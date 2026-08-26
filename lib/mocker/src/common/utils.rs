// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

use crate::common::handoff::HandoffTransferTiming;
use crate::common::protocols::{KvTransferTimingMode, MockEngineArgs, WorkerType};

pub fn prefill_handoff_transfer_timing(
    num_input_tokens: usize,
    block_size: usize,
    kv_block_footprint_bytes: Option<&[usize]>,
    kv_transfer_bandwidth: Option<f64>,
    kv_bytes_per_token: Option<usize>,
    mode: KvTransferTimingMode,
) -> HandoffTransferTiming {
    HandoffTransferTiming {
        mode,
        full_prompt_tokens: num_input_tokens,
        full_prompt_bytes: full_prompt_kv_bytes(
            num_input_tokens,
            block_size,
            kv_block_footprint_bytes,
            kv_bytes_per_token,
        ),
        kv_bytes_per_token,
        bandwidth_gb_s: kv_transfer_bandwidth,
    }
}

/// Resolve the KV bytes represented by a prompt.
///
/// Dynamic AIC footprints are indexed by logical block position and are the
/// byte authority whenever present. The scheduler and KVBM reserve whole
/// logical blocks, so a partial final prompt block is charged as one block as
/// well. The scalar path is retained only for legacy non-exact callers.
pub fn full_prompt_kv_bytes(
    num_input_tokens: usize,
    block_size: usize,
    kv_block_footprint_bytes: Option<&[usize]>,
    kv_bytes_per_token: Option<usize>,
) -> Option<usize> {
    if let Some(profile) = kv_block_footprint_bytes {
        if block_size == 0 {
            return None;
        }
        let required_blocks = num_input_tokens.div_ceil(block_size);
        let blocks = profile.get(..required_blocks)?;
        return blocks
            .iter()
            .try_fold(0usize, |total, bytes| total.checked_add(*bytes));
    }
    kv_bytes_per_token.and_then(|bytes_per_token| num_input_tokens.checked_mul(bytes_per_token))
}

/// Compute the modeled handoff delay after a prefill worker emits its terminal token.
///
/// NOTE: this intentionally does not model the internal prefill TTFT itself accurately, and the
/// exact prefill/decode boundary is backend dependent. For now we only care about decode-visible
/// TTFT, which is what the client observes, so modeling the delay as prefill-to-decode handoff is
/// good enough.
pub fn compute_prefill_handoff_delay_ms(
    worker_type: WorkerType,
    completed: bool,
    num_input_tokens: usize,
    block_size: usize,
    kv_block_footprint_bytes: Option<&[usize]>,
    kv_transfer_bandwidth: Option<f64>,
    kv_bytes_per_token: Option<usize>,
) -> Option<f64> {
    if worker_type != WorkerType::Prefill || !completed {
        return None;
    }
    let timing = prefill_handoff_transfer_timing(
        num_input_tokens,
        block_size,
        kv_block_footprint_bytes,
        kv_transfer_bandwidth,
        kv_bytes_per_token,
        KvTransferTimingMode::FullPrompt,
    );
    match timing.full_prompt_delay_ms() {
        Some(delay_ms) => {
            tracing::debug!(
                num_input_tokens,
                bandwidth_gb_s = kv_transfer_bandwidth,
                delay_ms = format!("{delay_ms:.2}"),
                "KV handoff delay for prefill completion"
            );
            Some(delay_ms)
        }
        None => None,
    }
}

/// Compute the KV transfer delay duration for a given number of input tokens.
///
/// Returns `None` if KV transfer simulation is disabled (bandwidth is 0 or not configured).
pub fn compute_kv_transfer_delay(
    args: &MockEngineArgs,
    num_input_tokens: usize,
) -> Option<Duration> {
    compute_prefill_handoff_delay_ms(
        args.worker_type,
        true,
        num_input_tokens,
        args.block_size,
        args.kv_block_footprint_bytes.as_deref(),
        args.kv_transfer_bandwidth,
        args.kv_bytes_per_token,
    )
    .map(|delay_ms| Duration::from_secs_f64(delay_ms / 1000.0))
}

/// Sleep for the specified duration using timerfd on Linux for precision.
pub async fn sleep_precise(duration: Duration) {
    sleep_until_precise(Instant::now() + duration).await;
}

/// Sleep until the specified deadline using timerfd on Linux for precision.
///
/// Unlike `sleep_precise`, this accounts for time already elapsed since the
/// deadline's reference point, making it suitable for simulation loops where
/// computation time should be subtracted from the sleep.
pub async fn sleep_until_precise(deadline: Instant) {
    // Scheduler work may consume the modeled delay, especially at high speedup ratios. Avoid
    // allocating and registering a timerfd when there is no remaining time to sleep. Preserve
    // the scheduler loop's cooperative yield so other tasks on the runtime can make progress.
    if deadline <= Instant::now() {
        tokio::task::yield_now().await;
        return;
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(delay) = tokio_timerfd::Delay::new(deadline) {
            let _ = delay.await;
        } else {
            tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::task::Poll;

    #[tokio::test(flavor = "current_thread")]
    async fn test_expired_precise_sleep_yields_to_runtime() {
        let sleep = sleep_until_precise(Instant::now());
        tokio::pin!(sleep);

        let first_poll = futures::poll!(sleep.as_mut());

        assert!(matches!(first_poll, Poll::Pending));
        sleep.await;
    }

    #[test]
    fn test_prefill_handoff_delay_only_applies_to_completed_prefill() {
        let delay_ms = compute_prefill_handoff_delay_ms(
            WorkerType::Prefill,
            true,
            128,
            16,
            None,
            Some(1.0),
            Some(1_000_000),
        )
        .expect("prefill completion should produce a handoff delay");
        assert!((delay_ms - 128.0).abs() < 1e-9);

        assert!(
            compute_prefill_handoff_delay_ms(
                WorkerType::Prefill,
                false,
                128,
                16,
                None,
                Some(1.0),
                Some(1_000_000),
            )
            .is_none()
        );
        assert!(
            compute_prefill_handoff_delay_ms(
                WorkerType::Decode,
                true,
                128,
                16,
                None,
                Some(1.0),
                Some(1_000_000),
            )
            .is_none()
        );
    }

    #[test]
    fn dynamic_full_prompt_bytes_charge_whole_logical_blocks() {
        let profile = [100usize, 200, 300];
        assert_eq!(full_prompt_kv_bytes(0, 16, Some(&profile), None), Some(0));
        assert_eq!(full_prompt_kv_bytes(1, 16, Some(&profile), None), Some(100));
        assert_eq!(
            full_prompt_kv_bytes(16, 16, Some(&profile), None),
            Some(100)
        );
        assert_eq!(
            full_prompt_kv_bytes(17, 16, Some(&profile), None),
            Some(300)
        );
        assert_eq!(full_prompt_kv_bytes(49, 16, Some(&profile), None), None);
    }
}
