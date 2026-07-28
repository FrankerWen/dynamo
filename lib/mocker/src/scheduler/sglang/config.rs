// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::common::perf_model::PerfModel;
use crate::common::protocols::{KvTransferTimingMode, MockEngineArgs, WorkerType};

const DEFAULT_MAX_PREFILL_TOKENS: usize = 16384;
const DEFAULT_CHUNKED_PREFILL_SIZE: usize = 8192;
const DEFAULT_CLIP_MAX_NEW_TOKENS: usize = 4096;
const DEFAULT_INIT_NEW_TOKEN_RATIO: f64 = 0.7;
const DEFAULT_MIN_NEW_TOKEN_RATIO_FACTOR: f64 = 0.14;
const DEFAULT_NEW_TOKEN_RATIO_DECAY_STEPS: f64 = 600.0;
pub(super) const LPM_FALLBACK_THRESHOLD: usize = 128;
pub(super) const IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD: usize = 32;
pub(super) const IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD: usize = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SchedulePolicy {
    #[default]
    Fifo,
    Lpm,
}

pub(super) struct SglangConfig {
    pub(super) schedule_policy: SchedulePolicy,
    pub(super) max_prefill_tokens: usize,
    pub(super) max_running_requests: usize,
    pub(super) chunked_prefill_size: usize,
    pub(super) clip_max_new_tokens: usize,
    pub(super) init_new_token_ratio: f64,
    pub(super) min_new_token_ratio: f64,
    pub(super) new_token_ratio_decay_step: f64,
    pub(super) perf_model: Arc<PerfModel>,
    pub(super) speedup_ratio: f64,
    pub(super) decode_speedup_ratio: f64,
    pub(super) worker_type: WorkerType,
    pub(super) block_size: usize,
    pub(super) total_kv_tokens: usize,
    pub(super) kv_bytes_per_token: Option<usize>,
    pub(super) kv_transfer_bandwidth: Option<f64>,
    pub(super) kv_transfer_timing_mode: KvTransferTimingMode,
    pub(super) speculative_max_tokens: Option<usize>,
    pub(super) cuda_graph_enabled: bool,
    pub(super) cuda_graph_padding_enabled: bool,
    pub(super) cuda_graph_batch_sizes: Vec<usize>,
}

impl SglangConfig {
    pub(super) fn from_args(args: &MockEngineArgs) -> Self {
        let sglang = args.sglang.as_ref();
        let schedule_conservativeness = sglang
            .and_then(|s| s.schedule_conservativeness)
            .unwrap_or(1.0);
        let init_new_token_ratio = DEFAULT_INIT_NEW_TOKEN_RATIO * schedule_conservativeness;
        let min_new_token_ratio = init_new_token_ratio * DEFAULT_MIN_NEW_TOKEN_RATIO_FACTOR;
        let decay_steps = DEFAULT_NEW_TOKEN_RATIO_DECAY_STEPS;
        let decay_step = (init_new_token_ratio - min_new_token_ratio) / decay_steps;

        let policy_str = sglang.and_then(|s| s.schedule_policy.as_deref());
        let schedule_policy = match policy_str {
            Some("lpm") => SchedulePolicy::Lpm,
            Some("fifo") | Some("fcfs") | None => SchedulePolicy::Fifo,
            Some(other) => {
                tracing::warn!(
                    "Unknown sglang schedule_policy '{}', falling back to FIFO",
                    other
                );
                SchedulePolicy::Fifo
            }
        };

        Self {
            schedule_policy,
            max_running_requests: args.max_num_seqs.unwrap_or(usize::MAX),
            max_prefill_tokens: sglang
                .and_then(|s| s.max_prefill_tokens)
                .unwrap_or(DEFAULT_MAX_PREFILL_TOKENS),
            chunked_prefill_size: sglang
                .and_then(|s| s.chunked_prefill_size)
                .unwrap_or(DEFAULT_CHUNKED_PREFILL_SIZE),
            clip_max_new_tokens: sglang
                .and_then(|s| s.clip_max_new_tokens)
                .unwrap_or(DEFAULT_CLIP_MAX_NEW_TOKENS),
            init_new_token_ratio,
            min_new_token_ratio,
            new_token_ratio_decay_step: decay_step,
            perf_model: args.perf_model.clone(),
            speedup_ratio: args.speedup_ratio,
            decode_speedup_ratio: args.decode_speedup_ratio,
            worker_type: args.worker_type,
            block_size: args.block_size,
            total_kv_tokens: args.num_gpu_blocks * args.block_size,
            kv_bytes_per_token: args.kv_bytes_per_token,
            kv_transfer_bandwidth: args.kv_transfer_bandwidth,
            kv_transfer_timing_mode: args.kv_transfer_timing_mode,
            speculative_max_tokens: args.aic_nextn.map(|nextn| nextn + 1),
            cuda_graph_enabled: sglang.and_then(|s| s.cuda_graph_enabled).unwrap_or(false),
            cuda_graph_padding_enabled: sglang
                .and_then(|s| s.cuda_graph_padding_enabled)
                .unwrap_or(false),
            cuda_graph_batch_sizes: sglang
                .and_then(|s| s.cuda_graph_batch_sizes.clone())
                .unwrap_or_default(),
        }
    }

    pub(super) fn decode_timing_batch_size(&self, logical_batch_size: usize) -> usize {
        resolve_cuda_graph_batch_size(
            logical_batch_size,
            self.cuda_graph_enabled,
            self.cuda_graph_padding_enabled,
            &self.cuda_graph_batch_sizes,
        )
    }
}

fn resolve_cuda_graph_batch_size(
    logical_batch_size: usize,
    graph_enabled: bool,
    padding_enabled: bool,
    capture_sizes: &[usize],
) -> usize {
    if !graph_enabled || !padding_enabled {
        return logical_batch_size;
    }
    capture_sizes
        .iter()
        .copied()
        .find(|&capture_size| capture_size >= logical_batch_size)
        .unwrap_or(logical_batch_size)
}

pub(super) fn ceil_to_block(tokens: usize, block_size: usize) -> usize {
    if tokens == 0 {
        return 0;
    }

    tokens.div_ceil(block_size) * block_size
}

pub(super) fn floor_to_block(tokens: usize, block_size: usize) -> usize {
    tokens / block_size * block_size
}

#[cfg(test)]
mod tests {
    use super::resolve_cuda_graph_batch_size;

    const CAPTURE_SIZES: &[usize] = &[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64];

    #[test]
    fn cuda_graph_padding_uses_smallest_fitting_capture() {
        assert_eq!(
            resolve_cuda_graph_batch_size(9, true, true, CAPTURE_SIZES),
            12
        );
        assert_eq!(
            resolve_cuda_graph_batch_size(24, true, true, CAPTURE_SIZES),
            24
        );
    }

    #[test]
    fn cuda_graph_padding_keeps_batch_above_max_capture() {
        assert_eq!(
            resolve_cuda_graph_batch_size(65, true, true, CAPTURE_SIZES),
            65
        );
    }

    #[test]
    fn cuda_graph_disabled_or_unpadded_keeps_logical_batch() {
        assert_eq!(
            resolve_cuda_graph_batch_size(9, false, true, CAPTURE_SIZES),
            9
        );
        assert_eq!(
            resolve_cuda_graph_batch_size(9, true, false, CAPTURE_SIZES),
            9
        );
    }
}
