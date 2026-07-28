// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use super::super::AdmissionEvent;
use super::config::{SglangConfig, ceil_to_block};
use super::request::SglangRequest;
use crate::kv_manager::SglangKvManager;

/// Per-request prefill data needed for FPM snapshot construction.
pub(super) struct PrefillFpmItem {
    pub(super) prompt_len: usize,
    pub(super) tokens_computed: usize,
    pub(super) prefix_tokens: usize,
}

#[derive(Default)]
pub(super) struct AdmitResult {
    pub(super) can_run: Vec<SglangRequest>,
    pub(super) admissions: Vec<AdmissionEvent>,
    pub(super) total_isl: usize,
    pub(super) total_prefix: usize,
    pub(super) oom: bool,
    /// Per-request prefill info for building FPM snapshots.
    pub(super) prefill_fpm: Vec<PrefillFpmItem>,
}

pub(super) fn get_new_batch_prefill(
    waiting: &mut VecDeque<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    new_token_ratio: f64,
    running: &[SglangRequest],
) -> AdmitResult {
    let cache = kv_manager.cache();
    let reserved_decode_output: f64 = running
        .iter()
        .map(|req| {
            let remaining_output = req
                .remaining_output_tokens()
                .min(config.clip_max_new_tokens);
            remaining_output as f64 * new_token_ratio
        })
        .sum();
    let reserved_page_overhead = waiting
        .iter()
        .map(SglangRequest::extra_reserved_tokens)
        .sum::<usize>()
        + running
            .iter()
            .map(SglangRequest::extra_reserved_tokens)
            .sum::<usize>();

    let mut rem_total_tokens = (cache.available_tokens() + cache.evictable_size)
        .saturating_sub(reserved_page_overhead) as f64
        - reserved_decode_output;
    let mut rem_input_tokens = config.max_prefill_tokens as f64;
    let mut rem_chunk_tokens = config.chunked_prefill_size as f64;

    let mut can_run = Vec::new();
    let mut admissions = Vec::new();
    let mut prefill_fpm = Vec::new();
    let mut rejected = VecDeque::new();
    let mut oom = false;
    let mut total_isl = 0usize;
    let mut total_prefix = 0usize;

    let available_running_slots = config.max_running_requests.saturating_sub(running.len());
    while can_run.len() < available_running_slots
        && let Some(mut req) = waiting.pop_front()
    {
        let first_admission = req.materialized_tokens == 0;
        // Match the complete current sequence before applying the chunk budget.
        // SGLang's Req::init_next_round_input does the same: cached prefix
        // tokens do not consume the fresh-token chunk budget. Matching only
        // sequence_prefix(chunked_prefill_size) loses any shared prefix past
        // the first chunk and makes later chunks look like new compute.
        let prefix_len = if first_admission {
            kv_manager.cache().prefix_match_len(req.sequence_tokens())
        } else {
            req.materialized_tokens
        };
        let extend_input = req.current_sequence_len().saturating_sub(prefix_len);

        let chunk_tokens = if extend_input <= config.chunked_prefill_size {
            extend_input
        } else {
            let chunk = (rem_chunk_tokens as usize / config.block_size) * config.block_size;
            if chunk == 0 {
                rejected.push_back(req);
                break;
            }
            chunk.min(extend_input)
        };

        let charged_input_tokens = ceil_to_block(chunk_tokens, config.block_size) as f64;
        let output_reserve = if chunk_tokens < extend_input {
            0
        } else {
            req.remaining_output_tokens()
                .min(config.clip_max_new_tokens)
        };
        if charged_input_tokens + output_reserve as f64 >= rem_total_tokens {
            rejected.push_back(req);
            break;
        }
        if charged_input_tokens > rem_input_tokens || charged_input_tokens > rem_chunk_tokens {
            rejected.push_back(req);
            break;
        }

        let chunk_end = prefix_len + chunk_tokens;
        let mut lease = std::mem::take(&mut req.kv_lease);
        let alloc_tokens = req.sequence_prefix(chunk_end);
        // Keep the prefix discovered above protected while making room for
        // fresh tokens. Otherwise eviction could remove that exact radix path
        // between the read-only match and allocate_for_request, changing the
        // effective prefix after the chunk budget has already been computed.
        let protected_prefix_node = if first_admission {
            let (protected_prefix_len, node) =
                kv_manager.cache_mut().match_prefix(req.sequence_tokens());
            debug_assert_eq!(protected_prefix_len, prefix_len);
            kv_manager.cache_mut().inc_lock_ref(node);
            Some(node)
        } else {
            None
        };
        let actual_new_tokens = chunk_tokens;
        let available = kv_manager.cache().token_pool.available();
        if available < actual_new_tokens {
            kv_manager.evict(actual_new_tokens - available);
        }

        let allocated_prefix_len = if !first_admission {
            if !lease.is_active() {
                panic!(
                    "prefill: request {} has materialized_tokens={} but no active KV lease",
                    req.uuid, req.materialized_tokens
                );
            }
            kv_manager
                .extend_allocation(alloc_tokens, &mut lease)
                .then_some(req.materialized_tokens)
        } else {
            kv_manager.allocate_for_request(alloc_tokens).map(|alloc| {
                lease = alloc.lease;
                alloc.prefix_len
            })
        };
        if let Some(node) = protected_prefix_node {
            kv_manager.cache_mut().dec_lock_ref(node);
        }

        let Some(allocated_prefix_len) = allocated_prefix_len else {
            req.kv_lease = lease;
            rejected.push_back(req);
            oom = true;
            break;
        };
        let tokens_computed = chunk_end.saturating_sub(allocated_prefix_len);

        req.kv_lease = lease;
        req.materialized_tokens = chunk_end;
        req.allocated_tokens = ceil_to_block(chunk_end, config.block_size);
        req.debug_assert_invariants(config.block_size);

        if first_admission {
            admissions.push(AdmissionEvent {
                uuid: req.uuid,
                reused_input_tokens: allocated_prefix_len,
            });
        }
        prefill_fpm.push(PrefillFpmItem {
            prompt_len: req.prompt_len(),
            tokens_computed,
            prefix_tokens: allocated_prefix_len,
        });

        total_isl += chunk_end;
        total_prefix += allocated_prefix_len;
        rem_total_tokens -= charged_input_tokens + output_reserve as f64;
        rem_input_tokens -= charged_input_tokens;
        rem_chunk_tokens -= charged_input_tokens;
        can_run.push(req);

        if rem_chunk_tokens <= 0.0 {
            break;
        }
    }

    while let Some(req) = rejected.pop_back() {
        waiting.push_front(req);
    }

    AdmitResult {
        can_run,
        admissions,
        total_isl,
        total_prefix,
        oom,
        prefill_fpm,
    }
}
