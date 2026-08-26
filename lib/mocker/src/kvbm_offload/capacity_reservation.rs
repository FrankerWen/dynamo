// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Destination-capacity reservation helpers for mock offload pipelines.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use kvbm_engine::SequenceHash;
use kvbm_engine::offload::{
    EvalContext, OffloadPolicy, PolicyBatchFuture, PolicyFuture, sync_batch_result, sync_result,
};
use kvbm_logical::blocks::BlockMetadata;
use kvbm_logical::manager::BlockManager;

/// Shared destination-capacity reservations for lower-tier pipelines.
///
/// Policy evaluation can run concurrently across workers. `BlockManager`
/// exposes only currently-free blocks, so accepted-but-not-yet-registered
/// transfers need an atomic side reservation to avoid over-admitting into a
/// shared tier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ByteCapacityMetrics {
    pub capacity_bytes: usize,
    pub reserved_bytes: usize,
    pub occupied_bytes: usize,
    pub peak_committed_bytes: usize,
    pub admission_rejections: u64,
}

#[derive(Debug)]
struct ByteCapacityState {
    profile: Arc<[usize]>,
    pending: HashMap<SequenceHash, usize>,
    occupied: HashMap<SequenceHash, usize>,
    metrics: ByteCapacityMetrics,
}

#[derive(Debug)]
pub(crate) struct CapacityReservations {
    reserved_blocks: AtomicUsize,
    byte_state: Option<Mutex<ByteCapacityState>>,
}

impl Default for CapacityReservations {
    fn default() -> Self {
        Self {
            reserved_blocks: AtomicUsize::new(0),
            byte_state: None,
        }
    }
}

impl CapacityReservations {
    pub(crate) fn with_byte_capacity(capacity_bytes: usize, profile: Arc<[usize]>) -> Self {
        assert!(capacity_bytes > 0, "G2 byte capacity must be positive");
        assert!(
            !profile.is_empty() && !profile.contains(&0),
            "G2 byte capacity requires a positive AIC block-footprint profile"
        );
        Self {
            reserved_blocks: AtomicUsize::new(0),
            byte_state: Some(Mutex::new(ByteCapacityState {
                profile,
                pending: HashMap::new(),
                occupied: HashMap::new(),
                metrics: ByteCapacityMetrics {
                    capacity_bytes,
                    ..ByteCapacityMetrics::default()
                },
            })),
        }
    }

    pub(crate) fn uses_byte_capacity(&self) -> bool {
        self.byte_state.is_some()
    }

    pub(crate) fn reserved_blocks(&self) -> usize {
        self.reserved_blocks.load(Ordering::Acquire)
    }

    pub(crate) fn byte_metrics(&self) -> Option<ByteCapacityMetrics> {
        self.byte_state.as_ref().map(|state| {
            state
                .lock()
                .expect("byte capacity reservation mutex poisoned")
                .metrics
        })
    }

    fn bytes_for_hash(state: &ByteCapacityState, hash: &SequenceHash) -> Option<usize> {
        state.profile.get(hash.position() as usize).copied()
    }

    /// Reserve destination slots and physical bytes for a deterministic
    /// prefix of hashes. Each result corresponds to one input hash; once a
    /// capacity limit rejects an element, later elements are rejected too so
    /// pipeline batching preserves prefix order.
    pub(crate) fn try_reserve_hashes(
        &self,
        available_blocks: usize,
        hashes: &[SequenceHash],
    ) -> Vec<bool> {
        let Some(byte_state) = self.byte_state.as_ref() else {
            let reserved = self.try_reserve_up_to(available_blocks, hashes.len());
            let mut results = vec![true; reserved];
            results.resize(hashes.len(), false);
            return results;
        };
        if hashes.is_empty() {
            return Vec::new();
        }

        let mut state = byte_state
            .lock()
            .expect("byte capacity reservation mutex poisoned");
        let mut reserved_blocks = self.reserved_blocks.load(Ordering::Acquire);
        let mut results = Vec::with_capacity(hashes.len());
        let mut prefix_open = true;
        for hash in hashes {
            let Some(bytes) = Self::bytes_for_hash(&state, hash) else {
                state.metrics.admission_rejections =
                    state.metrics.admission_rejections.saturating_add(1);
                results.push(false);
                prefix_open = false;
                continue;
            };
            let duplicate = state.pending.contains_key(hash) || state.occupied.contains_key(hash);
            let committed = state
                .metrics
                .reserved_bytes
                .saturating_add(state.metrics.occupied_bytes);
            let fits_slots = reserved_blocks < available_blocks;
            let fits_bytes = committed
                .checked_add(bytes)
                .is_some_and(|next| next <= state.metrics.capacity_bytes);
            if prefix_open && !duplicate && fits_slots && fits_bytes {
                state.pending.insert(*hash, bytes);
                state.metrics.reserved_bytes = state.metrics.reserved_bytes.saturating_add(bytes);
                let committed = state
                    .metrics
                    .reserved_bytes
                    .saturating_add(state.metrics.occupied_bytes);
                state.metrics.peak_committed_bytes =
                    state.metrics.peak_committed_bytes.max(committed);
                reserved_blocks += 1;
                results.push(true);
            } else {
                state.metrics.admission_rejections =
                    state.metrics.admission_rejections.saturating_add(1);
                results.push(false);
                prefix_open = false;
            }
        }
        self.reserved_blocks
            .store(reserved_blocks, Ordering::Release);
        results
    }

    /// Move a completed reservation into byte-accounted resident G2 state.
    pub(crate) fn mark_published(&self, hash: SequenceHash) {
        let Some(byte_state) = self.byte_state.as_ref() else {
            return;
        };
        let mut state = byte_state
            .lock()
            .expect("byte capacity reservation mutex poisoned");
        if state.occupied.contains_key(&hash) {
            return;
        }
        let bytes = state
            .pending
            .remove(&hash)
            .or_else(|| Self::bytes_for_hash(&state, &hash))
            .expect("published G2 block position must exist in AIC footprint");
        state.metrics.reserved_bytes = state.metrics.reserved_bytes.saturating_sub(bytes);
        state.metrics.occupied_bytes = state.metrics.occupied_bytes.saturating_add(bytes);
        state.occupied.insert(hash, bytes);
        assert!(
            state.metrics.occupied_bytes <= state.metrics.capacity_bytes,
            "published G2 byte occupancy exceeded configured capacity"
        );
    }

    pub(crate) fn mark_removed(&self, hash: SequenceHash) {
        let Some(byte_state) = self.byte_state.as_ref() else {
            return;
        };
        let mut state = byte_state
            .lock()
            .expect("byte capacity reservation mutex poisoned");
        if let Some(bytes) = state.occupied.remove(&hash) {
            state.metrics.occupied_bytes = state.metrics.occupied_bytes.saturating_sub(bytes);
        }
    }

    pub(crate) fn release_hashes(&self, hashes: &[SequenceHash]) {
        let Some(byte_state) = self.byte_state.as_ref() else {
            return;
        };
        let mut state = byte_state
            .lock()
            .expect("byte capacity reservation mutex poisoned");
        for hash in hashes {
            if let Some(bytes) = state.pending.remove(hash) {
                state.metrics.reserved_bytes = state.metrics.reserved_bytes.saturating_sub(bytes);
            }
        }
    }

    pub(crate) fn try_reserve(self: &Arc<Self>, available_blocks: usize, blocks: usize) -> bool {
        assert!(
            self.byte_state.is_none(),
            "byte-accounted G2 reservations must name logical block hashes"
        );
        if blocks == 0 {
            return true;
        }

        self.reserved_blocks
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |reserved| {
                let next = reserved.checked_add(blocks)?;
                (next <= available_blocks).then_some(next)
            })
            .is_ok()
    }

    pub(crate) fn try_reserve_up_to(&self, available_blocks: usize, blocks: usize) -> usize {
        assert!(
            self.byte_state.is_none(),
            "byte-accounted G2 reservations must name logical block hashes"
        );
        if blocks == 0 {
            return 0;
        }

        let mut reserved = self.reserved_blocks.load(Ordering::Acquire);
        loop {
            let free = available_blocks.saturating_sub(reserved);
            let to_reserve = blocks.min(free);
            if to_reserve == 0 {
                return 0;
            }
            let next = reserved + to_reserve;
            match self.reserved_blocks.compare_exchange_weak(
                reserved,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return to_reserve,
                Err(actual) => reserved = actual,
            }
        }
    }

    pub(crate) fn release(&self, blocks: usize) {
        if blocks == 0 {
            return;
        }
        let result =
            self.reserved_blocks
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |reserved| {
                    reserved.checked_sub(blocks)
                });
        if let Err(reserved) = result {
            panic!(
                "capacity reservation underflow: tried to release {blocks} blocks with {reserved} reserved"
            );
        }
    }
}

#[derive(Debug)]
pub(crate) struct CapacityReservationGuard {
    reservations: Arc<CapacityReservations>,
    blocks: usize,
    hashes: Vec<SequenceHash>,
}

impl CapacityReservationGuard {
    pub(crate) fn new(reservations: Arc<CapacityReservations>, blocks: usize) -> Self {
        Self {
            reservations,
            blocks,
            hashes: Vec::new(),
        }
    }

    pub(crate) fn with_hashes(
        reservations: Arc<CapacityReservations>,
        hashes: Vec<SequenceHash>,
    ) -> Self {
        Self {
            blocks: hashes.len(),
            reservations,
            hashes,
        }
    }

    fn release(&mut self) {
        let blocks = std::mem::take(&mut self.blocks);
        self.reservations.release_hashes(&self.hashes);
        self.hashes.clear();
        self.reservations.release(blocks);
    }
}

impl Drop for CapacityReservationGuard {
    fn drop(&mut self) {
        self.release();
    }
}

/// Destination-capacity guard for an offload pipeline.
///
/// kvbm-engine's executor currently reports allocation failure after policy
/// evaluation if the destination tier is full. For mocker, a full lower tier
/// should behave like a cache miss/drop rather than producing an executor
/// error or pinning simulated source capacity, so we filter before enqueueing
/// work that cannot allocate a destination block.
pub(crate) struct CapacityReservationPolicy<Src: BlockMetadata, Dst: BlockMetadata> {
    dst_manager: Arc<BlockManager<Dst>>,
    capacity_reservations: Arc<CapacityReservations>,
    _marker: PhantomData<Src>,
}

impl<Src: BlockMetadata, Dst: BlockMetadata> CapacityReservationPolicy<Src, Dst> {
    pub(crate) fn new(
        dst_manager: Arc<BlockManager<Dst>>,
        capacity_reservations: Arc<CapacityReservations>,
    ) -> Self {
        Self {
            dst_manager,
            capacity_reservations,
            _marker: PhantomData,
        }
    }

    fn try_reserve_capacity(&self, blocks: usize) -> bool {
        debug_assert!(!self.capacity_reservations.uses_byte_capacity());
        let available = self.dst_manager.available_blocks();
        self.capacity_reservations.try_reserve(available, blocks)
    }
}

impl<Src: BlockMetadata, Dst: BlockMetadata> OffloadPolicy<Src>
    for CapacityReservationPolicy<Src, Dst>
{
    fn name(&self) -> &str {
        "CapacityReservationPolicy"
    }

    fn evaluate<'a>(&'a self, ctx: &'a EvalContext<Src>) -> PolicyFuture<'a> {
        let accepted = if self.capacity_reservations.uses_byte_capacity() {
            self.capacity_reservations
                .try_reserve_hashes(self.dst_manager.available_blocks(), &[ctx.sequence_hash])[0]
        } else {
            self.try_reserve_capacity(1)
        };
        sync_result(Ok(accepted))
    }

    fn evaluate_batch<'a>(&'a self, contexts: &'a [EvalContext<Src>]) -> PolicyBatchFuture<'a> {
        if contexts.is_empty() {
            return sync_batch_result(Ok(Vec::new()));
        }

        let available = self.dst_manager.available_blocks();
        if self.capacity_reservations.uses_byte_capacity() {
            let hashes = contexts
                .iter()
                .map(|context| context.sequence_hash)
                .collect::<Vec<_>>();
            return sync_batch_result(Ok(self
                .capacity_reservations
                .try_reserve_hashes(available, &hashes)));
        }
        let reserved = self
            .capacity_reservations
            .try_reserve_up_to(available, contexts.len());
        let mut results = vec![true; reserved];
        results.resize(contexts.len(), false);

        sync_batch_result(Ok(results))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use dynamo_tokens::PositionalLineageHash;
    use futures::future::Either;
    use kvbm_engine::offload::{EvalContext, OffloadPolicy};
    use kvbm_engine::{G1 as EngineG1, G2};
    use kvbm_logical::pools::BlockDuplicationPolicy;
    use kvbm_logical::registry::BlockRegistry;

    #[test]
    fn capacity_reservation_policy_rejects_when_reserved_capacity_is_full() {
        let g2_manager = Arc::new(
            BlockManager::<G2>::builder()
                .block_count(1)
                .block_size(64)
                .registry(BlockRegistry::new())
                .duplication_policy(BlockDuplicationPolicy::Reject)
                .with_lineage_backend()
                .build()
                .expect("BlockManager<G2> should build with valid config"),
        );
        let reservations = Arc::new(CapacityReservations::default());
        let filter =
            CapacityReservationPolicy::<EngineG1, G2>::new(g2_manager, reservations.clone());

        let ctx = EvalContext::from_external(0, PositionalLineageHash::new(10, None, 0));
        let Either::Left(result) = filter.evaluate(&ctx) else {
            panic!("CapacityReservationPolicy should evaluate synchronously");
        };
        assert!(
            futures::executor::block_on(result).expect("policy should succeed"),
            "first block should reserve the only free destination slot"
        );

        let ctx = EvalContext::from_external(1, PositionalLineageHash::new(11, None, 1));
        let Either::Left(result) = filter.evaluate(&ctx) else {
            panic!("CapacityReservationPolicy should evaluate synchronously");
        };
        assert!(
            !futures::executor::block_on(result).expect("policy should succeed"),
            "reserved destination capacity should reject another block"
        );

        reservations.release(1);
        let ctx = EvalContext::from_external(2, PositionalLineageHash::new(12, None, 2));
        let Either::Left(result) = filter.evaluate(&ctx) else {
            panic!("CapacityReservationPolicy should evaluate synchronously");
        };
        assert!(futures::executor::block_on(result).expect("policy should succeed"));
    }

    #[test]
    fn byte_capacity_tracks_pending_and_resident_aic_footprints() {
        let reservations =
            CapacityReservations::with_byte_capacity(5, Arc::from(vec![2usize, 4usize]));
        let first = PositionalLineageHash::new(10, None, 0);
        let second = PositionalLineageHash::new(11, Some(10), 1);

        assert_eq!(reservations.try_reserve_hashes(10, &[first]), vec![true]);
        assert_eq!(reservations.reserved_blocks(), 1);
        assert_eq!(
            reservations.byte_metrics(),
            Some(ByteCapacityMetrics {
                capacity_bytes: 5,
                reserved_bytes: 2,
                occupied_bytes: 0,
                peak_committed_bytes: 2,
                admission_rejections: 0,
            })
        );

        assert_eq!(reservations.try_reserve_hashes(10, &[second]), vec![false]);
        reservations.mark_published(first);
        let metrics = reservations.byte_metrics().unwrap();
        assert_eq!(metrics.reserved_bytes, 0);
        assert_eq!(metrics.occupied_bytes, 2);
        assert_eq!(metrics.admission_rejections, 1);

        reservations.release(1);
        reservations.mark_removed(first);
        assert_eq!(reservations.try_reserve_hashes(10, &[second]), vec![true]);
        reservations.release_hashes(&[second]);
        reservations.release(1);
        let metrics = reservations.byte_metrics().unwrap();
        assert_eq!(metrics.reserved_bytes, 0);
        assert_eq!(metrics.occupied_bytes, 0);
        assert_eq!(reservations.reserved_blocks(), 0);
    }

    #[test]
    #[should_panic(expected = "capacity reservation underflow")]
    fn capacity_reservation_release_panics_on_underflow() {
        let reservations = CapacityReservations::default();
        reservations.release(1);
    }
}
