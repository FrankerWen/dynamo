// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Destination-capacity reservation helpers for mock offload pipelines.

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
use rustc_hash::FxHashMap;

/// Shared destination-capacity reservations for lower-tier pipelines.
///
/// Policy evaluation can run concurrently across workers. `BlockManager`
/// exposes only currently-free blocks, so accepted-but-not-yet-registered
/// transfers need an atomic side reservation to avoid over-admitting into a
/// shared tier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ByteCapacityMetrics {
    pub capacity_bytes: usize,
    pub logical_slot_capacity: usize,
    pub reserved_bytes: usize,
    pub occupied_bytes: usize,
    pub peak_committed_bytes: usize,
    pub reserved_blocks: usize,
    pub occupied_blocks: usize,
    pub peak_committed_blocks: usize,
    pub admission_rejections: u64,
    pub slot_admission_rejections: u64,
    pub byte_admission_rejections: u64,
}

#[derive(Debug)]
struct ByteCapacityState {
    profile: Arc<[usize]>,
    hash_to_id: FxHashMap<SequenceHash, u32>,
    hashes: Vec<Option<SequenceHash>>,
    bytes: Vec<usize>,
    generation: Vec<u32>,
    occupancy: Vec<u8>, // 0=free, 1=pending, 2=occupied
    free_ids: Vec<u32>,
    enforce_logical_slots: bool,
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
    #[cfg(test)]
    pub(crate) fn with_byte_capacity(capacity_bytes: usize, profile: Arc<[usize]>) -> Self {
        Self::with_byte_capacity_mode(capacity_bytes, profile, usize::MAX, false)
    }

    pub(crate) fn with_byte_capacity_mode(
        capacity_bytes: usize,
        profile: Arc<[usize]>,
        logical_slot_capacity: usize,
        enforce_logical_slots: bool,
    ) -> Self {
        assert!(capacity_bytes > 0, "G2 byte capacity must be positive");
        assert!(
            !profile.is_empty() && !profile.contains(&0),
            "G2 byte capacity requires a positive AIC block-footprint profile"
        );
        Self {
            reserved_blocks: AtomicUsize::new(0),
            byte_state: Some(Mutex::new(ByteCapacityState {
                profile,
                hash_to_id: FxHashMap::default(),
                hashes: Vec::new(),
                bytes: Vec::new(),
                generation: Vec::new(),
                occupancy: Vec::new(),
                free_ids: Vec::new(),
                enforce_logical_slots,
                metrics: ByteCapacityMetrics {
                    capacity_bytes,
                    logical_slot_capacity,
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

    fn insert_dense(state: &mut ByteCapacityState, hash: SequenceHash, bytes: usize, kind: u8) {
        let id = state.free_ids.pop().unwrap_or_else(|| {
            let id =
                u32::try_from(state.hashes.len()).expect("compact G2 dense hash id exhausted u32");
            state.hashes.push(None);
            state.bytes.push(0);
            state.generation.push(0);
            state.occupancy.push(0);
            id
        });
        let index = id as usize;
        state.hashes[index] = Some(hash);
        state.bytes[index] = bytes;
        state.generation[index] = state.generation[index].wrapping_add(1).max(1);
        state.occupancy[index] = kind;
        state.hash_to_id.insert(hash, id);
    }

    fn remove_dense(
        state: &mut ByteCapacityState,
        hash: SequenceHash,
        expected: u8,
    ) -> Option<usize> {
        let id = state.hash_to_id.get(&hash).copied()?;
        let index = id as usize;
        if state.occupancy[index] != expected {
            return None;
        }
        state.hash_to_id.remove(&hash);
        state.hashes[index] = None;
        state.occupancy[index] = 0;
        state.free_ids.push(id);
        Some(state.bytes[index])
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
            let duplicate = state.hash_to_id.contains_key(hash);
            let committed = state
                .metrics
                .reserved_bytes
                .saturating_add(state.metrics.occupied_bytes);
            let committed_blocks = state
                .metrics
                .reserved_blocks
                .saturating_add(state.metrics.occupied_blocks);
            let fits_slots = !state.enforce_logical_slots
                || (reserved_blocks < available_blocks
                    && committed_blocks < state.metrics.logical_slot_capacity);
            let fits_bytes = committed
                .checked_add(bytes)
                .is_some_and(|next| next <= state.metrics.capacity_bytes);
            if prefix_open && !duplicate && fits_slots && fits_bytes {
                Self::insert_dense(&mut state, *hash, bytes, 1);
                state.metrics.reserved_bytes = state.metrics.reserved_bytes.saturating_add(bytes);
                state.metrics.reserved_blocks = state.metrics.reserved_blocks.saturating_add(1);
                let committed = state
                    .metrics
                    .reserved_bytes
                    .saturating_add(state.metrics.occupied_bytes);
                state.metrics.peak_committed_bytes =
                    state.metrics.peak_committed_bytes.max(committed);
                state.metrics.peak_committed_blocks = state
                    .metrics
                    .peak_committed_blocks
                    .max(committed_blocks.saturating_add(1));
                reserved_blocks += 1;
                results.push(true);
            } else {
                state.metrics.admission_rejections =
                    state.metrics.admission_rejections.saturating_add(1);
                if !fits_slots {
                    state.metrics.slot_admission_rejections =
                        state.metrics.slot_admission_rejections.saturating_add(1);
                }
                if !fits_bytes {
                    state.metrics.byte_admission_rejections =
                        state.metrics.byte_admission_rejections.saturating_add(1);
                }
                results.push(false);
                prefix_open = false;
            }
        }
        self.reserved_blocks
            .store(reserved_blocks, Ordering::Release);
        results
    }

    /// Move a completed reservation into byte-accounted resident G2 state.
    #[cfg(test)]
    pub(crate) fn mark_published(&self, hash: SequenceHash) {
        self.apply_lifecycle_batch(&[(hash, true)]);
    }

    #[cfg(test)]
    pub(crate) fn mark_removed(&self, hash: SequenceHash) {
        self.apply_lifecycle_batch(&[(hash, false)]);
    }

    /// Apply one completion wave under a single lock. The boolean is true for
    /// publish and false for remove; caller order is retained.
    #[cfg(test)]
    pub(crate) fn apply_lifecycle_batch(&self, events: &[(SequenceHash, bool)]) {
        self.apply_lifecycle_events(events.iter().copied());
    }

    /// Apply a completion wave from any ordered iterator without materializing
    /// a temporary `(hash, published)` vector at the caller.
    pub(crate) fn apply_lifecycle_events<I>(&self, events: I)
    where
        I: IntoIterator<Item = (SequenceHash, bool)>,
    {
        let Some(byte_state) = self.byte_state.as_ref() else {
            return;
        };
        let mut state = byte_state
            .lock()
            .expect("byte capacity reservation mutex poisoned");
        for (hash, published) in events {
            if published {
                let Some(id) = state.hash_to_id.get(&hash).copied() else {
                    continue;
                };
                let index = id as usize;
                if state.occupancy[index] == 2 {
                    continue;
                }
                assert_eq!(
                    state.occupancy[index], 1,
                    "published G2 block was not pending"
                );
                let bytes = state.bytes[index];
                state.occupancy[index] = 2;
                state.metrics.reserved_bytes = state.metrics.reserved_bytes.saturating_sub(bytes);
                state.metrics.occupied_bytes = state.metrics.occupied_bytes.saturating_add(bytes);
                state.metrics.reserved_blocks = state.metrics.reserved_blocks.saturating_sub(1);
                state.metrics.occupied_blocks = state.metrics.occupied_blocks.saturating_add(1);
                assert!(
                    state.metrics.occupied_bytes <= state.metrics.capacity_bytes,
                    "published G2 byte occupancy exceeded configured capacity"
                );
            } else if let Some(bytes) = Self::remove_dense(&mut state, hash, 2) {
                state.metrics.occupied_bytes = state.metrics.occupied_bytes.saturating_sub(bytes);
                state.metrics.occupied_blocks = state.metrics.occupied_blocks.saturating_sub(1);
            }
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
            if let Some(bytes) = Self::remove_dense(&mut state, *hash, 1) {
                state.metrics.reserved_bytes = state.metrics.reserved_bytes.saturating_sub(bytes);
                state.metrics.reserved_blocks = state.metrics.reserved_blocks.saturating_sub(1);
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
                logical_slot_capacity: usize::MAX,
                reserved_bytes: 2,
                occupied_bytes: 0,
                peak_committed_bytes: 2,
                reserved_blocks: 1,
                peak_committed_blocks: 1,
                admission_rejections: 0,
                ..ByteCapacityMetrics::default()
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
    fn sparse_byte_native_ignores_dense_available_slots_but_not_bytes() {
        let reservations = CapacityReservations::with_byte_capacity_mode(
            5,
            Arc::from(vec![2usize, 3usize, 4usize]),
            2,
            false,
        );
        let first = PositionalLineageHash::new(10, None, 0);
        let second = PositionalLineageHash::new(11, Some(10), 1);
        let third = PositionalLineageHash::new(12, Some(11), 2);

        assert_eq!(
            reservations.try_reserve_hashes(0, &[first, second]),
            vec![true, true]
        );
        assert_eq!(reservations.try_reserve_hashes(0, &[third]), vec![false]);
        let metrics = reservations.byte_metrics().unwrap();
        assert_eq!(metrics.slot_admission_rejections, 0);
        assert_eq!(metrics.byte_admission_rejections, 1);
        assert_eq!(metrics.peak_committed_blocks, 2);
        assert_eq!(metrics.peak_committed_bytes, 5);
    }

    #[test]
    fn capped_dense_reports_slot_limited_rejection_separately() {
        let reservations = CapacityReservations::with_byte_capacity_mode(
            100,
            Arc::from(vec![2usize, 3usize]),
            1,
            true,
        );
        let first = PositionalLineageHash::new(10, None, 0);
        let second = PositionalLineageHash::new(11, Some(10), 1);
        assert_eq!(
            reservations.try_reserve_hashes(1, &[first, second]),
            vec![true, false]
        );
        let metrics = reservations.byte_metrics().unwrap();
        assert_eq!(metrics.slot_admission_rejections, 1);
        assert_eq!(metrics.byte_admission_rejections, 0);
    }

    #[test]
    #[should_panic(expected = "capacity reservation underflow")]
    fn capacity_reservation_release_panics_on_underflow() {
        let reservations = CapacityReservations::default();
        reservations.release(1);
    }
}
