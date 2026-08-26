// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `MockWorker`: simulates G1↔G2/G2↔G3/G2↔G4 transfer timing without moving real memory.
//!
//! Flow per transfer:
//! 1. Caller sets the current simulation time via [`MockWorker::set_now_ms`].
//! 2. `OffloadEngine`'s pipeline drain task invokes
//!    [`WorkerTransfers::execute_local_transfer`]; the worker reserves a PS
//!    slot on the appropriate model, registers a [`velo::Event`], and
//!    returns a [`TransferCompleteNotification`] wired to the event.
//! 3. [`MockWorker::drain_completions`] advances both models under PS.
//!    Live mode triggers every due event immediately. Offline mode can detach
//!    due events and fire them one at a time around a causal settlement
//!    callback.
//!
//! Remote NIXL and cross-instance methods return `bail!` / all-Err futures.

use std::collections::HashMap;
#[cfg(test)]
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow, bail};
use futures::future::BoxFuture;

use kvbm_common::LogicalLayoutHandle;
use kvbm_engine::object::ObjectBlockOps;
use kvbm_engine::worker::{
    ConnectRemoteResponse, ImportMetadataResponse, RemoteDescriptor, SerializedLayoutResponse,
    Worker, WorkerTransfers,
};
use kvbm_engine::{BlockId, InstanceId, SequenceHash};
use kvbm_physical::manager::{LayoutHandle, SerializedLayout};
use kvbm_physical::transfer::{PhysicalLayout, TransferCompleteNotification, TransferOptions};
use tokio::sync::watch;
use velo::{Event, EventManager};

use super::KvbmDriveMode;
use super::bandwidth_sharing_model::{BandwidthSharingModel, TransferId};
use super::coordinator::SwapInStatus;
use super::shared_g2::SharedG2Fabric;
use super::shared_g3::SharedG3Pool;
use super::shared_g4::SharedG4Store;

/// Direction of a G1↔G2/G2↔G3/G2↔G4 transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    G1ToG2,
    G2ToG1,
    G2ToG3,
    G3ToG2,
    G2ToG4,
    G4ToG2,
}

impl TransferDirection {
    fn is_g3(self) -> bool {
        matches!(self, Self::G2ToG3 | Self::G3ToG2)
    }

    fn is_g4(self) -> bool {
        matches!(self, Self::G2ToG4 | Self::G4ToG2)
    }

    fn is_offload(self) -> bool {
        matches!(self, Self::G1ToG2 | Self::G2ToG3 | Self::G2ToG4)
    }

    fn label(self) -> &'static str {
        match self {
            Self::G1ToG2 => "G1→G2",
            Self::G2ToG1 => "G2→G1",
            Self::G2ToG3 => "G2→G3",
            Self::G3ToG2 => "G3→G2",
            Self::G2ToG4 => "G2→G4",
            Self::G4ToG2 => "G4→G2",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DirectionTransferMetrics {
    pub transfers: u64,
    pub completed_transfers: u64,
    pub blocks: u64,
    pub bytes: u64,
    /// End-to-end modeled reservation-to-completion time. Processor-sharing
    /// starts service immediately, so queue wait is explicitly zero today.
    pub total_transfer_time_us: u64,
    pub total_queue_wait_us: u64,
}

impl DirectionTransferMetrics {
    fn record(&mut self, blocks: usize, bytes: usize) {
        self.transfers = self.transfers.saturating_add(1);
        self.blocks = self.blocks.saturating_add(blocks as u64);
        self.bytes = self.bytes.saturating_add(bytes as u64);
    }

    fn record_completion(&mut self, elapsed_ms: f64) {
        self.completed_transfers = self.completed_transfers.saturating_add(1);
        self.total_transfer_time_us = self
            .total_transfer_time_us
            .saturating_add(ms_to_us(elapsed_ms));
        // The bandwidth model is processor sharing: admitted transfers begin
        // receiving service at reservation rather than waiting in a FIFO.
        self.total_queue_wait_us = self.total_queue_wait_us.saturating_add(0);
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OffloadTransferMetrics {
    pub g1_to_g2: DirectionTransferMetrics,
    pub g2_to_g1: DirectionTransferMetrics,
    pub g2_to_g3: DirectionTransferMetrics,
    pub g3_to_g2: DirectionTransferMetrics,
    pub g2_to_g4: DirectionTransferMetrics,
    pub g4_to_g2: DirectionTransferMetrics,
}

impl OffloadTransferMetrics {
    fn record(&mut self, direction: TransferDirection, blocks: usize, bytes: usize) {
        let metrics = match direction {
            TransferDirection::G1ToG2 => &mut self.g1_to_g2,
            TransferDirection::G2ToG1 => &mut self.g2_to_g1,
            TransferDirection::G2ToG3 => &mut self.g2_to_g3,
            TransferDirection::G3ToG2 => &mut self.g3_to_g2,
            TransferDirection::G2ToG4 => &mut self.g2_to_g4,
            TransferDirection::G4ToG2 => &mut self.g4_to_g2,
        };
        metrics.record(blocks, bytes);
    }

    fn record_completion(&mut self, direction: TransferDirection, elapsed_ms: f64) {
        let metrics = match direction {
            TransferDirection::G1ToG2 => &mut self.g1_to_g2,
            TransferDirection::G2ToG1 => &mut self.g2_to_g1,
            TransferDirection::G2ToG3 => &mut self.g2_to_g3,
            TransferDirection::G3ToG2 => &mut self.g3_to_g2,
            TransferDirection::G2ToG4 => &mut self.g2_to_g4,
            TransferDirection::G4ToG2 => &mut self.g4_to_g2,
        };
        metrics.record_completion(elapsed_ms);
    }
}

impl TryFrom<(LogicalLayoutHandle, LogicalLayoutHandle)> for TransferDirection {
    type Error = anyhow::Error;

    fn try_from((src, dst): (LogicalLayoutHandle, LogicalLayoutHandle)) -> Result<Self> {
        match (src, dst) {
            (LogicalLayoutHandle::G1, LogicalLayoutHandle::G2) => Ok(Self::G1ToG2),
            (LogicalLayoutHandle::G2, LogicalLayoutHandle::G1) => Ok(Self::G2ToG1),
            (LogicalLayoutHandle::G2, LogicalLayoutHandle::G3) => Ok(Self::G2ToG3),
            (LogicalLayoutHandle::G3, LogicalLayoutHandle::G2) => Ok(Self::G3ToG2),
            (s, d) => bail!(
                "MockWorker only simulates local G1↔G2 and G2↔G3 transfers; got src={:?} dst={:?}",
                s,
                d
            ),
        }
    }
}

struct PipelineAwaiter {
    event: Event,
    owner_id: u64,
    direction: TransferDirection,
    num_blocks: usize,
    started_at_ms: f64,
    metrics: Arc<Mutex<OffloadTransferMetrics>>,
}

struct SwapInAwaiter {
    owner_id: u64,
    num_blocks: usize,
    status: watch::Sender<SwapInStatus>,
    started_at_ms: f64,
    metrics: Arc<Mutex<OffloadTransferMetrics>>,
}

pub(crate) struct CompletionAction {
    id: TransferId,
    pipeline: Option<PipelineAwaiter>,
    swap_in: Option<SwapInAwaiter>,
    completed_at_ms: f64,
}

impl CompletionAction {
    pub(crate) fn owner_id(&self) -> Option<u64> {
        self.pipeline
            .as_ref()
            .map(|pipeline| pipeline.owner_id)
            .or_else(|| self.swap_in.as_ref().map(|swap_in| swap_in.owner_id))
    }

    pub(crate) fn completed_transfer(&self) -> CompletedTransfer {
        let (owner_id, direction, num_blocks, pipeline) =
            if let Some(pipeline) = self.pipeline.as_ref() {
                (
                    Some(pipeline.owner_id),
                    pipeline.direction,
                    pipeline.num_blocks,
                    true,
                )
            } else if let Some(swap_in) = self.swap_in.as_ref() {
                (
                    Some(swap_in.owner_id),
                    TransferDirection::G2ToG1,
                    swap_in.num_blocks,
                    false,
                )
            } else {
                (None, TransferDirection::G2ToG1, 0, false)
            };
        CompletedTransfer {
            id: self.id,
            owner_id,
            direction,
            num_blocks,
            pipeline,
        }
    }

    pub(crate) fn fire(self) -> CompletedTransfer {
        let completed = self.completed_transfer();
        if let Some(pipeline) = self.pipeline {
            pipeline
                .metrics
                .lock()
                .expect("offload transfer metrics mutex poisoned")
                .record_completion(
                    pipeline.direction,
                    (self.completed_at_ms - pipeline.started_at_ms).max(0.0),
                );
            // Ignore trigger errors — the velo event system may be shut down
            // during cleanup.
            let _ = pipeline.event.trigger();
        }
        if let Some(swap_in) = self.swap_in {
            swap_in
                .metrics
                .lock()
                .expect("offload transfer metrics mutex poisoned")
                .record_completion(
                    TransferDirection::G2ToG1,
                    (self.completed_at_ms - swap_in.started_at_ms).max(0.0),
                );
            let block_count = swap_in.status.borrow().block_count;
            swap_in
                .status
                .send_replace(SwapInStatus::completed(block_count));
        }
        completed
    }
}

/// Shared state between `MockWorker` and `MockOffloadEngine`. Both hold an
/// `Arc` clone. The single mutex keeps the two PS models and the
/// awaiter map consistent under concurrent access between the scheduler
/// thread (which calls `tick` + `set_now_ms` + any engine-level `enqueue_*`
/// helpers) and the pipeline drain worker thread (which calls
/// `execute_local_transfer`).
pub(crate) struct TransferState {
    offload_bw: BandwidthSharingModel,
    onboard_bw: BandwidthSharingModel,
    /// Pending `velo::Event`s keyed by the `TransferId` the model
    /// issued. When the model drains an id on `advance_to`, we
    /// `remove` the `Event` from this map and `trigger()` it.
    awaiters: HashMap<TransferId, PipelineAwaiter>,
    /// Terminal status publishers for swap-in reservations — kept
    /// separate from `awaiters` because swap-in does not feed a velo
    /// notification back into a kvbm-engine pipeline; the scheduler
    /// owns lifecycle directly.
    swap_in_status: HashMap<TransferId, SwapInAwaiter>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum WorkerFault {
    ExecutorError,
    ExecutorPanic,
    ChannelClosure,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DrainCounts {
    pub(crate) offload_transfers: usize,
    pub(crate) onboard_transfers: usize,
    pub(crate) offload_blocks: usize,
    pub(crate) onboard_blocks: usize,
}

impl DrainCounts {
    fn into_tuple(self) -> (usize, usize, usize, usize) {
        (
            self.offload_transfers,
            self.onboard_transfers,
            self.offload_blocks,
            self.onboard_blocks,
        )
    }

    fn add_transfer(&mut self, direction: TransferDirection, blocks: usize) {
        if direction.is_offload() {
            self.offload_transfers += 1;
            self.offload_blocks += blocks;
        } else {
            self.onboard_transfers += 1;
            self.onboard_blocks += blocks;
        }
    }

    pub(crate) fn add_counts(&mut self, other: DrainCounts) {
        self.offload_transfers += other.offload_transfers;
        self.onboard_transfers += other.onboard_transfers;
        self.offload_blocks += other.offload_blocks;
        self.onboard_blocks += other.onboard_blocks;
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DrainSummary {
    pub(crate) local: DrainCounts,
    pub(crate) shared_g3: SharedDrainCounts,
    pub(crate) shared_g4: SharedDrainCounts,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SharedDrainCounts {
    pub(crate) counts: DrainCounts,
    pub(crate) deferred_onboard_blocks: usize,
    pub(crate) offload_registration_baseline: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DeferredOwnerDrain {
    pub(crate) counts: SharedDrainCounts,
    pub(crate) deadline_ms: f64,
}

impl SharedDrainCounts {
    pub(crate) fn add_record(&mut self, record: SharedDrainCounts) {
        self.counts.add_counts(record.counts);
        self.deferred_onboard_blocks += record.deferred_onboard_blocks;
        if record.counts.offload_blocks > 0 {
            self.offload_registration_baseline = match (
                self.offload_registration_baseline,
                record.offload_registration_baseline,
            ) {
                (Some(existing), Some(incoming)) => Some(existing.min(incoming)),
                (None, incoming) => incoming,
                (existing, None) => existing,
            };
        }
    }

    pub(crate) fn add_deferred_record(&mut self, mut record: SharedDrainCounts) {
        record.deferred_onboard_blocks += record.counts.onboard_blocks;
        self.add_record(record);
    }
}

#[derive(Debug, Default)]
pub(crate) struct DrainResult {
    pub(crate) total: DrainCounts,
    pub(crate) by_owner: HashMap<u64, DrainCounts>,
    pub(crate) completed: Vec<CompletedTransfer>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CompletedTransfer {
    pub(crate) id: TransferId,
    pub(crate) owner_id: Option<u64>,
    pub(crate) direction: TransferDirection,
    pub(crate) num_blocks: usize,
    pub(crate) pipeline: bool,
}

impl DrainResult {
    pub(crate) fn record_completion(&mut self, completed: CompletedTransfer) {
        self.total
            .add_transfer(completed.direction, completed.num_blocks);
        if let Some(owner_id) = completed.owner_id {
            self.by_owner
                .entry(owner_id)
                .or_default()
                .add_transfer(completed.direction, completed.num_blocks);
        }
        if completed.pipeline {
            self.completed.push(completed);
        }
    }
}

pub(crate) fn fire_completion_actions_with(
    actions: Vec<CompletionAction>,
    mut after_pipeline_completion: impl FnMut(CompletedTransfer),
) -> DrainResult {
    let mut result = DrainResult::default();
    for action in actions {
        let completed = action.fire();
        result.record_completion(completed);
        if completed.pipeline {
            after_pipeline_completion(completed);
        }
    }
    result
}

impl TransferState {
    pub(crate) fn new(offload_gbps: f64, onboard_gbps: f64) -> Self {
        // One shared `TransferId` counter across both models so ids
        // are globally unique. `awaiters` and `swap_in_status` below are
        // single maps keyed by TransferId; per-model counters would
        // hand out overlapping ids and cause completion signals to
        // cross-fire between unrelated transfers.
        let id_counter = Arc::new(std::sync::atomic::AtomicU64::new(0));
        Self {
            offload_bw: BandwidthSharingModel::new(offload_gbps, id_counter.clone()),
            onboard_bw: BandwidthSharingModel::new(onboard_gbps, id_counter),
            awaiters: HashMap::new(),
            swap_in_status: HashMap::new(),
        }
    }

    pub(crate) fn earliest_finish(&self) -> Option<f64> {
        self.offload_bw
            .earliest_finish()
            .into_iter()
            .chain(self.onboard_bw.earliest_finish())
            .reduce(f64::min)
    }

    pub(crate) fn earliest_offload_finish(&self) -> Option<f64> {
        self.offload_bw.earliest_finish()
    }

    pub(crate) fn earliest_offload_finish_with_id(&self) -> Option<(TransferId, f64)> {
        self.offload_bw.earliest_finish_with_id()
    }

    pub(crate) fn owner_offload_active_count(&self, owner_id: u64) -> usize {
        self.awaiters
            .iter()
            .filter(|(id, awaiter)| {
                awaiter.owner_id == owner_id
                    && awaiter.direction.is_offload()
                    && self.offload_bw.is_active(**id)
            })
            .count()
    }

    pub(crate) fn owner_earliest_offload_finish_with_id(
        &self,
        owner_id: u64,
    ) -> Option<(TransferId, f64)> {
        self.awaiters
            .iter()
            .filter(|(id, awaiter)| {
                awaiter.owner_id == owner_id
                    && awaiter.direction.is_offload()
                    && self.offload_bw.is_active(**id)
            })
            .filter_map(|(id, _)| {
                self.offload_bw
                    .deadline_for(*id)
                    .map(|deadline| (*id, deadline))
            })
            .min_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.0.cmp(&right.0))
            })
    }

    pub(crate) fn earliest_onboard_finish(&self) -> Option<f64> {
        self.onboard_bw.earliest_finish()
    }

    /// Advance both models to `now_ms` and detach due completion actions.
    ///
    /// Callers that need to wait for causal publication can release the
    /// transfer-state mutex before firing these actions.
    pub(crate) fn take_completion_actions(
        &mut self,
        now_ms: f64,
        scope: &'static str,
    ) -> Vec<CompletionAction> {
        let offload_before = self.offload_bw.active_count();
        let onboard_before = self.onboard_bw.active_count();
        let offload_drained = self.offload_bw.advance_to(now_ms);
        let onboard_drained = self.onboard_bw.advance_to(now_ms);
        let offload_drained_count = offload_drained.len();
        let onboard_drained_count = onboard_drained.len();
        let drained: Vec<TransferId> = offload_drained.into_iter().chain(onboard_drained).collect();
        tracing::debug!(
            now_ms,
            offload_active_before = offload_before,
            onboard_active_before = onboard_before,
            drained_count = drained.len(),
            offload_drained_count,
            onboard_drained_count,
            awaiter_map_size = self.awaiters.len(),
            "kvbm-offload: drain transfer completions"
        );

        let mut actions = Vec::with_capacity(drained.len());
        for id in drained {
            let pipeline = self.awaiters.remove(&id);
            if let Some(awaiter) = pipeline.as_ref() {
                tracing::debug!(
                    now_ms,
                    scope,
                    transfer_id = id,
                    direction = awaiter.direction.label(),
                    blocks = awaiter.num_blocks,
                    "kvbm-offload: mock transfer complete"
                );
            }
            let swap_in = self.swap_in_status.remove(&id);
            if pipeline.is_some() || swap_in.is_some() {
                actions.push(CompletionAction {
                    id,
                    pipeline,
                    swap_in,
                    completed_at_ms: now_ms,
                });
            }
        }
        tracing::debug!(
            completion_actions = actions.len(),
            "kvbm-offload: detached completed transfer waiters"
        );
        actions
    }

    /// Advance both models to `now_ms` under PS and notify any completion
    /// sinks registered for drained `TransferId`s.
    pub(crate) fn drain_completions(&mut self, now_ms: f64, scope: &'static str) -> DrainResult {
        let actions = self.take_completion_actions(now_ms, scope);
        fire_completion_actions_with(actions, |_| {})
    }
}

static NEXT_WORKER_ID: AtomicU64 = AtomicU64::new(1);

// Store `now_ms` as integer microseconds so it can round-trip through
// `AtomicU64`. Microsecond precision is more than sufficient for the
// mocker's ms+ tick cadence.
fn ms_to_us(ms: f64) -> u64 {
    (ms.max(0.0) * 1000.0) as u64
}
fn us_to_ms(us: u64) -> f64 {
    (us as f64) / 1000.0
}

/// Mock implementation of kvbm-engine's `Worker`, `WorkerTransfers`, and
/// `ObjectBlockOps` traits. Simulates G1↔G2 transfer timing via a local
/// processor-sharing bandwidth model, and G2↔G3/G2↔G4 via shared ones;
/// never touches real memory.
///
/// Live mode preserves eager completion draining during reservations. Offline
/// mode disables those opportunistic drains so the engine's explicit drive
/// boundary owns completion ordering and publication.
pub struct MockWorker {
    drive_mode: KvbmDriveMode,
    /// Stable process-local owner id. Shared G3 completion accounting uses
    /// this to return drained transfer counts to the worker that reserved
    /// the transfer, even if a different worker's tick advanced the shared
    /// bandwidth queue first.
    owner_id: u64,
    /// Current simulation time, in integer microseconds. Engine stores via
    /// `set_now_ms`; worker reads via `now_ms` from inside the pipeline
    /// drain task.
    now_us: Arc<AtomicU64>,
    /// Shared transfer state. Cloned into `MockOffloadEngine` so its
    /// `tick(now_ms)` can drive completions.
    pub(crate) state: Arc<Mutex<TransferState>>,
    /// Event system for creating completion awaiters.
    event_manager: EventManager,
    /// Monotonic count of pipeline transfer reservations accepted by
    /// `reserve_transfer`. Offline replay uses this as a concrete barrier:
    /// enqueue should not let virtual time jump until the worker has
    /// actually reserved the simulated bandwidth slot.
    reservation_count_tx: watch::Sender<u64>,
    /// Bytes per block — used to derive transfer size from block-id counts.
    block_bytes: usize,
    /// Optional AIC-derived byte size indexed by logical block position.
    /// This supersedes `block_bytes` when transfer metadata carries positions.
    block_footprint_bytes: Option<Arc<[usize]>>,
    transfer_metrics: Arc<Mutex<OffloadTransferMetrics>>,
    g1_handle: Option<LayoutHandle>,
    g2_handle: Option<LayoutHandle>,
    shared_g2: Option<Arc<SharedG2Fabric>>,
    shared_g3: Option<Arc<SharedG3Pool>>,
    shared_g4: Option<Arc<SharedG4Store>>,
    #[cfg(test)]
    fault_script: Mutex<VecDeque<WorkerFault>>,
    #[cfg(test)]
    injected_staging_failure: Mutex<Option<Arc<str>>>,
}

impl MockWorker {
    #[cfg(test)]
    /// Build a new `MockWorker`.
    ///
    /// `offload_gbps` and `onboard_gbps` are throughput caps for the G1→G2
    /// and G2→G1 links respectively; zero or negative values mean
    /// "infinite bandwidth" (transfers complete instantly on next tick).
    /// `block_bytes` is how many bytes each simulation block represents —
    /// typically `block_size * kv_bytes_per_token`.
    pub(crate) fn new(
        block_bytes: usize,
        offload_gbps: f64,
        onboard_gbps: f64,
        g1_handle: Option<LayoutHandle>,
        g2_handle: Option<LayoutHandle>,
        shared_g3: Option<Arc<SharedG3Pool>>,
        shared_g4: Option<Arc<SharedG4Store>>,
    ) -> Self {
        Self::new_with_drive_mode(
            block_bytes,
            offload_gbps,
            onboard_gbps,
            g1_handle,
            g2_handle,
            shared_g3,
            shared_g4,
            KvbmDriveMode::Live,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_with_drive_mode(
        block_bytes: usize,
        offload_gbps: f64,
        onboard_gbps: f64,
        g1_handle: Option<LayoutHandle>,
        g2_handle: Option<LayoutHandle>,
        shared_g3: Option<Arc<SharedG3Pool>>,
        shared_g4: Option<Arc<SharedG4Store>>,
        drive_mode: KvbmDriveMode,
    ) -> Self {
        Self::new_with_footprint(
            block_bytes,
            None,
            offload_gbps,
            onboard_gbps,
            g1_handle,
            g2_handle,
            shared_g3,
            shared_g4,
            drive_mode,
        )
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_footprint(
        block_bytes: usize,
        block_footprint_bytes: Option<Arc<[usize]>>,
        offload_gbps: f64,
        onboard_gbps: f64,
        g1_handle: Option<LayoutHandle>,
        g2_handle: Option<LayoutHandle>,
        shared_g3: Option<Arc<SharedG3Pool>>,
        shared_g4: Option<Arc<SharedG4Store>>,
        drive_mode: KvbmDriveMode,
    ) -> Self {
        Self::new_with_footprint_and_fabric(
            block_bytes,
            block_footprint_bytes,
            offload_gbps,
            onboard_gbps,
            g1_handle,
            g2_handle,
            None,
            shared_g3,
            shared_g4,
            drive_mode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_footprint_and_fabric(
        block_bytes: usize,
        block_footprint_bytes: Option<Arc<[usize]>>,
        offload_gbps: f64,
        onboard_gbps: f64,
        g1_handle: Option<LayoutHandle>,
        g2_handle: Option<LayoutHandle>,
        shared_g2: Option<Arc<SharedG2Fabric>>,
        shared_g3: Option<Arc<SharedG3Pool>>,
        shared_g4: Option<Arc<SharedG4Store>>,
        drive_mode: KvbmDriveMode,
    ) -> Self {
        Self {
            drive_mode,
            owner_id: NEXT_WORKER_ID.fetch_add(1, Ordering::Relaxed),
            now_us: Arc::new(AtomicU64::new(0)),
            state: Arc::new(Mutex::new(TransferState::new(offload_gbps, onboard_gbps))),
            event_manager: EventManager::local(),
            reservation_count_tx: watch::channel(0).0,
            block_bytes,
            block_footprint_bytes,
            transfer_metrics: Arc::new(Mutex::new(OffloadTransferMetrics::default())),
            g1_handle,
            g2_handle,
            shared_g2,
            shared_g3,
            shared_g4,
            #[cfg(test)]
            fault_script: Mutex::new(VecDeque::new()),
            #[cfg(test)]
            injected_staging_failure: Mutex::new(None),
        }
    }

    fn transfer_bytes(&self, num_blocks: usize, positions: Option<&[u64]>) -> Result<usize> {
        let Some(profile) = self.block_footprint_bytes.as_ref() else {
            return Ok(num_blocks.saturating_mul(self.block_bytes));
        };
        let positions = positions.ok_or_else(|| {
            anyhow!("AIC dynamic KV footprint requires logical block positions for every transfer")
        })?;
        if positions.len() != num_blocks {
            bail!(
                "KV transfer position count {} does not match block count {}",
                positions.len(),
                num_blocks
            );
        }
        positions.iter().try_fold(0usize, |total, position| {
            let bytes = profile.get(*position as usize).copied().ok_or_else(|| {
                anyhow!(
                    "AIC KV footprint profile has no entry for logical block position {}",
                    position
                )
            })?;
            total
                .checked_add(bytes)
                .ok_or_else(|| anyhow!("KV transfer byte count overflow"))
        })
    }

    fn block_bytes_at(&self, position: u64) -> Result<usize> {
        match self.block_footprint_bytes.as_ref() {
            Some(profile) => profile.get(position as usize).copied().ok_or_else(|| {
                anyhow!(
                    "AIC KV footprint profile has no entry for logical block position {}",
                    position
                )
            }),
            None if self.block_bytes > 0 => Ok(self.block_bytes),
            None => bail!("KV block byte size is unresolved"),
        }
    }

    pub(crate) fn transfer_metrics(&self) -> OffloadTransferMetrics {
        *self
            .transfer_metrics
            .lock()
            .expect("offload transfer metrics mutex poisoned")
    }

    #[cfg(test)]
    pub(crate) fn inject_fault(&self, fault: WorkerFault) {
        self.fault_script
            .lock()
            .expect("worker fault script mutex poisoned")
            .push_back(fault);
    }

    #[cfg(test)]
    fn take_fault(&self) -> Option<WorkerFault> {
        self.fault_script
            .lock()
            .expect("worker fault script mutex poisoned")
            .pop_front()
    }

    #[cfg(test)]
    pub(crate) fn take_injected_staging_failure(&self) -> Option<Arc<str>> {
        self.injected_staging_failure
            .lock()
            .expect("injected staging failure mutex poisoned")
            .take()
    }

    /// Update the worker's notion of current simulation time. Engine calls
    /// this at the start of `tick(now_ms)` and before every enqueue/start
    /// operation so transfer reservation reads a fresh `now_ms`
    /// inside `execute_local_transfer`.
    pub fn set_now_ms(&self, now_ms: f64) {
        self.now_us.store(ms_to_us(now_ms), Ordering::Release);
    }

    pub fn now_ms(&self) -> f64 {
        us_to_ms(self.now_us.load(Ordering::Acquire))
    }

    pub fn reservation_count(&self) -> u64 {
        *self.reservation_count_tx.borrow()
    }

    pub(crate) fn subscribe_reservation_count(&self) -> watch::Receiver<u64> {
        self.reservation_count_tx.subscribe()
    }

    /// Advance both models to `now_ms` under PS and notify any
    /// completion sinks registered for drained `TransferId`s: `velo::Event`
    /// awaiters (for kvbm-engine pipeline transfers) and watch statuses
    /// (for swap-in reservations polled by the scheduler). Called
    /// from `MockOffloadEngine::tick` and implicitly before every new
    /// reservation — both uses need the model's active set to
    /// reflect completed transfers at the queried time.
    pub fn drain_completions(&self, now_ms: f64) -> (usize, usize, usize, usize) {
        self.drain_completions_summary(now_ms).local.into_tuple()
    }

    pub(crate) fn drain_completions_summary(&self, now_ms: f64) -> DrainSummary {
        if self.drive_mode == KvbmDriveMode::OfflineDeterministic {
            return self.drain_completions_summary_ordered(now_ms, |_| {}, |_| {}, |_| {});
        }
        let local = self
            .shared_g2
            .as_ref()
            .map(|fabric| fabric.drain_completions(now_ms, self.owner_id).counts)
            .unwrap_or_else(|| self.drain_local_completions_with(now_ms, |_| {}).total);
        let shared_g3 = self
            .shared_g3
            .as_ref()
            .map(|shared_g3| shared_g3.drain_completions(now_ms, self.owner_id))
            .unwrap_or_default();
        let shared_g4 = self
            .shared_g4
            .as_ref()
            .map(|shared_g4| shared_g4.drain_completions(now_ms, self.owner_id))
            .unwrap_or_default();
        DrainSummary {
            local,
            shared_g3,
            shared_g4,
        }
    }

    pub(crate) fn drain_completions_summary_ordered(
        &self,
        now_ms: f64,
        mut after_local_completion: impl FnMut(CompletedTransfer),
        after_shared_g3_completion: impl FnMut(CompletedTransfer),
        after_shared_g4_completion: impl FnMut(CompletedTransfer),
    ) -> DrainSummary {
        let local = if let Some(fabric) = self.shared_g2.as_ref() {
            fabric
                .drain_completions_ordered(now_ms, self.owner_id, &mut after_local_completion)
                .counts
        } else {
            self.drain_local_completions_with(now_ms, after_local_completion)
                .total
        };
        let shared_g3 = self
            .shared_g3
            .as_ref()
            .map(|shared_g3| {
                shared_g3.drain_completions_ordered(
                    now_ms,
                    self.owner_id,
                    after_shared_g3_completion,
                )
            })
            .unwrap_or_default();
        let shared_g4 = self
            .shared_g4
            .as_ref()
            .map(|shared_g4| {
                shared_g4.drain_completions_ordered(
                    now_ms,
                    self.owner_id,
                    after_shared_g4_completion,
                )
            })
            .unwrap_or_default();
        DrainSummary {
            local,
            shared_g3,
            shared_g4,
        }
    }

    pub(crate) fn drain_local_completions_with(
        &self,
        now_ms: f64,
        after_pipeline_completion: impl FnMut(CompletedTransfer),
    ) -> DrainResult {
        let actions = {
            let mut state = self.state.lock().expect("TransferState mutex poisoned");
            state.take_completion_actions(now_ms, "worker")
        };
        fire_completion_actions_with(actions, after_pipeline_completion)
    }

    /// Reserve an onboard (G2→G1) transfer whose completion is published
    /// through `status` when `MockOffloadEngine::tick` (or any drain path)
    /// drains the reservation from the PS model.
    pub(crate) fn reserve_swap_in(
        &self,
        now_ms: f64,
        num_blocks: usize,
        positions: Option<&[u64]>,
        status: watch::Sender<SwapInStatus>,
    ) -> TransferId {
        let bytes = self
            .transfer_bytes(num_blocks, positions)
            .expect("validated AIC KV footprint must cover every swap-in block");
        let (state_arc, already_drained) = if let Some(fabric) = self.shared_g2.as_ref() {
            if self.drive_mode == KvbmDriveMode::OfflineDeterministic {
                fabric.defer_completions_ordered(now_ms);
            } else {
                fabric.drain_completions_to_pending(now_ms);
            }
            (fabric.transfer_state(), true)
        } else {
            (self.state.clone(), false)
        };
        let mut state = state_arc.lock().expect("TransferState mutex poisoned");
        if !already_drained && self.drive_mode == KvbmDriveMode::Live {
            state.drain_completions(now_ms, "worker");
        }
        let id = state.onboard_bw.start_transfer(now_ms, bytes);
        self.transfer_metrics
            .lock()
            .expect("offload transfer metrics mutex poisoned")
            .record(TransferDirection::G2ToG1, num_blocks, bytes);
        let next_deadline_ms = state.onboard_bw.earliest_finish();
        tracing::debug!(
            now_ms,
            scope = "worker",
            transfer_id = id,
            direction = TransferDirection::G2ToG1.label(),
            blocks = num_blocks,
            bytes,
            next_deadline_ms = ?next_deadline_ms,
            "kvbm-offload: reserve mock swap-in transfer"
        );
        state.swap_in_status.insert(
            id,
            SwapInAwaiter {
                owner_id: self.owner_id,
                num_blocks,
                status,
                started_at_ms: now_ms,
                metrics: self.transfer_metrics.clone(),
            },
        );
        id
    }

    /// Earliest pending deadline across both link models. `None` if
    /// both are idle. Used by the scheduler's stall-advance.
    pub fn earliest_finish(&self) -> Option<f64> {
        let state = self.state.lock().expect("TransferState mutex poisoned");
        let rank_local = state.earliest_finish();
        drop(state);
        let local = self
            .shared_g2
            .as_ref()
            .and_then(|fabric| fabric.earliest_finish())
            .or(rank_local);
        let pending_g2_deadline = self
            .shared_g2
            .as_ref()
            .and_then(|fabric| fabric.pending_owner_deadline(self.owner_id));
        let pending_g3_deadline = self
            .shared_g3
            .as_ref()
            .and_then(|g3| g3.pending_owner_deadline(self.owner_id));
        let pending_g4_deadline = self
            .shared_g4
            .as_ref()
            .and_then(|g4| g4.pending_owner_deadline(self.owner_id));
        local
            .into_iter()
            .chain(pending_g2_deadline)
            .chain(self.shared_g3.as_ref().and_then(|g3| g3.earliest_finish()))
            .chain(self.shared_g4.as_ref().and_then(|g4| g4.earliest_finish()))
            .chain(pending_g3_deadline)
            .chain(pending_g4_deadline)
            .reduce(f64::min)
    }

    pub(crate) fn earliest_local_offload_finish_with_id(&self) -> Option<(TransferId, f64)> {
        if let Some(fabric) = self.shared_g2.as_ref() {
            return fabric.earliest_offload_finish_with_id(self.owner_id);
        }
        let state = self.state.lock().expect("TransferState mutex poisoned");
        state.earliest_offload_finish_with_id()
    }

    pub(crate) fn local_offload_active_count(&self) -> usize {
        if let Some(fabric) = self.shared_g2.as_ref() {
            return fabric.offload_active_count(self.owner_id);
        }
        let state = self.state.lock().expect("TransferState mutex poisoned");
        state.offload_bw.active_count()
    }

    pub(crate) fn earliest_shared_g3_offload_finish(&self) -> Option<f64> {
        self.shared_g3
            .as_ref()
            .and_then(|g3| g3.earliest_offload_finish())
    }

    pub(crate) fn earliest_shared_g4_offload_finish(&self) -> Option<f64> {
        self.shared_g4
            .as_ref()
            .and_then(|g4| g4.earliest_offload_finish())
    }

    /// Earliest deadline that can unblock scheduler-visible foreground work.
    ///
    /// Background G2→G3 copies are intentionally excluded: offline replay
    /// should drain them when time reaches an existing arrival/worker/swap-in
    /// timestamp, but they should not create an extra scheduling timestamp by
    /// themselves.
    pub fn earliest_foreground_finish(&self) -> Option<f64> {
        let state = self.state.lock().expect("TransferState mutex poisoned");
        let rank_local = state.earliest_finish();
        drop(state);
        let local = self
            .shared_g2
            .as_ref()
            .and_then(|fabric| fabric.earliest_foreground_finish())
            .or(rank_local);
        local
            .into_iter()
            .chain(
                self.shared_g2
                    .as_ref()
                    .and_then(|fabric| fabric.pending_owner_deadline(self.owner_id)),
            )
            .chain(
                self.shared_g3
                    .as_ref()
                    .and_then(|g3| g3.earliest_onboard_finish()),
            )
            .chain(
                self.shared_g4
                    .as_ref()
                    .and_then(|g4| g4.earliest_onboard_finish()),
            )
            .reduce(f64::min)
    }

    /// Reserve a pipeline transfer on the requested link, register a
    /// velo `Event` in the shared awaiter map, and return a
    /// `TransferCompleteNotification` backed by that event's awaiter.
    fn reserve_transfer(
        &self,
        direction: TransferDirection,
        now_ms: f64,
        num_blocks: usize,
        positions: Option<&[u64]>,
    ) -> Result<TransferCompleteNotification> {
        self.reserve_transfer_with(direction, now_ms, num_blocks, positions, |_| {})
            .map(|(_id, notification)| notification)
    }

    fn reserve_transfer_with(
        &self,
        direction: TransferDirection,
        now_ms: f64,
        num_blocks: usize,
        positions: Option<&[u64]>,
        on_reserved: impl FnOnce(TransferId),
    ) -> Result<(TransferId, TransferCompleteNotification)> {
        let bytes = self.transfer_bytes(num_blocks, positions)?;
        let (state_arc, scope, already_drained) =
            if direction.is_g3() {
                let shared_g3 = self.shared_g3.as_ref().ok_or_else(|| {
                    anyhow!("MockWorker: G2↔G3 transfer requested without shared G3")
                })?;
                if self.drive_mode == KvbmDriveMode::OfflineDeterministic {
                    shared_g3.defer_completions_ordered(now_ms);
                } else {
                    shared_g3.drain_completions_to_pending(now_ms);
                }
                (shared_g3.transfer_state(), "shared-g3", true)
            } else if direction.is_g4() {
                let shared_g4 = self.shared_g4.as_ref().ok_or_else(|| {
                    anyhow!("MockWorker: G2↔G4 transfer requested without shared G4")
                })?;
                if self.drive_mode == KvbmDriveMode::OfflineDeterministic {
                    shared_g4.defer_completions_ordered(now_ms);
                } else {
                    shared_g4.drain_completions_to_pending(now_ms);
                }
                (shared_g4.transfer_state(), "shared-g4", true)
            } else if let Some(shared_g2) = self.shared_g2.as_ref() {
                if self.drive_mode == KvbmDriveMode::OfflineDeterministic {
                    shared_g2.defer_completions_ordered(now_ms);
                } else {
                    shared_g2.drain_completions_to_pending(now_ms);
                }
                (shared_g2.transfer_state(), "shared-g2", true)
            } else {
                (self.state.clone(), "worker", false)
            };
        let mut state = state_arc.lock().expect("TransferState mutex poisoned");
        if !already_drained && self.drive_mode == KvbmDriveMode::Live {
            state.drain_completions(now_ms, scope);
        }

        let (id, next_deadline_ms) = if direction.is_offload() {
            let id = state.offload_bw.start_transfer(now_ms, bytes);
            (id, state.offload_bw.earliest_finish())
        } else {
            let id = state.onboard_bw.start_transfer(now_ms, bytes);
            (id, state.onboard_bw.earliest_finish())
        };
        self.transfer_metrics
            .lock()
            .expect("offload transfer metrics mutex poisoned")
            .record(direction, num_blocks, bytes);
        tracing::debug!(
            now_ms,
            scope,
            transfer_id = id,
            direction = direction.label(),
            blocks = num_blocks,
            bytes,
            next_deadline_ms = ?next_deadline_ms,
            "kvbm-offload: reserve mock transfer"
        );
        self.reservation_count_tx.send_modify(|count| {
            *count = count.checked_add(1).expect("reservation count overflow");
        });

        // Allocate a velo event + awaiter. Store the `Event` so we can
        // `trigger()` it later (triggering consumes `self`).
        let event = self
            .event_manager
            .new_event()
            .map_err(|e| anyhow!("MockWorker: failed to allocate velo event: {e}"))?;
        let awaiter = event
            .awaiter()
            .map_err(|e| anyhow!("MockWorker: failed to build event awaiter: {e}"))?;
        state.awaiters.insert(
            id,
            PipelineAwaiter {
                event,
                owner_id: self.owner_id,
                direction,
                num_blocks,
                started_at_ms: now_ms,
                metrics: self.transfer_metrics.clone(),
            },
        );
        on_reserved(id);
        drop(state);
        Ok((id, TransferCompleteNotification::from_awaiter(awaiter)))
    }
}

impl WorkerTransfers for MockWorker {
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        _dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let direction = TransferDirection::try_from((src, dst))?;
        #[cfg(test)]
        match self.take_fault() {
            Some(WorkerFault::ExecutorError) => {
                bail!("injected mock transfer executor error")
            }
            Some(WorkerFault::ExecutorPanic) => {
                panic!("injected mock transfer executor panic")
            }
            Some(WorkerFault::ChannelClosure) => {
                if direction == TransferDirection::G3ToG2 {
                    self.injected_staging_failure
                        .lock()
                        .expect("injected staging failure mutex poisoned")
                        .replace(Arc::from("injected lower-tier staging channel closure"));
                }
                let event = self.event_manager.new_event()?;
                let awaiter = event.awaiter()?;
                let _ = event.poison("injected mock transfer notification failure");
                return Ok(TransferCompleteNotification::from_awaiter(awaiter));
            }
            None => {}
        }
        let now_ms = self.now_ms();
        self.reserve_transfer(
            direction,
            now_ms,
            src_block_ids.len(),
            options.block_positions.as_deref(),
        )
    }

    fn execute_remote_onboard(
        &self,
        _src: RemoteDescriptor,
        _dst: LogicalLayoutHandle,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        bail!(
            "MockWorker: execute_remote_onboard not supported (mocker simulates local G1↔G2/G2↔G3 only)"
        )
    }

    fn execute_remote_offload(
        &self,
        _src: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst: RemoteDescriptor,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        bail!("MockWorker: execute_remote_offload not supported")
    }

    fn connect_remote(
        &self,
        _instance_id: InstanceId,
        _metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        bail!("MockWorker: connect_remote not supported")
    }

    fn has_remote_metadata(&self, _instance_id: InstanceId) -> bool {
        false
    }

    fn execute_remote_onboard_for_instance(
        &self,
        _instance_id: InstanceId,
        _remote_logical_type: LogicalLayoutHandle,
        _src_block_ids: Vec<BlockId>,
        _dst: LogicalLayoutHandle,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        bail!("MockWorker: execute_remote_onboard_for_instance not supported")
    }
}

impl Worker for MockWorker {
    fn g1_handle(&self) -> Option<LayoutHandle> {
        self.g1_handle
    }

    fn g2_handle(&self) -> Option<LayoutHandle> {
        self.g2_handle
    }

    fn g3_handle(&self) -> Option<LayoutHandle> {
        None
    }

    fn export_metadata(&self) -> Result<SerializedLayoutResponse> {
        bail!("MockWorker: export_metadata not supported (mocker is single-instance)")
    }

    fn import_metadata(&self, _metadata: SerializedLayout) -> Result<ImportMetadataResponse> {
        bail!("MockWorker: import_metadata not supported (mocker is single-instance)")
    }
}

// Mock implementation of the G4 ObjectBlockOps trait. This models object
// presence and G2<->G4 transfer bandwidth only; object-store request latency,
// HEAD/list overhead, retries, failures, and consistency effects are outside
// this mocker's fidelity boundary.
impl ObjectBlockOps for MockWorker {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        let shared_g4 = self.shared_g4.clone();
        Box::pin(async move {
            keys.into_iter()
                .map(|key| {
                    let size = shared_g4.as_ref().and_then(|store| store.has_object(&key));
                    (key, size)
                })
                .collect()
        })
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        src_layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        if keys.is_empty() {
            return Box::pin(async move { Vec::new() });
        }
        if src_layout != LogicalLayoutHandle::G2 || keys.len() != block_ids.len() {
            tracing::warn!(
                ?src_layout,
                keys = keys.len(),
                block_ids = block_ids.len(),
                "kvbm-offload: rejected mock G2→G4 object put"
            );
            return Box::pin(async move { keys.into_iter().map(Err).collect() });
        }
        let Some(shared_g4) = self.shared_g4.clone() else {
            tracing::warn!("kvbm-offload: put_blocks called without shared G4 store");
            return Box::pin(async move { keys.into_iter().map(Err).collect() });
        };
        let positions = keys.iter().map(SequenceHash::position).collect::<Vec<_>>();
        let sizes_bytes = match positions
            .iter()
            .map(|position| self.block_bytes_at(*position))
            .collect::<Result<Vec<_>>>()
        {
            Ok(sizes) => sizes,
            Err(error) => {
                tracing::warn!(error = %error, "kvbm-offload: unresolved G4 object sizes");
                return Box::pin(async move { keys.into_iter().map(Err).collect() });
            }
        };
        let publish_keys = keys.clone();
        let publish_sizes = sizes_bytes.clone();
        let publish_store = shared_g4.clone();
        let notification = match self.reserve_transfer_with(
            TransferDirection::G2ToG4,
            self.now_ms(),
            keys.len(),
            Some(&positions),
            move |transfer_id| {
                publish_store.register_pending_put(transfer_id, publish_keys, publish_sizes);
            },
        ) {
            Ok((_transfer_id, notification)) => notification,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "kvbm-offload: failed to reserve mock G2→G4 object put"
                );
                return Box::pin(async move { keys.into_iter().map(Err).collect() });
            }
        };
        Box::pin(async move {
            match notification.await {
                // SharedG4Store publishes the pending objects before firing
                // the completion event. Publishing again here would turn one
                // physical admission into two metric events and perturb LRU.
                Ok(()) => keys.into_iter().map(Ok).collect(),
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "kvbm-offload: mock G2→G4 object put did not complete"
                    );
                    keys.into_iter().map(Err).collect()
                }
            }
        })
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        dst_layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        if keys.is_empty() {
            return Box::pin(async move { Vec::new() });
        }
        if dst_layout != LogicalLayoutHandle::G2 || keys.len() != block_ids.len() {
            tracing::warn!(
                ?dst_layout,
                keys = keys.len(),
                block_ids = block_ids.len(),
                "kvbm-offload: rejected mock G4→G2 object get"
            );
            return Box::pin(async move { keys.into_iter().map(Err).collect() });
        }
        let Some(shared_g4) = self.shared_g4.clone() else {
            tracing::warn!("kvbm-offload: get_blocks called without shared G4 store");
            return Box::pin(async move { keys.into_iter().map(Err).collect() });
        };
        let present: Vec<bool> = keys
            .iter()
            .map(|key| shared_g4.has_object(key).is_some())
            .collect();
        let present_count = present.iter().filter(|found| **found).count();
        if present_count == 0 {
            return Box::pin(async move { keys.into_iter().map(Err).collect() });
        }
        let present_positions = keys
            .iter()
            .zip(&present)
            .filter_map(|(key, found)| found.then_some(key.position()))
            .collect::<Vec<_>>();
        let notification = match self.reserve_transfer(
            TransferDirection::G4ToG2,
            self.now_ms(),
            present_count,
            Some(&present_positions),
        ) {
            Ok(notification) => notification,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "kvbm-offload: failed to reserve mock G4→G2 object get"
                );
                return Box::pin(async move { keys.into_iter().map(Err).collect() });
            }
        };
        Box::pin(async move {
            match notification.await {
                Ok(()) => keys
                    .into_iter()
                    .zip(present)
                    .map(|(key, found)| if found { Ok(key) } else { Err(key) })
                    .collect(),
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "kvbm-offload: mock G4→G2 object get did not complete"
                    );
                    keys.into_iter().map(Err).collect()
                }
            }
        })
    }

    fn put_blocks_with_layout(
        &self,
        keys: Vec<SequenceHash>,
        _layout: PhysicalLayout,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    fn get_blocks_with_layout(
        &self,
        keys: Vec<SequenceHash>,
        _layout: PhysicalLayout,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::KvbmOffloadConfig;
    use super::super::coordinator::SwapInTerminal;
    use super::super::shared_g2::{SharedG2Fabric, shared_g2_test_guard};
    use super::super::shared_g3::shared_g3_test_guard;
    use super::super::shared_g4::{SharedG4Store, shared_g4_test_guard};
    use super::*;

    const EPS: f64 = 1e-6;

    fn make_worker() -> MockWorker {
        // 1 GB/s bandwidth on both links, 1 MB per block.
        MockWorker::new(1_000_000, 1.0, 1.0, None, None, None, None)
    }

    fn shared_g2_two_workers() -> (MockWorker, MockWorker) {
        let config = KvbmOffloadConfig {
            g2_fabric_id: Some("worker-test-node-0".to_string()),
            bandwidth_g1_to_g2_gbps: 1.0,
            bandwidth_g2_to_g1_gbps: 1.0,
            ..Default::default()
        };
        let fabric =
            SharedG2Fabric::get_or_create_with_mode(&config, KvbmDriveMode::OfflineDeterministic)
                .unwrap()
                .expect("shared G2 fabric enabled");
        (
            MockWorker::new_with_footprint_and_fabric(
                1_000_000,
                None,
                1.0,
                1.0,
                None,
                None,
                Some(fabric.clone()),
                None,
                None,
                KvbmDriveMode::OfflineDeterministic,
            ),
            MockWorker::new_with_footprint_and_fabric(
                1_000_000,
                None,
                1.0,
                1.0,
                None,
                None,
                Some(fabric),
                None,
                None,
                KvbmDriveMode::OfflineDeterministic,
            ),
        )
    }

    fn shared_g3_two_workers() -> (MockWorker, MockWorker) {
        let config = KvbmOffloadConfig {
            num_g3_blocks: Some(128),
            block_size_bytes: Some(1_000_000),
            bandwidth_g2_to_g3_gbps: 1.0,
            bandwidth_g3_to_g2_gbps: 1.0,
            ..Default::default()
        };
        let shared_g3 = SharedG3Pool::get_or_create(&config).unwrap();
        (
            MockWorker::new(1_000_000, 1.0, 1.0, None, None, shared_g3.clone(), None),
            MockWorker::new(1_000_000, 1.0, 1.0, None, None, shared_g3, None),
        )
    }

    fn reserve_shared_g3(worker: &MockWorker, blocks: usize) {
        let shared_g3 = worker
            .shared_g3
            .as_ref()
            .expect("test worker should have shared G3");
        assert!(
            shared_g3
                .capacity_reservations()
                .try_reserve(shared_g3.manager().available_blocks(), blocks)
        );
    }

    fn shared_g4_worker() -> (MockWorker, Arc<SharedG4Store>) {
        let config = KvbmOffloadConfig {
            enable_g4_storage: true,
            g4_capacity_bytes: Some(64_000_000),
            block_size_bytes: Some(1_000_000),
            bandwidth_g2_to_g4_gbps: 1.0,
            bandwidth_g4_to_g2_gbps: 1.0,
            ..Default::default()
        };
        let shared_g4 = SharedG4Store::get_or_create(&config)
            .unwrap()
            .expect("G4 enabled");
        (
            MockWorker::new(
                1_000_000,
                1.0,
                1.0,
                None,
                None,
                None,
                Some(shared_g4.clone()),
            ),
            shared_g4,
        )
    }

    fn shared_g4_two_workers() -> (MockWorker, MockWorker, Arc<SharedG4Store>) {
        let config = KvbmOffloadConfig {
            enable_g4_storage: true,
            g4_capacity_bytes: Some(64_000_000),
            block_size_bytes: Some(1_000_000),
            bandwidth_g2_to_g4_gbps: 1.0,
            bandwidth_g4_to_g2_gbps: 1.0,
            ..Default::default()
        };
        let shared_g4 = SharedG4Store::get_or_create(&config)
            .unwrap()
            .expect("G4 enabled");
        (
            MockWorker::new(
                1_000_000,
                1.0,
                1.0,
                None,
                None,
                None,
                Some(shared_g4.clone()),
            ),
            MockWorker::new(
                1_000_000,
                1.0,
                1.0,
                None,
                None,
                None,
                Some(shared_g4.clone()),
            ),
            shared_g4,
        )
    }

    #[tokio::test]
    async fn mock_worker_single_transfer_completes_on_tick() {
        // A single G1→G2 transfer reserved at t=0 should complete after
        // drain_completions is called with now_ms >= bytes/bandwidth.
        let worker = make_worker();
        worker.set_now_ms(0.0);
        let src_ids: Arc<[BlockId]> = Arc::from(vec![0usize]);
        let dst_ids: Arc<[BlockId]> = Arc::from(vec![0usize]);
        let notification = worker
            .execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                src_ids,
                dst_ids,
                TransferOptions::default(),
            )
            .expect("reservation should succeed");
        // Before drain, the notification is pending (would yield).
        assert!(notification.could_yield());

        // Advance virtual clock past the transfer's finish time and drain.
        worker.drain_completions(1.0);

        // The notification should now resolve immediately.
        notification
            .await
            .expect("transfer notification should resolve Ok after drain");
    }

    #[tokio::test]
    async fn shared_g2_fabric_contends_across_ranks_and_fires_only_at_owner_boundary() {
        let _guard = shared_g2_test_guard().await;
        let (worker_a, worker_b) = shared_g2_two_workers();
        let ids: Arc<[BlockId]> = Arc::from(vec![0usize]);
        worker_a.set_now_ms(0.0);
        worker_b.set_now_ms(0.0);
        let notification_a = worker_a
            .execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                ids.clone(),
                ids.clone(),
                TransferOptions::default(),
            )
            .unwrap();
        let notification_b = worker_b
            .execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                ids.clone(),
                ids,
                TransferOptions::default(),
            )
            .unwrap();

        // Two 1 MB transfers sharing one 1 GB/s fabric each finish at 2 ms.
        let mut notification_a = Box::pin(notification_a.into_future());
        let mut notification_b = Box::pin(notification_b.into_future());
        assert_eq!(
            worker_a
                .drain_completions_summary(1.999)
                .local
                .offload_transfers,
            0
        );
        assert!(matches!(
            futures::poll!(&mut notification_a),
            std::task::Poll::Pending
        ));
        assert!(matches!(
            futures::poll!(&mut notification_b),
            std::task::Poll::Pending
        ));

        // Advancing A detaches both due completions, but offline ownership
        // keeps B's notification pending until B reaches its own boundary.
        assert_eq!(
            worker_a
                .drain_completions_summary(2.0)
                .local
                .offload_transfers,
            1
        );
        assert!(matches!(
            futures::poll!(&mut notification_a),
            std::task::Poll::Ready(Ok(()))
        ));
        assert!(matches!(
            futures::poll!(&mut notification_b),
            std::task::Poll::Pending
        ));
        assert_eq!(
            worker_b
                .drain_completions_summary(2.0)
                .local
                .offload_transfers,
            1
        );
        assert!(matches!(
            futures::poll!(&mut notification_b),
            std::task::Poll::Ready(Ok(()))
        ));
        for worker in [&worker_a, &worker_b] {
            let metrics = worker.transfer_metrics().g1_to_g2;
            assert_eq!(metrics.transfers, 1);
            assert_eq!(metrics.completed_transfers, 1);
            assert_eq!(metrics.total_transfer_time_us, 2_000);
            assert_eq!(metrics.total_queue_wait_us, 0);
        }
    }

    #[tokio::test]
    async fn mock_worker_dynamic_footprint_charges_each_logical_position() {
        let worker = MockWorker::new_with_footprint(
            0,
            Some(Arc::from(vec![1_000_000usize, 3_000_000usize])),
            1.0,
            1.0,
            None,
            None,
            None,
            None,
            KvbmDriveMode::Live,
        );
        worker.set_now_ms(0.0);
        let ids: Arc<[BlockId]> = Arc::from(vec![0usize, 1usize]);
        let options = TransferOptions::builder()
            .block_positions(Arc::from(vec![0u64, 1u64]))
            .build()
            .expect("position metadata should build");
        let notification = worker
            .execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                ids.clone(),
                ids,
                options,
            )
            .expect("dynamic transfer should reserve");

        worker.drain_completions(3.999);
        assert!(notification.could_yield());
        worker.drain_completions(4.0);
        notification
            .await
            .expect("four MB at one GB/s should finish at four ms");
    }

    #[test]
    fn mock_worker_dynamic_footprint_fails_closed_without_positions() {
        let worker = MockWorker::new_with_footprint(
            0,
            Some(Arc::from(vec![1_000_000usize])),
            1.0,
            1.0,
            None,
            None,
            None,
            None,
            KvbmDriveMode::Live,
        );
        let ids: Arc<[BlockId]> = Arc::from(vec![0usize]);
        let error = match worker.execute_local_transfer(
            LogicalLayoutHandle::G1,
            LogicalLayoutHandle::G2,
            ids.clone(),
            ids,
            TransferOptions::default(),
        ) {
            Ok(_) => panic!("missing positions must not fall back to a scalar"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("requires logical block positions")
        );
    }

    #[tokio::test]
    async fn mock_worker_two_concurrent_transfers_complete_at_2x() {
        // PS regression at the Worker layer: two G1→G2 transfers at t=0
        // should both complete at t = 2 * single_transfer_duration.
        let worker = make_worker();
        worker.set_now_ms(0.0);
        let mk_ids = || -> Arc<[BlockId]> { Arc::from(vec![0usize]) };

        let n1 = worker
            .execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                mk_ids(),
                mk_ids(),
                TransferOptions::default(),
            )
            .unwrap();
        let n2 = worker
            .execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                mk_ids(),
                mk_ids(),
                TransferOptions::default(),
            )
            .unwrap();

        // At t = 1 ms (single-transfer duration), both must still be
        // pending under PS — each got 0.5 MB/ms so both have 0.5 MB left.
        worker.drain_completions(1.0);
        assert!(n1.could_yield());
        assert!(n2.could_yield());

        // At t = 2 ms, both finish simultaneously.
        worker.drain_completions(2.0);
        n1.await.expect("n1 should resolve Ok");
        n2.await.expect("n2 should resolve Ok");
    }

    #[tokio::test]
    async fn ordered_completion_hook_follows_processor_sharing_order() {
        let worker = MockWorker::new_with_drive_mode(
            1_000_000,
            1.0,
            1.0,
            None,
            None,
            None,
            None,
            KvbmDriveMode::OfflineDeterministic,
        );
        let (long_id, _long) = worker
            .reserve_transfer_with(TransferDirection::G1ToG2, 0.0, 2, None, |_| {})
            .expect("long transfer reservation");
        let (short_id, _short) = worker
            .reserve_transfer_with(TransferDirection::G1ToG2, 0.0, 1, None, |_| {})
            .expect("short transfer reservation");
        let mut observed = Vec::new();

        let drained = worker.drain_completions_summary_ordered(
            3.0,
            |completed| observed.push(completed.id),
            |_| {},
            |_| {},
        );

        assert_eq!(drained.local.offload_transfers, 2);
        assert_eq!(observed, vec![short_id, long_id]);
    }

    #[tokio::test]
    async fn mock_worker_rejects_unsupported_directions() {
        // Direct G1↔G3 and G4 directions must fail at the Worker layer
        // (not silently succeed as no-ops).
        let worker = make_worker();
        worker.set_now_ms(0.0);
        let ids: Arc<[BlockId]> = Arc::from(vec![0usize]);

        let result = worker.execute_local_transfer(
            LogicalLayoutHandle::G1,
            LogicalLayoutHandle::G3,
            ids.clone(),
            ids,
            TransferOptions::default(),
        );
        let err = match result {
            Ok(_) => panic!("G1→G3 must be rejected"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(msg.contains("G2↔G3"), "unexpected error: {msg}");
    }

    #[tokio::test]
    async fn mock_worker_g2_to_g3_bandwidth_is_shared_across_workers() {
        let _guard = shared_g3_test_guard().await;
        let (worker_a, worker_b) = shared_g3_two_workers();
        let ids = || -> Arc<[BlockId]> { Arc::from(vec![0usize]) };
        reserve_shared_g3(&worker_a, 2);

        worker_a.set_now_ms(0.0);
        worker_b.set_now_ms(0.0);
        let a = worker_a
            .execute_local_transfer(
                LogicalLayoutHandle::G2,
                LogicalLayoutHandle::G3,
                ids(),
                ids(),
                TransferOptions::default(),
            )
            .unwrap();
        let b = worker_b
            .execute_local_transfer(
                LogicalLayoutHandle::G2,
                LogicalLayoutHandle::G3,
                ids(),
                ids(),
                TransferOptions::default(),
            )
            .unwrap();

        let early = worker_a.drain_completions_summary(1.0).shared_g3.counts;
        assert_eq!(early.offload_blocks, 0);
        assert!(a.could_yield());
        assert!(b.could_yield());

        let b_drained = worker_b.drain_completions_summary(2.0).shared_g3;
        assert_eq!(b_drained.offload_registration_baseline, Some(0));
        let b_drained = b_drained.counts;
        assert_eq!(b_drained.offload_blocks, 1);
        let a_drained = worker_a.drain_completions_summary(2.0).shared_g3;
        assert_eq!(a_drained.offload_registration_baseline, Some(0));
        let a_drained = a_drained.counts;
        assert_eq!(a_drained.offload_blocks, 1);
        a.await
            .expect("worker A G2→G3 should complete at shared PS 2x");
        b.await
            .expect("worker B G2→G3 should complete at shared PS 2x");
    }

    #[tokio::test]
    async fn offline_shared_g3_completions_wait_for_their_owner_boundary() {
        let _guard = shared_g3_test_guard().await;
        let config = KvbmOffloadConfig {
            num_g3_blocks: Some(128),
            block_size_bytes: Some(1_000_000),
            bandwidth_g2_to_g3_gbps: 1.0,
            bandwidth_g3_to_g2_gbps: 1.0,
            ..Default::default()
        };
        let shared_g3 =
            SharedG3Pool::get_or_create_with_mode(&config, KvbmDriveMode::OfflineDeterministic)
                .unwrap();
        let make_worker = || {
            MockWorker::new_with_drive_mode(
                1_000_000,
                1.0,
                1.0,
                None,
                None,
                shared_g3.clone(),
                None,
                KvbmDriveMode::OfflineDeterministic,
            )
        };
        let worker_a = make_worker();
        let worker_b = make_worker();
        let ids = || -> Arc<[BlockId]> { Arc::from(vec![0usize]) };

        let a = worker_a
            .execute_local_transfer(
                LogicalLayoutHandle::G2,
                LogicalLayoutHandle::G3,
                ids(),
                ids(),
                TransferOptions::default(),
            )
            .unwrap();
        let b = worker_b
            .execute_local_transfer(
                LogicalLayoutHandle::G2,
                LogicalLayoutHandle::G3,
                ids(),
                ids(),
                TransferOptions::default(),
            )
            .unwrap();

        let mut b_completions = 0;
        let b_drained =
            worker_b.drain_completions_summary_ordered(2.0, |_| {}, |_| b_completions += 1, |_| {});
        assert_eq!(b_drained.shared_g3.counts.offload_blocks, 1);
        assert_eq!(b_completions, 1);
        assert!(a.could_yield());
        b.await.expect("worker B completion");

        let mut a_completions = 0;
        let a_drained =
            worker_a.drain_completions_summary_ordered(2.0, |_| {}, |_| a_completions += 1, |_| {});
        assert_eq!(a_drained.shared_g3.counts.offload_blocks, 1);
        assert_eq!(a_completions, 1);
        a.await.expect("worker A completion");
    }

    #[tokio::test]
    async fn mock_worker_shared_prereservation_drain_preserves_owner_accounting() {
        let _guard = shared_g3_test_guard().await;
        let (worker_a, worker_b) = shared_g3_two_workers();
        let ids = || -> Arc<[BlockId]> { Arc::from(vec![0usize]) };
        reserve_shared_g3(&worker_a, 2);

        worker_a.set_now_ms(0.0);
        let a = worker_a
            .execute_local_transfer(
                LogicalLayoutHandle::G2,
                LogicalLayoutHandle::G3,
                ids(),
                ids(),
                TransferOptions::default(),
            )
            .unwrap();

        worker_b.set_now_ms(1.0);
        let b = worker_b
            .execute_local_transfer(
                LogicalLayoutHandle::G2,
                LogicalLayoutHandle::G3,
                ids(),
                ids(),
                TransferOptions::default(),
            )
            .unwrap();

        let a_drained = worker_a.drain_completions_summary(1.0).shared_g3;
        assert_eq!(a_drained.counts.offload_blocks, 1);
        assert_eq!(a_drained.offload_registration_baseline, Some(0));
        a.await
            .expect("worker A G2→G3 should complete during worker B reservation");

        let b_drained = worker_b.drain_completions_summary(2.0).shared_g3;
        assert_eq!(b_drained.counts.offload_blocks, 1);
        b.await
            .expect("worker B G2→G3 should complete after its own reservation");
    }

    #[tokio::test]
    async fn mock_worker_shared_prereservation_drain_marks_deferred_onboard_blocks() {
        let _guard = shared_g3_test_guard().await;
        let (worker_a, worker_b) = shared_g3_two_workers();
        let ids = || -> Arc<[BlockId]> { Arc::from(vec![0usize]) };
        reserve_shared_g3(&worker_b, 1);

        worker_a.set_now_ms(0.0);
        let a = worker_a
            .execute_local_transfer(
                LogicalLayoutHandle::G3,
                LogicalLayoutHandle::G2,
                ids(),
                ids(),
                TransferOptions::default(),
            )
            .unwrap();

        worker_b.set_now_ms(1.0);
        let b = worker_b
            .execute_local_transfer(
                LogicalLayoutHandle::G2,
                LogicalLayoutHandle::G3,
                ids(),
                ids(),
                TransferOptions::default(),
            )
            .unwrap();

        let a_drained = worker_a.drain_completions_summary(1.0).shared_g3;
        assert_eq!(a_drained.counts.onboard_blocks, 1);
        assert_eq!(a_drained.deferred_onboard_blocks, 1);
        a.await
            .expect("worker A G3→G2 should complete during worker B reservation");

        let b_drained = worker_b.drain_completions_summary(2.0).shared_g3;
        assert_eq!(b_drained.counts.offload_blocks, 1);
        b.await
            .expect("worker B G2→G3 should complete after its own reservation");
    }

    #[tokio::test]
    async fn mock_worker_object_put_updates_shared_g4_after_transfer() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g4_test_guard().await;
        let (worker, shared_g4) = shared_g4_worker();
        let plh = PositionalLineageHash::new(6_000, None, 0);

        worker.set_now_ms(0.0);
        let put = worker.put_blocks(vec![plh], LogicalLayoutHandle::G2, vec![0]);
        assert_eq!(shared_g4.has_object(&plh), None);

        let early = worker.drain_completions_summary(0.5).shared_g4.counts;
        assert_eq!(early.offload_blocks, 0);
        assert_eq!(shared_g4.has_object(&plh), None);

        let drained = worker.drain_completions_summary(1.0).shared_g4.counts;
        assert_eq!(drained.offload_blocks, 1);
        assert_eq!(shared_g4.has_object(&plh), Some(1_000_000));
        let results = put.await;
        assert_eq!(results, vec![Ok(plh)]);
        assert_eq!(shared_g4.has_object(&plh), Some(1_000_000));
    }

    #[tokio::test]
    async fn offline_g4_put_is_visible_before_its_completion_callback() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g4_test_guard().await;
        let config = KvbmOffloadConfig {
            enable_g4_storage: true,
            g4_capacity_bytes: Some(64_000_000),
            block_size_bytes: Some(1_000_000),
            bandwidth_g2_to_g4_gbps: 1.0,
            bandwidth_g4_to_g2_gbps: 1.0,
            ..Default::default()
        };
        let shared_g4 =
            SharedG4Store::get_or_create_with_mode(&config, KvbmDriveMode::OfflineDeterministic)
                .unwrap()
                .expect("G4 enabled");
        let worker = MockWorker::new_with_drive_mode(
            1_000_000,
            1.0,
            1.0,
            None,
            None,
            None,
            Some(shared_g4.clone()),
            KvbmDriveMode::OfflineDeterministic,
        );
        let plh = PositionalLineageHash::new(6_025, None, 0);
        let put = worker.put_blocks(vec![plh], LogicalLayoutHandle::G2, vec![0]);
        let mut callback_observed_object = false;

        let drained = worker.drain_completions_summary_ordered(
            1.0,
            |_| {},
            |_| {},
            |_| callback_observed_object = shared_g4.has_object(&plh).is_some(),
        );

        assert_eq!(drained.shared_g4.counts.offload_blocks, 1);
        assert!(callback_observed_object);
        assert_eq!(put.await, vec![Ok(plh)]);
    }

    #[tokio::test]
    async fn mock_worker_object_put_visible_when_another_worker_drains_shared_g4() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g4_test_guard().await;
        let (worker_a, worker_b, shared_g4) = shared_g4_two_workers();
        let plh = PositionalLineageHash::new(6_050, None, 0);

        worker_a.set_now_ms(0.0);
        let put = worker_a.put_blocks(vec![plh], LogicalLayoutHandle::G2, vec![0]);
        assert_eq!(shared_g4.has_object(&plh), None);

        let b_drained = worker_b.drain_completions_summary(1.0).shared_g4.counts;
        assert_eq!(
            b_drained.offload_blocks, 0,
            "worker B should not consume worker A's owner accounting"
        );
        assert_eq!(
            shared_g4.has_object(&plh),
            Some(1_000_000),
            "shared G4 completion should publish the object even when another worker drains"
        );

        let results = put.await;
        assert_eq!(results, vec![Ok(plh)]);
    }

    #[tokio::test]
    async fn mock_worker_object_get_charges_present_g4_blocks_only() {
        use dynamo_tokens::PositionalLineageHash;

        let _guard = shared_g4_test_guard().await;
        let (worker, shared_g4) = shared_g4_worker();
        let present = PositionalLineageHash::new(6_100, None, 0);
        let missing = PositionalLineageHash::new(6_101, None, 1);
        shared_g4.insert_object(present, 1_000_000);

        worker.set_now_ms(0.0);
        let get = worker.get_blocks(vec![present, missing], LogicalLayoutHandle::G2, vec![0, 1]);
        let drained = worker.drain_completions_summary(1.0).shared_g4.counts;
        assert_eq!(drained.onboard_blocks, 1);
        let results = get.await;
        assert_eq!(results, vec![Ok(present), Err(missing)]);
    }

    #[tokio::test]
    async fn mock_worker_offload_and_swap_in_share_id_keyspace() {
        // Invariant: pipeline transfers (`awaiters`) and G2→G1 swap-ins
        // (`swap_in_status`) live in two HashMaps but share one TransferId
        // keyspace, because `TransferState::drain_completions` looks up every
        // drained id in both maps. `TransferState::new` enforces this by handing the
        // same Arc<AtomicU64> counter to `offload_bw` and `onboard_bw`.
        //
        // If a future refactor gives each BandwidthSharingModel its own
        // counter, both would start at 0 and the first offload + first
        // swap-in would alias on id=0 — causing a completing offload to
        // falsely complete the swap-in status (and vice versa). This test pins
        // that invariant: ids drawn across the two models must be disjoint.
        let worker = make_worker();
        worker.set_now_ms(0.0);

        // Reserve one swap-in (onboard model) and one offload (offload model)
        // at the same virtual time, so both counters are at their initial value.
        let (status, _) = watch::channel(SwapInStatus::pending(1));
        let swap_id = worker.reserve_swap_in(0.0, 1, None, status);
        let ids: Arc<[BlockId]> = Arc::from(vec![0usize]);
        let _offload = worker
            .execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                ids.clone(),
                ids,
                TransferOptions::default(),
            )
            .unwrap();

        let state = worker.state.lock().unwrap();
        let awaiter_id = *state
            .awaiters
            .keys()
            .next()
            .expect("offload must register an awaiter");
        assert_ne!(
            awaiter_id, swap_id,
            "offload and swap-in must draw distinct TransferIds"
        );
    }

    #[tokio::test]
    async fn mock_worker_swap_in_status_completes_on_drain() {
        // Reserve a G2→G1 swap-in for 1 block (1 MB at 1 GB/s → 1 ms).
        // Before drain the status must be pending; after advancing past the
        // finish time the same drain must publish completion.
        let worker = make_worker();
        worker.set_now_ms(0.0);
        let (status, status_rx) = watch::channel(SwapInStatus::pending(1));
        let _id = worker.reserve_swap_in(0.0, 1, None, status);
        assert_eq!(status_rx.borrow().terminal, SwapInTerminal::Pending);
        worker.drain_completions(0.5);
        assert!(
            matches!(status_rx.borrow().terminal, SwapInTerminal::Pending),
            "swap-in must not complete before its finish time"
        );
        worker.drain_completions(1.0);
        assert!(
            matches!(status_rx.borrow().terminal, SwapInTerminal::Completed),
            "swap-in status must complete after drain past finish time"
        );
    }

    #[tokio::test]
    async fn mock_worker_earliest_finish_min_of_both_links() {
        // With both directions active, `earliest_finish` returns the
        // minimum of the two models' next-completion times.
        let worker = make_worker();
        worker.set_now_ms(0.0);
        let ids: Arc<[BlockId]> = Arc::from(vec![0usize]);

        // G1→G2 at t=0, finishes at 1.0 ms under PS with N=1.
        let _n1 = worker
            .execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                ids.clone(),
                ids.clone(),
                TransferOptions::default(),
            )
            .unwrap();

        // G2→G1 at t=0 on the OTHER link, finishes at 1.0 ms with N=1 too.
        let _n2 = worker
            .execute_local_transfer(
                LogicalLayoutHandle::G2,
                LogicalLayoutHandle::G1,
                ids.clone(),
                ids,
                TransferOptions::default(),
            )
            .unwrap();

        let earliest = worker.earliest_finish().unwrap();
        assert!(
            (earliest - 1.0).abs() < EPS,
            "expected 1.0 ms, got {earliest}"
        );
    }
}
