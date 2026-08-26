// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-local shared G4 object-store simulation for mock KVBM offload.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use anyhow::{Result, bail};
use kvbm_engine::SequenceHash;
use kvbm_engine::offload::PendingTracker;

use super::KvbmDriveMode;
use super::bandwidth_sharing_model::TransferId;
use super::config::KvbmOffloadConfig;
use super::worker::{
    CompletedTransfer, CompletionAction, DeferredOwnerDrain, DrainResult, SharedDrainCounts,
    TransferDirection, TransferState,
};

struct DeferredOwnerCompletions {
    actions: Vec<CompletionAction>,
    deadline_ms: f64,
}

/// Object metadata tracked by the mock G4 tier.
#[derive(Debug, Clone, Copy)]
pub(crate) struct G4ObjectRecord {
    pub(crate) size_bytes: usize,
    last_access: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct G4CacheMetrics {
    pub capacity_bytes: usize,
    pub current_bytes: usize,
    pub peak_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub admissions: u64,
    pub evictions: u64,
    pub evicted_bytes: u64,
    pub rejected_oversize: u64,
}

#[derive(Debug, Default)]
struct G4Catalog {
    objects: HashMap<SequenceHash, G4ObjectRecord>,
    clock: u64,
    metrics: G4CacheMetrics,
}

#[derive(Debug)]
struct PendingG4Put {
    keys: Vec<SequenceHash>,
    sizes_bytes: Vec<usize>,
}

/// Shared process-local G4 resource. All mock workers in the process share
/// the object registry and the G2<->G4 processor-sharing model.
pub(crate) struct SharedG4Store {
    config: KvbmOffloadConfig,
    drive_mode: KvbmDriveMode,
    catalog: Mutex<G4Catalog>,
    pending_puts: Mutex<HashMap<TransferId, PendingG4Put>>,
    state: Arc<Mutex<TransferState>>,
    pending_tracker: Arc<PendingTracker>,
    pending_owner_drains: Mutex<HashMap<u64, DeferredOwnerDrain>>,
    pending_owner_completions: Mutex<HashMap<u64, DeferredOwnerCompletions>>,
}

static SHARED_G4_STORE: OnceLock<Mutex<Option<Weak<SharedG4Store>>>> = OnceLock::new();

#[cfg(test)]
static SHARED_G4_TEST_LOCK: OnceLock<Arc<tokio::sync::Mutex<()>>> = OnceLock::new();

impl SharedG4Store {
    #[cfg(test)]
    pub(crate) fn get_or_create(config: &KvbmOffloadConfig) -> Result<Option<Arc<Self>>> {
        Self::get_or_create_with_mode(config, KvbmDriveMode::Live)
    }

    pub(crate) fn get_or_create_with_mode(
        config: &KvbmOffloadConfig,
        drive_mode: KvbmDriveMode,
    ) -> Result<Option<Arc<Self>>> {
        if !config.enable_g4_storage {
            return Ok(None);
        }

        let store_slot = SHARED_G4_STORE.get_or_init(|| Mutex::new(None));
        let mut store_slot = store_slot
            .lock()
            .expect("shared G4 store registry poisoned");

        if let Some(existing) = store_slot.as_ref().and_then(Weak::upgrade) {
            existing.validate_config(config, drive_mode)?;
            return Ok(Some(existing));
        }

        *store_slot = None;
        let store = Arc::new(Self {
            config: config.clone(),
            drive_mode,
            catalog: Mutex::new(G4Catalog {
                metrics: G4CacheMetrics {
                    capacity_bytes: config
                        .g4_capacity_bytes
                        .expect("validated finite G4 capacity"),
                    ..G4CacheMetrics::default()
                },
                ..G4Catalog::default()
            }),
            pending_puts: Mutex::new(HashMap::new()),
            state: Arc::new(Mutex::new(TransferState::new(
                config.bandwidth_g2_to_g4_gbps,
                config.bandwidth_g4_to_g2_gbps,
            ))),
            pending_tracker: Arc::new(PendingTracker::new()),
            pending_owner_drains: Mutex::new(HashMap::new()),
            pending_owner_completions: Mutex::new(HashMap::new()),
        });
        *store_slot = Some(Arc::downgrade(&store));
        Ok(Some(store))
    }

    fn validate_config(&self, config: &KvbmOffloadConfig, drive_mode: KvbmDriveMode) -> Result<()> {
        if self.config.enable_g4_storage != config.enable_g4_storage
            || self.config.block_size_tokens != config.block_size_tokens
            || self.config.block_size_bytes.unwrap_or(0) != config.block_size_bytes.unwrap_or(0)
            || self.config.block_footprint_bytes != config.block_footprint_bytes
            || self.config.g4_capacity_bytes != config.g4_capacity_bytes
            || self.config.bandwidth_g2_to_g4_gbps != config.bandwidth_g2_to_g4_gbps
            || self.config.bandwidth_g4_to_g2_gbps != config.bandwidth_g4_to_g2_gbps
            || self.drive_mode != drive_mode
        {
            bail!(
                "process-local shared G4 store already exists with a different capacity, footprint, bandwidth, or drive mode"
            );
        }
        Ok(())
    }

    pub(crate) fn transfer_state(&self) -> Arc<Mutex<TransferState>> {
        self.state.clone()
    }

    pub(crate) fn pending_tracker(&self) -> Arc<PendingTracker> {
        self.pending_tracker.clone()
    }

    pub(crate) fn has_object(&self, hash: &SequenceHash) -> Option<usize> {
        let mut catalog = self.catalog.lock().expect("shared G4 catalog poisoned");
        catalog.clock = catalog.clock.wrapping_add(1);
        let access = catalog.clock;
        if let Some(record) = catalog.objects.get_mut(hash) {
            record.last_access = access;
            let size = record.size_bytes;
            catalog.metrics.hits = catalog.metrics.hits.saturating_add(1);
            Some(size)
        } else {
            catalog.metrics.misses = catalog.metrics.misses.saturating_add(1);
            None
        }
    }

    pub(crate) fn insert_object(&self, hash: SequenceHash, size_bytes: usize) {
        let mut catalog = self.catalog.lock().expect("shared G4 catalog poisoned");
        let capacity = catalog.metrics.capacity_bytes;
        if size_bytes > capacity {
            catalog.metrics.rejected_oversize = catalog.metrics.rejected_oversize.saturating_add(1);
            return;
        }
        if let Some(existing) = catalog.objects.remove(&hash) {
            catalog.metrics.current_bytes = catalog
                .metrics
                .current_bytes
                .saturating_sub(existing.size_bytes);
        }
        while catalog.metrics.current_bytes.saturating_add(size_bytes) > capacity {
            let victim = catalog
                .objects
                .iter()
                .min_by_key(|(key, record)| (record.last_access, key.as_u128()))
                .map(|(key, _)| *key);
            let Some(victim) = victim else {
                break;
            };
            if let Some(evicted) = catalog.objects.remove(&victim) {
                catalog.metrics.current_bytes = catalog
                    .metrics
                    .current_bytes
                    .saturating_sub(evicted.size_bytes);
                catalog.metrics.evictions = catalog.metrics.evictions.saturating_add(1);
                catalog.metrics.evicted_bytes = catalog
                    .metrics
                    .evicted_bytes
                    .saturating_add(evicted.size_bytes as u64);
            }
        }
        catalog.clock = catalog.clock.wrapping_add(1);
        let last_access = catalog.clock;
        catalog.objects.insert(
            hash,
            G4ObjectRecord {
                size_bytes,
                last_access,
            },
        );
        catalog.metrics.current_bytes = catalog.metrics.current_bytes.saturating_add(size_bytes);
        catalog.metrics.peak_bytes = catalog
            .metrics
            .peak_bytes
            .max(catalog.metrics.current_bytes);
        catalog.metrics.admissions = catalog.metrics.admissions.saturating_add(1);
    }

    pub(crate) fn register_pending_put(
        &self,
        transfer_id: TransferId,
        keys: Vec<SequenceHash>,
        sizes_bytes: Vec<usize>,
    ) {
        let mut pending = self
            .pending_puts
            .lock()
            .expect("shared G4 pending put map poisoned");
        debug_assert_eq!(keys.len(), sizes_bytes.len());
        pending.insert(transfer_id, PendingG4Put { keys, sizes_bytes });
    }

    #[cfg(test)]
    pub(crate) fn object_count(&self) -> usize {
        let catalog = self.catalog.lock().expect("shared G4 catalog poisoned");
        catalog.objects.len()
    }

    pub fn metrics(&self) -> G4CacheMetrics {
        self.catalog
            .lock()
            .expect("shared G4 catalog poisoned")
            .metrics
    }

    pub(crate) fn drain_completions(&self, now_ms: f64, owner_id: u64) -> SharedDrainCounts {
        let mut state = self.state.lock().expect("shared G4 state poisoned");
        let drained = state.drain_completions(now_ms, "shared-g4");
        drop(state);

        self.record_drained(drained, Some(owner_id), now_ms)
    }

    pub(crate) fn drain_completions_to_pending(&self, now_ms: f64) {
        let mut state = self.state.lock().expect("shared G4 state poisoned");
        let drained = state.drain_completions(now_ms, "shared-g4");
        drop(state);

        self.record_drained(drained, None, now_ms);
    }

    pub(crate) fn drain_completions_ordered(
        &self,
        now_ms: f64,
        owner_id: u64,
        mut after_pipeline_completion: impl FnMut(CompletedTransfer),
    ) -> SharedDrainCounts {
        self.defer_due_completions(now_ms);
        let pending = self
            .pending_owner_completions
            .lock()
            .expect("shared G4 ordered completion map poisoned")
            .remove(&owner_id);
        let Some(pending) = pending else {
            return SharedDrainCounts::default();
        };

        let mut result = DrainResult::default();
        for action in pending.actions {
            let metadata = action.completed_transfer();
            self.publish_completed_puts(std::slice::from_ref(&metadata));
            let completed = action.fire();
            result.record_completion(completed);
            if completed.pipeline {
                after_pipeline_completion(completed);
            }
        }
        SharedDrainCounts {
            counts: result.by_owner.get(&owner_id).copied().unwrap_or_default(),
            // Ordered actions fire only at this owner's boundary, so onboard
            // blocks are current rather than already-published deferred work.
            deferred_onboard_blocks: 0,
            offload_registration_baseline: None,
        }
    }

    pub(crate) fn defer_completions_ordered(&self, now_ms: f64) {
        self.defer_due_completions(now_ms);
    }

    fn defer_due_completions(&self, now_ms: f64) {
        let actions = {
            let mut state = self.state.lock().expect("shared G4 state poisoned");
            state.take_completion_actions(now_ms, "shared-g4")
        };
        if actions.is_empty() {
            return;
        }

        let mut pending = self
            .pending_owner_completions
            .lock()
            .expect("shared G4 ordered completion map poisoned");
        for action in actions {
            let owner_id = action
                .owner_id()
                .expect("shared G4 completion must have an owner");
            match pending.entry(owner_id) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let pending = entry.get_mut();
                    pending.actions.push(action);
                    pending.deadline_ms = pending.deadline_ms.min(now_ms);
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(DeferredOwnerCompletions {
                        actions: vec![action],
                        deadline_ms: now_ms,
                    });
                }
            }
        }
    }

    fn record_drained(
        &self,
        drained: DrainResult,
        owner_id: Option<u64>,
        now_ms: f64,
    ) -> SharedDrainCounts {
        self.publish_completed_puts(&drained.completed);

        let mut owner_result = SharedDrainCounts::default();
        let mut pending = self
            .pending_owner_drains
            .lock()
            .expect("shared G4 owner drain map poisoned");
        if let Some(owner_id) = owner_id
            && let Some(record) = pending.remove(&owner_id)
        {
            owner_result.add_deferred_record(record.counts);
        }
        for (owner, counts) in drained.by_owner {
            let record = SharedDrainCounts {
                counts,
                deferred_onboard_blocks: 0,
                offload_registration_baseline: None,
            };
            if Some(owner) == owner_id {
                owner_result.add_record(record);
            } else {
                pending
                    .entry(owner)
                    .and_modify(|pending| {
                        pending.counts.add_record(record);
                        pending.deadline_ms = pending.deadline_ms.min(now_ms);
                    })
                    .or_insert(DeferredOwnerDrain {
                        counts: record,
                        deadline_ms: now_ms,
                    });
            }
        }
        owner_result
    }

    fn publish_completed_puts(&self, completed: &[CompletedTransfer]) {
        let mut puts = Vec::new();
        {
            let mut pending = self
                .pending_puts
                .lock()
                .expect("shared G4 pending put map poisoned");
            for transfer in completed {
                if transfer.direction == TransferDirection::G2ToG4
                    && let Some(pending_put) = pending.remove(&transfer.id)
                {
                    puts.push(pending_put);
                }
            }
        }

        if puts.is_empty() {
            return;
        }

        for pending_put in puts {
            for (key, size_bytes) in pending_put.keys.into_iter().zip(pending_put.sizes_bytes) {
                self.insert_object(key, size_bytes);
            }
        }
    }

    pub(crate) fn earliest_finish(&self) -> Option<f64> {
        let state = self.state.lock().expect("shared G4 state poisoned");
        state.earliest_finish()
    }

    pub(crate) fn pending_owner_deadline(&self, owner_id: u64) -> Option<f64> {
        let drained = self
            .pending_owner_drains
            .lock()
            .expect("shared G4 owner drain map poisoned")
            .get(&owner_id)
            .map(|pending| pending.deadline_ms);
        let completions = self
            .pending_owner_completions
            .lock()
            .expect("shared G4 ordered completion map poisoned")
            .get(&owner_id)
            .map(|pending| pending.deadline_ms);
        drained.into_iter().chain(completions).reduce(f64::min)
    }

    pub(crate) fn earliest_offload_finish(&self) -> Option<f64> {
        let state = self.state.lock().expect("shared G4 state poisoned");
        state.earliest_offload_finish()
    }

    pub(crate) fn earliest_onboard_finish(&self) -> Option<f64> {
        let state = self.state.lock().expect("shared G4 state poisoned");
        state.earliest_onboard_finish()
    }

    #[cfg(test)]
    fn reset_for_tests() {
        let store_slot = SHARED_G4_STORE.get_or_init(|| Mutex::new(None));
        let mut store_slot = store_slot
            .lock()
            .expect("shared G4 store registry poisoned");
        *store_slot = None;
    }
}

#[cfg(test)]
pub(crate) async fn shared_g4_test_guard() -> tokio::sync::OwnedMutexGuard<()> {
    let guard = SHARED_G4_TEST_LOCK
        .get_or_init(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
        .lock_owned()
        .await;
    SharedG4Store::reset_for_tests();
    guard
}

#[cfg(test)]
pub(crate) fn shared_g4_test_guard_blocking() -> tokio::sync::OwnedMutexGuard<()> {
    let guard = SHARED_G4_TEST_LOCK
        .get_or_init(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
        .blocking_lock_owned();
    SharedG4Store::reset_for_tests();
    guard
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_tokens::PositionalLineageHash;

    fn key(position: u64, value: u64) -> SequenceHash {
        PositionalLineageHash::new(value, None, position)
    }

    fn finite_config(capacity_bytes: usize) -> KvbmOffloadConfig {
        KvbmOffloadConfig {
            enable_g4_storage: true,
            g4_capacity_bytes: Some(capacity_bytes),
            block_size_bytes: Some(1),
            ..KvbmOffloadConfig::default()
        }
    }

    #[test]
    fn finite_g4_uses_deterministic_lru_and_byte_accounting() {
        let _guard = shared_g4_test_guard_blocking();
        let store = SharedG4Store::get_or_create(&finite_config(8))
            .expect("finite G4 config should build")
            .expect("G4 should be enabled");
        let oldest = key(0, 10);
        let retained = key(1, 11);
        let newest = key(2, 12);

        store.insert_object(oldest, 4);
        store.insert_object(retained, 4);
        // Refresh `oldest`; the other four-byte object is now the LRU victim.
        assert_eq!(store.has_object(&oldest), Some(4));
        store.insert_object(newest, 4);

        assert_eq!(store.has_object(&retained), None);
        assert_eq!(store.has_object(&oldest), Some(4));
        assert_eq!(store.has_object(&newest), Some(4));
        let metrics = store.metrics();
        assert_eq!(metrics.capacity_bytes, 8);
        assert_eq!(metrics.current_bytes, 8);
        assert_eq!(metrics.peak_bytes, 8);
        assert_eq!(metrics.admissions, 3);
        assert_eq!(metrics.evictions, 1);
        assert_eq!(metrics.evicted_bytes, 4);
        assert_eq!(metrics.rejected_oversize, 0);
    }

    #[test]
    fn finite_g4_rejects_an_object_larger_than_the_deployment_tier() {
        let _guard = shared_g4_test_guard_blocking();
        let store = SharedG4Store::get_or_create(&finite_config(8))
            .expect("finite G4 config should build")
            .expect("G4 should be enabled");

        store.insert_object(key(0, 20), 9);

        assert_eq!(store.object_count(), 0);
        let metrics = store.metrics();
        assert_eq!(metrics.current_bytes, 0);
        assert_eq!(metrics.admissions, 0);
        assert_eq!(metrics.rejected_oversize, 1);
    }
}
