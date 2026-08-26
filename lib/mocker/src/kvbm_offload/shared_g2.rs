// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Node-scoped shared G1<->G2 bandwidth fabric for offline KVBM replay.
//!
//! G2 capacity remains rank-local, but ranks attached to the same physical
//! host contend on one pair of directional processor-sharing queues.  The
//! owner-partitioned completion maps are required because advancing one rank
//! may make another rank's transfer due; only the owning engine may perform
//! its pipeline settlement and publish into its rank-local G2 manager.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use anyhow::{Result, bail};

use super::KvbmDriveMode;
use super::config::KvbmOffloadConfig;
use super::worker::{
    CompletedTransfer, CompletionAction, DeferredOwnerDrain, DrainResult, SharedDrainCounts,
    TransferState, fire_completion_actions_with,
};

struct DeferredOwnerCompletions {
    actions: Vec<CompletionAction>,
    deadline_ms: f64,
}

/// One physical host-memory transfer fabric. Capacity is intentionally not
/// stored here: each scheduler rank owns an independently byte-accounted G2
/// pool while all ranks in this domain contend for the node-level links.
pub(crate) struct SharedG2Fabric {
    id: String,
    drive_mode: KvbmDriveMode,
    offload_gbps: f64,
    onboard_gbps: f64,
    state: Arc<Mutex<TransferState>>,
    pending_owner_drains: Mutex<HashMap<u64, DeferredOwnerDrain>>,
    pending_owner_completions: Mutex<HashMap<u64, DeferredOwnerCompletions>>,
}

static SHARED_G2_FABRICS: OnceLock<Mutex<HashMap<String, Weak<SharedG2Fabric>>>> = OnceLock::new();

#[cfg(test)]
static SHARED_G2_TEST_LOCK: OnceLock<Arc<tokio::sync::Mutex<()>>> = OnceLock::new();

impl SharedG2Fabric {
    pub(crate) fn get_or_create_with_mode(
        config: &KvbmOffloadConfig,
        drive_mode: KvbmDriveMode,
    ) -> Result<Option<Arc<Self>>> {
        let Some(id) = config.g2_fabric_id.as_deref() else {
            return Ok(None);
        };
        let id = id.trim();
        if id.is_empty() {
            bail!("shared G2 fabric id must be non-empty");
        }

        let registry = SHARED_G2_FABRICS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut registry = registry.lock().expect("shared G2 fabric registry poisoned");
        registry.retain(|_, fabric| fabric.strong_count() > 0);
        if let Some(existing) = registry.get(id).and_then(Weak::upgrade) {
            existing.validate_config(config, drive_mode)?;
            return Ok(Some(existing));
        }

        let fabric = Arc::new(Self {
            id: id.to_string(),
            drive_mode,
            offload_gbps: config.bandwidth_g1_to_g2_gbps,
            onboard_gbps: config.bandwidth_g2_to_g1_gbps,
            state: Arc::new(Mutex::new(TransferState::new(
                config.bandwidth_g1_to_g2_gbps,
                config.bandwidth_g2_to_g1_gbps,
            ))),
            pending_owner_drains: Mutex::new(HashMap::new()),
            pending_owner_completions: Mutex::new(HashMap::new()),
        });
        registry.insert(id.to_string(), Arc::downgrade(&fabric));
        Ok(Some(fabric))
    }

    fn validate_config(&self, config: &KvbmOffloadConfig, drive_mode: KvbmDriveMode) -> Result<()> {
        if self.offload_gbps != config.bandwidth_g1_to_g2_gbps
            || self.onboard_gbps != config.bandwidth_g2_to_g1_gbps
            || self.drive_mode != drive_mode
        {
            bail!(
                "shared G2 fabric {:?} already exists with different directional bandwidth or drive mode",
                self.id
            );
        }
        Ok(())
    }

    pub(crate) fn transfer_state(&self) -> Arc<Mutex<TransferState>> {
        self.state.clone()
    }

    pub(crate) fn drain_completions(&self, now_ms: f64, owner_id: u64) -> SharedDrainCounts {
        let drained = {
            let mut state = self.state.lock().expect("shared G2 state poisoned");
            state.drain_completions(now_ms, "shared-g2")
        };
        self.record_drained(drained, Some(owner_id), now_ms)
    }

    pub(crate) fn drain_completions_to_pending(&self, now_ms: f64) {
        let drained = {
            let mut state = self.state.lock().expect("shared G2 state poisoned");
            state.drain_completions(now_ms, "shared-g2")
        };
        self.record_drained(drained, None, now_ms);
    }

    pub(crate) fn drain_completions_ordered(
        &self,
        now_ms: f64,
        owner_id: u64,
        after_pipeline_completion: impl FnMut(CompletedTransfer),
    ) -> SharedDrainCounts {
        self.defer_due_completions(now_ms);
        let pending = self
            .pending_owner_completions
            .lock()
            .expect("shared G2 ordered completion map poisoned")
            .remove(&owner_id);
        let Some(pending) = pending else {
            return SharedDrainCounts::default();
        };
        let result = fire_completion_actions_with(pending.actions, after_pipeline_completion);
        SharedDrainCounts {
            counts: result.by_owner.get(&owner_id).copied().unwrap_or_default(),
            deferred_onboard_blocks: 0,
            offload_registration_baseline: None,
        }
    }

    pub(crate) fn defer_completions_ordered(&self, now_ms: f64) {
        self.defer_due_completions(now_ms);
    }

    fn defer_due_completions(&self, now_ms: f64) {
        let actions = {
            let mut state = self.state.lock().expect("shared G2 state poisoned");
            state.take_completion_actions(now_ms, "shared-g2")
        };
        if actions.is_empty() {
            return;
        }

        let mut pending = self
            .pending_owner_completions
            .lock()
            .expect("shared G2 ordered completion map poisoned");
        for action in actions {
            let owner_id = action
                .owner_id()
                .expect("shared G2 completion must have an owner");
            match pending.entry(owner_id) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let record = entry.get_mut();
                    record.actions.push(action);
                    record.deadline_ms = record.deadline_ms.min(now_ms);
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
        let mut owner_result = SharedDrainCounts::default();
        let mut pending = self
            .pending_owner_drains
            .lock()
            .expect("shared G2 owner drain map poisoned");
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

    pub(crate) fn earliest_finish(&self) -> Option<f64> {
        self.state
            .lock()
            .expect("shared G2 state poisoned")
            .earliest_finish()
    }

    pub(crate) fn earliest_offload_finish_with_id(
        &self,
        owner_id: u64,
    ) -> Option<(super::bandwidth_sharing_model::TransferId, f64)> {
        self.state
            .lock()
            .expect("shared G2 state poisoned")
            .owner_earliest_offload_finish_with_id(owner_id)
    }

    pub(crate) fn offload_active_count(&self, owner_id: u64) -> usize {
        self.state
            .lock()
            .expect("shared G2 state poisoned")
            .owner_offload_active_count(owner_id)
    }

    pub(crate) fn pending_owner_deadline(&self, owner_id: u64) -> Option<f64> {
        let drained = self
            .pending_owner_drains
            .lock()
            .expect("shared G2 owner drain map poisoned")
            .get(&owner_id)
            .map(|pending| pending.deadline_ms);
        let completions = self
            .pending_owner_completions
            .lock()
            .expect("shared G2 ordered completion map poisoned")
            .get(&owner_id)
            .map(|pending| pending.deadline_ms);
        drained.into_iter().chain(completions).reduce(f64::min)
    }

    pub(crate) fn earliest_foreground_finish(&self) -> Option<f64> {
        self.earliest_finish()
    }

    #[cfg(test)]
    fn reset_for_tests() {
        SHARED_G2_FABRICS
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .expect("shared G2 fabric registry poisoned")
            .clear();
    }
}

#[cfg(test)]
pub(crate) async fn shared_g2_test_guard() -> tokio::sync::OwnedMutexGuard<()> {
    let guard = SHARED_G2_TEST_LOCK
        .get_or_init(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
        .lock_owned()
        .await;
    SharedG2Fabric::reset_for_tests();
    guard
}
