// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Simulated-time counterpart of the live random router's endpoint affinity.
//!
//! A binding survives while requests are active and expires only after an idle
//! TTL following completion. The live random router binds a worker endpoint,
//! with no DP rank unless explicitly requested. Contiguous rank groups model
//! per-node endpoints without changing attention-DP/EP topology. Internal rank
//! selection is an explicit round-robin proxy, not a model of vLLM's live load
//! telemetry. No KV overlap or lower-tier residency influences this policy.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, ensure};
use rustc_hash::FxHashMap;
use uuid::Uuid;

pub(crate) const SESSION_AFFINITY_SEED: u64 = 0xD1A0_5EED;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Endpoint {
    worker_id: usize,
    rank_group: usize,
}

#[derive(Debug)]
struct Binding {
    endpoint: Endpoint,
    revision: u64,
    active_leases: usize,
    idle_deadline_ms: f64,
}

#[derive(Debug)]
pub(in crate::replay::offline) struct SessionAffinityState {
    ttl_ms: f64,
    ranks_per_endpoint: usize,
    rng: fastrand::Rng,
    bindings: FxHashMap<String, Binding>,
    active: FxHashMap<Uuid, (String, u64)>,
    next_rank: FxHashMap<Endpoint, usize>,
    next_revision: u64,
    next_reap_ms: f64,
}

impl SessionAffinityState {
    pub(in crate::replay::offline) fn new(
        ttl_secs: u64,
        ranks_per_endpoint: u32,
        seed: u64,
    ) -> Result<Self> {
        ensure!(
            (1..=31_536_000).contains(&ttl_secs),
            "session affinity TTL must be between 1 and 31536000 seconds"
        );
        ensure!(
            ranks_per_endpoint > 0,
            "ranks_per_endpoint must be positive"
        );
        Ok(Self {
            ttl_ms: ttl_secs as f64 * 1000.0,
            ranks_per_endpoint: ranks_per_endpoint as usize,
            rng: fastrand::Rng::with_seed(seed),
            bindings: FxHashMap::default(),
            active: FxHashMap::default(),
            next_rank: FxHashMap::default(),
            next_revision: 0,
            next_reap_ms: 0.0,
        })
    }

    pub(in crate::replay::offline) fn select(
        &mut self,
        request_id: Uuid,
        session_id: Option<&str>,
        now_ms: f64,
        workers: &BTreeMap<usize, Vec<usize>>,
    ) -> Result<usize> {
        ensure!(
            now_ms.is_finite() && now_ms >= 0.0,
            "invalid affinity replay time"
        );
        ensure!(
            !self.active.contains_key(&request_id),
            "request already holds a session affinity lease"
        );
        if now_ms >= self.next_reap_ms {
            self.bindings.retain(|_, binding| {
                binding.active_leases > 0 || binding.idle_deadline_ms > now_ms
            });
            self.next_reap_ms = now_ms + self.ttl_ms.min(30_000.0);
        }
        let endpoints = workers
            .iter()
            .flat_map(|(&worker_id, ranks)| {
                (0..ranks.len().div_ceil(self.ranks_per_endpoint)).map(move |rank_group| Endpoint {
                    worker_id,
                    rank_group,
                })
            })
            .collect::<Vec<_>>();
        ensure!(
            !endpoints.is_empty(),
            "session affinity has no active endpoint"
        );

        let bound = session_id
            .and_then(|session| self.bindings.get(session))
            .filter(|binding| {
                (binding.active_leases > 0 || binding.idle_deadline_ms > now_ms)
                    && endpoints.contains(&binding.endpoint)
            })
            .map(|binding| binding.endpoint);
        let endpoint = bound.unwrap_or_else(|| endpoints[self.rng.usize(..endpoints.len())]);
        if let Some(session_id) = session_id {
            if bound.is_none() {
                self.next_revision += 1;
                self.bindings.insert(
                    session_id.to_string(),
                    Binding {
                        endpoint,
                        revision: self.next_revision,
                        active_leases: 0,
                        idle_deadline_ms: now_ms + self.ttl_ms,
                    },
                );
            }
            let binding = self
                .bindings
                .get_mut(session_id)
                .expect("selected binding exists");
            binding.active_leases += 1;
            self.active
                .insert(request_id, (session_id.to_string(), binding.revision));
        }

        let ranks = workers
            .get(&endpoint.worker_id)
            .expect("selected endpoint exists");
        let first = endpoint.rank_group * self.ranks_per_endpoint;
        let count = self.ranks_per_endpoint.min(ranks.len() - first);
        let next = self.next_rank.entry(endpoint).or_default();
        let scheduler = *ranks
            .get(first + *next % count)
            .ok_or_else(|| anyhow!("selected endpoint has no scheduler"))?;
        *next = (*next + 1) % count;
        Ok(scheduler)
    }

    pub(in crate::replay::offline) fn complete(&mut self, request_id: Uuid, now_ms: f64) {
        let Some((session_id, revision)) = self.active.remove(&request_id) else {
            return;
        };
        if let Some(binding) = self.bindings.get_mut(&session_id)
            && binding.revision == revision
        {
            debug_assert!(binding.active_leases > 0);
            binding.active_leases = binding.active_leases.saturating_sub(1);
            binding.idle_deadline_ms = now_ms + self.ttl_ms;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    #[test]
    fn endpoint_affinity_keeps_node_but_not_dp_rank() {
        let workers = BTreeMap::from([(0, vec![0, 1, 2, 3, 4, 5, 6, 7])]);
        let mut state = SessionAffinityState::new(900, 4, SESSION_AFFINITY_SEED).unwrap();
        let selected = (1..=8)
            .map(|n| {
                let rank = state
                    .select(request(n), Some("stream-a"), n as f64, &workers)
                    .unwrap();
                state.complete(request(n), n as f64 + 0.5);
                rank
            })
            .collect::<Vec<_>>();
        assert!(selected.iter().all(|rank| rank / 4 == selected[0] / 4));
        assert_eq!(
            selected[..4]
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            4
        );
        assert_eq!(selected[..4], selected[4..]);
    }

    #[test]
    fn rank_affinity_is_only_the_explicit_one_rank_endpoint_case() {
        let workers = BTreeMap::from([(0, vec![0, 1])]);
        let mut state = SessionAffinityState::new(1, 1, SESSION_AFFINITY_SEED).unwrap();
        let first = state.select(request(1), Some("a"), 0.0, &workers).unwrap();
        state.complete(request(1), 8.0);
        assert_eq!(
            state
                .select(request(2), Some("a"), 1007.0, &workers)
                .unwrap(),
            first
        );
    }

    #[test]
    fn ttl_starts_at_completion_and_does_not_expire_active_leases() {
        let workers = BTreeMap::from([(0, vec![0, 1])]);
        let mut state = SessionAffinityState::new(1, 1, SESSION_AFFINITY_SEED).unwrap();
        let first = state.select(request(1), Some("a"), 0.0, &workers).unwrap();
        assert_eq!(
            state
                .select(request(2), Some("a"), 50_000.0, &workers)
                .unwrap(),
            first
        );
        state.complete(request(1), 50_010.0);
        state.complete(request(2), 50_020.0);
        let revision = state.bindings["a"].revision;
        assert_eq!(
            state
                .select(request(3), Some("a"), 51_019.0, &workers)
                .unwrap(),
            first
        );
        state.complete(request(3), 51_020.0);
        state
            .select(request(4), Some("a"), 52_020.0, &workers)
            .unwrap();
        assert_ne!(
            state.bindings["a"].revision, revision,
            "exact TTL boundary must reselect"
        );
    }

    #[test]
    fn child_streams_have_independent_bindings_and_missing_ids_do_not_bind() {
        let workers = BTreeMap::from([(0, vec![0, 1, 2, 3])]);
        let mut state = SessionAffinityState::new(1, 1, SESSION_AFFINITY_SEED).unwrap();
        state
            .select(request(1), Some("play:main"), 0.0, &workers)
            .unwrap();
        state
            .select(request(2), Some("play:child"), 0.0, &workers)
            .unwrap();
        state.select(request(3), None, 0.0, &workers).unwrap();
        assert_eq!(state.bindings.len(), 2);
        assert_eq!(state.active.len(), 2);
    }

    #[test]
    fn removed_endpoint_is_reselected_and_old_completion_cannot_release_new_binding() {
        let mut workers = BTreeMap::from([(0, vec![0]), (1, vec![1])]);
        let mut state = SessionAffinityState::new(1, 1, SESSION_AFFINITY_SEED).unwrap();
        let first = state.select(request(1), Some("a"), 0.0, &workers).unwrap();
        workers.remove(&first);
        let next = state.select(request(2), Some("a"), 1.0, &workers).unwrap();
        assert_ne!(first, next);
        state.complete(request(1), 2.0);
        assert_eq!(state.bindings["a"].active_leases, 1);
        state.complete(request(2), 3.0);
        assert_eq!(state.bindings["a"].active_leases, 0);
    }

    #[test]
    fn fixed_seed_is_independent_of_wall_clock_and_rejects_invalid_configuration() {
        let workers = BTreeMap::from([(0, vec![0, 1, 2, 3])]);
        let run = || {
            let mut state = SessionAffinityState::new(1, 1, SESSION_AFFINITY_SEED).unwrap();
            (1..=20)
                .map(|n| state.select(request(n), None, n as f64, &workers).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
        assert!(SessionAffinityState::new(0, 1, 0).is_err());
        assert!(SessionAffinityState::new(1, 0, 0).is_err());
    }
}
