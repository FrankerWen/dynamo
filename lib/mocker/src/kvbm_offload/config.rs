// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration for the G1↔G2/G2↔G3 offload simulation.
//!
//! [`KvbmOffloadConfig`] holds the parameters that shape both the PS
//! bandwidth model ([`super::bandwidth_sharing_model::BandwidthSharingModel`]) and the
//! kvbm-engine pipeline topology (G2/G3 capacity, offload batch sizes).

use crate::common::protocols::MockEngineArgs;

const DEFAULT_G1_G2_BANDWIDTH_GBPS: f64 = 14.0;
const DEFAULT_G2_G3_BANDWIDTH_GBPS: f64 = 7.0;
const DEFAULT_G2_G4_BANDWIDTH_GBPS: f64 = 4.0;
/// Upper bound on derived G2 logical slots to avoid wasteful dense metadata
/// preallocation for huge theoretical tier sizes (e.g. 17 TB / 69 KB = 246M slots).
/// Physical admission remains strictly enforced in bytes by CapacityReservations.
pub const DEFAULT_MAX_DERIVED_G2_SLOTS: usize = 2_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G2SlotMode {
    CappedDense,
    RawDerivedDense,
    SparseByteNative,
}

impl G2SlotMode {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "capped_dense" => Ok(Self::CappedDense),
            "raw_derived_dense" => Ok(Self::RawDerivedDense),
            "sparse_byte_native" => Ok(Self::SparseByteNative),
            other => anyhow::bail!("unsupported g2_slot_mode={other:?}"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::CappedDense => "capped_dense",
            Self::RawDerivedDense => "raw_derived_dense",
            Self::SparseByteNative => "sparse_byte_native",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OfflineKvbmBackend {
    Reference,
    /// Diagnostic A/B mode: reference G1 plus compact sparse/lazy G2.
    CompactG2Only,
    Compact,
}

impl OfflineKvbmBackend {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "reference" => Ok(Self::Reference),
            "compact_g2_only" => Ok(Self::CompactG2Only),
            "compact" => Ok(Self::Compact),
            other => anyhow::bail!("unsupported offline_kvbm_backend={other:?}"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::CompactG2Only => "compact_g2_only",
            Self::Compact => "compact",
        }
    }

    pub fn uses_compact_g2(self) -> bool {
        matches!(self, Self::CompactG2Only | Self::Compact)
    }
}

#[derive(Debug, Clone)]
pub struct KvbmOffloadConfig {
    /// Number of G2 blocks to simulate. Sets the capacity of the
    /// `BlockManager<G2>` owned by the engine.
    pub num_g2_blocks: usize,
    pub raw_derived_g2_slots: Option<usize>,
    pub g2_slot_mode: G2SlotMode,
    pub offline_kvbm_backend: OfflineKvbmBackend,
    /// Rank-local physical G2 byte capacity. When set, admission is enforced
    /// against AIC-derived per-position block sizes in addition to slots.
    pub g2_capacity_bytes: Option<usize>,

    /// Optional physical resource-domain id. Scheduler ranks using the same
    /// id share the node-level G1<->G2 directional bandwidth queues while
    /// retaining independent G2 capacity pools.
    pub g2_fabric_id: Option<String>,

    /// Tokens per logical KV block. Used as the kvbm-logical block size for
    /// G2 so its block identity matches the engine/G1 block geometry.
    pub block_size_tokens: usize,

    /// Batch size for the G1→G2 pipeline. Offloads are grouped into
    /// batches of this size before being handed to the worker.
    pub offload_batch_size: usize,

    /// Optional number of shared G3 blocks to simulate. When set, the mocker
    /// wires a G2→G3 presence pipeline chained after G1→G2.
    pub num_g3_blocks: Option<usize>,

    /// Enable shared G4 object-storage simulation. When true, the mocker
    /// wires a G2→G4 object pipeline chained after G1→G2 and lets lookup
    /// stage G4 hits back through G2.
    pub enable_g4_storage: bool,

    /// Finite deployment-shared G4 capacity in bytes. G4 uses deterministic
    /// LRU eviction when admitting an object would exceed this limit.
    pub g4_capacity_bytes: Option<usize>,

    /// Bytes per block — used by `MockWorker` to compute transfer size
    /// from block counts. Typically `block_size_tokens * kv_bytes_per_token`.
    /// `None` means "unknown" and the worker will use `0` as a sentinel
    /// (every transfer completes instantly).
    pub block_size_bytes: Option<usize>,

    /// AIC-derived incremental bytes by logical block position. When set,
    /// this profile supersedes `block_size_bytes` for transfer timing.
    pub block_footprint_bytes: Option<Vec<usize>>,

    /// Auditable provenance for the dynamic footprint.
    pub footprint_source: Option<String>,

    /// Throughput of the G1→G2 offload link in GB/s. Non-positive values
    /// mean "infinite bandwidth" (transfers complete instantly).
    pub bandwidth_g1_to_g2_gbps: f64,

    /// Throughput of the G2→G1 onboard link in GB/s. Non-positive values
    /// mean "infinite bandwidth" (transfers complete instantly).
    pub bandwidth_g2_to_g1_gbps: f64,

    /// Throughput of the G2→G3 offload link in GB/s. Defaults to a
    /// conservative single local NVMe/SSD bandwidth; non-positive values
    /// mean "infinite bandwidth" (transfers complete instantly).
    pub bandwidth_g2_to_g3_gbps: f64,

    /// Throughput of the G3→G2 staging link in GB/s. Defaults to the same
    /// conservative single local NVMe/SSD bandwidth as G2→G3; non-positive
    /// values mean "infinite bandwidth" (transfers complete instantly).
    pub bandwidth_g3_to_g2_gbps: f64,

    /// Throughput of the G2→G4 object offload link in GB/s. Defaults to a
    /// conservative object-tier bandwidth; non-positive values mean
    /// "infinite bandwidth" (transfers complete instantly).
    pub bandwidth_g2_to_g4_gbps: f64,

    /// Throughput of the G4→G2 object staging link in GB/s. Defaults to the
    /// same value as G2→G4; non-positive values mean "infinite bandwidth"
    /// (transfers complete instantly).
    pub bandwidth_g4_to_g2_gbps: f64,
}

impl Default for KvbmOffloadConfig {
    fn default() -> Self {
        Self {
            num_g2_blocks: 100_000,
            raw_derived_g2_slots: None,
            g2_slot_mode: G2SlotMode::CappedDense,
            offline_kvbm_backend: OfflineKvbmBackend::Reference,
            g2_capacity_bytes: None,
            g2_fabric_id: None,
            block_size_tokens: 64,
            offload_batch_size: 32,
            num_g3_blocks: None,
            enable_g4_storage: false,
            g4_capacity_bytes: None,
            block_size_bytes: None,
            block_footprint_bytes: None,
            footprint_source: None,
            bandwidth_g1_to_g2_gbps: DEFAULT_G1_G2_BANDWIDTH_GBPS,
            bandwidth_g2_to_g1_gbps: DEFAULT_G1_G2_BANDWIDTH_GBPS,
            bandwidth_g2_to_g3_gbps: DEFAULT_G2_G3_BANDWIDTH_GBPS,
            bandwidth_g3_to_g2_gbps: DEFAULT_G2_G3_BANDWIDTH_GBPS,
            bandwidth_g2_to_g4_gbps: DEFAULT_G2_G4_BANDWIDTH_GBPS,
            bandwidth_g4_to_g2_gbps: DEFAULT_G2_G4_BANDWIDTH_GBPS,
        }
    }
}

impl KvbmOffloadConfig {
    /// Derive an offload config from scheduler-level [`MockEngineArgs`].
    ///
    /// Returns `Ok(None)` unless both `num_g2_blocks` and a resolved
    /// `kv_bytes_per_token` are set. Positive `num_g2_blocks` is the explicit
    /// opt-in for the G2 tier; `kv_bytes_per_token` may be populated by the
    /// Python entrypoints from model metadata and is required to compute
    /// `block_size_bytes`.
    /// Caller should interpret `Ok(None)` as "don't attach an offload engine
    /// for this run".
    pub fn from_args(args: &MockEngineArgs) -> anyhow::Result<Option<Self>> {
        let num_g3_blocks = args
            .num_g3_blocks
            .and_then(|block_count| (block_count > 0).then_some(block_count));
        let enable_g4_storage = args.enable_g4_storage;
        let g2_slot_mode = G2SlotMode::parse(&args.g2_slot_mode)?;
        let offline_kvbm_backend = OfflineKvbmBackend::parse(&args.offline_kvbm_backend)?;
        if offline_kvbm_backend.uses_compact_g2() && g2_slot_mode != G2SlotMode::SparseByteNative {
            anyhow::bail!("compact G2 backend requires g2_slot_mode=sparse_byte_native");
        }
        let raw_derived_g2_slots = match (
            args.g2_capacity_bytes,
            args.kv_block_footprint_bytes.as_ref(),
        ) {
            (Some(capacity), Some(profile)) => {
                let min_block = profile.iter().copied().min().ok_or_else(|| {
                    anyhow::anyhow!("cannot derive G2 slots from an empty KV footprint")
                })?;
                let raw_slots = capacity / min_block;
                if raw_slots == 0 {
                    anyhow::bail!(
                        "g2_capacity_bytes={capacity} cannot hold the smallest AIC KV block ({min_block} bytes)"
                    );
                }
                Some(raw_slots)
            }
            _ => None,
        };
        let derived_num_g2_blocks = raw_derived_g2_slots.map(|raw_slots| match g2_slot_mode {
            G2SlotMode::CappedDense => raw_slots.min(DEFAULT_MAX_DERIVED_G2_SLOTS),
            G2SlotMode::RawDerivedDense | G2SlotMode::SparseByteNative => raw_slots,
        });
        let Some(num_g2_blocks) = args.num_g2_blocks.or(derived_num_g2_blocks) else {
            if num_g3_blocks.is_some() || enable_g4_storage {
                anyhow::bail!(
                    "G3/G4 offload requires num_g2_blocks because mocker stages lower tiers through G2"
                );
            }
            return Ok(None);
        };
        if num_g2_blocks == 0 {
            return Ok(None);
        }
        let has_dynamic_footprint = args
            .kv_block_footprint_bytes
            .as_ref()
            .is_some_and(|profile| !profile.is_empty());
        if args.g2_capacity_bytes.is_some() && !has_dynamic_footprint {
            anyhow::bail!(
                "g2_capacity_bytes requires an AIC kv_block_footprint_bytes profile; exact byte admission must not use the linear fallback"
            );
        }
        let bpt = args.kv_bytes_per_token;
        if !has_dynamic_footprint && bpt.is_none() {
            if num_g3_blocks.is_some() || enable_g4_storage {
                anyhow::bail!(
                    "G3/G4 offload requires an AIC kv_block_footprint_bytes profile or kv_bytes_per_token so mocker can size lower-tier transfers"
                );
            }
            return Ok(None);
        }
        let defaults = Self::default();
        let offload_batch_size = args
            .offload_batch_size
            .filter(|batch_size| *batch_size > 0)
            .unwrap_or(defaults.offload_batch_size);
        Ok(Some(Self {
            num_g2_blocks,
            raw_derived_g2_slots,
            g2_slot_mode,
            offline_kvbm_backend,
            g2_capacity_bytes: args.g2_capacity_bytes,
            g2_fabric_id: args.g2_fabric_id.clone(),
            block_size_tokens: args.block_size,
            offload_batch_size,
            num_g3_blocks,
            enable_g4_storage,
            g4_capacity_bytes: args.g4_capacity_bytes,
            block_size_bytes: bpt.map(|bytes| args.block_size * bytes),
            block_footprint_bytes: args.kv_block_footprint_bytes.clone(),
            footprint_source: args.kv_footprint_source.clone(),
            bandwidth_g1_to_g2_gbps: args
                .bandwidth_g1_to_g2_gbps
                .unwrap_or(defaults.bandwidth_g1_to_g2_gbps),
            bandwidth_g2_to_g1_gbps: args
                .bandwidth_g2_to_g1_gbps
                .unwrap_or(defaults.bandwidth_g2_to_g1_gbps),
            bandwidth_g2_to_g3_gbps: args
                .bandwidth_g2_to_g3_gbps
                .unwrap_or(defaults.bandwidth_g2_to_g3_gbps),
            bandwidth_g3_to_g2_gbps: args
                .bandwidth_g3_to_g2_gbps
                .unwrap_or(defaults.bandwidth_g3_to_g2_gbps),
            bandwidth_g2_to_g4_gbps: args
                .bandwidth_g2_to_g4_gbps
                .unwrap_or(defaults.bandwidth_g2_to_g4_gbps),
            bandwidth_g4_to_g2_gbps: args
                .bandwidth_g4_to_g2_gbps
                .unwrap_or(defaults.bandwidth_g4_to_g2_gbps),
        }))
    }

    pub fn block_bytes_at(&self, position: u64) -> anyhow::Result<usize> {
        if let Some(profile) = self.block_footprint_bytes.as_ref() {
            return profile.get(position as usize).copied().ok_or_else(|| {
                anyhow::anyhow!(
                    "KV footprint profile from {:?} has no entry for logical block position {}",
                    self.footprint_source,
                    position
                )
            });
        }
        self.block_size_bytes
            .ok_or_else(|| anyhow::anyhow!("offload block byte size is unresolved"))
    }

    pub fn bytes_for_positions(
        &self,
        positions: impl IntoIterator<Item = u64>,
    ) -> anyhow::Result<usize> {
        positions.into_iter().try_fold(0usize, |total, position| {
            total
                .checked_add(self.block_bytes_at(position)?)
                .ok_or_else(|| anyhow::anyhow!("KV transfer byte count overflow"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_args_none_when_kv_bytes_per_token_missing() {
        let args = MockEngineArgs::builder()
            .num_g2_blocks(Some(10_000))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        assert!(args.kv_bytes_per_token.is_none());
        assert!(KvbmOffloadConfig::from_args(&args).unwrap().is_none());
    }

    #[test]
    fn from_args_errors_when_g3_kv_bytes_per_token_missing() {
        let args = MockEngineArgs::builder()
            .num_g2_blocks(Some(10_000))
            .num_g3_blocks(Some(20_000))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        let error = KvbmOffloadConfig::from_args(&args).unwrap_err();
        assert!(
            error.to_string().contains("kv_block_footprint_bytes")
                && error.to_string().contains("kv_bytes_per_token"),
            "unexpected error: {error}",
        );
    }

    #[test]
    fn from_args_none_when_num_g2_blocks_missing() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .kv_bytes_per_token(Some(131_072))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        assert!(args.num_g2_blocks.is_none());
        assert!(KvbmOffloadConfig::from_args(&args).unwrap().is_none());
    }

    #[test]
    fn from_args_none_when_num_g2_blocks_zero() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .kv_bytes_per_token(Some(131_072))
            .num_g2_blocks(Some(0))
            .build()
            .unwrap();

        assert!(KvbmOffloadConfig::from_args(&args).unwrap().is_none());
    }

    #[test]
    fn from_args_computes_block_size_bytes() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .kv_bytes_per_token(Some(131_072))
            .num_g2_blocks(Some(10_000))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        let cfg = KvbmOffloadConfig::from_args(&args)
            .unwrap()
            .expect("bpt set");
        assert_eq!(cfg.block_size_bytes, Some(64 * 131_072));
        // Defaults preserved.
        assert_eq!(cfg.num_g2_blocks, 10_000);
        assert_eq!(cfg.block_size_tokens, 64);
        assert_eq!(cfg.offload_batch_size, 32);
        assert_eq!(cfg.num_g3_blocks, None);
        assert!(!cfg.enable_g4_storage);
        assert_eq!(cfg.bandwidth_g1_to_g2_gbps, DEFAULT_G1_G2_BANDWIDTH_GBPS);
        assert_eq!(cfg.bandwidth_g2_to_g1_gbps, DEFAULT_G1_G2_BANDWIDTH_GBPS);
        assert_eq!(cfg.bandwidth_g2_to_g3_gbps, DEFAULT_G2_G3_BANDWIDTH_GBPS);
        assert_eq!(cfg.bandwidth_g3_to_g2_gbps, DEFAULT_G2_G3_BANDWIDTH_GBPS);
        assert_eq!(cfg.bandwidth_g2_to_g4_gbps, DEFAULT_G2_G4_BANDWIDTH_GBPS);
        assert_eq!(cfg.bandwidth_g4_to_g2_gbps, DEFAULT_G2_G4_BANDWIDTH_GBPS);
    }

    #[test]
    fn from_args_threads_aic_dynamic_footprint_and_g2_byte_capacity() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .num_g2_blocks(Some(10_000))
            .g2_capacity_bytes(Some(1_000_000))
            .kv_block_footprint_bytes(Some(vec![200_000, 220_000, 180_000]))
            .kv_footprint_source(Some("aic:test".to_string()))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        let cfg = KvbmOffloadConfig::from_args(&args)
            .unwrap()
            .expect("dynamic profile should enable G2");

        assert_eq!(cfg.g2_capacity_bytes, Some(1_000_000));
        assert_eq!(
            cfg.block_footprint_bytes,
            Some(vec![200_000, 220_000, 180_000])
        );
        assert_eq!(cfg.footprint_source.as_deref(), Some("aic:test"));
        assert_eq!(cfg.block_size_bytes, None);
        assert_eq!(cfg.block_bytes_at(1).unwrap(), 220_000);
        assert_eq!(cfg.bytes_for_positions([0, 2]).unwrap(), 380_000);
    }

    #[test]
    fn from_args_caps_derived_g2_slots_to_active_bound() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .g2_capacity_bytes(Some(17_000_000_000_000)) // 17 TB
            .kv_block_footprint_bytes(Some(vec![69_120])) // 17 TB / 69 KB = 245.9M raw slots
            .kv_footprint_source(Some("aic:test".to_string()))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        let cfg = KvbmOffloadConfig::from_args(&args)
            .unwrap()
            .expect("should build offload config");

        assert_eq!(cfg.g2_capacity_bytes, Some(17_000_000_000_000));
        assert_eq!(cfg.num_g2_blocks, DEFAULT_MAX_DERIVED_G2_SLOTS);
        assert_eq!(cfg.g2_slot_mode, G2SlotMode::CappedDense);
        assert_eq!(cfg.raw_derived_g2_slots, Some(245_949_074));
    }

    #[test]
    fn raw_dense_and_sparse_modes_keep_the_byte_derived_logical_ceiling() {
        let raw_slots = 17_000_000_000_000usize / 69_120usize;
        for (mode, backend, expected_mode, expected_backend) in [
            (
                "raw_derived_dense",
                "reference",
                G2SlotMode::RawDerivedDense,
                OfflineKvbmBackend::Reference,
            ),
            (
                "sparse_byte_native",
                "compact_g2_only",
                G2SlotMode::SparseByteNative,
                OfflineKvbmBackend::CompactG2Only,
            ),
            (
                "sparse_byte_native",
                "compact",
                G2SlotMode::SparseByteNative,
                OfflineKvbmBackend::Compact,
            ),
        ] {
            let args = MockEngineArgs::builder()
                .block_size(64)
                .g2_capacity_bytes(Some(17_000_000_000_000))
                .kv_block_footprint_bytes(Some(vec![69_120]))
                .kv_footprint_source(Some("aic:test".to_string()))
                .g2_slot_mode(mode.to_string())
                .offline_kvbm_backend(backend.to_string())
                .build()
                .unwrap()
                .normalized()
                .unwrap();
            let cfg = KvbmOffloadConfig::from_args(&args)
                .unwrap()
                .expect("byte-native G2 should build");
            assert_eq!(cfg.num_g2_blocks, raw_slots);
            assert_eq!(cfg.raw_derived_g2_slots, Some(raw_slots));
            assert_eq!(cfg.g2_slot_mode, expected_mode);
            assert_eq!(cfg.offline_kvbm_backend, expected_backend);
        }
    }

    #[test]
    fn from_args_rejects_g2_byte_capacity_without_aic_profile() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .num_g2_blocks(Some(10_000))
            .g2_capacity_bytes(Some(1_000_000))
            .kv_bytes_per_token(Some(1024))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        let error = KvbmOffloadConfig::from_args(&args).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires an AIC kv_block_footprint_bytes profile"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn from_args_threads_optional_knobs_when_set() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .kv_bytes_per_token(Some(131_072))
            .num_g2_blocks(Some(10_000))
            .offload_batch_size(Some(16))
            .num_g3_blocks(Some(20_000))
            .enable_g4_storage(true)
            .g4_capacity_bytes(Some(64_000_000))
            .bandwidth_g1_to_g2_gbps(Some(8.0))
            .bandwidth_g2_to_g1_gbps(Some(12.0))
            .bandwidth_g2_to_g3_gbps(Some(3.0))
            .bandwidth_g3_to_g2_gbps(Some(4.0))
            .bandwidth_g2_to_g4_gbps(Some(5.0))
            .bandwidth_g4_to_g2_gbps(Some(6.0))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        let cfg = KvbmOffloadConfig::from_args(&args)
            .unwrap()
            .expect("bpt set");
        assert_eq!(cfg.num_g2_blocks, 10_000);
        assert_eq!(cfg.block_size_tokens, 64);
        assert_eq!(cfg.offload_batch_size, 16);
        assert_eq!(cfg.num_g3_blocks, Some(20_000));
        assert!(cfg.enable_g4_storage);
        assert_eq!(cfg.g4_capacity_bytes, Some(64_000_000));
        assert_eq!(cfg.bandwidth_g1_to_g2_gbps, 8.0);
        assert_eq!(cfg.bandwidth_g2_to_g1_gbps, 12.0);
        assert_eq!(cfg.bandwidth_g2_to_g3_gbps, 3.0);
        assert_eq!(cfg.bandwidth_g3_to_g2_gbps, 4.0);
        assert_eq!(cfg.bandwidth_g2_to_g4_gbps, 5.0);
        assert_eq!(cfg.bandwidth_g4_to_g2_gbps, 6.0);
    }

    #[test]
    fn from_args_errors_when_g4_num_g2_blocks_missing() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .kv_bytes_per_token(Some(131_072))
            .enable_g4_storage(true)
            .build()
            .unwrap();
        let error = KvbmOffloadConfig::from_args(&args).unwrap_err();
        assert!(
            error.to_string().contains("requires num_g2_blocks"),
            "unexpected error: {error}",
        );
    }

    #[test]
    fn from_args_errors_when_g4_kv_bytes_per_token_missing() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .num_g2_blocks(Some(10_000))
            .enable_g4_storage(true)
            .g4_capacity_bytes(Some(64_000_000))
            .build()
            .unwrap();
        let error = KvbmOffloadConfig::from_args(&args).unwrap_err();
        assert!(
            error.to_string().contains("kv_block_footprint_bytes")
                && error.to_string().contains("kv_bytes_per_token"),
            "unexpected error: {error}",
        );
    }

    #[test]
    fn from_args_treats_zero_optional_g3_and_batch_size_as_disabled() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .kv_bytes_per_token(Some(131_072))
            .num_g2_blocks(Some(10_000))
            .num_g3_blocks(Some(0))
            .offload_batch_size(Some(0))
            .build()
            .unwrap();

        let cfg = KvbmOffloadConfig::from_args(&args)
            .unwrap()
            .expect("G2 remains enabled");
        assert_eq!(cfg.num_g2_blocks, 10_000);
        assert_eq!(cfg.num_g3_blocks, None);
        assert_eq!(
            cfg.offload_batch_size,
            KvbmOffloadConfig::default().offload_batch_size
        );
    }
}
