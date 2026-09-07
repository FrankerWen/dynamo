// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use dynamo_kv_router::config::RouterQueuePolicy;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use uuid::Uuid;
use validator::{Validate, ValidationError};

use crate::common::perf_model::PerfModel;
use dynamo_kv_router::protocols::{KvCacheEvent, StorageTier};
use dynamo_tokens::blocks::UniqueBlock;
use dynamo_tokens::{BlockHash, PositionalLineageHash, SequenceHash, Token};

/// Metadata marker type for kvbm-logical blocks in the mocker's G1 pool.
#[derive(Clone, Debug)]
pub struct G1;

/// Eviction strategy for the kvbm-logical inactive pool.
///
/// `Lineage` is the default and matches kvbm-logical's own default — it evicts
/// leaf blocks first, which subsumes the preemption-priority behaviour that the
/// mocker's old `LRUEvictor::push_front` provided.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum MockerEvictionBackend {
    Lru,
    MultiLru,
    #[default]
    Lineage,
}

/// G1 implementation used by the shared vLLM/TRT-LLM mock scheduler.
///
/// `Native` is the default self-contained physical-copy pool. `Kvbm` preserves
/// the existing kvbm-logical implementation and is selected automatically when
/// the legacy G2/G3/G4 offload path is enabled. SGLang ignores this setting and
/// uses its own KV manager.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum G1Backend {
    Kvbm,
    #[default]
    Native,
}

/// Trait for publishing KV cache events.
/// This abstracts the runtime dependency so mocker components can remain generic.
pub trait KvCacheEventSink: Send + Sync {
    fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()>;

    fn publish_with_storage_tier(
        &self,
        event: KvCacheEvent,
        _storage_tier: StorageTier,
    ) -> anyhow::Result<()> {
        self.publish(event)
    }

    /// Publishes events that share one source visibility boundary.
    ///
    /// Implementations that do not have a native batch representation retain
    /// singleton delivery semantics by default.
    fn publish_batch_with_storage_tiers(
        &self,
        events: Vec<(KvCacheEvent, StorageTier)>,
    ) -> anyhow::Result<()> {
        let mut first_error = None;
        for (event, storage_tier) in events {
            if let Err(error) = self.publish_with_storage_tier(event, storage_tier) {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

/// Raw KV event payload used by transport-specific publishers such as the
/// vLLM-native ZMQ event stream.
#[derive(Debug, Clone)]
pub struct RawKvEvent {
    pub event: KvCacheEvent,
    pub block_token_ids: Option<Vec<Vec<u32>>>,
    pub storage_tier: StorageTier,
}

/// Trait for publishing transport-specific raw KV event payloads.
pub trait RawKvEventSink: Send + Sync {
    fn publish(&self, event: RawKvEvent) -> anyhow::Result<()>;

    /// Publishes raw events that share one source visibility boundary.
    ///
    /// Implementations that do not have a native batch representation retain
    /// singleton delivery semantics by default.
    fn publish_batch(&self, events: Vec<RawKvEvent>) -> anyhow::Result<()> {
        let mut first_error = None;
        for event in events {
            if let Err(error) = self.publish(event) {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

/// Shared KV event publisher bundle used by schedulers and KV managers.
#[derive(Clone, Default)]
pub struct KvEventPublishers {
    event_sink: Option<Arc<dyn KvCacheEventSink>>,
    raw_sink: Option<Arc<dyn RawKvEventSink>>,
}

impl KvEventPublishers {
    pub fn new(
        event_sink: Option<Arc<dyn KvCacheEventSink>>,
        raw_sink: Option<Arc<dyn RawKvEventSink>>,
    ) -> Self {
        Self {
            event_sink,
            raw_sink,
        }
    }

    pub fn raw_enabled(&self) -> bool {
        self.raw_sink.is_some()
    }

    pub fn is_empty(&self) -> bool {
        self.event_sink.is_none() && self.raw_sink.is_none()
    }

    pub fn publish(
        &self,
        event: KvCacheEvent,
        block_token_ids: Option<&[Vec<u32>]>,
    ) -> anyhow::Result<()> {
        self.publish_with_storage_tier(event, block_token_ids, StorageTier::Device)
    }

    pub fn publish_with_storage_tier(
        &self,
        event: KvCacheEvent,
        block_token_ids: Option<&[Vec<u32>]>,
        storage_tier: StorageTier,
    ) -> anyhow::Result<()> {
        if let Some(sink) = self.event_sink.as_ref() {
            sink.publish_with_storage_tier(event.clone(), storage_tier)?;
        }

        if let Some(sink) = self.raw_sink.as_ref() {
            sink.publish(RawKvEvent {
                event,
                block_token_ids: block_token_ids.map(|token_ids| token_ids.to_vec()),
                storage_tier,
            })?;
        }

        Ok(())
    }

    /// Publishes normal KV events without also forwarding them to a raw sink.
    ///
    /// Deferred live-scheduler forwarding uses this to preserve its source
    /// visibility boundary for normal and raw sinks independently.
    pub(crate) fn publish_event_sink_batch_only(
        &self,
        events: Vec<(KvCacheEvent, StorageTier)>,
    ) -> anyhow::Result<()> {
        if let Some(sink) = self.event_sink.as_ref() {
            sink.publish_batch_with_storage_tiers(events)?;
        }
        Ok(())
    }

    /// Publishes raw events as one source visibility boundary.
    pub(crate) fn publish_raw_batch(&self, events: Vec<RawKvEvent>) -> anyhow::Result<()> {
        if let Some(sink) = self.raw_sink.as_ref() {
            sink.publish_batch(events)?;
        }
        Ok(())
    }
}

/// Per-iteration forward pass snapshot, mirroring the Python `ForwardPassMetrics`
/// schema in `components/src/dynamo/common/forward_pass_metrics.py`.
///
/// Produced by the scheduler core after each `execute_pass_internal()` call.
/// Runtime publishers may either stamp identity at serialization time or fill
/// the identity fields directly when snapshots are consumed in-process.
#[derive(Debug, Clone, Default)]
pub struct ForwardPassSnapshot {
    // -- identity --
    // `Default::default()` leaves `version == 0` and identity fields empty or
    // zero, which means an unstamped local snapshot. Runtime publishers may
    // stamp or overwrite these fields at the serialization boundary.
    pub version: u32,
    pub worker_id: String,
    pub dp_rank: u32,
    pub counter_id: u64,
    // -- scheduled requests (executed this iteration) --
    pub num_prefill_requests: u32,
    pub sum_prefill_tokens: u64,
    pub var_prefill_length: f64,
    pub sum_prefill_kv_tokens: u64,
    pub num_decode_requests: u32,
    pub sum_decode_kv_tokens: u64,
    pub var_decode_kv_tokens: f64,
    // -- queued requests (waiting, not scheduled) --
    pub num_queued_prefill: u32,
    pub sum_queued_prefill_tokens: u64,
    pub var_queued_prefill_length: f64,
    pub num_queued_decode: u32,
    pub sum_queued_decode_kv_tokens: u64,
    pub var_queued_decode_kv_tokens: f64,
    // -- timing --
    pub wall_time_secs: f64,
}

/// Trait for publishing forward pass metrics snapshots.
/// This abstracts the FPM publishing pipeline so mocker schedulers remain generic.
pub trait FpmSink: Send + Sync {
    fn publish(&self, snapshot: ForwardPassSnapshot) -> anyhow::Result<()>;
}

/// Optional FPM sink used by schedulers.
/// Wraps `Option<Arc<dyn FpmSink>>` for ergonomic passing and no-op default behavior.
#[derive(Clone, Default)]
pub struct FpmPublisher {
    sink: Option<Arc<dyn FpmSink>>,
}

impl FpmPublisher {
    pub fn new(sink: Option<Arc<dyn FpmSink>>) -> Self {
        Self { sink }
    }

    pub fn publish(&self, snapshot: ForwardPassSnapshot) -> anyhow::Result<()> {
        if let Some(sink) = &self.sink {
            sink.publish(snapshot)?;
        }
        Ok(())
    }
}

pub type NumBlocks = usize;

/// Represents different block movement operations in the cache
/// For Use and Promote variants, block hashes are included for KV event publishing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MoveBlock {
    Use(
        Vec<UniqueBlock>,
        Vec<BlockHash>,
        Vec<PositionalLineageHash>,
        Option<Vec<Vec<u32>>>,
        Option<UniqueBlock>,
    ),
    Deref(Vec<UniqueBlock>),
    Promote(
        Uuid,
        SequenceHash,
        Option<u64>,
        Option<BlockHash>,
        PositionalLineageHash,
        Option<Vec<u32>>,
    ),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MoveBlockResponse {
    Store(Vec<SequenceHash>, Option<u64>),
    Remove(Vec<SequenceHash>),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DirectRequest {
    pub tokens: Vec<Token>,
    pub max_output_tokens: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_ids: Option<Vec<Token>>,
    pub uuid: Option<Uuid>,
    pub dp_rank: u32,
    pub arrival_timestamp_ms: Option<f64>,
    /// TODO: Replay maps this to router queue priority only; mock-engine
    /// scheduling does not consume it yet.
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub priority: i32,
    /// NOTE: Strict priority orders the router's pending queue only. It does
    /// not affect scheduling inside the selected mock engine.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub strict_priority: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_class: Option<String>,
}

impl DirectRequest {
    pub fn router_priorities(&self) -> (f64, u32) {
        (f64::from(self.priority.max(0)), self.strict_priority)
    }

    #[inline]
    pub(crate) fn effective_max_output_tokens(&self) -> usize {
        self.output_token_ids
            .as_ref()
            .map_or(self.max_output_tokens, Vec::len)
    }

    pub(crate) fn clone_with_output_limit(&self, limit: usize) -> Self {
        let max_output_tokens = self.effective_max_output_tokens().min(limit);
        Self {
            tokens: self.tokens.clone(),
            max_output_tokens,
            output_token_ids: self
                .output_token_ids
                .as_ref()
                .map(|ids| ids[..max_output_tokens].to_vec()),
            uuid: self.uuid,
            dp_rank: self.dp_rank,
            arrival_timestamp_ms: self.arrival_timestamp_ms,
            priority: self.priority,
            strict_priority: self.strict_priority,
            policy_class: self.policy_class.clone(),
        }
    }
}

fn is_zero_i32(value: &i32) -> bool {
    *value == 0
}

fn is_zero_u32(value: &u32) -> bool {
    *value == 0
}

/// Represents the cost of prefilling content in the cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillCost {
    pub new_blocks: usize,
    pub new_tokens: usize,
    /// Number of tokens already cached (prefix hit).
    /// isl = cached_tokens + new_tokens
    pub cached_tokens: usize,
    /// Subset of `cached_tokens` backed by active blocks. Physical-capacity
    /// admission discounts only these because inactive reuse is re-consumed.
    pub active_cached_tokens: usize,
}

impl PrefillCost {
    pub fn predict_prefill_compute(
        &self,
        new_tokens: Option<usize>,
        perf_model: &PerfModel,
    ) -> anyhow::Result<f64> {
        let tokens = new_tokens.unwrap_or(self.new_tokens);
        let isl = self.cached_tokens + tokens;
        perf_model.predict_prefill_time(1, isl, self.cached_tokens)
    }
}

/// Signal for output token generation with completion status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSignal {
    pub uuid: Uuid,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_id: Option<Token>,
    /// Terminal flag: the request's lifecycle has ended. Replay drivers free
    /// resources and advance/notify on this.
    pub completed: bool,
    /// Set with `completed` when the request was rejected without ever running
    /// (its footprint exceeds the whole KV pool); drivers free/advance but
    /// exclude it from token/latency/throughput stats.
    #[serde(default)]
    pub rejected: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handoff_delay_ms: Option<f64>,
    /// Prompt tokens served from KV cache at admission (scheduler truth,
    /// post-eviction). Set once, on the request's first output signal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<usize>,
}

/// Preemption policy for evicting decode requests under memory pressure
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PreemptionMode {
    /// Evict the newest request (matches vLLM v1 default)
    #[default]
    Lifo,
    /// Evict the oldest request
    Fifo,
}

impl FromStr for PreemptionMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "lifo" => Ok(Self::Lifo),
            "fifo" => Ok(Self::Fifo),
            _ => Err(format!(
                "Invalid preemption_mode: '{value}'. Must be 'lifo' or 'fifo'."
            )),
        }
    }
}

/// Engine type for selecting scheduling and KV cache simulation behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EngineType {
    /// vLLM-style scheduling with hash-based block KV cache
    #[default]
    Vllm,
    /// SGLang-style scheduling with radix-tree KV cache
    Sglang,
    /// TensorRT-LLM-style scheduling. Reuses the vLLM scheduler
    /// core with a TensorRT-LLM-style admission policy.
    Trtllm,
}

impl FromStr for EngineType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "vllm" => Ok(Self::Vllm),
            "sglang" => Ok(Self::Sglang),
            "trtllm" => Ok(Self::Trtllm),
            _ => Err(format!(
                "Invalid engine_type '{value}'. Must be 'vllm', 'sglang', or 'trtllm'."
            )),
        }
    }
}

/// Scheduling policy applied by the shared vLLM scheduler core.
///
/// Derived from [`EngineType`] (+ engine-specific args) so the core reads a
/// single discriminant instead of re-deriving engine behavior per pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchedulingPolicy {
    /// vLLM semantics: require the current known sequence to fit at waiting
    /// admission, then permit preemption under later KV pressure.
    #[default]
    Vllm,
    /// TRT-LLM `GUARANTEED_NO_EVICT`: reserve `prompt + max_output` per
    /// admitted request up front; never preempt.
    TrtllmGuaranteedNoEvict,
}

/// Worker type for disaggregated serving configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum WorkerType {
    /// Standard aggregated worker handling both prefill and decode
    #[default]
    Aggregated,
    /// Dedicated prefill worker in disaggregated mode
    Prefill,
    /// Dedicated decode worker in disaggregated mode
    Decode,
}

impl FromStr for WorkerType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "aggregated" => Ok(Self::Aggregated),
            "prefill" => Ok(Self::Prefill),
            "decode" => Ok(Self::Decode),
            _ => Err(format!(
                "Invalid worker_type '{value}'. Must be 'aggregated', 'prefill', or 'decode'."
            )),
        }
    }
}

/// Physical KV footprint used to model a coordinated disaggregated transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum KvTransferTimingMode {
    /// Charge the source request's full logical prompt length.
    #[default]
    FullPrompt,
    /// Charge only the physical prompt footprint missing at the destination.
    DestinationMissing,
}

impl FromStr for KvTransferTimingMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "full_prompt" => Ok(Self::FullPrompt),
            "destination_missing" => Ok(Self::DestinationMissing),
            _ => Err(format!(
                "Invalid kv_transfer_timing_mode '{value}'. Must be 'full_prompt' or 'destination_missing'."
            )),
        }
    }
}

/// Configuration for reasoning/thinking token output in the mocker.
///
/// When set, the mocker wraps the first portion of each response in thinking
/// boundary tokens: `[start_token, random..., end_token, random...]`.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ReasoningConfig {
    pub start_thinking_token_id: u32,
    pub end_thinking_token_id: u32,
    #[validate(range(min = 0.0, max = 1.0))]
    pub thinking_ratio: f64,
}

impl ReasoningConfig {
    /// Number of thinking tokens (including start/end boundaries) for a given osl.
    /// Returns 0 if osl < 2 (thinking disabled). Otherwise clamps to [2, osl].
    pub fn num_thinking_tokens(&self, max_output_tokens: usize) -> usize {
        if max_output_tokens < 2 {
            return 0;
        }
        let raw = (max_output_tokens as f64 * self.thinking_ratio).floor() as usize;
        if raw == 0 {
            return 0;
        }
        raw.max(2).min(max_output_tokens)
    }

    /// Number of response tokens after the thinking block.
    pub fn num_response_tokens(&self, max_output_tokens: usize) -> usize {
        max_output_tokens.saturating_sub(self.num_thinking_tokens(max_output_tokens))
    }
}

/// SGLang-specific configuration parameters.
///
/// Grouped into a nested struct to keep the `MockEngineArgs` namespace clean,
/// following the same pattern as [`ReasoningConfig`].
#[derive(Debug, Clone, Serialize, Deserialize, Validate, Default)]
pub struct SglangArgs {
    /// Scheduling policy: "fifo"/"fcfs" or "lpm". Default: "fifo".
    pub schedule_policy: Option<String>,
    /// Radix cache page size in tokens. Default: 1.
    #[validate(range(min = 1))]
    pub page_size: Option<usize>,
    /// Maximum prefill tokens budget per batch. Default: 16384.
    #[validate(range(min = 1))]
    pub max_prefill_tokens: Option<usize>,
    /// Chunked prefill size (max tokens per chunk). Default: 8192.
    #[validate(range(min = 1))]
    pub chunked_prefill_size: Option<usize>,
    /// Clip max new tokens for admission budget. Default: 4096.
    #[validate(range(min = 1))]
    pub clip_max_new_tokens: Option<usize>,
    /// Schedule conservativeness factor (0.0–1.0). Default: 1.0.
    #[validate(range(min = 0.0, max = 1.0))]
    pub schedule_conservativeness: Option<f64>,
}

/// TensorRT-LLM-specific configuration parameters.
///
/// Grouped into a nested struct to keep the `MockEngineArgs` namespace clean,
/// following the same pattern as [`SglangArgs`].
#[derive(Debug, Clone, Serialize, Deserialize, Validate, Default)]
pub struct TrtllmArgs {
    /// Capacity scheduler policy, supported only `"guaranteed_no_evict"`
    /// (TensorRT-LLM's default). Default: `"guaranteed_no_evict"`.
    pub capacity_scheduler_policy: Option<String>,
}

/// Physical G2 resources assigned to one offline scheduler rank.
///
/// Exact replay may place different `(worker_id, dp_rank)` schedulers on
/// different physical nodes.  The role-level [`MockEngineArgs`] remains the
/// immutable scheduler template; this record replaces only the rank-local G2
/// capacity and the node-scoped directional fabric at worker construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G2RankResourceConfig {
    pub worker_id: usize,
    pub dp_rank: u32,
    pub g2_capacity_bytes: usize,
    pub g2_fabric_id: String,
    pub bandwidth_g1_to_g2_gbps: f64,
    pub bandwidth_g2_to_g1_gbps: f64,
}

/// Keeps omitted JSON fields distinct from explicit `null` so serde can replace
/// the old hand-written parser without losing input-config semantics.
#[derive(Debug, Clone, Default)]
enum OptionalConfigValue<T> {
    #[default]
    Missing,
    Present(Option<T>),
}

impl<'de, T> Deserialize<'de> for OptionalConfigValue<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer).map(Self::Present)
    }
}

impl<T> OptionalConfigValue<T> {
    fn into_nullable(self) -> Option<Option<T>> {
        match self {
            Self::Missing => None,
            Self::Present(value) => Some(value),
        }
    }

    fn into_non_null(self, field: &str) -> Result<Option<T>, String> {
        match self {
            Self::Missing => Ok(None),
            Self::Present(Some(value)) => Ok(Some(value)),
            Self::Present(None) => Err(format!("{field} must not be null")),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct MockEngineArgsSerde {
    engine_type: OptionalConfigValue<String>,
    num_gpu_blocks: OptionalConfigValue<usize>,
    block_size: OptionalConfigValue<usize>,
    max_model_len: OptionalConfigValue<usize>,
    max_num_seqs: OptionalConfigValue<usize>,
    max_num_batched_tokens: OptionalConfigValue<usize>,
    enable_prefix_caching: OptionalConfigValue<bool>,
    g1_backend: OptionalConfigValue<G1Backend>,
    enable_chunked_prefill: OptionalConfigValue<bool>,
    speedup_ratio: OptionalConfigValue<f64>,
    decode_speedup_ratio: OptionalConfigValue<f64>,
    dp_size: OptionalConfigValue<u32>,
    startup_time: OptionalConfigValue<f64>,
    worker_type: OptionalConfigValue<String>,
    is_prefill: OptionalConfigValue<bool>,
    is_decode: OptionalConfigValue<bool>,
    planner_profile_data: OptionalConfigValue<PathBuf>,
    aic_backend: OptionalConfigValue<String>,
    aic_system: OptionalConfigValue<String>,
    aic_backend_version: OptionalConfigValue<String>,
    aic_database_mode: OptionalConfigValue<String>,
    aic_systems_path: OptionalConfigValue<String>,
    aic_database_snapshot_sha256: OptionalConfigValue<String>,
    aic_analytical_config: OptionalConfigValue<String>,
    aic_tp_size: OptionalConfigValue<usize>,
    aic_model_path: OptionalConfigValue<String>,
    aic_moe_tp_size: OptionalConfigValue<usize>,
    aic_moe_ep_size: OptionalConfigValue<usize>,
    aic_attention_dp_size: OptionalConfigValue<usize>,
    aic_gemm_dtype: OptionalConfigValue<String>,
    aic_moe_dtype: OptionalConfigValue<String>,
    aic_fmha_dtype: OptionalConfigValue<String>,
    aic_kv_cache_dtype: OptionalConfigValue<String>,
    aic_comm_dtype: OptionalConfigValue<String>,
    aic_nextn: OptionalConfigValue<usize>,
    aic_nextn_accept_rates: OptionalConfigValue<String>,
    aic_mtp_seed: OptionalConfigValue<u64>,
    gpu_memory_utilization: OptionalConfigValue<f64>,
    mem_fraction_static: OptionalConfigValue<f64>,
    free_gpu_memory_fraction: OptionalConfigValue<f64>,
    enable_local_indexer: OptionalConfigValue<bool>,
    bootstrap_port: OptionalConfigValue<u16>,
    handoff_session_timeout_ms: OptionalConfigValue<u64>,
    kv_bytes_per_token: OptionalConfigValue<usize>,
    kv_block_footprint_bytes: OptionalConfigValue<Vec<usize>>,
    kv_footprint_source: OptionalConfigValue<String>,
    kv_transfer_bandwidth: OptionalConfigValue<f64>,
    kv_transfer_timing_mode: OptionalConfigValue<String>,
    num_g2_blocks: OptionalConfigValue<usize>,
    g2_capacity_bytes: OptionalConfigValue<usize>,
    g2_slot_mode: OptionalConfigValue<String>,
    offline_kvbm_backend: OptionalConfigValue<String>,
    g2_fabric_id: OptionalConfigValue<String>,
    g2_rank_resource_configs: OptionalConfigValue<Vec<G2RankResourceConfig>>,
    num_g3_blocks: OptionalConfigValue<usize>,
    enable_g4_storage: OptionalConfigValue<bool>,
    g4_capacity_bytes: OptionalConfigValue<usize>,
    offload_batch_size: OptionalConfigValue<usize>,
    bandwidth_g1_to_g2_gbps: OptionalConfigValue<f64>,
    bandwidth_g2_to_g1_gbps: OptionalConfigValue<f64>,
    bandwidth_g2_to_g3_gbps: OptionalConfigValue<f64>,
    bandwidth_g3_to_g2_gbps: OptionalConfigValue<f64>,
    bandwidth_g2_to_g4_gbps: OptionalConfigValue<f64>,
    bandwidth_g4_to_g2_gbps: OptionalConfigValue<f64>,
    reasoning: OptionalConfigValue<ReasoningConfig>,
    response_replay_trace_path: OptionalConfigValue<PathBuf>,
    zmq_kv_events_port: OptionalConfigValue<u16>,
    zmq_replay_port: OptionalConfigValue<u16>,
    preemption_mode: OptionalConfigValue<String>,
    router_queue_policy: OptionalConfigValue<String>,
    sglang: OptionalConfigValue<SglangArgs>,
    trtllm: OptionalConfigValue<TrtllmArgs>,
    #[serde(rename = "has_perf_model")]
    _has_perf_model: OptionalConfigValue<serde_json::Value>,
}

fn load_perf_model(path: &Path) -> Arc<PerfModel> {
    match PerfModel::from_npz(path) {
        Ok(model) => {
            tracing::info!("Successfully loaded performance model from: {:?}", path);
            Arc::new(model)
        }
        Err(e) => {
            tracing::error!(
                "Failed to load performance model from {:?}: {}. Falling back to polynomial model.",
                path,
                e
            );
            Arc::new(PerfModel::default())
        }
    }
}

/// Configuration arguments for MockEngine
#[derive(Debug, Clone, Serialize, Deserialize, Builder, Validate)]
#[serde(try_from = "MockEngineArgsSerde")]
#[validate(schema(function = "validate_mock_engine_args"))]
#[builder(pattern = "owned", build_fn(public))]
pub struct MockEngineArgs {
    /// Engine type: vLLM, SGLang, or TensorRT-LLM simulation
    #[builder(default = "EngineType::Vllm")]
    pub engine_type: EngineType,

    /// Usable simulated G1 capacity. This preserves the mocker's historical
    /// convention across backends. A raw vLLM `num_gpu_blocks` value also
    /// includes its reserved null block, so parity runs configure real vLLM
    /// with one additional total block.
    #[builder(default = "16384")]
    #[validate(range(min = 1))]
    pub num_gpu_blocks: usize,

    #[builder(default = "0")]
    pub block_size: usize,

    /// Optional vLLM sequence-length limit, including prompt and generated
    /// tokens. Requests with no room to generate are rejected before admission.
    #[builder(default = "None")]
    #[validate(range(min = 1))]
    pub max_model_len: Option<usize>,

    // This was 1024 in the past but reverted back to 256
    #[builder(default = Some(256))]
    #[validate(range(min = 1))]
    pub max_num_seqs: Option<usize>,

    // default for open api server, for llm class it's 16384
    #[builder(default = Some(8192))]
    #[validate(range(min = 1))]
    pub max_num_batched_tokens: Option<usize>,

    #[builder(default = true)]
    pub enable_prefix_caching: bool,

    /// Requested G1 block-manager implementation for the shared vLLM/TRT-LLM
    /// scheduler. `None` selects native unless legacy offload requires KVBM.
    /// Ignored by the SGLang scheduler, which uses `SglangKvManager`.
    #[builder(default = "None", setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub g1_backend: Option<G1Backend>,

    #[builder(default = true)]
    pub enable_chunked_prefill: bool,

    #[builder(default = "1.0")]
    #[validate(range(min = 0.0))]
    pub speedup_ratio: f64,

    /// Additional speedup multiplier applied only to decode steps.
    /// Models speculative decoding (e.g. Eagle) where decode throughput improves
    /// without affecting prefill latency. The effective decode speedup is
    /// `speedup_ratio * decode_speedup_ratio`.
    #[builder(default = "1.0")]
    #[validate(range(min = 0.0))]
    pub decode_speedup_ratio: f64,

    #[builder(default = "1")]
    #[validate(range(min = 1))]
    pub dp_size: u32,

    /// Optional startup time in seconds to simulate engine initialization delay
    #[builder(default = "None")]
    #[validate(range(min = 0.0))]
    pub startup_time: Option<f64>,

    /// Worker type for disaggregated serving (Aggregated, Prefill, or Decode)
    #[builder(default = "WorkerType::Aggregated")]
    pub worker_type: WorkerType,

    /// Original planner profile NPZ path used to materialize `perf_model`.
    #[builder(default = "None")]
    pub planner_profile_data: Option<PathBuf>,

    /// Performance model for timing predictions (not serialized, loaded from planner_profile_data)
    #[serde(skip)]
    #[builder(default = "Arc::new(PerfModel::default())")]
    pub perf_model: Arc<PerfModel>,

    /// If set, indicates direct AIC SDK calls should be used.
    /// The value is the backend name (e.g., "sglang", "vllm").
    /// The Python layer reads this and overrides perf_model with an Aiconfigurator callback.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_backend: Option<String>,

    /// AIC GPU system name (e.g., "h200_sxm"). Required when aic_backend is set.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_system: Option<String>,

    /// AIC backend engine version (e.g., "0.12.0" for vLLM, "0.5.6.post2" for SGLang).
    /// If None, uses the default version for the backend.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_backend_version: Option<String>,

    /// AIC perf query mode. `None` preserves the SILICON default.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_database_mode: Option<String>,

    /// Single systems root selected from the verified multi-root AIC snapshot
    /// for this concrete system. The compiled Rust engine cannot consume a
    /// search-path list and therefore must never guess the first directory.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_systems_path: Option<String>,

    /// Content identity of the verified AIC database snapshot. It salts the
    /// native compiled-engine cache so a database update at the same path can
    /// never reuse a stale callback.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_database_snapshot_sha256: Option<String>,

    /// Canonical JSON for AIC's immutable ANALYTICAL KernelSim policy.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_analytical_config: Option<String>,

    /// Tensor parallel size for AIC latency prediction.
    /// Only affects AIC performance model lookups, not mocker scheduling.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_tp_size: Option<usize>,

    /// HuggingFace model path for AIC latency prediction (e.g., "nvidia/Llama-3.1-8B-Instruct-FP8").
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_model_path: Option<String>,

    /// MoE tensor-parallel size for AIC latency prediction (e.g., 4 for pure MoE-TP).
    /// Required for MoE models; must satisfy: aic_tp_size * aic_attention_dp_size == aic_moe_tp_size * aic_moe_ep_size.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_moe_tp_size: Option<usize>,

    /// MoE expert-parallel size for AIC latency prediction (e.g., 4 for pure EP).
    /// Required for MoE models; must satisfy: aic_tp_size * aic_attention_dp_size == aic_moe_tp_size * aic_moe_ep_size.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_moe_ep_size: Option<usize>,

    /// Attention data-parallel size for AIC latency prediction (default: 1).
    /// Corresponds to the `dp` dimension in AIC CLI output.
    /// Must satisfy: aic_tp_size * aic_attention_dp_size == aic_moe_tp_size * aic_moe_ep_size.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_attention_dp_size: Option<usize>,

    /// Weight dtype override for AIC latency prediction.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_gemm_dtype: Option<String>,

    /// MoE kernel dtype override for AIC latency prediction.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_moe_dtype: Option<String>,

    /// Activation dtype override for AIC latency prediction.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_fmha_dtype: Option<String>,

    /// KV-cache dtype override for AIC latency prediction.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_kv_cache_dtype: Option<String>,

    /// Communication (collective) dtype override for AIC latency prediction.
    #[serde(skip)]
    #[builder(default = "None")]
    pub aic_comm_dtype: Option<String>,

    /// MTP/Eagle speculative-decoding draft-token count (1..=5).
    /// The mocker samples accepted drafts while AIC supplies undiscounted
    /// verification-round latency.
    #[builder(default = "None")]
    #[validate(range(min = 1, max = 5))]
    pub aic_nextn: Option<usize>,

    /// Conditional acceptance rates for draft tokens, comma-separated.
    /// Entry i is P(draft i accepted | every earlier draft was accepted).
    #[builder(default = "None")]
    pub aic_nextn_accept_rates: Option<String>,

    /// Base RNG seed for MTP burst sampling. Worker rank is added with
    /// wrapping arithmetic before constructing each worker-local sampler.
    #[builder(default = "42")]
    pub aic_mtp_seed: u64,

    /// GPU memory fraction for AIC KV capacity estimation with vLLM.
    #[builder(default = "None")]
    #[validate(range(min = 0.0, max = 1.0))]
    pub gpu_memory_utilization: Option<f64>,

    /// Static memory fraction for AIC KV capacity estimation with SGLang.
    #[builder(default = "None")]
    #[validate(range(min = 0.0, max = 1.0))]
    pub mem_fraction_static: Option<f64>,

    /// Fraction of *free* GPU memory (after weights/buffers) allocated to the KV
    /// cache, for AIC KV capacity estimation with TRT-LLM. Mirrors TRT-LLM's
    /// `KvCacheConfig.free_gpu_memory_fraction`. Unlike vLLM's
    /// `gpu_memory_utilization` (a fraction of *total* memory), this is a
    /// fraction of what remains after the model is loaded.
    #[builder(default = "None")]
    #[validate(range(min = 0.0, max = 1.0))]
    pub free_gpu_memory_fraction: Option<f64>,

    /// Enable worker-local KV indexer for tracking this worker's own KV cache state
    #[builder(default = "false")]
    pub enable_local_indexer: bool,

    /// Bootstrap port for disaggregated serving rendezvous.
    /// Prefill workers listen on this port; decode workers connect to it.
    /// If None, bootstrap rendezvous is disabled.
    #[builder(default = "None")]
    pub bootstrap_port: Option<u16>,

    /// Absolute live handoff session timeout, excluding modeled transfer delay.
    #[builder(default = "300_000")]
    #[validate(range(min = 1))]
    pub handoff_session_timeout_ms: u64,

    /// KV cache bytes per token, auto-computed from model config by Python CLI.
    /// Formula: num_layers * 2 * num_kv_heads * head_dim * dtype_bytes
    #[builder(default = "None")]
    pub kv_bytes_per_token: Option<usize>,

    /// Rank-local incremental KV bytes for each logical block position.
    /// When present this AIC-derived profile supersedes the legacy linear
    /// `kv_bytes_per_token` scalar for offload capacity and transfer timing.
    #[builder(default = "None")]
    pub kv_block_footprint_bytes: Option<Vec<usize>>,

    /// Auditable provenance for `kv_block_footprint_bytes`.
    #[builder(default = "None")]
    pub kv_footprint_source: Option<String>,

    /// KV cache transfer bandwidth in GB/s for disaggregated serving latency simulation.
    /// Default: 64.0 (inter-node InfiniBand). Set to 0 to disable KV transfer delay.
    /// For intra-node NVLink, typical value is ~450.
    #[builder(default = "None")]
    #[validate(range(min = 0.0))]
    pub kv_transfer_bandwidth: Option<f64>,

    /// Selects whether disaggregated transfer timing charges the full prompt
    /// or only the physical prompt footprint missing at the destination.
    #[builder(default = "KvTransferTimingMode::FullPrompt")]
    pub kv_transfer_timing_mode: KvTransferTimingMode,

    /// KVBM G2 (host DRAM) block capacity. When the `kvbm-offload`
    /// feature is enabled, setting this explicitly opts the mocker into
    /// G2 offload simulation. When unset or set to 0, no G2 offload engine
    /// is attached.
    #[builder(default = "None")]
    #[validate(range(min = 1))]
    pub num_g2_blocks: Option<usize>,

    /// Rank-local physical host-memory budget backing G2. Exact AgentX
    /// trials use this byte cap together with the AIC block-footprint profile;
    /// `num_g2_blocks` remains a scheduler slot bound for legacy KVBM APIs.
    #[builder(default = "None")]
    #[validate(range(min = 1))]
    pub g2_capacity_bytes: Option<usize>,

    /// G2 logical-slot derivation: `capped_dense` (legacy 2M cap),
    /// `raw_derived_dense` (diagnostic only), or `sparse_byte_native`.
    #[builder(default = "\"capped_dense\".to_string()")]
    pub g2_slot_mode: String,

    /// Offline G2 metadata backend. `reference` preserves the generic KVBM
    /// layout; `compact` lazily materializes dense slot ids and uses sparse
    /// hash indexes so memory follows resident/pending state.
    #[builder(default = "\"reference\".to_string()")]
    pub offline_kvbm_backend: String,

    /// Physical host-memory fabric shared by scheduler ranks. Ranks with the
    /// same non-empty id contend on one pair of G1<->G2 directional
    /// processor-sharing queues. `None` preserves the legacy rank-local
    /// bandwidth model.
    #[builder(default = "None")]
    pub g2_fabric_id: Option<String>,

    /// Offline-only physical resource overrides keyed by `(worker_id,
    /// dp_rank)`.  When present, every constructed scheduler rank must have
    /// exactly one entry.  This prevents a role-wide G2 capacity or fabric
    /// from being silently replicated across ranks placed on different nodes.
    #[builder(default = "None")]
    pub g2_rank_resource_configs: Option<Vec<G2RankResourceConfig>>,

    /// KVBM G3 shared lower-tier block capacity. Positive values require
    /// `num_g2_blocks` and a resolvable KV block byte size; 0 disables G3.
    #[builder(default = "None")]
    #[validate(range(min = 1))]
    pub num_g3_blocks: Option<usize>,

    /// Enable KVBM mock G4 object-storage simulation. G4 stages through G2
    /// and uses object presence operations instead of a `BlockManager<G4>`.
    #[builder(default = "false")]
    pub enable_g4_storage: bool,

    /// Deployment-shared finite G4 byte capacity. Required whenever G4 is
    /// enabled; the exact mocker uses deterministic LRU eviction at this cap.
    #[builder(default = "None")]
    #[validate(range(min = 1))]
    pub g4_capacity_bytes: Option<usize>,

    /// Batch size for the G1→G2 offload pipeline. Offloads are grouped
    /// into batches of this size before being handed to the worker.
    /// Only consulted when the `kvbm-offload` feature is enabled;
    /// falls back to the `KvbmOffloadConfig` default when unset or 0.
    #[builder(default = "None")]
    #[validate(range(min = 1))]
    pub offload_batch_size: Option<usize>,

    /// G1→G2 offload bandwidth in GB/s for the PS-queue simulation.
    /// Only consulted when the `kvbm-offload` feature is enabled;
    /// falls back to the `KvbmOffloadConfig` default (host DRAM PCIe
    /// ballpark) when unset.
    #[builder(default = "None")]
    #[validate(range(min = 0.0))]
    pub bandwidth_g1_to_g2_gbps: Option<f64>,

    /// G2→G1 onboard bandwidth in GB/s for the PS-queue simulation.
    /// Only consulted when the `kvbm-offload` feature is enabled;
    /// falls back to the `KvbmOffloadConfig` default when unset.
    #[builder(default = "None")]
    #[validate(range(min = 0.0))]
    pub bandwidth_g2_to_g1_gbps: Option<f64>,

    /// G2→G3 offload bandwidth in GB/s for the shared PS-queue simulation.
    #[builder(default = "None")]
    #[validate(range(min = 0.0))]
    pub bandwidth_g2_to_g3_gbps: Option<f64>,

    /// G3→G2 staging bandwidth in GB/s for the shared PS-queue simulation.
    #[builder(default = "None")]
    #[validate(range(min = 0.0))]
    pub bandwidth_g3_to_g2_gbps: Option<f64>,

    /// G2→G4 object offload bandwidth in GB/s for the shared PS-queue simulation.
    #[builder(default = "None")]
    #[validate(range(min = 0.0))]
    pub bandwidth_g2_to_g4_gbps: Option<f64>,

    /// G4→G2 object staging bandwidth in GB/s for the shared PS-queue simulation.
    #[builder(default = "None")]
    #[validate(range(min = 0.0))]
    pub bandwidth_g4_to_g2_gbps: Option<f64>,

    /// Reasoning/thinking token configuration.
    /// When set, the mocker wraps output in thinking boundary tokens.
    #[builder(default = "None")]
    pub reasoning: Option<ReasoningConfig>,

    /// Optional Mooncake trace with exact output token IDs keyed by
    /// `output_replay_id` annotations. Direct replay paths carry the same token
    /// IDs on `DirectRequest` and do not need this lookup.
    #[builder(default = "None")]
    pub response_replay_trace_path: Option<PathBuf>,

    /// ZMQ port for publishing KV events in vLLM's native wire format.
    /// When set, the scheduler publishes to a ZMQ PUB socket instead of directly to NATS.
    /// A KvEventPublisher relay subscribes to this socket and forwards events to NATS.
    #[builder(default = "None")]
    pub zmq_kv_events_port: Option<u16>,

    /// ZMQ ROUTER port for replay of buffered KV event batches.
    /// When set alongside `zmq_kv_events_port`, the mocker binds a ROUTER socket
    /// that streams back buffered batches by sequence number on request.
    /// Port is offset by dp_rank (replay_port + dp_rank).
    #[builder(default = "None")]
    pub zmq_replay_port: Option<u16>,

    /// Preemption mode for decode eviction under memory pressure.
    /// Lifo (default) evicts the newest request; Fifo evicts the oldest.
    #[builder(default)]
    pub preemption_mode: PreemptionMode,

    /// Optional replay-only override for the router queue policy.
    #[builder(default = "None")]
    pub router_queue_policy: Option<RouterQueuePolicy>,

    /// SGLang-specific configuration. Only used when `engine_type == Sglang`.
    #[builder(default = "None")]
    pub sglang: Option<SglangArgs>,

    /// TensorRT-LLM-specific configuration. Only used when `engine_type == Trtllm`.
    #[builder(default = "None")]
    pub trtllm: Option<TrtllmArgs>,
}

fn mock_engine_args_validation_error(code: &'static str, message: String) -> ValidationError {
    let mut error = ValidationError::new(code);
    error.message = Some(message.into());
    error
}

fn validate_mock_engine_args(args: &MockEngineArgs) -> Result<(), ValidationError> {
    if let Some(mode) = args.aic_database_mode.as_deref()
        && !matches!(
            mode,
            "SILICON" | "HYBRID" | "EMPIRICAL" | "SOL" | "SOL_FULL" | "ANALYTICAL"
        )
    {
        return Err(mock_engine_args_validation_error(
            "aic_database_mode_invalid",
            format!(
                "aic_database_mode must be one of SILICON, HYBRID, EMPIRICAL, SOL, SOL_FULL, or ANALYTICAL; got {mode:?}"
            ),
        ));
    }
    match (
        args.aic_systems_path.as_deref(),
        args.aic_database_snapshot_sha256.as_deref(),
    ) {
        (Some(path), Some(digest)) => {
            if path.trim().is_empty() {
                return Err(mock_engine_args_validation_error(
                    "aic_systems_path_empty",
                    "aic_systems_path must be non-empty".to_string(),
                ));
            }
            if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(mock_engine_args_validation_error(
                    "aic_database_snapshot_sha256_invalid",
                    "aic_database_snapshot_sha256 must be a 64-character hexadecimal digest"
                        .to_string(),
                ));
            }
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err(mock_engine_args_validation_error(
                "aic_database_identity_incomplete",
                "aic_systems_path and aic_database_snapshot_sha256 must be configured together"
                    .to_string(),
            ));
        }
        (None, None) => {}
    }
    if let Some(policy) = args.aic_analytical_config.as_deref() {
        if args.aic_database_mode.as_deref() != Some("ANALYTICAL") {
            return Err(mock_engine_args_validation_error(
                "aic_analytical_config_without_mode",
                "aic_analytical_config requires aic_database_mode=ANALYTICAL".to_string(),
            ));
        }
        match serde_json::from_str::<serde_json::Value>(policy) {
            Ok(serde_json::Value::Object(_)) => {}
            Ok(_) => {
                return Err(mock_engine_args_validation_error(
                    "aic_analytical_config_not_object",
                    "aic_analytical_config must be a JSON object".to_string(),
                ));
            }
            Err(error) => {
                return Err(mock_engine_args_validation_error(
                    "aic_analytical_config_invalid_json",
                    format!("aic_analytical_config must be valid JSON: {error}"),
                ));
            }
        }
    }

    if args.block_size == 0 {
        return Err(mock_engine_args_validation_error(
            "block_size_zero",
            "block_size must be greater than 0".to_string(),
        ));
    }

    if matches!(args.engine_type, EngineType::Vllm | EngineType::Trtllm) && args.block_size < 2 {
        return Err(mock_engine_args_validation_error(
            "shared_scheduler_block_size_too_small",
            format!(
                "the vLLM/TRT-LLM scheduler requires block_size to be at least 2 for engine_type={:?}, got block_size={}",
                args.engine_type, args.block_size,
            ),
        ));
    }

    if args.g1_backend == Some(G1Backend::Native) && args.requires_kvbm_g1() {
        return Err(mock_engine_args_validation_error(
            "native_g1_legacy_offload_conflict",
            "g1_backend=native cannot be combined with KVBM G2/G3/G4 offload; omit g1_backend to select KVBM automatically or set g1_backend=kvbm explicitly"
                .to_string(),
        ));
    }

    if !matches!(
        args.g2_slot_mode.as_str(),
        "capped_dense" | "raw_derived_dense" | "sparse_byte_native"
    ) {
        return Err(mock_engine_args_validation_error(
            "invalid_g2_slot_mode",
            format!("unsupported g2_slot_mode={:?}", args.g2_slot_mode),
        ));
    }
    if !matches!(
        args.offline_kvbm_backend.as_str(),
        "reference" | "compact_g2_only" | "compact"
    ) {
        return Err(mock_engine_args_validation_error(
            "invalid_offline_kvbm_backend",
            format!(
                "unsupported offline_kvbm_backend={:?}",
                args.offline_kvbm_backend
            ),
        ));
    }
    if matches!(
        args.offline_kvbm_backend.as_str(),
        "compact_g2_only" | "compact"
    ) && args.g2_slot_mode != "sparse_byte_native"
    {
        return Err(mock_engine_args_validation_error(
            "compact_backend_requires_sparse_g2",
            "compact G2 backend requires g2_slot_mode=sparse_byte_native".to_string(),
        ));
    }

    if let Some(fabric_id) = args.g2_fabric_id.as_deref() {
        if fabric_id.trim().is_empty() {
            return Err(mock_engine_args_validation_error(
                "empty_g2_fabric_id",
                "g2_fabric_id must be a non-empty physical resource-domain id".to_string(),
            ));
        }
        if args.num_g2_blocks.is_none() && args.g2_capacity_bytes.is_none() {
            return Err(mock_engine_args_validation_error(
                "g2_fabric_without_g2",
                "g2_fabric_id requires an enabled G2 tier".to_string(),
            ));
        }
    }

    if let Some(configs) = args.g2_rank_resource_configs.as_ref() {
        if configs.is_empty() {
            return Err(mock_engine_args_validation_error(
                "empty_g2_rank_resource_configs",
                "g2_rank_resource_configs must contain one entry per offline scheduler rank"
                    .to_string(),
            ));
        }
        if args.g2_capacity_bytes.is_some() || args.g2_fabric_id.is_some() {
            return Err(mock_engine_args_validation_error(
                "mixed_role_and_rank_g2_resources",
                "g2_rank_resource_configs cannot be combined with role-wide g2_capacity_bytes or g2_fabric_id"
                    .to_string(),
            ));
        }
        let mut identities = std::collections::HashSet::with_capacity(configs.len());
        for config in configs {
            if config.g2_capacity_bytes == 0 {
                return Err(mock_engine_args_validation_error(
                    "zero_rank_g2_capacity",
                    format!(
                        "G2 rank resource ({}, {}) has zero capacity",
                        config.worker_id, config.dp_rank
                    ),
                ));
            }
            if config.g2_fabric_id.trim().is_empty() {
                return Err(mock_engine_args_validation_error(
                    "empty_rank_g2_fabric_id",
                    format!(
                        "G2 rank resource ({}, {}) has an empty fabric id",
                        config.worker_id, config.dp_rank
                    ),
                ));
            }
            for (name, value) in [
                ("bandwidth_g1_to_g2_gbps", config.bandwidth_g1_to_g2_gbps),
                ("bandwidth_g2_to_g1_gbps", config.bandwidth_g2_to_g1_gbps),
            ] {
                if !value.is_finite() || value <= 0.0 {
                    return Err(mock_engine_args_validation_error(
                        "invalid_rank_g2_bandwidth",
                        format!(
                            "G2 rank resource ({}, {}) requires positive finite {name}, got {value}",
                            config.worker_id, config.dp_rank
                        ),
                    ));
                }
            }
            if !identities.insert((config.worker_id, config.dp_rank)) {
                return Err(mock_engine_args_validation_error(
                    "duplicate_rank_g2_resource",
                    format!(
                        "duplicate G2 rank resource for ({}, {})",
                        config.worker_id, config.dp_rank
                    ),
                ));
            }
        }
    }

    if args.num_g3_blocks.is_some()
        && args.num_g2_blocks.is_none()
        && args.g2_capacity_bytes.is_none()
        && args.g2_rank_resource_configs.is_none()
    {
        return Err(mock_engine_args_validation_error(
            "g3_requires_g2",
            "num_g3_blocks requires num_g2_blocks because mocker stages G3 through G2".to_string(),
        ));
    }

    if args.max_model_len.is_some() && args.engine_type != EngineType::Vllm {
        return Err(mock_engine_args_validation_error(
            "max_model_len_requires_vllm",
            format!(
                "max_model_len is supported only for engine_type=vllm, got engine_type={:?}",
                args.engine_type
            ),
        ));
    }
    if args.enable_g4_storage
        && args.num_g2_blocks.is_none()
        && args.g2_capacity_bytes.is_none()
        && args.g2_rank_resource_configs.is_none()
    {
        return Err(mock_engine_args_validation_error(
            "g4_requires_g2",
            "enable_g4_storage requires num_g2_blocks because mocker stages G4 through G2"
                .to_string(),
        ));
    }
    if args.enable_g4_storage && args.g4_capacity_bytes.is_none() {
        return Err(mock_engine_args_validation_error(
            "g4_capacity_required",
            "enable_g4_storage requires finite g4_capacity_bytes".to_string(),
        ));
    }
    if !args.enable_g4_storage && args.g4_capacity_bytes.is_some() {
        return Err(mock_engine_args_validation_error(
            "g4_capacity_without_g4",
            "g4_capacity_bytes requires enable_g4_storage=true".to_string(),
        ));
    }
    match (
        args.kv_block_footprint_bytes.as_ref(),
        args.kv_footprint_source.as_ref(),
    ) {
        (Some(profile), Some(source)) => {
            if profile.is_empty() || profile.contains(&0) {
                return Err(mock_engine_args_validation_error(
                    "invalid_kv_block_footprint",
                    "kv_block_footprint_bytes must contain only positive byte sizes".to_string(),
                ));
            }
            if source.trim().is_empty() {
                return Err(mock_engine_args_validation_error(
                    "invalid_kv_footprint_source",
                    "kv_footprint_source must be non-empty".to_string(),
                ));
            }
            if let Some(max_model_len) = args.max_model_len {
                let required_blocks = max_model_len.div_ceil(args.block_size);
                if profile.len() < required_blocks {
                    return Err(mock_engine_args_validation_error(
                        "short_kv_block_footprint",
                        format!(
                            "kv_block_footprint_bytes has {} entries but max_model_len={} and block_size={} require at least {}",
                            profile.len(),
                            max_model_len,
                            args.block_size,
                            required_blocks
                        ),
                    ));
                }
            }
        }
        (Some(_), None) => {
            return Err(mock_engine_args_validation_error(
                "missing_kv_footprint_source",
                "kv_block_footprint_bytes requires kv_footprint_source".to_string(),
            ));
        }
        (None, Some(_)) => {
            return Err(mock_engine_args_validation_error(
                "kv_footprint_source_without_profile",
                "kv_footprint_source requires kv_block_footprint_bytes".to_string(),
            ));
        }
        (None, None) => {}
    }

    if args.aic_nextn.is_some() && args.decode_speedup_ratio != 1.0 {
        return Err(mock_engine_args_validation_error(
            "mtp_decode_speedup_conflict",
            format!(
                "aic_nextn requires decode_speedup_ratio=1.0 because MTP output acceleration is modeled by burst sampling, got {}",
                args.decode_speedup_ratio
            ),
        ));
    }

    if args.aic_nextn.is_none() && args.aic_nextn_accept_rates.is_some() {
        return Err(mock_engine_args_validation_error(
            "mtp_rates_without_nextn",
            "aic_nextn_accept_rates requires aic_nextn".to_string(),
        ));
    }

    if let Some(policy) = args
        .trtllm
        .as_ref()
        .and_then(|trtllm| trtllm.capacity_scheduler_policy.as_deref())
        && policy != "guaranteed_no_evict"
    {
        return Err(mock_engine_args_validation_error(
            "trtllm_unsupported_capacity_scheduler_policy",
            format!(
                "engine_type=trtllm v1 supports only capacity_scheduler_policy='guaranteed_no_evict', got '{policy}'",
            ),
        ));
    }

    if args.engine_type != EngineType::Sglang {
        return Ok(());
    }

    if let Some(page_size) = args.sglang.as_ref().and_then(|sglang| sglang.page_size)
        && args.block_size != page_size
    {
        return Err(mock_engine_args_validation_error(
            "sglang_block_size_page_size_mismatch",
            format!(
                "engine_type=sglang requires block_size and sglang.page_size to match when both are set, got block_size={} and sglang.page_size={page_size}",
                args.block_size,
            ),
        ));
    }

    if let Some(chunked_prefill_size) = args
        .sglang
        .as_ref()
        .and_then(|sglang| sglang.chunked_prefill_size)
        && chunked_prefill_size % args.block_size != 0
    {
        return Err(mock_engine_args_validation_error(
            "sglang_chunked_prefill_size_not_divisible_by_block_size",
            format!(
                "engine_type=sglang requires sglang.chunked_prefill_size to be divisible by block_size, got chunked_prefill_size={} and block_size={}",
                chunked_prefill_size, args.block_size,
            ),
        ));
    }

    Ok(())
}

impl TryFrom<MockEngineArgsSerde> for MockEngineArgs {
    type Error = String;

    fn try_from(compat: MockEngineArgsSerde) -> Result<Self, Self::Error> {
        let mut builder = Self::builder();

        if let Some(engine_type) = compat.engine_type.into_non_null("engine_type")? {
            builder = builder.engine_type(engine_type.parse()?);
        }
        if let Some(Some(num_gpu_blocks)) = compat.num_gpu_blocks.into_nullable() {
            builder = builder.num_gpu_blocks(num_gpu_blocks);
        }
        if let Some(block_size) = compat.block_size.into_non_null("block_size")? {
            builder = builder.block_size(block_size);
        }
        if let Some(max_model_len) = compat.max_model_len.into_nullable() {
            builder = builder.max_model_len(max_model_len);
        }
        if let Some(max_num_seqs) = compat.max_num_seqs.into_nullable() {
            builder = builder.max_num_seqs(max_num_seqs);
        }
        if let Some(max_num_batched_tokens) = compat.max_num_batched_tokens.into_nullable() {
            builder = builder.max_num_batched_tokens(max_num_batched_tokens);
        }
        if let Some(enable_prefix_caching) = compat
            .enable_prefix_caching
            .into_non_null("enable_prefix_caching")?
        {
            builder = builder.enable_prefix_caching(enable_prefix_caching);
        }
        if let Some(g1_backend) = compat.g1_backend.into_non_null("g1_backend")? {
            builder = builder.g1_backend(g1_backend);
        }
        if let Some(enable_chunked_prefill) = compat
            .enable_chunked_prefill
            .into_non_null("enable_chunked_prefill")?
        {
            builder = builder.enable_chunked_prefill(enable_chunked_prefill);
        }
        if let Some(speedup_ratio) = compat.speedup_ratio.into_non_null("speedup_ratio")? {
            builder = builder.speedup_ratio(speedup_ratio);
        }
        if let Some(decode_speedup_ratio) = compat
            .decode_speedup_ratio
            .into_non_null("decode_speedup_ratio")?
        {
            builder = builder.decode_speedup_ratio(decode_speedup_ratio);
        }
        if let Some(dp_size) = compat.dp_size.into_non_null("dp_size")? {
            builder = builder.dp_size(dp_size);
        }
        if let Some(startup_time) = compat.startup_time.into_nullable() {
            builder = builder.startup_time(startup_time);
        }

        let worker_type = if let Some(worker_type) =
            compat.worker_type.into_non_null("worker_type")?
        {
            worker_type.parse()?
        } else {
            let is_prefill = compat
                .is_prefill
                .into_non_null("is_prefill")?
                .unwrap_or(false);
            let is_decode = compat
                .is_decode
                .into_non_null("is_decode")?
                .unwrap_or(false);

            match (is_prefill, is_decode) {
                (false, false) => WorkerType::Aggregated,
                (true, false) => WorkerType::Prefill,
                (false, true) => WorkerType::Decode,
                (true, true) => {
                    return Err(
                        "Invalid worker configuration: is_prefill and is_decode cannot both be true."
                            .to_string(),
                    );
                }
            }
        };
        builder = builder.worker_type(worker_type);

        if let Some(planner_profile_data) = compat.planner_profile_data.into_nullable() {
            builder = builder.planner_profile_data(planner_profile_data.clone());
            if let Some(path) = planner_profile_data {
                builder = builder.perf_model(load_perf_model(&path));
            }
        }

        if let Some(aic_backend) = compat.aic_backend.into_nullable() {
            builder = builder.aic_backend(aic_backend);
        }
        if let Some(aic_system) = compat.aic_system.into_nullable() {
            builder = builder.aic_system(aic_system);
        }
        if let Some(aic_backend_version) = compat.aic_backend_version.into_nullable() {
            builder = builder.aic_backend_version(aic_backend_version);
        }
        if let Some(aic_database_mode) = compat.aic_database_mode.into_nullable() {
            builder = builder.aic_database_mode(aic_database_mode);
        }
        if let Some(aic_systems_path) = compat.aic_systems_path.into_nullable() {
            builder = builder.aic_systems_path(aic_systems_path);
        }
        if let Some(aic_database_snapshot_sha256) =
            compat.aic_database_snapshot_sha256.into_nullable()
        {
            builder = builder.aic_database_snapshot_sha256(aic_database_snapshot_sha256);
        }
        if let Some(aic_analytical_config) = compat.aic_analytical_config.into_nullable() {
            builder = builder.aic_analytical_config(aic_analytical_config);
        }
        if let Some(aic_tp_size) = compat.aic_tp_size.into_nullable() {
            builder = builder.aic_tp_size(aic_tp_size);
        }
        if let Some(aic_model_path) = compat.aic_model_path.into_nullable() {
            builder = builder.aic_model_path(aic_model_path);
        }
        if let Some(aic_moe_tp_size) = compat.aic_moe_tp_size.into_nullable() {
            builder = builder.aic_moe_tp_size(aic_moe_tp_size);
        }
        if let Some(aic_moe_ep_size) = compat.aic_moe_ep_size.into_nullable() {
            builder = builder.aic_moe_ep_size(aic_moe_ep_size);
        }
        if let Some(aic_attention_dp_size) = compat.aic_attention_dp_size.into_nullable() {
            builder = builder.aic_attention_dp_size(aic_attention_dp_size);
        }
        if let Some(aic_gemm_dtype) = compat.aic_gemm_dtype.into_nullable() {
            builder = builder.aic_gemm_dtype(aic_gemm_dtype);
        }
        if let Some(aic_moe_dtype) = compat.aic_moe_dtype.into_nullable() {
            builder = builder.aic_moe_dtype(aic_moe_dtype);
        }
        if let Some(aic_fmha_dtype) = compat.aic_fmha_dtype.into_nullable() {
            builder = builder.aic_fmha_dtype(aic_fmha_dtype);
        }
        if let Some(aic_kv_cache_dtype) = compat.aic_kv_cache_dtype.into_nullable() {
            builder = builder.aic_kv_cache_dtype(aic_kv_cache_dtype);
        }
        if let Some(aic_comm_dtype) = compat.aic_comm_dtype.into_nullable() {
            builder = builder.aic_comm_dtype(aic_comm_dtype);
        }
        if let Some(aic_nextn) = compat.aic_nextn.into_nullable() {
            builder = builder.aic_nextn(aic_nextn);
        }
        if let Some(aic_nextn_accept_rates) = compat.aic_nextn_accept_rates.into_nullable() {
            builder = builder.aic_nextn_accept_rates(aic_nextn_accept_rates);
        }
        if let Some(aic_mtp_seed) = compat.aic_mtp_seed.into_non_null("aic_mtp_seed")? {
            builder = builder.aic_mtp_seed(aic_mtp_seed);
        }
        if let Some(gpu_memory_utilization) = compat.gpu_memory_utilization.into_nullable() {
            builder = builder.gpu_memory_utilization(gpu_memory_utilization);
        }
        if let Some(mem_fraction_static) = compat.mem_fraction_static.into_nullable() {
            builder = builder.mem_fraction_static(mem_fraction_static);
        }
        if let Some(free_gpu_memory_fraction) = compat.free_gpu_memory_fraction.into_nullable() {
            builder = builder.free_gpu_memory_fraction(free_gpu_memory_fraction);
        }
        if let Some(enable_local_indexer) = compat
            .enable_local_indexer
            .into_non_null("enable_local_indexer")?
        {
            builder = builder.enable_local_indexer(enable_local_indexer);
        }
        if let Some(bootstrap_port) = compat.bootstrap_port.into_nullable() {
            builder = builder.bootstrap_port(bootstrap_port);
        }
        if let Some(timeout_ms) = compat
            .handoff_session_timeout_ms
            .into_non_null("handoff_session_timeout_ms")?
        {
            builder = builder.handoff_session_timeout_ms(timeout_ms);
        }
        if let Some(kv_bytes_per_token) = compat.kv_bytes_per_token.into_nullable() {
            builder = builder.kv_bytes_per_token(kv_bytes_per_token);
        }
        if let Some(kv_block_footprint_bytes) = compat.kv_block_footprint_bytes.into_nullable() {
            builder = builder.kv_block_footprint_bytes(kv_block_footprint_bytes);
        }
        if let Some(kv_footprint_source) = compat.kv_footprint_source.into_nullable() {
            builder = builder.kv_footprint_source(kv_footprint_source);
        }
        if let Some(kv_transfer_bandwidth) = compat.kv_transfer_bandwidth.into_nullable() {
            builder = builder.kv_transfer_bandwidth(kv_transfer_bandwidth);
        }
        if let Some(mode) = compat
            .kv_transfer_timing_mode
            .into_non_null("kv_transfer_timing_mode")?
        {
            builder = builder.kv_transfer_timing_mode(mode.parse()?);
        }
        if let Some(num_g2_blocks) = compat.num_g2_blocks.into_nullable() {
            builder = builder.num_g2_blocks(num_g2_blocks);
        }
        if let Some(g2_capacity_bytes) = compat.g2_capacity_bytes.into_nullable() {
            builder = builder.g2_capacity_bytes(g2_capacity_bytes);
        }
        if let Some(g2_slot_mode) = compat.g2_slot_mode.into_non_null("g2_slot_mode")? {
            builder = builder.g2_slot_mode(g2_slot_mode);
        }
        if let Some(offline_kvbm_backend) = compat
            .offline_kvbm_backend
            .into_non_null("offline_kvbm_backend")?
        {
            builder = builder.offline_kvbm_backend(offline_kvbm_backend);
        }
        if let Some(g2_fabric_id) = compat.g2_fabric_id.into_nullable() {
            builder = builder.g2_fabric_id(g2_fabric_id);
        }
        if let Some(g2_rank_resource_configs) = compat.g2_rank_resource_configs.into_nullable() {
            builder = builder.g2_rank_resource_configs(g2_rank_resource_configs);
        }
        if let Some(num_g3_blocks) = compat.num_g3_blocks.into_nullable() {
            builder = builder.num_g3_blocks(num_g3_blocks);
        }
        if let Some(enable_g4_storage) = compat
            .enable_g4_storage
            .into_non_null("enable_g4_storage")?
        {
            builder = builder.enable_g4_storage(enable_g4_storage);
        }
        if let Some(g4_capacity_bytes) = compat.g4_capacity_bytes.into_nullable() {
            builder = builder.g4_capacity_bytes(g4_capacity_bytes);
        }
        if let Some(offload_batch_size) = compat.offload_batch_size.into_nullable() {
            builder = builder.offload_batch_size(offload_batch_size);
        }
        if let Some(bandwidth_g1_to_g2_gbps) = compat.bandwidth_g1_to_g2_gbps.into_nullable() {
            builder = builder.bandwidth_g1_to_g2_gbps(bandwidth_g1_to_g2_gbps);
        }
        if let Some(bandwidth_g2_to_g1_gbps) = compat.bandwidth_g2_to_g1_gbps.into_nullable() {
            builder = builder.bandwidth_g2_to_g1_gbps(bandwidth_g2_to_g1_gbps);
        }
        if let Some(bandwidth_g2_to_g3_gbps) = compat.bandwidth_g2_to_g3_gbps.into_nullable() {
            builder = builder.bandwidth_g2_to_g3_gbps(bandwidth_g2_to_g3_gbps);
        }
        if let Some(bandwidth_g3_to_g2_gbps) = compat.bandwidth_g3_to_g2_gbps.into_nullable() {
            builder = builder.bandwidth_g3_to_g2_gbps(bandwidth_g3_to_g2_gbps);
        }
        if let Some(bandwidth_g2_to_g4_gbps) = compat.bandwidth_g2_to_g4_gbps.into_nullable() {
            builder = builder.bandwidth_g2_to_g4_gbps(bandwidth_g2_to_g4_gbps);
        }
        if let Some(bandwidth_g4_to_g2_gbps) = compat.bandwidth_g4_to_g2_gbps.into_nullable() {
            builder = builder.bandwidth_g4_to_g2_gbps(bandwidth_g4_to_g2_gbps);
        }
        if let Some(reasoning) = compat.reasoning.into_nullable() {
            builder = builder.reasoning(reasoning);
        }
        if let Some(response_replay_trace_path) = compat.response_replay_trace_path.into_nullable()
        {
            builder = builder.response_replay_trace_path(response_replay_trace_path);
        }
        if let Some(zmq_kv_events_port) = compat.zmq_kv_events_port.into_nullable() {
            builder = builder.zmq_kv_events_port(zmq_kv_events_port);
        }
        if let Some(zmq_replay_port) = compat.zmq_replay_port.into_nullable() {
            builder = builder.zmq_replay_port(zmq_replay_port);
        }
        if let Some(preemption_mode) = compat.preemption_mode.into_non_null("preemption_mode")? {
            builder = builder.preemption_mode(preemption_mode.parse()?);
        }
        if let Some(router_queue_policy) = compat.router_queue_policy.into_nullable() {
            let router_queue_policy = router_queue_policy
                .map(|policy| policy.parse().map_err(|e: String| e))
                .transpose()?;
            builder = builder.router_queue_policy(router_queue_policy);
        }
        if let Some(sglang) = compat.sglang.into_nullable() {
            builder = builder.sglang(sglang);
        }
        if let Some(trtllm) = compat.trtllm.into_nullable() {
            builder = builder.trtllm(trtllm);
        }

        builder
            .build()
            .map_err(|e| format!("Failed to build MockEngineArgs: {e}"))?
            .normalized()
            .map_err(|e| e.to_string())
    }
}

impl Default for MockEngineArgs {
    fn default() -> MockEngineArgs {
        MockEngineArgsBuilder::default()
            .build()
            .expect("Failed to build default MockEngineArgs")
            .normalized()
            .expect("Failed to normalize default MockEngineArgs")
    }
}

impl MockEngineArgs {
    const DEFAULT_VLLM_BLOCK_SIZE: usize = 64;
    const DEFAULT_SGLANG_BLOCK_SIZE: usize = 1;
    const DEFAULT_TRTLLM_BLOCK_SIZE: usize = 32;

    pub fn builder() -> MockEngineArgsBuilder {
        MockEngineArgsBuilder::default()
    }

    /// GPUs occupied by one worker (engine), derived from tensor parallelism
    /// and the materialized DP topology. AIC-backed replay uses
    /// `aic_tp_size × aic_attention_dp_size`; non-AIC replay still counts one
    /// GPU for every independently modeled `dp_size` rank. Used to turn
    /// provisioned worker-seconds into GPU-hours.
    pub fn aic_gpus_per_worker(&self) -> usize {
        self.aic_tp_size.unwrap_or(1) * self.dp_size.max(1) as usize
    }

    /// Finite ownership bound for live handoff queues and sessions.
    ///
    /// An unset runnable-sequence limit is semantically unbounded, so use the
    /// physical KV block count as the conservative process-local bound.
    pub fn effective_handoff_capacity(&self) -> usize {
        self.max_num_seqs.unwrap_or(self.num_gpu_blocks).max(1)
    }

    pub fn normalized(mut self) -> anyhow::Result<Self> {
        self.materialize_defaults();
        self.resolve_g1_backend();
        self.validate_config()?;
        Ok(self)
    }

    fn materialize_defaults(&mut self) {
        if let Some(mode) = self.aic_database_mode.as_mut() {
            *mode = mode.trim().to_uppercase();
        }
        if let Some(path) = self.aic_systems_path.as_mut() {
            *path = path.trim().to_string();
        }
        if let Some(digest) = self.aic_database_snapshot_sha256.as_mut() {
            *digest = digest.trim().to_lowercase();
        }
        if let Some(policy) = self.aic_analytical_config.as_mut()
            && let Ok(serde_json::Value::Object(object)) =
                serde_json::from_str::<serde_json::Value>(policy)
        {
            *policy = serde_json::to_string(&object)
                .expect("serializing a parsed analytical policy cannot fail");
        }

        match self.engine_type {
            EngineType::Vllm => {
                if self.block_size == 0 {
                    self.block_size = Self::DEFAULT_VLLM_BLOCK_SIZE;
                }
            }
            EngineType::Sglang => {
                let page_size = self.sglang.as_ref().and_then(|sglang| sglang.page_size);
                match (self.block_size, page_size) {
                    (0, None) => {
                        self.block_size = Self::DEFAULT_SGLANG_BLOCK_SIZE;
                    }
                    (0, Some(page_size)) => {
                        self.block_size = page_size;
                    }
                    (_, Some(_)) => {}
                    (_, None) => {}
                }
            }
            EngineType::Trtllm => {
                if self.block_size == 0 {
                    self.block_size = Self::DEFAULT_TRTLLM_BLOCK_SIZE;
                }
            }
        }

        if self.num_g2_blocks == Some(0) {
            self.num_g2_blocks = None;
        }
        if let Some(fabric_id) = self.g2_fabric_id.as_mut() {
            *fabric_id = fabric_id.trim().to_string();
        }
        self.g2_slot_mode = self.g2_slot_mode.trim().to_ascii_lowercase();
        self.offline_kvbm_backend = self.offline_kvbm_backend.trim().to_ascii_lowercase();
        if let Some(configs) = self.g2_rank_resource_configs.as_mut() {
            for config in configs {
                config.g2_fabric_id = config.g2_fabric_id.trim().to_string();
            }
        }
        if self.num_g3_blocks == Some(0) {
            self.num_g3_blocks = None;
        }
        if self.offload_batch_size == Some(0) {
            self.offload_batch_size = None;
        }
    }

    fn requires_kvbm_g1(&self) -> bool {
        matches!(self.engine_type, EngineType::Vllm | EngineType::Trtllm)
            && (self.num_g2_blocks.is_some_and(|blocks| blocks > 0)
                || self.g2_capacity_bytes.is_some()
                || self.g2_rank_resource_configs.is_some()
                || self.num_g3_blocks.is_some_and(|blocks| blocks > 0)
                || self.enable_g4_storage)
    }

    fn resolve_g1_backend(&mut self) {
        if self.g1_backend.is_none() {
            self.g1_backend = Some(if self.requires_kvbm_g1() {
                G1Backend::Kvbm
            } else {
                G1Backend::Native
            });
        }
    }

    /// Return the selected backend, resolving an unset raw configuration from
    /// its engine and lower-tier offload settings.
    pub fn resolved_g1_backend(&self) -> G1Backend {
        self.g1_backend.unwrap_or_else(|| {
            if self.requires_kvbm_g1() {
                G1Backend::Kvbm
            } else {
                G1Backend::Native
            }
        })
    }

    /// Materialize the exact G2 resources assigned to one offline scheduler
    /// rank. Role-wide legacy settings are returned unchanged when no rank map
    /// is configured.
    pub(crate) fn for_offline_rank(&self, worker_id: usize, dp_rank: u32) -> anyhow::Result<Self> {
        let Some(configs) = self.g2_rank_resource_configs.as_ref() else {
            return Ok(self.clone());
        };
        let config = configs
            .iter()
            .find(|config| config.worker_id == worker_id && config.dp_rank == dp_rank)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "missing G2 rank resource for offline scheduler ({worker_id}, {dp_rank})"
                )
            })?;
        let mut args = self.clone();
        args.g2_rank_resource_configs = None;
        args.g2_capacity_bytes = Some(config.g2_capacity_bytes);
        args.g2_fabric_id = Some(config.g2_fabric_id.clone());
        args.bandwidth_g1_to_g2_gbps = Some(config.bandwidth_g1_to_g2_gbps);
        args.bandwidth_g2_to_g1_gbps = Some(config.bandwidth_g2_to_g1_gbps);
        let block_bytes = if let Some(profile) = args.kv_block_footprint_bytes.as_ref() {
            profile.iter().copied().min()
        } else {
            args.kv_bytes_per_token
                .and_then(|bytes| args.block_size.checked_mul(bytes))
        }
        .ok_or_else(|| {
            anyhow::anyhow!(
                "G2 rank resource ({worker_id}, {dp_rank}) requires a resolved KV block footprint"
            )
        })?;
        let raw_slots = config.g2_capacity_bytes / block_bytes;
        if raw_slots == 0 {
            anyhow::bail!(
                "G2 rank resource ({worker_id}, {dp_rank}) capacity {} cannot hold one {block_bytes}-byte block",
                config.g2_capacity_bytes
            );
        }
        #[cfg(feature = "kvbm-offload")]
        let max_slots = crate::kvbm_offload::config::DEFAULT_MAX_DERIVED_G2_SLOTS;
        #[cfg(not(feature = "kvbm-offload"))]
        let max_slots = 2_000_000;
        let slots = match args.g2_slot_mode.as_str() {
            "capped_dense" => raw_slots.min(max_slots),
            "raw_derived_dense" | "sparse_byte_native" => raw_slots,
            other => anyhow::bail!("unsupported g2_slot_mode={other:?}"),
        };
        args.num_g2_blocks = Some(slots);
        args.normalized()
    }

    fn validate_config(&mut self) -> anyhow::Result<()> {
        self.validate()
            .map_err(|error| anyhow::anyhow!("Failed to validate MockEngineArgs: {error}"))?;
        if let Some(nextn) = self.aic_nextn {
            let rates = crate::common::speculative::normalize_conditional_accept_rates(
                nextn,
                self.aic_nextn_accept_rates.as_deref(),
            )?;
            self.aic_nextn_accept_rates =
                Some(crate::common::speculative::format_accept_rates(&rates));
        }
        Ok(())
    }

    /// Scheduling policy applied by the shared vLLM scheduler core, derived
    /// from the engine type. TRT-LLM uses `GUARANTEED_NO_EVICT`.
    pub fn scheduling_policy(&self) -> SchedulingPolicy {
        match self.engine_type {
            EngineType::Trtllm => SchedulingPolicy::TrtllmGuaranteedNoEvict,
            EngineType::Vllm | EngineType::Sglang => SchedulingPolicy::Vllm,
        }
    }

    pub fn is_prefill(&self) -> bool {
        self.worker_type == WorkerType::Prefill
    }

    pub fn is_decode(&self) -> bool {
        self.worker_type == WorkerType::Decode
    }

    pub fn needs_kv_publisher(&self) -> bool {
        self.enable_prefix_caching && !self.is_decode()
    }

    pub fn undiscounted_aic_accept_rates(&self) -> Option<String> {
        crate::common::speculative::undiscounted_aic_accept_rates(self.aic_nextn)
    }

    /// Create MockEngineArgs from a JSON file containing extra engine arguments
    pub fn from_json_file(path: &Path) -> anyhow::Result<Self> {
        let file_content = std::fs::read_to_string(path)?;
        Self::from_json_str(&file_content)
    }

    pub fn from_json_str(content: &str) -> anyhow::Result<Self> {
        let mut deserializer = serde_json::Deserializer::from_str(content);
        let args = serde_path_to_error::deserialize(&mut deserializer)
            .map_err(|error| anyhow::anyhow!("{error}"))?;
        deserializer
            .end()
            .map_err(|error| anyhow::anyhow!("{error}"))?;
        Ok(args)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use serde_json::json;

    #[derive(Default)]
    struct FailingRawSink {
        attempts: Mutex<Vec<u64>>,
    }

    impl RawKvEventSink for FailingRawSink {
        fn publish(&self, event: RawKvEvent) -> anyhow::Result<()> {
            self.attempts.lock().unwrap().push(event.event.event_id);
            if event.event.event_id == 2 {
                anyhow::bail!("injected raw sink failure");
            }
            Ok(())
        }
    }

    #[test]
    fn raw_sink_batch_fallback_attempts_later_events_after_failure() {
        let sink = FailingRawSink::default();
        let error = sink
            .publish_batch(
                (1..=3)
                    .map(|event_id| RawKvEvent {
                        event: KvCacheEvent {
                            event_id,
                            data: dynamo_kv_router::protocols::KvCacheEventData::Cleared,
                            dp_rank: 0,
                        },
                        block_token_ids: None,
                        storage_tier: StorageTier::Device,
                    })
                    .collect(),
            )
            .unwrap_err();

        assert_eq!(error.to_string(), "injected raw sink failure");
        assert_eq!(*sink.attempts.lock().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn direct_request_output_plan_is_authoritative_when_limited() {
        let request = DirectRequest {
            tokens: vec![1, 2],
            max_output_tokens: 0,
            output_token_ids: Some(vec![7, 8]),
            ..Default::default()
        };

        assert_eq!(request.effective_max_output_tokens(), 2);
        let limited = request.clone_with_output_limit(1);
        assert_eq!(limited.max_output_tokens, 1);
        assert_eq!(limited.output_token_ids.as_deref(), Some(&[7][..]));
        assert_eq!(limited.output_token_ids.unwrap().capacity(), 1);
        assert_eq!(request.output_token_ids.as_deref(), Some(&[7, 8][..]));

        let empty_plan = DirectRequest {
            max_output_tokens: 4,
            output_token_ids: Some(Vec::new()),
            ..Default::default()
        };
        assert_eq!(empty_plan.effective_max_output_tokens(), 0);
        let limited = empty_plan.clone_with_output_limit(1);
        assert_eq!(limited.max_output_tokens, 0);
        assert_eq!(limited.output_token_ids.as_deref(), Some(&[][..]));

        let unplanned = DirectRequest::default();
        assert_eq!(unplanned.effective_max_output_tokens(), 0);
        let limited = unplanned.clone_with_output_limit(1);
        assert_eq!(limited.max_output_tokens, 0);
        assert!(limited.output_token_ids.is_none());
    }

    #[test]
    fn direct_request_priorities_are_backward_compatible() {
        let legacy = json!({
            "tokens": [1, 2],
            "max_output_tokens": 3,
            "uuid": null,
            "dp_rank": 0,
            "arrival_timestamp_ms": null
        });
        let request: DirectRequest = serde_json::from_value(legacy).unwrap();
        assert_eq!(request.priority, 0);
        assert_eq!(request.strict_priority, 0);
        assert_eq!(request.router_priorities(), (0.0, 0));

        let rendered = serde_json::to_value(&request).unwrap();
        assert!(rendered.get("priority").is_none());
        assert!(rendered.get("strict_priority").is_none());
    }

    #[test]
    fn direct_request_derives_router_priorities() {
        let negative: DirectRequest = serde_json::from_value(json!({
            "tokens": [1],
            "max_output_tokens": 1,
            "uuid": null,
            "dp_rank": 0,
            "arrival_timestamp_ms": null,
            "priority": -7,
            "strict_priority": 4
        }))
        .unwrap();
        assert_eq!(negative.router_priorities(), (0.0, 4));

        let positive = DirectRequest {
            priority: 9,
            strict_priority: 5,
            ..negative
        };
        assert_eq!(positive.router_priorities(), (9.0, 5));
        let rendered = serde_json::to_value(&positive).unwrap();
        assert_eq!(rendered["priority"], 9);
        assert_eq!(rendered["strict_priority"], 5);
    }

    #[test]
    fn test_mock_engine_args_json_round_trip_preserves_worker_type_and_nulls() {
        let args = MockEngineArgs::builder()
            .worker_type(WorkerType::Decode)
            .g1_backend(G1Backend::Native)
            .max_model_len(Some(32768))
            .max_num_seqs(None)
            .max_num_batched_tokens(None)
            .reasoning(None)
            .sglang(None)
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        // Exercise the real serializer rather than mirroring every field in a
        // giant `json!` invocation. Besides reducing macro recursion, this
        // keeps the round-trip test automatically aligned with new resource
        // fields such as dynamic KV footprints and finite tier capacities.
        let payload = serde_json::to_value(&args).unwrap();

        let restored = MockEngineArgs::from_json_str(&payload.to_string()).unwrap();

        assert_eq!(restored.worker_type, WorkerType::Decode);
        assert_eq!(restored.max_model_len, Some(32768));
        assert_eq!(restored.max_num_seqs, None);
        assert_eq!(restored.max_num_batched_tokens, None);
        assert_eq!(restored.g1_backend, Some(G1Backend::Native));
        assert_eq!(
            restored.kv_transfer_timing_mode,
            KvTransferTimingMode::FullPrompt
        );
    }

    #[test]
    fn analytical_contract_normalizes_mode_and_policy_json() {
        let args = MockEngineArgs::builder()
            .aic_database_mode(Some(" analytical ".to_string()))
            .aic_analytical_config(Some(
                r#"{"level":"high","attention_algorithm":"fa3"}"#.to_string(),
            ))
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        assert_eq!(args.aic_database_mode.as_deref(), Some("ANALYTICAL"));
        assert_eq!(
            args.aic_analytical_config.as_deref(),
            Some(r#"{"attention_algorithm":"fa3","level":"high"}"#)
        );
    }

    #[test]
    fn aic_database_identity_normalizes_and_requires_a_complete_pair() {
        let args = MockEngineArgs::builder()
            .aic_systems_path(Some(" /tmp/aic-systems ".to_string()))
            .aic_database_snapshot_sha256(Some("A".repeat(64)))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        assert_eq!(args.aic_systems_path.as_deref(), Some("/tmp/aic-systems"));
        assert_eq!(
            args.aic_database_snapshot_sha256.as_deref(),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );

        let error = MockEngineArgs::builder()
            .aic_systems_path(Some("/tmp/aic-systems".to_string()))
            .build()
            .unwrap()
            .normalized()
            .unwrap_err();
        assert!(error.to_string().contains("must be configured together"));
    }

    #[test]
    fn analytical_contract_fails_closed_on_invalid_policy() {
        let without_mode = MockEngineArgs::builder()
            .aic_analytical_config(Some("{}".to_string()))
            .build()
            .unwrap()
            .normalized()
            .unwrap_err();
        assert!(
            without_mode
                .to_string()
                .contains("aic_analytical_config requires aic_database_mode=ANALYTICAL")
        );

        let non_object = MockEngineArgs::builder()
            .aic_database_mode(Some("ANALYTICAL".to_string()))
            .aic_analytical_config(Some("[]".to_string()))
            .build()
            .unwrap()
            .normalized()
            .unwrap_err();
        assert!(
            non_object
                .to_string()
                .contains("aic_analytical_config must be a JSON object")
        );
    }

    #[test]
    fn test_mock_engine_args_json_omits_unset_g1_backend() {
        let args = MockEngineArgs::builder().build().unwrap();
        let serialized = serde_json::to_value(args).unwrap();

        assert!(serialized.get("g1_backend").is_none());
    }

    #[test]
    fn test_mock_engine_args_accepts_legacy_enum_case_and_writes_lowercase() {
        let args = MockEngineArgs::from_json_str(
            &json!({
                "engine_type": "VLLM",
                "worker_type": "Aggregated",
                "preemption_mode": "Lifo",
                "num_g2_blocks": 8,
            })
            .to_string(),
        )
        .unwrap();

        assert_eq!(args.g1_backend, Some(G1Backend::Kvbm));
        let serialized = serde_json::to_value(args).unwrap();
        assert_eq!(serialized["engine_type"], "vllm");
        assert_eq!(serialized["worker_type"], "aggregated");
        assert_eq!(serialized["preemption_mode"], "lifo");
    }

    #[test]
    fn test_mock_engine_args_json_accepts_aic_quant_dtypes() {
        let args = MockEngineArgs::from_json_str(
            &json!({
                "aic_gemm_dtype": "fp8_block",
                "aic_moe_dtype": "w4a16_mxfp4",
                "aic_fmha_dtype": "bfloat16",
                "aic_kv_cache_dtype": "fp8",
                "aic_comm_dtype": "fp8",
            })
            .to_string(),
        )
        .unwrap();

        assert_eq!(args.aic_gemm_dtype.as_deref(), Some("fp8_block"));
        assert_eq!(args.aic_moe_dtype.as_deref(), Some("w4a16_mxfp4"));
        assert_eq!(args.aic_fmha_dtype.as_deref(), Some("bfloat16"));
        assert_eq!(args.aic_kv_cache_dtype.as_deref(), Some("fp8"));
        assert_eq!(args.aic_comm_dtype.as_deref(), Some("fp8"));
    }

    #[test]
    fn test_mock_engine_args_json_rejects_unknown_and_invalid_types() {
        let unknown = MockEngineArgs::from_json_str(&json!({"unknown": true}).to_string())
            .expect_err("unknown fields should be rejected");
        assert!(
            unknown.to_string().contains("unknown field"),
            "unexpected error: {unknown}",
        );

        let invalid =
            MockEngineArgs::from_json_str(&json!({"gpu_memory_utilization": "bad"}).to_string())
                .expect_err("wrongly typed fields should be rejected");
        assert!(
            invalid.to_string().contains("gpu_memory_utilization"),
            "unexpected error: {invalid}",
        );

        let trailing = MockEngineArgs::from_json_str(r#"{"block_size": 16} true"#)
            .expect_err("trailing JSON should be rejected");
        assert!(
            trailing.to_string().contains("trailing characters"),
            "unexpected error: {trailing}",
        );
    }

    #[test]
    fn test_unique_block_default_uniqueness() {
        // Create 10 default UniqueBlock instances
        let blocks: Vec<UniqueBlock> = (0..10).map(|_| UniqueBlock::default()).collect();

        // Extract UUIDs from each block
        let mut uuids = Vec::new();
        for block in blocks {
            match block {
                UniqueBlock::PartialBlock(uuid) => uuids.push(uuid),
                _ => panic!("Expected UuidIdentifier variant"),
            }
        }

        // Check that all UUIDs are unique by comparing each with every other
        for i in 0..uuids.len() {
            for j in i + 1..uuids.len() {
                assert_ne!(
                    uuids[i], uuids[j],
                    "UUID at index {} and {} are identical",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_normalized_sglang_uses_page_size_alias_for_block_size() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .sglang(Some(SglangArgs {
                page_size: Some(16),
                ..Default::default()
            }))
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        assert_eq!(args.block_size, 16);
    }

    #[test]
    fn test_normalized_sglang_accepts_equal_block_size_and_page_size() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(8)
            .sglang(Some(SglangArgs {
                page_size: Some(8),
                ..Default::default()
            }))
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        assert_eq!(args.block_size, 8);
    }

    #[test]
    fn test_normalized_sglang_rejects_mismatched_block_size_and_page_size() {
        let error = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(8)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                ..Default::default()
            }))
            .build()
            .unwrap()
            .normalized()
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("block_size and sglang.page_size to match"),
            "unexpected error: {error}",
        );
    }

    #[test]
    fn test_normalized_g3_requires_g2() {
        let missing_g2 = MockEngineArgs::builder()
            .num_g3_blocks(Some(10))
            .kv_bytes_per_token(Some(1024))
            .build()
            .unwrap()
            .normalized()
            .unwrap_err();
        assert!(
            missing_g2.to_string().contains("requires num_g2_blocks"),
            "unexpected error: {missing_g2}",
        );
    }

    #[test]
    fn test_normalized_g4_requires_g2() {
        let missing_g2 = MockEngineArgs::builder()
            .enable_g4_storage(true)
            .kv_bytes_per_token(Some(1024))
            .build()
            .unwrap()
            .normalized()
            .unwrap_err();
        assert!(
            missing_g2.to_string().contains("requires num_g2_blocks"),
            "unexpected error: {missing_g2}",
        );
    }

    #[test]
    fn test_native_g1_accepts_both_shared_scheduler_engines() {
        for engine_type in [EngineType::Vllm, EngineType::Trtllm] {
            let args = MockEngineArgs::builder()
                .engine_type(engine_type)
                .g1_backend(G1Backend::Native)
                .build()
                .unwrap()
                .normalized()
                .unwrap_or_else(|error| {
                    panic!("native G1 should support {engine_type:?}: {error}")
                });
            assert_eq!(args.g1_backend, Some(G1Backend::Native));
        }
    }

    #[test]
    fn test_g1_backend_defaults_to_native() {
        let default_args = MockEngineArgs::default();
        assert_eq!(default_args.g1_backend, Some(G1Backend::Native));

        let json_args = MockEngineArgs::from_json_str("{}").unwrap();
        assert_eq!(json_args.g1_backend, Some(G1Backend::Native));
    }

    #[test]
    fn test_legacy_kvbm_offload_selects_kvbm_g1() {
        let configs = [
            MockEngineArgs::builder()
                .num_g2_blocks(Some(8))
                .build()
                .unwrap(),
            MockEngineArgs::builder()
                .num_g2_blocks(Some(8))
                .num_g3_blocks(Some(16))
                .build()
                .unwrap(),
            MockEngineArgs::builder()
                .num_g2_blocks(Some(8))
                .enable_g4_storage(true)
                .g4_capacity_bytes(Some(8 * 1024 * 1024))
                .build()
                .unwrap(),
        ];

        for config in configs {
            assert_eq!(config.resolved_g1_backend(), G1Backend::Kvbm);
            let args = config.normalized().unwrap();
            assert_eq!(args.g1_backend, Some(G1Backend::Kvbm));
        }
    }

    #[test]
    fn test_explicit_native_g1_with_offload_is_rejected() {
        for engine_type in [EngineType::Vllm, EngineType::Trtllm] {
            let error = MockEngineArgs::builder()
                .engine_type(engine_type)
                .g1_backend(G1Backend::Native)
                .num_g2_blocks(Some(8))
                .build()
                .unwrap()
                .normalized()
                .unwrap_err();

            assert!(
                error.to_string().contains("omit g1_backend"),
                "unexpected error for {engine_type:?}: {error}"
            );
        }
    }

    #[test]
    fn test_explicit_native_g1_accepts_disabled_offload() {
        let args = MockEngineArgs::builder()
            .g1_backend(G1Backend::Native)
            .num_g2_blocks(Some(0))
            .num_g3_blocks(Some(0))
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        assert_eq!(args.g1_backend, Some(G1Backend::Native));
        assert_eq!(args.num_g2_blocks, None);
        assert_eq!(args.num_g3_blocks, None);
    }

    #[test]
    fn test_g1_backend_is_ignored_for_sglang() {
        for g1_backend in [G1Backend::Kvbm, G1Backend::Native] {
            let args = MockEngineArgs::builder()
                .engine_type(EngineType::Sglang)
                .g1_backend(g1_backend)
                .build()
                .unwrap()
                .normalized()
                .unwrap();
            assert_eq!(args.engine_type, EngineType::Sglang);
            assert_eq!(args.g1_backend, Some(g1_backend));
        }
    }

    #[test]
    fn test_native_g1_accepts_mtp() {
        for engine_type in [EngineType::Vllm, EngineType::Trtllm] {
            let args = MockEngineArgs::builder()
                .engine_type(engine_type)
                .g1_backend(G1Backend::Native)
                .aic_nextn(Some(1))
                .build()
                .unwrap()
                .normalized()
                .unwrap_or_else(|error| {
                    panic!("native G1 MTP should support {engine_type:?}: {error}")
                });
            assert_eq!(args.g1_backend, Some(G1Backend::Native));
            assert_eq!(args.aic_nextn, Some(1));
        }
    }

    #[test]
    fn test_normalized_rejects_out_of_range_aic_nextn() {
        // The mocker/replay JSON path must share AicPerfConfig's 1..=5 contract.
        for bad in [0_usize, 6, usize::MAX] {
            let err = MockEngineArgs::builder()
                .aic_nextn(Some(bad))
                .build()
                .unwrap()
                .normalized()
                .unwrap_err();
            assert!(
                err.to_string().contains("aic_nextn"),
                "unexpected error for nextn={bad}: {err}",
            );
        }
        MockEngineArgs::builder()
            .aic_nextn(Some(3))
            .build()
            .unwrap()
            .normalized()
            .expect("in-range aic_nextn should validate");
    }

    #[test]
    fn test_normalized_rejects_zero_max_model_len() {
        let error = MockEngineArgs::builder()
            .max_model_len(Some(0))
            .build()
            .unwrap()
            .normalized()
            .unwrap_err();

        assert!(
            error.to_string().contains("max_model_len"),
            "unexpected error: {error}",
        );
    }

    #[test]
    fn test_mtp_defaults_and_json_round_trip() {
        let args = MockEngineArgs::builder()
            .aic_nextn(Some(3))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        assert_eq!(args.aic_nextn_accept_rates.as_deref(), Some("0.85,0.3,0"));
        assert_eq!(args.aic_mtp_seed, 42);
        assert_eq!(
            args.undiscounted_aic_accept_rates().as_deref(),
            Some("0,0,0")
        );

        let json = serde_json::to_string(&args).unwrap();
        let round_trip = MockEngineArgs::from_json_str(&json).unwrap();
        assert_eq!(round_trip.aic_nextn, Some(3));
        assert_eq!(
            round_trip.aic_nextn_accept_rates.as_deref(),
            Some("0.85,0.3,0")
        );
        assert_eq!(round_trip.aic_mtp_seed, 42);
    }

    #[test]
    fn test_mtp_rates_are_validated_before_normalization() {
        for rates in ["nan", "inf", "-0.1", "1.1", "bad"] {
            let err = MockEngineArgs::builder()
                .aic_nextn(Some(1))
                .aic_nextn_accept_rates(Some(rates.to_string()))
                .build()
                .unwrap()
                .normalized()
                .unwrap_err();
            assert!(
                err.to_string().contains("aic_nextn_accept_rates"),
                "unexpected error for rates={rates:?}: {err}"
            );
        }
    }

    #[test]
    fn test_mtp_rates_are_padded_and_truncated_to_nextn() {
        let padded = MockEngineArgs::builder()
            .aic_nextn(Some(3))
            .aic_nextn_accept_rates(Some("1,0.5".to_string()))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        assert_eq!(padded.aic_nextn_accept_rates.as_deref(), Some("1,0.5,0"));

        let truncated = MockEngineArgs::builder()
            .aic_nextn(Some(2))
            .aic_nextn_accept_rates(Some("1,0.5,0.25".to_string()))
            .build()
            .unwrap()
            .normalized()
            .unwrap();
        assert_eq!(truncated.aic_nextn_accept_rates.as_deref(), Some("1,0.5"));
    }

    #[test]
    fn test_mtp_rejects_decode_speedup_ratio() {
        let err = MockEngineArgs::builder()
            .aic_nextn(Some(1))
            .decode_speedup_ratio(2.0)
            .build()
            .unwrap()
            .normalized()
            .unwrap_err();
        assert!(err.to_string().contains("decode_speedup_ratio=1.0"));
    }

    #[test]
    fn test_normalized_zero_disables_optional_offload_knobs() {
        let args = MockEngineArgs::builder()
            .num_g2_blocks(Some(0))
            .num_g3_blocks(Some(0))
            .offload_batch_size(Some(0))
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        assert_eq!(args.num_g2_blocks, None);
        assert_eq!(args.num_g3_blocks, None);
        assert!(!args.enable_g4_storage);
        assert_eq!(args.offload_batch_size, None);
    }

    #[test]
    fn test_normalized_zero_g3_does_not_require_g2_or_kv_bytes() {
        let args = MockEngineArgs::builder()
            .num_g3_blocks(Some(0))
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        assert_eq!(args.num_g3_blocks, None);
    }

    #[test]
    fn test_normalized_g3_allows_missing_kv_bytes_for_cli_auto_compute() {
        let args = MockEngineArgs::builder()
            .num_g2_blocks(Some(10))
            .num_g3_blocks(Some(10))
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        assert_eq!(args.num_g2_blocks, Some(10));
        assert_eq!(args.num_g3_blocks, Some(10));
        assert_eq!(args.kv_bytes_per_token, None);
    }

    #[test]
    fn test_normalized_g4_allows_missing_kv_bytes_for_cli_auto_compute() {
        let args = MockEngineArgs::builder()
            .num_g2_blocks(Some(10))
            .enable_g4_storage(true)
            .g4_capacity_bytes(Some(10 * 1024 * 1024))
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        assert_eq!(args.num_g2_blocks, Some(10));
        assert!(args.enable_g4_storage);
        assert_eq!(args.kv_bytes_per_token, None);
    }

    #[test]
    fn test_rank_scoped_g2_resources_materialize_exact_offline_rank() {
        let args = MockEngineArgs::from_json_str(
            &json!({
                "engine_type": "vllm",
                "block_size": 16,
                "kv_block_footprint_bytes": [1024, 2048],
                "kv_footprint_source": "aic:test",
                "g2_rank_resource_configs": [
                    {
                        "worker_id": 0,
                        "dp_rank": 0,
                        "g2_capacity_bytes": 4096,
                        "g2_fabric_id": "node-0",
                        "bandwidth_g1_to_g2_gbps": 64.0,
                        "bandwidth_g2_to_g1_gbps": 32.0
                    },
                    {
                        "worker_id": 0,
                        "dp_rank": 1,
                        "g2_capacity_bytes": 8192,
                        "g2_fabric_id": "node-1",
                        "bandwidth_g1_to_g2_gbps": 128.0,
                        "bandwidth_g2_to_g1_gbps": 96.0
                    }
                ]
            })
            .to_string(),
        )
        .unwrap();

        let rank = args.for_offline_rank(0, 1).unwrap();
        assert_eq!(rank.g2_capacity_bytes, Some(8192));
        assert_eq!(rank.num_g2_blocks, Some(8));
        assert_eq!(rank.g2_fabric_id.as_deref(), Some("node-1"));
        assert_eq!(rank.bandwidth_g1_to_g2_gbps, Some(128.0));
        assert_eq!(rank.bandwidth_g2_to_g1_gbps, Some(96.0));
        assert!(rank.g2_rank_resource_configs.is_none());
        assert!(args.for_offline_rank(1, 0).is_err());
    }

    #[test]
    fn test_rank_scoped_g2_resources_reject_role_wide_capacity() {
        let error = MockEngineArgs::from_json_str(
            &json!({
                "g2_capacity_bytes": 4096,
                "g2_rank_resource_configs": [{
                    "worker_id": 0,
                    "dp_rank": 0,
                    "g2_capacity_bytes": 4096,
                    "g2_fabric_id": "node-0",
                    "bandwidth_g1_to_g2_gbps": 64.0,
                    "bandwidth_g2_to_g1_gbps": 64.0
                }]
            })
            .to_string(),
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("cannot be combined with role-wide g2_capacity_bytes")
        );
    }

    #[test]
    fn test_normalized_sglang_defaults_block_size_to_one() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        assert_eq!(args.block_size, 1);
    }

    #[test]
    fn test_normalized_shared_scheduler_rejects_block_size_one_for_every_backend() {
        for engine_type in [EngineType::Vllm, EngineType::Trtllm] {
            let args_for = |g1_backend| {
                MockEngineArgs::builder()
                    .engine_type(engine_type)
                    .g1_backend(g1_backend)
                    .block_size(1)
                    .build()
                    .unwrap()
            };
            let explicit_native = args_for(G1Backend::Native);
            let explicit_kvbm = args_for(G1Backend::Kvbm);
            let mut unset = explicit_native.clone();
            unset.g1_backend = None;
            let mut automatic_kvbm = unset.clone();
            automatic_kvbm.num_g2_blocks = Some(1);

            for args in [explicit_native, explicit_kvbm, unset, automatic_kvbm] {
                let error = args.normalized().unwrap_err();
                let message = error.to_string();
                assert!(
                    message.contains(
                        "the vLLM/TRT-LLM scheduler requires block_size to be at least 2"
                    ),
                    "engine_type={engine_type:?}, error={error:#}"
                );
            }
        }
    }

    #[test]
    fn test_normalized_sglang_accepts_block_size_one() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(1)
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        assert_eq!(args.block_size, 1);
    }

    #[test]
    fn test_from_json_file_normalizes_sglang_page_size() {
        let tempdir = tempfile::tempdir().unwrap();
        let path = tempdir.path().join("args.json");
        std::fs::write(
            &path,
            serde_json::to_string(&json!({
                "engine_type": "sglang",
                "sglang": {
                    "page_size": 32
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let args = MockEngineArgs::from_json_file(&path).unwrap();
        assert_eq!(args.block_size, 32);
    }

    #[test]
    fn test_normalized_sglang_rejects_chunked_prefill_not_divisible_by_block_size() {
        let error = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(6),
                ..Default::default()
            }))
            .build()
            .unwrap()
            .normalized()
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("chunked_prefill_size to be divisible by block_size"),
            "unexpected error: {error}",
        );
    }

    #[test]
    fn test_normalized_sglang_accepts_chunked_prefill_divisible_by_block_size() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(8),
                ..Default::default()
            }))
            .build()
            .unwrap()
            .normalized()
            .unwrap();

        assert_eq!(args.block_size, 4);
    }
}
