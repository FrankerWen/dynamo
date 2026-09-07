// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact AgentX session-tree replay corpus and closed-loop controller.
//!
//! The controller is deliberately independent of aggregated/disaggregated
//! engine plumbing. Both offline runtimes consume it through the same
//! `AdmissionQueue`, so tree admission, recycling, guards, and phase semantics
//! cannot drift between topologies.

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;
use uuid::Uuid;

use super::trace::validate_synthesizable_prompt;
use super::{
    ReplayRequestHashes, ReplayRequestPayload, SYNTHETIC_OUTPUT_SEED, planned_output_token_ids,
};
use crate::common::protocols::DirectRequest;

pub const AGENTX_EXACT_SCHEMA_VERSION: &str = "agentx_exact_v1";

fn default_start_min_ratio() -> f64 {
    0.25
}

fn default_start_max_ratio() -> f64 {
    0.75
}

fn default_seed() -> u64 {
    42
}

fn default_primer_max_tokens() -> usize {
    1
}

fn default_warmup_requests_per_lane() -> usize {
    10
}

fn default_profile_duration_s() -> f64 {
    3600.0
}

fn default_tree_idle_gap_cap_s() -> f64 {
    300.0
}

fn default_global_idle_gap_cap_s() -> f64 {
    10.0
}

fn default_drain_grace_s() -> f64 {
    600.0
}

fn default_benchmark_id() -> String {
    "agentx".to_string()
}

fn default_marker_token_count() -> usize {
    1
}

fn default_marker_tokenizer_fingerprint() -> String {
    "test-only-single-token-marker".to_string()
}

/// Strict line-level envelope for `agentx_exact_v1` JSONL. Each non-empty
/// line is one complete source session tree.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AgentxExactTreeRecord {
    pub schema_version: String,
    pub tree_id: String,
    pub source_workflow_id: String,
    pub source_index: usize,
    pub hash_namespace: String,
    pub hash_id_scope: String,
    pub trace_hash_block_size: usize,
    pub models: Vec<String>,
    pub source_start_s: f64,
    pub source_last_request_start_s: f64,
    pub source_end_s: f64,
    pub source_duration_s: f64,
    pub root_stream_id: String,
    pub streams: Vec<AgentxStream>,
    pub branches: Vec<AgentxBranch>,
    pub tree_terminal_request_ids: Vec<String>,
    pub compile_evidence: AgentxCompileEvidence,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AgentxStream {
    pub stream_id: String,
    pub role: String,
    pub parent_stream_id: Option<String>,
    pub agent_depth: usize,
    pub source_agent_id: Option<String>,
    pub source_agent_path: Vec<String>,
    pub detected_chain_index: usize,
    pub fork_prefix_blocks: Option<usize>,
    pub worker_group: Option<Vec<usize>>,
    pub source_start_s: f64,
    pub source_end_s: f64,
    pub requests: Vec<AgentxRequest>,
    pub terminal_request_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AgentxRequest {
    pub request_id: String,
    pub stream_id: String,
    pub stream_turn_index: usize,
    pub source_index_path: Vec<usize>,
    pub source_outer_index: usize,
    pub source_inner_index: Option<usize>,
    pub source_type: String,
    pub source_model: String,
    pub source_t_s: f64,
    pub source_start_s: f64,
    pub source_api_time_s: f64,
    pub source_end_s: f64,
    pub source_think_time_s: Option<f64>,
    pub source_ttft_s: Option<f64>,
    pub input_length: usize,
    /// Source-corpus value, retained exactly and allowed to be zero.
    pub output_length: usize,
    /// Explicit sendable replay value. The compiler maps a source zero-output
    /// request to one token and records that policy in the bundle manifest.
    pub replay_max_output_tokens: usize,
    pub hash_ids: Vec<u32>,
    pub dependencies: Vec<AgentxDependency>,
    pub spawn_branch_ids: Vec<String>,
    pub join_branch_ids: Vec<String>,
    pub is_stream_start: bool,
    pub is_stream_terminal: bool,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentxDependencyKind {
    Stream,
    Join,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AgentxDependency {
    pub request_id: String,
    pub kind: AgentxDependencyKind,
    pub source_end_to_start_gap_s: f64,
    pub replay_delay_after_dependency_s: f64,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentxBranchKind {
    Spawn,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AgentxBranch {
    pub branch_id: String,
    pub branch_kind: AgentxBranchKind,
    pub source_kind: String,
    pub parent_stream_id: String,
    pub spawn_request_id: String,
    pub source_start_s: f64,
    pub source_recorded_end_values_s: Vec<f64>,
    pub spawn_offset_from_parent_start_s: f64,
    pub source_parent_end_to_branch_start_gap_s: f64,
    pub child_stream_ids: Vec<String>,
    pub descendant_stream_ids: Vec<String>,
    pub join_request_id: Option<String>,
    pub is_background: bool,
    pub terminal_request_ids: Vec<String>,
    /// Provenance-only compiler evidence. It never controls admission; causal
    /// replay semantics are represented by the typed fields above.
    pub source_refs: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AgentxCompileEvidence {
    pub source_requests: usize,
    pub zero_output_requests_replayed_as_one_token: usize,
    pub source_explicit_subagents: usize,
    pub preamble_requests: usize,
    pub chain_seams_merged: usize,
    pub unclassified_empty_hash_requests: usize,
    pub subagent_recorded_end_before_descendant: usize,
    pub hash_ids_preserved: bool,
    pub token_lengths_preserved: bool,
    pub source_timestamps_preserved: bool,
    pub source_api_times_preserved: bool,
    pub source_think_times_preserved_as_metadata: bool,
    pub delay_cap_applied: bool,
}

#[derive(Debug, Clone)]
pub struct AgentxExactCorpus {
    pub trace_block_size: usize,
    pub trees: Vec<AgentxExactTreeRecord>,
}

/// Immutable validated/compiled representation shared by every candidate in
/// one Python process. A trial clones only this `Arc`, not the full JSON DAG.
#[derive(Debug, Clone)]
pub struct AgentxCompiledCorpus {
    trace_block_size: usize,
    trees: Arc<Vec<CompiledTree>>,
    hash_globalization: AgentxHashGlobalizationEvidence,
}

/// Runtime-only namespace conversion. Source records stay local-per-tree so
/// provenance and DAG inference remain exact; compiled requests use a dense
/// global u32 namespace so equal local ids from different workflows cannot
/// create KV-router affinity or cache identity collisions.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct AgentxHashGlobalizationEvidence {
    pub source_hash_id_scope: String,
    pub runtime_hash_id_scope: String,
    pub mapping_algorithm: String,
    pub global_unique_hash_ids: u64,
    pub mapping_sha256: String,
    pub collision_free: bool,
}

impl AgentxExactCorpus {
    pub fn from_jsonl_files(paths: &[PathBuf]) -> Result<Self> {
        if paths.is_empty() {
            bail!("agentx_exact_v1 replay requires at least one trace file");
        }

        let mut trace_block_size = None;
        let mut trees = Vec::new();
        for path in paths {
            let file = File::open(path)
                .with_context(|| format!("failed to open AgentX trace file {}", path.display()))?;
            for (line_index, line) in BufReader::new(file).lines().enumerate() {
                let line = line.with_context(|| {
                    format!(
                        "failed to read line {} from {}",
                        line_index + 1,
                        path.display()
                    )
                })?;
                if line.trim().is_empty() {
                    continue;
                }
                let tree: AgentxExactTreeRecord =
                    serde_json::from_str(&line).with_context(|| {
                        format!(
                            "failed to parse line {} from {} as agentx_exact_v1 JSON",
                            line_index + 1,
                            path.display()
                        )
                    })?;
                ensure!(
                    tree.schema_version == AGENTX_EXACT_SCHEMA_VERSION,
                    "line {} from {} has schema_version '{}', expected '{}'",
                    line_index + 1,
                    path.display(),
                    tree.schema_version,
                    AGENTX_EXACT_SCHEMA_VERSION
                );
                if let Some(expected) = trace_block_size {
                    ensure!(
                        tree.trace_hash_block_size == expected,
                        "line {} from {} has trace_block_size {}, expected {}",
                        line_index + 1,
                        path.display(),
                        tree.trace_hash_block_size,
                        expected
                    );
                } else {
                    trace_block_size = Some(tree.trace_hash_block_size);
                }
                trees.push(tree);
            }
        }

        let corpus = Self {
            trace_block_size: trace_block_size
                .ok_or_else(|| anyhow!("agentx_exact_v1 files did not contain any trees"))?,
            trees,
        };
        corpus.validate()?;
        Ok(corpus)
    }

    pub fn compile(self) -> Result<AgentxCompiledCorpus> {
        self.validate()?;
        let mut next_global_hash_id = 0_u64;
        let mut mapping_hasher = Sha256::new();
        mapping_hasher.update(b"agentx_exact_global_dense_u32_v1\0");
        let trees = self
            .trees
            .into_iter()
            .enumerate()
            .map(|(tree_ordinal, tree)| {
                compile_tree(
                    tree,
                    tree_ordinal,
                    &mut next_global_hash_id,
                    &mut mapping_hasher,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mapping_sha256 = mapping_hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        Ok(AgentxCompiledCorpus {
            trace_block_size: self.trace_block_size,
            trees: Arc::new(trees),
            hash_globalization: AgentxHashGlobalizationEvidence {
                source_hash_id_scope: "local_per_tree".to_string(),
                runtime_hash_id_scope: "global_dense_u32".to_string(),
                mapping_algorithm: "tree_source_order_then_local_id_ascending_dense_u32_v1"
                    .to_string(),
                global_unique_hash_ids: next_global_hash_id,
                mapping_sha256: format!("sha256:{mapping_sha256}"),
                collision_free: true,
            },
        })
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.trace_block_size > 0,
            "trace_block_size must be positive"
        );
        ensure!(
            !self.trees.is_empty(),
            "AgentX corpus must contain at least one tree"
        );
        let mut tree_ids = HashSet::new();
        for tree in &self.trees {
            ensure!(
                tree.trace_hash_block_size == self.trace_block_size,
                "tree {} has inconsistent trace_hash_block_size {}",
                tree.tree_id,
                tree.trace_hash_block_size
            );
            ensure!(
                !tree.tree_id.trim().is_empty(),
                "AgentX tree_id must not be empty"
            );
            ensure!(
                tree_ids.insert(tree.tree_id.clone()),
                "duplicate AgentX tree_id {}",
                tree.tree_id
            );
            ensure!(
                tree.source_start_s.is_finite() && tree.source_start_s >= 0.0,
                "tree {} source_start_s must be finite and non-negative",
                tree.tree_id
            );
            ensure!(
                tree.source_end_s.is_finite() && tree.source_end_s > tree.source_start_s,
                "tree {} source_end_s must be finite and later than source_start_s",
                tree.tree_id
            );
            ensure!(
                tree.source_last_request_start_s.is_finite()
                    && tree.source_last_request_start_s >= tree.source_start_s
                    && tree.source_last_request_start_s <= tree.source_end_s,
                "tree {} has an invalid source_last_request_start_s",
                tree.tree_id
            );
            ensure!(
                tree.source_duration_s.is_finite()
                    && (tree.source_duration_s - (tree.source_end_s - tree.source_start_s)).abs()
                        <= 1.0e-9_f64.max(tree.source_duration_s.abs() * 1.0e-9),
                "tree {} source_duration_s does not equal source_end_s - source_start_s",
                tree.tree_id
            );
            ensure!(
                !tree.source_workflow_id.trim().is_empty()
                    && !tree.hash_namespace.trim().is_empty(),
                "tree {} has empty source identity/hash namespace",
                tree.tree_id
            );
            ensure!(
                tree.hash_id_scope == "local",
                "tree {} hash_id_scope must be 'local'",
                tree.tree_id
            );
            ensure!(
                !tree.models.is_empty() && tree.models.iter().all(|model| !model.trim().is_empty()),
                "tree {} models must contain non-empty values",
                tree.tree_id
            );
            ensure!(
                tree.compile_evidence.hash_ids_preserved
                    && tree.compile_evidence.token_lengths_preserved
                    && tree.compile_evidence.source_timestamps_preserved
                    && tree.compile_evidence.source_api_times_preserved
                    && !tree.compile_evidence.delay_cap_applied,
                "tree {} compiler evidence is not exact",
                tree.tree_id
            );
            ensure!(
                !tree.streams.is_empty(),
                "tree {} must contain at least one stream",
                tree.tree_id
            );

            let stream_ids = tree
                .streams
                .iter()
                .map(|stream| stream.stream_id.clone())
                .collect::<HashSet<_>>();
            ensure!(
                stream_ids.len() == tree.streams.len(),
                "tree {} contains duplicate stream_id values",
                tree.tree_id
            );
            let root = tree
                .streams
                .iter()
                .find(|stream| stream.stream_id == tree.root_stream_id)
                .ok_or_else(|| anyhow!("tree {} root_stream_id is undeclared", tree.tree_id))?;
            ensure!(
                root.parent_stream_id.is_none() && root.role == "root" && root.agent_depth == 0,
                "tree {} root stream has invalid role/parent/depth",
                tree.tree_id
            );
            ensure!(
                tree.streams
                    .iter()
                    .filter(|stream| stream.parent_stream_id.is_none())
                    .count()
                    == 1,
                "tree {} must contain exactly one parentless root stream",
                tree.tree_id
            );

            let requests = tree
                .streams
                .iter()
                .flat_map(|stream| stream.requests.iter())
                .collect::<Vec<_>>();
            ensure!(
                !requests.is_empty(),
                "tree {} must contain at least one request",
                tree.tree_id
            );
            let mut request_index = HashMap::new();
            let mut max_end_s = tree.source_start_s;
            for (stream_index, stream) in tree.streams.iter().enumerate() {
                ensure!(
                    !stream.stream_id.trim().is_empty() && !stream.role.trim().is_empty(),
                    "tree {} contains an empty stream identity/role",
                    tree.tree_id
                );
                if let Some(parent) = &stream.parent_stream_id {
                    ensure!(
                        stream_ids.contains(parent) && parent != &stream.stream_id,
                        "tree {} stream {} has invalid parent {}",
                        tree.tree_id,
                        stream.stream_id,
                        parent
                    );
                }
                if let Some(group) = &stream.worker_group {
                    ensure!(
                        group.len() == 2,
                        "tree {} stream {} worker_group must contain exactly two integers",
                        tree.tree_id,
                        stream.stream_id
                    );
                }
                ensure!(
                    stream.source_start_s.is_finite()
                        && stream.source_end_s.is_finite()
                        && stream.source_start_s >= tree.source_start_s
                        && stream.source_end_s >= stream.source_start_s
                        && stream.source_end_s <= tree.source_end_s,
                    "tree {} stream {} has invalid source bounds",
                    tree.tree_id,
                    stream.stream_id
                );
                ensure!(
                    !stream.requests.is_empty(),
                    "tree {} stream {} is empty",
                    tree.tree_id,
                    stream.stream_id
                );
                ensure!(
                    stream.terminal_request_id
                        == stream
                            .requests
                            .last()
                            .expect("stream is non-empty")
                            .request_id,
                    "tree {} stream {} terminal_request_id is not its last request",
                    tree.tree_id,
                    stream.stream_id
                );
                for (turn_index, request) in stream.requests.iter().enumerate() {
                    let index = request_index.len();
                    ensure!(
                        !request.request_id.trim().is_empty(),
                        "tree {} has an empty request_id",
                        tree.tree_id
                    );
                    ensure!(
                        request_index
                            .insert(request.request_id.as_str(), index)
                            .is_none(),
                        "tree {} duplicates request_id {}",
                        tree.tree_id,
                        request.request_id
                    );
                    ensure!(
                        request.stream_id == stream.stream_id
                            && request.stream_turn_index == turn_index,
                        "tree {} request {} has inconsistent stream identity/turn index",
                        tree.tree_id,
                        request.request_id
                    );
                    ensure!(
                        request.source_start_s.is_finite()
                            && request.source_start_s >= stream.source_start_s,
                        "tree {} request {} source_start_s is invalid",
                        tree.tree_id,
                        request.request_id
                    );
                    ensure!(
                        request.source_api_time_s.is_finite()
                            && request.source_api_time_s >= 0.0
                            && request.source_end_s.is_finite()
                            && request.source_end_s >= request.source_start_s
                            && (request.source_end_s
                                - (request.source_start_s + request.source_api_time_s))
                                .abs()
                                <= 1.0e-9_f64.max(request.source_end_s.abs() * 1.0e-9),
                        "tree {} request {} source end/api timing is inconsistent",
                        tree.tree_id,
                        request.request_id
                    );
                    for (name, value) in [
                        ("source_t_s", Some(request.source_t_s)),
                        ("source_think_time_s", request.source_think_time_s),
                        ("source_ttft_s", request.source_ttft_s),
                    ] {
                        if let Some(value) = value {
                            ensure!(
                                value.is_finite(),
                                "tree {} request {} {name} must be finite when present",
                                tree.tree_id,
                                request.request_id
                            );
                        }
                    }
                    ensure!(
                        request.input_length > 0,
                        "tree {} request {} input_length must be positive",
                        tree.tree_id,
                        request.request_id
                    );
                    ensure!(
                        request.replay_max_output_tokens > 0
                            && request.replay_max_output_tokens == request.output_length.max(1),
                        "tree {} request {} replay_max_output_tokens must equal max(1, output_length)",
                        tree.tree_id,
                        request.request_id
                    );
                    validate_synthesizable_prompt(
                        request.input_length,
                        &request.hash_ids,
                        self.trace_block_size,
                    )
                    .with_context(|| {
                        format!(
                            "tree {} request {} has an invalid prompt",
                            tree.tree_id, request.request_id
                        )
                    })?;
                    ensure!(
                        request.is_stream_start == (turn_index == 0)
                            && request.is_stream_terminal
                                == (turn_index + 1 == stream.requests.len()),
                        "tree {} request {} has invalid stream start/terminal flags",
                        tree.tree_id,
                        request.request_id
                    );
                    ensure!(
                        !request.source_index_path.is_empty()
                            && request.source_outer_index == request.source_index_path[0]
                            && request.source_inner_index
                                == (request.source_index_path.len() > 1)
                                    .then(|| *request.source_index_path.last().expect("non-empty")),
                        "tree {} request {} has inconsistent source index path",
                        tree.tree_id,
                        request.request_id
                    );
                    ensure!(
                        tree.models.contains(&request.source_model),
                        "tree {} request {} references undeclared source model {}",
                        tree.tree_id,
                        request.request_id,
                        request.source_model
                    );
                    max_end_s = max_end_s.max(request.source_end_s);
                }
                let first = stream.requests.first().expect("stream is non-empty");
                let stream_max_end_s = stream
                    .requests
                    .iter()
                    .map(|request| request.source_end_s)
                    .max_by(f64::total_cmp)
                    .expect("stream is non-empty");
                ensure!(
                    (stream.source_start_s - first.source_start_s).abs() <= 1.0e-9
                        && (stream.source_end_s - stream_max_end_s).abs() <= 1.0e-9,
                    "tree {} stream {} source bounds do not match requests",
                    tree.tree_id,
                    stream.stream_id
                );
                let _ = stream_index;
            }
            ensure!(
                (tree.source_end_s - max_end_s).abs() <= 1.0e-9,
                "tree {} source_end_s {} does not match last source_end_s {}",
                tree.tree_id,
                tree.source_end_s,
                max_end_s
            );
            validate_tree_topology(tree, &requests, &request_index, &stream_ids)?;
        }
        Ok(())
    }
}

fn validate_tree_topology(
    tree: &AgentxExactTreeRecord,
    requests: &[&AgentxRequest],
    request_index: &HashMap<&str, usize>,
    stream_ids: &HashSet<String>,
) -> Result<()> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Mark {
        Visiting,
        Done,
    }
    fn visit(
        index: usize,
        tree: &AgentxExactTreeRecord,
        requests: &[&AgentxRequest],
        request_index: &HashMap<&str, usize>,
        marks: &mut [Option<Mark>],
    ) -> Result<()> {
        match marks[index] {
            Some(Mark::Done) => return Ok(()),
            Some(Mark::Visiting) => bail!(
                "tree {} contains a dependency cycle at request {}",
                tree.tree_id,
                requests[index].request_id
            ),
            None => {}
        }
        marks[index] = Some(Mark::Visiting);
        let mut seen = HashSet::new();
        for dependency in &requests[index].dependencies {
            ensure!(
                seen.insert(dependency.request_id.as_str()),
                "tree {} request {} duplicates dependency {}",
                tree.tree_id,
                requests[index].request_id,
                dependency.request_id
            );
            let dependency_index = *request_index
                .get(dependency.request_id.as_str())
                .ok_or_else(|| {
                    anyhow!(
                        "tree {} request {} references unknown dependency {}",
                        tree.tree_id,
                        requests[index].request_id,
                        dependency.request_id
                    )
                })?;
            ensure!(
                dependency_index != index,
                "tree {} request {} depends on itself",
                tree.tree_id,
                requests[index].request_id
            );
            ensure!(
                dependency.source_end_to_start_gap_s.is_finite()
                    && dependency.replay_delay_after_dependency_s.is_finite()
                    && dependency.replay_delay_after_dependency_s >= 0.0
                    && (dependency.replay_delay_after_dependency_s
                        - dependency.source_end_to_start_gap_s.max(0.0))
                    .abs()
                        <= 1.0e-9_f64
                            .max(dependency.replay_delay_after_dependency_s.abs() * 1.0e-9),
                "tree {} request {} has an invalid dependency delay",
                tree.tree_id,
                requests[index].request_id
            );
            visit(dependency_index, tree, requests, request_index, marks)?;
        }
        marks[index] = Some(Mark::Done);
        Ok(())
    }

    let mut marks = vec![None; requests.len()];
    for index in 0..requests.len() {
        visit(index, tree, requests, request_index, &mut marks)?;
    }

    let terminal_ids = tree
        .streams
        .iter()
        .map(|stream| stream.terminal_request_id.as_str())
        .collect::<HashSet<_>>();
    ensure!(
        tree.tree_terminal_request_ids.len() == terminal_ids.len()
            && tree
                .tree_terminal_request_ids
                .iter()
                .all(|terminal| terminal_ids.contains(terminal.as_str())),
        "tree {} tree_terminal_request_ids is not the complete stream-terminal set",
        tree.tree_id
    );
    let request_by_id = requests
        .iter()
        .map(|request| (request.request_id.as_str(), *request))
        .collect::<HashMap<_, _>>();
    let stream_by_id = tree
        .streams
        .iter()
        .map(|stream| (stream.stream_id.as_str(), stream))
        .collect::<HashMap<_, _>>();
    let mut branch_ids = HashSet::new();
    let mut owned_children = HashSet::new();
    for branch in &tree.branches {
        ensure!(
            !branch.branch_id.trim().is_empty() && branch_ids.insert(branch.branch_id.as_str()),
            "tree {} contains an empty/duplicate branch_id {}",
            tree.tree_id,
            branch.branch_id
        );
        ensure!(
            branch.source_start_s.is_finite()
                && branch.spawn_offset_from_parent_start_s.is_finite()
                && branch.spawn_offset_from_parent_start_s >= 0.0
                && branch.source_parent_end_to_branch_start_gap_s.is_finite()
                && branch
                    .source_recorded_end_values_s
                    .iter()
                    .all(|value| value.is_finite()),
            "tree {} branch {} has invalid source timing",
            tree.tree_id,
            branch.branch_id
        );
        let spawn = request_by_id
            .get(branch.spawn_request_id.as_str())
            .ok_or_else(|| {
                anyhow!(
                    "tree {} branch {} has unknown spawn request",
                    tree.tree_id,
                    branch.branch_id
                )
            })?;
        ensure!(
            spawn.stream_id == branch.parent_stream_id
                && stream_ids.contains(&branch.parent_stream_id)
                && spawn.spawn_branch_ids.contains(&branch.branch_id)
                && (branch.spawn_offset_from_parent_start_s
                    - (branch.source_start_s - spawn.source_start_s))
                    .abs()
                    <= 1.0e-9_f64.max(branch.spawn_offset_from_parent_start_s.abs() * 1.0e-9),
            "tree {} branch {} has inconsistent SPAWN anchor/backlink",
            tree.tree_id,
            branch.branch_id
        );
        ensure!(
            !branch.child_stream_ids.is_empty(),
            "tree {} branch {} has no child streams",
            tree.tree_id,
            branch.branch_id
        );
        for child_id in &branch.child_stream_ids {
            let child = stream_by_id.get(child_id.as_str()).ok_or_else(|| {
                anyhow!(
                    "tree {} branch {} references unknown child stream {}",
                    tree.tree_id,
                    branch.branch_id,
                    child_id
                )
            })?;
            ensure!(
                owned_children.insert(child_id.as_str())
                    && child.parent_stream_id.as_deref() == Some(branch.parent_stream_id.as_str()),
                "tree {} branch {} reuses/misparents child stream {}",
                tree.tree_id,
                branch.branch_id,
                child_id
            );
        }
        ensure!(
            branch
                .descendant_stream_ids
                .iter()
                .all(|stream| stream_ids.contains(stream)),
            "tree {} branch {} has an unknown descendant stream",
            tree.tree_id,
            branch.branch_id
        );
        ensure!(
            branch.is_background == branch.join_request_id.is_none(),
            "tree {} branch {} background/JOIN declaration is inconsistent",
            tree.tree_id,
            branch.branch_id
        );
        if let Some(join_id) = &branch.join_request_id {
            let join = request_by_id.get(join_id.as_str()).ok_or_else(|| {
                anyhow!(
                    "tree {} branch {} has unknown JOIN request",
                    tree.tree_id,
                    branch.branch_id
                )
            })?;
            ensure!(
                join.join_branch_ids.contains(&branch.branch_id),
                "tree {} branch {} JOIN backlink is missing",
                tree.tree_id,
                branch.branch_id
            );
            for terminal in &branch.terminal_request_ids {
                ensure!(
                    terminal_ids.contains(terminal.as_str())
                        && join.dependencies.iter().any(|dependency| {
                            dependency.kind == AgentxDependencyKind::Join
                                && dependency.request_id == *terminal
                        }),
                    "tree {} branch {} JOIN is missing terminal dependency {}",
                    tree.tree_id,
                    branch.branch_id,
                    terminal
                );
            }
        } else {
            ensure!(
                branch
                    .terminal_request_ids
                    .iter()
                    .all(|terminal| terminal_ids.contains(terminal.as_str())),
                "tree {} background branch {} has an invalid terminal",
                tree.tree_id,
                branch.branch_id
            );
        }
    }
    let non_root_streams = stream_ids
        .iter()
        .filter(|stream| stream.as_str() != tree.root_stream_id)
        .map(String::as_str)
        .collect::<HashSet<_>>();
    ensure!(
        owned_children == non_root_streams,
        "tree {} non-root streams are not owned exactly once by SPAWN branches",
        tree.tree_id
    );
    for request in requests {
        ensure!(
            request
                .spawn_branch_ids
                .iter()
                .all(|branch| branch_ids.contains(branch.as_str()))
                && request
                    .join_branch_ids
                    .iter()
                    .all(|branch| branch_ids.contains(branch.as_str())),
            "tree {} request {} references an unknown branch",
            tree.tree_id,
            request.request_id
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentxCacheBustMode {
    FirstTurnPrefix,
}

impl Default for AgentxCacheBustMode {
    fn default() -> Self {
        Self::FirstTurnPrefix
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentxReplayConfig {
    pub session_tree_concurrency: usize,
    #[serde(default = "default_start_min_ratio")]
    pub trajectory_start_min_ratio: f64,
    #[serde(default = "default_start_max_ratio")]
    pub trajectory_start_max_ratio: f64,
    #[serde(default = "default_seed")]
    pub random_seed: u64,
    /// NumPy-PCG64 ratios precomputed by the Python protocol layer with the
    /// pinned AIPerf SHA256(seed:tree_id:lane) rule. Formal non-degenerate
    /// 25%-75% replay requires exactly one value per initial lane.
    #[serde(default)]
    pub initial_start_ratios: Option<Vec<f64>>,
    #[serde(default)]
    pub initial_tree_ids: Option<Vec<String>>,
    #[serde(default = "default_primer_max_tokens")]
    pub primer_max_tokens: usize,
    #[serde(default = "default_warmup_requests_per_lane")]
    pub warmup_requests_per_lane: usize,
    #[serde(default = "default_profile_duration_s")]
    pub profile_duration_s: f64,
    #[serde(
        default = "default_tree_idle_gap_cap_s",
        alias = "trace_idle_gap_cap_s"
    )]
    pub tree_idle_gap_cap_s: f64,
    #[serde(
        default = "default_global_idle_gap_cap_s",
        alias = "system_idle_gap_cap_s"
    )]
    pub global_idle_gap_cap_s: f64,
    #[serde(default)]
    pub cache_bust_mode: AgentxCacheBustMode,
    /// Token cost of the real `[rid:<12hex>]\n\n` marker under the pinned
    /// deployment tokenizer. DynoSim synthesizes deterministic token ids with
    /// this exact length because its trace IR stores token/hash lineage rather
    /// than message text.
    #[serde(default = "default_marker_token_count")]
    pub cache_bust_marker_token_count: usize,
    #[serde(default)]
    pub cache_bust_tokenizer_path: Option<PathBuf>,
    #[serde(default = "default_marker_tokenizer_fingerprint")]
    pub cache_bust_tokenizer_fingerprint: String,
    #[serde(default = "default_drain_grace_s")]
    pub drain_grace_s: f64,
    #[serde(default = "default_benchmark_id")]
    pub benchmark_id: String,
    #[serde(default)]
    pub capture_events: bool,
}

impl Default for AgentxReplayConfig {
    fn default() -> Self {
        Self {
            session_tree_concurrency: 1,
            trajectory_start_min_ratio: default_start_min_ratio(),
            trajectory_start_max_ratio: default_start_max_ratio(),
            random_seed: default_seed(),
            initial_start_ratios: None,
            initial_tree_ids: None,
            primer_max_tokens: default_primer_max_tokens(),
            warmup_requests_per_lane: default_warmup_requests_per_lane(),
            profile_duration_s: default_profile_duration_s(),
            tree_idle_gap_cap_s: default_tree_idle_gap_cap_s(),
            global_idle_gap_cap_s: default_global_idle_gap_cap_s(),
            cache_bust_mode: AgentxCacheBustMode::default(),
            cache_bust_marker_token_count: default_marker_token_count(),
            cache_bust_tokenizer_path: None,
            cache_bust_tokenizer_fingerprint: default_marker_tokenizer_fingerprint(),
            drain_grace_s: default_drain_grace_s(),
            benchmark_id: default_benchmark_id(),
            capture_events: false,
        }
    }
}

impl AgentxReplayConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.session_tree_concurrency > 0,
            "session_tree_concurrency must be positive"
        );
        ensure!(
            self.trajectory_start_min_ratio.is_finite()
                && self.trajectory_start_max_ratio.is_finite(),
            "trajectory start ratios must be finite"
        );
        ensure!(
            (0.0..=1.0).contains(&self.trajectory_start_min_ratio),
            "trajectory_start_min_ratio must be in [0, 1]"
        );
        ensure!(
            (0.0..=1.0).contains(&self.trajectory_start_max_ratio),
            "trajectory_start_max_ratio must be in [0, 1]"
        );
        ensure!(
            self.trajectory_start_min_ratio <= self.trajectory_start_max_ratio,
            "trajectory_start_min_ratio must not exceed trajectory_start_max_ratio"
        );
        ensure!(
            self.primer_max_tokens > 0,
            "primer_max_tokens must be positive"
        );
        if let Some(ratios) = &self.initial_start_ratios {
            ensure!(
                ratios.len() == self.session_tree_concurrency,
                "initial_start_ratios must contain exactly one value per session-tree lane"
            );
            ensure!(
                ratios.iter().all(|ratio| {
                    ratio.is_finite()
                        && *ratio >= self.trajectory_start_min_ratio
                        && *ratio <= self.trajectory_start_max_ratio
                }),
                "initial_start_ratios must be finite and inside the configured start window"
            );
        } else {
            ensure!(
                self.trajectory_start_min_ratio == self.trajectory_start_max_ratio,
                "non-degenerate trajectory sampling requires Python-precomputed initial_start_ratios for NumPy PCG64 parity"
            );
        }
        if let Some(tree_ids) = &self.initial_tree_ids {
            ensure!(
                tree_ids.len() == self.session_tree_concurrency
                    && tree_ids.iter().all(|tree_id| !tree_id.trim().is_empty()),
                "initial_tree_ids must contain one non-empty source workflow id per lane"
            );
        } else {
            ensure!(
                self.initial_start_ratios.is_none(),
                "initial_tree_ids are required with an explicit start-ratio schedule"
            );
        }
        ensure!(
            self.warmup_requests_per_lane > 0,
            "warmup_requests_per_lane must be positive"
        );
        for (name, value) in [
            ("profile_duration_s", self.profile_duration_s),
            ("tree_idle_gap_cap_s", self.tree_idle_gap_cap_s),
            ("global_idle_gap_cap_s", self.global_idle_gap_cap_s),
            ("drain_grace_s", self.drain_grace_s),
        ] {
            ensure!(
                value.is_finite() && value > 0.0,
                "{name} must be finite and positive"
            );
        }
        ensure!(
            !self.benchmark_id.trim().is_empty(),
            "benchmark_id must not be empty"
        );
        ensure!(
            !self.cache_bust_tokenizer_fingerprint.trim().is_empty(),
            "cache_bust_tokenizer_fingerprint must not be empty"
        );
        if self.cache_bust_tokenizer_path.is_some() {
            ensure!(
                self.cache_bust_tokenizer_fingerprint.starts_with("sha256:")
                    && self.cache_bust_tokenizer_fingerprint.len() == 71,
                "formal cache-bust tokenizer fingerprint must be sha256:<64 hex>"
            );
        } else {
            ensure!(
                self.cache_bust_marker_token_count > 0
                    && self.cache_bust_tokenizer_fingerprint == "test-only-single-token-marker",
                "a tokenizer path is required outside the explicit test-only marker mode"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentxPhase {
    Primer,
    Warmup,
    Profile,
    Drain,
    Complete,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentxEventKind {
    PlayStarted,
    RequestDispatched,
    RequestCompleted,
    BarrierOpened,
    ProfileEnded,
    PlayCompleted,
    PlayRecycled,
    TreeIdleGuard,
    GlobalIdleGuard,
    DrainCompleted,
    ProtocolInvalid,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AgentxEvent {
    pub kind: AgentxEventKind,
    pub phase: AgentxPhase,
    pub logical_time_ms: f64,
    pub lane_id: Option<usize>,
    pub play_id: Option<String>,
    pub source_tree_id: Option<String>,
    pub stream_id: Option<String>,
    pub source_request_id: Option<String>,
    pub request_uuid: Option<String>,
    pub cache_bust_id: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AgentxPhaseRequestCounts {
    pub dispatched: usize,
    pub completed: usize,
    pub rejected: usize,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AgentxEventCounts {
    pub play_started: usize,
    pub play_completed: usize,
    pub play_recycled: usize,
    pub tree_idle_guard: usize,
    pub global_idle_guard: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AgentxReplaySummary {
    pub schema_version: String,
    pub benchmark_id: String,
    pub protocol_valid: bool,
    pub profile_completed: bool,
    pub drain_completed: bool,
    pub censored: bool,
    pub invalid_reason: Option<String>,
    pub phase: AgentxPhase,
    pub session_tree_concurrency: usize,
    pub source_tree_count: usize,
    pub source_stream_count: usize,
    pub source_request_count: usize,
    pub source_branch_count: usize,
    pub source_joined_branch_count: usize,
    pub source_background_branch_count: usize,
    pub source_terminal_request_count: usize,
    pub source_zero_output_request_count: usize,
    pub hash_globalization: AgentxHashGlobalizationEvidence,
    pub spawn_start_anchored: bool,
    pub join_terminal_gated: bool,
    pub background_terminal_gated: bool,
    pub dependency_delay_semantics: String,
    pub snapshot_inflight_policy: String,
    pub target_model_policy: String,
    pub zero_output_policy: String,
    pub random_seed: u64,
    pub initial_sampler_order: String,
    pub initial_ratio_sampler: String,
    pub trajectory_start_min_ratio: f64,
    pub trajectory_start_max_ratio: f64,
    pub initial_tree_ids: Vec<String>,
    pub sampled_start_ratios: Vec<f64>,
    pub sampled_start_ms: Vec<f64>,
    pub primer_max_tokens: usize,
    pub warmup_requests_per_lane: usize,
    pub tree_idle_gap_cap_ms: f64,
    pub global_idle_gap_cap_ms: f64,
    pub drain_grace_ms: f64,
    pub primer: AgentxPhaseRequestCounts,
    pub warmup: AgentxPhaseRequestCounts,
    pub profile: AgentxPhaseRequestCounts,
    pub event_counts: AgentxEventCounts,
    pub profile_start_ms: Option<f64>,
    pub profile_end_ms: Option<f64>,
    pub requested_profile_duration_ms: f64,
    pub actual_profile_duration_ms: Option<f64>,
    pub drain_deadline_ms: Option<f64>,
    pub in_flight_at_profile_cutoff: usize,
    pub cache_bust_mode: AgentxCacheBustMode,
    pub cache_bust_strategy: String,
    pub cache_bust_marker_text_format: String,
    pub cache_bust_marker_token_count: usize,
    pub cache_bust_marker_token_count_min: usize,
    pub cache_bust_marker_token_count_max: usize,
    pub cache_bust_tokenization_mode: String,
    pub cache_bust_tokenizer_fingerprint: String,
    pub profile_worker_time_isolated: bool,
    /// Scheduler semantics and service-time lookup provenance are separate.
    /// In particular, an SGLang AIC database does not make a vLLM scheduler
    /// replay "SGLang faithful".
    pub topology: Option<String>,
    pub scheduler_backend: Option<String>,
    pub prefill_scheduler_backend: Option<String>,
    pub decode_scheduler_backend: Option<String>,
    pub aic_lookup_backend: Option<String>,
    pub aic_lookup_backend_version: Option<String>,
    pub aic_system: Option<String>,
    pub aic_model_path: Option<String>,
    pub aic_systems_path: Option<String>,
    pub aic_database_snapshot_sha256: Option<String>,
    pub sglang_scheduler_faithful: bool,
    pub backend_fidelity_note: Option<String>,
    pub events: Vec<AgentxEvent>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AgentxRequestContext {
    pub phase: AgentxPhase,
    pub measurement_eligible: bool,
    pub lane_id: usize,
    pub play_index: usize,
    pub play_id: String,
    pub source_tree_id: String,
    pub stream_id: String,
    pub source_request_id: String,
    pub cache_bust_id: String,
}

pub(crate) struct AgentxReadyRequest {
    pub(crate) request: ReplayRequestPayload,
    pub(crate) arrival_time_ms: f64,
    pub(crate) replay_hashes: Option<ReplayRequestHashes>,
    pub(crate) session_id: String,
    pub(crate) turn_index: usize,
    pub(crate) context: AgentxRequestContext,
}

#[derive(Debug, Clone)]
struct CompiledTree {
    source: AgentxExactTreeRecord,
    requests: Vec<AgentxRequest>,
    dependencies: Vec<Vec<CompiledDependency>>,
    dependents: Vec<Vec<CompiledDependent>>,
    spawn_children: Vec<Vec<CompiledSpawn>>,
    spawned_by: Vec<Option<usize>>,
    terminal_indexes: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
struct CompiledDependency {
    request_index: usize,
    replay_delay_ms: f64,
}

#[derive(Debug, Clone, Copy)]
struct CompiledDependent {
    request_index: usize,
    replay_delay_ms: f64,
}

#[derive(Debug, Clone, Copy)]
struct CompiledSpawn {
    request_index: usize,
    offset_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeStatus {
    Historical,
    Pending,
    InFlight,
    Complete,
}

#[derive(Debug, Clone)]
struct NodeRuntime {
    status: NodeStatus,
    remaining_dependencies: usize,
    dependency_ready_at_ms: Option<f64>,
    spawn_pending: bool,
    spawn_ready_at_ms: Option<f64>,
    ready_at_ms: Option<f64>,
}

#[derive(Debug, Clone)]
struct ActivePlay {
    lane_id: usize,
    play_index: usize,
    tree_index: usize,
    play_id: String,
    cache_bust_id: String,
    cache_bust_tokens: Vec<u32>,
    source_origin_ms: f64,
    nodes: Vec<NodeRuntime>,
    in_flight: usize,
}

impl ActivePlay {
    fn is_complete(&self, tree: &CompiledTree) -> bool {
        self.in_flight == 0
            && tree.terminal_indexes.iter().all(|index| {
                matches!(
                    self.nodes[*index].status,
                    NodeStatus::Historical | NodeStatus::Complete
                )
            })
    }

    fn earliest_ready_ms(&self) -> Option<f64> {
        self.nodes
            .iter()
            .filter(|node| node.status == NodeStatus::Pending)
            .filter_map(|node| node.ready_at_ms)
            .min_by(f64::total_cmp)
    }

    fn has_ready_at(&self, now_ms: f64) -> bool {
        self.nodes.iter().any(|node| {
            node.status == NodeStatus::Pending
                && node.ready_at_ms.is_some_and(|ready_at| ready_at <= now_ms)
        })
    }

    fn shift_ready_times(&mut self, delta_ms: f64) {
        for node in &mut self.nodes {
            if let Some(ready_at_ms) = node.ready_at_ms.as_mut() {
                *ready_at_ms = (*ready_at_ms - delta_ms).max(0.0);
            }
            if let Some(ready_at_ms) = node.dependency_ready_at_ms.as_mut() {
                *ready_at_ms = (*ready_at_ms - delta_ms).max(0.0);
            }
            if let Some(ready_at_ms) = node.spawn_ready_at_ms.as_mut() {
                *ready_at_ms = (*ready_at_ms - delta_ms).max(0.0);
            }
        }
    }
}

#[derive(Debug)]
struct LaneState {
    play: ActivePlay,
    next_play_index: usize,
    warmup_dispatched: usize,
    warmup_completed: usize,
    last_wire_activity_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InFlightKind {
    Primer,
    WarmupNode(usize),
    ProfileNode(usize),
}

#[derive(Debug, Clone)]
struct InFlightRequest {
    lane_id: usize,
    kind: InFlightKind,
    context: AgentxRequestContext,
}

#[derive(Debug, Clone)]
struct PrimerTemplate {
    lane_id: usize,
    request_index: usize,
    dispatch_at_ms: f64,
}

#[derive(Clone)]
enum MarkerTokenizer {
    HuggingFace(Arc<Tokenizer>),
    Synthetic { token_count: usize },
}

impl std::fmt::Debug for MarkerTokenizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HuggingFace(_) => formatter.write_str("MarkerTokenizer::HuggingFace"),
            Self::Synthetic { token_count } => formatter
                .debug_struct("MarkerTokenizer::Synthetic")
                .field("token_count", token_count)
                .finish(),
        }
    }
}

impl MarkerTokenizer {
    fn from_config(config: &AgentxReplayConfig) -> Result<Self> {
        let Some(path) = config.cache_bust_tokenizer_path.as_ref() else {
            return Ok(Self::Synthetic {
                token_count: config.cache_bust_marker_token_count,
            });
        };
        let actual = sha256_file(path)?;
        let expected = config
            .cache_bust_tokenizer_fingerprint
            .strip_prefix("sha256:")
            .expect("validated sha256 marker tokenizer fingerprint");
        ensure!(
            actual == expected,
            "cache-bust tokenizer fingerprint mismatch for {}: expected {}, got {}",
            path.display(),
            expected,
            actual
        );
        let tokenizer = Tokenizer::from_file(path).map_err(|error| {
            anyhow!(
                "failed to load cache-bust tokenizer {}: {error}",
                path.display()
            )
        })?;
        Ok(Self::HuggingFace(Arc::new(tokenizer)))
    }

    fn encode(&self, marker: &str) -> Result<Vec<u32>> {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer
                .encode(marker, false)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|error| anyhow!("failed to tokenize cache-bust marker: {error}")),
            Self::Synthetic { token_count } => Ok(synthetic_marker_tokens(marker, *token_count)),
        }
    }

    fn mode(&self) -> &'static str {
        match self {
            Self::HuggingFace(_) => "tokenizer_encoded_per_play",
            Self::Synthetic { .. } => "test_only_synthetic_fixed_count",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ReadyNodeItem {
    ready_at_ms: f64,
    lane_id: usize,
    play_index: usize,
    request_index: usize,
}

impl Eq for ReadyNodeItem {}

impl Ord for ReadyNodeItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .ready_at_ms
            .total_cmp(&self.ready_at_ms)
            .then_with(|| other.lane_id.cmp(&self.lane_id))
            .then_with(|| other.request_index.cmp(&self.request_index))
    }
}

impl PartialOrd for ReadyNodeItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Closed-loop exact session-tree controller shared by aggregate and disagg
/// replay. It emits requests, then advances dependencies only from terminal
/// callbacks supplied by the target runtime.
pub(crate) struct AgentxReplayController {
    corpus: Arc<Vec<CompiledTree>>,
    trace_block_size: usize,
    engine_block_size: u32,
    include_replay_hashes: bool,
    config: AgentxReplayConfig,
    lanes: Vec<LaneState>,
    next_tree_cursor: usize,
    recycle_pass_by_tree: HashMap<usize, usize>,
    marker_tokenizer: MarkerTokenizer,
    primers: VecDeque<PrimerTemplate>,
    in_flight: HashMap<Uuid, InFlightRequest>,
    phase: AgentxPhase,
    last_now_ms: f64,
    last_global_wire_activity_ms: f64,
    profile_start_ms: Option<f64>,
    profile_deadline_ms: Option<f64>,
    drain_deadline_ms: Option<f64>,
    drain_timeout_fired: bool,
    ready_queue: BinaryHeap<ReadyNodeItem>,
    summary: AgentxReplaySummary,
}

impl AgentxReplayController {
    #[cfg(test)]
    pub(crate) fn new(
        corpus: AgentxExactCorpus,
        config: AgentxReplayConfig,
        engine_block_size: usize,
        include_replay_hashes: bool,
    ) -> Result<Self> {
        Self::new_compiled(
            corpus.compile()?,
            config,
            engine_block_size,
            include_replay_hashes,
        )
    }

    pub(crate) fn new_compiled(
        corpus: AgentxCompiledCorpus,
        config: AgentxReplayConfig,
        engine_block_size: usize,
        include_replay_hashes: bool,
    ) -> Result<Self> {
        config.validate()?;
        let engine_block_size =
            u32::try_from(engine_block_size).context("engine block size does not fit in u32")?;
        ensure!(engine_block_size > 0, "engine block size must be positive");
        let trace_block_size = corpus.trace_block_size;
        let hash_globalization = corpus.hash_globalization;
        let compiled = corpus.trees;
        let marker_tokenizer = MarkerTokenizer::from_config(&config)?;

        let mut sampled_start_ratios = Vec::with_capacity(config.session_tree_concurrency);
        let mut sampled_start_ms = Vec::with_capacity(config.session_tree_concurrency);
        let mut initial_tree_ids = Vec::with_capacity(config.session_tree_concurrency);
        let mut lanes = Vec::with_capacity(config.session_tree_concurrency);
        let mut primer_leads = Vec::<(usize, usize, f64)>::new();
        let mut recycle_pass_by_tree = HashMap::<usize, usize>::new();
        for lane_id in 0..config.session_tree_concurrency {
            // The pinned AIPerf default sampler is sequential and wraps. It is
            // deliberately not shuffled here: the ratio schedule is computed
            // independently per (seed, tree_id, absolute lane) by Python.
            let tree_index = lane_id % compiled.len();
            if let Some(tree_ids) = &config.initial_tree_ids {
                ensure!(
                    tree_ids[lane_id] == compiled[tree_index].source.source_workflow_id,
                    "initial sampler identity mismatch for lane {lane_id}: schedule has '{}', corpus has '{}'",
                    tree_ids[lane_id],
                    compiled[tree_index].source.source_workflow_id
                );
            }
            let ratio = config
                .initial_start_ratios
                .as_ref()
                .map(|ratios| ratios[lane_id])
                .unwrap_or(config.trajectory_start_min_ratio);
            sampled_start_ratios.push(ratio);
            let snapshot_ms = (compiled[tree_index].source.source_start_s
                + (compiled[tree_index].source.source_last_request_start_s
                    - compiled[tree_index].source.source_start_s)
                    * ratio)
                * 1000.0;
            sampled_start_ms.push(snapshot_ms);
            initial_tree_ids.push(compiled[tree_index].source.source_workflow_id.clone());
            let recycle_pass = next_recycle_pass(&mut recycle_pass_by_tree, tree_index);
            let play = build_play(
                &compiled[tree_index],
                tree_index,
                lane_id,
                0,
                Some(snapshot_ms),
                0.0,
                &config.benchmark_id,
                recycle_pass,
                &marker_tokenizer,
            )?;

            let streams_with_future = compiled[tree_index]
                .requests
                .iter()
                .filter(|request| request.source_start_s * 1000.0 >= snapshot_ms)
                .map(|request| request.stream_id.as_str())
                .collect::<HashSet<_>>();
            let mut latest_by_stream = HashMap::<&str, usize>::new();
            for (request_index, request) in compiled[tree_index].requests.iter().enumerate() {
                if request.source_start_s * 1000.0 < snapshot_ms
                    && streams_with_future.contains(request.stream_id.as_str())
                {
                    let replace =
                        latest_by_stream
                            .get(request.stream_id.as_str())
                            .is_none_or(|current| {
                                compiled[tree_index].requests[*current].source_start_s
                                    <= request.source_start_s
                            });
                    if replace {
                        latest_by_stream.insert(&request.stream_id, request_index);
                    }
                }
            }
            let mut primer_indexes = latest_by_stream.into_values().collect::<Vec<_>>();
            primer_indexes.sort_unstable();
            for request_index in &primer_indexes {
                let lead_ms = (snapshot_ms
                    - compiled[tree_index].requests[*request_index].source_start_s * 1000.0)
                    .max(0.0)
                    .min(config.global_idle_gap_cap_s * 1000.0);
                primer_leads.push((lane_id, *request_index, lead_ms));
            }
            lanes.push(LaneState {
                play,
                next_play_index: 1,
                warmup_dispatched: 0,
                warmup_completed: 0,
                last_wire_activity_ms: 0.0,
            });
        }
        let max_primer_lead_ms = primer_leads
            .iter()
            .map(|(_, _, lead_ms)| *lead_ms)
            .max_by(f64::total_cmp)
            .unwrap_or(0.0);
        let mut primer_templates = primer_leads
            .into_iter()
            .map(|(lane_id, request_index, lead_ms)| PrimerTemplate {
                lane_id,
                request_index,
                dispatch_at_ms: max_primer_lead_ms - lead_ms,
            })
            .collect::<Vec<_>>();
        primer_templates.sort_by(|left, right| {
            left.dispatch_at_ms
                .total_cmp(&right.dispatch_at_ms)
                .then_with(|| left.lane_id.cmp(&right.lane_id))
                .then_with(|| left.request_index.cmp(&right.request_index))
        });
        let primers = VecDeque::from(primer_templates);

        let initial_ratio_sampler = if config.initial_start_ratios.is_some() {
            "python_numpy_default_rng_sha256_seed_tree_id_lane_big_endian_u64"
        } else {
            "fixed_ratio_no_rng"
        };
        let marker_token_count_min = lanes
            .iter()
            .map(|lane| lane.play.cache_bust_tokens.len())
            .min()
            .unwrap_or(0);
        let marker_token_count_max = lanes
            .iter()
            .map(|lane| lane.play.cache_bust_tokens.len())
            .max()
            .unwrap_or(0);
        let summary = AgentxReplaySummary {
            schema_version: AGENTX_EXACT_SCHEMA_VERSION.to_string(),
            benchmark_id: config.benchmark_id.clone(),
            protocol_valid: true,
            profile_completed: false,
            drain_completed: false,
            censored: false,
            invalid_reason: None,
            phase: AgentxPhase::Primer,
            session_tree_concurrency: config.session_tree_concurrency,
            source_tree_count: compiled.len(),
            source_stream_count: compiled.iter().map(|tree| tree.source.streams.len()).sum(),
            source_request_count: compiled.iter().map(|tree| tree.requests.len()).sum(),
            source_branch_count: compiled.iter().map(|tree| tree.source.branches.len()).sum(),
            source_joined_branch_count: compiled
                .iter()
                .flat_map(|tree| tree.source.branches.iter())
                .filter(|branch| !branch.is_background)
                .count(),
            source_background_branch_count: compiled
                .iter()
                .flat_map(|tree| tree.source.branches.iter())
                .filter(|branch| branch.is_background)
                .count(),
            source_terminal_request_count: compiled
                .iter()
                .map(|tree| tree.terminal_indexes.len())
                .sum(),
            source_zero_output_request_count: compiled
                .iter()
                .map(|tree| {
                    tree.source
                        .compile_evidence
                        .zero_output_requests_replayed_as_one_token
                })
                .sum(),
            hash_globalization,
            spawn_start_anchored: true,
            join_terminal_gated: true,
            background_terminal_gated: true,
            dependency_delay_semantics: "dependency_terminal_plus_per_edge_replay_delay"
                .to_string(),
            snapshot_inflight_policy:
                "requests_started_before_snapshot_are_primerized_not_mid_decode_resumed".to_string(),
            target_model_policy: "deployment_model_for_all_source_model_labels".to_string(),
            zero_output_policy: "preserve_source_and_replay_one_token".to_string(),
            random_seed: config.random_seed,
            initial_sampler_order: "sequential_wrap".to_string(),
            initial_ratio_sampler: initial_ratio_sampler.to_string(),
            trajectory_start_min_ratio: config.trajectory_start_min_ratio,
            trajectory_start_max_ratio: config.trajectory_start_max_ratio,
            initial_tree_ids,
            sampled_start_ratios,
            sampled_start_ms,
            primer_max_tokens: config.primer_max_tokens,
            warmup_requests_per_lane: config.warmup_requests_per_lane,
            tree_idle_gap_cap_ms: config.tree_idle_gap_cap_s * 1000.0,
            global_idle_gap_cap_ms: config.global_idle_gap_cap_s * 1000.0,
            drain_grace_ms: config.drain_grace_s * 1000.0,
            primer: AgentxPhaseRequestCounts::default(),
            warmup: AgentxPhaseRequestCounts::default(),
            profile: AgentxPhaseRequestCounts::default(),
            event_counts: AgentxEventCounts::default(),
            profile_start_ms: None,
            profile_end_ms: None,
            requested_profile_duration_ms: config.profile_duration_s * 1000.0,
            actual_profile_duration_ms: None,
            drain_deadline_ms: None,
            in_flight_at_profile_cutoff: 0,
            cache_bust_mode: config.cache_bust_mode,
            cache_bust_strategy: "prepend_tokenizer_calibrated_unique_marker_prefix_per_play"
                .to_string(),
            cache_bust_marker_text_format: "[rid:<sha256-12hex>]\\n\\n".to_string(),
            cache_bust_marker_token_count: config.cache_bust_marker_token_count,
            cache_bust_marker_token_count_min: marker_token_count_min,
            cache_bust_marker_token_count_max: marker_token_count_max,
            cache_bust_tokenization_mode: marker_tokenizer.mode().to_string(),
            cache_bust_tokenizer_fingerprint: config.cache_bust_tokenizer_fingerprint.clone(),
            profile_worker_time_isolated: true,
            topology: None,
            scheduler_backend: None,
            prefill_scheduler_backend: None,
            decode_scheduler_backend: None,
            aic_lookup_backend: None,
            aic_lookup_backend_version: None,
            aic_system: None,
            aic_model_path: None,
            aic_systems_path: None,
            aic_database_snapshot_sha256: None,
            sglang_scheduler_faithful: false,
            backend_fidelity_note: None,
            events: Vec::new(),
        };

        let next_tree_cursor = config.session_tree_concurrency;
        let mut controller = Self {
            corpus: compiled,
            trace_block_size,
            engine_block_size,
            include_replay_hashes,
            config,
            lanes,
            next_tree_cursor,
            recycle_pass_by_tree,
            marker_tokenizer,
            primers,
            in_flight: HashMap::new(),
            phase: AgentxPhase::Primer,
            last_now_ms: 0.0,
            last_global_wire_activity_ms: 0.0,
            profile_start_ms: None,
            profile_deadline_ms: None,
            drain_deadline_ms: None,
            drain_timeout_fired: false,
            ready_queue: BinaryHeap::new(),
            summary,
        };
        for lane_id in 0..controller.lanes.len() {
            controller.record_play_event(AgentxEventKind::PlayStarted, lane_id, 0.0);
        }
        Ok(controller)
    }

    pub(crate) fn next_ready_time_ms(&mut self) -> Option<f64> {
        match self.phase {
            AgentxPhase::Primer => self
                .primers
                .front()
                .map(|primer| primer.dispatch_at_ms.max(self.last_now_ms))
                .or_else(|| self.in_flight.is_empty().then_some(self.last_now_ms)),
            AgentxPhase::Warmup => (self.warmup_complete()
                || self.lanes.iter().any(|lane| {
                    lane.warmup_dispatched < self.config.warmup_requests_per_lane
                        && (lane.play.is_complete(&self.corpus[lane.play.tree_index])
                            || lane.play.nodes.iter().any(warmup_node_dispatchable))
                }))
            .then_some(self.last_now_ms),
            AgentxPhase::Profile => {
                if self.ready_queue.is_empty() && self.any_ready_pending() {
                    self.rebuild_ready_queue();
                }
                let mut next = self.profile_deadline_ms;
                self.purge_stale_ready_queue_top();
                if let Some(top) = self.ready_queue.peek() {
                    next = min_time(next, Some(top.ready_at_ms));
                }
                for lane in &self.lanes {
                    if lane.play.in_flight == 0 && !lane.play.has_ready_at(self.last_now_ms) {
                        if let Some(_ready_at_ms) = lane.play.earliest_ready_ms() {
                            next = min_time(
                                next,
                                Some(
                                    (lane.last_wire_activity_ms
                                        + self.config.tree_idle_gap_cap_s * 1000.0)
                                        .max(self.last_now_ms),
                                ),
                            );
                        }
                    }
                }
                if self.in_flight.is_empty()
                    && !self.any_ready_at(self.last_now_ms)
                    && self
                        .lanes
                        .iter()
                        .any(|lane| lane.play.earliest_ready_ms().is_some())
                {
                    next = min_time(
                        next,
                        Some(
                            (self.last_global_wire_activity_ms
                                + self.config.global_idle_gap_cap_s * 1000.0)
                                .max(self.last_now_ms),
                        ),
                    );
                }
                next
            }
            AgentxPhase::Drain if !self.drain_timeout_fired && !self.in_flight.is_empty() => {
                self.drain_deadline_ms
            }
            AgentxPhase::Drain | AgentxPhase::Complete => None,
        }
    }

    pub(crate) fn drain_ready(&mut self, now_ms: f64) -> Result<Vec<AgentxReadyRequest>> {
        ensure!(
            now_ms.is_finite() && now_ms >= self.last_now_ms,
            "AgentX logical clock moved backwards from {} to {}",
            self.last_now_ms,
            now_ms
        );
        self.last_now_ms = now_ms;
        let mut emitted = Vec::new();

        loop {
            match self.phase {
                AgentxPhase::Primer => {
                    while self
                        .primers
                        .front()
                        .is_some_and(|template| template.dispatch_at_ms <= now_ms)
                    {
                        let template = self.primers.pop_front().expect("front was present");
                        emitted.push(self.emit_template_request(
                            template.lane_id,
                            template.request_index,
                            AgentxPhase::Primer,
                            InFlightKind::Primer,
                            now_ms,
                        )?);
                    }
                    if self.primers.is_empty() && self.in_flight.is_empty() {
                        self.begin_warmup();
                        continue;
                    }
                }
                AgentxPhase::Warmup => {
                    loop {
                        let mut progressed = false;
                        for lane_id in 0..self.lanes.len() {
                            while self.lanes[lane_id].warmup_dispatched
                                < self.config.warmup_requests_per_lane
                            {
                                let tree_index = self.lanes[lane_id].play.tree_index;
                                if self.lanes[lane_id]
                                    .play
                                    .is_complete(&self.corpus[tree_index])
                                {
                                    self.recycle_lane(lane_id, now_ms)?;
                                    progressed = true;
                                    continue;
                                }
                                let Some(request_index) = self.lanes[lane_id]
                                    .play
                                    .nodes
                                    .iter()
                                    .position(warmup_node_dispatchable)
                                else {
                                    break;
                                };
                                self.lanes[lane_id].warmup_dispatched += 1;
                                emitted.push(self.emit_graph_request(
                                    lane_id,
                                    request_index,
                                    AgentxPhase::Warmup,
                                    now_ms,
                                )?);
                                progressed = true;
                            }
                        }
                        if !progressed {
                            break;
                        }
                    }
                    if self.warmup_complete() {
                        self.begin_profile(now_ms)?;
                        continue;
                    }
                    if self.in_flight.is_empty() {
                        self.invalidate(
                            "warmup DAG made no progress before every lane reached its wire quota",
                            now_ms,
                        );
                        continue;
                    }
                }
                AgentxPhase::Profile => {
                    if self
                        .profile_deadline_ms
                        .is_some_and(|deadline| now_ms >= deadline)
                    {
                        self.begin_drain(now_ms);
                        continue;
                    }
                    self.apply_idle_guards(now_ms);
                    if self.ready_queue.is_empty() && self.any_ready_pending() {
                        self.rebuild_ready_queue();
                    }
                    while let Some(top) = self.ready_queue.peek() {
                        if top.ready_at_ms > now_ms {
                            break;
                        }
                        let item = self.ready_queue.pop().expect("top was present");
                        let lane = &self.lanes[item.lane_id];
                        if lane.play.play_index == item.play_index {
                            let node = &lane.play.nodes[item.request_index];
                            if node.status == NodeStatus::Pending
                                && node.ready_at_ms.is_some_and(|ready_at| ready_at <= now_ms)
                            {
                                emitted.push(self.emit_graph_request(
                                    item.lane_id,
                                    item.request_index,
                                    AgentxPhase::Profile,
                                    now_ms,
                                )?);
                            }
                        }
                    }
                }
                AgentxPhase::Drain => {
                    if !self.drain_timeout_fired
                        && self
                            .drain_deadline_ms
                            .is_some_and(|deadline| now_ms >= deadline)
                        && !self.in_flight.is_empty()
                    {
                        self.drain_timeout_fired = true;
                        self.invalidate(
                            "drain grace expired before emitted requests completed",
                            now_ms,
                        );
                    }
                    self.complete_drain_if_ready(now_ms);
                }
                AgentxPhase::Complete => {}
            }
            break;
        }
        Ok(emitted)
    }

    pub(crate) fn on_output_token(&mut self, request_id: Uuid, _token_id: u32) -> Result<()> {
        ensure!(
            self.in_flight.contains_key(&request_id),
            "unknown AgentX output request {request_id}"
        );
        Ok(())
    }

    pub(crate) fn on_terminal(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
        rejected: bool,
    ) -> Result<()> {
        ensure!(
            now_ms.is_finite() && now_ms >= self.last_now_ms,
            "AgentX logical clock moved backwards from {} to {}",
            self.last_now_ms,
            now_ms
        );
        self.last_now_ms = now_ms;
        let flight = self
            .in_flight
            .remove(&request_id)
            .ok_or_else(|| anyhow!("unknown AgentX terminal request {request_id}"))?;
        self.lanes[flight.lane_id].last_wire_activity_ms = now_ms;
        self.last_global_wire_activity_ms = now_ms;
        let counts = self.phase_counts_mut(flight.context.phase);
        if rejected {
            counts.rejected += 1;
        } else {
            counts.completed += 1;
        }
        self.record_request_event(
            AgentxEventKind::RequestCompleted,
            &flight.context,
            request_id,
            now_ms,
        );

        if rejected {
            self.invalidate("an emitted AgentX request was rejected", now_ms);
        }

        match flight.kind {
            InFlightKind::Primer => {
                if self.phase == AgentxPhase::Primer
                    && self.primers.is_empty()
                    && self.in_flight.is_empty()
                {
                    self.begin_warmup();
                }
            }
            InFlightKind::WarmupNode(request_index) => {
                let lane = &mut self.lanes[flight.lane_id];
                lane.warmup_completed += 1;
                self.complete_graph_node(
                    flight.lane_id,
                    request_index,
                    now_ms,
                    AgentxPhase::Warmup,
                )?;
                if self.phase == AgentxPhase::Warmup && self.warmup_complete() {
                    self.begin_profile(now_ms)?;
                }
            }
            InFlightKind::ProfileNode(request_index) => {
                self.complete_graph_node(
                    flight.lane_id,
                    request_index,
                    now_ms,
                    AgentxPhase::Profile,
                )?;
            }
        }
        if self.phase == AgentxPhase::Drain {
            self.complete_drain_if_ready(now_ms);
        }
        Ok(())
    }

    pub(crate) fn is_drained(&self) -> bool {
        self.phase == AgentxPhase::Complete
    }

    pub(crate) fn total_requests_hint(&self) -> usize {
        let primer = self.summary.primer.dispatched.max(self.primers.len());
        let warmup = self
            .config
            .session_tree_concurrency
            .saturating_mul(self.config.warmup_requests_per_lane);
        let one_pass = self
            .lanes
            .iter()
            .map(|lane| self.corpus[lane.play.tree_index].requests.len())
            .sum::<usize>();
        primer.saturating_add(warmup).saturating_add(one_pass)
    }

    pub(crate) fn profile_overlap_ms(&self, from_ms: f64, to_ms: f64) -> f64 {
        let Some(start_ms) = self.profile_start_ms else {
            return 0.0;
        };
        let end_ms = self.profile_deadline_ms.unwrap_or(to_ms);
        (to_ms.min(end_ms) - from_ms.max(start_ms)).max(0.0)
    }

    pub(crate) fn summary(&self) -> AgentxReplaySummary {
        let mut summary = self.summary.clone();
        summary.phase = self.phase;
        if let Some(start_ms) = self.profile_start_ms {
            let end_ms = self.summary.profile_end_ms.unwrap_or(
                self.last_now_ms
                    .min(self.profile_deadline_ms.unwrap_or(self.last_now_ms)),
            );
            summary.actual_profile_duration_ms = Some((end_ms - start_ms).max(0.0));
        }
        summary
    }

    fn begin_warmup(&mut self) {
        if self.phase != AgentxPhase::Primer {
            return;
        }
        self.phase = AgentxPhase::Warmup;
        self.summary.phase = self.phase;
    }

    fn warmup_complete(&self) -> bool {
        self.in_flight.is_empty()
            && self.lanes.iter().all(|lane| {
                lane.warmup_dispatched >= self.config.warmup_requests_per_lane
                    && lane.warmup_completed == lane.warmup_dispatched
            })
    }

    fn begin_profile(&mut self, now_ms: f64) -> Result<()> {
        if self.phase != AgentxPhase::Warmup || !self.warmup_complete() {
            return Ok(());
        }
        // A lane whose Nth warmup request drained its tree must still hold a
        // live tree when profiling opens. AIPerf draws the next sequential
        // root at handoff, starting it at turn zero with a fresh marker.
        for lane_id in 0..self.lanes.len() {
            let tree_index = self.lanes[lane_id].play.tree_index;
            if self.lanes[lane_id]
                .play
                .is_complete(&self.corpus[tree_index])
            {
                self.recycle_lane(lane_id, now_ms)?;
            }
        }
        self.phase = AgentxPhase::Profile;
        self.summary.phase = self.phase;
        self.profile_start_ms = Some(now_ms);
        let deadline = now_ms + self.config.profile_duration_s * 1000.0;
        self.profile_deadline_ms = Some(deadline);
        self.summary.profile_start_ms = Some(now_ms);
        self.last_global_wire_activity_ms = now_ms;
        for lane in &mut self.lanes {
            lane.last_wire_activity_ms = now_ms;
        }
        self.record_simple_event(AgentxEventKind::BarrierOpened, now_ms);
        let global_min_offset_ms = self
            .lanes
            .iter()
            .flat_map(|lane| {
                lane.play.nodes.iter().filter_map(|node| {
                    warmup_node_dispatchable(node)
                        .then_some(node.ready_at_ms)
                        .flatten()
                })
            })
            .min_by(f64::total_cmp)
            .unwrap_or(0.0);
        for lane in &mut self.lanes {
            anchor_play_ready_times(&mut lane.play, now_ms, global_min_offset_ms);
        }
        self.rebuild_ready_queue();
        Ok(())
    }

    fn begin_drain(&mut self, now_ms: f64) {
        if self.phase != AgentxPhase::Profile {
            return;
        }
        let cutoff = self.profile_deadline_ms.unwrap_or(now_ms);
        self.phase = AgentxPhase::Drain;
        self.summary.phase = self.phase;
        self.summary.profile_completed = true;
        self.summary.profile_end_ms = Some(cutoff);
        self.summary.actual_profile_duration_ms = self
            .profile_start_ms
            .map(|start_ms| (cutoff - start_ms).max(0.0));
        self.summary.in_flight_at_profile_cutoff = self.in_flight.len();
        let drain_deadline = cutoff + self.config.drain_grace_s * 1000.0;
        self.drain_deadline_ms = Some(drain_deadline);
        self.summary.drain_deadline_ms = Some(drain_deadline);
        self.record_simple_event(AgentxEventKind::ProfileEnded, cutoff);
        self.complete_drain_if_ready(now_ms);
    }

    fn complete_drain_if_ready(&mut self, now_ms: f64) {
        if self.phase != AgentxPhase::Drain || !self.in_flight.is_empty() {
            return;
        }
        if !self.drain_timeout_fired && !self.summary.censored {
            self.summary.drain_completed = true;
        }
        self.phase = AgentxPhase::Complete;
        self.summary.phase = self.phase;
        self.record_simple_event(AgentxEventKind::DrainCompleted, now_ms);
    }

    fn invalidate(&mut self, reason: &str, now_ms: f64) {
        if self.summary.protocol_valid {
            self.summary.protocol_valid = false;
            self.summary.censored = true;
            self.summary.invalid_reason = Some(reason.to_string());
            self.record_simple_event(AgentxEventKind::ProtocolInvalid, now_ms);
        }
        if matches!(
            self.phase,
            AgentxPhase::Primer | AgentxPhase::Warmup | AgentxPhase::Profile
        ) {
            self.phase = AgentxPhase::Drain;
            self.summary.phase = self.phase;
            self.drain_deadline_ms = Some(now_ms);
            self.drain_timeout_fired = true;
        }
        self.complete_drain_if_ready(now_ms);
    }

    fn emit_template_request(
        &mut self,
        lane_id: usize,
        request_index: usize,
        phase: AgentxPhase,
        kind: InFlightKind,
        now_ms: f64,
    ) -> Result<AgentxReadyRequest> {
        let play = &self.lanes[lane_id].play;
        let compiled = &self.corpus[play.tree_index];
        let tree = &compiled.source;
        let source = &compiled.requests[request_index];
        let context = request_context(play, tree, source, phase);
        let nonce = match phase {
            AgentxPhase::Primer => self.summary.primer.dispatched,
            AgentxPhase::Warmup => self.summary.warmup.dispatched,
            _ => 0,
        };
        let uuid = stable_uuid(&format!(
            "{}:{}:{:?}:{nonce}",
            self.config.benchmark_id, play.play_id, phase
        ));
        let output_limit = match phase {
            AgentxPhase::Primer => self.config.primer_max_tokens,
            AgentxPhase::Warmup => 1,
            _ => source.replay_max_output_tokens,
        };
        let ready = self.build_ready_request(
            source,
            play,
            context.clone(),
            uuid,
            request_index,
            output_limit,
            now_ms,
        )?;
        self.in_flight.insert(
            uuid,
            InFlightRequest {
                lane_id,
                kind,
                context: context.clone(),
            },
        );
        self.lanes[lane_id].last_wire_activity_ms = now_ms;
        self.last_global_wire_activity_ms = now_ms;
        self.phase_counts_mut(phase).dispatched += 1;
        self.record_request_event(AgentxEventKind::RequestDispatched, &context, uuid, now_ms);
        Ok(ready)
    }

    fn emit_graph_request(
        &mut self,
        lane_id: usize,
        request_index: usize,
        phase: AgentxPhase,
        now_ms: f64,
    ) -> Result<AgentxReadyRequest> {
        ensure!(
            matches!(phase, AgentxPhase::Warmup | AgentxPhase::Profile),
            "AgentX graph requests are only valid during warmup or profile"
        );
        let play = &self.lanes[lane_id].play;
        let compiled = &self.corpus[play.tree_index];
        let tree = &compiled.source;
        let source = &compiled.requests[request_index];
        let context = request_context(play, tree, source, phase);
        let uuid = stable_uuid(&format!(
            "{}:{}:{phase:?}:{}",
            self.config.benchmark_id, play.play_id, source.request_id,
        ));
        let output_limit = if phase == AgentxPhase::Warmup {
            1
        } else {
            source.replay_max_output_tokens
        };
        let ready = self.build_ready_request(
            source,
            play,
            context.clone(),
            uuid,
            request_index,
            output_limit,
            now_ms,
        )?;
        let lane = &mut self.lanes[lane_id];
        lane.play.nodes[request_index].status = NodeStatus::InFlight;
        lane.play.nodes[request_index].ready_at_ms = None;
        lane.play.in_flight += 1;
        self.in_flight.insert(
            uuid,
            InFlightRequest {
                lane_id,
                kind: if phase == AgentxPhase::Warmup {
                    InFlightKind::WarmupNode(request_index)
                } else {
                    InFlightKind::ProfileNode(request_index)
                },
                context: context.clone(),
            },
        );
        self.lanes[lane_id].last_wire_activity_ms = now_ms;
        self.last_global_wire_activity_ms = now_ms;
        self.phase_counts_mut(phase).dispatched += 1;
        self.record_request_event(AgentxEventKind::RequestDispatched, &context, uuid, now_ms);
        self.release_spawn_children(lane_id, request_index, now_ms, phase);
        Ok(ready)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_ready_request(
        &self,
        source: &AgentxRequest,
        play: &ActivePlay,
        context: AgentxRequestContext,
        uuid: Uuid,
        turn_index: usize,
        output_limit: usize,
        now_ms: f64,
    ) -> Result<AgentxReadyRequest> {
        let hash_ids = source.hash_ids.clone();
        validate_synthesizable_prompt(source.input_length, &hash_ids, self.trace_block_size)?;
        let max_output_tokens = source.replay_max_output_tokens.min(output_limit).max(1);
        // AgentX synthetic payloads are part of the deterministic benchmark
        // identity. Leaving this unset delegates decode token generation to
        // `rand::random`, which preserves lengths/timing but makes completed
        // output-block PLHs (and therefore G1/G2 resident checksums) differ
        // between otherwise identical replay processes. Derive an independent
        // stream from the already-stable request UUID so dispatch order cannot
        // perturb another request's output tokens.
        let uuid_bits = uuid.as_u128();
        let output_seed =
            SYNTHETIC_OUTPUT_SEED ^ (uuid_bits as u64) ^ ((uuid_bits >> 64) as u64).rotate_left(29);
        let mut output_rng = StdRng::seed_from_u64(output_seed);
        let request_metadata = DirectRequest {
            tokens: Vec::new(),
            max_output_tokens,
            output_token_ids: Some(planned_output_token_ids(
                None,
                max_output_tokens,
                &mut output_rng,
            )),
            uuid: Some(uuid),
            dp_rank: 0,
            arrival_timestamp_ms: Some(now_ms),
            priority: 0,
            strict_priority: 0,
            policy_class: None,
        };
        let request = ReplayRequestPayload::deferred_with_prefix(
            request_metadata,
            source.input_length,
            hash_ids,
            self.trace_block_size,
            play.cache_bust_tokens.clone(),
        );
        let replay_hashes = self.include_replay_hashes.then(|| {
            ReplayRequestHashes::from_tokens(&request.prompt_tokens(), self.engine_block_size)
        });
        Ok(AgentxReadyRequest {
            request,
            arrival_time_ms: now_ms,
            replay_hashes,
            session_id: context.play_id.clone(),
            turn_index,
            context,
        })
    }

    fn complete_graph_node(
        &mut self,
        lane_id: usize,
        request_index: usize,
        now_ms: f64,
        phase: AgentxPhase,
    ) -> Result<()> {
        let tree_index = self.lanes[lane_id].play.tree_index;
        let dependents = self.corpus[tree_index].dependents[request_index].clone();
        let source_origin_ms = self.lanes[lane_id].play.source_origin_ms;
        {
            let play = &mut self.lanes[lane_id].play;
            ensure!(
                play.nodes[request_index].status == NodeStatus::InFlight,
                "AgentX node terminal callback does not match an in-flight node"
            );
            play.nodes[request_index].status = NodeStatus::Complete;
            play.in_flight = play.in_flight.saturating_sub(1);
            for dependent in dependents {
                let node = &mut play.nodes[dependent.request_index];
                if node.status != NodeStatus::Pending || node.remaining_dependencies == 0 {
                    continue;
                }
                let ready_at_ms = if phase == AgentxPhase::Warmup {
                    (self.corpus[tree_index].requests[dependent.request_index].source_start_s
                        * 1000.0
                        - source_origin_ms)
                        .max(0.0)
                } else {
                    now_ms + dependent.replay_delay_ms
                };
                node.dependency_ready_at_ms =
                    max_time(node.dependency_ready_at_ms, Some(ready_at_ms));
                node.remaining_dependencies -= 1;
                update_node_ready_at(node);
                if phase == AgentxPhase::Profile && node.status == NodeStatus::Pending {
                    if let Some(ready_ms) = node.ready_at_ms {
                        self.ready_queue.push(ReadyNodeItem {
                            ready_at_ms: ready_ms,
                            lane_id,
                            play_index: play.play_index,
                            request_index: dependent.request_index,
                        });
                    }
                }
            }
        }

        if self.lanes[lane_id]
            .play
            .is_complete(&self.corpus[tree_index])
        {
            self.summary.event_counts.play_completed += 1;
            self.record_play_event(AgentxEventKind::PlayCompleted, lane_id, now_ms);
            if phase == AgentxPhase::Profile
                && self.phase == AgentxPhase::Profile
                && self
                    .profile_deadline_ms
                    .is_some_and(|deadline| now_ms < deadline)
            {
                self.recycle_lane(lane_id, now_ms)?;
            } else if phase == AgentxPhase::Warmup
                && self.lanes[lane_id].warmup_dispatched < self.config.warmup_requests_per_lane
            {
                self.recycle_lane(lane_id, now_ms)?;
            }
        }
        Ok(())
    }

    /// Release child-stream roots from the parent's actual dispatch time.
    /// This is deliberately separate from completion dependencies: SPAWN is
    /// start-anchored, whereas stream/JOIN edges are terminal-anchored.
    fn release_spawn_children(
        &mut self,
        lane_id: usize,
        request_index: usize,
        now_ms: f64,
        phase: AgentxPhase,
    ) {
        let tree_index = self.lanes[lane_id].play.tree_index;
        let spawns = self.corpus[tree_index].spawn_children[request_index].clone();
        let source_origin_ms = self.lanes[lane_id].play.source_origin_ms;
        let play = &mut self.lanes[lane_id].play;
        for spawn in spawns {
            let node = &mut play.nodes[spawn.request_index];
            if node.status != NodeStatus::Pending || !node.spawn_pending {
                continue;
            }
            node.spawn_pending = false;
            node.spawn_ready_at_ms = Some(if phase == AgentxPhase::Warmup {
                (self.corpus[tree_index].requests[spawn.request_index].source_start_s * 1000.0
                    - source_origin_ms)
                    .max(0.0)
            } else {
                now_ms + spawn.offset_ms
            });
            update_node_ready_at(node);
            if phase == AgentxPhase::Profile && node.status == NodeStatus::Pending {
                if let Some(ready_ms) = node.ready_at_ms {
                    self.ready_queue.push(ReadyNodeItem {
                        ready_at_ms: ready_ms,
                        lane_id,
                        play_index: play.play_index,
                        request_index: spawn.request_index,
                    });
                }
            }
        }
    }

    fn recycle_lane(&mut self, lane_id: usize, now_ms: f64) -> Result<()> {
        let play_index = self.lanes[lane_id].next_play_index;
        let tree_index = self.next_tree_cursor % self.corpus.len();
        self.next_tree_cursor = self.next_tree_cursor.saturating_add(1);
        let recycle_pass = next_recycle_pass(&mut self.recycle_pass_by_tree, tree_index);
        let base_ms = if self.phase == AgentxPhase::Profile {
            now_ms
        } else {
            0.0
        };
        self.lanes[lane_id].play = build_play(
            &self.corpus[tree_index],
            tree_index,
            lane_id,
            play_index,
            None,
            base_ms,
            &self.config.benchmark_id,
            recycle_pass,
            &self.marker_tokenizer,
        )?;
        self.lanes[lane_id].next_play_index += 1;
        self.summary.event_counts.play_recycled += 1;
        self.record_play_event(AgentxEventKind::PlayRecycled, lane_id, now_ms);
        self.record_play_event(AgentxEventKind::PlayStarted, lane_id, now_ms);
        if self.phase == AgentxPhase::Profile {
            let play = &self.lanes[lane_id].play;
            for (request_index, node) in play.nodes.iter().enumerate() {
                if node.status == NodeStatus::Pending {
                    if let Some(ready_at_ms) = node.ready_at_ms {
                        self.ready_queue.push(ReadyNodeItem {
                            ready_at_ms,
                            lane_id,
                            play_index: play.play_index,
                            request_index,
                        });
                    }
                }
            }
        }
        let marker_tokens = self.lanes[lane_id].play.cache_bust_tokens.len();
        self.summary.cache_bust_marker_token_count_min = self
            .summary
            .cache_bust_marker_token_count_min
            .min(marker_tokens);
        self.summary.cache_bust_marker_token_count_max = self
            .summary
            .cache_bust_marker_token_count_max
            .max(marker_tokens);
        Ok(())
    }

    fn apply_idle_guards(&mut self, now_ms: f64) {
        let tree_cap_ms = self.config.tree_idle_gap_cap_s * 1000.0;
        let mut shifted = false;
        for lane_id in 0..self.lanes.len() {
            let should_shift = {
                let lane = &self.lanes[lane_id];
                let play = &lane.play;
                play.in_flight == 0
                    && !play.has_ready_at(now_ms)
                    && now_ms >= lane.last_wire_activity_ms + tree_cap_ms
                    && play
                        .earliest_ready_ms()
                        .is_some_and(|ready_at| ready_at > now_ms)
            };
            if should_shift {
                let earliest = self.lanes[lane_id]
                    .play
                    .earliest_ready_ms()
                    .expect("tree guard requires a pending timer");
                self.lanes[lane_id]
                    .play
                    .shift_ready_times(earliest - now_ms);
                self.summary.event_counts.tree_idle_guard += 1;
                self.record_play_event(AgentxEventKind::TreeIdleGuard, lane_id, now_ms);
                shifted = true;
            }
        }

        let global_cap_ms = self.config.global_idle_gap_cap_s * 1000.0;
        if self.in_flight.is_empty()
            && !self.any_ready_at(now_ms)
            && now_ms >= self.last_global_wire_activity_ms + global_cap_ms
        {
            let earliest = self
                .lanes
                .iter()
                .filter_map(|lane| lane.play.earliest_ready_ms())
                .min_by(f64::total_cmp);
            if let Some(earliest) = earliest
                && earliest > now_ms
            {
                let delta = earliest - now_ms;
                for lane in &mut self.lanes {
                    lane.play.shift_ready_times(delta);
                }
                self.summary.event_counts.global_idle_guard += 1;
                self.record_simple_event(AgentxEventKind::GlobalIdleGuard, now_ms);
                shifted = true;
            }
        }
        if shifted {
            self.rebuild_ready_queue();
        }
    }

    fn rebuild_ready_queue(&mut self) {
        self.ready_queue.clear();
        if self.phase != AgentxPhase::Profile {
            return;
        }
        for lane_id in 0..self.lanes.len() {
            let play = &self.lanes[lane_id].play;
            for (request_index, node) in play.nodes.iter().enumerate() {
                if node.status == NodeStatus::Pending {
                    if let Some(ready_at_ms) = node.ready_at_ms {
                        self.ready_queue.push(ReadyNodeItem {
                            ready_at_ms,
                            lane_id,
                            play_index: play.play_index,
                            request_index,
                        });
                    }
                }
            }
        }
    }

    fn purge_stale_ready_queue_top(&mut self) {
        while let Some(top) = self.ready_queue.peek() {
            let lane = &self.lanes[top.lane_id];
            if lane.play.play_index != top.play_index {
                self.ready_queue.pop();
                continue;
            }
            let node = &lane.play.nodes[top.request_index];
            if node.status != NodeStatus::Pending
                || node
                    .ready_at_ms
                    .is_none_or(|t| (t - top.ready_at_ms).abs() > 1e-9)
            {
                self.ready_queue.pop();
                continue;
            }
            break;
        }
    }

    fn any_ready_pending(&self) -> bool {
        self.lanes.iter().any(|lane| {
            lane.play
                .nodes
                .iter()
                .any(|node| node.status == NodeStatus::Pending && node.ready_at_ms.is_some())
        })
    }

    fn any_ready_at(&self, now_ms: f64) -> bool {
        self.lanes.iter().any(|lane| lane.play.has_ready_at(now_ms))
    }

    fn phase_counts_mut(&mut self, phase: AgentxPhase) -> &mut AgentxPhaseRequestCounts {
        match phase {
            AgentxPhase::Primer => &mut self.summary.primer,
            AgentxPhase::Warmup => &mut self.summary.warmup,
            AgentxPhase::Profile => &mut self.summary.profile,
            AgentxPhase::Drain | AgentxPhase::Complete => &mut self.summary.profile,
        }
    }

    fn record_request_event(
        &mut self,
        kind: AgentxEventKind,
        context: &AgentxRequestContext,
        request_uuid: Uuid,
        now_ms: f64,
    ) {
        if !self.config.capture_events {
            return;
        }
        self.summary.events.push(AgentxEvent {
            kind,
            phase: context.phase,
            logical_time_ms: now_ms,
            lane_id: Some(context.lane_id),
            play_id: Some(context.play_id.clone()),
            source_tree_id: Some(context.source_tree_id.clone()),
            stream_id: Some(context.stream_id.clone()),
            source_request_id: Some(context.source_request_id.clone()),
            request_uuid: Some(request_uuid.to_string()),
            cache_bust_id: Some(context.cache_bust_id.clone()),
        });
    }

    fn record_play_event(&mut self, kind: AgentxEventKind, lane_id: usize, now_ms: f64) {
        if kind == AgentxEventKind::PlayStarted {
            self.summary.event_counts.play_started += 1;
        }
        if !self.config.capture_events {
            return;
        }
        let play = &self.lanes[lane_id].play;
        let tree = &self.corpus[play.tree_index].source;
        self.summary.events.push(AgentxEvent {
            kind,
            phase: self.phase,
            logical_time_ms: now_ms,
            lane_id: Some(lane_id),
            play_id: Some(play.play_id.clone()),
            source_tree_id: Some(tree.tree_id.clone()),
            stream_id: None,
            source_request_id: None,
            request_uuid: None,
            cache_bust_id: Some(play.cache_bust_id.clone()),
        });
    }

    fn record_simple_event(&mut self, kind: AgentxEventKind, now_ms: f64) {
        if !self.config.capture_events {
            return;
        }
        self.summary.events.push(AgentxEvent {
            kind,
            phase: self.phase,
            logical_time_ms: now_ms,
            lane_id: None,
            play_id: None,
            source_tree_id: None,
            stream_id: None,
            source_request_id: None,
            request_uuid: None,
            cache_bust_id: None,
        });
    }
}

fn compile_tree(
    mut source: AgentxExactTreeRecord,
    tree_ordinal: usize,
    next_global_hash_id: &mut u64,
    mapping_hasher: &mut Sha256,
) -> Result<CompiledTree> {
    let mut requests = source
        .streams
        .iter()
        .flat_map(|stream| stream.requests.iter().cloned())
        .collect::<Vec<_>>();
    globalize_tree_hash_ids(
        &source,
        tree_ordinal,
        &mut requests,
        next_global_hash_id,
        mapping_hasher,
    )?;
    let index = requests
        .iter()
        .enumerate()
        .map(|(index, request)| (request.request_id.as_str(), index))
        .collect::<HashMap<_, _>>();
    let dependencies = requests
        .iter()
        .map(|request| {
            request
                .dependencies
                .iter()
                .map(|dependency| {
                    Ok(CompiledDependency {
                        request_index: index
                            .get(dependency.request_id.as_str())
                            .copied()
                            .ok_or_else(|| {
                                anyhow!(
                                    "unknown dependency {} in tree {}",
                                    dependency.request_id,
                                    source.tree_id
                                )
                            })?,
                        replay_delay_ms: dependency.replay_delay_after_dependency_s * 1000.0,
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let mut dependents = vec![Vec::new(); requests.len()];
    for (request_index, request_dependencies) in dependencies.iter().enumerate() {
        for dependency in request_dependencies {
            dependents[dependency.request_index].push(CompiledDependent {
                request_index,
                replay_delay_ms: dependency.replay_delay_ms,
            });
        }
    }
    let stream_by_id = source
        .streams
        .iter()
        .map(|stream| (stream.stream_id.as_str(), stream))
        .collect::<HashMap<_, _>>();
    let mut spawn_children = vec![Vec::new(); requests.len()];
    let mut spawned_by = vec![None; requests.len()];
    for branch in &source.branches {
        let spawn_index = index[branch.spawn_request_id.as_str()];
        let spawn_start_s = requests[spawn_index].source_start_s;
        for child_stream_id in &branch.child_stream_ids {
            let child = stream_by_id[child_stream_id.as_str()];
            let child_request = child.requests.first().expect("validated non-empty stream");
            let child_index = index[child_request.request_id.as_str()];
            ensure!(
                spawned_by[child_index].replace(spawn_index).is_none(),
                "tree {} child request {} has multiple SPAWN owners",
                source.tree_id,
                child_request.request_id
            );
            spawn_children[spawn_index].push(CompiledSpawn {
                request_index: child_index,
                // A worker group may contain children with different source
                // starts. Deriving each offset preserves those differences;
                // branch.spawn_offset is the earliest group-level SPAWN.
                offset_ms: (child_request.source_start_s - spawn_start_s).max(0.0) * 1000.0,
            });
        }
    }
    let terminal_indexes = source
        .tree_terminal_request_ids
        .iter()
        .map(|terminal| index[terminal.as_str()])
        .collect::<Vec<_>>();
    // The compiled request vector is authoritative at runtime. Drop the
    // nested source copies (especially hash lineage) while retaining stream
    // and branch metadata needed for evidence; otherwise a cached corpus
    // would hold the ~GB-scale prompt hashes twice.
    for stream in &mut source.streams {
        stream.requests.clear();
        stream.requests.shrink_to_fit();
    }

    Ok(CompiledTree {
        source,
        requests,
        dependencies,
        dependents,
        spawn_children,
        spawned_by,
        terminal_indexes,
    })
}

fn globalize_tree_hash_ids(
    source: &AgentxExactTreeRecord,
    tree_ordinal: usize,
    requests: &mut [AgentxRequest],
    next_global_hash_id: &mut u64,
    mapping_hasher: &mut Sha256,
) -> Result<()> {
    // A long workflow repeats its prefix lineage in many requests. Deduplicate
    // while scanning so globalization uses O(unique ids in one tree) scratch
    // space instead of copying every repeated lineage entry.
    let mut unique_local_ids = HashSet::new();
    for request in requests.iter() {
        unique_local_ids.extend(request.hash_ids.iter().copied());
    }
    let mut local_ids = unique_local_ids.into_iter().collect::<Vec<_>>();
    local_ids.sort_unstable();

    mapping_hasher.update((tree_ordinal as u64).to_le_bytes());
    mapping_hasher.update((source.tree_id.len() as u64).to_le_bytes());
    mapping_hasher.update(source.tree_id.as_bytes());
    mapping_hasher.update((local_ids.len() as u64).to_le_bytes());

    let mut mapping = HashMap::with_capacity(local_ids.len());
    for local_id in local_ids {
        let global_id = u32::try_from(*next_global_hash_id).with_context(|| {
            format!(
                "AgentX global hash namespace exceeds u32 while compiling tree {}",
                source.tree_id
            )
        })?;
        mapping_hasher.update(local_id.to_le_bytes());
        mapping_hasher.update(global_id.to_le_bytes());
        ensure!(
            mapping.insert(local_id, global_id).is_none(),
            "duplicate local hash id after deduplication in tree {}",
            source.tree_id
        );
        *next_global_hash_id = next_global_hash_id
            .checked_add(1)
            .ok_or_else(|| anyhow!("AgentX global hash id count overflow"))?;
    }

    for request in requests {
        for hash_id in &mut request.hash_ids {
            *hash_id = mapping[hash_id];
        }
    }
    Ok(())
}

fn build_play(
    tree: &CompiledTree,
    tree_index: usize,
    lane_id: usize,
    play_index: usize,
    snapshot_ms: Option<f64>,
    base_ms: f64,
    benchmark_id: &str,
    recycle_pass: usize,
    marker_tokenizer: &MarkerTokenizer,
) -> Result<ActivePlay> {
    let historical = tree
        .requests
        .iter()
        .map(|request| {
            snapshot_ms.is_some_and(|snapshot| request.source_start_s * 1000.0 < snapshot)
        })
        .collect::<Vec<_>>();
    let source_origin_ms = snapshot_ms.unwrap_or_else(|| {
        tree.requests
            .iter()
            .map(|request| request.source_start_s * 1000.0)
            .min_by(f64::total_cmp)
            .unwrap_or(0.0)
    });
    let nodes = tree
        .requests
        .iter()
        .enumerate()
        .map(|(request_index, request)| {
            if historical[request_index] {
                return NodeRuntime {
                    status: NodeStatus::Historical,
                    remaining_dependencies: 0,
                    dependency_ready_at_ms: None,
                    spawn_pending: false,
                    spawn_ready_at_ms: None,
                    ready_at_ms: None,
                };
            }
            let remaining_dependencies = tree.dependencies[request_index]
                .iter()
                .filter(|dependency| !historical[dependency.request_index])
                .count();
            let has_historical_dependency = tree.dependencies[request_index]
                .iter()
                .any(|dependency| historical[dependency.request_index]);
            let source_residual_ms =
                base_ms + (request.source_start_s * 1000.0 - source_origin_ms).max(0.0);
            let spawn_pending =
                tree.spawned_by[request_index].is_some_and(|spawn_index| !historical[spawn_index]);
            let spawn_ready_at_ms = tree.spawned_by[request_index]
                .is_some_and(|spawn_index| historical[spawn_index])
                .then_some(source_residual_ms);
            let dependency_ready_at_ms = has_historical_dependency.then_some(source_residual_ms);
            let ready_at_ms = (remaining_dependencies == 0 && !spawn_pending).then_some(
                max_time(dependency_ready_at_ms, spawn_ready_at_ms).unwrap_or(source_residual_ms),
            );
            NodeRuntime {
                status: NodeStatus::Pending,
                remaining_dependencies,
                dependency_ready_at_ms,
                spawn_pending,
                spawn_ready_at_ms,
                ready_at_ms,
            }
        })
        .collect();
    let play_id = format!("{benchmark_id}:lane-{lane_id}:play-{play_index}");
    let cache_bust_id =
        cache_bust_marker(benchmark_id, recycle_pass, lane_id, &tree.source.tree_id);
    let cache_bust_tokens = marker_tokenizer.encode(&cache_bust_id)?;
    ensure!(
        !cache_bust_tokens.is_empty(),
        "cache-bust marker tokenized to an empty prefix"
    );
    Ok(ActivePlay {
        lane_id,
        play_index,
        tree_index,
        cache_bust_id,
        cache_bust_tokens,
        source_origin_ms,
        play_id,
        nodes,
        in_flight: 0,
    })
}

fn anchor_play_ready_times(
    play: &mut ActivePlay,
    profile_start_ms: f64,
    global_min_offset_ms: f64,
) {
    for node in &mut play.nodes {
        if let Some(ready_at_ms) = node.ready_at_ms.as_mut() {
            *ready_at_ms = profile_start_ms + (*ready_at_ms - global_min_offset_ms).max(0.0);
        }
        if let Some(ready_at_ms) = node.dependency_ready_at_ms.as_mut() {
            *ready_at_ms = profile_start_ms + (*ready_at_ms - global_min_offset_ms).max(0.0);
        }
        if let Some(ready_at_ms) = node.spawn_ready_at_ms.as_mut() {
            *ready_at_ms = profile_start_ms + (*ready_at_ms - global_min_offset_ms).max(0.0);
        }
    }
}

fn update_node_ready_at(node: &mut NodeRuntime) {
    if node.status == NodeStatus::Pending && node.remaining_dependencies == 0 && !node.spawn_pending
    {
        node.ready_at_ms = max_time(node.dependency_ready_at_ms, node.spawn_ready_at_ms);
    }
}

fn request_context(
    play: &ActivePlay,
    tree: &AgentxExactTreeRecord,
    request: &AgentxRequest,
    phase: AgentxPhase,
) -> AgentxRequestContext {
    AgentxRequestContext {
        phase,
        measurement_eligible: phase == AgentxPhase::Profile,
        lane_id: play.lane_id,
        play_index: play.play_index,
        play_id: play.play_id.clone(),
        source_tree_id: tree.tree_id.clone(),
        stream_id: request.stream_id.clone(),
        source_request_id: request.request_id.clone(),
        cache_bust_id: play.cache_bust_id.clone(),
    }
}

fn cache_bust_marker(
    benchmark_id: &str,
    recycle_pass: usize,
    lane_id: usize,
    trace_id: &str,
) -> String {
    let identity = format!("{benchmark_id}:{recycle_pass}:{lane_id}:{trace_id}");
    let digest = Sha256::digest(identity.as_bytes());
    let hex = digest[..6]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("[rid:{hex}]\n\n")
}

fn synthetic_marker_tokens(marker: &str, count: usize) -> Vec<u32> {
    (0..count)
        .map(|index| {
            let digest = Sha256::digest(format!("{marker}:{index}").as_bytes());
            let mut bytes = [0_u8; 4];
            bytes.copy_from_slice(&digest[..4]);
            u32::from_be_bytes(bytes) | 0x8000_0000
        })
        .collect()
}

fn next_recycle_pass(recycle_pass_by_tree: &mut HashMap<usize, usize>, tree_index: usize) -> usize {
    let next = recycle_pass_by_tree.entry(tree_index).or_insert(0);
    let current = *next;
    *next = next.saturating_add(1);
    current
}

fn warmup_node_dispatchable(node: &NodeRuntime) -> bool {
    node.status == NodeStatus::Pending && node.remaining_dependencies == 0 && !node.spawn_pending
}

fn stable_uuid(value: &str) -> Uuid {
    let high = stable_hash64(value.as_bytes());
    let mut salted = Vec::with_capacity(value.len() + 1);
    salted.extend_from_slice(value.as_bytes());
    salted.push(0xA5);
    let low = stable_hash64(&salted);
    Uuid::from_u128((u128::from(high) << 64) | u128::from(low))
}

fn stable_hash64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

fn sha256_file(path: &PathBuf) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open tokenizer {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read tokenizer {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn min_time(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn max_time(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

#[cfg(test)]
pub(crate) fn test_agentx_corpus() -> AgentxExactCorpus {
    let tree: AgentxExactTreeRecord = serde_json::from_str(
        r#"{
        "schema_version": "agentx_exact_v1",
        "tree_id": "tree",
        "source_workflow_id": "workflow",
        "source_index": 0,
        "hash_namespace": "tree",
        "hash_id_scope": "local",
        "trace_hash_block_size": 4,
        "models": ["model"],
        "source_start_s": 0.0,
        "source_last_request_start_s": 0.5,
        "source_end_s": 0.6,
        "source_duration_s": 0.6,
        "root_stream_id": "root",
        "streams": [{
            "stream_id": "root",
            "role": "root",
            "parent_stream_id": null,
            "agent_depth": 0,
            "source_agent_id": null,
            "source_agent_path": [],
            "detected_chain_index": 0,
            "fork_prefix_blocks": null,
            "worker_group": null,
            "source_start_s": 0.0,
            "source_end_s": 0.6,
            "requests": [{
                "request_id": "a",
                "stream_id": "root",
                "stream_turn_index": 0,
                "source_index_path": [0],
                "source_outer_index": 0,
                "source_inner_index": null,
                "source_type": "s",
                "source_model": "model",
                "source_t_s": 0.0,
                "source_start_s": 0.0,
                "source_api_time_s": 0.1,
                "source_end_s": 0.1,
                "source_think_time_s": null,
                "source_ttft_s": 0.01,
                "input_length": 4,
                "output_length": 1,
                "replay_max_output_tokens": 1,
                "hash_ids": [1],
                "dependencies": [],
                "spawn_branch_ids": [],
                "join_branch_ids": [],
                "is_stream_start": true,
                "is_stream_terminal": false
            }, {
                "request_id": "b",
                "stream_id": "root",
                "stream_turn_index": 1,
                "source_index_path": [1],
                "source_outer_index": 1,
                "source_inner_index": null,
                "source_type": "s",
                "source_model": "model",
                "source_t_s": 0.5,
                "source_start_s": 0.5,
                "source_api_time_s": 0.1,
                "source_end_s": 0.6,
                "source_think_time_s": 0.4,
                "source_ttft_s": 0.01,
                "input_length": 8,
                "output_length": 1,
                "replay_max_output_tokens": 1,
                "hash_ids": [1, 2],
                "dependencies": [{
                    "request_id": "a",
                    "kind": "stream",
                    "source_end_to_start_gap_s": 0.4,
                    "replay_delay_after_dependency_s": 0.4
                }],
                "spawn_branch_ids": [],
                "join_branch_ids": [],
                "is_stream_start": false,
                "is_stream_terminal": true
            }],
            "terminal_request_id": "b"
        }],
        "branches": [],
        "tree_terminal_request_ids": ["b"],
        "compile_evidence": {
            "source_requests": 2,
            "zero_output_requests_replayed_as_one_token": 0,
            "source_explicit_subagents": 0,
            "preamble_requests": 0,
            "chain_seams_merged": 0,
            "unclassified_empty_hash_requests": 0,
            "subagent_recorded_end_before_descendant": 0,
            "hash_ids_preserved": true,
            "token_lengths_preserved": true,
            "source_timestamps_preserved": true,
            "source_api_times_preserved": true,
            "source_think_times_preserved_as_metadata": true,
            "delay_cap_applied": false
        }
    }"#,
    )
    .expect("test AgentX tree schema must deserialize");
    AgentxExactCorpus {
        trace_block_size: 4,
        trees: vec![tree],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> AgentxExactCorpus {
        test_agentx_corpus()
    }

    fn config() -> AgentxReplayConfig {
        AgentxReplayConfig {
            session_tree_concurrency: 1,
            trajectory_start_min_ratio: 0.25,
            trajectory_start_max_ratio: 0.25,
            warmup_requests_per_lane: 1,
            profile_duration_s: 1.0,
            tree_idle_gap_cap_s: 0.1,
            global_idle_gap_cap_s: 0.05,
            drain_grace_s: 1.0,
            benchmark_id: "test".to_string(),
            ..Default::default()
        }
    }

    fn branching_corpus() -> AgentxExactCorpus {
        fn request(
            request_id: &str,
            stream_id: &str,
            turn: usize,
            source_index: usize,
            start_s: f64,
            end_s: f64,
            input_length: usize,
            hash_ids: Vec<u32>,
        ) -> AgentxRequest {
            AgentxRequest {
                request_id: request_id.to_string(),
                stream_id: stream_id.to_string(),
                stream_turn_index: turn,
                source_index_path: vec![source_index],
                source_outer_index: source_index,
                source_inner_index: None,
                source_type: "s".to_string(),
                source_model: "model".to_string(),
                source_t_s: start_s,
                source_start_s: start_s,
                source_api_time_s: end_s - start_s,
                source_end_s: end_s,
                source_think_time_s: None,
                source_ttft_s: Some(0.01),
                input_length,
                output_length: 1,
                replay_max_output_tokens: 1,
                hash_ids,
                dependencies: Vec::new(),
                spawn_branch_ids: Vec::new(),
                join_branch_ids: Vec::new(),
                is_stream_start: turn == 0,
                is_stream_terminal: true,
            }
        }

        let mut root_a = request("root-a", "root", 0, 0, 0.0, 0.1, 4, vec![1]);
        root_a.is_stream_terminal = false;
        root_a.spawn_branch_ids.push("joined".to_string());
        let mut root_b = request("root-b", "root", 1, 1, 0.2, 0.3, 8, vec![1, 2]);
        root_b.dependencies = vec![
            AgentxDependency {
                request_id: "root-a".to_string(),
                kind: AgentxDependencyKind::Stream,
                source_end_to_start_gap_s: 0.1,
                replay_delay_after_dependency_s: 0.1,
            },
            AgentxDependency {
                request_id: "child-terminal".to_string(),
                kind: AgentxDependencyKind::Join,
                source_end_to_start_gap_s: 0.05,
                replay_delay_after_dependency_s: 0.05,
            },
        ];
        root_b.join_branch_ids.push("joined".to_string());
        root_b.spawn_branch_ids.push("background".to_string());
        let child = request("child-terminal", "child", 0, 2, 0.05, 0.15, 8, vec![1, 3]);
        let background = request(
            "background-terminal",
            "background-stream",
            0,
            3,
            0.25,
            0.4,
            8,
            vec![1, 4],
        );
        let tree = AgentxExactTreeRecord {
            schema_version: AGENTX_EXACT_SCHEMA_VERSION.to_string(),
            tree_id: "branching-tree".to_string(),
            source_workflow_id: "branching-workflow".to_string(),
            source_index: 0,
            hash_namespace: "branching-tree".to_string(),
            hash_id_scope: "local".to_string(),
            trace_hash_block_size: 4,
            models: vec!["model".to_string()],
            source_start_s: 0.0,
            source_last_request_start_s: 0.25,
            source_end_s: 0.4,
            source_duration_s: 0.4,
            root_stream_id: "root".to_string(),
            streams: vec![
                AgentxStream {
                    stream_id: "root".to_string(),
                    role: "root".to_string(),
                    parent_stream_id: None,
                    agent_depth: 0,
                    source_agent_id: None,
                    source_agent_path: Vec::new(),
                    detected_chain_index: 0,
                    fork_prefix_blocks: None,
                    worker_group: None,
                    source_start_s: 0.0,
                    source_end_s: 0.3,
                    requests: vec![root_a, root_b],
                    terminal_request_id: "root-b".to_string(),
                },
                AgentxStream {
                    stream_id: "child".to_string(),
                    role: "auxiliary".to_string(),
                    parent_stream_id: Some("root".to_string()),
                    agent_depth: 1,
                    source_agent_id: None,
                    source_agent_path: Vec::new(),
                    detected_chain_index: 1,
                    fork_prefix_blocks: Some(1),
                    worker_group: None,
                    source_start_s: 0.05,
                    source_end_s: 0.15,
                    requests: vec![child],
                    terminal_request_id: "child-terminal".to_string(),
                },
                AgentxStream {
                    stream_id: "background-stream".to_string(),
                    role: "auxiliary".to_string(),
                    parent_stream_id: Some("root".to_string()),
                    agent_depth: 1,
                    source_agent_id: None,
                    source_agent_path: Vec::new(),
                    detected_chain_index: 2,
                    fork_prefix_blocks: Some(1),
                    worker_group: None,
                    source_start_s: 0.25,
                    source_end_s: 0.4,
                    requests: vec![background],
                    terminal_request_id: "background-terminal".to_string(),
                },
            ],
            branches: vec![
                AgentxBranch {
                    branch_id: "joined".to_string(),
                    branch_kind: AgentxBranchKind::Spawn,
                    source_kind: "detected_flat_chain".to_string(),
                    parent_stream_id: "root".to_string(),
                    spawn_request_id: "root-a".to_string(),
                    source_start_s: 0.05,
                    source_recorded_end_values_s: vec![0.15],
                    spawn_offset_from_parent_start_s: 0.05,
                    source_parent_end_to_branch_start_gap_s: -0.05,
                    child_stream_ids: vec!["child".to_string()],
                    descendant_stream_ids: vec!["child".to_string()],
                    join_request_id: Some("root-b".to_string()),
                    is_background: false,
                    terminal_request_ids: vec!["child-terminal".to_string()],
                    source_refs: Vec::new(),
                },
                AgentxBranch {
                    branch_id: "background".to_string(),
                    branch_kind: AgentxBranchKind::Spawn,
                    source_kind: "detected_flat_chain".to_string(),
                    parent_stream_id: "root".to_string(),
                    spawn_request_id: "root-b".to_string(),
                    source_start_s: 0.25,
                    source_recorded_end_values_s: vec![0.4],
                    spawn_offset_from_parent_start_s: 0.05,
                    source_parent_end_to_branch_start_gap_s: -0.05,
                    child_stream_ids: vec!["background-stream".to_string()],
                    descendant_stream_ids: vec!["background-stream".to_string()],
                    join_request_id: None,
                    is_background: true,
                    terminal_request_ids: vec!["background-terminal".to_string()],
                    source_refs: Vec::new(),
                },
            ],
            tree_terminal_request_ids: vec![
                "root-b".to_string(),
                "child-terminal".to_string(),
                "background-terminal".to_string(),
            ],
            compile_evidence: AgentxCompileEvidence {
                source_requests: 4,
                zero_output_requests_replayed_as_one_token: 0,
                source_explicit_subagents: 0,
                preamble_requests: 0,
                chain_seams_merged: 0,
                unclassified_empty_hash_requests: 0,
                subagent_recorded_end_before_descendant: 0,
                hash_ids_preserved: true,
                token_lengths_preserved: true,
                source_timestamps_preserved: true,
                source_api_times_preserved: true,
                source_think_times_preserved_as_metadata: true,
                delay_cap_applied: false,
            },
        };
        AgentxExactCorpus {
            trace_block_size: 4,
            trees: vec![tree],
        }
    }

    #[test]
    fn corpus_rejects_dependency_cycles() {
        let mut corpus = corpus();
        corpus.trees[0].streams[0].requests[0]
            .dependencies
            .push(AgentxDependency {
                request_id: "b".to_string(),
                kind: AgentxDependencyKind::Stream,
                source_end_to_start_gap_s: -0.6,
                replay_delay_after_dependency_s: 0.0,
            });
        assert!(corpus.validate().unwrap_err().to_string().contains("cycle"));
    }

    #[test]
    fn primer_warmup_barrier_then_profile() {
        let mut controller = AgentxReplayController::new(corpus(), config(), 4, true).unwrap();
        let primer = controller.drain_ready(0.0).unwrap();
        assert_eq!(primer.len(), 1);
        assert_eq!(primer[0].context.phase, AgentxPhase::Primer);
        let primer_output_ids = primer[0]
            .request
            .metadata()
            .output_token_ids
            .clone()
            .expect("AgentX synthetic output tokens must be preplanned");
        assert_eq!(primer_output_ids.len(), 1);
        let mut twin = AgentxReplayController::new(corpus(), config(), 4, true).unwrap();
        let twin_primer = twin.drain_ready(0.0).unwrap();
        assert_eq!(
            twin_primer[0].request.metadata().output_token_ids,
            Some(primer_output_ids)
        );
        let primer_uuid = primer[0].request.metadata().uuid.unwrap();
        controller.on_terminal(primer_uuid, 1.0, false).unwrap();
        let warmup = controller.drain_ready(1.0).unwrap();
        assert_eq!(warmup.len(), 1);
        assert_eq!(warmup[0].context.phase, AgentxPhase::Warmup);
        let warmup_uuid = warmup[0].request.metadata().uuid.unwrap();
        controller.on_terminal(warmup_uuid, 2.0, false).unwrap();
        assert_eq!(controller.phase, AgentxPhase::Profile);
        assert_eq!(controller.summary.profile_start_ms, Some(2.0));
    }

    #[test]
    fn pinned_aiperf_phase_oracle_matches_normalized_events() {
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "fixtures/agentx_aiperf_56a0cf70_phase_oracle.json"
        ))
        .unwrap();
        assert_eq!(oracle["commit"], "56a0cf70f4c0359454ee4bd15a17770b541a3e3e");

        let mut protocol = config();
        protocol.capture_events = true;
        let mut controller = AgentxReplayController::new(corpus(), protocol, 4, true).unwrap();
        let primer = controller.drain_ready(0.0).unwrap().remove(0);
        let primer_marker = primer.context.cache_bust_id.clone();
        controller
            .on_terminal(primer.request.metadata().uuid.unwrap(), 1.0, false)
            .unwrap();
        let warmup = controller.drain_ready(1.0).unwrap().remove(0);
        assert_eq!(warmup.context.cache_bust_id, primer_marker);
        controller
            .on_terminal(warmup.request.metadata().uuid.unwrap(), 2.0, false)
            .unwrap();

        let normalized = controller
            .summary
            .events
            .iter()
            .map(|event| {
                serde_json::json!({
                    "kind": event.kind,
                    "phase": event.phase,
                    "logical_time_ms": event.logical_time_ms,
                    "source_request_id": event.source_request_id,
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            serde_json::Value::Array(normalized),
            oracle["normalized_events"]
        );
    }

    #[test]
    fn cache_bust_is_stable_inside_play_and_changes_across_plays() {
        let compiled = corpus().compile().unwrap();
        let tree = compiled.trees[0].clone();
        let marker_tokenizer = MarkerTokenizer::Synthetic { token_count: 3 };
        let first = build_play(&tree, 0, 0, 0, None, 0.0, "bench", 0, &marker_tokenizer).unwrap();
        let second = build_play(&tree, 0, 0, 1, None, 0.0, "bench", 1, &marker_tokenizer).unwrap();
        assert_eq!(first.cache_bust_tokens, first.cache_bust_tokens);
        assert_ne!(first.cache_bust_id, second.cache_bust_id);
        assert_ne!(first.cache_bust_tokens, second.cache_bust_tokens);
        assert_eq!(first.cache_bust_tokens.len(), 3);
        assert!(first.cache_bust_id.starts_with("[rid:"));
        assert!(first.cache_bust_id.ends_with("]\n\n"));
    }

    #[test]
    fn corpus_compile_globalizes_equal_local_hash_ids_across_workflows() {
        let mut source = corpus();
        let mut second = source.trees[0].clone();
        second.tree_id = "tree-1".to_string();
        second.source_workflow_id = "workflow-1".to_string();
        second.source_index = 1;
        second.hash_namespace = "tree-1".to_string();
        source.trees.push(second);

        let compiled = source.clone().compile().unwrap();
        let repeated_in_first = compiled.trees[0].requests[0].hash_ids[0];
        assert_eq!(repeated_in_first, compiled.trees[0].requests[1].hash_ids[0]);
        assert_ne!(
            repeated_in_first, compiled.trees[1].requests[0].hash_ids[0],
            "equal source-local ids from different workflows must not share runtime identity"
        );
        assert_eq!(compiled.hash_globalization.global_unique_hash_ids, 4);
        assert_eq!(
            compiled.hash_globalization.runtime_hash_id_scope,
            "global_dense_u32"
        );
        assert!(compiled.hash_globalization.collision_free);

        let left_tokens = crate::loadgen::trace::synthesize_trace_tokens(
            compiled.trees[0].requests[1].input_length,
            &compiled.trees[0].requests[1].hash_ids,
            compiled.trace_block_size,
        )
        .unwrap();
        let right_tokens = crate::loadgen::trace::synthesize_trace_tokens(
            compiled.trees[1].requests[1].input_length,
            &compiled.trees[1].requests[1].hash_ids,
            compiled.trace_block_size,
        )
        .unwrap();
        let left_hashes = ReplayRequestHashes::from_tokens(&left_tokens, 4);
        let right_hashes = ReplayRequestHashes::from_tokens(&right_tokens, 4);
        assert!(
            left_hashes
                .local_block_hashes
                .iter()
                .all(|hash| !right_hashes.local_block_hashes.contains(hash)),
            "KV-router local overlap hashes must be disjoint across workflows"
        );
        assert!(
            left_hashes
                .sequence_hashes
                .iter()
                .all(|hash| !right_hashes.sequence_hashes.contains(hash)),
            "KV cache sequence identities must be disjoint across workflows"
        );

        let recompiled = source.compile().unwrap();
        assert_eq!(
            compiled.hash_globalization, recompiled.hash_globalization,
            "global mapping and provenance must be deterministic"
        );
    }

    #[test]
    fn initial_snapshot_is_deterministic() {
        let left = AgentxReplayController::new(corpus(), config(), 4, false).unwrap();
        let right = AgentxReplayController::new(corpus(), config(), 4, false).unwrap();
        assert_eq!(
            left.summary.sampled_start_ratios,
            right.summary.sampled_start_ratios
        );
        assert_eq!(left.summary.sampled_start_ratios, vec![0.25]);
        assert_eq!(left.summary.initial_ratio_sampler, "fixed_ratio_no_rng");
    }

    #[test]
    fn initial_sampler_wraps_and_mints_lane_unique_markers() {
        let mut protocol = config();
        protocol.session_tree_concurrency = 3;
        protocol.initial_start_ratios = Some(vec![0.25, 0.25, 0.25]);
        protocol.initial_tree_ids = Some(vec![
            "workflow-0".to_string(),
            "workflow-1".to_string(),
            "workflow-0".to_string(),
        ]);
        let mut second = corpus().trees.remove(0);
        second.tree_id = "tree-1".to_string();
        second.source_workflow_id = "workflow-1".to_string();
        second.hash_namespace = "tree-1".to_string();
        let mut first = corpus().trees.remove(0);
        first.source_workflow_id = "workflow-0".to_string();
        let corpus = AgentxExactCorpus {
            trace_block_size: 4,
            trees: vec![first, second],
        };
        let controller = AgentxReplayController::new(corpus, protocol, 4, false).unwrap();
        assert_eq!(
            controller.summary.initial_tree_ids,
            vec!["workflow-0", "workflow-1", "workflow-0"]
        );
        assert_ne!(
            controller.lanes[0].play.cache_bust_id,
            controller.lanes[2].play.cache_bust_id
        );
        assert_eq!(controller.next_tree_cursor, 3);
    }

    #[test]
    fn primers_are_globally_snapshot_aligned_not_burst_dispatched() {
        let mut protocol = config();
        protocol.session_tree_concurrency = 2;
        protocol.trajectory_start_min_ratio = 0.25;
        protocol.trajectory_start_max_ratio = 0.75;
        protocol.initial_start_ratios = Some(vec![0.25, 0.75]);
        protocol.initial_tree_ids = Some(vec!["workflow".to_string(); 2]);
        protocol.global_idle_gap_cap_s = 10.0;
        let mut source = corpus();
        source.trees[0].source_workflow_id = "workflow".to_string();
        let mut controller = AgentxReplayController::new(source, protocol, 4, true).unwrap();

        let early = controller.drain_ready(0.0).unwrap();
        assert_eq!(early.len(), 1);
        assert_eq!(early[0].context.lane_id, 1);
        assert!(controller.drain_ready(249.999).unwrap().is_empty());
        let late = controller.drain_ready(250.0).unwrap();
        assert_eq!(late.len(), 1);
        assert_eq!(late[0].context.lane_id, 0);
    }

    #[test]
    fn tree_fanout_can_exceed_session_tree_concurrency() {
        let mut protocol = config();
        protocol.trajectory_start_min_ratio = 0.0;
        protocol.trajectory_start_max_ratio = 0.0;
        protocol.profile_duration_s = 10.0;
        let mut controller =
            AgentxReplayController::new(branching_corpus(), protocol, 4, true).unwrap();
        controller.phase = AgentxPhase::Profile;
        controller.summary.phase = AgentxPhase::Profile;
        controller.profile_start_ms = Some(0.0);
        controller.profile_deadline_ms = Some(10_000.0);
        controller.primers.clear();
        controller.in_flight.clear();
        controller.lanes[0].play = build_play(
            &controller.corpus[0],
            0,
            0,
            0,
            None,
            0.0,
            "fanout",
            0,
            &MarkerTokenizer::Synthetic { token_count: 1 },
        )
        .unwrap();

        let root = controller.drain_ready(0.0).unwrap();
        assert_eq!(root.len(), 1);
        assert_eq!(root[0].context.source_request_id, "root-a");
        let child = controller.drain_ready(50.0).unwrap();
        assert_eq!(child.len(), 1);
        assert_eq!(child[0].context.source_request_id, "child-terminal");
        assert_eq!(controller.config.session_tree_concurrency, 1);
        assert_eq!(controller.in_flight.len(), 2);
    }

    #[test]
    fn tree_idle_guard_fires_once_from_last_wire_activity() {
        let mut protocol = config();
        protocol.tree_idle_gap_cap_s = 0.1;
        protocol.global_idle_gap_cap_s = 10.0;
        let mut controller = AgentxReplayController::new(corpus(), protocol, 4, true).unwrap();
        controller.phase = AgentxPhase::Profile;
        controller.profile_start_ms = Some(0.0);
        controller.profile_deadline_ms = Some(10_000.0);
        controller.primers.clear();
        controller.in_flight.clear();
        controller.lanes[0].play.in_flight = 0;
        controller.lanes[0].last_wire_activity_ms = 0.0;
        controller.last_global_wire_activity_ms = 0.0;
        controller.lanes[0].play.nodes[0].status = NodeStatus::Complete;
        let future = &mut controller.lanes[0].play.nodes[1];
        future.status = NodeStatus::Pending;
        future.remaining_dependencies = 0;
        future.spawn_pending = false;
        future.dependency_ready_at_ms = Some(1_000.0);
        future.ready_at_ms = Some(1_000.0);

        assert_eq!(controller.next_ready_time_ms(), Some(100.0));
        controller.apply_idle_guards(99.0);
        assert_eq!(controller.lanes[0].play.earliest_ready_ms(), Some(1_000.0));
        controller.apply_idle_guards(100.0);
        assert_eq!(controller.lanes[0].play.earliest_ready_ms(), Some(100.0));
        assert_eq!(controller.summary.event_counts.tree_idle_guard, 1);
        assert_eq!(controller.summary.event_counts.global_idle_guard, 0);
    }

    #[test]
    fn global_idle_guard_preserves_relative_timer_spacing() {
        let mut protocol = config();
        protocol.tree_idle_gap_cap_s = 10.0;
        protocol.global_idle_gap_cap_s = 0.1;
        let mut controller =
            AgentxReplayController::new(branching_corpus(), protocol, 4, true).unwrap();
        controller.phase = AgentxPhase::Profile;
        controller.profile_start_ms = Some(0.0);
        controller.profile_deadline_ms = Some(10_000.0);
        controller.primers.clear();
        controller.in_flight.clear();
        controller.lanes[0].play.in_flight = 0;
        controller.lanes[0].last_wire_activity_ms = 0.0;
        controller.last_global_wire_activity_ms = 0.0;
        for node in &mut controller.lanes[0].play.nodes {
            node.status = NodeStatus::Complete;
            node.ready_at_ms = None;
        }
        for (request_index, ready_at_ms) in [(0, 1_000.0), (1, 1_250.0)] {
            let node = &mut controller.lanes[0].play.nodes[request_index];
            node.status = NodeStatus::Pending;
            node.remaining_dependencies = 0;
            node.spawn_pending = false;
            node.dependency_ready_at_ms = Some(ready_at_ms);
            node.ready_at_ms = Some(ready_at_ms);
        }

        assert_eq!(controller.next_ready_time_ms(), Some(100.0));
        controller.apply_idle_guards(100.0);
        let mut ready_times = controller.lanes[0]
            .play
            .nodes
            .iter()
            .filter_map(|node| node.ready_at_ms)
            .collect::<Vec<_>>();
        ready_times.sort_by(f64::total_cmp);
        assert_eq!(ready_times, vec![100.0, 350.0]);
        assert_eq!(controller.summary.event_counts.tree_idle_guard, 0);
        assert_eq!(controller.summary.event_counts.global_idle_guard, 1);
    }

    #[test]
    fn spawn_join_and_background_terminal_drive_the_lane() {
        let mut protocol = config();
        protocol.trajectory_start_min_ratio = 0.0;
        protocol.trajectory_start_max_ratio = 0.0;
        protocol.profile_duration_s = 10.0;
        protocol.tree_idle_gap_cap_s = 10.0;
        protocol.global_idle_gap_cap_s = 10.0;
        let mut controller =
            AgentxReplayController::new(branching_corpus(), protocol, 4, true).unwrap();

        // t*=0 has no historical request, so the lane correctly has no
        // mandatory primer and immediately spends its one-request warmup
        // quota on the real root DAG node.
        let warmup_root = controller.drain_ready(0.0).unwrap();
        assert_eq!(warmup_root.len(), 1);
        assert_eq!(warmup_root[0].context.phase, AgentxPhase::Warmup);
        assert_eq!(warmup_root[0].context.source_request_id, "root-a");
        controller
            .on_terminal(warmup_root[0].request.metadata().uuid.unwrap(), 0.5, false)
            .unwrap();
        // The child was start-anchored by the warmup root dispatch. The global
        // handoff minimum makes it the first profiling request at t=0.5.
        let child = controller.drain_ready(0.5).unwrap();
        assert_eq!(child[0].context.source_request_id, "child-terminal");
        controller
            .on_terminal(child[0].request.metadata().uuid.unwrap(), 1.0, false)
            .unwrap();

        // JOIN releases at max(child terminal + 50 ms, the authored root
        // stream deadline restored across the warmup barrier). The latter is
        // 150.5 ms after subtracting the one global handoff minimum.
        let root_b = controller.drain_ready(150.5).unwrap();
        assert_eq!(root_b[0].context.source_request_id, "root-b");
        // Background SPAWN remains parent-start anchored at +50 ms.
        let background = controller.drain_ready(200.5).unwrap();
        assert_eq!(
            background[0].context.source_request_id,
            "background-terminal"
        );
        controller
            .on_terminal(root_b[0].request.metadata().uuid.unwrap(), 201.0, false)
            .unwrap();
        assert_eq!(controller.lanes[0].play.play_index, 0);
        controller
            .on_terminal(background[0].request.metadata().uuid.unwrap(), 251.0, false)
            .unwrap();
        assert_eq!(controller.lanes[0].play.play_index, 1);
    }

    #[test]
    fn high_concurrency_ready_queue_drains_in_exact_order() {
        let mut protocol = config();
        protocol.session_tree_concurrency = 128;
        protocol.trajectory_start_min_ratio = 0.0;
        protocol.trajectory_start_max_ratio = 0.0;
        protocol.profile_duration_s = 100.0;
        let mut controller = AgentxReplayController::new(corpus(), protocol, 4, true).unwrap();
        controller.phase = AgentxPhase::Profile;
        controller.summary.phase = AgentxPhase::Profile;
        controller.profile_start_ms = Some(0.0);
        controller.profile_deadline_ms = Some(100_000.0);
        controller.primers.clear();
        controller.in_flight.clear();

        // Initialize ready queue across 128 lanes
        controller.rebuild_ready_queue();
        assert_eq!(controller.ready_queue.len(), 128);

        // Drain first 10ms
        let ready = controller.drain_ready(0.0).unwrap();
        assert_eq!(ready.len(), 128);
        assert_eq!(controller.ready_queue.len(), 0);

        // Verify all 128 lanes are in flight
        assert_eq!(controller.in_flight.len(), 128);
    }

    /// Cross-package contract hook. CI or the Stage 0 compiler tests can point
    /// this at a freshly compiled `trees.jsonl`; ordinary unit-test runs stay
    /// hermetic when the variable is absent.
    #[test]
    fn compiler_fixture_loads_when_requested() {
        let Some(path) = std::env::var_os("AGENTX_EXACT_COMPILER_FIXTURE") else {
            return;
        };
        let corpus = AgentxExactCorpus::from_jsonl_files(&[PathBuf::from(path)])
            .expect("compiler fixture must satisfy the native exact schema");
        assert!(!corpus.trees.is_empty());
        assert!(corpus.trees.iter().any(|tree| !tree.branches.is_empty()));
    }
}
