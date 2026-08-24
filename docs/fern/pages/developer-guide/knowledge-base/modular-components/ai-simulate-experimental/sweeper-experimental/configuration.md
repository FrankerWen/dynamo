---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sweeper Configuration
subtitle: Core fields and optional adapter-owned search spaces
---

<!--
Generated from `aisimulate/docs/sweeper/configuration.md` by `docs/fern/scripts/sync_aisimulate_docs.py`.
Edit the canonical source instead of this Fern copy.
-->

> [!WARNING]
> **Experimental.** Sweeper's configuration schema may change without a standard deprecation period.

`SmartSearchConfig.search_space` contains backend and deployment fields. Optional feature-specific
search spaces are mappings under `SmartSearchConfig.adapters`.

## Top-Level Shape

```yaml
search_space:
  model_name: example/model
  hardware_sku: h200_sxm
  gpu_budget: 32
  deployment_mode: [disagg, agg]
  backend: [vllm, sglang]
  pd_transfer:
    kv_bytes_per_token: 131072
    bandwidth_gb_s: 64.0
    timing_mode: full_prompt

adapters:
  example.policy:
    search_space:
      mode: [balanced, latency]

workload:
  isl: 1024
  osl: 128
  request_rate: 4
  num_request_ratio: 10

goal:
  target: throughput_per_gpu

sweep:
  max_rounds: 10
  candidates_per_round: 8
  parallel_evals: 4
```

The adapter value is a search space, not one concrete runtime configuration. Its provider validates
the whole mapping, contributes optimizer dimensions, and later materializes one concrete adapter
configuration for each candidate.

## Outer Trace Load Scan

An open-loop trace can decouple its request-feature distribution from arrival pressure:

```yaml
workload:
  trace_path: /data/replay/traffic.jsonl
  trace_format: mooncake

load_sweep:
  axis: arrival_speedup_ratio
  values: [0.5, 1.0, 2.0, 4.0]

goal:
  target: pareto
  pareto_objectives: [throughput_per_gpu, throughput_per_user]
```

The only supported outer axis is currently `arrival_speedup_ratio`. Values are explicit positive
points, not a continuous optimizer range. Sweeper runs a fresh deployment study at every point and
constructs both load-conditioned Pareto fronts and a cross-load operating envelope. See
[Traffic](traffic.md) for the execution semantics.

## Backend Fields

| Field | Default | Purpose |
|---|---|---|
| `model_name` | required | model identifier |
| `hardware_sku` | required | AI Configurator system identifier |
| `deployment_mode` | `[disagg, agg]` | deployment branches to search |
| `backend` | `[vllm]` | engine backends to search |
| `gpu_budget` | `32` | maximum GPUs per candidate |
| `min_gpu_budget` | `None` | optional lower bound during enumeration |
| `context_length` | `None` | optional KV-feasibility sequence length |
| `parallel_configs` | `[]` | optional pinned parallel configurations |
| `startup_time` | `None` | optional simulated worker startup time |
| `aic_nextn` | `None` | optional speculative-decoding depth |
| `pd_transfer` | `None` | optional pinned per-request P-to-D KV transfer model for disaggregated candidates |

Each engine role also has lists for `max_num_batched_tokens` and `max_num_seqs`, plus pinned block
size, GPU-memory-utilization, and prefix-caching fields. A one-item list pins a searched field.

## Pinned Parallel Configurations

Pinning `parallel_configs` requires exactly one deployment mode. An aggregated entry is one shape:

```yaml
search_space:
  deployment_mode: [agg]
  parallel_configs:
    - tp: 4
      attention_dp: 2
      replicas: 2
```

A disaggregated entry contains `prefill` and `decode` shapes. Every pinned shape must be legal,
KV-feasible, and supported by at least one selected backend.

For offline disaggregated replay, prefill and decode can use different `attention_dp` values when
the runner uses fixed round-robin placement. This restricted attention-DP mode does not accept
Planner scaling or Router adapter hooks.

## P-to-D KV Transfer Timing

`pd_transfer` enables the existing scalar handoff timing model for every disaggregated candidate.
All three values are pinned workload/deployment assumptions rather than optimizer decisions:

```yaml
search_space:
  deployment_mode: [disagg]
  pd_transfer:
    kv_bytes_per_token: 131072
    bandwidth_gb_s: 64.0
    timing_mode: full_prompt
```

The charged delay is `tokens * kv_bytes_per_token / (bandwidth_gb_s * 1e9) * 1000` milliseconds.
`full_prompt` charges every prompt token. `destination_missing` charges only the destination's
missing prompt footprint. Each request receives the full configured effective bandwidth; this
model does not simulate shared-link contention, NIXL topology, or per-GPU transfers. Omit
`pd_transfer` to disable P-to-D transfer delay explicitly.

## Provider Selection

Adapter names are provider entry-point names. A provider can be installed through the
`aisimulate.sweep_config_providers` entry-point group or injected into the `Sweeper` constructor:

```python
sweeper = Sweeper(
    runner_factory=my_runner_factory,
    providers={"example.policy": my_provider},
)
```

Sweeper loads only names present under `adapters`. See [Sweep Configuration
Providers](sweep-config-provider.md) for the complete ABI.

## Sampler Algorithm Override

The experimental `AISIMULATE_SWEEPER_VIZIER_ALGO` environment variable overrides the Vizier
algorithm. For example, set it to `RANDOM_SEARCH` to bypass the default GP-bandit designer.
`SPICA_VIZIER_ALGO` remains a deprecated fallback during migration; when both are set, the
AI Simulate variable takes precedence.

## Removed KVBM Fields

Sweeper rejects the old flat KVBM block-count, transfer-bandwidth, offload-batch-size, and cache-hit
fields. The nested `pd_transfer` model above represents P-to-D request handoff and is separate from
KVBM tier bandwidths. Native G2 replaces KVBM, so the removed KVBM fields have no adapter migration.
