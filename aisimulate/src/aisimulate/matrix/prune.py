# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative AIC-backed structural pruning before replay optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aisimulate.spica import SmartSearchConfig
from aisimulate.spica.parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
)
from aisimulate.spica.search_space import enumerate_branches

from .config import ModelSpec, StaticPrunePolicy


@dataclass(frozen=True)
class StaticPruneDecision:
    config: dict[str, Any]
    retained: bool
    reason: str
    static_pareto: bool
    static_rank: int | None
    objectives: dict[str, float]


@dataclass(frozen=True)
class StaticPruneResult:
    config: SmartSearchConfig
    generated: int
    retained: int
    decisions: tuple[StaticPruneDecision, ...]


def _shape_allowed(shape: ParallelShape, model: ModelSpec) -> bool:
    constraint = model.parallel
    return all(
        allowed is None or value in allowed
        for value, allowed in (
            (shape.tp, constraint.tp),
            (shape.dp, constraint.attention_dp),
            (shape.moe_tp, constraint.moe_tp),
            (shape.moe_ep, constraint.moe_ep),
        )
    ) and (
        constraint.max_gpus_per_worker is None
        or shape.gpus_per_worker <= constraint.max_gpus_per_worker
    )


def _replica_payload(config: ReplicaParallelConfig) -> dict[str, Any]:
    shape = config.shape
    return {
        "tp": shape.tp,
        "attention_dp": shape.dp,
        "moe_tp": shape.moe_tp,
        "moe_ep": shape.moe_ep,
        "pp": shape.pp,
        "cp": 1,
        "replicas": config.replicas,
    }


def _payload(config: ReplicaParallelConfig | DisaggParallelConfig) -> dict[str, Any]:
    if isinstance(config, ReplicaParallelConfig):
        return _replica_payload(config)
    return {
        "prefill": _replica_payload(config.prefill),
        "decode": _replica_payload(config.decode),
    }


def _strategy(config: ReplicaParallelConfig | DisaggParallelConfig) -> str:
    if isinstance(config, ReplicaParallelConfig):
        return config.shape.strategy
    return f"p:{config.prefill.shape.strategy}/d:{config.decode.shape.strategy}"


def _used_gpus(config: ReplicaParallelConfig | DisaggParallelConfig) -> int:
    return config.total_gpus


def _worker_count(config: ReplicaParallelConfig | DisaggParallelConfig) -> int:
    if isinstance(config, ReplicaParallelConfig):
        return config.replicas
    return config.prefill.replicas + config.decode.replicas


def _max_worker_gpus(config: ReplicaParallelConfig | DisaggParallelConfig) -> int:
    if isinstance(config, ReplicaParallelConfig):
        return config.shape.gpus_per_worker
    return max(
        config.prefill.shape.gpus_per_worker,
        config.decode.shape.gpus_per_worker,
    )


def _static_objectives(
    config: ReplicaParallelConfig | DisaggParallelConfig,
) -> dict[str, float]:
    return {
        "used_gpus_min": float(_used_gpus(config)),
        "worker_count_max": float(_worker_count(config)),
        "max_worker_gpus_min": float(_max_worker_gpus(config)),
    }


def _static_pareto(
    configs: list[ReplicaParallelConfig | DisaggParallelConfig],
) -> set[ReplicaParallelConfig | DisaggParallelConfig]:
    front = set()
    for candidate in configs:
        candidate_metrics = _static_objectives(candidate)
        dominated = False
        for other in configs:
            if other == candidate:
                continue
            other_metrics = _static_objectives(other)
            no_worse = (
                other_metrics["used_gpus_min"] <= candidate_metrics["used_gpus_min"]
                and other_metrics["worker_count_max"]
                >= candidate_metrics["worker_count_max"]
                and other_metrics["max_worker_gpus_min"]
                <= candidate_metrics["max_worker_gpus_min"]
            )
            strictly_better = any(
                (
                    other_metrics[key] < candidate_metrics[key]
                    if key.endswith("_min")
                    else other_metrics[key] > candidate_metrics[key]
                )
                for key in candidate_metrics
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.add(candidate)
    return front


def _allowed(
    config: ReplicaParallelConfig | DisaggParallelConfig, model: ModelSpec
) -> bool:
    if isinstance(config, ReplicaParallelConfig):
        return _shape_allowed(config.shape, model)
    return _shape_allowed(config.prefill.shape, model) and _shape_allowed(
        config.decode.shape, model
    )


def _replay_supported(
    config: ReplicaParallelConfig | DisaggParallelConfig,
) -> bool:
    """Mirror replay constraints that are narrower than AIC shape support."""
    if isinstance(config, ReplicaParallelConfig):
        return True
    return config.prefill.shape.dp == 1 and config.decode.shape.dp == 1


def prune_parallel_configs(
    config: SmartSearchConfig,
    model: ModelSpec,
    policy: StaticPrunePolicy,
) -> StaticPruneResult:
    """Use Spica's AIC support/KV checks, then apply diversity-preserving caps."""
    branches = enumerate_branches(
        config, max_seq_len=config.search_space.context_length
    )
    if len(branches) != 1:
        raise ValueError(
            "static pruning requires exactly one deployment_mode per study"
        )
    generated = list(branches[0].parallel_configs)
    legal = [
        candidate
        for candidate in generated
        if _allowed(candidate, model) and _replay_supported(candidate)
    ]

    def static_key(candidate):
        return (
            _used_gpus(candidate),
            -_worker_count(candidate),
            _max_worker_gpus(candidate),
            _strategy(candidate),
            repr(candidate),
        )

    legal.sort(key=static_key)
    static_ranks = {candidate: index for index, candidate in enumerate(legal, start=1)}
    static_pareto = _static_pareto(legal)

    by_strategy: dict[str, list[ReplicaParallelConfig | DisaggParallelConfig]] = {}
    for candidate in legal:
        by_strategy.setdefault(_strategy(candidate), []).append(candidate)

    retained: list[ReplicaParallelConfig | DisaggParallelConfig] = []
    retention_source: dict[ReplicaParallelConfig | DisaggParallelConfig, str] = {}
    for strategy in sorted(by_strategy):
        for candidate in by_strategy[strategy][: policy.topk_per_strategy]:
            if len(retained) >= policy.max_candidates:
                break
            if candidate not in retention_source:
                retained.append(candidate)
                retention_source[candidate] = "strategy_topk"
    for candidate in sorted(static_pareto, key=static_key):
        if len(retained) >= policy.max_candidates:
            break
        if candidate not in retention_source:
            retained.append(candidate)
            retention_source[candidate] = "static_pareto"
    retained_set = set(retained)
    for candidate in legal:
        if len(retained) >= policy.max_candidates:
            break
        if candidate not in retained_set:
            retained.append(candidate)
            retained_set.add(candidate)
            retention_source[candidate] = "global_fill"
    retained = sorted(
        retained[: policy.max_candidates],
        key=static_key,
    )
    retained_set = set(retained)

    decisions = []
    for candidate in generated:
        if not _allowed(candidate, model):
            reason = "model_catalog_parallel_constraint"
            keep = False
        elif not _replay_supported(candidate):
            reason = "replay_disagg_attention_dp_unsupported"
            keep = False
        elif candidate in retained_set:
            reason = (
                "aic_supported_kv_feasible_"
                f"{retention_source.get(candidate, 'global_fill')}"
            )
            keep = True
        else:
            reason = "static_candidate_cap"
            keep = False
        decisions.append(
            StaticPruneDecision(
                config=_payload(candidate),
                retained=keep,
                reason=reason,
                static_pareto=candidate in static_pareto,
                static_rank=static_ranks.get(candidate),
                objectives=_static_objectives(candidate),
            )
        )

    if not retained:
        raise ValueError("static pruning removed every AIC-supported parallel config")
    search_space = config.search_space.model_copy(
        update={"parallel_configs": [_payload(candidate) for candidate in retained]}
    )
    return StaticPruneResult(
        config=config.model_copy(update={"search_space": search_space}),
        generated=len(generated),
        retained=len(retained),
        decisions=tuple(decisions),
    )
