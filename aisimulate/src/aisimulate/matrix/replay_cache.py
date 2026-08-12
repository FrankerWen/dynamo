# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy-independent cache keys for exact replay computations.

The cache contract is deliberately below MatrixCell/SearchPolicy identity. Two
experiments may share a replay only when the actual evaluator inputs are equal:
the prepared workload contents, replay workload parameters, SLA/goal, resolved
deployment plan, load/concurrency, model/SKU carried by that plan, and explicit
simulator revision. Search breadth, GPU budget, rounds, and price are not inputs
to a static replay and therefore do not participate in this key.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from aisimulate.spica.config import OptimizationGoal, Workload
from aisimulate.spica.deploy import DeploymentPlan
from aisimulate.spica.evaluator import (
    EVALUATION_CACHE_HIT_METRIC,
    ReplayEvaluator,
)

from ._utils import fingerprint

REPLAY_CACHE_CONTRACT_VERSION = 3


@dataclass(frozen=True)
class ReplayCacheDescription:
    evaluation_cache_key: str
    cache_namespace_key: str
    semantics: dict[str, Any]


@dataclass(frozen=True)
class CachedReplayResult:
    metrics: dict[str, float]
    description: ReplayCacheDescription
    cache_hit: bool


class CacheAwareReplayEvaluator(ReplayEvaluator):
    """Replay evaluator backed by an in-memory, read-only cache snapshot.

    Worker processes receive a pickle of this object and never access DuckDB.
    The owning Matrix process persists misses after workers return.
    """

    def __init__(
        self,
        workload: Workload,
        goal: OptimizationGoal,
        *,
        workload_content_key: str,
        simulator_revision: str | None,
        model_id: str,
        sku_id: str,
        model_variant_id: str | None = None,
        cache_entries: dict[str, dict[str, float]] | None = None,
        trace_block_size: int | None = None,
        benchmark_granularity: int = 8,
    ):
        super().__init__(
            workload,
            goal,
            trace_block_size=trace_block_size,
            benchmark_granularity=benchmark_granularity,
        )
        self.workload_content_key = workload_content_key
        self.simulator_revision = simulator_revision
        self.model_id = model_id
        self.model_variant_id = model_variant_id
        self.sku_id = sku_id
        self._cache_entries = cache_entries if cache_entries is not None else {}

    @property
    def cache_enabled(self) -> bool:
        """Cross-experiment reuse requires an explicit simulator identity."""
        return bool(self.simulator_revision)

    @property
    def cache_namespace_key(self) -> str:
        return fingerprint(
            {
                "contract_version": REPLAY_CACHE_CONTRACT_VERSION,
                "workload_content_key": self.workload_content_key,
                "simulator_revision": self.simulator_revision,
                "model_id": self.model_id,
                "sku_id": self.sku_id,
            }
        )

    def fork(self, workload: Workload) -> CacheAwareReplayEvaluator:
        """Use another load/seed while sharing the same local cache snapshot."""
        return CacheAwareReplayEvaluator(
            workload,
            self.goal,
            workload_content_key=self.workload_content_key,
            simulator_revision=self.simulator_revision,
            model_id=self.model_id,
            model_variant_id=self.model_variant_id,
            sku_id=self.sku_id,
            cache_entries=self._cache_entries,
            trace_block_size=self.trace_block_size,
            benchmark_granularity=self.benchmark_granularity,
        )

    def describe(
        self,
        plan: DeploymentPlan,
        *,
        concurrency_override: int | None = None,
    ) -> ReplayCacheDescription:
        workload = self.workload.model_dump(mode="python")
        # File identity comes from the source content hash + normalization key.
        # A materialization path and the surrounding search axis are not replay
        # semantics; the selected load itself remains in this Workload payload.
        workload.pop("trace_path", None)
        semantics = {
            "contract_version": REPLAY_CACHE_CONTRACT_VERSION,
            "namespace": {
                "workload_content_key": self.workload_content_key,
                "simulator_revision": self.simulator_revision,
                "model_id": self.model_id,
                "sku_id": self.sku_id,
            },
            "replay": {
                "workload": workload,
                "goal": self.goal.model_dump(mode="python"),
                "trace_block_size": self.trace_block_size,
                "benchmark_granularity": self.benchmark_granularity,
                "concurrency_override": concurrency_override,
                "deployment_plan": asdict(plan),
            },
        }
        return ReplayCacheDescription(
            evaluation_cache_key=fingerprint(semantics),
            cache_namespace_key=self.cache_namespace_key,
            semantics=semantics,
        )

    def remember(
        self, description: ReplayCacheDescription, metrics: dict[str, float]
    ) -> None:
        if self.cache_enabled:
            self._cache_entries[description.evaluation_cache_key] = dict(metrics)

    def load_entries(self, entries: dict[str, dict[str, float]]) -> None:
        if self.cache_enabled:
            self._cache_entries.update(entries)

    def evaluate_with_cache(
        self,
        plan: DeploymentPlan,
        *,
        concurrency_override: int | None = None,
    ) -> CachedReplayResult:
        description = self.describe(plan, concurrency_override=concurrency_override)
        cached = (
            self._cache_entries.get(description.evaluation_cache_key)
            if self.cache_enabled
            else None
        )
        if cached is not None:
            return CachedReplayResult(
                metrics=dict(cached),
                description=description,
                cache_hit=True,
            )
        metrics = super().evaluate(plan, concurrency_override=concurrency_override)
        return CachedReplayResult(
            metrics=dict(metrics),
            description=description,
            cache_hit=False,
        )

    def evaluate(
        self,
        plan: DeploymentPlan,
        *,
        concurrency_override: int | None = None,
    ) -> dict[str, float]:
        """Spica-compatible API; a private metric carries hit provenance."""
        result = self.evaluate_with_cache(
            plan, concurrency_override=concurrency_override
        )
        metrics = dict(result.metrics)
        if result.cache_hit:
            metrics[EVALUATION_CACHE_HIT_METRIC] = 1.0
        return metrics
