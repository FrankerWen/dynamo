# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

from aisimulate.matrix.replay_cache import CacheAwareReplayEvaluator
from aisimulate.matrix.store import ResultStore
from aisimulate.spica import OptimizationGoal, Workload
from aisimulate.spica.deploy import DeploymentPlan
from aisimulate.spica.evaluator import ReplayEvaluator


def _plan() -> DeploymentPlan:
    return DeploymentPlan(
        deployment_mode="agg",
        is_static=True,
        agg_engine_args={
            "aic_model_path": "example/model",
            "aic_system": "b300_sxm",
            "aic_tp_size": 2,
            "max_num_batched_tokens": 8192,
        },
        prefill_engine_args=None,
        decode_engine_args=None,
        num_workers=1,
        num_prefill_workers=0,
        num_decode_workers=0,
        router_mode="round_robin",
        router_config=None,
        planner_config=None,
    )


def _evaluator(
    *,
    request_rate: float = 1.0,
    revision: str = "sim-v1",
    entries: dict[str, dict[str, float]] | None = None,
    model_variant_id: str = "model_native_vllm",
) -> CacheAwareReplayEvaluator:
    return CacheAwareReplayEvaluator(
        Workload(
            isl=128,
            osl=16,
            request_rate=request_rate,
            num_request_ratio=10,
        ),
        OptimizationGoal(
            target="throughput",
            sla={"ttft_ms": 1000, "itl_ms": 100},
        ),
        workload_content_key="workload-content-sha",
        simulator_revision=revision,
        model_id="model",
        model_variant_id=model_variant_id,
        sku_id="b300_sxm",
        cache_entries=entries,
    )


def test_evaluation_cache_key_tracks_replay_semantics_not_search_policy():
    base = _evaluator().describe(_plan()).evaluation_cache_key

    # Search rounds/GPU budget are intentionally absent: a static replay sees
    # only the resolved deployment plan, not the policy that discovered it.
    assert _evaluator().describe(_plan()).evaluation_cache_key == base
    assert _evaluator(request_rate=2).describe(_plan()).evaluation_cache_key != base
    assert _evaluator(revision="sim-v2").describe(_plan()).evaluation_cache_key != base
    # Resolver/native/fallback labels are provenance, not replay semantics.
    # The concrete checkpoint + quant profile live in the deployment plan.
    assert (
        _evaluator(model_variant_id="model_fp8_vllm")
        .describe(_plan())
        .evaluation_cache_key
        == base
    )
    changed_plan = replace(
        _plan(),
        agg_engine_args={**_plan().agg_engine_args, "aic_tp_size": 4},
    )
    assert _evaluator().describe(changed_plan).evaluation_cache_key != base


def test_cache_hit_skips_underlying_replay(monkeypatch):
    evaluator = _evaluator()
    description = evaluator.describe(_plan())
    evaluator.load_entries(
        {description.evaluation_cache_key: {"output_throughput_tok_s": 123.0}}
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("underlying replay must not run on an exact cache hit")

    monkeypatch.setattr(ReplayEvaluator, "evaluate", fail_if_called)
    result = evaluator.evaluate_with_cache(_plan())

    assert result.cache_hit
    assert result.metrics == {"output_throughput_tok_s": 123.0}


def test_result_store_imports_cache_and_records_hit_provenance(tmp_path):
    evaluator = _evaluator()
    description = evaluator.describe(_plan())
    source_dir = tmp_path / "source"
    with ResultStore.open(source_dir) as source:
        source.store_replay_cache_entry(
            evaluation_cache_key=description.evaluation_cache_key,
            cache_namespace_key=description.cache_namespace_key,
            semantics=description.semantics,
            metrics={"output_throughput_tok_s": 123.0},
            source_cell_id="source-cell",
            source_stage="spica",
        )

    with ResultStore.open(tmp_path / "target") as target:
        imported = target.import_replay_cache([source_dir])
        snapshot = target.replay_cache_snapshot(description.cache_namespace_key)
        target.record_evaluation(
            cell_id="target-cell",
            evaluation_key="spica:test",
            stage="spica",
            outcome="feasible",
            reason="evaluation_cache_hit",
            selection={},
            metrics=snapshot[description.evaluation_cache_key],
            sla_pass=True,
            candidate=None,
            evaluation_cache_key=description.evaluation_cache_key,
            cache_hit=True,
        )
        row = target.connection.execute(
            """
            SELECT evaluation_cache_key, cache_hit
            FROM evaluations WHERE evaluation_key = 'spica:test'
            """
        ).fetchone()

    assert imported["imported"] == 1
    assert snapshot[description.evaluation_cache_key]["output_throughput_tok_s"] == 123
    assert row == (description.evaluation_cache_key, True)
