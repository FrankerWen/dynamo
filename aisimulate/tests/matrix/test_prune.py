# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from aisimulate.matrix.config import ModelSpec, StaticPrunePolicy
from aisimulate.matrix.prune import prune_parallel_configs
from aisimulate.spica import SearchSpace, SmartSearchConfig
from aisimulate.spica.parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
)


def test_static_prune_retains_structural_pareto_before_topk(monkeypatch):
    candidates = (
        ReplicaParallelConfig(
            shape=ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1
        ),
        ReplicaParallelConfig(
            shape=ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=2
        ),
        ReplicaParallelConfig(
            shape=ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=1), replicas=1
        ),
    )
    monkeypatch.setattr(
        "aisimulate.matrix.prune.enumerate_branches",
        lambda config, max_seq_len: [SimpleNamespace(parallel_configs=candidates)],
    )
    search_space = SearchSpace.model_construct(
        context_length=4096,
        parallel_configs=[],
    )
    config = SmartSearchConfig.model_construct(search_space=search_space)
    model = ModelSpec(id="model", model_path="example/model")

    result = prune_parallel_configs(
        config,
        model,
        StaticPrunePolicy(max_candidates=2, topk_per_strategy=1),
    )

    assert result.generated == 3
    assert result.retained == 2
    assert sum(decision.static_pareto for decision in result.decisions) == 2
    assert sum(decision.retained for decision in result.decisions) == 2
    assert all(
        decision.config["cp"] == 1
        for decision in result.decisions
        if "tp" in decision.config
    )
    dominated = result.decisions[2]
    assert not dominated.static_pareto
    assert not dominated.retained
    assert dominated.reason == "static_candidate_cap"


def test_static_prune_drops_disagg_attention_dp_before_replay(monkeypatch):
    tensor = ReplicaParallelConfig(
        shape=ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=2), replicas=1
    )
    attention_dp = ReplicaParallelConfig(
        shape=ParallelShape(tp=1, dp=2, moe_tp=1, moe_ep=2), replicas=1
    )
    candidates = (
        DisaggParallelConfig(prefill=tensor, decode=tensor),
        DisaggParallelConfig(prefill=attention_dp, decode=tensor),
    )
    monkeypatch.setattr(
        "aisimulate.matrix.prune.enumerate_branches",
        lambda config, max_seq_len: [SimpleNamespace(parallel_configs=candidates)],
    )
    search_space = SearchSpace.model_construct(
        context_length=4096,
        parallel_configs=[],
    )
    config = SmartSearchConfig.model_construct(search_space=search_space)

    result = prune_parallel_configs(
        config,
        ModelSpec(id="model", model_path="example/model"),
        StaticPrunePolicy(max_candidates=2, topk_per_strategy=2),
    )

    assert result.generated == 2
    assert result.retained == 1
    assert result.decisions[0].retained
    assert not result.decisions[1].retained
    assert result.decisions[1].reason == "replay_disagg_attention_dp_unsupported"
