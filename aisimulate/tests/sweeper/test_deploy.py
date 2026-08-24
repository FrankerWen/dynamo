# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend sample to runner-neutral deployment contract."""

import pytest

from aisimulate.sweeper.config import SearchSpace
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
)
from aisimulate.sweeper.sample import unroll_sample

BACKEND_VERSION = "1.3.0rc10"


def _space(**overrides) -> SearchSpace:
    values = {"model_name": "example/model", "hardware_sku": "example_sku"}
    values.update(overrides)
    return SearchSpace(**values)


def _agg_selection(**overrides) -> dict:
    values = {
        "deployment_mode": "agg",
        "backend": "trtllm",
        "agg_max_num_batched_tokens": 16384,
        "agg_max_num_seqs": 512,
    }
    values.update(overrides)
    return values


AGG_MOE = ReplicaParallelConfig(
    ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2
)


def _agg_deployment(*, space=None, selection=None, parallel_config=AGG_MOE):
    sample = unroll_sample(
        search_space=space or _space(),
        selection=selection or _agg_selection(),
        parallel_config=parallel_config,
    )
    return build_backend_deployment(sample, backend_version=BACKEND_VERSION)


def test_agg_backend_deployment_preserves_engine_payload():
    deployment = _agg_deployment()
    engine = deployment.agg_engine_args

    assert deployment.deployment_mode == "agg"
    assert deployment.backend == "trtllm"
    assert deployment.backend_version == BACKEND_VERSION
    assert deployment.num_workers == 2
    assert deployment.num_prefill_workers == 0
    assert deployment.num_decode_workers == 0
    assert deployment.prefill_engine_args is None
    assert deployment.decode_engine_args is None
    assert deployment.parallel_config == {
        "tp": 4,
        "pp": 1,
        "attention_dp": 1,
        "moe_tp": 1,
        "moe_ep": 4,
        "strategy": "tep",
        "replicas": 2,
    }
    assert engine == {
        "worker_type": "aggregated",
        "engine_type": "trtllm",
        "aic_backend": "trtllm",
        "aic_backend_version": BACKEND_VERSION,
        "aic_system": "example_sku",
        "aic_model_path": "example/model",
        "aic_tp_size": 4,
        "aic_attention_dp_size": 1,
        "aic_moe_tp_size": 1,
        "aic_moe_ep_size": 4,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 512,
        "block_size": 64,
        "gpu_memory_utilization": 0.9,
        "enable_prefix_caching": True,
    }


def test_disagg_backend_deployment_preserves_both_roles():
    parallel = DisaggParallelConfig(
        prefill=ReplicaParallelConfig(ParallelShape(tp=8, dp=1, moe_tp=1, moe_ep=8), 1),
        decode=ReplicaParallelConfig(ParallelShape(tp=1, dp=8, moe_tp=1, moe_ep=8), 2),
    )
    selection = _agg_selection(
        deployment_mode="disagg",
        backend="sglang",
        prefill_max_num_batched_tokens=32768,
        prefill_max_num_seqs=4,
        decode_max_num_batched_tokens=8192,
        decode_max_num_seqs=1024,
    )
    sample = unroll_sample(
        search_space=_space(), selection=selection, parallel_config=parallel
    )

    deployment = build_backend_deployment(sample, backend_version=BACKEND_VERSION)

    assert deployment.deployment_mode == "disagg"
    assert deployment.backend == "sglang"
    assert deployment.num_workers == 0
    assert deployment.num_prefill_workers == 1
    assert deployment.num_decode_workers == 2
    assert deployment.agg_engine_args is None
    assert deployment.prefill_engine_args["worker_type"] == "prefill"
    assert deployment.prefill_engine_args["aic_tp_size"] == 8
    assert deployment.prefill_engine_args["engine_type"] == "sglang"
    assert deployment.decode_engine_args["worker_type"] == "decode"
    assert deployment.decode_engine_args["aic_attention_dp_size"] == 8
    assert deployment.decode_engine_args["engine_type"] == "sglang"


def test_disagg_backend_deployment_injects_scalar_pd_transfer_on_both_pools():
    parallel = DisaggParallelConfig(
        prefill=ReplicaParallelConfig(ParallelShape(tp=1, dp=2, moe_tp=1, moe_ep=2), 1),
        decode=ReplicaParallelConfig(ParallelShape(tp=1, dp=4, moe_tp=1, moe_ep=4), 1),
    )
    selection = _agg_selection(
        deployment_mode="disagg",
        backend="sglang",
        prefill_max_num_batched_tokens=32768,
        prefill_max_num_seqs=4,
        decode_max_num_batched_tokens=8192,
        decode_max_num_seqs=1024,
    )
    sample = unroll_sample(
        search_space=_space(
            deployment_mode=["disagg"],
            pd_transfer={
                "kv_bytes_per_token": 1_000_000,
                "bandwidth_gb_s": 1.0,
                "timing_mode": "full_prompt",
            },
        ),
        selection=selection,
        parallel_config=parallel,
    )

    deployment = build_backend_deployment(sample, backend_version=BACKEND_VERSION)

    expected = {
        "kv_bytes_per_token": 1_000_000,
        "kv_transfer_bandwidth": 1.0,
        "kv_transfer_timing_mode": "full_prompt",
    }
    assert {key: deployment.prefill_engine_args[key] for key in expected} == expected
    assert {key: deployment.decode_engine_args[key] for key in expected} == expected


def test_dense_shape_omits_moe_sizes():
    dense = ReplicaParallelConfig(
        ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=1), replicas=1
    )
    engine = _agg_deployment(
        space=_space(model_name="example/dense"),
        parallel_config=dense,
    ).agg_engine_args

    assert engine["aic_tp_size"] == 2
    assert "aic_moe_tp_size" not in engine
    assert "aic_moe_ep_size" not in engine


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_engine_type_tracks_swept_backend(backend):
    engine = _agg_deployment(selection=_agg_selection(backend=backend)).agg_engine_args

    assert engine["engine_type"] == backend
    assert engine["aic_backend"] == backend


def test_optional_backend_runtime_values_are_forwarded():
    engine = _agg_deployment(
        space=_space(startup_time=45.0, aic_nextn=2)
    ).agg_engine_args

    assert engine["startup_time"] == 45.0
    assert engine["aic_nextn"] == 2


def test_analytical_contract_uses_canonical_policy_json():
    engine = _agg_deployment(
        space=_space(
            database_mode="ANALYTICAL",
            analytical_config={"attention_algorithm": "fa3", "level": "high"},
            aic_gemm_dtype="fp8_block",
            aic_moe_dtype="nvfp4",
            aic_fmha_dtype="bfloat16",
            aic_kv_cache_dtype="fp8",
            aic_comm_dtype="half",
        )
    ).agg_engine_args

    assert engine["aic_database_mode"] == "ANALYTICAL"
    assert engine["aic_analytical_config"] == (
        '{"attention_algorithm":"fa3","communication_mode":"empirical",'
        '"fp8_gemm_recipe":"sglang","level":"high",'
        '"moe_combine_dtype":"half","moe_dispatch_dtype":"half",'
        '"sparse_attention_head_quantum":null,"wideep_combine_dtype":"half",'
        '"wideep_dispatch_dtype":"half"}'
    )
    assert engine["aic_gemm_dtype"] == "fp8_block"
    assert engine["aic_moe_dtype"] == "nvfp4"
    assert engine["aic_fmha_dtype"] == "bfloat16"
    assert engine["aic_kv_cache_dtype"] == "fp8"
    assert engine["aic_comm_dtype"] == "half"


def test_backend_deployment_contains_no_dynamo_policy_fields():
    deployment = _agg_deployment()

    assert not hasattr(deployment, "planner_config")
    assert not hasattr(deployment, "router_config")
    assert not hasattr(deployment, "is_static")
