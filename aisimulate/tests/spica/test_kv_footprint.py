# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest

from aisimulate.spica.config import KVFootprintSpec, Workload
from aisimulate.spica.kv_footprint import (
    UnsupportedKVFootprint,
    resolve_candidate_kv_footprint,
    resolve_role_footprint,
)


def _sample(
    model_name: str,
    *,
    prefill_tp: int = 1,
    decode_tp: int = 1,
    layout: str = "test",
    dtype: str | None = None,
) -> dict:
    return {
        "deployment_mode": "disagg",
        "backend": "vllm",
        "model_name": model_name,
        "aic_nextn": None,
        "context_length": 32768,
        "kv_footprint": {
            "source": "aic_native",
            "layout": layout,
            "scope": "per_rank",
            "dtype": dtype,
        },
        "kv_footprint_policy": {
            "nonlinear_kvbm": "fixed_synthetic_average",
            "heterogeneous_trace": "fail_closed",
        },
        "prefill_tp": prefill_tp,
        "prefill_attention_dp": 1,
        "prefill_moe_tp": 1,
        "prefill_moe_ep": prefill_tp,
        "prefill_block_size": 64,
        "prefill_replicas": 1,
        "decode_tp": decode_tp,
        "decode_attention_dp": 1,
        "decode_moe_tp": 1,
        "decode_moe_ep": decode_tp,
        "decode_block_size": 64,
        "decode_replicas": 1,
        "num_g2_blocks": 0,
        "g2_capacity_gib": None,
        "g2_capacity_scope": "deployment_total",
        "kv_transfer_bandwidth": 100.0,
        "kv_transfer_timing_mode": "full_prompt",
    }


def _synthetic(isl: int = 8192) -> Workload:
    return Workload(
        isl=isl,
        osl=1024,
        request_rate=1,
        num_request_ratio=1,
    )


def test_aic_native_deepseek_v4_curve_retains_fixed_and_nonlinear_bytes():
    sample = _sample(
        "deepseek-ai/DeepSeek-V4-Flash",
        prefill_tp=4,
        layout="compressed_sparse_mla",
        dtype="fp8",
    )
    footprint = resolve_role_footprint(
        sample,
        role="prefill",
        spec=KVFootprintSpec.model_validate(sample["kv_footprint"]),
        reference_length=8192,
    )

    assert footprint.model_class == "DeepSeekV4Model"
    assert footprint.scope == "per_rank"
    assert footprint.bytes_at_zero == 11_345_920
    assert footprint.bytes_at_reference == 39_591_936
    assert footprint.marginal_bytes_per_token == 22_016
    assert footprint.growth_kind == "nonlinear"


def test_aic_native_standard_gqa_rate_depends_on_tp():
    tp1 = _sample("MiniMaxAI/MiniMax-M3", prefill_tp=1)
    tp4 = _sample("MiniMaxAI/MiniMax-M3", prefill_tp=4)
    spec = KVFootprintSpec(source="aic_native", layout="sparse_attention_gqa")

    one = resolve_role_footprint(tp1, role="prefill", spec=spec, reference_length=8192)
    four = resolve_role_footprint(tp4, role="prefill", spec=spec, reference_length=8192)

    assert one.marginal_bytes_per_token == 122_880
    assert four.marginal_bytes_per_token == 30_720
    assert one.bytes_at_zero == four.bytes_at_zero == 0
    assert one.growth_kind == four.growth_kind == "linear_zero_intercept"


def test_variant_kv_precision_overrides_checkpoint_default():
    sample = _sample(
        "MiniMaxAI/MiniMax-M3",
        prefill_tp=1,
        layout="sparse_attention_gqa",
        dtype="fp8",
    )
    sample.update(
        {
            "aic_gemm_dtype": "bfloat16",
            "aic_moe_dtype": "bfloat16",
            "aic_fmha_dtype": "bfloat16",
            "aic_kv_cache_dtype": "fp8",
            "aic_comm_dtype": "half",
        }
    )

    footprint = resolve_role_footprint(
        sample,
        role="prefill",
        spec=KVFootprintSpec.model_validate(sample["kv_footprint"]),
        reference_length=8192,
    )

    assert footprint.dtype == "fp8"
    assert footprint.marginal_bytes_per_token == 61_440


def test_aic_native_kimi_retains_tp_dependent_fixed_kda_state():
    tp1 = _sample("moonshotai/Kimi-K3", prefill_tp=1)
    tp8 = _sample("moonshotai/Kimi-K3", prefill_tp=8)
    spec = KVFootprintSpec(source="aic_native", layout="hybrid_mla_kda_fixed_state")

    one = resolve_role_footprint(tp1, role="prefill", spec=spec, reference_length=8192)
    eight = resolve_role_footprint(
        tp8, role="prefill", spec=spec, reference_length=8192
    )

    assert one.bytes_at_zero == 2_246_860_800
    assert eight.bytes_at_zero == 280_857_600
    assert one.marginal_bytes_per_token == eight.marginal_bytes_per_token == 27_648
    assert one.growth_kind == eight.growth_kind == "affine_fixed_state"


def test_fixed_synthetic_resolves_pd_and_kvbm_as_separate_scalars():
    sample = _sample(
        "MiniMaxAI/MiniMax-M3",
        prefill_tp=4,
        decode_tp=1,
        layout="sparse_attention_gqa",
        dtype="bfloat16",
    )
    sample["g2_capacity_gib"] = 64.0
    resolved = resolve_candidate_kv_footprint(sample, _synthetic())

    # KVBM is attached to TP4 prefill, while handoff is bottlenecked by the
    # larger TP1 decode rank. The two rates must not alias.
    assert resolved["kv_bytes_per_token"] == 30_720
    assert resolved["kv_transfer_bytes_per_token"] == 122_880
    assert resolved["kvbm_footprint_contract"] == "exact_linear"
    assert resolved["pd_transfer_footprint_contract"].startswith("exact_fixed")
    assert resolved["g2_effective_total_capacity_gib"] <= 64.0
    assert math.isclose(resolved["g2_effective_total_capacity_gib"], 64.0, rel_tol=2e-5)


def test_pd_scalar_accounts_for_replicated_kv_broadcast_fanout():
    sample = _sample(
        "deepseek-ai/DeepSeek-V4-Flash",
        prefill_tp=1,
        decode_tp=4,
        layout="compressed_sparse_mla",
        dtype="fp8",
    )

    resolved = resolve_candidate_kv_footprint(sample, _synthetic())

    decode_bytes = resolved["decode_kv_footprint"]["bytes_at_reference"]
    assert resolved["kv_transfer_bytes_per_token"] == math.ceil(decode_bytes * 4 / 8192)
    assert resolved["pd_transfer_physical_bytes_per_request_estimate"] == (
        decode_bytes * 4
    )
    assert "ideal_balanced_reshard" in resolved["pd_transfer_footprint_contract"]


def test_deployment_total_g2_capacity_includes_attention_dp_rank_pools():
    sample = _sample(
        "MiniMaxAI/MiniMax-M3",
        prefill_tp=4,
        layout="sparse_attention_gqa",
        dtype="bfloat16",
    )
    sample["prefill_attention_dp"] = 2
    sample["prefill_moe_ep"] = 8
    sample["g2_capacity_gib"] = 64.0

    resolved = resolve_candidate_kv_footprint(sample, _synthetic())

    per_rank = resolved["g2_effective_capacity_gib_per_rank"]
    assert per_rank == pytest.approx(8.0, rel=2e-5)
    assert resolved["g2_effective_total_capacity_gib"] == pytest.approx(64.0, rel=2e-5)


def test_non_linear_trace_and_destination_missing_fail_closed(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    sample = _sample(
        "deepseek-ai/DeepSeek-V4-Flash",
        layout="compressed_sparse_mla",
        dtype="fp8",
    )
    trace_workload = Workload(trace_path=str(trace))
    with pytest.raises(UnsupportedKVFootprint, match="heterogeneous trace"):
        resolve_candidate_kv_footprint(dict(sample), trace_workload)

    sample["kv_transfer_timing_mode"] = "destination_missing"
    with pytest.raises(UnsupportedKVFootprint, match="destination_missing"):
        resolve_candidate_kv_footprint(sample, _synthetic())


def test_catalog_dtype_disagreement_fails_closed():
    sample = _sample(
        "MiniMaxAI/MiniMax-M3",
        layout="sparse_attention_gqa",
        dtype="fp8",
    )
    with pytest.raises(UnsupportedKVFootprint, match="disagrees"):
        resolve_candidate_kv_footprint(sample, _synthetic())
