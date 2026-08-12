# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate-specific KV-cache byte contracts backed by AIConfigurator.

AIConfigurator reports the bytes occupied by one sequence on one GPU rank.
DynoSim currently consumes scalar byte rates for KVBM blocks and P/D handoff,
so this module is the only place allowed to reduce a native sequence-footprint
curve to those scalars.  Every reduction is stored on the candidate together
with its scope and approximation status.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any

from aiconfigurator_core.sdk.config_builders import apply_nextn, build_model_config
from aiconfigurator_core.sdk.models import get_model

from .config import KVFootprintPolicy, KVFootprintSpec, Workload


class UnsupportedKVFootprint(ValueError):
    """The scalar replay contract cannot faithfully represent this footprint."""


@dataclass(frozen=True)
class KVRoleFootprint:
    """Resolved native footprint for one engine role and parallel rank."""

    source: str
    layout: str
    scope: str
    dtype: str
    dtype_bytes: float | None
    model_class: str
    tp: int
    attention_dp: int
    moe_tp: int
    moe_ep: int
    reference_length: int
    bytes_at_zero: int
    bytes_at_reference: int
    marginal_bytes_per_token: int
    growth_kind: str
    probe_lengths: tuple[int, ...]
    probe_bytes: tuple[int, ...]

    @property
    def average_bytes_per_token(self) -> int:
        if self.reference_length <= 0:
            raise ValueError("KV footprint reference_length must be positive")
        return math.ceil(self.bytes_at_reference / self.reference_length)

    @property
    def is_zero_intercept_linear(self) -> bool:
        return self.growth_kind == "linear_zero_intercept"


def _normalize_dtype(value: str) -> str:
    normalized = value.strip().lower().replace("torch.", "")
    return {"bf16": "bfloat16", "fp16": "float16"}.get(normalized, normalized)


def _shape_fields(sample: dict[str, Any], role: str) -> tuple[int, int, int, int]:
    prefix = "" if role == "agg" else f"{role}_"
    return (
        int(sample[f"{prefix}tp"]),
        int(sample[f"{prefix}attention_dp"]),
        int(sample[f"{prefix}moe_tp"]),
        int(sample[f"{prefix}moe_ep"]),
    )


def _probe_lengths(reference_length: int, block_size: int) -> tuple[int, ...]:
    # 4K/8K catch sliding/compression transitions used by current native
    # models; the workload reference and one following block catch local
    # discontinuities without assuming a particular model family.
    return tuple(
        sorted(
            {
                0,
                1,
                max(1, block_size),
                4096,
                8192,
                max(1, reference_length),
                max(1, reference_length + block_size),
            }
        )
    )


@cache
def _resolve_role_cached(
    *,
    source: str,
    layout: str,
    scope: str,
    expected_dtype: str | None,
    explicit_bytes_per_token: int | None,
    model_name: str,
    backend: str,
    tp: int,
    attention_dp: int,
    moe_tp: int,
    moe_ep: int,
    nextn: int,
    reference_length: int,
    block_size: int,
    gemm_quant_mode: str | None,
    moe_quant_mode: str | None,
    fmha_quant_mode: str | None,
    kvcache_quant_mode: str | None,
    comm_quant_mode: str | None,
) -> KVRoleFootprint:
    lengths = _probe_lengths(reference_length, block_size)
    if source == "explicit_linear":
        if explicit_bytes_per_token is None:  # guarded by Pydantic; defensive here.
            raise ValueError("explicit_linear footprint has no byte rate")
        rate = int(explicit_bytes_per_token)
        values = tuple(length * rate for length in lengths)
        dtype = expected_dtype or "catalog_unspecified"
        dtype_bytes = None
        model_class = "ExplicitLinear"
    else:
        config = build_model_config(
            tp_size=tp,
            pp_size=1,
            attention_dp_size=attention_dp,
            moe_tp_size=moe_tp,
            moe_ep_size=moe_ep,
            gemm_quant_mode=gemm_quant_mode,
            moe_quant_mode=moe_quant_mode,
            fmha_quant_mode=fmha_quant_mode,
            kvcache_quant_mode=kvcache_quant_mode,
            comm_quant_mode=comm_quant_mode,
        )
        apply_nextn(config, nextn)
        model = get_model(model_name, config, backend)
        dtype = str(model.config.kvcache_quant_mode.name)
        dtype_bytes = float(model.config.kvcache_quant_mode.value.memory)
        if expected_dtype is not None and _normalize_dtype(
            expected_dtype
        ) != _normalize_dtype(dtype):
            raise UnsupportedKVFootprint(
                f"catalog KV dtype {expected_dtype!r} disagrees with AIC-resolved "
                f"dtype {dtype!r} for model={model_name!r}, backend={backend!r}"
            )
        values = tuple(
            math.ceil(float(model.get_kvcache_bytes_per_sequence(length)))
            for length in lengths
        )
        model_class = type(model).__name__

    by_length = dict(zip(lengths, values, strict=True))
    zero = by_length[0]
    one = by_length[1]
    marginal = one - zero
    affine = all(
        math.isclose(value, zero + length * marginal, rel_tol=0.0, abs_tol=1.0)
        for length, value in by_length.items()
    )
    if affine and zero == 0:
        growth_kind = "linear_zero_intercept"
    elif affine:
        growth_kind = "affine_fixed_state"
    else:
        growth_kind = "nonlinear"
    return KVRoleFootprint(
        source=source,
        layout=layout,
        scope=scope,
        dtype=dtype,
        dtype_bytes=dtype_bytes,
        model_class=model_class,
        tp=tp,
        attention_dp=attention_dp,
        moe_tp=moe_tp,
        moe_ep=moe_ep,
        reference_length=reference_length,
        bytes_at_zero=zero,
        bytes_at_reference=by_length[reference_length],
        marginal_bytes_per_token=marginal,
        growth_kind=growth_kind,
        probe_lengths=lengths,
        probe_bytes=values,
    )


def resolve_role_footprint(
    sample: dict[str, Any],
    *,
    role: str,
    spec: KVFootprintSpec,
    reference_length: int,
) -> KVRoleFootprint:
    """Resolve one role's candidate-specific per-rank footprint."""

    tp, attention_dp, moe_tp, moe_ep = _shape_fields(sample, role)
    block_size = int(sample[f"{role}_block_size"])
    return _resolve_role_cached(
        source=spec.source,
        layout=spec.layout,
        scope=spec.scope,
        expected_dtype=spec.dtype,
        explicit_bytes_per_token=spec.explicit_bytes_per_token,
        model_name=str(sample["model_name"]),
        backend=str(sample["backend"]),
        tp=tp,
        attention_dp=attention_dp,
        moe_tp=moe_tp,
        moe_ep=moe_ep,
        nextn=int(sample.get("aic_nextn") or 0),
        reference_length=reference_length,
        block_size=block_size,
        gemm_quant_mode=sample.get("aic_gemm_dtype"),
        moe_quant_mode=sample.get("aic_moe_dtype"),
        fmha_quant_mode=sample.get("aic_fmha_dtype"),
        kvcache_quant_mode=sample.get("aic_kv_cache_dtype"),
        comm_quant_mode=sample.get("aic_comm_dtype"),
    )


def _scalar_for_kvbm(
    footprint: KVRoleFootprint,
    *,
    workload: Workload,
    policy: KVFootprintPolicy,
) -> tuple[int, str]:
    if footprint.is_zero_intercept_linear:
        return footprint.marginal_bytes_per_token, "exact_linear"
    if workload.is_synthetic and policy.nonlinear_kvbm == "fixed_synthetic_average":
        return footprint.average_bytes_per_token, (
            "fixed_synthetic_sequence_average; aggregate bytes at the configured "
            "ISL are preserved, but per-block position timing is approximate"
        )
    raise UnsupportedKVFootprint(
        "KVBM's uniform bytes-per-block contract cannot represent "
        f"growth_kind={footprint.growth_kind!r} for this workload; use a fixed "
        "synthetic workload with nonlinear_kvbm='fixed_synthetic_average' or "
        "disable G2"
    )


def resolve_candidate_kv_footprint(
    sample: dict[str, Any], workload: Workload
) -> dict[str, Any]:
    """Attach role footprints and the two replay byte scalars to ``sample``.

    ``kv_bytes_per_token`` sizes rank-local KVBM blocks.
    ``kv_transfer_bytes_per_token`` sizes P/D handoff independently.
    """

    raw_spec = sample.get("kv_footprint")
    if raw_spec is None:
        return sample  # legacy SmartSearchConfig: retain the historical scalar path.
    spec = KVFootprintSpec.model_validate(raw_spec)
    policy = KVFootprintPolicy.model_validate(sample.get("kv_footprint_policy") or {})
    if workload.is_synthetic:
        assert workload.isl is not None
        reference_length = int(workload.isl)
    else:
        reference_length = max(1, int(sample.get("context_length") or 8192))

    mode = str(sample["deployment_mode"])
    roles = ("agg",) if mode == "agg" else ("prefill", "decode")
    resolved = {
        role: resolve_role_footprint(
            sample, role=role, spec=spec, reference_length=reference_length
        )
        for role in roles
    }
    for role, footprint in resolved.items():
        sample[f"{role}_kv_footprint"] = asdict(footprint)

    offload_enabled = (
        int(sample.get("num_g2_blocks") or 0) > 0
        or float(sample.get("g2_capacity_gib") or 0.0) > 0.0
    )
    if offload_enabled:
        kvbm_role = "agg" if mode == "agg" else "prefill"
        role_prefix = "" if kvbm_role == "agg" else f"{kvbm_role}_"
        tp = int(sample[f"{role_prefix}tp"])
        attention_dp = int(sample[f"{role_prefix}attention_dp"])
        replicas = int(
            sample["replicas"]
            if kvbm_role == "agg"
            else sample[f"{kvbm_role}_replicas"]
        )
        physical_rank_pools = tp * attention_dp * replicas
        kvbm_bpt, approximation = _scalar_for_kvbm(
            resolved[kvbm_role], workload=workload, policy=policy
        )
        sample["kv_bytes_per_token"] = kvbm_bpt
        sample["kvbm_footprint_contract"] = approximation
        sample["kvbm_physical_bytes_per_token_estimate"] = kvbm_bpt * tp
        capacity_gib = float(sample.get("g2_capacity_gib") or 0.0)
        if capacity_gib > 0.0:
            block_size = int(sample[f"{kvbm_role}_block_size"])
            scope = str(sample.get("g2_capacity_scope") or "deployment_total")
            divisor = kvbm_bpt * block_size
            if scope == "deployment_total":
                divisor *= physical_rank_pools
            blocks = math.floor(capacity_gib * 1024**3 / divisor)
            if blocks <= 0:
                raise UnsupportedKVFootprint(
                    f"G2 capacity {capacity_gib} GiB holds no {block_size}-token KV block"
                )
            sample["num_g2_blocks"] = blocks
            per_rank_gib = blocks * block_size * kvbm_bpt / 1024**3
            sample["g2_effective_capacity_gib_per_rank"] = per_rank_gib
            sample["g2_effective_total_capacity_gib"] = (
                per_rank_gib * physical_rank_pools
            )
        elif int(sample.get("num_g2_blocks") or 0) > 0:
            block_size = int(sample[f"{kvbm_role}_block_size"])
            per_rank_gib = (
                int(sample["num_g2_blocks"]) * block_size * kvbm_bpt / 1024**3
            )
            sample["g2_effective_capacity_gib_per_rank"] = per_rank_gib
            sample["g2_effective_total_capacity_gib"] = (
                per_rank_gib * physical_rank_pools
            )
            sample["g2_capacity_gib"] = sample["g2_effective_total_capacity_gib"]
            sample["g2_capacity_scope"] = "deployment_total"

    if mode == "disagg" and sample.get("kv_transfer_bandwidth") is not None:
        prefill = resolved["prefill"]
        decode = resolved["decode"]
        timing_mode = str(sample.get("kv_transfer_timing_mode") or "full_prompt")
        if workload.is_trace_based and not (
            prefill.is_zero_intercept_linear and decode.is_zero_intercept_linear
        ):
            raise UnsupportedKVFootprint(
                "one P/D byte scalar cannot represent a heterogeneous trace for an "
                f"AIC footprint with growth {prefill.growth_kind}/{decode.growth_kind}"
            )
        if timing_mode == "destination_missing" and not (
            prefill.is_zero_intercept_linear and decode.is_zero_intercept_linear
        ):
            raise UnsupportedKVFootprint(
                "destination_missing P/D timing requires a zero-intercept linear KV "
                "footprint until DynoSim accepts a sequence-position byte curve"
            )
        decode_total_at_reference = decode.bytes_at_reference * decode.tp
        if workload.is_synthetic:
            critical_link_bytes = max(
                decode.bytes_at_reference,
                math.ceil(decode_total_at_reference / prefill.tp),
            )
            transfer_bpt = math.ceil(critical_link_bytes / reference_length)
            transfer_contract = (
                "exact_fixed_prompt_ideal_balanced_reshard_critical_link"
            )
        else:
            decode_total_per_token = decode.marginal_bytes_per_token * decode.tp
            transfer_bpt = max(
                decode.marginal_bytes_per_token,
                math.ceil(decode_total_per_token / prefill.tp),
            )
            transfer_contract = "exact_linear_ideal_balanced_reshard_critical_link"
        sample["kv_transfer_bytes_per_token"] = transfer_bpt
        sample["pd_transfer_footprint_contract"] = transfer_contract
        sample["pd_prefill_bytes_per_sequence_per_rank"] = prefill.bytes_at_reference
        sample["pd_decode_bytes_per_sequence_per_rank"] = decode.bytes_at_reference
        # Cost/accounting estimate, separate from critical-path timing. A
        # prefill->decode handoff must materialize every destination rank's KV.
        # The scalar above models the ideal balanced source fan-out and the
        # busiest destination link; shared-fabric contention remains outside
        # DynoSim's one-bandwidth contract.
        sample["pd_transfer_physical_bytes_per_request_estimate"] = (
            decode_total_at_reference
        )
        sample["pd_transfer_physical_bytes_per_token_estimate"] = math.ceil(
            sample["pd_transfer_physical_bytes_per_request_estimate"] / reference_length
        )
        sample["pd_transfer_physical_bytes_contract"] = (
            "destination_materialized_total; ideal_balanced_reshard; "
            "shared_fabric_contention_not_modeled"
        )
    return sample
