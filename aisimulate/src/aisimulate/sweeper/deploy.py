# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate an unrolled backend sample into a replay deployment specification."""

from __future__ import annotations

from typing import Any

from .replay import BackendDeploymentSpec, canonical_json


def _role_prefix(role: str) -> str:
    """Field prefix in the unrolled sample for a role (empty for agg shape fields)."""
    return "" if role == "agg" else f"{role}_"


def _engine_args_payload(
    sample: dict[str, Any], role: str, *, backend_version: str
) -> dict[str, Any]:
    """Build the runner-neutral engine argument payload for one role."""
    prefix = _role_prefix(role)
    tp = int(sample[f"{prefix}tp"])
    attention_dp = int(sample[f"{prefix}attention_dp"])
    moe_tp = int(sample[f"{prefix}moe_tp"])
    moe_ep = int(sample[f"{prefix}moe_ep"])
    payload: dict[str, Any] = {
        "worker_type": "aggregated" if role == "agg" else role,
        "engine_type": sample["backend"],
        "aic_backend": sample["backend"],
        "aic_backend_version": backend_version,
        "aic_system": sample["hardware_sku"],
        "aic_model_path": sample["model_name"],
        "aic_tp_size": tp,
        "aic_attention_dp_size": attention_dp,
        "max_num_batched_tokens": int(sample[f"{role}_max_num_batched_tokens"]),
        "max_num_seqs": int(sample[f"{role}_max_num_seqs"]),
        "block_size": int(sample[f"{role}_block_size"]),
        "gpu_memory_utilization": float(sample[f"{role}_gpu_memory_utilization"]),
        "enable_prefix_caching": bool(sample[f"{role}_enable_prefix_caching"]),
    }
    if sample["database_mode"] == "ANALYTICAL":
        payload["aic_database_mode"] = "ANALYTICAL"
        payload["aic_analytical_config"] = canonical_json(
            sample.get("analytical_config", {})
        )
    if moe_tp * moe_ep > 1:
        payload["aic_moe_tp_size"] = moe_tp
        payload["aic_moe_ep_size"] = moe_ep
    if sample.get("aic_nextn") is not None:
        payload["aic_nextn"] = int(sample["aic_nextn"])
    for field in (
        "aic_gemm_dtype",
        "aic_moe_dtype",
        "aic_fmha_dtype",
        "aic_kv_cache_dtype",
        "aic_comm_dtype",
    ):
        if sample.get(field) is not None:
            payload[field] = str(sample[field])
    if sample.get("startup_time") is not None:
        payload["startup_time"] = float(sample["startup_time"])
    pd_transfer = sample.get("pd_transfer")
    if role != "agg" and pd_transfer is not None:
        if not isinstance(pd_transfer, dict):
            raise TypeError("pd_transfer must be a mapping")
        payload["kv_bytes_per_token"] = int(pd_transfer["kv_bytes_per_token"])
        payload["kv_transfer_bandwidth"] = float(pd_transfer["bandwidth_gb_s"])
        payload["kv_transfer_timing_mode"] = str(pd_transfer["timing_mode"])
    return payload


def build_backend_deployment(
    sample: dict[str, Any], *, backend_version: str
) -> BackendDeploymentSpec:
    """Build the Dynamo-independent backend part of a :class:`ReplaySpec`."""
    mode = sample["deployment_mode"]
    common = {
        "deployment_mode": mode,
        "backend": sample["backend"],
        "backend_version": backend_version,
        "parallel_config": {
            key: value
            for key, value in sample.items()
            if key
            in {
                "tp",
                "pp",
                "attention_dp",
                "moe_tp",
                "moe_ep",
                "strategy",
                "replicas",
                "prefill_tp",
                "prefill_pp",
                "prefill_attention_dp",
                "prefill_moe_tp",
                "prefill_moe_ep",
                "prefill_strategy",
                "prefill_replicas",
                "decode_tp",
                "decode_pp",
                "decode_attention_dp",
                "decode_moe_tp",
                "decode_moe_ep",
                "decode_strategy",
                "decode_replicas",
            }
        },
    }
    if mode == "agg":
        return BackendDeploymentSpec(
            agg_engine_args=_engine_args_payload(
                sample, "agg", backend_version=backend_version
            ),
            num_workers=int(sample["replicas"]),
            **common,
        )
    return BackendDeploymentSpec(
        prefill_engine_args=_engine_args_payload(
            sample, "prefill", backend_version=backend_version
        ),
        decode_engine_args=_engine_args_payload(
            sample, "decode", backend_version=backend_version
        ),
        num_prefill_workers=int(sample["prefill_replicas"]),
        num_decode_workers=int(sample["decode_replicas"]),
        **common,
    )
