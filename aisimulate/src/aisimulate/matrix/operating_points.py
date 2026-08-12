# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIC-backed closed-loop operating-point discovery for Matrix studies.

AIC is used only to select auditable concurrency anchors.  DynoSim still
evaluates every deployment/router/KVBM candidate at each selected anchor and
is the authority for the formal SLA and Pareto result.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from aiconfigurator.sdk import common
from aiconfigurator.sdk.task_v2 import Task
from pydantic import BaseModel, ConfigDict, Field

from .expand import MatrixCell
from .prune import StaticPruneResult
from .workload import LoadAxis, PreparedWorkload


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AICOperatingPoint(_StrictModel):
    mode: str
    concurrency: int = Field(ge=1)
    request_rate: float = Field(gt=0)
    tokens_per_second_per_gpu: float = Field(ge=0)
    tokens_per_second_per_user: float = Field(ge=0)
    num_total_gpus: int = Field(ge=1)
    parallel_key: str
    config: dict[str, Any]
    selected: bool = False
    selection_reasons: list[str] = Field(default_factory=list)


class AICOperatingPointResolution(_StrictModel):
    mode: str
    points: list[AICOperatingPoint]
    selected_concurrencies: list[int]
    source: str = "aic_operating_points"
    compression_applied: bool = False


def _enum(enum_type: type, value: str):
    try:
        return enum_type[value]
    except KeyError as exc:  # pragma: no cover - catalog validation normally catches this
        raise ValueError(f"unknown AIC quantization value {value!r}") from exc


def _retained_payloads(prune_result: StaticPruneResult) -> list[dict[str, Any]]:
    return [decision.config for decision in prune_result.decisions if decision.retained]


def _role_candidates(
    payloads: list[dict[str, Any]], role: str
) -> dict[str, list[int]]:
    shapes = [payload[role] for payload in payloads]
    return {
        "num_gpu": sorted(
            {
                int(shape["tp"])
                * int(shape["pp"])
                * int(shape["attention_dp"])
                for shape in shapes
            }
        ),
        "tp": sorted({int(shape["tp"]) for shape in shapes}),
        "pp": sorted({int(shape["pp"]) for shape in shapes}),
        "dp": sorted({int(shape["attention_dp"]) for shape in shapes}),
        "moe_tp": sorted({int(shape["moe_tp"]) for shape in shapes}),
        "moe_ep": sorted({int(shape["moe_ep"]) for shape in shapes}),
        "cp": sorted({int(shape.get("cp", 1)) for shape in shapes}),
    }


def _base_task_kwargs(
    cell: MatrixCell, prepared: PreparedWorkload, mode: str
) -> dict[str, Any]:
    quant = cell.model_variant.quantization
    context_length = int(
        prepared.summary.get("max_context_length")
        or prepared.summary.get("context_length")
        or (int(prepared.replay_args["isl"]) + int(prepared.replay_args["osl"]))
    )
    return {
        "serving_mode": mode,
        "isl": int(prepared.replay_args["isl"]),
        "osl": int(prepared.replay_args["osl"]),
        "prefix": int(prepared.summary.get("prefix_length", 0)),
        "ttft": float(cell.sla.ttft_ms),
        # AIC exposes mean TPOT rather than a separate TPS/u gate. Use the
        # stricter equivalent so an AIC anchor cannot pass only because the two
        # user thresholds were configured inconsistently.
        "tpot": min(
            float(cell.sla.tpot_ms),
            1000.0 / float(cell.sla.tps_per_user),
        ),
        "pareto_sweep": False,
        "total_gpus": int(cell.search_policy.spica.search_space.get("gpu_budget", 32)),
        "database_mode": (
            common.DatabaseMode.SILICON.name
            if cell.variant_resolution.data_mode == "silicon"
            else common.DatabaseMode.HYBRID.name
        ),
        "max_seq_len": context_length,
        "gemm_quant_mode": _enum(common.GEMMQuantMode, quant.gemm),
        "moe_quant_mode": _enum(common.MoEQuantMode, quant.moe),
        "fmha_quant_mode": _enum(common.FMHAQuantMode, quant.fmha),
        "kvcache_quant_mode": _enum(common.KVCacheQuantMode, quant.kv_cache),
        "comm_quant_mode": _enum(common.CommQuantMode, quant.comm),
    }


def _task_for_mode(
    cell: MatrixCell,
    prepared: PreparedWorkload,
    mode: str,
    prune_result: StaticPruneResult,
) -> tuple[Task, list[dict[str, Any]]]:
    payloads = _retained_payloads(prune_result)
    if not payloads:
        raise ValueError("AIC operating-point discovery has no retained parallel config")
    base = _base_task_kwargs(cell, prepared, mode)
    variant = cell.model_variant
    backend_version = (
        variant.backend_version or cell.variant_resolution.support.backend_version
    )
    if mode == "agg":
        shapes = payloads
        candidates = {
            "num_gpu": sorted(
                {
                    int(shape["tp"])
                    * int(shape["pp"])
                    * int(shape["attention_dp"])
                    for shape in shapes
                }
            ),
            "tp": sorted({int(shape["tp"]) for shape in shapes}),
            "pp": sorted({int(shape["pp"]) for shape in shapes}),
            "dp": sorted({int(shape["attention_dp"]) for shape in shapes}),
            "moe_tp": sorted({int(shape["moe_tp"]) for shape in shapes}),
            "moe_ep": sorted({int(shape["moe_ep"]) for shape in shapes}),
            "cp": sorted({int(shape.get("cp", 1)) for shape in shapes}),
        }
        return (
            Task(
                **base,
                model_path=variant.model_path,
                system_name=cell.sku.aic_system,
                backend_name=variant.backend,
                backend_version=backend_version,
                agg_num_gpu_candidates=candidates["num_gpu"],
                agg_tp_candidates=candidates["tp"],
                agg_pp_candidates=candidates["pp"],
                agg_dp_candidates=candidates["dp"],
                agg_moe_tp_candidates=candidates["moe_tp"],
                agg_moe_ep_candidates=candidates["moe_ep"],
                agg_cp_candidates=candidates["cp"],
            ),
            payloads,
        )

    prefill = _role_candidates(payloads, "prefill")
    decode = _role_candidates(payloads, "decode")
    total_gpus = sorted(
        {
            int(payload["prefill"]["tp"])
            * int(payload["prefill"]["pp"])
            * int(payload["prefill"]["attention_dp"])
            * int(payload["prefill"]["replicas"])
            + int(payload["decode"]["tp"])
            * int(payload["decode"]["pp"])
            * int(payload["decode"]["attention_dp"])
            * int(payload["decode"]["replicas"])
            for payload in payloads
        }
    )
    role_quant = {
        "gemm_quant_mode": base.pop("gemm_quant_mode"),
        "moe_quant_mode": base.pop("moe_quant_mode"),
        "fmha_quant_mode": base.pop("fmha_quant_mode"),
        "kvcache_quant_mode": base.pop("kvcache_quant_mode"),
        "comm_quant_mode": base.pop("comm_quant_mode"),
    }
    return (
        Task(
            **base,
            prefill_model_path=variant.model_path,
            prefill_system_name=cell.sku.aic_system,
            prefill_backend_name=variant.backend,
            prefill_backend_version=backend_version,
            decode_model_path=variant.model_path,
            decode_system_name=cell.sku.aic_system,
            decode_backend_name=variant.backend,
            decode_backend_version=backend_version,
            **{f"prefill_{key}": value for key, value in role_quant.items()},
            **{f"decode_{key}": value for key, value in role_quant.items()},
            prefill_num_gpu_candidates=prefill["num_gpu"],
            prefill_tp_candidates=prefill["tp"],
            prefill_pp_candidates=prefill["pp"],
            prefill_dp_candidates=prefill["dp"],
            prefill_moe_tp_candidates=prefill["moe_tp"],
            prefill_moe_ep_candidates=prefill["moe_ep"],
            prefill_cp_candidates=prefill["cp"],
            decode_num_gpu_candidates=decode["num_gpu"],
            decode_tp_candidates=decode["tp"],
            decode_pp_candidates=decode["pp"],
            decode_dp_candidates=decode["dp"],
            decode_moe_tp_candidates=decode["moe_tp"],
            decode_moe_ep_candidates=decode["moe_ep"],
            decode_cp_candidates=decode["cp"],
            num_gpu_per_replica=total_gpus,
            max_gpu_per_replica=max(total_gpus),
            max_prefill_workers=max(int(p["prefill"]["replicas"]) for p in payloads),
            max_decode_workers=max(int(p["decode"]["replicas"]) for p in payloads),
        ),
        payloads,
    )


def _matches_shape(row: dict[str, Any], shape: dict[str, Any], prefix: str = "") -> bool:
    return all(
        int(row.get(f"{prefix}{row_name}", -1)) == int(shape[field_name])
        for row_name, field_name in (
            ("tp", "tp"),
            ("pp", "pp"),
            ("dp", "attention_dp"),
            ("moe_tp", "moe_tp"),
            ("moe_ep", "moe_ep"),
        )
    ) and int(row.get(f"{prefix}cp", 1)) == int(shape.get("cp", 1))


def _basic_row(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if key.startswith("_"):
            continue
        if value is None or isinstance(value, (str, bool, int, float)):
            result[key] = value
        elif hasattr(value, "item"):
            result[key] = value.item()
        else:
            result[key] = str(value)
    return result


def _expand_and_filter_rows(
    mode: str, raw_rows: list[dict[str, Any]], payloads: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if mode == "agg":
        for raw in raw_rows:
            for payload in payloads:
                if not _matches_shape(raw, payload):
                    continue
                replicas = int(payload["replicas"])
                row = _basic_row(raw)
                row["replicas"] = replicas
                row["concurrency"] = int(round(float(raw["concurrency"]) * replicas))
                for key in ("request_rate", "seq/s", "tokens/s"):
                    if key in raw:
                        row[key] = float(raw[key]) * replicas
                row["num_total_gpus"] = int(raw["num_total_gpus"]) * replicas
                rows.append(row)
        return rows

    for raw in raw_rows:
        for payload in payloads:
            if not _matches_shape(raw, payload["prefill"], "(p)"):
                continue
            if not _matches_shape(raw, payload["decode"], "(d)"):
                continue
            if int(raw.get("(p)workers", -1)) != int(payload["prefill"]["replicas"]):
                continue
            if int(raw.get("(d)workers", -1)) != int(payload["decode"]["replicas"]):
                continue
            rows.append(_basic_row(raw))
    return rows


def _parallel_key(mode: str, row: dict[str, Any]) -> str:
    if mode == "agg":
        return (
            f"tp{row.get('tp')}-pp{row.get('pp')}-dp{row.get('dp')}-"
            f"mtp{row.get('moe_tp')}-mep{row.get('moe_ep')}-r{row.get('replicas', 1)}"
        )
    return (
        f"p:{row.get('(p)parallel')}x{row.get('(p)workers')}/"
        f"d:{row.get('(d)parallel')}x{row.get('(d)workers')}"
    )


def _static_pareto_indexes(points: list[AICOperatingPoint]) -> set[int]:
    front: set[int] = set()
    for index, point in enumerate(points):
        dominated = False
        for other_index, other in enumerate(points):
            if index == other_index:
                continue
            no_worse = (
                other.tokens_per_second_per_gpu >= point.tokens_per_second_per_gpu
                and other.tokens_per_second_per_user >= point.tokens_per_second_per_user
            )
            better = (
                other.tokens_per_second_per_gpu > point.tokens_per_second_per_gpu
                or other.tokens_per_second_per_user > point.tokens_per_second_per_user
            )
            if no_worse and better:
                dominated = True
                break
        if not dominated:
            front.add(index)
    return front


def _compress(values: list[int], limit: int) -> list[int]:
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[-1]]
    logs = [math.log(value) for value in values]
    chosen: set[int] = {values[0], values[-1]}
    for index in range(limit):
        target = logs[0] + (logs[-1] - logs[0]) * index / (limit - 1)
        chosen.add(values[min(range(len(values)), key=lambda i: abs(logs[i] - target))])
    if len(chosen) < limit:
        for value in values:
            chosen.add(value)
            if len(chosen) == limit:
                break
    ordered = sorted(chosen)
    if len(ordered) <= limit:
        return ordered
    # Quantile collisions can make the first pass exceed the limit; remove the
    # least isolated interior point while always preserving both boundaries.
    while len(ordered) > limit:
        remove_index = min(
            range(1, len(ordered) - 1),
            key=lambda index: (
                min(
                    math.log(ordered[index]) - math.log(ordered[index - 1]),
                    math.log(ordered[index + 1]) - math.log(ordered[index]),
                ),
                ordered[index],
            ),
        )
        ordered.pop(remove_index)
    return ordered


def resolve_aic_operating_points(
    cell: MatrixCell,
    prepared: PreparedWorkload,
    mode: str,
    prune_result: StaticPruneResult,
    load_axis: LoadAxis,
) -> AICOperatingPointResolution:
    """Return bounded, deterministic AIC-derived concurrency anchors."""

    if load_axis.source != "aic_operating_points":
        raise ValueError("resolve_aic_operating_points requires an AIC load axis")
    task, payloads = _task_for_mode(cell, prepared, mode, prune_result)
    frame = task.run()
    if frame is None or frame.empty:
        raise ValueError(f"AIC produced no SLA-feasible {mode} operating point")
    raw_rows = frame.to_dict(orient="records")
    rows = _expand_and_filter_rows(mode, raw_rows, payloads)
    points: list[AICOperatingPoint] = []
    for row in rows:
        concurrency = int(round(float(row.get("concurrency", 0))))
        request_rate = float(row.get("request_rate", row.get("seq/s", 0.0)))
        if concurrency < int(load_axis.min_value or 1):
            continue
        if load_axis.max_value is not None and concurrency > load_axis.max_value:
            continue
        if request_rate <= 0 or int(row.get("num_total_gpus", 0)) <= 0:
            continue
        points.append(
            AICOperatingPoint(
                mode=mode,
                concurrency=concurrency,
                request_rate=request_rate,
                tokens_per_second_per_gpu=float(row.get("tokens/s/gpu", 0.0)),
                tokens_per_second_per_user=float(row.get("tokens/s/user", 0.0)),
                num_total_gpus=int(row["num_total_gpus"]),
                parallel_key=_parallel_key(mode, row),
                config=row,
            )
        )
    if not points:
        raise ValueError(
            f"AIC {mode} operating points do not overlap the configured concurrency bounds"
        )

    reasons: dict[int, set[str]] = defaultdict(set)
    if load_axis.include_static_pareto:
        for index in _static_pareto_indexes(points):
            reasons[points[index].concurrency].add("aic_static_pareto")
    if load_axis.topk_per_parallel:
        grouped: dict[str, list[AICOperatingPoint]] = defaultdict(list)
        for point in points:
            grouped[point.parallel_key].append(point)
        for group in grouped.values():
            ranked = sorted(
                group,
                key=lambda point: (
                    -point.tokens_per_second_per_gpu,
                    -point.tokens_per_second_per_user,
                    -point.request_rate,
                    point.concurrency,
                ),
            )
            for point in ranked[: load_axis.topk_per_parallel]:
                reasons[point.concurrency].add("aic_parallel_topk")
    if not reasons:
        for point in points:
            reasons[point.concurrency].add("aic_feasible")

    selected = sorted(reasons)
    max_points = int(load_axis.max_points or len(selected))
    compressed = _compress(selected, max_points)
    compression_applied = compressed != selected
    if compression_applied:
        for concurrency in compressed:
            reasons[concurrency].add("deterministic_log_compression")
    selected_set = set(compressed)
    resolved_points = [
        point.model_copy(
            update={
                "selected": point.concurrency in selected_set,
                "selection_reasons": sorted(reasons.get(point.concurrency, set())),
            }
        )
        for point in points
    ]
    return AICOperatingPointResolution(
        mode=mode,
        points=resolved_points,
        selected_concurrencies=compressed,
        compression_applied=compression_applied,
    )
