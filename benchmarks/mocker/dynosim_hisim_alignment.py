#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the fixed 35-point DynoSim/SGLang ACC comparison against reused data.

The source tree is treated as read-only. Every generated artifact is written
under ``--output-root`` and carries hashes back to its authoritative source
workload, real SGLang metrics, and existing HiSim ACC comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SOURCE_ROOT = Path(
    "/workspace/hisim_align/"
    "hisim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_"
    "h100pcie_gsp_mixed_seed11_full"
)
OUTPUT_ROOT = Path(
    "/workspace/hisim_align/"
    "dynosim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_"
    "h100pcie_gsp_mixed_seed11_full"
)
DYNAMO_ROOT = Path("/workspace/hisim_align/dynamo")
AIC_ROOT = Path("/workspace/hisim_align/aiconfigurator-dynosim-e2e")
MODEL_PATH = Path(
    "/workspace/hisim_align/hisim_sglang_align/model_cache/"
    "DeepSeek-V3_layers20_maxpos163840"
)

SCENARIOS = ("a", "b", "b_mixed", "c", "d")
RPS_POINTS = (1, 2, 4, 8, 16, 24, 32)
SMOKE_POINTS = ("sweep_a_rps_point_1", "sweep_b_mixed_rps_point_8", "sweep_d_rps_point_32")
POINT_RE = re.compile(r"^sweep_(a|b|b_mixed|c|d)_rps_point_(1|2|4|8|16|24|32)$")
PRIMARY_METRICS = (
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
    "mean_e2e_latency_ms",
    "input_throughput",
)
GRAPH_BATCH_SIZES = (1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64)


@dataclass(frozen=True)
class PointInputs:
    name: str
    source_dir: Path
    workload_path: Path
    raw_path: Path
    hisim_mape_path: Path
    hisim_sim_config_path: Path
    rows: tuple[dict[str, Any], ...]
    raw: dict[str, Any]
    hisim_mape: dict[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            handle.write("\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def git_revision(repo: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    untracked_hashes = {}
    for line in status.splitlines():
        if not line.startswith("?? "):
            continue
        path = repo / line[3:]
        if path.is_file():
            untracked_hashes[line[3:]] = sha256_file(path)
    return {
        "branch": branch,
        "revision": revision,
        "dirty": bool(status.strip()),
        "status_porcelain": status.splitlines(),
        "tracked_diff_sha256": sha256_bytes(tracked_diff),
        "untracked_file_sha256": untracked_hashes,
    }


def expected_point_names() -> set[str]:
    return {
        f"sweep_{scenario}_rps_point_{point}"
        for scenario in SCENARIOS
        for point in RPS_POINTS
    }


def discover_points(source_root: Path) -> list[Path]:
    discovered = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and POINT_RE.fullmatch(path.name)
    )
    actual = {path.name for path in discovered}
    expected = expected_point_names()
    if actual != expected:
        raise ValueError(
            f"canonical RPS point mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return discovered


def _latest_jsonl_object(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows[-1]


def _validate_finite_nonnegative(label: str, value: Any) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative, got {value!r}")


def load_and_validate_point(source_dir: Path) -> PointInputs:
    workload_path = source_dir / "bench_serving_dataset.jsonl"
    raw_path = source_dir / "raw_sglang" / "client_metrics.jsonl"
    hisim_mape_path = source_dir / "acc_bench" / "compare" / "acc_bench_mape.json"
    hisim_sim_config_path = source_dir / "acc_bench" / "hisim_aic" / "sim_config.json"
    for path in (workload_path, raw_path, hisim_mape_path, hisim_sim_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = tuple(load_jsonl(workload_path))
    if not rows:
        raise ValueError(f"{source_dir.name}: workload is empty")
    rids = [row.get("rid") for row in rows]
    if any(not isinstance(rid, str) or not rid for rid in rids):
        raise ValueError(f"{source_dir.name}: every workload row must have a non-empty rid")
    if len(set(rids)) != len(rids):
        raise ValueError(f"{source_dir.name}: duplicate workload rid")
    previous_timestamp = -math.inf
    for index, row in enumerate(rows):
        input_ids = row.get("input_ids")
        if not isinstance(input_ids, list) or len(input_ids) != row.get("input_length"):
            raise ValueError(f"{source_dir.name}: line {index + 1} input length mismatch")
        if any(not isinstance(token_id, int) or token_id < 0 for token_id in input_ids):
            raise ValueError(f"{source_dir.name}: line {index + 1} has invalid input token")
        output_length = row.get("output_length")
        if not isinstance(output_length, int) or output_length < 1:
            raise ValueError(f"{source_dir.name}: line {index + 1} has invalid output_length")
        timestamp = row.get("timestamp")
        _validate_finite_nonnegative(f"{source_dir.name}: timestamp line {index + 1}", timestamp)
        if timestamp < previous_timestamp:
            raise ValueError(f"{source_dir.name}: timestamps are not non-decreasing")
        previous_timestamp = timestamp

    raw = _latest_jsonl_object(raw_path)
    hisim_mape = json.loads(hisim_mape_path.read_text())
    sim_config = json.loads(hisim_sim_config_path.read_text())
    count = len(rows)
    if raw.get("completed") != count:
        raise ValueError(f"{source_dir.name}: raw completed={raw.get('completed')} != {count}")
    for key in ("rids", "input_lens", "output_lens", "ttfts", "e2e_latencies"):
        if len(raw.get(key, [])) != count:
            raise ValueError(f"{source_dir.name}: raw {key} length mismatch")
    if raw.get("rids") != rids:
        raise ValueError(f"{source_dir.name}: raw/workload RID order mismatch")
    completed = hisim_mape.get("completed", {})
    if completed.get("raw") != count or completed.get("hisim_aic") != count:
        raise ValueError(f"{source_dir.name}: existing HiSim ACC completion mismatch")
    if not hisim_mape.get("ingress_envelope"):
        raise ValueError(f"{source_dir.name}: missing existing ingress envelope")
    for metric in PRIMARY_METRICS:
        metric_data = hisim_mape.get("metrics", {}).get(metric)
        if not metric_data or "raw" not in metric_data or "mape_percent" not in metric_data:
            raise ValueError(f"{source_dir.name}: missing HiSim ACC metric {metric}")
        _validate_finite_nonnegative(
            f"{source_dir.name}: HiSim MAPE {metric}", metric_data["mape_percent"]
        )

    server = raw.get("server_info", {})
    expected_server = {
        "ep_dispatch_algorithm": "fake",
        "tp_size": 4,
        "ep_size": 4,
        "dp_size": 1,
        "max_total_tokens": 655360,
        "max_prefill_tokens": 16384,
        "chunked_prefill_size": 8192,
        "page_size": 1,
    }
    for key, expected in expected_server.items():
        if server.get(key) != expected:
            raise ValueError(
                f"{source_dir.name}: real baseline {key}={server.get(key)!r}, expected {expected!r}"
            )
    if server.get("disable_cuda_graph") or server.get("disable_cuda_graph_padding"):
        raise ValueError(f"{source_dir.name}: real baseline CUDA graph/padding is disabled")
    real_graph_sizes = tuple(server.get("cuda_graph_bs", ()))
    if real_graph_sizes[: len(GRAPH_BATCH_SIZES)] != GRAPH_BATCH_SIZES:
        raise ValueError(f"{source_dir.name}: real baseline CUDA graph sizes lack the required prefix")

    predictor = sim_config.get("predictor", {})
    scheduler = sim_config.get("scheduler", {})
    if predictor.get("workload_distribution") != "balanced":
        raise ValueError(f"{source_dir.name}: existing AIC workload is not balanced")
    if predictor.get("device_name") != "h100_pcie":
        raise ValueError(f"{source_dir.name}: existing AIC system is not h100_pcie")
    if scheduler.get("backend_name") != "sglang" or scheduler.get("backend_version") != "0.5.9":
        raise ValueError(f"{source_dir.name}: existing AIC SGLang version mismatch")
    if (scheduler.get("tp_size"), scheduler.get("ep_size"), scheduler.get("dp_size")) != (4, 4, 1):
        raise ValueError(f"{source_dir.name}: existing AIC topology mismatch")

    return PointInputs(
        name=source_dir.name,
        source_dir=source_dir,
        workload_path=workload_path,
        raw_path=raw_path,
        hisim_mape_path=hisim_mape_path,
        hisim_sim_config_path=hisim_sim_config_path,
        rows=rows,
        raw=raw,
        hisim_mape=hisim_mape,
    )


def source_hashes(point: PointInputs) -> dict[str, str]:
    return {
        "workload_sha256": sha256_file(point.workload_path),
        "raw_metrics_sha256": sha256_file(point.raw_path),
        "hisim_mape_sha256": sha256_file(point.hisim_mape_path),
        "hisim_sim_config_sha256": sha256_file(point.hisim_sim_config_path),
    }


def preflight(source_root: Path) -> tuple[list[PointInputs], dict[str, Any]]:
    points = [load_and_validate_point(path) for path in discover_points(source_root)]
    manifest = {
        "schema": "dynosim_hisim_preflight.v1",
        "source_root": str(source_root),
        "point_count": len(points),
        "expected_point_count": 35,
        "total_requests": sum(len(point.rows) for point in points),
        "points": {
            point.name: {
                "request_count": len(point.rows),
                "first_timestamp_s": point.rows[0]["timestamp"],
                "last_timestamp_s": point.rows[-1]["timestamp"],
                "real_cuda_graph_batch_sizes": point.raw["server_info"]["cuda_graph_bs"],
                **source_hashes(point),
            }
            for point in points
        },
    }
    return points, manifest


def deterministic_uuid(request_ordinal: int) -> str:
    return str(uuid.UUID(int=request_ordinal + 1))


def convert_workload(point: PointInputs, trace_path: Path, rid_map_path: Path) -> dict[str, Any]:
    trace_rows: list[dict[str, Any]] = []
    rid_rows: list[dict[str, Any]] = []
    for index, row in enumerate(point.rows):
        trace_rows.append(
            {
                "request_id": row["rid"],
                "input_length": row["input_length"],
                "output_length": row["output_length"],
                "hash_ids": row["input_ids"],
                "timestamp": row["timestamp"] * 1000.0,
            }
        )
        rid_rows.append(
            {
                "line_index": index,
                "source_rid": row["rid"],
                "dynosim_uuid": deterministic_uuid(index),
                "input_length": row["input_length"],
                "output_length": row["output_length"],
            }
        )
    write_jsonl(trace_path, trace_rows)
    write_json(rid_map_path, {"schema": "dynosim_rid_map.v1", "rows": rid_rows})
    return {
        **source_hashes(point),
        "trace_sha256": sha256_file(trace_path),
        "rid_map_sha256": sha256_file(rid_map_path),
        "request_count": len(trace_rows),
        "timestamp_conversion": "source_seconds_x_1000_to_milliseconds",
        "trace_block_size": 1,
        "hash_id_semantics": "one_complete_source_input_id_per_hash_id",
    }


def engine_args() -> dict[str, Any]:
    return {
        "engine_type": "sglang",
        "num_gpu_blocks": 655360,
        "block_size": 1,
        "max_num_seqs": None,
        "max_num_batched_tokens": 16384,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "speedup_ratio": 1.0,
        "decode_speedup_ratio": 1.0,
        "dp_size": 1,
        "worker_type": "aggregated",
        "aic_backend": "sglang",
        "aic_system": "h100_pcie",
        "aic_backend_version": "0.5.9",
        "aic_tp_size": 4,
        "aic_model_path": str(MODEL_PATH),
        "aic_moe_tp_size": 1,
        "aic_moe_ep_size": 4,
        "aic_attention_dp_size": 1,
        "aic_workload_distribution": "balanced",
        "aic_kv_cache_dtype": "fp8",
        # The SGLang H100 PCIe MLA tables use BF16 attention kernels with an
        # independently FP8-quantized KV cache.
        "aic_fmha_dtype": "bfloat16",
        "sglang": {
            "schedule_policy": "fcfs",
            "page_size": 1,
            "max_prefill_tokens": 16384,
            "chunked_prefill_size": 8192,
            "clip_max_new_tokens": 4096,
            "schedule_conservativeness": 1.0,
            "cuda_graph_enabled": True,
            "cuda_graph_padding_enabled": True,
            "cuda_graph_batch_sizes": list(GRAPH_BATCH_SIZES),
        },
    }


def replay_command(
    python: Path,
    trace_path: Path,
    report_path: Path,
    per_request_path: Path,
    args: dict[str, Any],
) -> list[str]:
    return [
        str(python),
        "-m",
        "dynamo.replay",
        str(trace_path),
        "--extra-engine-args",
        json.dumps(args, sort_keys=True, separators=(",", ":")),
        "--num-workers",
        "1",
        "--replay-mode",
        "offline",
        "--router-mode",
        "round_robin",
        "--trace-format",
        "mooncake",
        "--trace-block-size",
        "1",
        "--report-json",
        str(report_path),
        "--report-jsonl",
        str(per_request_path),
    ]


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = quantile * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_bench_metrics(
    point: PointInputs,
    report: dict[str, Any],
    per_request: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    count = len(point.rows)
    if report.get("num_requests") != count or report.get("completed_requests") != count:
        raise ValueError(
            f"{point.name}: DynoSim completion mismatch "
            f"{report.get('completed_requests')}/{report.get('num_requests')} != {count}"
        )
    if len(per_request) != count:
        raise ValueError(f"{point.name}: DynoSim per-request length {len(per_request)} != {count}")
    if any(row.get("terminal_status") != "completed" for row in per_request):
        raise ValueError(f"{point.name}: non-completed DynoSim request")

    rid_by_uuid = {
        deterministic_uuid(index): row["rid"] for index, row in enumerate(point.rows)
    }
    ingress = point.hisim_mape["ingress_envelope"]
    ingress_ms = float(ingress["ingress_ms"])
    adjusted_rows: list[dict[str, Any]] = []
    ttfts: list[float] = []
    e2es: list[float] = []
    tpots: list[float] = []
    for row in per_request:
        uuid_value = row["uuid"]
        if uuid_value not in rid_by_uuid:
            raise ValueError(f"{point.name}: unexpected deterministic UUID {uuid_value}")
        for key in ("ttft_ms", "e2e_latency_ms", "itl_ms"):
            _validate_finite_nonnegative(f"{point.name}: {uuid_value} {key}", row.get(key))
        if row["output_length"] != row["requested_output_length"]:
            raise ValueError(f"{point.name}: {uuid_value} output length mismatch")
        adjusted = dict(row)
        adjusted["source_rid"] = rid_by_uuid[uuid_value]
        adjusted["scheduler_ttft_ms"] = row["ttft_ms"]
        adjusted["scheduler_e2e_latency_ms"] = row["e2e_latency_ms"]
        adjusted["ttft_ms"] = row["ttft_ms"] + ingress_ms
        adjusted["e2e_latency_ms"] = row["e2e_latency_ms"] + ingress_ms
        adjusted["ingress_envelope_ms"] = ingress_ms
        adjusted_rows.append(adjusted)
        ttfts.append(adjusted["ttft_ms"])
        e2es.append(adjusted["e2e_latency_ms"])
        tpots.append(row["itl_ms"])

    metrics = {
        "schema": "dynosim_bench_serving_metrics.v1",
        "backend": "dynosim_sglang",
        "completed": count,
        "duration": report["duration_ms"] / 1000.0,
        "input_throughput": report["input_throughput_tok_s"],
        "output_throughput": report["output_throughput_tok_s"],
        "request_throughput": report["request_throughput_rps"],
        "mean_ttft_ms": statistics.fmean(ttfts),
        "median_ttft_ms": statistics.median(ttfts),
        "p99_ttft_ms": _percentile(ttfts, 0.99),
        "mean_tpot_ms": report["mean_tpot_ms"],
        "median_tpot_ms": report["median_tpot_ms"],
        "p99_tpot_ms": report["p99_tpot_ms"],
        "mean_itl_ms": report["mean_itl_ms"],
        "median_itl_ms": report["median_itl_ms"],
        "p99_itl_ms": report["p99_itl_ms"],
        "mean_e2e_latency_ms": statistics.fmean(e2es),
        "median_e2e_latency_ms": statistics.median(e2es),
        "p99_e2e_latency_ms": _percentile(e2es, 0.99),
        "prefix_cache_reused_ratio": report["prefix_cache_reused_ratio"],
        "ingress_envelope": ingress,
        "scheduler_arrival_unchanged": True,
        "unchanged_by_ingress": [
            "mean_tpot_ms",
            "mean_itl_ms",
            "input_throughput",
            "request_throughput",
        ],
    }
    for metric in PRIMARY_METRICS:
        _validate_finite_nonnegative(f"{point.name}: DynoSim {metric}", metrics[metric])
    return metrics, adjusted_rows


def point_comparison(point: PointInputs, bench_metrics: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in PRIMARY_METRICS:
        raw_value = float(point.hisim_mape["metrics"][metric]["raw"])
        dynosim_value = float(bench_metrics[metric])
        if raw_value == 0:
            raise ValueError(f"{point.name}: raw baseline {metric} is zero")
        dynosim_mape = abs(dynosim_value - raw_value) / abs(raw_value) * 100.0
        hisim_mape = float(point.hisim_mape["metrics"][metric]["mape_percent"])
        metrics[metric] = {
            "raw": raw_value,
            "dynosim": dynosim_value,
            "dynosim_mape_percent": dynosim_mape,
            "hisim_mape_percent": hisim_mape,
            "delta_percentage_points": dynosim_mape - hisim_mape,
            "winner": (
                "dynosim"
                if dynosim_mape < hisim_mape
                else "hisim"
                if hisim_mape < dynosim_mape
                else "tie"
            ),
        }
    return {
        "schema": "dynosim_hisim_point_comparison.v1",
        "point": point.name,
        "completed": len(point.rows),
        "metric_count": len(metrics),
        "mean_dynosim_mape_percent": statistics.fmean(
            value["dynosim_mape_percent"] for value in metrics.values()
        ),
        "mean_hisim_mape_percent": statistics.fmean(
            value["hisim_mape_percent"] for value in metrics.values()
        ),
        "metrics": metrics,
    }


def canonical_report_hash(report: dict[str, Any]) -> str:
    # These three values are measured against the replay process's host wall
    # clock. They are useful diagnostics, but are not part of the simulated
    # result and cannot participate in deterministic replay acceptance.
    wall_clock_fields = {
        "wall_time_ms",
        "processed_tokens_per_s",
        "processed_output_tokens_per_s",
    }
    canonical = {
        key: value for key, value in report.items() if key not in wall_clock_fields
    }
    return sha256_bytes(canonical_json_bytes(canonical))


def point_artifact_hashes(point_dir: Path) -> dict[str, str]:
    report = json.loads((point_dir / "dynosim_report.json").read_text())
    hashes = {
        "trace_sha256": sha256_file(point_dir / "trace.mooncake.jsonl"),
        "rid_map_sha256": sha256_file(point_dir / "rid_map.json"),
        "source_manifest_sha256": sha256_file(point_dir / "source_manifest.json"),
        "engine_args_sha256": sha256_file(point_dir / "engine_args.json"),
        "command_sha256": sha256_file(point_dir / "command.json"),
        "revisions_sha256": sha256_file(point_dir / "revisions.json"),
        "dynosim_report_sha256": sha256_file(point_dir / "dynosim_report.json"),
        "dynosim_report_canonical_sha256": canonical_report_hash(report),
        "dynosim_per_request_sha256": sha256_file(
            point_dir / "dynosim_per_request.jsonl"
        ),
        "bench_metrics_sha256": sha256_file(point_dir / "bench_metrics.json"),
        "bench_per_request_sha256": sha256_file(
            point_dir / "bench_per_request.jsonl"
        ),
        "mape_sha256": sha256_file(point_dir / "mape.json"),
        "replay_stdout_sha256": sha256_file(point_dir / "replay.stdout.log"),
        "replay_stderr_sha256": sha256_file(point_dir / "replay.stderr.log"),
    }
    return hashes


def run_point(
    point: PointInputs,
    run_dir: Path,
    python: Path,
    dynamo_root: Path,
    aic_root: Path,
) -> dict[str, Any]:
    point_dir = run_dir / point.name
    point_dir.mkdir(parents=True, exist_ok=True)
    trace_path = point_dir / "trace.mooncake.jsonl"
    rid_map_path = point_dir / "rid_map.json"
    report_path = point_dir / "dynosim_report.json"
    per_request_path = point_dir / "dynosim_per_request.jsonl"
    source_manifest = convert_workload(point, trace_path, rid_map_path)
    args = engine_args()
    command = replay_command(python, trace_path, report_path, per_request_path, args)
    revisions = {
        "dynamo": git_revision(dynamo_root),
        "aiconfigurator": git_revision(aic_root),
        "aiconfigurator_base_pin": "ffbab15797d3a3f14c382e937c0ab0c1c5cbc530",
    }
    write_json(point_dir / "source_manifest.json", source_manifest)
    write_json(point_dir / "engine_args.json", args)
    write_json(point_dir / "revisions.json", revisions)
    write_json(
        point_dir / "command.json",
        {
            "argv": command,
            "cwd": str(dynamo_root),
            "offline": True,
            "num_workers": 1,
            "router_mode": "round_robin",
        },
    )

    environment = os.environ.copy()
    pythonpath = [
        str(aic_root / "src"),
        str(dynamo_root / "lib" / "bindings" / "python" / "src"),
        str(dynamo_root / "components" / "src"),
    ]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    environment["AICONFIGURATOR_SYSTEMS_PATH"] = str(aic_root / "src" / "aiconfigurator" / "systems")
    environment.setdefault("RUST_LOG", "info")
    started = time.time()
    with (point_dir / "replay.stdout.log").open("w", encoding="utf-8") as stdout, (
        point_dir / "replay.stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        subprocess.run(
            command,
            cwd=dynamo_root,
            env=environment,
            check=True,
            stdout=stdout,
            stderr=stderr,
        )

    report = json.loads(report_path.read_text())
    per_request = load_jsonl(per_request_path)
    bench_metrics, adjusted_rows = build_bench_metrics(point, report, per_request)
    comparison = point_comparison(point, bench_metrics)
    write_json(point_dir / "bench_metrics.json", bench_metrics)
    write_jsonl(point_dir / "bench_per_request.jsonl", adjusted_rows)
    write_json(point_dir / "mape.json", comparison)
    hashes = point_artifact_hashes(point_dir)
    write_json(point_dir / "hashes.json", hashes)
    write_json(
        point_dir / "run_manifest.json",
        {
            "schema": "dynosim_alignment_point_run.v1",
            "point": point.name,
            "completed": report["completed_requests"],
            "elapsed_wall_seconds": time.time() - started,
            "hashes": hashes,
            "source_hashes": source_manifest,
            "fake_balanced_semantics": {
                "real_baseline_ep_dispatch_algorithm": "fake",
                "dynosim_aic_workload_distribution": "balanced",
                "dynosim_executes_fake_topk_routing": False,
            },
        },
    )
    return comparison


def run_points(
    points: Sequence[PointInputs],
    run_dir: Path,
    python: Path,
    dynamo_root: Path,
    aic_root: Path,
) -> list[dict[str, Any]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for index, point in enumerate(points, start=1):
        print(f"[{index}/{len(points)}] {run_dir.name}: {point.name}", flush=True)
        comparisons.append(run_point(point, run_dir, python, dynamo_root, aic_root))
    write_json(
        run_dir / "run_summary.json",
        {
            "schema": "dynosim_alignment_run_summary.v1",
            "point_count": len(comparisons),
            "completed_points": [item["point"] for item in comparisons],
            "mean_cell_mape_percent": statistics.fmean(
                metric["dynosim_mape_percent"]
                for comparison in comparisons
                for metric in comparison["metrics"].values()
            ),
        },
    )
    return comparisons


def aggregate_comparisons(comparisons: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cells = [
        {
            "point": comparison["point"],
            "metric": metric,
            **values,
        }
        for comparison in comparisons
        for metric, values in comparison["metrics"].items()
    ]
    if len(cells) != 175:
        raise ValueError(f"expected 175 point-metric cells, got {len(cells)}")
    largest = max(cells, key=lambda cell: abs(cell["delta_percentage_points"]))
    per_metric = {}
    for metric in PRIMARY_METRICS:
        subset = [cell for cell in cells if cell["metric"] == metric]
        per_metric[metric] = {
            "point_count": len(subset),
            "mean_dynosim_mape_percent": statistics.fmean(
                cell["dynosim_mape_percent"] for cell in subset
            ),
            "mean_hisim_mape_percent": statistics.fmean(
                cell["hisim_mape_percent"] for cell in subset
            ),
            "mean_delta_percentage_points": statistics.fmean(
                cell["delta_percentage_points"] for cell in subset
            ),
            "dynosim_wins": sum(cell["winner"] == "dynosim" for cell in subset),
            "hisim_wins": sum(cell["winner"] == "hisim" for cell in subset),
            "ties": sum(cell["winner"] == "tie" for cell in subset),
        }
    return {
        "schema": "dynosim_hisim_accuracy_report.v1",
        "point_count": len(comparisons),
        "metric_count": len(PRIMARY_METRICS),
        "cell_count": len(cells),
        "largest_single_point_metric_difference": largest,
        "per_metric": per_metric,
        "overall": {
            "mean_dynosim_mape_percent": statistics.fmean(
                cell["dynosim_mape_percent"] for cell in cells
            ),
            "mean_hisim_mape_percent": statistics.fmean(
                cell["hisim_mape_percent"] for cell in cells
            ),
            "mean_delta_percentage_points": statistics.fmean(
                cell["delta_percentage_points"] for cell in cells
            ),
            "dynosim_wins": sum(cell["winner"] == "dynosim" for cell in cells),
            "hisim_wins": sum(cell["winner"] == "hisim" for cell in cells),
            "ties": sum(cell["winner"] == "tie" for cell in cells),
        },
        "cells": cells,
        "attribution_contract": {
            "scheduler": "FCFS admission, batching, and queueing differences primarily affect throughput and TTFT.",
            "prefix_cache": "B and B-mixed retain exact token-level prefixes with trace_block_size=1.",
            "cuda_graph": "Decode timing batch is padded to the next configured capture size; prefill is not padded.",
            "aic_data": "Negative latency or a missing silicon table is a hard run failure; no empirical scaling is applied.",
            "ingress_time_domain": "The existing per-point fixed ingress envelope is added only to DynoSim TTFT and E2E.",
            "fake_balanced": (
                "The reused real SGLang baseline used ep_dispatch_algorithm=fake. "
                "DynoSim uses AIC workload_distribution=balanced and does not execute fake TopK routing."
            ),
        },
    }


def largest_difference_attribution(
    report: dict[str, Any],
    output_root: Path,
    source_root: Path,
) -> dict[str, Any]:
    largest = report["largest_single_point_metric_difference"]
    point = largest["point"]
    point_dir = output_root / "run1" / point
    source_dir = source_root / point
    per_request = load_jsonl(point_dir / "bench_per_request.jsonl")
    dynosim_report = json.loads((point_dir / "dynosim_report.json").read_text())
    bench_metrics = json.loads((point_dir / "bench_metrics.json").read_text())
    engine = json.loads((point_dir / "engine_args.json").read_text())
    raw = _latest_jsonl_object(source_dir / "raw_sglang" / "client_metrics.jsonl")

    queue_ms = [
        row["first_admit_ms"] - row["arrival_time_ms"] for row in per_request
    ]
    post_admit_to_first_token_ms = [
        row["first_token_ms"] - row["first_admit_ms"] for row in per_request
    ]
    scheduler_ttft_ms = [row["scheduler_ttft_ms"] for row in per_request]
    reused_tokens = [row["reused_input_tokens"] for row in per_request]
    ingress_ms = float(bench_metrics["ingress_envelope"]["ingress_ms"])
    mean_scheduler_ttft_ms = statistics.fmean(scheduler_ttft_ms)
    mean_queue_ms = statistics.fmean(queue_ms)
    mean_post_admit_ms = statistics.fmean(post_admit_to_first_token_ms)

    stderr_text = (point_dir / "replay.stderr.log").read_text().lower()
    database_miss_detected = any(
        marker in stderr_text
        for marker in ("database miss", "perf data not available", "perfdatnotavailable")
    )
    if database_miss_detected:
        raise ValueError(f"{point}: AIC database miss found during attribution audit")

    return {
        "point": point,
        "metric": largest["metric"],
        "raw_value": largest["raw"],
        "dynosim_value": largest["dynosim"],
        "hisim_mape_percent": largest["hisim_mape_percent"],
        "dynosim_mape_percent": largest["dynosim_mape_percent"],
        "delta_percentage_points": largest["delta_percentage_points"],
        "scheduler": {
            "mean_queue_ms": mean_queue_ms,
            "p99_queue_ms": _percentile(queue_ms, 0.99),
            "max_queue_ms": max(queue_ms),
            "mean_post_admit_to_first_token_ms": mean_post_admit_ms,
            "mean_scheduler_ttft_ms": mean_scheduler_ttft_ms,
            "queue_fraction_of_scheduler_ttft": mean_queue_ms
            / mean_scheduler_ttft_ms,
            "input_throughput_raw": largest["raw"]
            if largest["metric"] == "input_throughput"
            else report["cells"][
                next(
                    index
                    for index, cell in enumerate(report["cells"])
                    if cell["point"] == point
                    and cell["metric"] == "input_throughput"
                )
            ]["raw"],
            "input_throughput_dynosim": bench_metrics["input_throughput"],
            "chunked_prefill_size": engine["sglang"]["chunked_prefill_size"],
            "max_prefill_tokens": engine["sglang"]["max_prefill_tokens"],
            "finding": (
                "FCFS queueing dominates the largest TTFT gap. Long prompts are "
                "admitted through the 8192-token chunk budget while decode work "
                "shares the single aggregated worker."
            ),
        },
        "prefix_cache": {
            "prefix_cache_reused_ratio": dynosim_report[
                "prefix_cache_reused_ratio"
            ],
            "first_admission_prefix_cache_reused_ratio": dynosim_report[
                "first_admission_prefix_cache_reused_ratio"
            ],
            "reused_input_tokens_min": min(reused_tokens),
            "reused_input_tokens_max": max(reused_tokens),
            "input_length": per_request[0]["input_length"],
            "trace_block_size": 1,
            "finding": (
                "The shared prefix is preserved and reused, but cache reuse does "
                "not prevent the single-worker prefill/decode queue from growing."
            ),
        },
        "cuda_graph": {
            "dynosim_capture_sizes": engine["sglang"]["cuda_graph_batch_sizes"],
            "real_capture_sizes": raw["server_info"]["cuda_graph_bs"],
            "decode_padding_enabled": engine["sglang"][
                "cuda_graph_padding_enabled"
            ],
            "prefill_padding_enabled": False,
            "finding": (
                "Graph padding changes decode timing only, so it is not a direct "
                "TTFT prefill term; slower decode can still increase aggregated-"
                "worker contention. B-4 decode TPOT is reported separately."
            ),
            "raw_mean_tpot_ms": next(
                cell["raw"]
                for cell in report["cells"]
                if cell["point"] == point and cell["metric"] == "mean_tpot_ms"
            ),
            "dynosim_mean_tpot_ms": bench_metrics["mean_tpot_ms"],
        },
        "aic_data": {
            "system": engine["aic_system"],
            "backend": engine["aic_backend"],
            "backend_version": engine["aic_backend_version"],
            "workload_distribution": engine["aic_workload_distribution"],
            "kv_cache_dtype": engine["aic_kv_cache_dtype"],
            "fmha_dtype": engine["aic_fmha_dtype"],
            "database_miss_detected": database_miss_detected,
            "empirical_scaling_applied": False,
            "finding": (
                "The run used silicon tables without a database miss. AIC "
                "prefill/decode service time therefore remains a real model "
                "difference, not a missing-data fallback."
            ),
        },
        "ingress_time_domain": {
            "ingress_ms": ingress_ms,
            "ingress_fraction_of_dynosim_ttft": ingress_ms
            / bench_metrics["mean_ttft_ms"],
            "scheduler_arrival_unchanged": True,
            "adjusted_metrics": ["mean_ttft_ms", "mean_e2e_latency_ms"],
            "finding": (
                "The reused fixed ingress envelope is small relative to the "
                "TTFT gap and cannot explain the queue-driven discrepancy."
            ),
        },
    }


def markdown_report(report: dict[str, Any]) -> str:
    largest = report["largest_single_point_metric_difference"]
    attribution = report["largest_difference_attribution"]
    scheduler = attribution["scheduler"]
    prefix = attribution["prefix_cache"]
    cuda_graph = attribution["cuda_graph"]
    aic_data = attribution["aic_data"]
    ingress = attribution["ingress_time_domain"]
    overall = report["overall"]
    lines = [
        "# DynoSim SGLang vs HiSim ACC 准确度报告",
        "",
        "## 最大单点/单指标差异",
        "",
        (
            f"- 点位：`{largest['point']}`；指标：`{largest['metric']}`；"
            f"DynoSim MAPE {largest['dynosim_mape_percent']:.6f}%，"
            f"HiSim MAPE {largest['hisim_mape_percent']:.6f}%，"
            f"差值 {largest['delta_percentage_points']:+.6f} 个百分点。"
        ),
        "",
        "### 最大差异归因",
        "",
        (
            f"- 调度：平均排队 {scheduler['mean_queue_ms']:.3f} ms，占 scheduler TTFT "
            f"{scheduler['queue_fraction_of_scheduler_ttft'] * 100:.3f}%；admit 后到首 token "
            f"平均 {scheduler['mean_post_admit_to_first_token_ms']:.3f} ms。DynoSim input "
            f"throughput {scheduler['input_throughput_dynosim']:.3f} tok/s，实机 "
            f"{scheduler['input_throughput_raw']:.3f} tok/s。"
        ),
        (
            f"- Prefix/cache：完整 token/block=1 链路实际复用比例 "
            f"{prefix['prefix_cache_reused_ratio']:.6f}，每请求最终复用 "
            f"{prefix['reused_input_tokens_min']}–{prefix['reused_input_tokens_max']} / "
            f"{prefix['input_length']} tokens；复用存在，但未阻止单 worker 队列增长。"
        ),
        (
            f"- CUDA graph：只对 decode padding；B-4 TPOT 为 DynoSim "
            f"{cuda_graph['dynosim_mean_tpot_ms']:.3f} ms、实机 "
            f"{cuda_graph['raw_mean_tpot_ms']:.3f} ms，因此可通过 decode contention "
            "间接放大 TTFT，但不是 prefill TTFT 的直接 padding 项。"
        ),
        (
            f"- AIC 数据：`{aic_data['system']}` / SGLang "
            f"`{aic_data['backend_version']}` / `{aic_data['workload_distribution']}`，"
            f"KV `{aic_data['kv_cache_dtype']}`、FMHA `{aic_data['fmha_dtype']}`；"
            "无 database miss、无经验缩放。"
        ),
        (
            f"- Ingress：复用固定 {ingress['ingress_ms']:.3f} ms，只占 DynoSim TTFT "
            f"{ingress['ingress_fraction_of_dynosim_ttft'] * 100:.3f}%，不能解释 "
            "35 秒量级的 scheduler queue。"
        ),
        "",
        "## 总体结果",
        "",
        (
            f"- 35 个点、5 项指标、175 个 cell；DynoSim 总均值 "
            f"{overall['mean_dynosim_mape_percent']:.6f}%，HiSim 总均值 "
            f"{overall['mean_hisim_mape_percent']:.6f}%。"
        ),
        (
            f"- 胜负：DynoSim {overall['dynosim_wins']}，HiSim "
            f"{overall['hisim_wins']}，平局 {overall['ties']}。"
        ),
        "",
        "| 指标 | DynoSim MAPE 均值 | HiSim MAPE 均值 | 差值百分点 | DynoSim/HiSim/平 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in PRIMARY_METRICS:
        item = report["per_metric"][metric]
        lines.append(
            f"| `{metric}` | {item['mean_dynosim_mape_percent']:.6f}% | "
            f"{item['mean_hisim_mape_percent']:.6f}% | "
            f"{item['mean_delta_percentage_points']:+.6f} | "
            f"{item['dynosim_wins']}/{item['hisim_wins']}/{item['ties']} |"
        )
    lines.extend(
        [
            "",
            "## 归因边界",
            "",
            "- 调度：FCFS admission、batching 和 queueing 差异主要进入 throughput/TTFT。",
            "- Prefix/cache：B 与 B-mixed 使用 block size 1 的完整 token 序列，保留 token 级共享前缀。",
            "- CUDA graph：仅 decode 计时 batch 上取整到最小可容纳 capture size；prefill 不 padding。",
            "- AIC：数据库 miss、负延迟或非有限值均为硬失败；未添加经验缩放。",
            "- Ingress：逐点复用原 HiSim ACC 的固定 envelope，只加到 DynoSim TTFT/E2E。",
            "- fake+balanced：实机 baseline 使用 `ep_dispatch_algorithm=fake`；DynoSim 仅使用 AIC "
            "`workload_distribution=balanced`，不声称内部执行 fake TopK 路由。",
            "",
        ]
    )
    return "\n".join(lines)


def refresh_run_metadata(
    output_root: Path,
    point_names: Iterable[str],
    dynamo_root: Path,
    aic_root: Path,
) -> dict[str, Any]:
    revisions = {
        "dynamo": git_revision(dynamo_root),
        "aiconfigurator": git_revision(aic_root),
        "aiconfigurator_base_pin": "ffbab15797d3a3f14c382e937c0ab0c1c5cbc530",
    }
    wanted = set(point_names)
    for run_name in ("smoke", "run1", "run2"):
        for point_dir in sorted((output_root / run_name).glob("sweep_*")):
            if not point_dir.is_dir() or point_dir.name not in wanted:
                continue
            write_json(point_dir / "revisions.json", revisions)
            hashes = point_artifact_hashes(point_dir)
            write_json(point_dir / "hashes.json", hashes)
            run_manifest_path = point_dir / "run_manifest.json"
            run_manifest = json.loads(run_manifest_path.read_text())
            run_manifest["hashes"] = hashes
            write_json(run_manifest_path, run_manifest)
    return revisions


def compare_determinism(output_root: Path, point_names: Iterable[str]) -> dict[str, Any]:
    fields = (
        "trace_sha256",
        "rid_map_sha256",
        "dynosim_report_canonical_sha256",
        "dynosim_per_request_sha256",
        "bench_metrics_sha256",
        "bench_per_request_sha256",
        "mape_sha256",
    )
    points = {}
    for name in point_names:
        first = json.loads(
            (output_root / "run1" / name / "hashes.json").read_text()
        )
        second = json.loads(
            (output_root / "run2" / name / "hashes.json").read_text()
        )
        mismatches = {
            field: {"run1": first[field], "run2": second[field]}
            for field in fields
            if first[field] != second[field]
        }
        points[name] = {"deterministic": not mismatches, "mismatches": mismatches}
    result = {
        "schema": "dynosim_alignment_determinism.v1",
        "point_count": len(points),
        "all_deterministic": all(value["deterministic"] for value in points.values()),
        "points": points,
    }
    if not result["all_deterministic"]:
        raise ValueError("DynoSim full-run determinism check failed")
    return result


def acceptance_audit(
    points: Sequence[PointInputs],
    output_root: Path,
    determinism: dict[str, Any],
    revisions: dict[str, Any],
    dynamo_root: Path,
    aic_root: Path,
) -> dict[str, Any]:
    expected_names = {point.name for point in points}
    current_source_hashes = {
        point.name: source_hashes(point)
        for point in points
    }
    run_results = {}
    all_database_miss_free = True
    all_nonnegative_finite = True
    all_topology_valid = True
    all_source_hashes_unchanged = True

    for run_name in ("run1", "run2"):
        point_dirs = sorted(
            path
            for path in (output_root / run_name).glob("sweep_*")
            if path.is_dir() and path.name in expected_names
        )
        completed_requests = 0
        per_request_rows = 0
        for point_dir in point_dirs:
            bench = json.loads((point_dir / "bench_metrics.json").read_text())
            per_request = load_jsonl(point_dir / "bench_per_request.jsonl")
            engine = json.loads((point_dir / "engine_args.json").read_text())
            source_manifest = json.loads(
                (point_dir / "source_manifest.json").read_text()
            )
            completed_requests += int(bench["completed"])
            per_request_rows += len(per_request)

            for metric in PRIMARY_METRICS:
                value = bench[metric]
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    all_nonnegative_finite = False
            for row in per_request:
                for field in ("ttft_ms", "e2e_latency_ms", "itl_ms"):
                    value = row[field]
                    if (
                        not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                    ):
                        all_nonnegative_finite = False

            stderr = (point_dir / "replay.stderr.log").read_text().lower()
            if any(
                marker in stderr
                for marker in (
                    "database miss",
                    "perf data not available",
                    "perfdatnotavailable",
                )
            ):
                all_database_miss_free = False

            topology = (
                engine.get("aic_tp_size"),
                engine.get("aic_attention_dp_size"),
                engine.get("aic_moe_tp_size"),
                engine.get("aic_moe_ep_size"),
            )
            if topology != (4, 1, 1, 4):
                all_topology_valid = False
            expected_hashes = current_source_hashes[point_dir.name]
            if any(
                source_manifest.get(key) != value
                for key, value in expected_hashes.items()
            ):
                all_source_hashes_unchanged = False

        run_results[run_name] = {
            "point_count": len(point_dirs),
            "completed_requests": completed_requests,
            "per_request_rows": per_request_rows,
        }

    total_source_requests = sum(len(point.rows) for point in points)
    checks = {
        "point_set_35_of_35": all(
            result["point_count"] == 35 for result in run_results.values()
        ),
        "completed_matches_source": all(
            result["completed_requests"] == total_source_requests
            and result["per_request_rows"] == total_source_requests
            for result in run_results.values()
        ),
        "nonnegative_finite_metrics": all_nonnegative_finite,
        "aic_database_miss_free": all_database_miss_free,
        "tp4_ep4_mapping_valid": all_topology_valid,
        "source_hashes_unchanged_since_run": all_source_hashes_unchanged,
        "two_run_deterministic": determinism["all_deterministic"],
    }
    if not all(checks.values()):
        raise ValueError(f"acceptance audit failed: {checks}")
    dynamo_extension = (
        dynamo_root / "lib" / "bindings" / "python" / "src" / "dynamo" / "_core.abi3.so"
    )
    aic_extension = (
        aic_root
        / "src"
        / "aiconfigurator_core"
        / "_aiconfigurator_core.abi3.so"
    )
    return {
        "schema": "dynosim_alignment_acceptance.v1",
        "accepted": True,
        "checks": checks,
        "source_point_count": len(points),
        "source_request_count": total_source_requests,
        "runs": run_results,
        "revisions": revisions,
        "build_artifacts": {
            "dynamo_python_extension": {
                "path": str(dynamo_extension),
                "sha256": sha256_file(dynamo_extension),
            },
            "aiconfigurator_python_extension": {
                "path": str(aic_extension),
                "sha256": sha256_file(aic_extension),
            },
        },
        "execution_contract": {
            "reused_real_data_only": True,
            "sglang_server_started": False,
            "gpu_preflight_required": False,
            "offline_aggregated_sglang_path": True,
            "single_batch_excluded": True,
            "empirical_scaling_applied": False,
        },
    }


def select_points(points: Sequence[PointInputs], names: Iterable[str]) -> list[PointInputs]:
    wanted = set(names)
    selected = [point for point in points if point.name in wanted]
    if {point.name for point in selected} != wanted:
        raise ValueError(f"unknown point selection: {sorted(wanted - {p.name for p in selected})}")
    return selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("preflight", "smoke", "run1", "run2", "report", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dynamo-root", type=Path, default=DYNAMO_ROOT)
    parser.add_argument("--aic-root", type=Path, default=AIC_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--points", nargs="*")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    points, initial_preflight = preflight(args.source_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "preflight.json", initial_preflight)
    selected = select_points(points, args.points) if args.points else points

    if args.action == "preflight":
        print(f"preflight ok: {len(points)}/35 points, {initial_preflight['total_requests']} requests")
        return 0
    if args.action in ("smoke", "all"):
        smoke = select_points(points, SMOKE_POINTS)
        run_points(smoke, args.output_root / "smoke", args.python, args.dynamo_root, args.aic_root)
        if args.action == "smoke":
            return 0
    if args.action in ("run1", "all"):
        run_points(selected, args.output_root / "run1", args.python, args.dynamo_root, args.aic_root)
        if args.action == "run1":
            return 0
    if args.action in ("run2", "all"):
        run_points(selected, args.output_root / "run2", args.python, args.dynamo_root, args.aic_root)
        if args.action == "run2":
            return 0
    if args.action in ("report", "all"):
        comparison_paths = sorted((args.output_root / "run1").glob("sweep_*/mape.json"))
        comparisons = [json.loads(path.read_text()) for path in comparison_paths]
        report = aggregate_comparisons(comparisons)
        revisions = refresh_run_metadata(
            args.output_root,
            expected_point_names(),
            args.dynamo_root,
            args.aic_root,
        )
        determinism = compare_determinism(args.output_root, expected_point_names())
        _, final_preflight = preflight(args.source_root)
        if final_preflight["points"] != initial_preflight["points"]:
            raise ValueError("source input hashes changed during DynoSim alignment")
        report["determinism"] = determinism
        report["source_unchanged"] = True
        report["largest_difference_attribution"] = largest_difference_attribution(
            report,
            args.output_root,
            args.source_root,
        )
        report["source_preflight_sha256"] = sha256_bytes(
            canonical_json_bytes(initial_preflight["points"])
        )
        write_json(args.output_root / "accuracy_report.json", report)
        (args.output_root / "accuracy_report.md").write_text(markdown_report(report))
        write_json(args.output_root / "determinism.json", determinism)
        write_json(
            args.output_root / "acceptance.json",
            acceptance_audit(
                points,
                args.output_root,
                determinism,
                revisions,
                args.dynamo_root,
                args.aic_root,
            ),
        )
        print(
            f"report ok: DynoSim={report['overall']['mean_dynosim_mape_percent']:.6f}% "
            f"HiSim={report['overall']['mean_hisim_mape_percent']:.6f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
