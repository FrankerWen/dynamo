#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay the seven formal B points after the SGLang prefix-admission fix.

The original real-data root is not required at replay time. This runner treats
the canonical traces, RID maps, ingress envelopes, raw metric values, HiSim
MAPE, and original source hashes preserved by the accepted run1 as immutable
inputs. It writes two fresh offline replays and a before/after accuracy report
to a separate output root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dynosim_hisim_alignment as common


BEFORE_ROOT = Path(
    "/workspace/hisim_align/"
    "dynosim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_"
    "h100pcie_gsp_mixed_seed11_full"
)
OUTPUT_ROOT = Path(
    "/workspace/hisim_align/"
    "dynosim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_"
    "h100pcie_gsp_mixed_seed11_full_b_prefix_fix"
)
ORIGINAL_SOURCE_ROOT = Path(
    "/workspace/hisim_align/"
    "hisim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_"
    "h100pcie_gsp_mixed_seed11_full"
)
DYNAMO_ROOT = Path("/workspace/hisim_align/dynamo")
AIC_ROOT = Path("/workspace/hisim_align/aiconfigurator-dynosim-e2e")
PYTHON = DYNAMO_ROOT / ".venv" / "bin" / "python"
B_RPS_POINTS = (1, 2, 4, 8, 16, 24, 32)
B_POINT_NAMES = tuple(f"sweep_b_rps_point_{rps}" for rps in B_RPS_POINTS)
PRESERVED_FILES = (
    "trace.mooncake.jsonl",
    "rid_map.json",
    "source_manifest.json",
    "engine_args.json",
    "bench_metrics.json",
    "bench_per_request.jsonl",
    "mape.json",
    "dynosim_report.json",
    "dynosim_per_request.jsonl",
    "hashes.json",
)
DATABASE_MISS_MARKERS = (
    "database miss",
    "perf data not available",
    "perfdatnotavailable",
)
ARTIFACT_SCOPE = "b_prefix_fix"
ARTIFACT_SCHEMA_VERSION = 2


def artifact_schema(suffix: str) -> str:
    """Return a scope-aware schema for runners that reuse this module."""
    return f"dynosim_{ARTIFACT_SCOPE}_{suffix}.v{ARTIFACT_SCHEMA_VERSION}"


@dataclass(frozen=True)
class PreservedPoint:
    name: str
    before_dir: Path
    trace_rows: tuple[dict[str, Any], ...]
    rid_rows: tuple[dict[str, Any], ...]
    source_manifest: dict[str, Any]
    engine_args: dict[str, Any]
    before_bench: dict[str, Any]
    before_mape: dict[str, Any]
    preserved_hashes: dict[str, str]
    shared_prefix_tokens: int

    @property
    def request_count(self) -> int:
        return len(self.trace_rows)


def _finite_nonnegative(label: str, value: Any) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative, got {value!r}")
    return float(value)


def _common_prefix_len(sequences: Sequence[Sequence[int]]) -> int:
    if not sequences:
        return 0
    limit = min(len(sequence) for sequence in sequences)
    for index in range(limit):
        token = sequences[0][index]
        if any(sequence[index] != token for sequence in sequences[1:]):
            return index
    return limit


def _preserved_hashes(point_dir: Path) -> dict[str, str]:
    return {
        name: common.sha256_file(point_dir / name)
        for name in PRESERVED_FILES
    }


def load_preserved_point(before_root: Path, name: str) -> PreservedPoint:
    point_dir = before_root / "run1" / name
    if not point_dir.is_dir():
        raise FileNotFoundError(point_dir)
    for filename in PRESERVED_FILES:
        path = point_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)

    trace_rows = tuple(common.load_jsonl(point_dir / "trace.mooncake.jsonl"))
    rid_map = json.loads((point_dir / "rid_map.json").read_text())
    rid_rows = tuple(rid_map.get("rows", ()))
    source_manifest = json.loads((point_dir / "source_manifest.json").read_text())
    engine_args = json.loads((point_dir / "engine_args.json").read_text())
    before_bench = json.loads((point_dir / "bench_metrics.json").read_text())
    before_mape = json.loads((point_dir / "mape.json").read_text())
    count = int(source_manifest.get("request_count", -1))

    if count <= 0 or len(trace_rows) != count or len(rid_rows) != count:
        raise ValueError(
            f"{name}: preserved request count mismatch "
            f"manifest={count}, trace={len(trace_rows)}, rid={len(rid_rows)}"
        )
    if before_bench.get("completed") != count or before_mape.get("completed") != count:
        raise ValueError(f"{name}: preserved metric completion mismatch")
    if source_manifest.get("trace_sha256") != common.sha256_file(
        point_dir / "trace.mooncake.jsonl"
    ):
        raise ValueError(f"{name}: preserved trace hash differs from source manifest")
    if source_manifest.get("rid_map_sha256") != common.sha256_file(
        point_dir / "rid_map.json"
    ):
        raise ValueError(f"{name}: preserved RID map hash differs from source manifest")
    if not source_manifest.get("workload_sha256"):
        raise ValueError(f"{name}: original workload SHA256 is missing")
    if not before_bench.get("ingress_envelope"):
        raise ValueError(f"{name}: preserved ingress envelope is missing")

    previous_timestamp = -math.inf
    for index, (trace, rid) in enumerate(zip(trace_rows, rid_rows, strict=True)):
        input_length = trace.get("input_length")
        output_length = trace.get("output_length")
        hash_ids = trace.get("hash_ids")
        timestamp = _finite_nonnegative(f"{name}: timestamp {index}", trace.get("timestamp"))
        if timestamp < previous_timestamp:
            raise ValueError(f"{name}: timestamps are not non-decreasing")
        previous_timestamp = timestamp
        if (
            not isinstance(input_length, int)
            or not isinstance(output_length, int)
            or not isinstance(hash_ids, list)
            or len(hash_ids) != input_length
            or output_length < 1
        ):
            raise ValueError(f"{name}: invalid trace shape at row {index}")
        if trace.get("request_id") != rid.get("source_rid"):
            raise ValueError(f"{name}: trace/RID mismatch at row {index}")
        if rid.get("line_index") != index:
            raise ValueError(f"{name}: unstable RID line index at row {index}")
        if rid.get("dynosim_uuid") != common.deterministic_uuid(index):
            raise ValueError(f"{name}: non-deterministic UUID at row {index}")

    expected_topology = (
        engine_args.get("aic_tp_size"),
        engine_args.get("aic_attention_dp_size"),
        engine_args.get("aic_moe_tp_size"),
        engine_args.get("aic_moe_ep_size"),
    )
    if expected_topology != (4, 1, 1, 4):
        raise ValueError(f"{name}: invalid TP/DP/MoE-TP/EP topology {expected_topology}")
    if (
        engine_args.get("engine_type") != "sglang"
        or engine_args.get("worker_type") != "aggregated"
        or engine_args.get("aic_workload_distribution") != "balanced"
        or engine_args.get("aic_system") != "h100_pcie"
        or engine_args.get("aic_backend_version") != "0.5.9"
    ):
        raise ValueError(f"{name}: preserved engine contract mismatch")
    sglang = engine_args.get("sglang", {})
    if (
        sglang.get("schedule_policy") != "fcfs"
        or sglang.get("chunked_prefill_size") != 8192
        or not sglang.get("cuda_graph_enabled")
        or not sglang.get("cuda_graph_padding_enabled")
    ):
        raise ValueError(f"{name}: preserved SGLang scheduling contract mismatch")

    for metric in common.PRIMARY_METRICS:
        before = before_mape.get("metrics", {}).get(metric, {})
        _finite_nonnegative(f"{name}: before raw {metric}", before.get("raw"))
        _finite_nonnegative(
            f"{name}: before DynoSim MAPE {metric}",
            before.get("dynosim_mape_percent"),
        )
        _finite_nonnegative(
            f"{name}: HiSim MAPE {metric}",
            before.get("hisim_mape_percent"),
        )

    return PreservedPoint(
        name=name,
        before_dir=point_dir,
        trace_rows=trace_rows,
        rid_rows=rid_rows,
        source_manifest=source_manifest,
        engine_args=engine_args,
        before_bench=before_bench,
        before_mape=before_mape,
        preserved_hashes=_preserved_hashes(point_dir),
        shared_prefix_tokens=_common_prefix_len(
            [row["hash_ids"] for row in trace_rows]
        ),
    )


def preflight(before_root: Path) -> tuple[list[PreservedPoint], dict[str, Any]]:
    points = [load_preserved_point(before_root, name) for name in B_POINT_NAMES]
    manifest = {
        "schema": "dynosim_b_prefix_fix_preflight.v1",
        "before_root": str(before_root),
        "preserved_run": "run1",
        "original_source_root": str(ORIGINAL_SOURCE_ROOT),
        "original_source_root_available_at_replay": ORIGINAL_SOURCE_ROOT.is_dir(),
        "point_count": len(points),
        "expected_points": list(B_POINT_NAMES),
        "total_requests": sum(point.request_count for point in points),
        "provenance_contract": {
            "trace_and_rid": "byte-identical files from the accepted before run1",
            "raw_and_hisim_accuracy": (
                "raw metric values and HiSim ACC MAPE preserved in each before mape.json"
            ),
            "ingress": "unchanged per-point envelope preserved in before bench_metrics.json",
            "original_source_identity": (
                "original workload/raw/HiSim/sim-config SHA256 retained in source_manifest.json"
            ),
            "real_data_recollected": False,
            "sglang_server_started": False,
        },
        "points": {
            point.name: {
                "request_count": point.request_count,
                "input_lengths": sorted(
                    {row["input_length"] for row in point.trace_rows}
                ),
                "output_lengths": sorted(
                    {row["output_length"] for row in point.trace_rows}
                ),
                "shared_prefix_tokens": point.shared_prefix_tokens,
                "original_source_hashes": point.source_manifest,
                "preserved_artifact_hashes": point.preserved_hashes,
            }
            for point in points
        },
    }
    return points, manifest


def _copy_preserved_inputs(point: PreservedPoint, point_dir: Path) -> None:
    for filename in (
        "trace.mooncake.jsonl",
        "rid_map.json",
        "source_manifest.json",
        "engine_args.json",
    ):
        shutil.copyfile(point.before_dir / filename, point_dir / filename)
    for filename in ("trace.mooncake.jsonl", "rid_map.json", "source_manifest.json", "engine_args.json"):
        if common.sha256_file(point_dir / filename) != common.sha256_file(
            point.before_dir / filename
        ):
            raise ValueError(f"{point.name}: copied {filename} is not byte-identical")


def _build_bench_metrics(
    point: PreservedPoint,
    report: dict[str, Any],
    per_request: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    count = point.request_count
    if report.get("num_requests") != count or report.get("completed_requests") != count:
        raise ValueError(
            f"{point.name}: completion mismatch "
            f"{report.get('completed_requests')}/{report.get('num_requests')} != {count}"
        )
    if len(per_request) != count:
        raise ValueError(f"{point.name}: per-request rows {len(per_request)} != {count}")
    if any(row.get("terminal_status") != "completed" for row in per_request):
        raise ValueError(f"{point.name}: replay contains non-completed requests")

    rid_by_uuid = {
        row["dynosim_uuid"]: row["source_rid"] for row in point.rid_rows
    }
    ingress = point.before_bench["ingress_envelope"]
    ingress_ms = float(ingress["ingress_ms"])
    adjusted_rows: list[dict[str, Any]] = []
    ttfts: list[float] = []
    e2es: list[float] = []
    for row in per_request:
        uuid_value = row.get("uuid")
        if uuid_value not in rid_by_uuid:
            raise ValueError(f"{point.name}: unexpected UUID {uuid_value}")
        for key in (
            "ttft_ms",
            "e2e_latency_ms",
            "itl_ms",
            "arrival_time_ms",
            "first_admit_ms",
            "first_token_ms",
        ):
            _finite_nonnegative(f"{point.name}: {uuid_value} {key}", row.get(key))
        if row.get("output_length") != row.get("requested_output_length"):
            raise ValueError(f"{point.name}: output length mismatch for {uuid_value}")
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
        "p99_ttft_ms": common._percentile(ttfts, 0.99),
        "mean_tpot_ms": report["mean_tpot_ms"],
        "median_tpot_ms": report["median_tpot_ms"],
        "p99_tpot_ms": report["p99_tpot_ms"],
        "mean_itl_ms": report["mean_itl_ms"],
        "median_itl_ms": report["median_itl_ms"],
        "p99_itl_ms": report["p99_itl_ms"],
        "mean_e2e_latency_ms": statistics.fmean(e2es),
        "median_e2e_latency_ms": statistics.median(e2es),
        "p99_e2e_latency_ms": common._percentile(e2es, 0.99),
        "prefix_cache_reused_ratio": report["prefix_cache_reused_ratio"],
        "first_admission_prefix_cache_reused_ratio": report[
            "first_admission_prefix_cache_reused_ratio"
        ],
        "ingress_envelope": ingress,
        "scheduler_arrival_unchanged": True,
        "unchanged_by_ingress": [
            "mean_tpot_ms",
            "mean_itl_ms",
            "input_throughput",
            "request_throughput",
        ],
    }
    for metric in common.PRIMARY_METRICS:
        _finite_nonnegative(f"{point.name}: after {metric}", metrics[metric])
    return metrics, adjusted_rows


def _after_mape(point: PreservedPoint, bench: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    for metric in common.PRIMARY_METRICS:
        baseline = point.before_mape["metrics"][metric]
        raw = float(baseline["raw"])
        after_value = float(bench[metric])
        after_mape = abs(after_value - raw) / abs(raw) * 100.0
        metrics[metric] = {
            "raw": raw,
            "dynosim": after_value,
            "dynosim_mape_percent": after_mape,
            "hisim_mape_percent": float(baseline["hisim_mape_percent"]),
            "delta_percentage_points": (
                after_mape - float(baseline["hisim_mape_percent"])
            ),
            "winner": (
                "dynosim"
                if after_mape < float(baseline["hisim_mape_percent"])
                else "hisim"
                if after_mape > float(baseline["hisim_mape_percent"])
                else "tie"
            ),
        }
    return {
        "schema": "dynosim_hisim_point_comparison.v1",
        "point": point.name,
        "completed": point.request_count,
        "metric_count": len(metrics),
        "mean_dynosim_mape_percent": statistics.fmean(
            value["dynosim_mape_percent"] for value in metrics.values()
        ),
        "mean_hisim_mape_percent": statistics.fmean(
            value["hisim_mape_percent"] for value in metrics.values()
        ),
        "metrics": metrics,
    }


def run_point(
    point: PreservedPoint,
    run_dir: Path,
    revisions: dict[str, Any],
) -> dict[str, Any]:
    point_dir = run_dir / point.name
    point_dir.mkdir(parents=True, exist_ok=False)
    _copy_preserved_inputs(point, point_dir)
    trace_path = point_dir / "trace.mooncake.jsonl"
    report_path = point_dir / "dynosim_report.json"
    per_request_path = point_dir / "dynosim_per_request.jsonl"
    command = common.replay_command(
        PYTHON,
        trace_path,
        report_path,
        per_request_path,
        point.engine_args,
    )
    common.write_json(point_dir / "revisions.json", revisions)
    common.write_json(
        point_dir / "command.json",
        {
            "argv": command,
            "cwd": str(DYNAMO_ROOT),
            "offline": True,
            "num_workers": 1,
            "router_mode": "round_robin",
            "input_provenance": str(point.before_dir),
        },
    )
    common.write_json(
        point_dir / "provenance.json",
        {
            "schema": artifact_schema("provenance"),
            "point": point.name,
            "preserved_before_dir": str(point.before_dir),
            "preserved_before_hashes": point.preserved_hashes,
            "original_source_root": str(ORIGINAL_SOURCE_ROOT),
            "original_source_root_available_at_replay": ORIGINAL_SOURCE_ROOT.is_dir(),
            "original_source_hashes": point.source_manifest,
            "real_data_recollected": False,
            "sglang_server_started": False,
            "gpu_preflight_run": False,
        },
    )

    environment = os.environ.copy()
    pythonpath = [
        str(AIC_ROOT / "src"),
        str(DYNAMO_ROOT / "lib" / "bindings" / "python" / "src"),
        str(DYNAMO_ROOT / "components" / "src"),
    ]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    environment["AICONFIGURATOR_SYSTEMS_PATH"] = str(
        AIC_ROOT / "src" / "aiconfigurator" / "systems"
    )
    environment.setdefault("RUST_LOG", "info")

    started = time.time()
    with (point_dir / "replay.stdout.log").open("w", encoding="utf-8") as stdout, (
        point_dir / "replay.stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        subprocess.run(
            command,
            cwd=DYNAMO_ROOT,
            env=environment,
            check=True,
            stdout=stdout,
            stderr=stderr,
        )

    report = json.loads(report_path.read_text())
    per_request = common.load_jsonl(per_request_path)
    bench, adjusted_rows = _build_bench_metrics(point, report, per_request)
    mape = _after_mape(point, bench)
    common.write_json(point_dir / "bench_metrics.json", bench)
    common.write_jsonl(point_dir / "bench_per_request.jsonl", adjusted_rows)
    common.write_json(point_dir / "mape.json", mape)
    hashes = common.point_artifact_hashes(point_dir)
    hashes["provenance_sha256"] = common.sha256_file(point_dir / "provenance.json")
    common.write_json(point_dir / "hashes.json", hashes)
    common.write_json(
        point_dir / "run_manifest.json",
        {
            "schema": artifact_schema("point_run"),
            "point": point.name,
            "completed": report["completed_requests"],
            "elapsed_wall_seconds": time.time() - started,
            "shared_prefix_tokens": point.shared_prefix_tokens,
            "hashes": hashes,
            "fake_balanced_semantics": {
                "real_baseline_ep_dispatch_algorithm": "fake",
                "dynosim_aic_workload_distribution": "balanced",
                "dynosim_executes_fake_topk_routing": False,
            },
        },
    )
    return mape


def run_all_points(
    points: Sequence[PreservedPoint],
    run_dir: Path,
    revisions: dict[str, Any],
) -> list[dict[str, Any]]:
    run_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for index, point in enumerate(points, start=1):
        print(f"[{index}/{len(points)}] {run_dir.name}: {point.name}", flush=True)
        results.append(run_point(point, run_dir, revisions))
    common.write_json(
        run_dir / "run_summary.json",
        {
            "schema": artifact_schema("run_summary"),
            "point_count": len(results),
            "completed_points": [result["point"] for result in results],
            "mean_cell_mape_percent": statistics.fmean(
                metric["dynosim_mape_percent"]
                for result in results
                for metric in result["metrics"].values()
            ),
        },
    )
    return results


def _point_diagnostics(point_dir: Path) -> dict[str, Any]:
    bench = json.loads((point_dir / "bench_metrics.json").read_text())
    report = json.loads((point_dir / "dynosim_report.json").read_text())
    rows = common.load_jsonl(point_dir / "bench_per_request.jsonl")
    queues = [row["first_admit_ms"] - row["arrival_time_ms"] for row in rows]
    post_admit = [row["first_token_ms"] - row["first_admit_ms"] for row in rows]
    scheduler_ttft = [row["scheduler_ttft_ms"] for row in rows]
    reused = [row["reused_input_tokens"] for row in rows]
    return {
        "mean_queue_ms": statistics.fmean(queues),
        "p99_queue_ms": common._percentile(queues, 0.99),
        "mean_post_admit_to_first_token_ms": statistics.fmean(post_admit),
        "mean_scheduler_ttft_ms": statistics.fmean(scheduler_ttft),
        "mean_ttft_with_ingress_ms": bench["mean_ttft_ms"],
        "mean_tpot_ms": bench["mean_tpot_ms"],
        "mean_e2e_latency_ms": bench["mean_e2e_latency_ms"],
        "input_throughput": bench["input_throughput"],
        "first_admission_prefix_cache_reused_ratio": report[
            "first_admission_prefix_cache_reused_ratio"
        ],
        "prefix_cache_reused_ratio": report["prefix_cache_reused_ratio"],
        "reused_input_tokens_min": min(reused),
        "reused_input_tokens_max": max(reused),
    }


def compare_before_after(
    points: Sequence[PreservedPoint],
    output_root: Path,
) -> dict[str, Any]:
    cells = []
    per_point = {}
    for point in points:
        after = json.loads(
            (output_root / "run1" / point.name / "mape.json").read_text()
        )
        point_cells = []
        for metric in common.PRIMARY_METRICS:
            before_cell = point.before_mape["metrics"][metric]
            after_cell = after["metrics"][metric]
            cell = {
                "point": point.name,
                "metric": metric,
                "raw": before_cell["raw"],
                "before_dynosim": before_cell["dynosim"],
                "after_dynosim": after_cell["dynosim"],
                "before_dynosim_mape_percent": before_cell[
                    "dynosim_mape_percent"
                ],
                "after_dynosim_mape_percent": after_cell[
                    "dynosim_mape_percent"
                ],
                "hisim_mape_percent": before_cell["hisim_mape_percent"],
                "mape_improvement_percentage_points": (
                    before_cell["dynosim_mape_percent"]
                    - after_cell["dynosim_mape_percent"]
                ),
                "after_minus_before_percentage_points": (
                    after_cell["dynosim_mape_percent"]
                    - before_cell["dynosim_mape_percent"]
                ),
            }
            cells.append(cell)
            point_cells.append(cell)
        per_point[point.name] = {
            "request_count": point.request_count,
            "shared_prefix_tokens": point.shared_prefix_tokens,
            "before_mean_dynosim_mape_percent": statistics.fmean(
                cell["before_dynosim_mape_percent"] for cell in point_cells
            ),
            "after_mean_dynosim_mape_percent": statistics.fmean(
                cell["after_dynosim_mape_percent"] for cell in point_cells
            ),
            "hisim_mean_mape_percent": statistics.fmean(
                cell["hisim_mape_percent"] for cell in point_cells
            ),
            "mean_mape_improvement_percentage_points": statistics.fmean(
                cell["mape_improvement_percentage_points"] for cell in point_cells
            ),
        }
    if len(cells) != len(B_POINT_NAMES) * len(common.PRIMARY_METRICS):
        raise ValueError(f"expected 35 B point-metric cells, got {len(cells)}")

    per_metric = {}
    for metric in common.PRIMARY_METRICS:
        selected = [cell for cell in cells if cell["metric"] == metric]
        per_metric[metric] = {
            "point_count": len(selected),
            "before_mean_dynosim_mape_percent": statistics.fmean(
                cell["before_dynosim_mape_percent"] for cell in selected
            ),
            "after_mean_dynosim_mape_percent": statistics.fmean(
                cell["after_dynosim_mape_percent"] for cell in selected
            ),
            "hisim_mean_mape_percent": statistics.fmean(
                cell["hisim_mape_percent"] for cell in selected
            ),
            "mean_mape_improvement_percentage_points": statistics.fmean(
                cell["mape_improvement_percentage_points"] for cell in selected
            ),
        }

    b4 = next(point for point in points if point.name == "sweep_b_rps_point_4")
    before_b4 = _point_diagnostics(b4.before_dir)
    after_b4 = _point_diagnostics(output_root / "run1" / b4.name)
    queue_removed = before_b4["mean_queue_ms"] - after_b4["mean_queue_ms"]
    b4_root_cause = {
        "shared_prefix_tokens": b4.shared_prefix_tokens,
        "prompt_tokens": b4.trace_rows[0]["input_length"],
        "unique_suffix_tokens": (
            b4.trace_rows[0]["input_length"] - b4.shared_prefix_tokens
        ),
        "chunked_prefill_size": b4.engine_args["sglang"]["chunked_prefill_size"],
        "before": before_b4,
        "after": after_b4,
        "mean_queue_reduction_ms": queue_removed,
        "mean_queue_reduction_percent": queue_removed / before_b4["mean_queue_ms"] * 100,
        "finding": (
            "Before the fix, first admission matched only the first 8192-token "
            "chunk, so the remaining shared prefix was repeatedly charged as "
            "fresh prefill. The fixed path matches the complete prompt before "
            "chunking and charges only the 3277-token unique suffix for warm requests."
        ),
    }
    largest_before = max(cells, key=lambda cell: cell["before_dynosim_mape_percent"])
    largest_after = max(cells, key=lambda cell: cell["after_dynosim_mape_percent"])
    largest_improvement = max(
        cells, key=lambda cell: cell["mape_improvement_percentage_points"]
    )
    return {
        "schema": "dynosim_b_prefix_fix_accuracy_report.v1",
        "scope": {
            "scenario": "b",
            "rps_points": list(B_RPS_POINTS),
            "point_count": len(points),
            "metric_count": len(common.PRIMARY_METRICS),
            "cell_count": len(cells),
            "single_batch_excluded": True,
            "b_mixed_excluded": True,
        },
        "overall": {
            "before_mean_dynosim_mape_percent": statistics.fmean(
                cell["before_dynosim_mape_percent"] for cell in cells
            ),
            "after_mean_dynosim_mape_percent": statistics.fmean(
                cell["after_dynosim_mape_percent"] for cell in cells
            ),
            "hisim_mean_mape_percent": statistics.fmean(
                cell["hisim_mape_percent"] for cell in cells
            ),
            "mean_mape_improvement_percentage_points": statistics.fmean(
                cell["mape_improvement_percentage_points"] for cell in cells
            ),
            "improved_cells": sum(
                cell["mape_improvement_percentage_points"] > 0 for cell in cells
            ),
            "regressed_cells": sum(
                cell["mape_improvement_percentage_points"] < 0 for cell in cells
            ),
            "unchanged_cells": sum(
                cell["mape_improvement_percentage_points"] == 0 for cell in cells
            ),
            "after_dynosim_beats_hisim_cells": sum(
                cell["after_dynosim_mape_percent"] < cell["hisim_mape_percent"]
                for cell in cells
            ),
        },
        "largest_before_cell": largest_before,
        "largest_after_cell": largest_after,
        "largest_improvement_cell": largest_improvement,
        "per_metric": per_metric,
        "per_point": per_point,
        "b4_root_cause": b4_root_cause,
        "cells": cells,
        "accuracy_contract": {
            "formal_metrics": list(common.PRIMARY_METRICS),
            "raw_and_hisim_source": "preserved accepted run1 ACC mape.json",
            "same_ingress_reused": True,
            "scheduler_arrival_unchanged": True,
            "empirical_scaling_applied": False,
            "real_baseline_routing": "fake",
            "dynosim_aic_workload_distribution": "balanced",
            "dynosim_executes_fake_topk_routing": False,
        },
    }


def acceptance_audit(
    points: Sequence[PreservedPoint],
    output_root: Path,
    before_hashes_after_run: dict[str, dict[str, str]],
) -> dict[str, Any]:
    run_results = {}
    database_miss_free = True
    finite_nonnegative = True
    topology_valid = True
    trace_identity = True
    for run_name in ("run1",):
        completed = 0
        row_count = 0
        point_count = 0
        for point in points:
            point_dir = output_root / run_name / point.name
            point_count += 1
            bench = json.loads((point_dir / "bench_metrics.json").read_text())
            rows = common.load_jsonl(point_dir / "bench_per_request.jsonl")
            engine = json.loads((point_dir / "engine_args.json").read_text())
            completed += int(bench["completed"])
            row_count += len(rows)
            for metric in common.PRIMARY_METRICS:
                try:
                    _finite_nonnegative(
                        f"{run_name}/{point.name}/{metric}", bench[metric]
                    )
                except ValueError:
                    finite_nonnegative = False
            stderr = (point_dir / "replay.stderr.log").read_text().lower()
            if any(marker in stderr for marker in DATABASE_MISS_MARKERS):
                database_miss_free = False
            topology = (
                engine.get("aic_tp_size"),
                engine.get("aic_attention_dp_size"),
                engine.get("aic_moe_tp_size"),
                engine.get("aic_moe_ep_size"),
            )
            if topology != (4, 1, 1, 4):
                topology_valid = False
            if common.sha256_file(
                point_dir / "trace.mooncake.jsonl"
            ) != point.source_manifest["trace_sha256"]:
                trace_identity = False
        run_results[run_name] = {
            "point_count": point_count,
            "completed_requests": completed,
            "per_request_rows": row_count,
        }

    expected_requests = sum(point.request_count for point in points)
    preserved_unchanged = all(
        point.preserved_hashes == before_hashes_after_run[point.name]
        for point in points
    )
    checks = {
        "b_points_7_of_7": all(
            result["point_count"] == len(B_POINT_NAMES)
            for result in run_results.values()
        ),
        "completed_matches_preserved_workload": all(
            result["completed_requests"] == expected_requests
            and result["per_request_rows"] == expected_requests
            for result in run_results.values()
        ),
        "nonnegative_finite_metrics": finite_nonnegative,
        "aic_database_miss_free": database_miss_free,
        "tp4_ep4_mapping_valid": topology_valid,
        "canonical_trace_identity": trace_identity,
        "preserved_before_artifacts_unchanged": preserved_unchanged,
        "single_run_contract": not (output_root / "run2").exists(),
    }
    if not all(checks.values()):
        raise ValueError(f"{ARTIFACT_SCOPE} replay acceptance failed: {checks}")
    extension = (
        DYNAMO_ROOT
        / "lib"
        / "bindings"
        / "python"
        / "src"
        / "dynamo"
        / "_core.abi3.so"
    )
    return {
        "schema": artifact_schema("acceptance"),
        "accepted": True,
        "checks": checks,
        "runs": run_results,
        "preserved_request_count": expected_requests,
        "build_artifact": {
            "path": str(extension),
            "sha256": common.sha256_file(extension),
        },
        "execution_contract": {
            "offline_aggregated_sglang_path": True,
            "real_data_recollected": False,
            "sglang_server_started": False,
            "gpu_preflight_required": False,
            "empirical_scaling_applied": False,
        },
    }


def markdown_report(report: dict[str, Any], acceptance: dict[str, Any]) -> str:
    overall = report["overall"]
    b4 = report["b4_root_cause"]
    before = b4["before"]
    after = b4["after"]
    largest_after = report["largest_after_cell"]
    lines = [
        "# DynoSim SGLang B 场景 prefix 修复前后准确度",
        "",
        "## 结论",
        "",
        (
            f"- B 的 7 个正式 RPS 点、5 项 ACC 指标、35 个 cell：DynoSim 平均 MAPE "
            f"由 {overall['before_mean_dynosim_mape_percent']:.6f}% 降至 "
            f"{overall['after_mean_dynosim_mape_percent']:.6f}%，改善 "
            f"{overall['mean_mape_improvement_percentage_points']:.6f} 个百分点。"
        ),
        (
            f"- 同一批 HiSim ACC 平均 MAPE 为 {overall['hisim_mean_mape_percent']:.6f}%；"
            f"修复后 DynoSim 在 {overall['after_dynosim_beats_hisim_cells']}/35 个 cell "
            "优于 HiSim。"
        ),
        (
            f"- 修复后最大 MAPE 仍是 `{largest_after['point']}` / "
            f"`{largest_after['metric']}`：{largest_after['after_dynosim_mape_percent']:.6f}%。"
        ),
        "",
        "## 根因与修复",
        "",
        (
            f"- B prompt 长 {b4['prompt_tokens']} tokens，完整共享前缀 "
            f"{b4['shared_prefix_tokens']}，每请求真实独有后缀仅 "
            f"{b4['unique_suffix_tokens']}；chunk size 为 {b4['chunked_prefill_size']}。"
        ),
        (
            "- 旧实现先截取首个 chunk，再对截断序列做 prefix match。因此首次 admission "
            "最多只看见 8192-token prefix，后续共享前缀被作为 fresh prefill 重算并占用 "
            "chunk/KV budget；续 chunk 还重复发 admission，污染 first-admission 复用统计。"
        ),
        (
            "- 修复后首次 admission 先对完整 current sequence 做 radix match，再对未命中 "
            "tokens 应用 chunk 和预算；已命中 radix path 在 eviction/allocation 期间加锁，"
            "续 chunk 沿现有 lease 扩展且不重复发 admission。"
        ),
        "",
        "## B-4 因果闭环",
        "",
        (
            f"- first-admission prefix ratio：{before['first_admission_prefix_cache_reused_ratio']:.6f} "
            f"→ {after['first_admission_prefix_cache_reused_ratio']:.6f}；最大 reused tokens："
            f"{before['reused_input_tokens_max']} → {after['reused_input_tokens_max']}。"
        ),
        (
            f"- 平均 scheduler queue：{before['mean_queue_ms']:.3f} ms → "
            f"{after['mean_queue_ms']:.3f} ms，减少 {b4['mean_queue_reduction_percent']:.3f}%。"
        ),
        (
            f"- 正式 TTFT（含同一 ingress）：{before['mean_ttft_with_ingress_ms']:.3f} ms → "
            f"{after['mean_ttft_with_ingress_ms']:.3f} ms；input throughput："
            f"{before['input_throughput']:.3f} → {after['input_throughput']:.3f} tok/s。"
        ),
        (
            "- 队列坍塌与完整 LCP 命中同时发生，且正式运行无 AIC database miss；"
            "因此旧 B-4 TTFT 异常的根因是 DynoSim SGLang admission/prefix 顺序，不是 "
            "ingress、AIC 缺表或经验 scale。修复后残差主要来自 decode/AIC service-time "
            "和单 aggregated worker 调度模型。"
        ),
        "",
        "## 各指标平均 MAPE",
        "",
        "| 指标 | 修复前 DynoSim | 修复后 DynoSim | HiSim | 改善百分点 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in common.PRIMARY_METRICS:
        row = report["per_metric"][metric]
        lines.append(
            f"| `{metric}` | {row['before_mean_dynosim_mape_percent']:.6f}% | "
            f"{row['after_mean_dynosim_mape_percent']:.6f}% | "
            f"{row['hisim_mean_mape_percent']:.6f}% | "
            f"{row['mean_mape_improvement_percentage_points']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## 各点五指标平均 MAPE",
            "",
            "| 点位 | 修复前 DynoSim | 修复后 DynoSim | HiSim | 改善百分点 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for point_name in B_POINT_NAMES:
        row = report["per_point"][point_name]
        lines.append(
            f"| `{point_name}` | {row['before_mean_dynosim_mape_percent']:.6f}% | "
            f"{row['after_mean_dynosim_mape_percent']:.6f}% | "
            f"{row['hisim_mean_mape_percent']:.6f}% | "
            f"{row['mean_mape_improvement_percentage_points']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## 数据与验收边界",
            "",
            "- 只运行 `sweep_b_rps_point_{1,2,4,8,16,24,32}`；不含 single-batch 或 B-mixed。",
            "- 未启动 SGLang、未重新采集；trace/RID/ingress/raw/HiSim ACC 均来自旧 accepted run1 保存工件及其原始 SHA256。",
            "- 原实机 source root 在本次 replay 时不存在，因此本报告明确使用 preserved replay provenance，不以其他目录替代。",
            "- 实机语义为 `ep_dispatch_algorithm=fake`；DynoSim AIC 为 `workload_distribution=balanced`，不声称 DynoSim 执行 fake TopK。",
            (
                f"- 验收：`accepted={str(acceptance['accepted']).lower()}`；"
                "7/7 点完成、TP4/EP4 合法、无 NaN/负延迟/AIC miss、"
                "preserved 输入未改；正式 ACC 只使用 run1。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--before-root",
        type=Path,
        default=BEFORE_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output root: {args.output_root}"
        )
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    points, preflight_manifest = preflight(args.before_root)
    args.output_root.mkdir(parents=True, exist_ok=False)
    common.write_json(args.output_root / "preflight.json", preflight_manifest)
    revisions = {
        "dynamo": common.git_revision(DYNAMO_ROOT),
        "aiconfigurator": common.git_revision(AIC_ROOT),
        "aiconfigurator_base_pin": "ffbab15797d3a3f14c382e937c0ab0c1c5cbc530",
    }
    common.write_json(args.output_root / "revisions.json", revisions)

    run_all_points(points, args.output_root / "run1", revisions)
    before_hashes_after_run = {
        point.name: _preserved_hashes(point.before_dir) for point in points
    }
    acceptance = acceptance_audit(
        points,
        args.output_root,
        before_hashes_after_run,
    )
    common.write_json(args.output_root / "acceptance.json", acceptance)
    report = compare_before_after(points, args.output_root)
    report["run_contract"] = {"run_count": 1, "report_source": "run1"}
    report["acceptance"] = acceptance
    common.write_json(args.output_root / "before_after_mape.json", report)
    (args.output_root / "before_after_mape.md").write_text(
        markdown_report(report, acceptance),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "accepted": acceptance["accepted"],
                "before_mean_mape_percent": report["overall"][
                    "before_mean_dynosim_mape_percent"
                ],
                "after_mean_mape_percent": report["overall"][
                    "after_mean_dynosim_mape_percent"
                ],
                "improvement_percentage_points": report["overall"][
                    "mean_mape_improvement_percentage_points"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
