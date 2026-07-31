#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run pure CPU single-batch replay & simulation for Dynamo+AIC alignment.

This script loads real ground truth SGLang data from the reference directory,
replays each of the 35 single-batch points using pure CPU Dynamo+AIC simulation,
computes MAPEs against raw SGLang metrics, updates accuracy_report.json, and
regenerates sweep_acc_report.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Allow direct execution by file path, where Python otherwise exposes only this
# script's directory rather than the Dynamo repository root.
DYNAMO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DYNAMO_ROOT))

from benchmarks.mocker.dynosim_hisim_alignment import (
    aic_database_manifest,
    sha256_file,
)

REAL_DATA_ROOT = Path(
    "/workspace/models/H100_bkup/"
    "hisim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_"
    "h100pcie_gsp_mixed_seed11_full"
)

TARGET_RUN_DIR = Path(
    "/workspace/hisim_align/"
    "dynosim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_"
    "h100pcie_gsp_mixed_seed11_full/prefix_fixed_full"
)

AIC_ROOT = Path("/workspace/hisim_align/aiconfigurator-dynosim-e2e")
AIC_SRC = AIC_ROOT / "src"
DYNAMO_SRC = Path("/workspace/hisim_align/dynamo/components/src")
PYTHON_EXE = Path("/workspace/hisim_align/dynamo/.venv/bin/python")
MODEL_PATH = Path(
    "/workspace/hisim_align/hisim_sglang_align/model_cache/"
    "DeepSeek-V3_layers20_maxpos163840"
)

SCENARIOS = ("a", "b", "b_mixed", "c", "d")
BATCH_SIZES = (1, 2, 4, 8, 16, 24, 32)
PRIMARY_METRICS = (
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
    "mean_e2e_latency_ms",
    "input_throughput",
)


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
        "aic_kv_cache_dtype": "bfloat16",
        "aic_mla_module_kv_cache_dtype": "fp8",
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
            "cuda_graph_batch_sizes": [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64],
            "enable_mixed_chunk": False,
        },
    }


def load_point_raw_and_hisim(source_point_dir: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Load raw ground truth metrics and HiSim MAPE for a single-batch point."""
    # Read raw metrics from raw_sglang/client_metrics.jsonl or compare_aic
    raw_metrics: dict[str, float] = {}
    hisim_mapes: dict[str, float] = {}

    mape_json_path = source_point_dir / "compare_aic" / "raw_hisim_aic_stdout_mape.json"
    if mape_json_path.exists():
        data = json.loads(mape_json_path.read_text())["metrics"]
        for m in PRIMARY_METRICS:
            if m in data:
                raw_metrics[m] = float(data[m]["raw"])
                hisim_mapes[m] = float(data[m]["mape_percent"])
    else:
        # Fallback to client_metrics.jsonl
        client_metrics_path = source_point_dir / "raw_sglang" / "client_metrics.jsonl"
        lines = [json.loads(l) for l in client_metrics_path.read_text().splitlines() if l.strip()]
        summary = lines[-1]
        for m in PRIMARY_METRICS:
            val_key = "input_throughput" if m == "input_throughput" else m
            raw_metrics[m] = float(summary[val_key])
            hisim_mapes[m] = 0.0

    return raw_metrics, hisim_mapes


def run_single_batch_point(
    scenario: str,
    batch_size: int,
    real_root: Path,
    output_run_dir: Path,
) -> dict[str, Any]:
    point_name = f"sweep_{scenario}_single_batch_point_{batch_size}"
    source_point_dir = real_root / point_name
    if not source_point_dir.exists():
        raise FileNotFoundError(f"Source point directory not found: {source_point_dir}")

    # Read requests from actual/requests.jsonl
    req_path = source_point_dir / "actual" / "requests.jsonl"
    if not req_path.exists():
        raise FileNotFoundError(f"Missing requests.jsonl in {source_point_dir}")

    requests = [json.loads(line) for line in req_path.read_text().splitlines() if line.strip()]

    point_out_dir = output_run_dir / "run1" / point_name
    point_out_dir.mkdir(parents=True, exist_ok=True)

    trace_path = point_out_dir / "trace.mooncake.jsonl"
    report_path = point_out_dir / "dynosim_report.json"
    per_request_path = point_out_dir / "dynosim_per_request.jsonl"

    trace_rows = []
    for r in requests:
        trace_rows.append(
            {
                "request_id": r["rid"],
                "input_length": r["input_length"],
                "output_length": r.get("requested_output_length", r.get("output_length")),
                "hash_ids": r["input_ids"],
                "timestamp": 0.0,
            }
        )

    trace_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in trace_rows) + "\n",
        encoding="utf-8",
    )

    # Execute dynamo.replay using pure CPU simulation
    args = engine_args()
    cmd = [
        str(PYTHON_EXE),
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

    env = os.environ.copy()
    pythonpath = [str(AIC_SRC), str(DYNAMO_SRC)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env["AICONFIGURATOR_SYSTEMS_PATH"] = str(AIC_SRC / "aiconfigurator" / "systems")

    (point_out_dir / "engine_args.json").write_text(
        json.dumps(args, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (point_out_dir / "command.json").write_text(
        json.dumps({"argv": cmd, "offline": True}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "schema": "dynosim_single_batch_aic_provenance.v1",
        "database": aic_database_manifest(
            Path(env["AICONFIGURATOR_SYSTEMS_PATH"]), str(args["aic_system"])
        ),
        "source": {
            "requests_path": str(req_path),
            "requests_sha256": sha256_file(req_path),
            "raw_hisim_mape_path": str(
                source_point_dir / "compare_aic" / "raw_hisim_aic_stdout_mape.json"
            ),
        },
        "historical_hisim_resolved_database_root": "unrecorded",
    }
    (point_out_dir / "aic_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    proc = subprocess.run(
        cmd, cwd=DYNAMO_ROOT, env=env, capture_output=True, text=True
    )
    (point_out_dir / "replay.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (point_out_dir / "replay.stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Replay failed for {point_name}:\n{proc.stderr}")
    stderr_lower = proc.stderr.lower()
    if any(
        marker in stderr_lower
        for marker in ("database miss", "perf data not available", "perfdatnotavailable")
    ):
        raise RuntimeError(f"AIC database miss detected for {point_name}")

    dynosim_report = json.loads(report_path.read_text())
    raw_metrics, hisim_mapes = load_point_raw_and_hisim(source_point_dir)

    metrics_res = {}
    cell_list = []
    for m in PRIMARY_METRICS:
        raw_val = raw_metrics[m]
        dynosim_val = (
            float(dynosim_report["input_throughput_tok_s"])
            if m == "input_throughput"
            else float(dynosim_report[m])
        )
        dynosim_mape = abs(dynosim_val - raw_val) / abs(raw_val) * 100.0 if raw_val != 0 else 0.0
        hisim_mape = hisim_mapes.get(m, 0.0)

        metrics_res[m] = {
            "raw": raw_val,
            "dynosim": dynosim_val,
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

        cell_list.append(
            {
                "point": point_name,
                "metric": m,
                "raw": raw_val,
                "dynosim": dynosim_val,
                "dynosim_mape_percent": dynosim_mape,
                "hisim_mape_percent": hisim_mape,
                "delta_percentage_points": dynosim_mape - hisim_mape,
                "original_dynosim": dynosim_val,
                "original_dynosim_mape_percent": dynosim_mape,
                "replacement_source": "pure_cpu_single_batch_replay",
                "winner": metrics_res[m]["winner"],
            }
        )

    mape_json = {
        "point": point_name,
        "completed": len(requests),
        "metric_count": len(PRIMARY_METRICS),
        "mean_dynosim_mape_percent": sum(v["dynosim_mape_percent"] for v in metrics_res.values())
        / len(metrics_res),
        "mean_hisim_mape_percent": sum(v["hisim_mape_percent"] for v in metrics_res.values())
        / len(metrics_res),
        "metrics": metrics_res,
        "schema": "dynosim_hisim_point_comparison.v1",
    }
    (point_out_dir / "mape.json").write_text(
        json.dumps(mape_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {"point": point_name, "cells": cell_list, "mape": mape_json}


def update_accuracy_report(target_dir: Path, new_cells: list[dict[str, Any]]) -> None:
    """Update accuracy_report.json and final_accuracy_report.json to include single-batch cells."""
    for report_filename in ("accuracy_report.json", "final_accuracy_report.json"):
        report_path = target_dir / report_filename
        if not report_path.exists():
            continue

        report_data = json.loads(report_path.read_text())
        existing_cells = report_data.get("cells", [])

        # Filter out old cells for the new points if any
        new_point_names = {c["point"] for c in new_cells}
        filtered_cells = [c for c in existing_cells if c["point"] not in new_point_names]
        merged_cells = filtered_cells + new_cells

        report_data["cells"] = merged_cells
        report_data.update(summarize_cells(merged_cells))


        report_path.write_text(
            json.dumps(report_data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[single_batch] Updated {report_path} (total cells: {len(merged_cells)})")


def summarize_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    def summary(subset: list[dict[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {
                "cell_count": 0,
                "mean_dynosim_mape_percent": None,
                "mean_hisim_mape_percent": None,
                "mean_delta_percentage_points": None,
                "dynosim_wins": 0,
                "hisim_wins": 0,
                "ties": 0,
            }
        return {
            "cell_count": len(subset),
            "mean_dynosim_mape_percent": sum(
                float(cell["dynosim_mape_percent"]) for cell in subset
            )
            / len(subset),
            "mean_hisim_mape_percent": sum(
                float(cell["hisim_mape_percent"]) for cell in subset
            )
            / len(subset),
            "mean_delta_percentage_points": sum(
                float(cell["delta_percentage_points"]) for cell in subset
            )
            / len(subset),
            "dynosim_wins": sum(cell["winner"] == "dynosim" for cell in subset),
            "hisim_wins": sum(cell["winner"] == "hisim" for cell in subset),
            "ties": sum(cell["winner"] == "tie" for cell in subset),
        }

    cells = sorted(cells, key=lambda cell: (cell["point"], cell["metric"]))
    methods = {
        "single_batch": [cell for cell in cells if "_single_batch_point_" in cell["point"]],
        "rps": [cell for cell in cells if "_rps_point_" in cell["point"]],
    }
    per_metric = {
        metric: summary([cell for cell in cells if cell["metric"] == metric])
        for metric in PRIMARY_METRICS
    }
    per_scenario = {}
    for scenario in SCENARIOS:
        per_scenario[scenario] = summary(
            [
                cell
                for cell in cells
                if cell["point"].startswith(f"sweep_{scenario}_rps_point_")
                or cell["point"].startswith(
                    f"sweep_{scenario}_single_batch_point_"
                )
            ]
        )
    return {
        "schema": "dynosim_offline_replay_accuracy_diagnostic.v2",
        "point_count": len({cell["point"] for cell in cells}),
        "cell_count": len(cells),
        "overall": summary(cells),
        "per_method": {key: summary(value) for key, value in methods.items()},
        "per_metric": per_metric,
        "per_scenario": per_scenario,
        "largest_dynosim_mape_cell": max(
            cells, key=lambda cell: float(cell["dynosim_mape_percent"])
        ),
        "scope": {
            "offline_replay_diagnostic": True,
            "formal_hisim_acc": False,
            "point_count": len({cell["point"] for cell in cells}),
            "cell_count": len(cells),
            "single_batch_excluded": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pure CPU single batch replay and simulation for Dynamo+AIC."
    )
    parser.add_argument(
        "--real-root",
        type=Path,
        default=REAL_DATA_ROOT,
        help="Path to real SGLang single batch reference root.",
    )
    parser.add_argument(
        "--target-run-dir",
        type=Path,
        default=TARGET_RUN_DIR,
        help="Path to target prefix_fixed_full run directory.",
    )
    args = parser.parse_args()

    real_root = args.real_root.resolve()
    target_run_dir = args.target_run_dir.resolve()

    print(f"[single_batch] Running 35 single batch points...")
    all_cells = []
    completed_points = 0

    for sc in SCENARIOS:
        for bs in BATCH_SIZES:
            res = run_single_batch_point(sc, bs, real_root, target_run_dir)
            all_cells.extend(res["cells"])
            completed_points += 1
            print(f"  [{completed_points}/35] Completed Scenario {sc} Batch Size {bs}")

    # Update accuracy_report.json in target_run_dir and its parent
    update_accuracy_report(target_run_dir, all_cells)
    if target_run_dir.parent.exists() and (target_run_dir.parent / "accuracy_report.json").exists():
        update_accuracy_report(target_run_dir.parent, all_cells)

    # Regenerate reports using dynosim_generate_acc_report.py
    print("[single_batch] Regenerating sweep_acc_report.md and plots...")
    generator_script = DYNAMO_ROOT / "benchmarks" / "mocker" / "dynosim_generate_acc_report.py"
    subprocess.run(
        [
            sys.executable,
            str(generator_script),
            "--run-dir",
            str(target_run_dir),
            "--output-dir",
            str(target_run_dir),
        ],
        check=True,
    )
    print("[single_batch] Single batch alignment completed successfully!")


if __name__ == "__main__":
    main()
