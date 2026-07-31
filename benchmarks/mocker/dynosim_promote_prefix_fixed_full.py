#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Promote accepted prefix-fixed run1 points into the original 35-point root.

The promotion keeps the original pre-fix ``run1`` untouched and creates one
self-contained ``prefix_fixed_full/run1``. A/C/D come from the original run1;
B and B-mixed come from their accepted prefix-fix run1 artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


WORKSPACE = Path("/workspace/hisim_align")
FULL_ROOT = WORKSPACE / (
    "dynosim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_"
    "h100pcie_gsp_mixed_seed11_full"
)
B_FIX_ROOT = WORKSPACE / (
    "dynosim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_"
    "h100pcie_gsp_mixed_seed11_full_b_prefix_fix"
)
BMIXED_FIX_ROOT = WORKSPACE / (
    "dynosim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_"
    "h100pcie_gsp_mixed_seed11_full_bmixed_prefix_fix"
)
OUTPUT_NAME = "prefix_fixed_full"
FINAL_REPORT_JSON = "final_accuracy_report.json"
FINAL_REPORT_MD = "final_accuracy_report.md"
FINAL_ACCEPTANCE_JSON = "final_acceptance.json"
RPS_POINTS = (1, 2, 4, 8, 16, 24, 32)
PRIMARY_METRICS = (
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
    "mean_e2e_latency_ms",
    "input_throughput",
)
DATABASE_MISS_MARKERS = (
    "perf data not available",
    "perfdatnotavailable",
    "database miss",
    "negative latency",
)
TOP_LEVEL_OUTPUTS = {
    OUTPUT_NAME,
    FINAL_REPORT_JSON,
    FINAL_REPORT_MD,
    FINAL_ACCEPTANCE_JSON,
}


def point_names() -> tuple[str, ...]:
    names = []
    for scenario in ("a", "b", "b_mixed", "c", "d"):
        names.extend(f"sweep_{scenario}_rps_point_{rps}" for rps in RPS_POINTS)
    return tuple(names)


EXPECTED_POINTS = point_names()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield value


def file_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def protected_manifest(root: Path) -> dict[str, str]:
    manifest = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] in TOP_LEVEL_OUTPUTS:
            continue
        if relative.parts[0].startswith(f".{OUTPUT_NAME}.tmp."):
            continue
        manifest[str(relative)] = sha256_file(path)
    return manifest


def component_status(root: Path, expected_points: int) -> dict[str, Any]:
    acceptance_path = root / "acceptance.json"
    acceptance = read_json(acceptance_path)
    run = acceptance.get("runs", {}).get("run1", {})
    status = {
        "root": str(root),
        "acceptance_sha256": sha256_file(acceptance_path),
        "accepted": acceptance.get("accepted") is True,
        "run_count": len(acceptance.get("runs", {})),
        "run1_point_count": run.get("point_count"),
    }
    if (
        not status["accepted"]
        or status["run_count"] != 1
        or status["run1_point_count"] != expected_points
    ):
        raise ValueError(f"component acceptance failed: {status}")
    return status


def source_root_for(point: str, full_root: Path, b_root: Path, bmixed_root: Path) -> Path:
    if point.startswith("sweep_b_mixed_rps_point_"):
        return bmixed_root
    if point.startswith("sweep_b_rps_point_"):
        return b_root
    return full_root


def validate_source_layout(
    full_root: Path,
    b_root: Path,
    bmixed_root: Path,
) -> dict[str, Any]:
    components = {
        "original_full": component_status(full_root, 35),
        "b_prefix_fix": component_status(b_root, 7),
        "bmixed_prefix_fix": component_status(bmixed_root, 7),
    }
    original_points = {
        path.name for path in (full_root / "run1").glob("sweep_*_rps_point_*")
    }
    if original_points != set(EXPECTED_POINTS):
        raise ValueError(
            f"{full_root}/run1: expected 35 canonical points, "
            f"got {len(original_points)}"
        )
    for point in EXPECTED_POINTS:
        source_root = source_root_for(point, full_root, b_root, bmixed_root)
        source = source_root / "run1" / point
        if not source.is_dir():
            raise FileNotFoundError(source)
    return components


def copy_component_reports(
    staging: Path,
    full_root: Path,
    b_root: Path,
    bmixed_root: Path,
) -> None:
    selections = {
        "original_full": (
            full_root,
            ("acceptance.json", "accuracy_report.json", "accuracy_report.md"),
        ),
        "b_prefix_fix": (
            b_root,
            (
                "acceptance.json",
                "before_after_mape.json",
                "before_after_mape.md",
                "revisions.json",
            ),
        ),
        "bmixed_prefix_fix": (
            bmixed_root,
            (
                "acceptance.json",
                "prefix_acceptance.json",
                "before_after_mape.json",
                "before_after_mape.md",
                "merged_accuracy_report.json",
                "merged_accuracy_report.md",
                "revisions.json",
            ),
        ),
    }
    for name, (root, files) in selections.items():
        destination = staging / "components" / name
        destination.mkdir(parents=True)
        for filename in files:
            source = root / filename
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, destination / filename)


def validate_point(point_dir: Path) -> tuple[int, Counter[int]]:
    mape = read_json(point_dir / "mape.json")
    bench_rows = list(load_jsonl(point_dir / "bench_per_request.jsonl"))
    dynosim_rows = list(load_jsonl(point_dir / "dynosim_per_request.jsonl"))
    trace_rows = list(load_jsonl(point_dir / "trace.mooncake.jsonl"))
    completed = int(mape["completed"])
    if not (len(bench_rows) == len(dynosim_rows) == len(trace_rows) == completed):
        raise ValueError(
            f"{point_dir}: completed/bench/dynosim/trace mismatch "
            f"{completed}/{len(bench_rows)}/{len(dynosim_rows)}/{len(trace_rows)}"
        )
    if any("HEALTH" in str(row.get("source_rid", "")) for row in bench_rows):
        raise ValueError(f"{point_dir}: health request found")
    numeric_fields = (
        "arrival_time_ms",
        "e2e_latency_ms",
        "first_admit_ms",
        "first_token_ms",
        "input_length",
        "itl_ms",
        "last_token_ms",
        "output_length",
        "requested_output_length",
        "reused_input_tokens",
        "scheduler_e2e_latency_ms",
        "scheduler_ttft_ms",
        "terminal_time_ms",
        "ttft_ms",
    )
    for row in bench_rows:
        for field in numeric_fields:
            value = row.get(field)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{point_dir}: invalid {field}={value!r}")
    stderr = (point_dir / "replay.stderr.log").read_text(errors="replace").lower()
    if any(marker in stderr for marker in DATABASE_MISS_MARKERS):
        raise ValueError(f"{point_dir}: AIC database miss marker found")
    return completed, Counter(
        int(row["requested_output_length"]) for row in bench_rows
    )


def copy_full_run(
    staging: Path,
    full_root: Path,
    b_root: Path,
    bmixed_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_mapping = {}
    run_dir = staging / "run1"
    run_dir.mkdir()
    total_completed = 0
    output_lengths: Counter[int] = Counter()
    point_means = []
    for point in EXPECTED_POINTS:
        source_root = source_root_for(point, full_root, b_root, bmixed_root)
        source = source_root / "run1" / point
        destination = run_dir / point
        source_hashes = file_manifest(source)
        shutil.copytree(source, destination, copy_function=shutil.copy2)
        destination_hashes = file_manifest(destination)
        if source_hashes != destination_hashes:
            raise ValueError(f"{point}: copied artifact hash mismatch")
        completed, lengths = validate_point(destination)
        total_completed += completed
        output_lengths.update(lengths)
        point_means.append(
            float(read_json(destination / "mape.json")["mean_dynosim_mape_percent"])
        )
        source_mapping[f"run1/{point}"] = {
            "source": str(source),
            "source_file_count": len(source_hashes),
            "source_tree_sha256": hashlib.sha256(
                json.dumps(source_hashes, sort_keys=True).encode()
            ).hexdigest(),
            "destination": f"run1/{point}",
        }
    summary = {
        "schema": "dynosim_prefix_fixed_full_run_summary.v2",
        "run": "run1",
        "point_count": len(EXPECTED_POINTS),
        "completed_requests": total_completed,
        "per_request_rows": total_completed,
        "output_length_distribution": {
            str(key): value for key, value in sorted(output_lengths.items())
        },
        "mean_of_point_mean_mape_percent": statistics.fmean(point_means),
    }
    write_json(run_dir / "run_summary.json", summary)
    return source_mapping, summary


def values_match(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def final_report(
    staging: Path,
    full_root: Path,
    merged_report_path: Path,
    components: dict[str, Any],
) -> dict[str, Any]:
    report = read_json(merged_report_path)
    if len(report.get("cells", [])) != 175:
        raise ValueError("merged report does not contain 175 cells")
    seen = set()
    for cell in report["cells"]:
        point = cell["point"]
        metric = cell["metric"]
        seen.add((point, metric))
        current = read_json(staging / "run1" / point / "mape.json")["metrics"][metric]
        for key in (
            "raw",
            "dynosim",
            "dynosim_mape_percent",
            "hisim_mape_percent",
            "delta_percentage_points",
            "winner",
        ):
            if not values_match(cell[key], current[key]):
                raise ValueError(
                    f"run1/{point}/{metric}/{key}: "
                    f"report={cell[key]!r}, point={current[key]!r}"
                )
    expected_cells = {
        (point, metric) for point in EXPECTED_POINTS for metric in PRIMARY_METRICS
    }
    if seen != expected_cells:
        raise ValueError("final report point/metric coverage mismatch")
    report.pop("determinism", None)
    report["schema"] = "dynosim_prefix_fixed_full_accuracy_report.v2"
    report["scope"]["result_root"] = str(full_root)
    report["scope"]["full_run_dir"] = OUTPUT_NAME
    report["run_contract"] = {"run_count": 1, "report_source": "run1"}
    report["promotion"] = {
        "layout": {
            "pre_fix_runs": ["run1"],
            "prefix_fixed_runs": [f"{OUTPUT_NAME}/run1"],
        },
        "scenario_sources": {
            "a": "original accepted run1",
            "b": "accepted B prefix-fix run1",
            "b_mixed": "accepted B-mixed prefix-fix run1",
            "c": "original accepted run1",
            "d": "original accepted run1",
        },
        "component_acceptance": components,
        "data_contract": {
            "real_data_recollected": False,
            "sglang_server_started": False,
            "gpu_preflight_required": False,
            "offline_aggregated_sglang_path": True,
            "real_baseline_routing": "fake",
            "dynosim_aic_workload_distribution": "balanced",
            "dynosim_executes_fake_topk_routing": False,
            "empirical_scaling_applied": False,
            "ingress_envelope_reused": True,
        },
    }
    report["provenance"]["final_report_source"] = str(merged_report_path)
    report["provenance"]["final_report_source_sha256"] = sha256_file(
        merged_report_path
    )
    return report


def report_markdown(report: dict[str, Any], acceptance: dict[str, Any]) -> str:
    overall = report["overall"]
    largest = report["largest_updated_cell"]
    lines = [
        "# DynoSim+AIC prefix 修复后最终 ACC 报告",
        "",
        "## 正式结论",
        "",
        (
            f"- 35 个正式点、175 个 point-metric cell：DynoSim+AIC 平均 "
            f"MAPE {overall['original_dynosim_mean_mape_percent']:.6f}% → "
            f"{overall['updated_dynosim_mean_mape_percent']:.6f}%；"
            f"HiSim 为 {overall['hisim_mean_mape_percent']:.6f}%。"
        ),
        (
            f"- 修复后胜负：DynoSim {overall['dynosim_wins']}，"
            f"HiSim {overall['hisim_wins']}，tie {overall['ties']}。"
        ),
        (
            f"- 修复后的全局最大 cell：`{largest['point']}` / "
            f"`{largest['metric']}` = {largest['dynosim_mape_percent']:.6f}%。"
        ),
        "",
        "## 分场景平均 MAPE",
        "",
        "| 场景 | 修复前 DynoSim | 修复后 DynoSim | HiSim |",
        "|---|---:|---:|---:|",
    ]
    for scenario in ("a", "b", "b_mixed", "c", "d"):
        item = report["per_scenario"][scenario]
        lines.append(
            f"| `{scenario}` | "
            f"{item['original_dynosim_mean_mape_percent']:.6f}% | "
            f"{item['updated_dynosim_mean_mape_percent']:.6f}% | "
            f"{item['hisim_mean_mape_percent']:.6f}% |"
        )
    lines.extend(
        [
            "",
            "## 完整运行布局",
            "",
            "- 原始修复前运行保留在 `run1/`。",
            "- 修复后完整运行位于 `prefix_fixed_full/run1/`。",
            "- A/C/D 复用原 accepted run1；B 与 B-mixed 使用各自已验收的 "
            "prefix-fix run1。",
            "",
            "## 验收与数据边界",
            "",
            (
                f"- `accepted={str(acceptance['accepted']).lower()}`："
                f"{acceptance['runs']['run1']['point_count']}/35 点、"
                f"{acceptance['runs']['run1']['completed_requests']}/3432 请求。"
            ),
            "- ACC report 仅由 `prefix_fixed_full/run1` 生成，不进行双跑平均。",
            "- 无 NaN、负延迟、health request 或 AIC database miss；"
            "TP4/EP4 映射沿用三个 accepted component 的验收。",
            "- 未启动 SGLang、未重新采集实机数据、未做 GPU preflight、"
            "未使用经验缩放。",
            "- 实机 baseline 为 fake 路由语义；DynoSim 仅使用 AIC balanced "
            "算子计时，不声称执行真实 fake TopK。",
            "- 本报告的 DynoSim TTFT/E2E 按任务既定合同复用实机 ingress "
            "envelope；它不是一次新的独立实机 client benchmark。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-root", type=Path, default=FULL_ROOT)
    parser.add_argument("--b-fix-root", type=Path, default=B_FIX_ROOT)
    parser.add_argument("--bmixed-fix-root", type=Path, default=BMIXED_FIX_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.full_root / OUTPUT_NAME
    top_level_outputs = [
        output,
        args.full_root / FINAL_REPORT_JSON,
        args.full_root / FINAL_REPORT_MD,
        args.full_root / FINAL_ACCEPTANCE_JSON,
    ]
    existing = [str(path) for path in top_level_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite promoted outputs: {existing}")

    components = validate_source_layout(
        args.full_root, args.b_fix_root, args.bmixed_fix_root
    )
    protected_before = protected_manifest(args.full_root)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{OUTPUT_NAME}.tmp.", dir=args.full_root)
    )
    try:
        source_mapping, run_summary = copy_full_run(
            staging, args.full_root, args.b_fix_root, args.bmixed_fix_root
        )
        copy_component_reports(
            staging, args.full_root, args.b_fix_root, args.bmixed_fix_root
        )
        report = final_report(
            staging,
            args.full_root,
            args.bmixed_fix_root / "merged_accuracy_report.json",
            components,
        )
        acceptance = {
            "schema": "dynosim_prefix_fixed_full_acceptance.v2",
            "accepted": True,
            "checks": {
                "points_35_of_35": True,
                "completed_matches_preserved_workload": True,
                "single_run_contract": True,
                "component_acceptance": all(
                    item["accepted"] for item in components.values()
                ),
                "copied_source_hashes_exact": True,
                "aic_database_miss_free": True,
                "health_request_free": True,
                "nonnegative_finite_metrics": True,
                "tp4_ep4_mapping_valid": True,
                "original_pre_fix_artifacts_unchanged": False,
            },
            "runs": {
                "run1": {
                    "point_count": run_summary["point_count"],
                    "completed_requests": run_summary["completed_requests"],
                    "per_request_rows": run_summary["per_request_rows"],
                    "output_length_distribution": run_summary[
                        "output_length_distribution"
                    ],
                }
            },
            "execution_contract": report["promotion"]["data_contract"],
            "component_acceptance": components,
        }
        if (
            run_summary["point_count"] != 35
            or run_summary["completed_requests"] != 3432
            or run_summary["per_request_rows"] != 3432
        ):
            raise ValueError(f"full-run request acceptance failed: {run_summary}")
        write_json(staging / "acceptance.json", acceptance)
        write_json(
            staging / "promotion_manifest.json",
            {
                "schema": "dynosim_prefix_fixed_full_promotion.v2",
                "result_root": str(args.full_root),
                "output": OUTPUT_NAME,
                "run_contract": {"run_count": 1, "report_source": "run1"},
                "scenario_sources": report["promotion"]["scenario_sources"],
                "source_mapping": source_mapping,
                "protected_original_file_count": len(protected_before),
                "protected_original_hashes": protected_before,
            },
        )
        staging.rename(output)

        protected_after = {
            relative: sha256_file(args.full_root / relative)
            for relative in protected_before
            if (args.full_root / relative).is_file()
        }
        acceptance["checks"]["original_pre_fix_artifacts_unchanged"] = (
            protected_after == protected_before
        )
        acceptance["accepted"] = all(acceptance["checks"].values())
        if not acceptance["accepted"]:
            raise ValueError(f"promotion acceptance failed: {acceptance['checks']}")

        report["acceptance"] = acceptance
        report_text = json.dumps(
            report, indent=2, sort_keys=True, ensure_ascii=False
        ) + "\n"
        acceptance_text = json.dumps(
            acceptance, indent=2, sort_keys=True, ensure_ascii=False
        ) + "\n"
        markdown = report_markdown(report, acceptance)
        atomic_write(output / "acceptance.json", acceptance_text)
        atomic_write(output / "accuracy_report.json", report_text)
        atomic_write(output / "accuracy_report.md", markdown)
        atomic_write(args.full_root / FINAL_REPORT_JSON, report_text)
        atomic_write(args.full_root / FINAL_REPORT_MD, markdown)
        atomic_write(args.full_root / FINAL_ACCEPTANCE_JSON, acceptance_text)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(
        json.dumps(
            {
                "result_root": str(args.full_root),
                "full_run": str(output),
                "final_report": str(args.full_root / FINAL_REPORT_JSON),
                "accepted": True,
                "run_count": 1,
                "point_count": 35,
                "completed_requests": 3432,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
