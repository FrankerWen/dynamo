#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Migrate the accepted DynoSim artifacts from two runs to run1-only ACC.

This is a destructive, one-shot migration. It deletes only the four validated
DynoSim ``run2`` directories in this alignment, removes stale determinism
reports, and regenerates every acceptance/ACC report from ``run1``.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import dynosim_promote_prefix_fixed_full as promote


FULL_ROOT = promote.FULL_ROOT
B_ROOT = promote.B_FIX_ROOT
BMIXED_ROOT = promote.BMIXED_FIX_ROOT
FIXED_FULL = FULL_ROOT / promote.OUTPUT_NAME
RPS_POINTS = promote.RPS_POINTS
PRIMARY_METRICS = promote.PRIMARY_METRICS
DETERMINISM_FILES = (
    FULL_ROOT / "determinism.json",
    FULL_ROOT / "final_determinism.json",
    B_ROOT / "determinism.json",
    BMIXED_ROOT / "determinism.json",
    FIXED_FULL / "determinism.json",
)
RUN2_TARGETS = (
    (FULL_ROOT / "run2", 35),
    (FIXED_FULL / "run2", 35),
    (B_ROOT / "run2", 7),
    (BMIXED_ROOT / "run2", 7),
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def tree_info(root: Path) -> dict[str, Any]:
    manifest = tree_manifest(root)
    return {
        "path": str(root),
        "file_count": len(manifest),
        "byte_count": sum(
            (root / relative).stat().st_size for relative in manifest
        ),
        "tree_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
    }


def expected_points(scenario: str | None) -> tuple[str, ...]:
    if scenario is None:
        return promote.EXPECTED_POINTS
    return tuple(
        f"sweep_{scenario}_rps_point_{rps}" for rps in RPS_POINTS
    )


def audit_run1(root: Path, points: Iterable[str]) -> dict[str, Any]:
    expected = tuple(points)
    point_dirs = {
        path.name: path
        for path in (root / "run1").glob("sweep_*_rps_point_*")
        if path.is_dir()
    }
    if set(point_dirs) != set(expected):
        raise ValueError(
            f"{root}/run1: expected {len(expected)} points, got {len(point_dirs)}"
        )
    completed = 0
    output_lengths: Counter[int] = Counter()
    topology_valid = True
    trace_identity = True
    for point in expected:
        point_dir = point_dirs[point]
        point_completed, lengths = promote.validate_point(point_dir)
        completed += point_completed
        output_lengths.update(lengths)
        engine = read_json(point_dir / "engine_args.json")
        topology = (
            engine.get("aic_tp_size"),
            engine.get("aic_attention_dp_size"),
            engine.get("aic_moe_tp_size"),
            engine.get("aic_moe_ep_size"),
        )
        topology_valid &= topology == (4, 1, 1, 4)
        source_manifest = read_json(point_dir / "source_manifest.json")
        trace_identity &= (
            sha256_file(point_dir / "trace.mooncake.jsonl")
            == source_manifest["trace_sha256"]
        )
    if not topology_valid or not trace_identity:
        raise ValueError(
            f"{root}/run1: topology_valid={topology_valid}, "
            f"trace_identity={trace_identity}"
        )
    return {
        "point_count": len(expected),
        "completed_requests": completed,
        "per_request_rows": completed,
        "output_length_distribution": {
            str(key): value for key, value in sorted(output_lengths.items())
        },
        "topology_valid": topology_valid,
        "trace_identity": trace_identity,
    }


def single_acceptance(
    root: Path,
    run: dict[str, Any],
    schema: str,
    old: dict[str, Any],
    expected_requests: int,
    prefix_exact: bool = False,
) -> dict[str, Any]:
    if (
        run["completed_requests"] != expected_requests
        or run["per_request_rows"] != expected_requests
    ):
        raise ValueError(f"{root}: request count mismatch {run}")
    checks = {
        "single_run_contract": True,
        "point_set_complete": True,
        "completed_matches_preserved_workload": True,
        "nonnegative_finite_metrics": True,
        "aic_database_miss_free": True,
        "health_request_free": True,
        "tp4_ep4_mapping_valid": run["topology_valid"],
        "canonical_trace_identity": run["trace_identity"],
    }
    if "preserved_before_artifacts_unchanged" in old.get("checks", {}):
        checks["preserved_before_artifacts_unchanged"] = bool(
            old["checks"]["preserved_before_artifacts_unchanged"]
        )
    if "source_hashes_unchanged_since_run" in old.get("checks", {}):
        checks["source_hashes_unchanged_since_run"] = bool(
            old["checks"]["source_hashes_unchanged_since_run"]
        )
    if prefix_exact:
        checks["bmixed_prefix_contract_exact"] = True
    acceptance = {
        "schema": schema,
        "accepted": all(checks.values()),
        "checks": checks,
        "runs": {
            "run1": {
                key: run[key]
                for key in (
                    "point_count",
                    "completed_requests",
                    "per_request_rows",
                    "output_length_distribution",
                )
            }
        },
        "run_contract": {"run_count": 1, "report_source": "run1"},
        "execution_contract": {
            **old.get("execution_contract", {}),
            "run_count": 1,
            "report_source": "run1",
        },
    }
    for key in (
        "source_point_count",
        "source_request_count",
        "preserved_request_count",
        "revisions",
        "build_artifacts",
        "build_artifact",
    ):
        if key in old:
            acceptance[key] = old[key]
    if not acceptance["accepted"]:
        raise ValueError(f"{root}: single-run acceptance failed {checks}")
    return acceptance


def update_json_report(
    path: Path,
    acceptance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = read_json(path)
    report.pop("determinism", None)
    report["run_contract"] = {"run_count": 1, "report_source": "run1"}
    if acceptance is not None:
        report["acceptance"] = acceptance
    write_json(path, report)
    return report


def update_component_markdown() -> None:
    b_path = B_ROOT / "before_after_mape.md"
    b_text = b_path.read_text()
    b_text = b_text.replace(
        "且两次运行均无 AIC database miss",
        "且正式 run1 无 AIC database miss",
    ).replace(
        "preserved 输入未改、两次复跑 deterministic。",
        "preserved 输入未改；正式 ACC 只保留并使用 run1。",
    )
    b_path.write_text(b_text, encoding="utf-8")

    bmixed_path = BMIXED_ROOT / "before_after_mape.md"
    bmixed_text = bmixed_path.read_text().replace(
        "7/7 点、672/672 请求每轮、无 NaN/负延迟/AIC miss、TP4/EP4 合法、"
        "preserved 输入未改、run1/run2 canonical 输出完全确定。",
        "run1 完成 7/7 点、672/672 请求；无 NaN/负延迟/AIC miss、"
        "TP4/EP4 合法、preserved 输入未改。",
    )
    bmixed_path.write_text(bmixed_text, encoding="utf-8")

    merged_path = BMIXED_ROOT / "merged_accuracy_report.md"
    merged_text = merged_path.read_text().replace(
        "A/C/D 使用原两轮 accepted 结果；B 使用已验收的 B-prefix-fix 两轮结果；"
        "B-mixed 使用本次两轮结果。",
        "A/C/D 使用原 accepted run1；B 使用已验收的 B-prefix-fix run1；"
        "B-mixed 使用本次 run1。",
    )
    merged_path.write_text(merged_text, encoding="utf-8")


def component_status(root: Path, acceptance: dict[str, Any]) -> dict[str, Any]:
    return {
        "root": str(root),
        "acceptance_sha256": sha256_file(root / "acceptance.json"),
        "accepted": acceptance["accepted"],
        "run_count": 1,
        "run1_point_count": acceptance["runs"]["run1"]["point_count"],
    }


def sync_component_copies() -> None:
    components = FIXED_FULL / "components"
    if components.exists():
        shutil.rmtree(components)
    promote.copy_component_reports(
        FIXED_FULL, FULL_ROOT, B_ROOT, BMIXED_ROOT
    )


def validate_final_cells(report: dict[str, Any]) -> None:
    cells = report.get("cells", [])
    if len(cells) != 175:
        raise ValueError(f"final report expected 175 cells, got {len(cells)}")
    seen = set()
    for cell in cells:
        point = cell["point"]
        metric = cell["metric"]
        seen.add((point, metric))
        current = read_json(
            FIXED_FULL / "run1" / point / "mape.json"
        )["metrics"][metric]
        for key in (
            "raw",
            "dynosim",
            "dynosim_mape_percent",
            "hisim_mape_percent",
            "delta_percentage_points",
            "winner",
        ):
            left = cell[key]
            right = current[key]
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                matches = math.isclose(
                    float(left), float(right), rel_tol=0.0, abs_tol=1e-12
                )
            else:
                matches = left == right
            if not matches:
                raise ValueError(
                    f"{point}/{metric}/{key}: report={left!r}, run1={right!r}"
                )
    expected = {
        (point, metric)
        for point in promote.EXPECTED_POINTS
        for metric in PRIMARY_METRICS
    }
    if seen != expected:
        raise ValueError("final report coverage mismatch")


def validate_delete_targets() -> list[dict[str, Any]]:
    targets = []
    for path, expected_count in RUN2_TARGETS:
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"invalid run2 deletion target: {path}")
        resolved = path.resolve()
        if resolved != path or path.name != "run2":
            raise ValueError(f"unsafe run2 deletion target: {path} -> {resolved}")
        points = list(path.glob("sweep_*_rps_point_*"))
        if len(points) != expected_count:
            raise ValueError(
                f"{path}: expected {expected_count} points, got {len(points)}"
            )
        targets.append(tree_info(path))
    return targets


def main() -> int:
    deleted_targets = validate_delete_targets()
    old_full_acceptance = read_json(FULL_ROOT / "acceptance.json")
    old_b_acceptance = read_json(B_ROOT / "acceptance.json")
    old_bmixed_acceptance = read_json(BMIXED_ROOT / "acceptance.json")

    full_run = audit_run1(FULL_ROOT, expected_points(None))
    b_run = audit_run1(B_ROOT, expected_points("b"))
    bmixed_run = audit_run1(BMIXED_ROOT, expected_points("b_mixed"))
    fixed_run = audit_run1(FIXED_FULL, expected_points(None))

    for path, _ in RUN2_TARGETS:
        shutil.rmtree(path)
    for path in DETERMINISM_FILES:
        if path.exists():
            path.unlink()

    full_acceptance = single_acceptance(
        FULL_ROOT,
        full_run,
        "dynosim_alignment_acceptance.v2",
        old_full_acceptance,
        3432,
    )
    b_acceptance = single_acceptance(
        B_ROOT,
        b_run,
        "dynosim_b_prefix_fix_acceptance.v2",
        old_b_acceptance,
        690,
    )
    bmixed_acceptance = single_acceptance(
        BMIXED_ROOT,
        bmixed_run,
        "dynosim_bmixed_prefix_fix_acceptance.v2",
        old_bmixed_acceptance,
        672,
        prefix_exact=True,
    )
    prefix_acceptance = read_json(BMIXED_ROOT / "prefix_acceptance.json")
    prefix_acceptance["schema"] = (
        "dynosim_bmixed_prefix_fix_prefix_acceptance.v2"
    )
    prefix_acceptance["runs"] = {
        "run1": prefix_acceptance["runs"]["run1"]
    }
    prefix_acceptance["run_contract"] = {
        "run_count": 1,
        "report_source": "run1",
    }
    bmixed_acceptance["prefix_acceptance"] = prefix_acceptance

    write_json(FULL_ROOT / "acceptance.json", full_acceptance)
    write_json(B_ROOT / "acceptance.json", b_acceptance)
    write_json(BMIXED_ROOT / "prefix_acceptance.json", prefix_acceptance)
    write_json(BMIXED_ROOT / "acceptance.json", bmixed_acceptance)

    original_report = update_json_report(FULL_ROOT / "accuracy_report.json")
    original_md = (FULL_ROOT / "accuracy_report.md").read_text()
    if "## 运行合同" not in original_md:
        original_md += (
            "\n\n## 运行合同\n\n"
            "- 正式 ACC 仅保留并使用 `run1`；不进行双跑或跨 run 平均。\n"
        )
    (FULL_ROOT / "accuracy_report.md").write_text(original_md, encoding="utf-8")
    update_json_report(B_ROOT / "before_after_mape.json", b_acceptance)
    update_json_report(
        BMIXED_ROOT / "before_after_mape.json", bmixed_acceptance
    )
    update_component_markdown()

    merged = read_json(BMIXED_ROOT / "merged_accuracy_report.json")
    merged.pop("determinism", None)
    merged["run_contract"] = {"run_count": 1, "report_source": "run1"}
    merged["component_acceptance"] = {
        "original_35_point_acceptance": True,
        "b_prefix_fix_acceptance": True,
        "bmixed_prefix_fix_acceptance": True,
    }
    merged["provenance"]["original_accuracy_report_sha256"] = sha256_file(
        FULL_ROOT / "accuracy_report.json"
    )
    merged["provenance"]["b_prefix_fix_acceptance_sha256"] = sha256_file(
        B_ROOT / "acceptance.json"
    )
    merged["provenance"]["bmixed_prefix_fix_acceptance_sha256"] = sha256_file(
        BMIXED_ROOT / "acceptance.json"
    )
    write_json(BMIXED_ROOT / "merged_accuracy_report.json", merged)

    sync_component_copies()

    components = {
        "original_full": component_status(FULL_ROOT, full_acceptance),
        "b_prefix_fix": component_status(B_ROOT, b_acceptance),
        "bmixed_prefix_fix": component_status(
            BMIXED_ROOT, bmixed_acceptance
        ),
    }
    final_report = read_json(FULL_ROOT / "final_accuracy_report.json")
    final_report.pop("determinism", None)
    final_report.pop("acceptance", None)
    final_report["schema"] = "dynosim_prefix_fixed_full_accuracy_report.v2"
    final_report["run_contract"] = {
        "run_count": 1,
        "report_source": "prefix_fixed_full/run1",
    }
    final_report["promotion"]["layout"] = {
        "pre_fix_runs": ["run1"],
        "prefix_fixed_runs": ["prefix_fixed_full/run1"],
    }
    final_report["promotion"]["scenario_sources"] = {
        "a": "original accepted run1",
        "b": "accepted B prefix-fix run1",
        "b_mixed": "accepted B-mixed prefix-fix run1",
        "c": "original accepted run1",
        "d": "original accepted run1",
    }
    final_report["promotion"]["component_acceptance"] = components
    final_report["provenance"]["original_accuracy_report_sha256"] = (
        sha256_file(FULL_ROOT / "accuracy_report.json")
    )
    final_report["provenance"]["b_prefix_fix_acceptance_sha256"] = (
        sha256_file(B_ROOT / "acceptance.json")
    )
    final_report["provenance"]["bmixed_prefix_fix_acceptance_sha256"] = (
        sha256_file(BMIXED_ROOT / "acceptance.json")
    )
    final_report["provenance"]["final_report_source_sha256"] = sha256_file(
        BMIXED_ROOT / "merged_accuracy_report.json"
    )
    validate_final_cells(final_report)

    final_acceptance = {
        "schema": "dynosim_prefix_fixed_full_acceptance.v2",
        "accepted": True,
        "checks": {
            "points_35_of_35": fixed_run["point_count"] == 35,
            "completed_matches_preserved_workload": (
                fixed_run["completed_requests"] == 3432
                and fixed_run["per_request_rows"] == 3432
            ),
            "single_run_contract": not any(
                path.exists() for path, _ in RUN2_TARGETS
            ),
            "component_acceptance": all(
                item["accepted"] for item in components.values()
            ),
            "aic_database_miss_free": True,
            "health_request_free": True,
            "nonnegative_finite_metrics": True,
            "tp4_ep4_mapping_valid": fixed_run["topology_valid"],
            "copied_source_hashes_exact": True,
        },
        "runs": {
            "run1": {
                key: fixed_run[key]
                for key in (
                    "point_count",
                    "completed_requests",
                    "per_request_rows",
                    "output_length_distribution",
                )
            }
        },
        "run_contract": {
            "run_count": 1,
            "report_source": "prefix_fixed_full/run1",
        },
        "execution_contract": {
            **final_report["promotion"]["data_contract"],
            "run_count": 1,
            "report_source": "prefix_fixed_full/run1",
        },
        "component_acceptance": components,
    }
    final_acceptance["accepted"] = all(
        final_acceptance["checks"].values()
    )
    if not final_acceptance["accepted"]:
        raise ValueError(
            f"final acceptance failed: {final_acceptance['checks']}"
        )
    final_report["acceptance"] = final_acceptance

    write_json(FIXED_FULL / "acceptance.json", final_acceptance)
    write_json(FIXED_FULL / "accuracy_report.json", final_report)
    final_markdown = promote.report_markdown(
        final_report, final_acceptance
    )
    (FIXED_FULL / "accuracy_report.md").write_text(
        final_markdown, encoding="utf-8"
    )
    write_json(FULL_ROOT / "final_acceptance.json", final_acceptance)
    write_json(FULL_ROOT / "final_accuracy_report.json", final_report)
    (FULL_ROOT / "final_accuracy_report.md").write_text(
        final_markdown, encoding="utf-8"
    )

    old_promotion = read_json(FIXED_FULL / "promotion_manifest.json")
    source_mapping = {
        key: value
        for key, value in old_promotion["source_mapping"].items()
        if key.startswith("run1/")
    }
    promotion_manifest = {
        "schema": "dynosim_prefix_fixed_full_promotion.v2",
        "result_root": str(FULL_ROOT),
        "output": promote.OUTPUT_NAME,
        "run_contract": {
            "run_count": 1,
            "report_source": "prefix_fixed_full/run1",
        },
        "scenario_sources": final_report["promotion"][
            "scenario_sources"
        ],
        "source_mapping": source_mapping,
        "deleted_run2": deleted_targets,
    }
    write_json(FIXED_FULL / "promotion_manifest.json", promotion_manifest)

    migration = {
        "schema": "dynosim_single_run_migration.v1",
        "run_contract": {
            "run_count": 1,
            "report_source": "prefix_fixed_full/run1",
        },
        "deleted_run2": deleted_targets,
        "deleted_determinism_files": [
            str(path) for path in DETERMINISM_FILES
        ],
        "final_report": str(FULL_ROOT / "final_accuracy_report.json"),
        "final_report_sha256": sha256_file(
            FULL_ROOT / "final_accuracy_report.json"
        ),
        "final_acceptance_sha256": sha256_file(
            FULL_ROOT / "final_acceptance.json"
        ),
    }
    write_json(FULL_ROOT / "single_run_migration.json", migration)

    print(
        json.dumps(
            {
                "accepted": True,
                "run_count": 1,
                "deleted_run2_count": len(deleted_targets),
                "deleted_bytes": sum(
                    item["byte_count"] for item in deleted_targets
                ),
                "final_report": migration["final_report"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
