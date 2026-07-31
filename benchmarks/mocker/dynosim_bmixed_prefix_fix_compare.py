#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay all seven formal B-mixed points with the SGLang prefix fix.

The accepted DynoSim run1 artifacts are immutable inputs. The script reuses
their canonical traces, deterministic RID maps, raw metrics, HiSim MAPE,
ingress envelopes, engine args, and source hashes. It performs two offline
aggregated SGLang replays, validates the eight-group prefix contract, compares
before/after accuracy, and emits a merged 175-cell conclusion that also
includes the already accepted B-prefix-fix results.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dynosim_b_prefix_fix_compare as base
import dynosim_hisim_alignment as common


BEFORE_ROOT = Path(
    "/workspace/hisim_align/"
    "dynosim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_"
    "h100pcie_gsp_mixed_seed11_full"
)
B_FIX_ROOT = Path(
    "/workspace/hisim_align/"
    "dynosim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_"
    "h100pcie_gsp_mixed_seed11_full_b_prefix_fix"
)
OUTPUT_ROOT = Path(
    "/workspace/hisim_align/"
    "dynosim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_"
    "h100pcie_gsp_mixed_seed11_full_bmixed_prefix_fix"
)
RPS_POINTS = (1, 2, 4, 8, 16, 24, 32)
POINT_NAMES = tuple(f"sweep_b_mixed_rps_point_{rps}" for rps in RPS_POINTS)
GROUP_COUNT = 8
REQUESTS_PER_GROUP = 12
PROMPT_TOKENS = 32768
SHARED_PREFIX_TOKENS = 29491
CHUNK_TOKENS = 8192
RID_PATTERN = re.compile(r"acc_bench_g(?P<group>\d{4})_o(?P<ordinal>\d{4})")


def _lcp(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (lhs, rhs) in enumerate(zip(left, right, strict=False)):
        if lhs != rhs:
            return index
    return min(len(left), len(right))


def _prefix_contract(point: base.PreservedPoint) -> dict[str, Any]:
    groups: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    best_previous: list[int] = []
    seen_by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(point.trace_rows):
        match = RID_PATTERN.fullmatch(str(row["request_id"]))
        if not match:
            raise ValueError(f"{point.name}: invalid B-mixed RID {row['request_id']!r}")
        group = int(match.group("group"))
        ordinal = int(match.group("ordinal"))
        groups[group].append((ordinal, row))
        previous = seen_by_group[group]
        best_previous.append(
            max(
                (_lcp(row["hash_ids"], item["hash_ids"]) for item in previous),
                default=0,
            )
        )
        previous.append(row)

    if set(groups) != set(range(GROUP_COUNT)):
        raise ValueError(f"{point.name}: expected groups 0..7, got {sorted(groups)}")
    for group, rows in groups.items():
        rows.sort(key=lambda item: item[0])
        if [ordinal for ordinal, _ in rows] != list(range(REQUESTS_PER_GROUP)):
            raise ValueError(f"{point.name}: group {group} ordinal/count mismatch")
        leader = rows[0][1]["hash_ids"]
        within = [_lcp(leader, row["hash_ids"]) for _, row in rows[1:]]
        if set(within) != {SHARED_PREFIX_TOKENS}:
            raise ValueError(f"{point.name}: group {group} LCP mismatch {set(within)}")

    leaders = [groups[group][0][1]["hash_ids"] for group in range(GROUP_COUNT)]
    cross = [
        _lcp(leaders[left], leaders[right])
        for left in range(GROUP_COUNT)
        for right in range(left + 1, GROUP_COUNT)
    ]
    if set(cross) != {0}:
        raise ValueError(f"{point.name}: cross-group LCP mismatch {set(cross)}")
    expected_best = [0] * GROUP_COUNT + [SHARED_PREFIX_TOKENS] * (
        point.request_count - GROUP_COUNT
    )
    if sorted(best_previous) != sorted(expected_best):
        raise ValueError(f"{point.name}: best-previous prefix distribution mismatch")

    total_input = sum(int(row["input_length"]) for row in point.trace_rows)
    expected_ratio = sum(best_previous) / total_input
    old_chunk_limited_ratio = (
        (point.request_count - GROUP_COUNT) * CHUNK_TOKENS / total_input
    )
    before_report = json.loads(
        (point.before_dir / "dynosim_report.json").read_text()
    )
    observed_before = float(
        before_report["first_admission_prefix_cache_reused_ratio"]
    )
    if not math.isclose(
        observed_before, old_chunk_limited_ratio, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"{point.name}: old first-admission ratio {observed_before} "
            f"!= chunk-limited {old_chunk_limited_ratio}"
        )
    return {
        "group_count": GROUP_COUNT,
        "requests_per_group": REQUESTS_PER_GROUP,
        "prompt_tokens": PROMPT_TOKENS,
        "shared_prefix_tokens": SHARED_PREFIX_TOKENS,
        "unique_suffix_tokens": PROMPT_TOKENS - SHARED_PREFIX_TOKENS,
        "within_group_lcp_tokens": SHARED_PREFIX_TOKENS,
        "cross_group_lcp_tokens": 0,
        "group_leader_requests": GROUP_COUNT,
        "warm_requests": point.request_count - GROUP_COUNT,
        "expected_first_admission_reuse_ratio": expected_ratio,
        "before_first_admission_reuse_ratio": observed_before,
        "old_chunk_limited_reuse_ratio": old_chunk_limited_ratio,
        "missing_reused_tokens_before_fix": (
            point.request_count - GROUP_COUNT
        )
        * (SHARED_PREFIX_TOKENS - CHUNK_TOKENS),
    }


def preflight(
    before_root: Path,
) -> tuple[list[base.PreservedPoint], dict[str, dict[str, Any]], dict[str, Any]]:
    base.B_POINT_NAMES = POINT_NAMES
    points, manifest = base.preflight(before_root)
    contracts = {point.name: _prefix_contract(point) for point in points}
    manifest["schema"] = "dynosim_bmixed_prefix_fix_preflight.v1"
    manifest["scenario"] = "b_mixed"
    manifest["expected_points"] = list(POINT_NAMES)
    manifest["prefix_contracts"] = contracts
    for point in points:
        manifest["points"][point.name]["prefix_contract"] = contracts[point.name]
    return points, contracts, manifest


def _prefix_acceptance(
    points: Sequence[base.PreservedPoint],
    contracts: dict[str, dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    results = {}
    all_ratios = True
    all_request_reuse = True
    for run_name in ("run1",):
        results[run_name] = {}
        for point in points:
            point_dir = output_root / run_name / point.name
            report = json.loads((point_dir / "dynosim_report.json").read_text())
            rows = common.load_jsonl(point_dir / "bench_per_request.jsonl")
            expected_ratio = contracts[point.name][
                "expected_first_admission_reuse_ratio"
            ]
            observed_ratio = float(
                report["first_admission_prefix_cache_reused_ratio"]
            )
            ratio_ok = math.isclose(
                observed_ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-12
            )
            bad_rows = []
            for row in rows:
                match = RID_PATTERN.fullmatch(str(row["source_rid"]))
                if not match:
                    bad_rows.append(row["source_rid"])
                    continue
                expected = (
                    0
                    if int(match.group("ordinal")) == 0
                    else SHARED_PREFIX_TOKENS
                )
                if row["reused_input_tokens"] != expected:
                    bad_rows.append(row["source_rid"])
            all_ratios &= ratio_ok
            all_request_reuse &= not bad_rows
            results[run_name][point.name] = {
                "expected_first_admission_reuse_ratio": expected_ratio,
                "observed_first_admission_reuse_ratio": observed_ratio,
                "ratio_matches": ratio_ok,
                "request_reuse_matches": not bad_rows,
                "bad_request_count": len(bad_rows),
                "expected_reuse_distribution": {
                    "0": GROUP_COUNT,
                    str(SHARED_PREFIX_TOKENS): point.request_count - GROUP_COUNT,
                },
            }
    audit = {
        "schema": "dynosim_bmixed_prefix_fix_prefix_acceptance.v1",
        "all_first_admission_ratios_match": all_ratios,
        "all_request_reuse_matches": all_request_reuse,
        "runs": results,
    }
    if not all_ratios or not all_request_reuse:
        raise ValueError(f"B-mixed prefix acceptance failed: {audit}")
    return audit


def _comparison(
    points: Sequence[base.PreservedPoint],
    contracts: dict[str, dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    per_point: dict[str, Any] = {}
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
            }
            cells.append(cell)
            point_cells.append(cell)
        per_point[point.name] = {
            "request_count": point.request_count,
            "before_mean_dynosim_mape_percent": statistics.fmean(
                item["before_dynosim_mape_percent"] for item in point_cells
            ),
            "after_mean_dynosim_mape_percent": statistics.fmean(
                item["after_dynosim_mape_percent"] for item in point_cells
            ),
            "hisim_mean_mape_percent": statistics.fmean(
                item["hisim_mape_percent"] for item in point_cells
            ),
            "mean_mape_improvement_percentage_points": statistics.fmean(
                item["mape_improvement_percentage_points"] for item in point_cells
            ),
        }

    per_metric = {}
    for metric in common.PRIMARY_METRICS:
        selected = [item for item in cells if item["metric"] == metric]
        per_metric[metric] = {
            "point_count": len(selected),
            "before_mean_dynosim_mape_percent": statistics.fmean(
                item["before_dynosim_mape_percent"] for item in selected
            ),
            "after_mean_dynosim_mape_percent": statistics.fmean(
                item["after_dynosim_mape_percent"] for item in selected
            ),
            "hisim_mean_mape_percent": statistics.fmean(
                item["hisim_mape_percent"] for item in selected
            ),
            "mean_mape_improvement_percentage_points": statistics.fmean(
                item["mape_improvement_percentage_points"] for item in selected
            ),
        }

    max_point = next(
        point for point in points if point.name == "sweep_b_mixed_rps_point_2"
    )
    before_diag = base._point_diagnostics(max_point.before_dir)
    after_diag = base._point_diagnostics(output_root / "run1" / max_point.name)
    queue_drop = before_diag["mean_queue_ms"] - after_diag["mean_queue_ms"]
    scheduler_ttft_drop = (
        before_diag["mean_scheduler_ttft_ms"]
        - after_diag["mean_scheduler_ttft_ms"]
    )
    return {
        "schema": "dynosim_bmixed_prefix_fix_accuracy_report.v1",
        "scope": {
            "scenario": "b_mixed",
            "rps_points": list(RPS_POINTS),
            "point_count": len(points),
            "metric_count": len(common.PRIMARY_METRICS),
            "cell_count": len(cells),
            "single_batch_excluded": True,
            "plain_b_excluded": True,
        },
        "overall": {
            "before_mean_dynosim_mape_percent": statistics.fmean(
                item["before_dynosim_mape_percent"] for item in cells
            ),
            "after_mean_dynosim_mape_percent": statistics.fmean(
                item["after_dynosim_mape_percent"] for item in cells
            ),
            "hisim_mean_mape_percent": statistics.fmean(
                item["hisim_mape_percent"] for item in cells
            ),
            "mean_mape_improvement_percentage_points": statistics.fmean(
                item["mape_improvement_percentage_points"] for item in cells
            ),
            "improved_cells": sum(
                item["mape_improvement_percentage_points"] > 0 for item in cells
            ),
            "regressed_cells": sum(
                item["mape_improvement_percentage_points"] < 0 for item in cells
            ),
            "after_dynosim_beats_hisim_cells": sum(
                item["after_dynosim_mape_percent"] < item["hisim_mape_percent"]
                for item in cells
            ),
        },
        "largest_before_cell": max(
            cells, key=lambda item: item["before_dynosim_mape_percent"]
        ),
        "largest_after_cell": max(
            cells, key=lambda item: item["after_dynosim_mape_percent"]
        ),
        "largest_improvement_cell": max(
            cells, key=lambda item: item["mape_improvement_percentage_points"]
        ),
        "per_metric": per_metric,
        "per_point": per_point,
        "bmixed2_root_cause": {
            "prefix_contract": contracts[max_point.name],
            "before": before_diag,
            "after": after_diag,
            "mean_queue_reduction_ms": queue_drop,
            "queue_share_of_scheduler_ttft_reduction_percent": (
                queue_drop / scheduler_ttft_drop * 100.0
            ),
            "finding": (
                "The old first admission could discover only the 8192-token "
                "chunk, not the 29491-token group prefix. It overcharged "
                "21299 fresh tokens on each of 88 warm requests and moved "
                "the RPS=2 point into a long FCFS queue. Full-sequence radix "
                "matching restores the exact eight-leader/eighty-eight-warm "
                "reuse distribution."
            ),
        },
        "per_point_prefix_contract": contracts,
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


def _scenario(point: str) -> str:
    if point.startswith("sweep_b_mixed_"):
        return "b_mixed"
    match = re.match(r"sweep_([abcd])_", point)
    if not match:
        raise ValueError(f"unknown point scenario: {point}")
    return match.group(1)


def _merged_accuracy(output_root: Path) -> dict[str, Any]:
    original_path = BEFORE_ROOT / "accuracy_report.json"
    original = json.loads(original_path.read_text())
    original_acceptance = json.loads((BEFORE_ROOT / "acceptance.json").read_text())
    b_acceptance = json.loads((B_FIX_ROOT / "acceptance.json").read_text())
    bmixed_acceptance = json.loads((output_root / "acceptance.json").read_text())
    cells = []
    for original_cell in original["cells"]:
        point = original_cell["point"]
        metric = original_cell["metric"]
        replacement_path = None
        if point.startswith("sweep_b_rps_point_"):
            replacement_path = B_FIX_ROOT / "run1" / point / "mape.json"
        elif point.startswith("sweep_b_mixed_rps_point_"):
            replacement_path = output_root / "run1" / point / "mape.json"
        cell = dict(original_cell)
        cell["original_dynosim"] = original_cell["dynosim"]
        cell["original_dynosim_mape_percent"] = original_cell[
            "dynosim_mape_percent"
        ]
        cell["replacement_source"] = "original_accepted_run1"
        if replacement_path is not None:
            replacement = json.loads(replacement_path.read_text())["metrics"][metric]
            if not math.isclose(
                float(replacement["raw"]),
                float(original_cell["raw"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ) or not math.isclose(
                float(replacement["hisim_mape_percent"]),
                float(original_cell["hisim_mape_percent"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{point}/{metric}: replacement baseline drift")
            cell.update(
                {
                    "dynosim": replacement["dynosim"],
                    "dynosim_mape_percent": replacement[
                        "dynosim_mape_percent"
                    ],
                    "delta_percentage_points": replacement[
                        "delta_percentage_points"
                    ],
                    "winner": replacement["winner"],
                    "replacement_source": (
                        "accepted_b_prefix_fix_run1"
                        if point.startswith("sweep_b_rps_point_")
                        else "accepted_bmixed_prefix_fix_run1"
                    ),
                }
            )
        cells.append(cell)
    if len(cells) != 175:
        raise ValueError(f"merged report expected 175 cells, got {len(cells)}")

    per_scenario = {}
    for scenario in ("a", "b", "b_mixed", "c", "d"):
        selected = [item for item in cells if _scenario(item["point"]) == scenario]
        per_scenario[scenario] = {
            "cell_count": len(selected),
            "original_dynosim_mean_mape_percent": statistics.fmean(
                item["original_dynosim_mape_percent"] for item in selected
            ),
            "updated_dynosim_mean_mape_percent": statistics.fmean(
                item["dynosim_mape_percent"] for item in selected
            ),
            "hisim_mean_mape_percent": statistics.fmean(
                item["hisim_mape_percent"] for item in selected
            ),
        }
    per_metric = {}
    for metric in common.PRIMARY_METRICS:
        selected = [item for item in cells if item["metric"] == metric]
        per_metric[metric] = {
            "cell_count": len(selected),
            "original_dynosim_mean_mape_percent": statistics.fmean(
                item["original_dynosim_mape_percent"] for item in selected
            ),
            "updated_dynosim_mean_mape_percent": statistics.fmean(
                item["dynosim_mape_percent"] for item in selected
            ),
            "hisim_mean_mape_percent": statistics.fmean(
                item["hisim_mape_percent"] for item in selected
            ),
        }
    component_acceptance = {
        "original_35_point_acceptance": bool(original_acceptance.get("accepted")),
        "b_prefix_fix_acceptance": bool(b_acceptance.get("accepted")),
        "bmixed_prefix_fix_acceptance": bool(bmixed_acceptance.get("accepted")),
    }
    if not all(component_acceptance.values()):
        raise ValueError(f"merged component acceptance failed: {component_acceptance}")
    return {
        "schema": "dynosim_prefix_fix_merged_accuracy_report.v1",
        "scope": {
            "point_count": 35,
            "metric_count": 5,
            "cell_count": 175,
            "single_batch_excluded": True,
        },
        "overall": {
            "original_dynosim_mean_mape_percent": statistics.fmean(
                item["original_dynosim_mape_percent"] for item in cells
            ),
            "updated_dynosim_mean_mape_percent": statistics.fmean(
                item["dynosim_mape_percent"] for item in cells
            ),
            "hisim_mean_mape_percent": statistics.fmean(
                item["hisim_mape_percent"] for item in cells
            ),
            "dynosim_improvement_percentage_points": statistics.fmean(
                item["original_dynosim_mape_percent"]
                - item["dynosim_mape_percent"]
                for item in cells
            ),
            "dynosim_wins": sum(item["winner"] == "dynosim" for item in cells),
            "hisim_wins": sum(item["winner"] == "hisim" for item in cells),
            "ties": sum(item["winner"] == "tie" for item in cells),
        },
        "largest_original_cell": max(
            cells, key=lambda item: item["original_dynosim_mape_percent"]
        ),
        "largest_updated_cell": max(
            cells, key=lambda item: item["dynosim_mape_percent"]
        ),
        "per_scenario": per_scenario,
        "per_metric": per_metric,
        "component_acceptance": component_acceptance,
        "provenance": {
            "original_accuracy_report": str(original_path),
            "original_accuracy_report_sha256": common.sha256_file(original_path),
            "b_prefix_fix_root": str(B_FIX_ROOT),
            "b_prefix_fix_acceptance_sha256": common.sha256_file(
                B_FIX_ROOT / "acceptance.json"
            ),
            "bmixed_prefix_fix_root": str(output_root),
            "bmixed_prefix_fix_acceptance_sha256": common.sha256_file(
                output_root / "acceptance.json"
            ),
            "unchanged_scenarios": ["a", "c", "d"],
            "replaced_scenarios": ["b", "b_mixed"],
        },
        "cells": cells,
    }


def _markdown(report: dict[str, Any], acceptance: dict[str, Any]) -> str:
    overall = report["overall"]
    cause = report["bmixed2_root_cause"]
    before = cause["before"]
    after = cause["after"]
    largest = report["largest_after_cell"]
    lines = [
        "# DynoSim SGLang B-mixed prefix 修复正式重跑",
        "",
        "## 正式结论",
        "",
        (
            f"- 7 个正式 RPS 点、35 个 point-metric cell：DynoSim 平均 MAPE "
            f"{overall['before_mean_dynosim_mape_percent']:.6f}% → "
            f"{overall['after_mean_dynosim_mape_percent']:.6f}%；HiSim 为 "
            f"{overall['hisim_mean_mape_percent']:.6f}%。"
        ),
        (
            f"- 修复后最大 B-mixed cell 为 `{largest['point']}` / "
            f"`{largest['metric']}`：{largest['after_dynosim_mape_percent']:.6f}%。"
        ),
        (
            f"- DynoSim 修复后在 {overall['after_dynosim_beats_hisim_cells']}/35 "
            "个 B-mixed cell 优于 HiSim。"
        ),
        "",
        "## B-mixed-2 根因闭环",
        "",
        (
            f"- 首次 admission prefix reuse："
            f"{before['first_admission_prefix_cache_reused_ratio']:.6f} → "
            f"{after['first_admission_prefix_cache_reused_ratio']:.6f}；"
            "对应 8 个 group leader 不复用、88 个 warm 请求各复用 29491 token。"
        ),
        (
            f"- 平均 scheduler queue：{before['mean_queue_ms']:.3f} ms → "
            f"{after['mean_queue_ms']:.3f} ms；正式 TTFT（同一 ingress）："
            f"{before['mean_ttft_with_ingress_ms']:.3f} ms → "
            f"{after['mean_ttft_with_ingress_ms']:.3f} ms。"
        ),
        (
            f"- TTFT 降幅中 {cause['queue_share_of_scheduler_ttft_reduction_percent']:.3f}% "
            "来自 queue 消失，证明最大 TTFT 异常由完整 prefix 被 8192 chunk 截断引起。"
        ),
        "",
        "## 各点五指标平均 MAPE",
        "",
        "| 点位 | 修复前 DynoSim | 修复后 DynoSim | HiSim | 改善百分点 |",
        "|---|---:|---:|---:|---:|",
    ]
    for point in POINT_NAMES:
        item = report["per_point"][point]
        lines.append(
            f"| `{point}` | {item['before_mean_dynosim_mape_percent']:.6f}% | "
            f"{item['after_mean_dynosim_mape_percent']:.6f}% | "
            f"{item['hisim_mean_mape_percent']:.6f}% | "
            f"{item['mean_mape_improvement_percentage_points']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## 验收边界",
            "",
            "- 只重放 `sweep_b_mixed_rps_point_{1,2,4,8,16,24,32}`，不含 single-batch。",
            "- 未启动 SGLang、未重新采集实机数据；复用 accepted run1 的 trace、RID、raw、HiSim ACC、ingress 及 SHA256。",
            "- DynoSim 是 offline aggregated SGLang scheduler + AIC balanced timing，不声称执行真实 fake TopK 路由。",
            (
                f"- `accepted={str(acceptance['accepted']).lower()}`：7/7 点、"
                "672/672 请求每轮、无 NaN/负延迟/AIC miss、TP4/EP4 合法、"
                "preserved 输入未改；正式 ACC 只使用 run1。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _merged_markdown(report: dict[str, Any]) -> str:
    overall = report["overall"]
    largest = report["largest_updated_cell"]
    lines = [
        "# DynoSim prefix 修复后的 35 点正式结论",
        "",
        (
            f"- 175 个正式 cell 的 DynoSim 总平均 MAPE："
            f"{overall['original_dynosim_mean_mape_percent']:.6f}% → "
            f"{overall['updated_dynosim_mean_mape_percent']:.6f}%；"
            f"HiSim 为 {overall['hisim_mean_mape_percent']:.6f}%。"
        ),
        (
            f"- 更新后胜负：DynoSim {overall['dynosim_wins']}，"
            f"HiSim {overall['hisim_wins']}，tie {overall['ties']}。"
        ),
        (
            f"- 更新后的全局最大 cell：`{largest['point']}` / "
            f"`{largest['metric']}` = {largest['dynosim_mape_percent']:.6f}%。"
        ),
        "",
        "## 分场景平均 MAPE",
        "",
        "| 场景 | 原 DynoSim | 更新后 DynoSim | HiSim |",
        "|---|---:|---:|---:|",
    ]
    for scenario in ("a", "b", "b_mixed", "c", "d"):
        item = report["per_scenario"][scenario]
        lines.append(
            f"| `{scenario}` | {item['original_dynosim_mean_mape_percent']:.6f}% | "
            f"{item['updated_dynosim_mean_mape_percent']:.6f}% | "
            f"{item['hisim_mean_mape_percent']:.6f}% |"
        )
    lines.extend(
        [
            "",
            "A/C/D 使用原 accepted run1；B 使用已验收的 B-prefix-fix run1；"
            "B-mixed 使用本次 run1。所有替换均保持原 raw 与 HiSim MAPE 不变。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", type=Path, default=BEFORE_ROOT)
    parser.add_argument("--b-fix-root", type=Path, default=B_FIX_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    global BEFORE_ROOT, B_FIX_ROOT
    BEFORE_ROOT = args.before_root
    B_FIX_ROOT = args.b_fix_root
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output root: {args.output_root}"
        )
    if not base.PYTHON.is_file():
        raise FileNotFoundError(base.PYTHON)
    base.B_POINT_NAMES = POINT_NAMES
    base.ARTIFACT_SCOPE = "bmixed_prefix_fix"
    base.ARTIFACT_SCHEMA_VERSION = 2
    points, contracts, preflight_manifest = preflight(args.before_root)
    args.output_root.mkdir(parents=True, exist_ok=False)
    common.write_json(args.output_root / "preflight.json", preflight_manifest)
    revisions = {
        "dynamo": common.git_revision(base.DYNAMO_ROOT),
        "aiconfigurator": common.git_revision(base.AIC_ROOT),
        "aiconfigurator_base_pin": "ffbab15797d3a3f14c382e937c0ab0c1c5cbc530",
        "prefix_fix_commit": "742eec4bc47b8ddf8ceeb744dadb828003b19c53",
    }
    common.write_json(args.output_root / "revisions.json", revisions)

    base.run_all_points(points, args.output_root / "run1", revisions)
    preserved_hashes_after = {
        point.name: base._preserved_hashes(point.before_dir) for point in points
    }
    acceptance = base.acceptance_audit(
        points, args.output_root, preserved_hashes_after
    )
    prefix_acceptance = _prefix_acceptance(points, contracts, args.output_root)
    acceptance["checks"]["bmixed_prefix_contract_exact"] = True
    acceptance["prefix_acceptance"] = prefix_acceptance
    common.write_json(args.output_root / "prefix_acceptance.json", prefix_acceptance)
    common.write_json(args.output_root / "acceptance.json", acceptance)

    report = _comparison(points, contracts, args.output_root)
    report["run_contract"] = {"run_count": 1, "report_source": "run1"}
    report["acceptance"] = acceptance
    common.write_json(args.output_root / "before_after_mape.json", report)
    (args.output_root / "before_after_mape.md").write_text(
        _markdown(report, acceptance), encoding="utf-8"
    )

    merged = _merged_accuracy(args.output_root)
    common.write_json(args.output_root / "merged_accuracy_report.json", merged)
    (args.output_root / "merged_accuracy_report.md").write_text(
        _merged_markdown(merged), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "accepted": acceptance["accepted"],
                "bmixed_before_mean_mape_percent": report["overall"][
                    "before_mean_dynosim_mape_percent"
                ],
                "bmixed_after_mean_mape_percent": report["overall"][
                    "after_mean_dynosim_mape_percent"
                ],
                "merged_updated_mean_mape_percent": merged["overall"][
                    "updated_dynosim_mean_mape_percent"
                ],
                "merged_largest_updated_cell": merged["largest_updated_cell"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
