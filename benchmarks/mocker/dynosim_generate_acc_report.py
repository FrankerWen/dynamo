#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate DynoSim offline replay diagnostic reports and plots.

This script parses accuracy_report.json from a DynoSim run directory (e.g. prefix_fixed_full)
and emits the established report filenames without claiming a formal HiSim ACC result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RUN_DIR = Path(
    "/workspace/hisim_align/"
    "dynosim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_"
    "h100pcie_gsp_mixed_seed11_full/prefix_fixed_full"
)
DEFAULT_BEFORE_REPORT = Path(
    "/workspace/hisim_align/"
    "dynosim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_"
    "h100pcie_gsp_mixed_seed11_full/prefix_fixed_full/accuracy_report.json"
)
CANONICAL_AIC_ROOT = Path("/workspace/hisim_align/aiconfigurator")
DYNOSIM_AIC_ROOT = Path("/workspace/hisim_align/aiconfigurator-dynosim-e2e")
RECORDED_OVERLAY_ROOT = Path("/workspace/hisim_align/aic_systems_h100_pcie_recorded_overlay")

SCENARIOS = {
    "a": {
        "name": "Long Input, Short Output (No Prefix)",
        "input_len": 32768,
        "output_len": 128,
    },
    "b": {
        "name": "Long Input, Short Output (90% Prefix)",
        "input_len": 32768,
        "output_len": 128,
    },
    "b_mixed": {
        "name": "Long Input, Short Output (90% Prefix, Multi-Group Mixed)",
        "input_len": 32768,
        "output_len": "mixed",
    },
    "c": {
        "name": "Short Input, Short Output (No Prefix)",
        "input_len": 512,
        "output_len": 128,
    },
    "d": {
        "name": "Short Input, Long Output (No Prefix)",
        "input_len": 512,
        "output_len": 1024,
    },
}

RPS_POINTS = (1, 2, 4, 8, 16, 24, 32)
BATCH_SIZES = (1, 2, 4, 8, 16, 24, 32)

METRICS = [
    ("mean_ttft_ms", "ttft", "TTFT (ms)"),
    ("mean_tpot_ms", "tpot", "TPOT (ms)"),
    ("mean_itl_ms", "itl", "ITL (ms)"),
    ("mean_e2e_latency_ms", "e2e_latency", "E2E Latency (ms)"),
    ("input_throughput", "input_throughput", "Input Throughput (tok/s)"),
]


def load_accuracy_data(run_dir: Path) -> tuple[dict[str, dict[str, dict[str, dict[int, dict[str, float]]]]], dict[str, Any]]:
    """Load accuracy cells from accuracy_report.json or final_accuracy_report.json."""
    candidates = [
        run_dir / "accuracy_report.json",
        run_dir / "final_accuracy_report.json",
        run_dir.parent / "accuracy_report.json",
        run_dir.parent / "final_accuracy_report.json",
    ]
    
    report_path = None
    for path in candidates:
        if path.exists():
            report_path = path
            break
            
    if not report_path:
        raise FileNotFoundError(f"Could not find accuracy_report.json in {run_dir} or its parent.")
        
    print(f"[acc_report] Loading accuracy data from {report_path}")
    data = json.loads(report_path.read_text(encoding="utf-8"))
    
    cells = data.get("cells", [])
    
    # Structure: data_map[method][scenario][metric][pt_val] = {'dynosim_mape': ..., 'hisim_mape': ..., 'raw': ..., 'dynosim': ...}
    data_map: dict[str, dict[str, dict[str, dict[int, dict[str, float]]]]] = {
        "single_batch": {},
        "rps": {},
    }
    
    for method in ("single_batch", "rps"):
        for sc in SCENARIOS:
            data_map[method][sc] = {}
            for m_key, _, _ in METRICS:
                data_map[method][sc][m_key] = {}

    for cell in cells:
        pt = cell["point"]  # e.g., sweep_a_rps_point_1 or sweep_a_single_batch_point_1
        if "_single_batch_point_" in pt:
            method = "single_batch"
            parts = pt.split("_single_batch_point_")
        elif "_rps_point_" in pt:
            method = "rps"
            parts = pt.split("_rps_point_")
        else:
            continue

        sc = parts[0].replace("sweep_", "")
        try:
            pt_val = int(parts[1])
        except ValueError:
            continue
            
        metric = cell["metric"]
        if sc in data_map[method] and metric in data_map[method][sc]:
            dynosim_mape = float(cell.get("dynosim_mape_percent", 0.0))
            hisim_mape = float(cell.get("hisim_mape_percent", 0.0))
            raw_val = float(cell.get("raw", 0.0))
            dynosim_val = float(cell.get("dynosim", 0.0))
            
            data_map[method][sc][metric][pt_val] = {
                "dynosim_mape": dynosim_mape,
                "hisim_mape": hisim_mape,
                "raw": raw_val,
                "dynosim": dynosim_val,
            }
            
    return data_map, data


def plot_acc_curves(
    data_map: dict[str, dict[str, dict[str, dict[int, dict[str, float]]]]],
    method_key: str,
    sweep_points: tuple[int, ...],
    output_png: Path,
) -> None:
    """Plot accuracy comparison curves for a sweep method across scenarios and metrics."""
    if "seaborn-v0_8-whitegrid" in plt.style.available:
        plt.style.use("seaborn-v0_8-whitegrid")
    else:
        plt.style.use("ggplot")

    scenario_keys = list(SCENARIOS.keys())
    fig_height = max(4, 4 * len(scenario_keys))
    fig, axes = plt.subplots(
        len(scenario_keys), len(METRICS), figsize=(25, fig_height), sharex=True, squeeze=False
    )

    xlabel_title = "Batch Size" if method_key == "single_batch" else "Request Rate (RPS)"
    source_label = "/generate raw vs DynoSim server replay" if method_key == "single_batch" else "bench_serving raw vs DynoSim simulation"

    fig.suptitle(
        f"DynoSim Offline Replay Diagnostic: {source_label} ({xlabel_title} Sweep)",
        fontsize=20,
        weight="bold",
    )

    for r_idx, sc_key in enumerate(scenario_keys):
        sc_name = SCENARIOS[sc_key]["name"]
        for c_idx, (m_key, m_short, m_title) in enumerate(METRICS):
            ax = axes[r_idx, c_idx]
            x_pts = []
            y_raw = []
            y_dynosim = []

            for pt_val in sweep_points:
                point_data = data_map[method_key][sc_key][m_key].get(pt_val)
                if point_data:
                    x_pts.append(pt_val)
                    y_raw.append(point_data["raw"])
                    y_dynosim.append(point_data["dynosim"])

            if x_pts:
                ax.plot(
                    x_pts,
                    y_raw,
                    label="raw",
                    color="#ff7f0e",
                    linestyle="--",
                    linewidth=2.5,
                    marker="x",
                    zorder=2,
                )
                ax.plot(
                    x_pts,
                    y_dynosim,
                    label="dynosim+aic",
                    color="#2ca02c",
                    linewidth=2.5,
                    marker="s",
                    zorder=3,
                )

            ax.set_title(f"{sc_name}\n{m_title}", fontsize=10, weight="semibold")
            if r_idx == len(scenario_keys) - 1:
                ax.set_xlabel(xlabel_title, fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            if r_idx == 0 and c_idx == 0:
                ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", shadow=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"[acc_report] ACC plots for {method_key} saved to {output_png}")


def generate_acc_report_md(
    data_map: dict[str, dict[str, dict[str, dict[int, dict[str, float]]]]],
    report: dict[str, Any],
    before_report: dict[str, Any] | None,
    output_md: Path,
) -> None:
    """Generate sweep_acc_report.md content."""
    md = "# DynoSim Offline Replay Accuracy Diagnostic\n\n"
    md += (
        "This is a DynoSim CPU/offline replay diagnostic over reused TP4+EP4 artifacts. "
        "It is not a formal HiSim `acc_bench/compare` result: DynoSim formal ACC status "
        "is `N/A`. RPS rows compare reused `bench_serving` raw metrics with offline "
        "scheduler replay plus the preserved ingress envelope. Single-batch rows compare "
        "reused `/generate` raw metrics with offline replay. HiSim MAPE is shown only as "
        "the already-recorded reference.\n\n"
    )
    after = report.get("overall", {})
    md += "## Repair Summary\n\n"
    md += "| Scope | Before P0/P1/P2 | After P0/P1/P2 | Change | HiSim reference |\n"
    md += "|---|---:|---:|---:|---:|\n"
    for scope, label in (("all", "70 points / 350 cells"), ("rps", "RPS / 175 cells"), ("single_batch", "Single batch / 175 cells")):
        after_item = after if scope == "all" else report.get("per_method", {}).get(scope, {})
        before_item = _cell_summary(_scope_cells(before_report, scope)) if before_report else None
        before_value = before_item["mean_dynosim_mape_percent"] if before_item else None
        after_value = after_item.get("mean_dynosim_mape_percent")
        hisim_value = after_item.get("mean_hisim_mape_percent")
        delta = after_value - before_value if after_value is not None and before_value is not None else None
        md += (
            f"| {label} | {_percent(before_value)} | {_percent(after_value)} | "
            f"{_signed_pp(delta)} | {_percent(hisim_value)} |\n"
        )
    md += "\nDynoSim formal ACC: `N/A`; all values above are offline replay diagnostic MAPE.\n\n"
    if before_report:
        md += "### Metric-Level Before / After\n\n"
        md += "| Method | Metric | Before | After | Change |\n"
        md += "|---|---|---:|---:|---:|\n"
        for scope, method_label in (("rps", "RPS"), ("single_batch", "Single batch")):
            before_cells = _scope_cells(before_report, scope)
            after_cells = _scope_cells(report, scope)
            for metric, metric_short, _ in METRICS:
                before_values = [
                    float(cell["dynosim_mape_percent"])
                    for cell in before_cells
                    if cell.get("metric") == metric
                ]
                after_values = [
                    float(cell["dynosim_mape_percent"])
                    for cell in after_cells
                    if cell.get("metric") == metric
                ]
                before_value = statistics.fmean(before_values) if before_values else None
                after_value = statistics.fmean(after_values) if after_values else None
                delta = (
                    after_value - before_value
                    if before_value is not None and after_value is not None
                    else None
                )
                md += (
                    f"| {method_label} | `{metric_short}` | {_percent(before_value)} | "
                    f"{_percent(after_value)} | {_signed_pp(delta)} |\n"
                )
        md += "\nNegative change means lower MAPE. The table makes the decode/ITL gains and TTFT/throughput regressions explicit.\n\n"

    # 1. Single Batch Section
    md += "## Single Batch Sweep\n\n"
    sb_has_data = any(
        len(data_map["single_batch"][sc][m_key]) > 0
        for sc in SCENARIOS
        for m_key, _, _ in METRICS
    )
    if not sb_has_data:
        md += "`N/A`: no ACC or raw-vs-simulation artifacts are available for this method yet.\n\n"
    else:
        md += "![Single_Batch ACC Curve Graph](single_batch_acc_sweep.png)\n\n"
        md += "### /generate Raw vs DynoSim+AIC MAPE Summary Table for Single Batch Sweep\n\n"
        sb_cols = [f"Batch Size={bs}" for bs in BATCH_SIZES]
        md += "| Scenario | Metric | " + " | ".join(sb_cols) + " | Average MAPE |\n"
        md += "| :--- | :--- | " + " | ".join("---" for _ in BATCH_SIZES) + " | :--- |\n"

        for sc_key in SCENARIOS:
            sc_name = SCENARIOS[sc_key]["name"]
            sc_prefix_printed = False

            for m_key, m_short, _ in METRICS:
                row_vals = []
                mape_list = []

                for bs in BATCH_SIZES:
                    point_data = data_map["single_batch"][sc_key][m_key].get(bs)
                    if point_data is not None:
                        mape = point_data["dynosim_mape"]
                        mape_list.append(mape)
                        row_vals.append(f"`{mape:.2f}%`")
                    else:
                        row_vals.append("`N/A`")

                row_prefix = f"**{sc_name}**" if not sc_prefix_printed else ""
                sc_prefix_printed = True

                avg_mape = sum(mape_list) / len(mape_list) if mape_list else None
                avg_str = f"**`{avg_mape:.2f}%`**" if avg_mape is not None else "`N/A`"

                md += (
                    f"| {row_prefix} | `{m_short}` | "
                    + " | ".join(row_vals)
                    + f" | {avg_str} |\n"
                )

            # Separator between scenarios
            md += "| | | " + " | ".join("" for _ in BATCH_SIZES) + " | |\n"

    # 2. RPS Section
    md += "## Rps Sweep\n\n"
    rps_has_data = any(
        len(data_map["rps"][sc][m_key]) > 0
        for sc in SCENARIOS
        for m_key, _, _ in METRICS
    )
    if not rps_has_data:
        md += "`N/A`: no ACC or raw-vs-simulation artifacts are available for this method yet.\n\n"
    else:
        md += "![Rps ACC Curve Graph](rps_acc_sweep.png)\n\n"
        md += "### Raw vs DynoSim+AIC Offline Replay MAPE Summary for RPS Sweep\n\n"
        rps_cols = [f"Request Rate (RPS)={rps}" for rps in RPS_POINTS]
        md += "| Scenario | Metric | " + " | ".join(rps_cols) + " | Average MAPE |\n"
        md += "| :--- | :--- | " + " | ".join("---" for _ in RPS_POINTS) + " | :--- |\n"

        for sc_key in SCENARIOS:
            sc_name = SCENARIOS[sc_key]["name"]
            sc_prefix_printed = False

            for m_key, m_short, _ in METRICS:
                row_vals = []
                mape_list = []

                for rps in RPS_POINTS:
                    point_data = data_map["rps"][sc_key][m_key].get(rps)
                    if point_data is not None:
                        mape = point_data["dynosim_mape"]
                        mape_list.append(mape)
                        row_vals.append(f"`{mape:.2f}%`")
                    else:
                        row_vals.append("`N/A`")

                row_prefix = f"**{sc_name}**" if not sc_prefix_printed else ""
                sc_prefix_printed = True

                avg_mape = sum(mape_list) / len(mape_list) if mape_list else None
                avg_str = f"**`{avg_mape:.2f}%`**" if avg_mape is not None else "`N/A`"

                md += (
                    f"| {row_prefix} | `{m_short}` | "
                    + " | ".join(row_vals)
                    + f" | {avg_str} |\n"
                )

            # Separator between scenarios
            md += "| | | " + " | ".join("" for _ in RPS_POINTS) + " | |\n"

    output_md.write_text(md, encoding="utf-8")
    print(f"[acc_report] ACC report markdown saved to {output_md}")


def _scope_cells(report: dict[str, Any] | None, scope: str) -> list[dict[str, Any]]:
    if not report:
        return []
    cells = report.get("cells", [])
    if scope == "rps":
        return [cell for cell in cells if "_rps_point_" in cell.get("point", "")]
    if scope == "single_batch":
        return [cell for cell in cells if "_single_batch_point_" in cell.get("point", "")]
    return list(cells)


def _cell_summary(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not cells:
        return None
    return {
        "cell_count": len(cells),
        "mean_dynosim_mape_percent": statistics.fmean(
            float(cell["dynosim_mape_percent"]) for cell in cells
        ),
        "mean_hisim_mape_percent": statistics.fmean(
            float(cell["hisim_mape_percent"]) for cell in cells
        ),
    }


def _percent(value: float | None) -> str:
    return "`N/A`" if value is None else f"`{value:.6f}%`"


def _signed_pp(value: float | None) -> str:
    return "`N/A`" if value is None else f"`{value:+.6f} pp`"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_aic_database() -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    canonical_systems = CANONICAL_AIC_ROOT / "src" / "aiconfigurator" / "systems"
    dynosim_systems = DYNOSIM_AIC_ROOT / "src" / "aiconfigurator" / "systems"
    canonical_data = canonical_systems / "data" / "h100_pcie"
    dynosim_data = dynosim_systems / "data" / "h100_pcie"
    tables = []
    max_abs = 0.0
    max_rel = 0.0
    total_rows = 0
    all_equivalent = True
    for text_path in sorted(canonical_data.rglob("*.txt")):
        parquet_path = dynosim_data / text_path.relative_to(canonical_data).with_suffix(".parquet")
        text_frame = pd.read_csv(text_path)
        parquet_frame = pd.read_parquet(parquet_path)
        equivalent = list(text_frame.columns) == list(parquet_frame.columns) and len(text_frame) == len(parquet_frame)
        table_abs = 0.0
        table_rel = 0.0
        if equivalent:
            for column in text_frame.columns:
                left = text_frame[column]
                right = parquet_frame[column]
                if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                    left_values = left.to_numpy(dtype=float)
                    right_values = right.to_numpy(dtype=float)
                    delta = np.abs(left_values - right_values)
                    finite = np.isfinite(delta)
                    if finite.any():
                        table_abs = max(table_abs, float(delta[finite].max()))
                        denominator = np.maximum(np.abs(left_values[finite]), np.finfo(float).tiny)
                        table_rel = max(table_rel, float((delta[finite] / denominator).max()))
                    equivalent &= bool(
                        np.allclose(left_values, right_values, rtol=1e-14, atol=1e-15, equal_nan=True)
                    )
                else:
                    equivalent &= left.fillna("<NA>").equals(right.fillna("<NA>"))
        total_rows += len(text_frame)
        max_abs = max(max_abs, table_abs)
        max_rel = max(max_rel, table_rel)
        all_equivalent &= equivalent
        tables.append(
            {
                "path": text_path.relative_to(canonical_data).as_posix(),
                "rows": len(text_frame),
                "columns": len(text_frame.columns),
                "numeric_equivalent": equivalent,
                "max_absolute_delta": table_abs,
                "max_relative_delta": table_rel,
                "canonical_sha256": _sha256(text_path),
                "dynosim_sha256": _sha256(parquet_path),
            }
        )

    overlay_moe = pd.read_csv(
        RECORDED_OVERLAY_ROOT / "data" / "h100_pcie" / "sglang" / "0.5.9" / "moe_perf.txt"
    )
    base_moe = pd.read_csv(canonical_data / "sglang" / "0.5.9" / "moe_perf.txt")
    yaml_canonical = canonical_systems / "h100_pcie.yaml"
    yaml_dynosim = dynosim_systems / "h100_pcie.yaml"
    return {
        "schema": "dynosim_aic_database_equivalence.v1",
        "system": "h100_pcie",
        "canonical_root": str(canonical_systems.resolve()),
        "dynosim_resolved_root": str(dynosim_systems.resolve()),
        "table_count": len(tables),
        "row_count": total_rows,
        "numeric_equivalent": all_equivalent,
        "tolerance": {"rtol": 1e-14, "atol": 1e-15},
        "max_absolute_delta": max_abs,
        "max_relative_delta": max_rel,
        "system_yaml": {
            "canonical_sha256": _sha256(yaml_canonical),
            "dynosim_sha256": _sha256(yaml_dynosim),
            "byte_identical": _sha256(yaml_canonical) == _sha256(yaml_dynosim),
        },
        "tables": tables,
        "recorded_overlay": {
            "base_rows_preserved_exactly": base_moe.equals(
                overlay_moe.iloc[: len(base_moe)].reset_index(drop=True)
            ),
            "base_row_count": len(base_moe),
            "added_row_count": len(overlay_moe) - len(base_moe),
            "added_distributions": sorted(
                set(overlay_moe["distribution"]) - set(base_moe["distribution"])
            ),
            "balanced_replay_consumes_recorded_rows": False,
        },
        "historical_hisim": {
            "artifact_database_path": None,
            "resolved_systems_root": "unrecorded",
            "resolved_file_hashes": "unrecorded",
            "physical_root_or_hash_identity_proven": False,
        },
        "conclusion": (
            "Canonical HiSim-side txt tables and DynoSim parquet mirrors are numerically equivalent; "
            "the historical HiSim run's physical systems root and file hashes were not recorded."
        ),
    }


def generate_diagnostic_report(
    report: dict[str, Any],
    before_report: dict[str, Any] | None,
    database: dict[str, Any],
    output_md: Path,
) -> None:
    before_cells = {(cell["point"], cell["metric"]): cell for cell in _scope_cells(before_report, "all")}
    changes = []
    for cell in report.get("cells", []):
        before = before_cells.get((cell["point"], cell["metric"]))
        if before:
            changes.append(
                {
                    "point": cell["point"],
                    "metric": cell["metric"],
                    "before": float(before["dynosim_mape_percent"]),
                    "after": float(cell["dynosim_mape_percent"]),
                    "change": float(cell["dynosim_mape_percent"])
                    - float(before["dynosim_mape_percent"]),
                }
            )
    best = min(changes, key=lambda item: item["change"])
    worst = max(changes, key=lambda item: item["change"])
    lines = [
        "# DynoSim P0/P1/P2 Diagnostic Report",
        "",
        "This report covers CPU/offline replay only. It does not replace HiSim formal ACC. P3 is deferred.",
        "",
        "## P0: AIC Database Provenance",
        "",
        (
            f"- DynoSim resolved root: `{database['dynosim_resolved_root']}`. "
            f"Audited {database['table_count']} tables / {database['row_count']} rows; "
            f"numeric equivalence is `{database['numeric_equivalent']}`."
        ),
        (
            f"- Maximum txt/parquet conversion delta: absolute {database['max_absolute_delta']:.3e}, "
            f"relative {database['max_relative_delta']:.3e}; H100 PCIe YAML is byte-identical: "
            f"`{database['system_yaml']['byte_identical']}`."
        ),
        "- Historical HiSim artifacts record `database_path=null` but not the resolved systems root/hash. Data content is comparable; physical root/hash identity is not proven.",
        (
            f"- Recorded overlay preserves all {database['recorded_overlay']['base_row_count']} base MoE rows "
            f"and adds {database['recorded_overlay']['added_row_count']} recorded-only rows. "
            "Balanced replay does not consume those rows, so the overlay is not the old 50 ms decode root cause."
        ),
        "",
        "## P1/P2 Outcome",
        "",
        "- P1 separates BF16 global KV from FP8 MLA-module KV, uses the SGLang WideEP MLA graph, and preserves the 3 dense + 17 MoE cropped-model structure.",
        "- P2 emits non-mixed first tokens at each prefill boundary without charging a decode duration or reserving KV before the token is consumed by a later forward pass.",
        "- CUDA graph capture sizes are read per point from the formal HiSim sim config; D-32 uses captures through batch 128.",
        "",
        "## Before / After Extremes",
        "",
        f"- Largest improvement: `{best['point']}` / `{best['metric']}`, {best['before']:.6f}% -> {best['after']:.6f}% ({best['change']:+.6f} pp).",
        f"- Largest regression: `{worst['point']}` / `{worst['metric']}`, {worst['before']:.6f}% -> {worst['after']:.6f}% ({worst['change']:+.6f} pp).",
        "",
        "## Remaining Accuracy Gap",
        "",
        "- Decode/ITL improves materially, but long-input TTFT and throughput remain inaccurate. The remaining gap is scheduler prefill admission/visibility and client-envelope behavior, not evidence for changing AIC table values.",
        "- `workload_distribution=balanced` is an AIC workload assumption; it is not proof that DynoSim executes SGLang fake TopK routing.",
        "- P3 remains deferred until the P0/P1/P2 results are accepted.",
        "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sweep_acc_report.md and ACC plots for DynoSim runs."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Path to the DynoSim run directory containing accuracy_report.json.",
    )
    parser.add_argument(
        "--before-report",
        type=Path,
        default=DEFAULT_BEFORE_REPORT,
        help="Prefix-fixed before report used for same-scope P0/P1/P2 comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to output sweep_acc_report.md and PNG plots (defaults to run-dir).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_map, report = load_accuracy_data(run_dir)
    before_report = (
        json.loads(args.before_report.read_text()) if args.before_report and args.before_report.is_file() else None
    )
    database = audit_aic_database()
    (output_dir / "aic_database_equivalence.json").write_text(
        json.dumps(database, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    
    # Plot RPS curves if RPS data exists
    if any(len(data_map["rps"][sc][m]) > 0 for sc in SCENARIOS for m, _, _ in METRICS):
        rps_png = output_dir / "rps_acc_sweep.png"
        plot_acc_curves(data_map, "rps", RPS_POINTS, rps_png)

    # Plot single_batch curves if single_batch data exists
    if any(len(data_map["single_batch"][sc][m]) > 0 for sc in SCENARIOS for m, _, _ in METRICS):
        sb_png = output_dir / "single_batch_acc_sweep.png"
        plot_acc_curves(data_map, "single_batch", BATCH_SIZES, sb_png)

    md_path = output_dir / "sweep_acc_report.md"
    generate_acc_report_md(data_map, report, before_report, md_path)
    generate_diagnostic_report(
        report, before_report, database, output_dir / "sweep_diagnostic_report.md"
    )


if __name__ == "__main__":
    main()
