# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable plots and a Chinese narrative for matrix-search reports."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES: tuple[tuple[str, str], ...] = (
    ("pareto_front", "00_pareto_front.png"),
    ("load_conditioned_pareto", "00a_load_conditioned_pareto.png"),
    ("model_sku_cost", "01_model_sku_cost.png"),
    ("tpsu_cost_envelope", "02_tpsu_cost_envelope.png"),
    ("context_cost", "03_context_cost.png"),
    ("kv_capacity_uplift", "04_kv_capacity_uplift.png"),
    ("kv_bandwidth_uplift", "05_kv_bandwidth_uplift.png"),
)


def _read_csv(output_dir: Path, filename: str) -> pd.DataFrame:
    path = output_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"report input is missing: {path}")
    return pd.read_csv(path)


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _label(identifier: Any) -> str:
    value = str(identifier)
    aliases = {
        "b300_sxm": "B300 SXM",
        "gb300": "GB300",
        "deepseek_v4_flash": "DeepSeek V4 Flash",
        "deepseek_v4_pro": "DeepSeek V4 Pro",
        "minimax_m3": "MiniMax M3",
        "glm_5_2": "GLM 5.2",
        "kimi_k3": "Kimi K3",
    }
    return aliases.get(value, value.replace("_", " ").title())


def _variant_label(model_id: Any, variant_id: Any) -> str:
    """Keep model variants visually distinct without hiding the base model."""

    if variant_id is None or pd.isna(variant_id) or not str(variant_id):
        return _label(model_id)
    return f"{_label(model_id)} / {variant_id}"


def _workload_label(identifier: Any) -> str:
    value = str(identifier)
    lowered = value.lower()
    if lowered == "synthetic_8k_1k_no_prefix":
        return "合成负载 ISL 8K / OSL 1K（无 prefix）"
    if lowered.startswith("synthetic"):
        suffix = next(
            (
                part.upper()
                for part in reversed(value.split("_"))
                if part.lower().endswith("k")
            ),
            "",
        )
        return f"合成负载 {suffix}".rstrip()
    return _label(value)


def _currency_label(frame: pd.DataFrame) -> str:
    if "currency" not in frame:
        return "currency units"
    values = [str(value) for value in frame["currency"].dropna().unique()]
    return values[0] if len(values) == 1 else "currency units"


def _context_label(value: Any) -> str:
    try:
        context = float(value)
    except (TypeError, ValueError):
        return "unknown context"
    if math.isnan(context):
        return "unknown context"
    if context >= 1024:
        return f"{context / 1024:g}K"
    return f"{context:g}"


def _style_axes(axis: Any) -> None:
    axis.grid(axis="y", color="#d9dde3", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save(fig: Any, path: Path) -> None:
    if getattr(fig, "_suptitle", None) is not None:
        # Reserve a fixed physical band for the title.  A fixed fractional margin
        # is either too small on short figures or enormous on 20-Cell Pareto grids.
        top = max(0.8, 1.0 - 0.5 / fig.get_figheight())
        fig.tight_layout(rect=(0, 0, 1, top))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _empty_figure(path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.axis("off")
    axis.set_title(title, fontsize=14, weight="bold")
    axis.text(
        0.5,
        0.48,
        "No SLA-feasible data",
        ha="center",
        va="center",
        fontsize=13,
        color="#667085",
    )
    _save(fig, path)


def _diagnostic_frontier_mask(frame: pd.DataFrame) -> pd.Series:
    """Return the non-dominated raw-throughput frontier for one Cell."""

    mask = pd.Series(False, index=frame.index)
    best_y = -math.inf
    for x_value in sorted(frame["throughput_per_user"].dropna().unique(), reverse=True):
        bucket = frame.loc[frame["throughput_per_user"] == x_value]
        bucket_best = bucket["raw_throughput_per_gpu"].max()
        if bucket_best > best_y:
            mask.loc[bucket.index[bucket["raw_throughput_per_gpu"] == bucket_best]] = (
                True
            )
            best_y = bucket_best
    return mask


def _plot_pareto_front(frame: pd.DataFrame, path: Path) -> None:
    frame = _numeric(
        frame,
        [
            "goodput_per_gpu",
            "raw_throughput_per_gpu",
            "throughput_per_user",
            "used_gpus",
        ],
    ).dropna(subset=["raw_throughput_per_gpu", "throughput_per_user"])
    if frame.empty:
        _empty_figure(path, "Throughput/GPU versus TPS/user Pareto frontier")
        return

    frame["strict_pareto"] = (
        frame["strict_pareto"].astype(str).str.lower().isin({"true", "1"})
    )
    frame["sla_pass"] = frame["sla_pass"].astype(str).str.lower().isin({"true", "1"})
    groups = list(
        frame.groupby(
            ["model_id", "model_variant_id", "sku_id", "workload_id"],
            sort=True,
        )
    )
    fig, axes = _subplot_grid(len(groups))
    for axis, ((model_id, variant_id, sku_id, workload_id), group) in zip(axes, groups):
        axis.scatter(
            group["throughput_per_user"],
            group["raw_throughput_per_gpu"],
            s=20,
            alpha=0.18,
            color="#98a2b3",
            label="Replay-success candidates",
        )
        sla_candidates = group.loc[group["sla_pass"]]
        if not sla_candidates.empty:
            axis.scatter(
                sla_candidates["throughput_per_user"],
                sla_candidates["raw_throughput_per_gpu"],
                s=24,
                alpha=0.45,
                color="#1570ef",
                label="SLA-feasible candidates",
            )
        front = group.loc[group["strict_pareto"] & group["sla_pass"]].sort_values(
            "throughput_per_user"
        )
        if not front.empty:
            axis.plot(
                front["throughput_per_user"],
                front["raw_throughput_per_gpu"],
                marker="o",
                markersize=5,
                linewidth=2,
                color="#d92d20",
                label="Formal SLA Pareto front",
            )
        diagnostic = group.loc[_diagnostic_frontier_mask(group)].sort_values(
            "throughput_per_user"
        )
        if not diagnostic.empty:
            axis.plot(
                diagnostic["throughput_per_user"],
                diagnostic["raw_throughput_per_gpu"],
                marker="o",
                markersize=4,
                linewidth=1.5,
                linestyle="--",
                color="#f79009",
                label="Diagnostic raw-throughput front",
            )
        axis.set_title(
            f"{_variant_label(model_id, variant_id)} / {_label(sku_id)} / "
            f"{_label(workload_id)}",
            weight="bold",
        )
        axis.set_xlabel("Output throughput per user (tokens/s/user)")
        axis.set_ylabel("Output throughput per GPU (tokens/s/GPU)")
        axis.legend(frameon=False)
        _style_axes(axis)
    fig.suptitle(
        "Formal SLA and diagnostic throughput Pareto frontiers",
        fontsize=15,
        weight="bold",
    )
    _save(fig, path)


def _plot_load_conditioned_pareto(
    load_front: pd.DataFrame,
    global_envelope: pd.DataFrame,
    path: Path,
) -> None:
    load_front = _numeric(
        load_front,
        ["concurrency", "goodput_per_gpu", "throughput_per_user"],
    ).dropna(subset=["concurrency", "goodput_per_gpu", "throughput_per_user"])
    global_envelope = _numeric(
        global_envelope,
        ["discovery_concurrency", "goodput_per_gpu", "throughput_per_user"],
    ).dropna(subset=["goodput_per_gpu", "throughput_per_user"])
    if load_front.empty:
        _empty_figure(path, "Load-conditioned Pareto and global operating envelope")
        return

    fig, axis = plt.subplots(figsize=(10.5, 6.2))
    palette = plt.cm.viridis(
        np.linspace(0.08, 0.88, max(load_front["concurrency"].nunique(), 1))
    )
    for color, (concurrency, group) in zip(
        palette, load_front.groupby("concurrency", sort=True)
    ):
        group = group.sort_values("throughput_per_user")
        axis.plot(
            group["throughput_per_user"],
            group["goodput_per_gpu"],
            marker="o",
            markersize=4,
            linewidth=1.3,
            alpha=0.75,
            color=color,
            label=f"Concurrency {concurrency:g}",
        )
    if not global_envelope.empty:
        global_envelope = global_envelope.sort_values("throughput_per_user")
        axis.plot(
            global_envelope["throughput_per_user"],
            global_envelope["goodput_per_gpu"],
            marker="D",
            markersize=6,
            linewidth=2.6,
            color="#d92d20",
            label="Global operating envelope",
        )
    axis.set_title("Load-conditioned Pareto frontiers", fontsize=14, weight="bold")
    axis.set_xlabel("Output throughput per user (tokens/s/user)")
    axis.set_ylabel("Goodput per GPU (tokens/s/GPU)")
    axis.legend(frameon=False, ncols=2, fontsize=8)
    _style_axes(axis)
    _save(fig, path)


def _plot_model_sku_cost(frame: pd.DataFrame, path: Path) -> None:
    frame = _numeric(
        frame,
        ["context_length", "all_in_cost_per_million_total_tokens"],
    ).dropna(subset=["all_in_cost_per_million_total_tokens"])
    if frame.empty:
        _empty_figure(path, "Model and SKU cost comparison")
        return

    frame["scenario"] = frame.apply(
        lambda row: (
            f"{_variant_label(row['model_id'], row['model_variant_id'])}\n"
            f"{_context_label(row['context_length'])}"
        ),
        axis=1,
    )
    if frame.groupby("scenario")["workload_id"].nunique().max() > 1:
        frame["scenario"] = frame.apply(
            lambda row: f"{row['scenario']}\n{_label(row['workload_id'])}", axis=1
        )
    scenarios = list(dict.fromkeys(frame["scenario"]))
    skus = sorted(frame["sku_id"].dropna().unique(), key=str)
    x_values = np.arange(len(scenarios))
    width = min(0.8 / max(len(skus), 1), 0.34)
    fig, axis = plt.subplots(figsize=(max(10, len(scenarios) * 1.65), 6.2))
    positive = frame.loc[
        frame["all_in_cost_per_million_total_tokens"] > 0,
        "all_in_cost_per_million_total_tokens",
    ]
    use_log = not positive.empty and positive.max() / positive.min() >= 20
    for index, sku in enumerate(skus):
        lookup = (
            frame.loc[frame["sku_id"] == sku]
            .groupby("scenario", sort=False)["all_in_cost_per_million_total_tokens"]
            .min()
        )
        values = [lookup.get(scenario, np.nan) for scenario in scenarios]
        offset = (index - (len(skus) - 1) / 2) * width
        bars = axis.bar(
            x_values + offset,
            values,
            width=width,
            label=_label(sku),
            alpha=0.9,
        )
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    if use_log:
        axis.set_yscale("log")
    axis.set_xticks(x_values, scenarios)
    axis.set_ylabel(f"All-in cost ({_currency_label(frame)} / 1M goodput tokens)")
    axis.set_title("Best SLA-feasible cost by model, workload and SKU", weight="bold")
    axis.legend(title="GPU SKU", frameon=False)
    _style_axes(axis)
    _save(fig, path)


def _subplot_grid(
    count: int, *, width: float = 6.0, height: float = 4.3
) -> tuple[Any, list[Any]]:
    columns = 2 if count > 1 else 1
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * width, rows * height))
    flattened = list(np.atleast_1d(axes).ravel())
    for axis in flattened[count:]:
        axis.set_visible(False)
    return fig, flattened[:count]


def _plot_tpsu_cost(frame: pd.DataFrame, path: Path) -> None:
    frame = _numeric(
        frame,
        ["required_tps_per_user", "all_in_cost_per_million_total_tokens"],
    ).dropna(subset=["required_tps_per_user", "all_in_cost_per_million_total_tokens"])
    if frame.empty:
        _empty_figure(path, "Cost envelope versus TPS/user requirement")
        return
    groups = list(
        frame.groupby(["model_id", "model_variant_id", "workload_id"], sort=True)
    )
    fig, axes = _subplot_grid(len(groups))
    for axis, ((model_id, variant_id, workload_id), group) in zip(axes, groups):
        for sku, series in group.groupby("sku_id", sort=True):
            points = (
                series.groupby("required_tps_per_user", as_index=False)[
                    "all_in_cost_per_million_total_tokens"
                ]
                .min()
                .sort_values("required_tps_per_user")
            )
            axis.step(
                points["required_tps_per_user"],
                points["all_in_cost_per_million_total_tokens"],
                where="post",
                marker="o",
                markersize=3.5,
                label=_label(sku),
            )
        context = ""
        context_source = group.get("workload_id")
        if context_source is not None and len(context_source):
            match = str(workload_id).lower()
            if "32k" in match:
                context = " / 32K"
            elif "4k" in match:
                context = " / 4K"
        axis.set_title(
            f"{_variant_label(model_id, variant_id)}{context}", weight="bold"
        )
        axis.set_xlabel("Required output TPS / user")
        axis.set_ylabel(f"{_currency_label(group)} / 1M goodput tokens")
        axis.legend(title="GPU SKU", frameon=False)
        _style_axes(axis)
    fig.suptitle(
        "Minimum feasible cost envelope versus TPS/user", fontsize=15, weight="bold"
    )
    _save(fig, path)


def _plot_context_cost(frame: pd.DataFrame, path: Path) -> None:
    frame = _numeric(
        frame,
        ["context_length", "best_cost_per_million"],
    ).dropna(subset=["context_length", "best_cost_per_million"])
    if frame.empty:
        _empty_figure(path, "Cost versus context length")
        return
    groups = list(frame.groupby(["model_id", "model_variant_id"], sort=True))
    fig, axes = _subplot_grid(len(groups))
    for axis, ((model_id, variant_id), group) in zip(axes, groups):
        for sku, series in group.groupby("sku_id", sort=True):
            points = (
                series.groupby("context_length", as_index=False)[
                    "best_cost_per_million"
                ]
                .min()
                .sort_values("context_length")
            )
            axis.plot(
                points["context_length"] / 1024,
                points["best_cost_per_million"],
                marker="o",
                linewidth=2,
                label=_label(sku),
            )
        axis.set_title(_variant_label(model_id, variant_id), weight="bold")
        axis.set_xlabel("Context length (K tokens)")
        axis.set_ylabel(f"{_currency_label(group)} / 1M goodput tokens")
        axis.legend(title="GPU SKU", frameon=False)
        _style_axes(axis)
    fig.suptitle(
        "Best feasible cost versus workload context", fontsize=15, weight="bold"
    )
    _save(fig, path)


def _plot_uplift_panels(
    frame: pd.DataFrame,
    path: Path,
    *,
    x_column: str,
    group_columns: Sequence[str],
    title: str,
    x_label: str,
    annotate_capacity: bool = False,
    footnote: str | None = None,
) -> None:
    numeric_columns = [
        x_column,
        "g2_capacity_gib",
        "target_hit_rate",
        "measured_hit_rate",
        "matched_config_cost_reduction_ratio",
        "matched_config_rps_increase_ratio",
    ]
    frame = _numeric(frame, numeric_columns).dropna(subset=[x_column])
    metrics = (
        ("matched_config_cost_reduction_ratio", "Matched cost reduction (%)"),
        ("matched_config_rps_increase_ratio", "Matched SLA-RPS increase (%)"),
    )
    if frame.empty or all(frame[metric].dropna().empty for metric, _ in metrics):
        _empty_figure(path, title)
        return

    aggregation = {metric: "median" for metric, _ in metrics}
    if "g2_capacity_gib" not in group_columns:
        aggregation["g2_capacity_gib"] = "median"
    grouped = frame.groupby(list(group_columns), as_index=False).agg(aggregation)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    for axis, (metric, y_label) in zip(axes, metrics):
        series_columns = [
            column
            for column in ("model_id", "model_variant_id", "sku_id")
            if column in grouped
        ]
        for series_key, series in grouped.groupby(series_columns, sort=True):
            if not isinstance(series_key, tuple):
                series_key = (series_key,)
            series_values = dict(zip(series_columns, series_key, strict=True))
            sku = series_values["sku_id"]
            variant_id = series_values.get("model_variant_id")
            model_id = series_values.get("model_id")
            points = series.sort_values(x_column)
            axis.plot(
                points[x_column],
                points[metric] * 100,
                marker="o",
                linewidth=2,
                label=(
                    _label(sku)
                    if model_id is None
                    else f"{_label(sku)} / {_variant_label(model_id, variant_id)}"
                ),
            )
            if annotate_capacity and "g2_capacity_gib" in points:
                for _, point in points.iterrows():
                    axis.annotate(
                        f"{point['g2_capacity_gib']:g} GiB",
                        (point[x_column], point[metric] * 100),
                        xytext=(4, 5),
                        textcoords="offset points",
                        fontsize=8,
                    )
        axis.axhline(0, color="#344054", linewidth=0.9)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.legend(title="GPU SKU", frameon=False)
        _style_axes(axis)
    fig.suptitle(title, fontsize=15, weight="bold")
    if footnote:
        fig.text(0.5, 0.005, footnote, ha="center", fontsize=9, color="#667085")
        fig.subplots_adjust(bottom=0.14)
    _save(fig, path)


def _plot_kv_capacity(frame: pd.DataFrame, path: Path) -> None:
    frame = _numeric(
        frame,
        [
            "g2_capacity_gib",
            "measured_hit_rate",
            "target_hit_rate",
            "matched_config_cost_reduction_ratio",
            "matched_config_rps_increase_ratio",
        ],
    )
    frame["plot_hit_rate"] = frame["measured_hit_rate"].fillna(frame["target_hit_rate"])
    frame = frame.groupby(
        [
            "model_id",
            "model_variant_id",
            "sku_id",
            "g2_capacity_gib",
            "target_hit_rate",
        ],
        as_index=False,
    ).agg(
        plot_hit_rate=("plot_hit_rate", "median"),
        matched_config_cost_reduction_ratio=(
            "matched_config_cost_reduction_ratio",
            "median",
        ),
        matched_config_rps_increase_ratio=(
            "matched_config_rps_increase_ratio",
            "median",
        ),
    )
    _plot_uplift_panels(
        frame,
        path,
        x_column="plot_hit_rate",
        group_columns=(
            "model_id",
            "model_variant_id",
            "sku_id",
            "g2_capacity_gib",
            "target_hit_rate",
            "plot_hit_rate",
        ),
        title="KVBM capacity / measured-hit scenarios (median matched uplift)",
        x_label="Measured prefix-cache hit ratio",
        annotate_capacity=True,
    )


def _plot_kv_bandwidth(frame: pd.DataFrame, path: Path) -> None:
    _plot_uplift_panels(
        frame,
        path,
        x_column="storage_to_compute_nic_ratio",
        group_columns=(
            "model_id",
            "model_variant_id",
            "sku_id",
            "storage_to_compute_nic_ratio",
        ),
        title="KVBM bandwidth scenarios (median matched uplift)",
        x_label="Storage bandwidth / compute-NIC bandwidth",
        footnote="Capacity and target hit rate may change with bandwidth; this is not a single-variable causal curve.",
    )


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def escape(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    materialized = [list(row) for row in rows]
    if not materialized:
        return "_无可用数据。_"
    lines = [
        "| " + " | ".join(map(escape, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(escape, row)) + " |" for row in materialized)
    return "\n".join(lines)


def _parse_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _topology_summary(value: Any) -> str:
    config = _parse_config(value)
    mode = config.get("deployment_mode")
    router = config.get("router_mode") or "—"
    if mode == "disagg":
        prefill = (
            f"P: TP{config.get('prefill_tp', 1)}/EP{config.get('prefill_moe_ep', 1)}"
            f" × {config.get('prefill_workers', config.get('prefill_instances', 1))}"
        )
        decode = (
            f"D: TP{config.get('decode_tp', 1)}/EP{config.get('decode_moe_ep', 1)}"
            f" × {config.get('decode_workers', config.get('decode_instances', 1))}"
        )
        domain = config.get("pd_bandwidth_domain") or "未解析"
        bandwidth = config.get("pd_transfer_bandwidth_gbytes_per_s")
        transfer = (
            f"PD域={domain}"
            if bandwidth is None
            else f"PD域={domain}/{_fmt(bandwidth, 1)} GB/s"
        )
        return f"PD 分离；{prefill}；{decode}；{transfer}；路由={router}"
    return (
        f"聚合；TP{config.get('tp', 1)}/EP{config.get('moe_ep', 1)}"
        f" × {config.get('instances', config.get('num_workers', 1))}；路由={router}"
    )


def _paired_sku_findings(costs: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    for (model, variant, workload), group in costs.groupby(
        ["model_id", "model_variant_id", "workload_id"], sort=True
    ):
        comparable = group.dropna(subset=["all_in_cost_per_million_total_tokens"])
        if comparable["sku_id"].nunique() < 2:
            continue
        best = comparable.sort_values("all_in_cost_per_million_total_tokens").iloc[0]
        second = comparable.sort_values("all_in_cost_per_million_total_tokens").iloc[1]
        delta = (
            second["all_in_cost_per_million_total_tokens"]
            / best["all_in_cost_per_million_total_tokens"]
            - 1
        ) * 100
        conclusion = (
            f"{_variant_label(model, variant)} / {_workload_label(workload)}："
            f"{_label(best['sku_id'])} 的最低成本比 "
            f"{_label(second['sku_id'])} 低 {delta:.2f}%。"
        )
        rps_points = comparable.dropna(subset=["sla_rps"])
        if len(rps_points) >= 2:
            faster = rps_points.sort_values("sla_rps", ascending=False).iloc[0]
            slower = rps_points.sort_values("sla_rps", ascending=False).iloc[1]
            if float(slower["sla_rps"]) > 0:
                rps_delta = (
                    float(faster["sla_rps"]) / float(slower["sla_rps"]) - 1
                ) * 100
                if rps_delta > 0.01:
                    conclusion += (
                        f"按各 SKU 的最低成本候选，{_label(faster['sku_id'])} 的 SLA RPS "
                        f"高 {rps_delta:.1f}%。"
                    )
                else:
                    conclusion += "两个最低成本候选的 SLA RPS 相同。"
        findings.append(conclusion)
    return findings


def _context_findings(context: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    for (model, variant, sku), group in context.groupby(
        ["model_id", "model_variant_id", "sku_id"], sort=True
    ):
        points = group.dropna(
            subset=["context_length", "best_cost_per_million"]
        ).sort_values("context_length")
        if len(points) < 2:
            continue
        first, last = points.iloc[0], points.iloc[-1]
        change = (
            last["best_cost_per_million"] / first["best_cost_per_million"] - 1
        ) * 100
        direction = "上升" if change >= 0 else "下降"
        conclusion = (
            f"{_variant_label(model, variant)} / {_label(sku)}：从 "
            f"{_context_label(first['context_length'])} "
            f"到 {_context_label(last['context_length'])}，每百万 goodput token 成本{direction} "
            f"{abs(change):.2f}%。"
        )
        first_rps = first.get("sla_rps")
        last_rps = last.get("sla_rps")
        if pd.notna(first_rps) and pd.notna(last_rps) and float(first_rps) > 0:
            rps_change = (float(last_rps) / float(first_rps) - 1) * 100
            rps_direction = "上升" if rps_change >= 0 else "下降"
            conclusion += f"同时，SLA RPS {rps_direction} {abs(rps_change):.1f}%。"
        findings.append(conclusion)
    return findings


def _ratio_range(frame: pd.DataFrame, column: str) -> str:
    values = pd.to_numeric(frame.get(column), errors="coerce").dropna() * 100
    if values.empty:
        return "无可用配对结果"
    return f"{values.min():.2f}% 至 {values.max():.2f}%"


def _write_chinese_markdown(
    output_dir: Path,
    summary: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    figure_paths: Mapping[str, str],
) -> Path:
    costs = _numeric(
        frames["costs"],
        [
            "context_length",
            "all_in_cost_per_million_total_tokens",
            "sla_rps",
            "used_gpus",
        ],
    )
    pareto = _numeric(
        frames["pareto"],
        [
            "goodput_per_gpu",
            "raw_throughput_per_gpu",
            "throughput_per_user",
            "used_gpus",
        ],
    )
    tpsu = _numeric(
        frames["tpsu"],
        ["required_tps_per_user", "all_in_cost_per_million_total_tokens"],
    )
    context = _numeric(
        frames["context"], ["context_length", "best_cost_per_million", "sla_rps"]
    )
    capacity = frames["capacity"]
    bandwidth = frames["bandwidth"]
    statuses = frames["statuses"]
    prices = frames["prices"]
    footprints = frames["footprints"]
    operating_points = frames.get("operating_points", pd.DataFrame())
    load_front = frames.get("load_front", pd.DataFrame())
    global_envelope = frames.get("global_envelope", pd.DataFrame())
    certification = frames.get("certification", pd.DataFrame())

    best_rows = []
    for _, row in costs.sort_values(
        ["model_id", "model_variant_id", "workload_id", "sku_id"]
    ).iterrows():
        best_rows.append(
            (
                _label(row["model_id"]),
                row["model_variant_id"],
                _workload_label(row["workload_id"]),
                _context_label(row["context_length"]),
                _label(row["sku_id"]),
                _fmt(row["all_in_cost_per_million_total_tokens"], 4),
                _fmt(row["sla_rps"], 3),
                _fmt(row["used_gpus"], 0),
                _topology_summary(row.get("config_json")),
            )
        )

    price_rows = []
    for _, row in prices.iterrows():
        price_id = str(row.get("price_id", ""))
        is_proxy = (
            "proxy" in price_id.lower() or "proxy" in str(row.get("basis", "")).lower()
        )
        source_url = row.get("source_url")
        source = (
            f"[来源]({source_url})"
            if isinstance(source_url, str) and source_url
            else "—"
        )
        price_rows.append(
            (
                _label(row.get("sku_id")),
                row.get("currency", "—"),
                _fmt(row.get("gpu_hour"), 4),
                _fmt(row.get("g2_gib_hour"), 7),
                _fmt(row.get("network_gb"), 4),
                "代理值" if is_proxy else "公开价格锚点",
                row.get("source_date", "—"),
                source,
            )
        )

    report_currency = _currency_label(costs)
    complete = (
        int((statuses.get("status") == "complete").sum()) if not statuses.empty else 0
    )
    failed = (
        int((statuses.get("status") == "failed").sum()) if not statuses.empty else 0
    )
    unsupported = (
        int((statuses.get("status") == "unsupported").sum())
        if not statuses.empty
        else 0
    )
    no_success_models = []
    if not statuses.empty:
        for (model, variant), group in statuses.groupby(
            ["model_id", "model_variant_id"], sort=True
        ):
            if not (group["status"] == "complete").any():
                no_success_models.append(_variant_label(model, variant))

    variant_rows = [
        (
            _label(row["model_id"]),
            row["model_variant_id"],
            row.get("weight_precision", "—"),
            row.get("aic_gemm_dtype", "—"),
            row.get("aic_moe_dtype", "—"),
            row.get("aic_fmha_dtype", "—"),
            row.get("aic_kv_cache_dtype", "—"),
            row.get("aic_comm_dtype", "—"),
            row.get("support_tier", "—"),
            row.get("variant_resolution_source", "—"),
            row.get("variant_support_status", "—"),
            row.get("deployment_fidelity", "—"),
        )
        for _, row in statuses.drop_duplicates(subset=["model_id", "model_variant_id"])
        .sort_values(["model_id", "model_variant_id"])
        .iterrows()
    ]

    paired_findings = _paired_sku_findings(costs)
    context_findings = _context_findings(context)
    matched_capacity = (
        pd.to_numeric(
            capacity.get("matched_config_cost_reduction_ratio"), errors="coerce"
        )
        .notna()
        .sum()
    )
    matched_rps_positive = (
        pd.to_numeric(
            capacity.get("matched_config_rps_increase_ratio"), errors="coerce"
        )
        > 0
    ).sum()
    system_cost_positive = (
        pd.to_numeric(
            capacity.get("system_optimal_cost_reduction_ratio"), errors="coerce"
        )
        > 0
    ).sum()
    system_rps_positive = (
        pd.to_numeric(
            capacity.get("system_optimal_rps_increase_ratio"), errors="coerce"
        )
        > 0
    ).sum()

    coupled_bandwidth = False
    if not bandwidth.empty:
        scenario_map = bandwidth.groupby("storage_to_compute_nic_ratio").agg(
            capacity=("g2_capacity_gib", "median"),
            target_hit=("target_hit_rate", "median"),
        )
        coupled_bandwidth = len(scenario_map) > 1 and (
            scenario_map["capacity"].nunique() > 1
            or scenario_map["target_hit"].nunique() > 1
        )

    lines = [
        "# DynoSim + AIC 矩阵搜索报告",
        "",
        "> 本报告由 `python -m aisimulate.matrix report` 自动生成。图表使用英文标注，正文和结论使用中文。",
        "",
        "## 运行概况",
        "",
        (
            f"本次数据库包含 **{int(summary.get('cells', len(statuses)))}** 个矩阵 Cell、"
            f"**{int(summary.get('evaluations', 0))}** 次评估、**{int(summary.get('candidates', 0))}** 个候选和 "
            f"**{int(summary.get('pareto', 0))}** 个严格 Pareto 成员。Cell 状态为："
            f"完成 {complete} 个、未找到满足 SLA 的候选 {failed} 个、不支持 {unsupported} 个。"
        ),
        "",
    ]
    if no_success_models:
        lines.extend(
            [
                "在当前搜索边界内，以下模型没有任何完成的 Cell："
                + "、".join(no_success_models)
                + "。这只表示本轮有限搜索没有找到满足 SLA 的配置，不等于这些组合不可部署。",
                "",
            ]
        )

    lines.extend(
        [
            "## 模型 Variant 与 AIC 精度契约",
            "",
            (
                "每个 Cell 固定一个 checkpoint/精度 variant；以下五个 AIC 字段会同时进入"
                "静态内存剪枝、KV footprint 和 DynoSim replay。自动替换只允许由 VariantResolver "
                "在硬件/Kernel 精度不支持时按不降精度规则显式记录。"
            ),
            "",
            _markdown_table(
                [
                    "模型",
                    "Variant",
                    "权重精度标签",
                    "GEMM",
                    "MoE",
                    "FMHA",
                    "KV cache",
                    "通信",
                    "支持级别",
                    "解析来源",
                    "AIC 支持状态",
                    "部署属性",
                ],
                variant_rows,
            ),
            "",
        ]
    )

    sla_pass = pareto.get("sla_pass", False).astype(str).str.lower().isin({"true", "1"})
    strict_pareto = pareto.loc[
        pareto.get("strict_pareto", False).astype(str).str.lower().isin({"true", "1"})
        & sla_pass
    ].sort_values(["throughput_per_user", "goodput_per_gpu"])
    pareto_rows = [
        (
            _label(row["model_id"]),
            row["model_variant_id"],
            _label(row["sku_id"]),
            _workload_label(row["workload_id"]),
            _fmt(row["throughput_per_user"], 3),
            _fmt(row["goodput_per_gpu"], 3),
            _fmt(row["used_gpus"], 0),
            "—"
            if pd.isna(row.get("pd_bandwidth_domain"))
            else row.get("pd_bandwidth_domain"),
            _fmt(row.get("pd_transfer_bandwidth_gbytes_per_s"), 1),
            "aggregate"
            if pd.isna(row.get("pd_profile_id"))
            else row.get("pd_profile_id"),
            "—" if pd.isna(row.get("kvbm_profile_id")) else row.get("kvbm_profile_id"),
        )
        for _, row in strict_pareto.head(40).iterrows()
    ]
    diagnostic_mask = pd.Series(False, index=pareto.index)
    for _, group in pareto.dropna(
        subset=["throughput_per_user", "raw_throughput_per_gpu"]
    ).groupby(["model_id", "model_variant_id", "sku_id", "workload_id"], sort=True):
        diagnostic_mask.loc[group.index] = _diagnostic_frontier_mask(group)
    diagnostic_front = (
        pareto.loc[diagnostic_mask]
        .sort_values(["throughput_per_user", "raw_throughput_per_gpu"])
        .drop_duplicates(
            subset=[
                "model_id",
                "model_variant_id",
                "sku_id",
                "workload_id",
                "throughput_per_user",
                "raw_throughput_per_gpu",
                "mean_ttft_ms",
                "mean_tpot_ms",
            ]
        )
    )
    diagnostic_rows = [
        (
            _label(row["model_id"]),
            row["model_variant_id"],
            _label(row["sku_id"]),
            _workload_label(row["workload_id"]),
            _fmt(row["throughput_per_user"], 3),
            _fmt(row["raw_throughput_per_gpu"], 3),
            "是" if str(row.get("sla_pass", "")).lower() in {"true", "1"} else "否",
            _fmt(row["mean_ttft_ms"], 1),
            _fmt(row["mean_tpot_ms"], 2),
        )
        for _, row in diagnostic_front.head(40).iterrows()
    ]
    lines.extend(
        [
            "## 两阶段负载搜索口径",
            "",
            (
                f"AIC 静态 sweep 记录 {len(operating_points)} 个可审计 operating point，"
                f"其中 {int(operating_points.get('selected', pd.Series(dtype=bool)).astype(str).str.lower().isin({'true', '1'}).sum())} 个并发点进入闭环 Spica。"
                f"DynoSim 保存 {len(load_front)} 个分负载严格 Pareto 成员和 "
                f"{len(global_envelope)} 个跨负载 operating-envelope 成员；"
                f"最终开放环 RPS 认证保留 {len(certification)} 个候选。"
            ),
            "",
            (
                "AIC 只负责从并行配置与 batch/context 静态笛卡尔积中生成 concurrency；"
                "每个 concurrency 下的 router、PD、KVBM 和部署 Pareto 由 DynoSim + Spica 搜索，"
                "最终候选再通过开放环 RPS 双向扩张/二分与正式 SLA 认证。"
            ),
            "",
            "## Pareto 曲线（中间结果）",
            "",
            f"![正式 SLA 与诊断 Pareto 曲线]({figure_paths['pareto_front']})",
            "",
            f"![分并发 Pareto 与跨负载包络]({figure_paths['load_conditioned_pareto']})",
            "",
            (
                "横轴是每用户输出吞吐，纵轴是每 GPU 的输出吞吐；两者均为越高越好。"
                "灰点是 replay 成功的最终阶段候选，蓝点满足正式 SLA，红线是正式 SLA 严格 Pareto，"
                "橙色虚线是原始吞吐诊断前沿。橙线用于观察搜索边界，不能当作满足 SLA 的候选。"
            ),
            "",
            _markdown_table(
                [
                    "模型",
                    "Variant",
                    "SKU",
                    "负载",
                    "TPS/u",
                    "goodput/GPU",
                    "GPU 数",
                    "PD 带宽域",
                    "PD GB/s",
                    "PD profile",
                    "KVBM profile",
                ],
                pareto_rows,
            ),
            "",
            "原始吞吐诊断前沿：",
            "",
            _markdown_table(
                [
                    "模型",
                    "Variant",
                    "SKU",
                    "负载",
                    "TPS/u",
                    "输出吞吐/GPU",
                    "满足 SLA",
                    "TTFT ms",
                    "TPOT ms",
                ],
                diagnostic_rows,
            ),
            "",
        ]
    )

    lines.extend(
        [
            "## 1. 不同模型与 SKU 的每百万 token 成本",
            "",
            f"![不同模型与 SKU 的成本对比]({figure_paths['model_sku_cost']})",
            "",
            _markdown_table(
                [
                    "模型",
                    "Variant",
                    "负载",
                    "上下文",
                    "SKU",
                    f"全量成本（{report_currency}/M）",
                    "SLA RPS",
                    "GPU 数",
                    "候选拓扑",
                ],
                best_rows,
            ),
            "",
        ]
    )
    if paired_findings:
        lines.extend(["同模型、同负载下的可比 SKU 结论：", ""])
        lines.extend(f"- {finding}" for finding in paired_findings)
        lines.append("")

    lines.extend(
        [
            "## 2. TPS/u 下限变化下的成本包络",
            "",
            f"![TPS/u 下限与成本包络]({figure_paths['tpsu_cost_envelope']})",
            "",
            (
                "曲线表示在每个 TPS/u 下限上，所有达到该下限的候选中成本最低的配置。曲线的跳点意味着"
                "原配置无法继续满足更高 TPS/u，需要切换 GPU 数、并行策略、实例数或路由配置。"
            ),
            "",
        ]
    )
    if not tpsu.empty:
        envelope_rows = []
        for (model, variant, workload, sku), group in tpsu.groupby(
            ["model_id", "model_variant_id", "workload_id", "sku_id"],
            sort=True,
        ):
            points = group.sort_values("required_tps_per_user")
            for position in sorted({0, len(points) - 1}):
                point = points.iloc[position]
                envelope_rows.append(
                    (
                        _label(model),
                        variant,
                        _workload_label(workload),
                        _label(sku),
                        _fmt(point["required_tps_per_user"], 2),
                        _fmt(point["all_in_cost_per_million_total_tokens"], 4),
                    )
                )
        lines.extend(
            [
                _markdown_table(
                    [
                        "模型",
                        "Variant",
                        "负载",
                        "SKU",
                        "TPS/u 下限",
                        f"最低成本（{report_currency}/M）",
                    ],
                    envelope_rows,
                ),
                "",
            ]
        )
    else:
        lines.extend(["_无满足正式 SLA 的成本包络数据。_", ""])

    lines.extend(
        [
            "## 3. 上下文长度变化下的成本趋势",
            "",
            f"![上下文长度与成本趋势]({figure_paths['context_cost']})",
            "",
        ]
    )
    lines.extend(f"- {finding}" for finding in context_findings)
    if context_findings:
        lines.append("")
    else:
        lines.extend(["_无满足正式 SLA 的上下文趋势数据。_", ""])
    lines.extend(
        [
            (
                "该图比较的是完整 workload 场景，而不是严格的单变量上下文实验。若不同场景同时改变输入长度、"
                "输出长度、prefix 或初始 RPS，成本变化不能只归因于上下文长度。"
            ),
            "",
            "## 4. KV 存储容量与命中率的收益",
            "",
            f"![KV 容量和命中率收益]({figure_paths['kv_capacity_uplift']})",
            "",
            (
                f"共有 {matched_capacity} 个候选具备相同计算与路由配置的 G2-off 精确基线。"
                f"匹配配置成本降幅范围为 **{_ratio_range(capacity, 'matched_config_cost_reduction_ratio')}**；"
                f"其中 {int(matched_rps_positive)} 个候选提高了匹配配置的 SLA RPS。"
                f"从系统最优口径看，成本改善候选 {int(system_cost_positive)} 个，RPS 改善候选 "
                f"{int(system_rps_positive)} 个。负的“降幅”表示启用 KVBM 后反而更贵。"
            ),
            "",
            (
                "图中横轴优先使用仿真实测 prefix-cache hit ratio，点旁标出 G2 容量。"
                "目标命中率只用于定义配置，研判时应以实测命中率为主。"
            ),
            "",
            "## 5. KV 存储带宽与 compute NIC 带宽比例的收益",
            "",
            f"![KV 带宽比例收益]({figure_paths['kv_bandwidth_uplift']})",
            "",
            (
                f"匹配配置成本降幅范围为 **{_ratio_range(bandwidth, 'matched_config_cost_reduction_ratio')}**；"
                f"匹配配置 RPS 增幅范围为 **{_ratio_range(bandwidth, 'matched_config_rps_increase_ratio')}**。"
            ),
            "",
        ]
    )
    if coupled_bandwidth:
        lines.extend(
            [
                (
                    "本轮不同带宽比例还绑定了不同的 G2 容量或目标命中率，因此这是组合场景对比，"
                    "不能作为带宽的单变量因果曲线。后续应对每个固定容量/命中率分别交叉全部带宽比例。"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## KV 字节量口径",
            "",
            (
                "候选的 KV footprint 来自 AIC 原生 `get_kvcache_bytes_per_sequence()`，作用域为每 GPU rank。"
                "KVBM block 字节和 P/D handoff 字节使用独立 scalar；P/D 计时按 decode 物化量计算理想均衡"
                "reshard/fan-out 的最忙 GPU 链路，物理网络用量另按 decode 端所有 TP rank 的物化总量估算。"
            ),
            "",
            _markdown_table(
                [
                    "模型",
                    "Variant",
                    "角色",
                    "布局",
                    "dtype",
                    "增长类型",
                    "TP",
                    "参考长度",
                    "固定 bytes/rank",
                    "参考 bytes/rank",
                    "KVBM bytes/token/rank",
                    "PD bytes/token/critical link",
                    "KVBM 口径",
                ],
                [
                    (
                        _label(row["model_id"]),
                        row["model_variant_id"],
                        row["role"],
                        row["layout"],
                        row["dtype"],
                        row["growth_kind"],
                        _fmt(row["tp"], 0),
                        _fmt(row["reference_length"], 0),
                        _fmt(row["fixed_bytes_per_sequence_per_rank"], 0),
                        _fmt(row["reference_bytes_per_sequence_per_rank"], 0),
                        _fmt(row["kvbm_bytes_per_token_per_rank"], 0),
                        _fmt(row["pd_transfer_bytes_per_token_critical_link"], 0),
                        row["kvbm_footprint_contract"]
                        if pd.notna(row["kvbm_footprint_contract"])
                        else "—",
                    )
                    for _, row in footprints.drop_duplicates(
                        subset=[
                            "model_id",
                            "model_variant_id",
                            "role",
                            "layout",
                            "dtype",
                            "growth_kind",
                            "tp",
                            "reference_length",
                            "fixed_bytes_per_sequence_per_rank",
                            "reference_bytes_per_sequence_per_rank",
                            "kvbm_bytes_per_token_per_rank",
                            "pd_transfer_bytes_per_token_critical_link",
                            "kvbm_footprint_contract",
                        ]
                    )
                    .head(60)
                    .iterrows()
                ],
            ),
            "",
            (
                "对非线性或带每请求固定 state 的模型，固定长度 Synthetic 的 sequence-average 只保证该 ISL 下"
                "每请求总字节一致，KVBM 单个 block 的位置相关传输时间仍是近似；异构 trace 和非线性 "
                "`destination_missing` 会 fail closed。"
            ),
            "",
            "## 定价口径",
            "",
            _markdown_table(
                [
                    "SKU",
                    "币种",
                    "GPU·小时",
                    "G2 GiB·小时",
                    "网络 GB",
                    "口径",
                    "来源日期",
                    "链接",
                ],
                price_rows,
            ),
            "",
            (
                "价格仅在报告阶段应用，不进入仿真 fingerprint；替换价格 catalog 后可直接重新定价，无需重跑搜索。"
                "标为“代理值”的 SKU 不能作为采购报价。"
            ),
            "",
            "## 结论边界",
            "",
            "- 本报告只对当前 catalog、搜索策略、GPU 上限、workload 和 SLA 生效；未找到候选不等于不可部署。",
            "- 容量搜索若标记为 right-censored，记录的最大 SLA RPS 是当前扩张上限内的下界。",
            "- DynoSim + AIC 结果是仿真估计；用于容量承诺或采购决策前，需要用目标软件版本和实机校准。",
            "- 若运行日志出现 AIC 低保真回退（low-fidelity fallback），相应模型/SKU 数据点只能视为初筛结果。",
            "- 默认 Pareto/TopK 和五类趋势只推荐 deployable；hypothetical 候选单独保存在 what-if Pareto。",
            "- KVBM 的 matched-config 与 system-optimal 是不同问题：前者隔离相同计算配置，后者允许全局重新选型。",
            "- 原始候选、失败原因、Pareto 标记和资源量保留在 DuckDB；CSV 是面向五类趋势的派生视图。",
            "",
            "## 产物索引",
            "",
            "- `sweep.duckdb`：原始评估、候选、资源用量、Pareto 与 TopK 数据库。",
            "- `aic_operating_points.csv`：AIC 原始/选中并发点与选择原因。",
            "- `load_conditioned_pareto.csv`：每个固定并发下的严格 Pareto。",
            "- `global_operating_envelope.csv`：跨并发 operating envelope。",
            "- `capacity_certification.csv`：入围配置的开放环 RPS 上下界和截尾状态。",
            "- `cost_by_model_sku.csv`：模型与 SKU 成本视图。",
            "- `cost_by_tpsu_requirement.csv`：TPS/u 下限成本包络。",
            "- `cost_by_context_sku.csv`：上下文场景成本视图。",
            "- `kv_capacity_impact.csv`：KV 容量/命中率收益视图。",
            "- `kv_bandwidth_impact.csv`：KV 带宽比收益视图。",
            "- `kv_footprint_audit.csv`：AIC 原生 KV 曲线、scope、dtype 与 scalar 化口径。",
            "- `cell_status.csv`：全部 Cell 的完成、失败或不支持状态。",
            "- `variant_resolution.csv`、`variant_resolution.md`：精度选择、支持探测及淘汰原因。",
            "- `what_if_pareto_front.csv`：不进入默认推荐的 hypothetical 候选 Pareto。",
            "- `manifest.json`、`report.json`、`price_catalog_used.csv`：代码、输入、binding 和价格来源记录。",
            "",
        ]
    )

    report_path = output_dir / "preliminary_findings.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def generate_visual_report(
    output_dir: str | Path,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build final/load Pareto plus five trend PNGs and a Chinese report."""
    output_dir = Path(output_dir).expanduser().resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "pareto": _read_csv(output_dir, "pareto_front.csv"),
        "costs": _read_csv(output_dir, "cost_by_model_sku.csv"),
        "tpsu": _read_csv(output_dir, "cost_by_tpsu_requirement.csv"),
        "context": _read_csv(output_dir, "cost_by_context_sku.csv"),
        "capacity": _read_csv(output_dir, "kv_capacity_impact.csv"),
        "bandwidth": _read_csv(output_dir, "kv_bandwidth_impact.csv"),
        "statuses": _read_csv(output_dir, "cell_status.csv"),
        "prices": _read_csv(output_dir, "price_catalog_used.csv"),
        "footprints": _read_csv(output_dir, "kv_footprint_audit.csv"),
        "operating_points": _read_csv(output_dir, "aic_operating_points.csv"),
        "load_front": _read_csv(output_dir, "load_conditioned_pareto.csv"),
        "global_envelope": _read_csv(output_dir, "global_operating_envelope.csv"),
        "certification": _read_csv(output_dir, "capacity_certification.csv"),
    }
    figure_paths = {key: f"figures/{filename}" for key, filename in FIGURES}
    _plot_pareto_front(frames["pareto"], output_dir / figure_paths["pareto_front"])
    _plot_load_conditioned_pareto(
        frames["load_front"],
        frames["global_envelope"],
        output_dir / figure_paths["load_conditioned_pareto"],
    )
    _plot_model_sku_cost(frames["costs"], output_dir / figure_paths["model_sku_cost"])
    _plot_tpsu_cost(frames["tpsu"], output_dir / figure_paths["tpsu_cost_envelope"])
    _plot_context_cost(frames["context"], output_dir / figure_paths["context_cost"])
    _plot_kv_capacity(
        frames["capacity"], output_dir / figure_paths["kv_capacity_uplift"]
    )
    _plot_kv_bandwidth(
        frames["bandwidth"], output_dir / figure_paths["kv_bandwidth_uplift"]
    )
    report_path = _write_chinese_markdown(output_dir, summary, frames, figure_paths)
    return {
        "figures": list(figure_paths.values()),
        "markdown_report": report_path.name,
    }
