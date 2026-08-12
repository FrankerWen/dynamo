#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed integrity check for a completed matrix run and its reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import duckdb

CSV_REPORTS = (
    "pareto_front.csv",
    "aic_operating_points.csv",
    "load_conditioned_pareto.csv",
    "global_operating_envelope.csv",
    "capacity_certification.csv",
    "cost_by_model_sku.csv",
    "cost_by_tpsu_requirement.csv",
    "cost_by_context_sku.csv",
    "kv_capacity_impact.csv",
    "kv_bandwidth_impact.csv",
    "kv_footprint_audit.csv",
)
FIGURES = (
    "00_pareto_front.png",
    "00a_load_conditioned_pareto.png",
    "01_model_sku_cost.png",
    "02_tpsu_cost_envelope.png",
    "03_context_cost.png",
    "04_kv_capacity_uplift.png",
    "05_kv_bandwidth_uplift.png",
)


def _row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _true_row_count(path: Path, column: str) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(
            str(row.get(column, "")).lower() in {"true", "1"}
            for row in csv.DictReader(handle)
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-cells", type=int)
    parser.add_argument("--model")
    parser.add_argument("--model-variant")
    parser.add_argument("--sku")
    parser.add_argument("--workload")
    parser.add_argument(
        "--require-pareto",
        action="store_true",
        help="fail when the formal SLA produced no Pareto member",
    )
    parser.add_argument(
        "--require-curve-data",
        action="store_true",
        help="fail when neither formal nor diagnostic frontier has any point",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    database = run_dir / "sweep.duckdb"
    required = [
        database,
        run_dir / "manifest.json",
        run_dir / "report.json",
        run_dir / "preliminary_findings.md",
        run_dir / "price_catalog_used.csv",
        *(run_dir / name for name in CSV_REPORTS),
        *(run_dir / "figures" / name for name in FIGURES),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing run artifacts:\n" + "\n".join(missing))

    connection = duckdb.connect(str(database), read_only=True)
    cells = connection.execute(
        """
        SELECT model_id, model_variant_id, sku_id, workload_id, status, count(*)
        FROM matrix_cells
        GROUP BY ALL
        ORDER BY ALL
        """
    ).fetchall()
    evaluation_count = connection.execute(
        "SELECT count(*) FROM evaluations"
    ).fetchone()[0]
    candidate_count = connection.execute("SELECT count(*) FROM candidates").fetchone()[
        0
    ]
    duplicate_keys = connection.execute(
        """
        SELECT count(*) FROM (
          SELECT cell_id, evaluation_key
          FROM evaluations
          GROUP BY ALL HAVING count(*) > 1
        )
        """
    ).fetchone()[0]
    incomplete = [row for row in cells if row[4] in {"pending", "running"}]
    connection.close()

    actual_cells = sum(row[5] for row in cells)
    if args.expected_cells is not None and actual_cells != args.expected_cells:
        raise SystemExit(f"expected {args.expected_cells} cells, got {cells!r}")
    for index, expected, label in (
        (0, args.model, "model"),
        (1, args.model_variant, "model variant"),
        (2, args.sku, "sku"),
        (3, args.workload, "workload"),
    ):
        if expected is not None and {row[index] for row in cells} != {expected}:
            raise SystemExit(f"unexpected {label} rows: {cells!r}")
    if incomplete:
        raise SystemExit(f"run still has incomplete cells: {incomplete!r}")
    if evaluation_count == 0 or duplicate_keys:
        raise SystemExit(
            f"invalid evaluation store: count={evaluation_count}, duplicate_keys={duplicate_keys}"
        )

    csv_rows = {name: _row_count(run_dir / name) for name in CSV_REPORTS}
    formal_pareto_rows = _true_row_count(run_dir / "pareto_front.csv", "strict_pareto")
    if args.require_curve_data and csv_rows["pareto_front.csv"] == 0:
        raise SystemExit("Pareto CSV has no formal or diagnostic candidate data")
    if args.require_pareto and formal_pareto_rows == 0:
        raise SystemExit("formal SLA produced no Pareto candidates")
    if args.require_curve_data:
        for name in (
            "aic_operating_points.csv",
            "load_conditioned_pareto.csv",
            "global_operating_envelope.csv",
            "capacity_certification.csv",
        ):
            if csv_rows[name] == 0:
                raise SystemExit(f"two-stage search report has no rows: {name}")

    markdown = (run_dir / "preliminary_findings.md").read_text(encoding="utf-8")
    for heading in (
        "## Pareto 曲线（中间结果）",
        "## 1. 不同模型与 SKU 的每百万 token 成本",
        "## 2. TPS/u 下限变化下的成本包络",
        "## 3. 上下文长度变化下的成本趋势",
        "## 4. KV 存储容量与命中率的收益",
        "## 5. KV 存储带宽与 compute NIC 带宽比例的收益",
        "## KV 字节量口径",
    ):
        if heading not in markdown:
            raise SystemExit(f"missing Chinese report section: {heading}")

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "cells": cells,
                "evaluations": evaluation_count,
                "candidates": candidate_count,
                "csv_rows": csv_rows,
                "formal_pareto_rows": formal_pareto_rows,
                "figures": len(FIGURES),
                "status": "PASS",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
