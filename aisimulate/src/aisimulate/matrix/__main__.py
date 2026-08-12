# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for reusable matrix experiments."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from pydantic import ValidationError

from .catalog import load_catalogs
from .config import MatrixExperimentConfig
from .expand import expand_matrix, resolve_matrix_variants, validate_experiment
from .report import generate_report
from .runner import MatrixRunner
from .store import ResultStore
from .variant import support_accepted


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aisimulate.matrix",
        description="Catalog-driven DynoSim + AIConfigurator matrix search",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "resolve-variants", "expand"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument(
        "--reuse-from",
        action="append",
        default=[],
        metavar="RUN_DIR_OR_DUCKDB",
        help="import exact replay cache entries from another run (repeatable)",
    )
    report = subparsers.add_parser("report")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--price-catalog")
    report.add_argument("--topk", type=int, default=20)
    return parser


def _load_config(parser: argparse.ArgumentParser, path: str) -> MatrixExperimentConfig:
    try:
        return MatrixExperimentConfig.from_file(path)
    except (OSError, ValueError, ValidationError) as exc:
        parser.error(f"invalid experiment config {path}: {exc}")


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.command == "report":
        with ResultStore.open(args.run_dir) as store:
            summary = generate_report(
                store, price_catalog=args.price_catalog, topk=args.topk
            )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    config = _load_config(parser, args.config)
    if args.command == "validate":
        report = validate_experiment(config)
        print(report.model_dump_json(indent=2))
        if not report.valid:
            sys.exit(1)
        return
    if args.command in {"resolve-variants", "expand"}:
        catalogs = load_catalogs(config)
        resolution = resolve_matrix_variants(config, catalogs)
        if args.command == "resolve-variants":
            payload = {
                "data_mode": config.matrix.variant_resolution.data_mode,
                "selected_count": len(resolution.selected_records),
                "record_count": len(resolution.records),
                "records": [
                    record.model_dump(mode="python") for record in resolution.records
                ],
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return
        cells = expand_matrix(config, catalogs, resolution=resolution)
        estimated_substudies = 0
        estimated_static_candidates = 0
        estimated_spica_trials = 0
        estimated_capacity_replays = 0
        estimated_robustness_replays = 0
        cell_payloads = []
        for cell in cells:
            support = cell.variant_resolution.support
            runnable = support_accepted(support, config.matrix.variant_resolution)
            if cell.workload.load_axis is None:
                load_points = 1
            elif cell.workload.load_axis.source == "aic_operating_points":
                # The exact count is intentionally unknown until the AIC
                # static sweep runs. This is a safe configured upper bound.
                load_points = int(cell.workload.load_axis.max_points or 1)
            else:
                load_points = len(cell.workload.load_axis.values)
            profile_branches = 0
            if (
                "agg" in cell.search_policy.spica.deployment_mode
                and cell.workload.capabilities.aggregate
            ):
                profile_branches += 1
            if (
                "disagg" in cell.search_policy.spica.deployment_mode
                and cell.workload.capabilities.disaggregate
            ):
                profile_branches += len(cell.search_policy.pd_profiles)
            substudies = (
                load_points * profile_branches * len(cell.search_policy.kvbm_profiles)
                if runnable
                else 0
            )
            trials_per_study = cell.search_policy.spica.max_rounds * (
                cell.search_policy.spica.candidates_per_round
                or cell.search_policy.spica.parallel_evals
            )
            spica_trials = substudies * trials_per_study
            robustness_multiplier = (
                len(cell.search_policy.robustness.seeds)
                * len(cell.search_policy.robustness.bandwidth_multipliers)
                if cell.search_policy.robustness.enabled
                else 0
            )
            capacity = cell.search_policy.capacity_search
            if capacity.enabled and capacity.stage == "finalists" and runnable:
                # Every enabled KVBM finalist can create one exact KVBM-off
                # matched baseline, so 2x is a conservative qualification cap.
                qualification_candidates = capacity.max_finalists_per_cell * 2
                probes_per_candidate = (
                    1 + capacity.max_expansions + capacity.max_binary_steps
                )
                capacity_replays = qualification_candidates * probes_per_candidate
                robustness_replays = (
                    qualification_candidates * robustness_multiplier
                )
            else:
                capacity_replays = (
                    spica_trials
                    * (1 + capacity.max_expansions + capacity.max_binary_steps)
                    if capacity.enabled and runnable
                    else 0
                )
                robustness_replays = spica_trials * robustness_multiplier
            estimated_substudies += substudies
            estimated_static_candidates += (
                substudies * cell.search_policy.static_prune.max_candidates
            )
            estimated_spica_trials += spica_trials
            estimated_capacity_replays += capacity_replays
            estimated_robustness_replays += robustness_replays
            cell_payloads.append(
                {
                    "id": cell.id,
                    "model": cell.model.id,
                    "model_variant": cell.model_variant.id,
                    "variant_resolution_source": cell.variant_resolution.source,
                    "variant_support_status": cell.variant_resolution.support.status,
                    "deployment_fidelity": cell.model_variant.deployment_fidelity,
                    "runnable": runnable,
                    "sku": cell.sku.id,
                    "workload": cell.workload.id,
                    "fingerprint": cell.fingerprint,
                    "estimated_substudies": substudies,
                    "estimated_spica_trials_upper_bound": spica_trials,
                    "estimated_capacity_replays_upper_bound": capacity_replays,
                    "estimated_robustness_replays_upper_bound": robustness_replays,
                    "load_point_estimate_kind": (
                        "configured_upper_bound"
                        if cell.workload.load_axis is not None
                        and cell.workload.load_axis.source == "aic_operating_points"
                        else "exact"
                    ),
                }
            )
        payload: dict[str, Any] = {
            "cell_count": len(cells),
            "estimated_substudies": estimated_substudies,
            "estimated_static_candidates_upper_bound": estimated_static_candidates,
            "estimated_spica_trials_upper_bound": estimated_spica_trials,
            "estimated_capacity_replays_upper_bound": estimated_capacity_replays,
            "estimated_robustness_replays_upper_bound": estimated_robustness_replays,
            "cells": cell_payloads,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    with ResultStore.open(args.output_dir) as store:
        summary = MatrixRunner().run(
            config,
            store,
            resume=args.resume,
            reuse_from=args.reuse_from,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
