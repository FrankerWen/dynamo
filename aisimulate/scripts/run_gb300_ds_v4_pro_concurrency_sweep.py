# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run independent fixed-concurrency Sweeper studies and build a Pareto envelope."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aisimulate.sweeper import (
    Candidate,
    LoadSweepResult,
    SmartSearchConfig,
    Sweeper,
)
from aisimulate.sweeper.score import pareto_front
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = SCRIPT_DIR / "gb300_ds_v4_pro_kv_ratio_sweep.yaml"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT.parent / "results" / "gb300-ds-v4-pro-concurrency-sweep"
)
DEFAULT_CONCURRENCIES = (1, 8, 32, 64, 256, 512, 1024, 4096, 8192)
DEFAULT_MAX_TOTAL_PARALLEL_EVALS = 32
POINT_SCHEMA = "aisimulate.fixed_concurrency_pareto_point.v1"
DATABASE_SCHEMA = "aisimulate.concurrency_pareto_database.v1"
ENVELOPE_SCHEMA = "aisimulate.global_pareto_envelope.v1"
MANIFEST_SCHEMA = "aisimulate.concurrency_sweep_manifest.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _point_path(output_dir: Path, concurrency: int) -> Path:
    return output_dir / "points" / f"concurrency_{concurrency:05d}.json"


def _checkpoint_path(output_dir: Path, concurrency: int) -> Path:
    return output_dir / "checkpoints" / f"concurrency_{concurrency:05d}.json"


def _validate_template(config: SmartSearchConfig) -> None:
    workload = config.workload
    if not config.goal.is_pareto:
        raise ValueError("the concurrency sweep template requires goal.target='pareto'")
    if len(config.goal.resolved_pareto_objectives) != 2:
        raise ValueError("the final 2D Pareto plot requires exactly two objectives")
    if not workload.is_synthetic:
        raise ValueError("the concurrency sweep template must use a synthetic workload")
    if workload.concurrency is None:
        raise ValueError("the concurrency sweep template must pin workload.concurrency")
    if workload.kv_load_ratio is not None or workload.request_rate is not None:
        raise ValueError(
            "the concurrency sweep template must not set kv_load_ratio or request_rate"
        )
    if config.load_sweep is not None:
        raise ValueError("the concurrency sweep is owned by this outer script")


def _config_for_concurrency(
    template: SmartSearchConfig, concurrency: int
) -> SmartSearchConfig:
    workload = template.workload.model_copy(update={"concurrency": concurrency})
    return SmartSearchConfig.model_validate(
        template.model_copy(
            update={"workload": workload, "load_sweep": None}
        ).model_dump(mode="python")
    )


def _effective_inner_parallelism(config: SmartSearchConfig) -> int:
    candidates_per_round = (
        config.sweep.candidates_per_round or config.sweep.parallel_evals
    )
    return min(config.sweep.parallel_evals, candidates_per_round)


def _outer_parallelism(
    point_count: int,
    *,
    inner_parallelism: int,
    max_total_parallel_evals: int,
) -> int:
    if max_total_parallel_evals < inner_parallelism:
        raise ValueError(
            "--max-total-parallel-evals must be at least the effective inner "
            f"parallelism ({inner_parallelism})"
        )
    return min(point_count, max_total_parallel_evals // inner_parallelism)


def _candidate_with_concurrency(
    candidate: Candidate, concurrency: int
) -> Candidate:
    config = dict(candidate.config)
    config["concurrency"] = concurrency
    return candidate.model_copy(update={"config": config})


def _run_fixed_concurrency(
    point_config: SmartSearchConfig,
    concurrency: int,
    runner_factory_type: type[Any],
    checkpoint_path: Path,
    resume_checkpoint: bool,
) -> list[Candidate]:
    sweeper = Sweeper(runner_factory=runner_factory_type(), show_progress=False)

    def report_round(round_number: int, candidates: list[Candidate]) -> None:
        tqdm.write(
            f"progress concurrency={concurrency}: round={round_number}, "
            f"candidates={len(candidates)}",
            file=sys.stdout,
        )

    result = sweeper.run(
        point_config,
        on_round=report_round,
        checkpoint_path=checkpoint_path,
        resume_checkpoint=resume_checkpoint,
    )
    if isinstance(result, LoadSweepResult):
        raise TypeError("inner fixed-concurrency sweep returned LoadSweepResult")
    return [
        _candidate_with_concurrency(candidate, concurrency)
        for candidate in result
    ]


def _load_completed_point(
    path: Path,
    concurrency: int,
    expected_config: SmartSearchConfig,
) -> list[Candidate] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("schema") != POINT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("concurrency") != concurrency
        or payload.get("config") != expected_config.model_dump(mode="json")
    ):
        return None
    try:
        return [
            _candidate_with_concurrency(
                Candidate.model_validate(candidate), concurrency
            )
            for candidate in payload["pareto_front"]
        ]
    except (KeyError, TypeError, ValueError):
        return None


def _candidate_id(concurrency: int, rank: int, candidate: Candidate) -> str:
    canonical = json.dumps(
        {
            "concurrency": concurrency,
            "load_rank": rank,
            "candidate": candidate.model_dump(mode="json"),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _flatten_fronts(
    fronts: dict[int, list[Candidate]], global_front: list[Candidate]
) -> list[dict[str, Any]]:
    global_ranks = {
        id(candidate): rank for rank, candidate in enumerate(global_front)
    }
    rows: list[dict[str, Any]] = []
    for concurrency, candidates in sorted(fronts.items()):
        for rank, candidate in enumerate(candidates):
            rows.append(
                {
                    "candidate_id": _candidate_id(concurrency, rank, candidate),
                    "concurrency": concurrency,
                    "load_rank": rank,
                    "global_envelope": id(candidate) in global_ranks,
                    "global_rank": global_ranks.get(id(candidate)),
                    "candidate": candidate.model_dump(mode="json"),
                }
            )
    return rows


def _csv_row(
    row: dict[str, Any], objective_names: list[str]
) -> dict[str, object]:
    candidate = row["candidate"]
    config = candidate["config"]
    metrics = candidate["metrics"]
    objectives = candidate.get("objectives") or {}
    output: dict[str, object] = {
        "candidate_id": row["candidate_id"],
        "concurrency": row["concurrency"],
        "load_rank": row["load_rank"],
        "global_envelope": row["global_envelope"],
        "global_rank": (
            "" if row["global_rank"] is None else row["global_rank"]
        ),
        "used_gpus": candidate["used_gpus"],
        "score": candidate["score"],
        "deployment_mode": config.get("deployment_mode", ""),
        "backend": config.get("backend", ""),
        "backend_version": config.get("backend_version", ""),
        "config_json": json.dumps(config, sort_keys=True, separators=(",", ":")),
        "metrics_json": json.dumps(metrics, sort_keys=True, separators=(",", ":")),
        "objectives_json": json.dumps(
            objectives, sort_keys=True, separators=(",", ":")
        ),
    }
    for name in objective_names:
        output[name] = objectives.get(name, "")
    return output


def _write_csv(
    path: Path, rows: list[dict[str, Any]], objective_names: list[str]
) -> None:
    fieldnames = [
        "candidate_id",
        "concurrency",
        "load_rank",
        "global_envelope",
        "global_rank",
        "used_gpus",
        "score",
        *objective_names,
        "deployment_mode",
        "backend",
        "backend_version",
        "config_json",
        "metrics_json",
        "objectives_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_row(row, objective_names))
    temporary.replace(path)


def _plot_envelope(
    path: Path,
    rows: list[dict[str, Any]],
    global_rows: list[dict[str, Any]],
    objective_names: list[str],
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError as exc:
        raise RuntimeError(
            "plotting requires matplotlib in the active Python environment"
        ) from exc

    y_name, x_name = objective_names[0], objective_names[1]

    def values(record: dict[str, Any]) -> tuple[float, float]:
        objectives = record["candidate"].get("objectives") or {}
        return float(objectives[x_name]), float(objectives[y_name])

    plotted_rows = []
    for row in rows:
        try:
            x_value, y_value = values(row)
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x_value) and math.isfinite(y_value):
            plotted_rows.append((row, x_value, y_value))
    if not plotted_rows:
        raise ValueError("no finite Pareto objective points are available to plot")

    concurrencies = [float(row["concurrency"]) for row, _, _ in plotted_rows]
    minimum = min(concurrencies)
    maximum = max(concurrencies)
    norm = LogNorm(vmin=minimum, vmax=maximum) if minimum < maximum else None

    figure, axis = plt.subplots(figsize=(11, 7), dpi=160)
    scatter = axis.scatter(
        [x_value for _, x_value, _ in plotted_rows],
        [y_value for _, _, y_value in plotted_rows],
        c=concurrencies,
        cmap="viridis",
        norm=norm,
        s=38,
        alpha=0.55,
        label="Load-conditioned Pareto points",
        zorder=2,
    )
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Fixed concurrency")

    envelope_values = []
    for row in global_rows:
        try:
            x_value, y_value = values(row)
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x_value) and math.isfinite(y_value):
            envelope_values.append((row, x_value, y_value))
    envelope_values.sort(key=lambda item: item[1])
    axis.plot(
        [x_value for _, x_value, _ in envelope_values],
        [y_value for _, _, y_value in envelope_values],
        color="#d62728",
        marker="o",
        markersize=5,
        linewidth=2,
        label="Global Pareto envelope",
        zorder=4,
    )
    for row, x_value, y_value in envelope_values:
        axis.annotate(
            f"c={row['concurrency']}",
            (x_value, y_value),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=7,
        )

    axis.set_title(title)
    axis.set_xlabel(x_name.replace("_", " "))
    axis.set_ylabel(y_name.replace("_", " "))
    axis.grid(True, linestyle=":", alpha=0.5)
    axis.legend(loc="best")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def write_pareto_artifacts(
    output_dir: Path,
    fronts: dict[int, list[Candidate]],
    config: SmartSearchConfig,
    *,
    plot: bool,
) -> list[Candidate]:
    objective_targets = config.goal.resolved_pareto_objectives
    objective_names = [target.value for target in objective_targets]
    all_candidates = [candidate for front in fronts.values() for candidate in front]
    global_front = pareto_front(all_candidates, objective_targets)
    rows = _flatten_fronts(fronts, global_front)
    global_rows = sorted(
        (row for row in rows if row["global_envelope"]),
        key=lambda row: int(row["global_rank"]),
    )

    load_database = {
        "schema": DATABASE_SCHEMA,
        "generated_at": _utc_now(),
        "objective_names": objective_names,
        "concurrency_values": sorted(fronts),
        "load_point_count": len(fronts),
        "candidate_count": len(rows),
        "load_points": [
            {
                "concurrency": concurrency,
                "pareto_front": [
                    row for row in rows if row["concurrency"] == concurrency
                ],
            }
            for concurrency in sorted(fronts)
        ],
    }
    envelope_database = {
        "schema": ENVELOPE_SCHEMA,
        "generated_at": _utc_now(),
        "objective_names": objective_names,
        "source_candidate_count": len(rows),
        "candidate_count": len(global_rows),
        "points": global_rows,
    }
    _write_json(output_dir / "load_conditioned_pareto.json", load_database)
    _write_csv(
        output_dir / "load_conditioned_pareto.csv", rows, objective_names
    )
    _write_json(output_dir / "global_pareto_envelope.json", envelope_database)
    _write_csv(
        output_dir / "global_pareto_envelope.csv", global_rows, objective_names
    )
    if plot and global_rows:
        title = (
            f"{config.search_space.hardware_sku} / "
            f"{config.search_space.model_name}\nGlobal Pareto envelope across fixed concurrency"
        )
        _plot_envelope(
            output_dir / "global_pareto_envelope.png",
            rows,
            global_rows,
            objective_names,
            title,
        )
    return global_front


def _manifest(
    *,
    config_path: Path,
    config_sha256: str,
    output_dir: Path,
    concurrencies: list[int],
    parallelism: dict[str, int],
    statuses: dict[int, str],
    started_at: str,
    status: str,
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "status": status,
        "started_at": started_at,
        "updated_at": _utc_now(),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "output_dir": str(output_dir),
        "concurrency_values": concurrencies,
        "parallelism": parallelism,
        "points": [
            {
                "concurrency": concurrency,
                "status": statuses.get(concurrency, "pending"),
                "result_path": str(_point_path(output_dir, concurrency)),
                "checkpoint_path": str(
                    _checkpoint_path(output_dir, concurrency)
                ),
            }
            for concurrency in concurrencies
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one fixed-concurrency Sweeper Pareto study per load point and "
            "build a global envelope"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONCURRENCIES),
    )
    parser.add_argument(
        "--max-total-parallel-evals",
        type=int,
        default=DEFAULT_MAX_TOTAL_PARALLEL_EVALS,
        help=(
            "upper bound for outer_parallelism * "
            "min(parallel_evals, candidates_per_round)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse completed points and resume incomplete points from the last "
            "committed inner round"
        ),
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="rebuild JSON/CSV/plot from completed point files without replay",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="write JSON and CSV databases without importing matplotlib",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the resolved per-load plan without replay",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    concurrencies = list(args.concurrencies)
    if not concurrencies or any(value <= 0 for value in concurrencies):
        raise ValueError("--concurrencies must contain positive integers")
    if len(set(concurrencies)) != len(concurrencies):
        raise ValueError("--concurrencies must not contain duplicates")
    if args.max_total_parallel_evals <= 0:
        raise ValueError("--max-total-parallel-evals must be positive")

    template = SmartSearchConfig.from_yaml(config_path)
    _validate_template(template)
    resolved_configs = {
        concurrency: _config_for_concurrency(template, concurrency)
        for concurrency in concurrencies
    }
    per_point_trials = (
        len(template.search_space.deployment_mode)
        * template.sweep.max_rounds
        * (template.sweep.candidates_per_round or template.sweep.parallel_evals)
    )
    inner_parallelism = _effective_inner_parallelism(template)
    planned_outer_parallelism = _outer_parallelism(
        len(concurrencies),
        inner_parallelism=inner_parallelism,
        max_total_parallel_evals=args.max_total_parallel_evals,
    )
    planned_total_parallelism = planned_outer_parallelism * inner_parallelism
    parallelism_manifest = {
        "max_total_parallel_evals": args.max_total_parallel_evals,
        "inner_parallelism": inner_parallelism,
        "outer_parallelism_limit": (
            args.max_total_parallel_evals // inner_parallelism
        ),
        "effective_total_limit": (
            args.max_total_parallel_evals // inner_parallelism
        )
        * inner_parallelism,
    }
    print(f"config: {config_path}")
    print(f"output: {output_dir}")
    print(f"concurrency points: {concurrencies}")
    print(
        "parallelism: "
        f"outer={planned_outer_parallelism}, inner={inner_parallelism}, "
        f"effective_total={planned_total_parallelism}, "
        f"configured_max={args.max_total_parallel_evals}"
    )
    print(f"upper-bound trials per point: {per_point_trials}")
    print(f"upper-bound total trials: {per_point_trials * len(concurrencies)}")
    if args.dry_run:
        for concurrency, config in resolved_configs.items():
            print(
                f"dry-run concurrency={concurrency}: "
                f"request_count={config.workload.resolved_request_count()}"
            )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    statuses: dict[int, str] = {}
    fronts: dict[int, list[Candidate]] = {}
    manifest_path = output_dir / "manifest.json"
    config_sha256 = _sha256(config_path)

    if args.resume or args.aggregate_only:
        for concurrency in concurrencies:
            path = _point_path(output_dir, concurrency)
            candidates = _load_completed_point(
                path, concurrency, resolved_configs[concurrency]
            )
            if candidates is None:
                continue
            fronts[concurrency] = candidates
            statuses[concurrency] = "complete"
            print(
                f"reuse concurrency={concurrency}: "
                f"{len(candidates)} Pareto candidate(s)"
            )

    if args.aggregate_only:
        missing = [value for value in concurrencies if value not in fronts]
        if missing:
            raise RuntimeError(
                f"aggregate-only is missing completed point files for {missing}"
            )
        if not any(fronts.values()):
            raise RuntimeError("completed point files contain no Pareto candidates")
        global_front = write_pareto_artifacts(
            output_dir, fronts, template, plot=not args.skip_plot
        )
        _write_json(
            manifest_path,
            _manifest(
                config_path=config_path,
                config_sha256=config_sha256,
                output_dir=output_dir,
                concurrencies=concurrencies,
                parallelism=parallelism_manifest,
                statuses=statuses,
                started_at=started_at,
                status="complete",
            ),
        )
        print(f"global envelope: {len(global_front)} candidate(s)")
        return 0

    pending = [value for value in concurrencies if value not in fronts]
    failures: dict[int, str] = {}
    if pending:
        try:
            from dynamo.replay.simulation import DynamoReplayRunnerFactory
        except ImportError as exc:
            raise RuntimeError(
                "Dynamo replay is unavailable; install/build the matching "
                "ai-dynamo runtime"
            ) from exc

        actual_outer_parallelism = _outer_parallelism(
            len(pending),
            inner_parallelism=inner_parallelism,
            max_total_parallel_evals=args.max_total_parallel_evals,
        )
        actual_total_parallelism = actual_outer_parallelism * inner_parallelism
        print(
            "remaining parallelism: "
            f"outer={actual_outer_parallelism}, inner={inner_parallelism}, "
            f"effective_total={actual_total_parallelism}"
        )
        statuses.update({concurrency: "queued" for concurrency in pending})
        _write_json(
            manifest_path,
            _manifest(
                config_path=config_path,
                config_sha256=config_sha256,
                output_dir=output_dir,
                concurrencies=concurrencies,
                parallelism=parallelism_manifest,
                statuses=statuses,
                started_at=started_at,
                status="running",
            ),
        )

        future_points: dict[Future[list[Candidate]], int] = {}
        completed_count = 0
        with (
            ThreadPoolExecutor(
                max_workers=actual_outer_parallelism,
                thread_name_prefix="concurrency-sweep",
            ) as executor,
            tqdm(
                total=len(pending),
                desc="Outer Sweeper",
                unit="point",
                dynamic_ncols=True,
                file=sys.stdout,
            ) as outer_bar,
        ):
            for concurrency in pending:
                tqdm.write(
                    f"submit concurrency={concurrency}", file=sys.stdout
                )
                future = executor.submit(
                    _run_fixed_concurrency,
                    resolved_configs[concurrency],
                    concurrency,
                    DynamoReplayRunnerFactory,
                    _checkpoint_path(output_dir, concurrency),
                    args.resume,
                )
                future_points[future] = concurrency

            for future in as_completed(future_points):
                concurrency = future_points[future]
                point_config = resolved_configs[concurrency]
                try:
                    candidates = future.result()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    failures[concurrency] = error
                    statuses[concurrency] = "failed"
                    _write_json(
                        _point_path(output_dir, concurrency),
                        {
                            "schema": POINT_SCHEMA,
                            "status": "failed",
                            "failed_at": _utc_now(),
                            "concurrency": concurrency,
                            "checkpoint_path": str(
                                _checkpoint_path(output_dir, concurrency)
                            ),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    tqdm.write(
                        f"failed concurrency={concurrency}: {error}",
                        file=sys.stdout,
                    )
                else:
                    point_payload = {
                        "schema": POINT_SCHEMA,
                        "status": "complete",
                        "completed_at": _utc_now(),
                        "concurrency": concurrency,
                        "config": point_config.model_dump(mode="json"),
                        "checkpoint_path": str(
                            _checkpoint_path(output_dir, concurrency)
                        ),
                        "pareto_front": [
                            candidate.model_dump(mode="json")
                            for candidate in candidates
                        ],
                    }
                    _write_json(
                        _point_path(output_dir, concurrency), point_payload
                    )
                    fronts[concurrency] = candidates
                    statuses[concurrency] = "complete"
                    completed_count += 1
                    tqdm.write(
                        f"complete concurrency={concurrency}: "
                        f"{len(candidates)} Pareto candidate(s)",
                        file=sys.stdout,
                    )

                    # Persist databases after each completed replay. Plot only
                    # after all points finish so plotting cannot invalidate it.
                    write_pareto_artifacts(
                        output_dir, fronts, template, plot=False
                    )

                _write_json(
                    manifest_path,
                    _manifest(
                        config_path=config_path,
                        config_sha256=config_sha256,
                        output_dir=output_dir,
                        concurrencies=concurrencies,
                        parallelism=parallelism_manifest,
                        statuses=statuses,
                        started_at=started_at,
                        status="running",
                    ),
                )
                outer_bar.update(1)
                outer_bar.set_postfix(
                    completed=completed_count,
                    failed=len(failures),
                    last=f"c={concurrency}",
                )

    if failures:
        _write_json(
            manifest_path,
            _manifest(
                config_path=config_path,
                config_sha256=config_sha256,
                output_dir=output_dir,
                concurrencies=concurrencies,
                parallelism=parallelism_manifest,
                statuses=statuses,
                started_at=started_at,
                status="failed",
            ),
        )
        detail = "; ".join(
            f"concurrency={concurrency}: {error}"
            for concurrency, error in sorted(failures.items())
        )
        raise RuntimeError(f"{len(failures)} concurrency point(s) failed: {detail}")

    if not any(fronts.values()):
        raise RuntimeError("no fixed-concurrency study produced a Pareto front")
    global_front = write_pareto_artifacts(
        output_dir, fronts, template, plot=not args.skip_plot
    )
    _write_json(
        manifest_path,
        _manifest(
            config_path=config_path,
            config_sha256=config_sha256,
            output_dir=output_dir,
            concurrencies=concurrencies,
            parallelism=parallelism_manifest,
            statuses=statuses,
            started_at=started_at,
            status="complete",
        ),
    )
    print(f"global envelope: {len(global_front)} candidate(s)")
    print(f"artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        raise
