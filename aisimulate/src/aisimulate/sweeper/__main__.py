# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental ``python -m aisimulate.sweeper`` entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import yaml
from pydantic import ValidationError

from .config import Candidate, LoadSweepResult, SmartSearchConfig


def load_config_or_parser_error(
    parser: argparse.ArgumentParser, path: str
) -> SmartSearchConfig:
    """Load a sweep config with the legacy CLI's concise error messages."""

    try:
        return SmartSearchConfig.from_yaml(path)
    except OSError as exc:  # missing file, a directory, unreadable, etc.
        parser.error(f"could not read config {path}: {exc}")
    except yaml.YAMLError as exc:
        parser.error(f"malformed YAML in {path}: {exc}")
    except ValidationError as exc:
        parser.error(f"invalid config {path}: {exc}")


def print_candidates_or_exit(
    config: SmartSearchConfig, candidates: Sequence[Candidate] | LoadSweepResult
) -> None:
    """Preserve the legacy CLI result and no-candidate behavior for wrappers."""

    if isinstance(candidates, LoadSweepResult):
        if not candidates.global_operating_envelope:
            print(
                "no feasible candidate found at any load point "
                "(check backends / SLA / gpu_budget / replay errors)",
                file=sys.stderr,
            )
            raise SystemExit(1)
        for point in candidates.load_conditioned_pareto:
            print(
                f"load-conditioned pareto "
                f"{candidates.load_axis}={point.load_value:.12g} "
                f"({len(point.candidates)} non-dominated):"
            )
            for index, candidate in enumerate(point.candidates):
                print(_pareto_candidate_line(index, candidate))
        print(
            "global operating envelope "
            f"({len(candidates.global_operating_envelope)} non-dominated):"
        )
        for index, candidate in enumerate(candidates.global_operating_envelope):
            print(_pareto_candidate_line(index, candidate))
        return

    if not candidates:
        print(
            "no feasible candidate found "
            "(check backends / SLA / gpu_budget / replay errors)",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if config.goal.is_pareto:
        print(f"pareto front ({len(candidates)} non-dominated):")
        for index, candidate in enumerate(candidates):
            print(_pareto_candidate_line(index, candidate))
        return
    for index, candidate in enumerate(candidates):
        print(f"{index}: score={candidate.score} used_gpus={candidate.used_gpus}")


def _pareto_candidate_line(index: int, candidate: Candidate) -> str:
    objectives = ", ".join(
        f"{key}={value:.4g}" for key, value in (candidate.objectives or {}).items()
    )
    concrete_load = "".join(
        f" {name}={candidate.config[name]}"
        for name in ("arrival_speedup_ratio", "request_rate", "concurrency")
        if candidate.config.get(name) is not None
    )
    return f"{index}: {objectives}{concrete_load} used_gpus={candidate.used_gpus}"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m aisimulate.sweeper",
        description=(
            "[EXPERIMENTAL] Sweeper is replay-runtime neutral. Use its Python API "
            "with an injected RunnerFactory to execute a sweep."
        ),
    )
    parser.add_argument(
        "--config", required=True, help="Path to a SmartSearchConfig YAML file"
    )
    args = parser.parse_args()

    config = load_config_or_parser_error(parser, args.config)

    del config
    parser.error(
        "the standalone CLI has no default replay runtime; call "
        "aisimulate.sweeper.Sweeper(runner_factory=...).run(config) from Python"
    )


if __name__ == "__main__":
    main()
