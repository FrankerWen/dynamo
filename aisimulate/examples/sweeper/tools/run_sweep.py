# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Sweeper with the transitional Dynamo replay composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from aisimulate.sweeper import LoadSweepResult, Sweeper
from aisimulate.sweeper.__main__ import (
    load_config_or_parser_error,
    print_candidates_or_exit,
)
from dynamo.replay.simulation import DynamoReplayRunnerFactory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Sweeper sweep with Dynamo Replay"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output-json",
        help="Write the resolved input and candidate or load-envelope result to JSON",
    )
    args = parser.parse_args()

    config = load_config_or_parser_error(parser, args.config)
    try:
        result = Sweeper(
            runner_factory=DynamoReplayRunnerFactory(),
        ).run(config)
    except ValidationError as exc:
        parser.error(f"invalid adapter search space in {args.config}: {exc}")
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "config_path": str(Path(args.config).resolve()),
            "config": config.model_dump(mode="json"),
        }
        if isinstance(result, LoadSweepResult):
            payload["load_sweep_result"] = result.model_dump(mode="json")
        else:
            payload["candidates"] = [
                candidate.model_dump(mode="json") for candidate in result
            ]
        output_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print_candidates_or_exit(config, result)


if __name__ == "__main__":
    main()
