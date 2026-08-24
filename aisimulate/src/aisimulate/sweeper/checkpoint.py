# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable round-barrier state for resumable Sweeper studies."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import Candidate
from .replay import validate_json_value


CHECKPOINT_SCHEMA = "aisimulate.sweeper.round_checkpoint.v1"


class SamplerObservation(BaseModel):
    """One completed optimizer trial needed to warm-start a fresh study."""

    model_config = ConfigDict(extra="forbid")

    parameters: dict[str, Any]
    metrics: dict[str, float] | None = None
    infeasible_reason: str | None = None

    @model_validator(mode="after")
    def _validate_outcome(self) -> SamplerObservation:
        validate_json_value(self.parameters, path="sampler observation parameters")
        if (self.metrics is None) == (self.infeasible_reason is None):
            raise ValueError(
                "a sampler observation requires exactly one of metrics or "
                "infeasible_reason"
            )
        return self


class ReplayCacheEntry(BaseModel):
    """One successful replay result keyed by its stable full-sample digest."""

    model_config = ConfigDict(extra="forbid")

    candidate: Candidate
    observe_metrics: dict[str, float]


class BranchCheckpoint(BaseModel):
    """Completed round barriers and optimizer observations for one branch."""

    model_config = ConfigDict(extra="forbid")

    completed_rounds: int = Field(default=0, ge=0)
    stalled: bool = False
    observations: list[SamplerObservation] = Field(default_factory=list)


class SweepRoundCheckpoint(BaseModel):
    """Complete state required to continue at the next round barrier."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_schema: Literal[CHECKPOINT_SCHEMA] = CHECKPOINT_SCHEMA
    fingerprint: str
    updated_at: str
    candidates: list[Candidate] = Field(default_factory=list)
    replay_cache: dict[str, ReplayCacheEntry] = Field(default_factory=dict)
    tally: dict[str, int]
    failure_reasons: dict[str, int] = Field(default_factory=dict)
    branches: dict[str, BranchCheckpoint]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_round_checkpoint(
    path: Path, *, expected_fingerprint: str
) -> SweepRoundCheckpoint | None:
    """Load one checkpoint, failing closed on corruption or config drift."""

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    checkpoint = SweepRoundCheckpoint.model_validate_json(text)
    if checkpoint.fingerprint != expected_fingerprint:
        raise ValueError(
            f"checkpoint fingerprint mismatch for {path}; rerun without resume "
            "to replace stale state"
        )
    return checkpoint


def write_round_checkpoint(path: Path, checkpoint: SweepRoundCheckpoint) -> None:
    """Atomically replace a checkpoint after a completed round barrier."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            checkpoint.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
