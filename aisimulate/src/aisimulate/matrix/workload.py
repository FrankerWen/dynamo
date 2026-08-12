# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extensible workload preparation and file normalization."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._utils import fingerprint, resolve_path, sha256_file
from .config import WorkloadSpec


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoaderContext(_StrictModel):
    base_dir: Path


class WorkloadCapabilities(_StrictModel):
    aggregate: bool = True
    disaggregate: bool = True
    open_loop: bool = True
    closed_loop: bool = True


class LoadAxis(_StrictModel):
    name: Literal[
        "request_rate", "arrival_speedup_ratio", "concurrency", "kv_load_ratio"
    ]
    source: Literal["explicit", "aic_operating_points"] = "explicit"
    values: list[float] = Field(default_factory=list)
    min_value: float | None = Field(default=None, gt=0)
    max_value: float | None = Field(default=None, gt=0)
    max_points: int | None = Field(default=None, ge=1)
    include_static_pareto: bool = True
    topk_per_parallel: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def _positive_values(self) -> LoadAxis:
        if any(value <= 0 for value in self.values):
            raise ValueError("load axis values must be positive")
        if self.source == "explicit" and not self.values:
            raise ValueError("an explicit load axis must contain at least one value")
        if self.source == "aic_operating_points":
            if self.name != "concurrency":
                raise ValueError(
                    "AIC operating-point discovery currently resolves concurrency only"
                )
            if self.values:
                raise ValueError(
                    "AIC operating-point discovery cannot also contain explicit values"
                )
            if self.max_points is None:
                raise ValueError("AIC operating-point discovery requires max_points")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("load axis min_value cannot exceed max_value")
        return self


class PreparedWorkload(_StrictModel):
    id: str
    loader: str
    fingerprint: str
    replay_args: dict[str, Any]
    summary: dict[str, Any]
    capabilities: WorkloadCapabilities
    source_path: str | None = None
    source_sha256: str | None = None
    load_axis: LoadAxis | None = None
    normalization: dict[str, Any] | None = None


class WorkloadLoader(Protocol):
    type_name: ClassVar[str]

    def prepare(
        self, spec: WorkloadSpec, context: LoaderContext
    ) -> PreparedWorkload: ...


_LOADERS: dict[str, Callable[[], WorkloadLoader]] = {}


def register_workload_loader(
    name: str, factory: Callable[[], WorkloadLoader], *, replace: bool = False
) -> None:
    if not name:
        raise ValueError("workload loader name must not be empty")
    if name in _LOADERS and not replace:
        raise ValueError(f"workload loader {name!r} is already registered")
    _LOADERS[name] = factory


def prepare_workload(spec: WorkloadSpec, context: LoaderContext) -> PreparedWorkload:
    factory = _LOADERS.get(spec.loader)
    if factory is None:
        raise ValueError(
            f"unknown workload loader {spec.loader!r}; registered={sorted(_LOADERS)}"
        )
    prepared = factory().prepare(spec, context)
    if prepared.id != spec.id or prepared.loader != spec.loader:
        raise ValueError(
            f"loader {spec.loader!r} returned inconsistent identity for {spec.id!r}"
        )
    return prepared


class _ConcurrencySweepConfig(_StrictModel):
    """Resolve realistic closed-loop loads from AIC's static serving sweep."""

    source: Literal["aic_operating_points"] = "aic_operating_points"
    min_concurrency: int = Field(default=1, ge=1)
    max_concurrency: int | None = Field(default=None, ge=1)
    max_points: int = Field(default=24, ge=1)
    include_static_pareto: bool = True
    topk_per_parallel: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> _ConcurrencySweepConfig:
        if (
            self.max_concurrency is not None
            and self.min_concurrency > self.max_concurrency
        ):
            raise ValueError("min_concurrency cannot exceed max_concurrency")
        return self


class _SyntheticConfig(_StrictModel):
    isl: int = Field(gt=0)
    osl: int = Field(gt=0)
    prefix_length: int = Field(default=0, ge=0)
    request_rate: float | None = Field(default=None, gt=0)
    request_rates: list[float] | None = None
    concurrency: int | None = Field(default=None, gt=0)
    concurrency_sweep: _ConcurrencySweepConfig | None = None
    kv_load_ratio: float | None = Field(default=None, gt=0, le=1)
    num_request_ratio: float = Field(default=10.0, gt=0)
    shared_prefix_ratio: float | None = Field(default=None, ge=0, le=1)
    num_prefix_groups: int = Field(default=0, ge=0)
    turns_per_session: int = Field(default=1, ge=1)
    inter_turn_delay_ms: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _one_load_mode(self) -> _SyntheticConfig:
        modes = (
            self.request_rate is not None or self.request_rates is not None,
            self.concurrency is not None or self.concurrency_sweep is not None,
            self.kv_load_ratio is not None,
        )
        if sum(modes) != 1:
            raise ValueError(
                "synthetic workload needs exactly one of request_rate/request_rates, "
                "concurrency/concurrency_sweep, or kv_load_ratio"
            )
        if self.concurrency is not None and self.concurrency_sweep is not None:
            raise ValueError(
                "synthetic workload cannot set both concurrency and concurrency_sweep"
            )
        if self.request_rates is not None and (
            not self.request_rates or any(value <= 0 for value in self.request_rates)
        ):
            raise ValueError("request_rates must be non-empty and positive")
        if self.prefix_length > self.isl:
            raise ValueError("prefix_length cannot exceed isl")
        derived_ratio = self.prefix_length / self.isl
        if (
            self.prefix_length > 0
            and self.shared_prefix_ratio is not None
            and not math.isclose(self.shared_prefix_ratio, derived_ratio)
        ):
            raise ValueError(
                "shared_prefix_ratio conflicts with prefix_length / isl; use prefix_length as the canonical synthetic prefix"
            )
        return self

    @property
    def resolved_shared_prefix_ratio(self) -> float:
        if self.shared_prefix_ratio is not None:
            return self.shared_prefix_ratio
        return self.prefix_length / self.isl

    @property
    def resolved_num_prefix_groups(self) -> int:
        return self.num_prefix_groups or (
            1 if self.resolved_shared_prefix_ratio > 0 else 0
        )


class SyntheticLoader:
    type_name = "synthetic"

    def prepare(self, spec: WorkloadSpec, context: LoaderContext) -> PreparedWorkload:
        del context
        config = _SyntheticConfig.model_validate(spec.config)
        request_rates = config.request_rates
        request_rate = config.request_rate
        load_axis = None
        if request_rates is not None:
            request_rate = request_rates[0]
            load_axis = LoadAxis(name="request_rate", values=request_rates)
        concurrency = config.concurrency
        if config.concurrency_sweep is not None:
            sweep = config.concurrency_sweep
            concurrency = sweep.min_concurrency
            load_axis = LoadAxis(
                name="concurrency",
                source=sweep.source,
                min_value=float(sweep.min_concurrency),
                max_value=None
                if sweep.max_concurrency is None
                else float(sweep.max_concurrency),
                max_points=sweep.max_points,
                include_static_pareto=sweep.include_static_pareto,
                topk_per_parallel=sweep.topk_per_parallel,
            )
        replay_args = {
            "isl": config.isl,
            "osl": config.osl,
            "request_rate": request_rate,
            "concurrency": concurrency,
            "kv_load_ratio": config.kv_load_ratio,
            "num_request_ratio": config.num_request_ratio,
            "shared_prefix_ratio": config.resolved_shared_prefix_ratio,
            "num_prefix_groups": config.resolved_num_prefix_groups,
            "turns_per_session": config.turns_per_session,
            "inter_turn_delay_ms": config.inter_turn_delay_ms,
        }
        payload = {
            "id": spec.id,
            "loader": spec.loader,
            "replay_args": replay_args,
            "prefix_length": config.prefix_length,
            "load_axis": load_axis,
        }
        return PreparedWorkload(
            id=spec.id,
            loader=spec.loader,
            fingerprint=fingerprint(payload),
            replay_args=replay_args,
            summary={
                "isl": config.isl,
                "osl": config.osl,
                "prefix_length": config.prefix_length,
                "context_length": config.isl + config.osl,
            },
            capabilities=WorkloadCapabilities(),
            load_axis=load_axis,
        )


class _TraceConfig(_StrictModel):
    path: str
    arrival_speedup_ratio: float = Field(default=1.0, gt=0)
    arrival_speedup_ratios: list[float] | None = None
    replay_concurrency: int | None = Field(default=None, gt=0)
    trace_block_size: int = Field(default=512, gt=0)
    request_count: int | None = Field(default=None, gt=0)
    shared_prefix_ratio: float = Field(default=0.0, ge=0, le=1)
    num_prefix_groups: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _valid_speedup_axis(self) -> _TraceConfig:
        if self.arrival_speedup_ratios is not None and (
            not self.arrival_speedup_ratios
            or any(value <= 0 for value in self.arrival_speedup_ratios)
        ):
            raise ValueError("arrival_speedup_ratios must be non-empty and positive")
        return self


def _jsonl_summary(path: Path, request_count: int | None) -> dict[str, Any]:
    row_count = 0
    timestamps: list[float] = []
    input_lengths: list[int] = []
    output_lengths: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"trace row {path}:{line_no} must be an object")
            timestamp = row.get("timestamp")
            if isinstance(timestamp, (int, float)):
                timestamps.append(float(timestamp))
            input_length = row.get("input_length")
            output_length = row.get("output_length")
            if isinstance(input_length, int):
                input_lengths.append(input_length)
            if isinstance(output_length, int):
                output_lengths.append(output_length)
    if row_count == 0:
        raise ValueError(f"trace {path} contains no records")
    return {
        "row_count": row_count,
        "request_count": request_count or row_count,
        "start_timestamp": min(timestamps) if timestamps else None,
        "end_timestamp": max(timestamps) if timestamps else None,
        "mean_isl": sum(input_lengths) / len(input_lengths) if input_lengths else None,
        "mean_osl": sum(output_lengths) / len(output_lengths)
        if output_lengths
        else None,
        "max_context_length": max(
            (isl + osl for isl, osl in zip(input_lengths, output_lengths)),
            default=None,
        ),
    }


class _NativeTraceLoader:
    type_name = ""
    trace_format = "mooncake"
    capabilities = WorkloadCapabilities()
    requires_concurrency = False

    def prepare(self, spec: WorkloadSpec, context: LoaderContext) -> PreparedWorkload:
        config = _TraceConfig.model_validate(spec.config)
        if self.requires_concurrency and config.replay_concurrency is None:
            raise ValueError(
                f"{self.type_name} requires replay_concurrency because its source has no first-turn timestamps"
            )
        path = resolve_path(context.base_dir, config.path)
        source_sha256 = sha256_file(path)
        speedups = config.arrival_speedup_ratios
        speedup = speedups[0] if speedups is not None else config.arrival_speedup_ratio
        load_axis = (
            LoadAxis(name="arrival_speedup_ratio", values=speedups)
            if speedups is not None
            else None
        )
        replay_args = {
            "trace_path": str(path),
            "trace_format": self.trace_format,
            "trace_block_size": config.trace_block_size,
            "arrival_speedup_ratio": speedup,
            "replay_concurrency": config.replay_concurrency,
            "shared_prefix_ratio": config.shared_prefix_ratio,
            "num_prefix_groups": config.num_prefix_groups,
        }
        summary = _jsonl_summary(path, config.request_count)
        summary["trace_block_size"] = config.trace_block_size
        summary["shared_prefix_ratio"] = config.shared_prefix_ratio
        summary["num_prefix_groups"] = config.num_prefix_groups
        return PreparedWorkload(
            id=spec.id,
            loader=spec.loader,
            fingerprint=fingerprint(
                {
                    "loader": spec.loader,
                    "source_sha256": source_sha256,
                    "replay_args": replay_args,
                    "load_axis": load_axis,
                }
            ),
            replay_args=replay_args,
            summary=summary,
            capabilities=self.capabilities,
            source_path=str(path),
            source_sha256=source_sha256,
            load_axis=load_axis,
        )


class MooncakeTraceLoader(_NativeTraceLoader):
    type_name = "mooncake_trace"
    trace_format = "mooncake"


class AgenticMooncakeTraceLoader(_NativeTraceLoader):
    type_name = "agentic_mooncake_trace"
    trace_format = "agentic_mooncake"
    capabilities = WorkloadCapabilities(closed_loop=False)


class AppliedComputeAgenticTraceLoader(_NativeTraceLoader):
    type_name = "applied_compute_agentic_trace"
    trace_format = "applied_compute_agentic"
    capabilities = WorkloadCapabilities(open_loop=False)
    requires_concurrency = True


class _TabularConfig(_TraceConfig):
    file_format: Literal["csv", "jsonl"] | None = None
    columns: dict[
        Literal["timestamp", "input_length", "output_length", "hash_ids"], str
    ]

    @model_validator(mode="after")
    def _required_columns(self) -> _TabularConfig:
        missing = {"timestamp", "input_length", "output_length"} - set(self.columns)
        if missing:
            raise ValueError(f"tabular trace columns missing {sorted(missing)}")
        return self


class TabularTraceLoader:
    type_name = "tabular_trace"

    def prepare(self, spec: WorkloadSpec, context: LoaderContext) -> PreparedWorkload:
        config = _TabularConfig.model_validate(spec.config)
        path = resolve_path(context.base_dir, config.path)
        source_sha256 = sha256_file(path)
        file_format = config.file_format or path.suffix.lower().lstrip(".")
        if file_format not in {"csv", "jsonl"}:
            raise ValueError(f"cannot infer tabular file format for {path}")
        rows = _read_tabular_rows(path, file_format)
        _validate_tabular_rows(path, rows, config.columns)
        speedups = config.arrival_speedup_ratios
        speedup = speedups[0] if speedups is not None else config.arrival_speedup_ratio
        load_axis = (
            LoadAxis(name="arrival_speedup_ratio", values=speedups)
            if speedups is not None
            else None
        )
        replay_args = {
            "trace_path": str(path),
            "trace_format": "mooncake",
            "trace_block_size": config.trace_block_size,
            "arrival_speedup_ratio": speedup,
            "replay_concurrency": config.replay_concurrency,
            "shared_prefix_ratio": config.shared_prefix_ratio,
            "num_prefix_groups": config.num_prefix_groups,
        }
        normalization = {
            "file_format": file_format,
            "columns": config.columns,
        }
        return PreparedWorkload(
            id=spec.id,
            loader=spec.loader,
            fingerprint=fingerprint(
                {
                    "loader": spec.loader,
                    "source_sha256": source_sha256,
                    "normalization": normalization,
                    "replay_args": replay_args,
                    "load_axis": load_axis,
                }
            ),
            replay_args=replay_args,
            summary={
                "row_count": len(rows),
                "request_count": config.request_count or len(rows),
                "trace_block_size": config.trace_block_size,
            },
            capabilities=WorkloadCapabilities(),
            source_path=str(path),
            source_sha256=source_sha256,
            load_axis=load_axis,
            normalization=normalization,
        )


def _read_tabular_rows(path: Path, file_format: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        if file_format == "csv":
            rows.extend(dict(row) for row in csv.DictReader(handle))
        else:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL at {path}:{line_no}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise TypeError(f"tabular row {path}:{line_no} must be an object")
                rows.append(row)
    if not rows:
        raise ValueError(f"tabular trace {path} contains no records")
    return rows


def _validate_tabular_rows(
    path: Path, rows: list[dict[str, Any]], columns: dict[str, str]
) -> None:
    required = [
        columns[name] for name in ("timestamp", "input_length", "output_length")
    ]
    for row_no, row in enumerate(rows, start=1):
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(f"tabular row {path}:{row_no} missing columns {missing}")


def materialize_workload(
    prepared: PreparedWorkload, output_dir: Path
) -> PreparedWorkload:
    """Normalize a tabular trace into a content-addressed Mooncake JSONL file."""
    if prepared.normalization is None:
        return prepared
    if prepared.source_path is None:
        raise ValueError("a normalized workload must retain its source_path")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{prepared.fingerprint}.jsonl"
    if not target.exists():
        source = Path(prepared.source_path)
        normalization = prepared.normalization
        rows = _read_tabular_rows(source, str(normalization["file_format"]))
        columns = dict(normalization["columns"])
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                normalized = {
                    "timestamp": float(row[columns["timestamp"]]),
                    "input_length": int(row[columns["input_length"]]),
                    "output_length": int(row[columns["output_length"]]),
                }
                hash_column = columns.get("hash_ids")
                if hash_column is not None and row.get(hash_column) not in (None, ""):
                    raw_hashes = row[hash_column]
                    normalized["hash_ids"] = (
                        json.loads(raw_hashes)
                        if isinstance(raw_hashes, str)
                        else raw_hashes
                    )
                handle.write(json.dumps(normalized, sort_keys=True) + "\n")
    replay_args = dict(prepared.replay_args)
    replay_args["trace_path"] = str(target)
    return prepared.model_copy(update={"replay_args": replay_args})


register_workload_loader(SyntheticLoader.type_name, SyntheticLoader)
register_workload_loader(MooncakeTraceLoader.type_name, MooncakeTraceLoader)
register_workload_loader(
    AgenticMooncakeTraceLoader.type_name, AgenticMooncakeTraceLoader
)
register_workload_loader(
    AppliedComputeAgenticTraceLoader.type_name, AppliedComputeAgenticTraceLoader
)
register_workload_loader(TabularTraceLoader.type_name, TabularTraceLoader)
