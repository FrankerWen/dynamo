# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append-first event log and DuckDB result store for matrix searches."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Concatenate, ParamSpec, Self, TypeVar

from ._utils import canonical_json, fingerprint, jsonable
from .config import CatalogBundle, MatrixExperimentConfig, PriceSpec
from .expand import MatrixCell
from .variant import VariantResolutionRecord, VariantSupportResult

_Params = ParamSpec("_Params")
_ReturnT = TypeVar("_ReturnT")


def _connect_database(path: Path, *, read_only: bool = False):
    """Import DuckDB only in the process that actually owns/reads a result store.

    Importing ``duckdb`` eagerly creates a host-sized thread pool in current wheels.
    Spawned replay workers import this module while unpickling MatrixRunner tasks, but
    never access the database; a lazy import avoids hundreds of idle threads per worker.
    """

    import duckdb

    return duckdb.connect(str(path), read_only=read_only)


def _synchronized(
    method: Callable[Concatenate[ResultStore, _Params], _ReturnT],
) -> Callable[Concatenate[ResultStore, _Params], _ReturnT]:
    """Serialize the process-owned DuckDB connection and append-only event log."""

    @wraps(method)
    def wrapped(
        self: ResultStore, *args: _Params.args, **kwargs: _Params.kwargs
    ) -> _ReturnT:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResultStore:
    """Single-writer store. Worker processes return records to this owner."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.output_dir / "sweep.duckdb"
        self.events_path = self.output_dir / "events.jsonl"
        self._lock = threading.RLock()
        self.connection = _connect_database(self.database_path)
        self._create_schema()

    @classmethod
    def open(cls, path: str | Path) -> ResultStore:
        raw_path = Path(path).expanduser().resolve()
        output_dir = raw_path.parent if raw_path.suffix == ".duckdb" else raw_path
        return cls(output_dir)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @_synchronized
    def close(self) -> None:
        self.connection.close()

    @_synchronized
    def _create_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id VARCHAR PRIMARY KEY,
                config_json JSON NOT NULL,
                created_at VARCHAR NOT NULL,
                updated_at VARCHAR NOT NULL,
                status VARCHAR NOT NULL
            );
            CREATE TABLE IF NOT EXISTS input_artifacts (
                experiment_id VARCHAR NOT NULL,
                role VARCHAR NOT NULL,
                path VARCHAR NOT NULL,
                sha256 VARCHAR NOT NULL,
                PRIMARY KEY (experiment_id, role, path)
            );
            CREATE TABLE IF NOT EXISTS matrix_cells (
                cell_id VARCHAR PRIMARY KEY,
                experiment_id VARCHAR NOT NULL,
                fingerprint VARCHAR NOT NULL,
                model_id VARCHAR NOT NULL,
                model_variant_id VARCHAR,
                sku_id VARCHAR NOT NULL,
                workload_id VARCHAR NOT NULL,
                sla_id VARCHAR NOT NULL,
                support_tier VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                detail VARCHAR,
                cell_json JSON NOT NULL,
                updated_at VARCHAR NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id VARCHAR PRIMARY KEY,
                evaluation_key VARCHAR NOT NULL,
                evaluation_cache_key VARCHAR,
                cell_id VARCHAR NOT NULL,
                stage VARCHAR NOT NULL,
                outcome VARCHAR NOT NULL,
                cache_hit BOOLEAN NOT NULL DEFAULT FALSE,
                reason VARCHAR NOT NULL,
                selection_json JSON NOT NULL,
                metrics_json JSON NOT NULL,
                sla_pass BOOLEAN,
                created_at VARCHAR NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS evaluations_cell_key
                ON evaluations(cell_id, evaluation_key);
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id VARCHAR PRIMARY KEY,
                evaluation_id VARCHAR NOT NULL,
                cell_id VARCHAR NOT NULL,
                score DOUBLE NOT NULL,
                used_gpus INTEGER NOT NULL,
                objectives_json JSON,
                config_json JSON NOT NULL,
                compute_config_key VARCHAR NOT NULL,
                pd_profile_id VARCHAR,
                kvbm_profile_id VARCHAR
            );
            CREATE TABLE IF NOT EXISTS resource_usage (
                candidate_id VARCHAR PRIMARY KEY,
                gpu_hours DOUBLE NOT NULL,
                g2_gib_hours DOUBLE NOT NULL,
                network_gb DOUBLE NOT NULL,
                input_tokens DOUBLE NOT NULL,
                output_tokens DOUBLE NOT NULL,
                goodput_input_tokens DOUBLE NOT NULL,
                goodput_output_tokens DOUBLE NOT NULL,
                duration_ms DOUBLE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS frontier_membership (
                candidate_id VARCHAR NOT NULL,
                frontier_type VARCHAR NOT NULL,
                rank INTEGER,
                PRIMARY KEY (candidate_id, frontier_type)
            );
            CREATE TABLE IF NOT EXISTS price_catalog (
                price_id VARCHAR PRIMARY KEY,
                sku_id VARCHAR NOT NULL,
                currency VARCHAR NOT NULL,
                gpu_hour DOUBLE NOT NULL,
                g2_gib_hour DOUBLE NOT NULL,
                network_gb DOUBLE NOT NULL,
                all_in_multiplier DOUBLE NOT NULL,
                source_url VARCHAR,
                source_date VARCHAR,
                basis VARCHAR,
                notes VARCHAR
            );
            CREATE TABLE IF NOT EXISTS replay_evaluation_cache (
                evaluation_cache_key VARCHAR PRIMARY KEY,
                cache_namespace_key VARCHAR NOT NULL,
                semantics_json JSON NOT NULL,
                metrics_json JSON NOT NULL,
                source_cell_id VARCHAR NOT NULL,
                source_stage VARCHAR NOT NULL,
                source_database VARCHAR,
                created_at VARCHAR NOT NULL,
                last_used_at VARCHAR NOT NULL,
                hit_count BIGINT NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS replay_cache_namespace
                ON replay_evaluation_cache(cache_namespace_key);
            CREATE TABLE IF NOT EXISTS variant_support_cache (
                variant_support_cache_key VARCHAR PRIMARY KEY,
                status VARCHAR NOT NULL,
                reason VARCHAR NOT NULL,
                failed_operators_json JSON NOT NULL,
                backend_version VARCHAR,
                data_source_json JSON NOT NULL,
                created_at VARCHAR NOT NULL,
                last_used_at VARCHAR NOT NULL,
                hit_count BIGINT NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS variant_resolution (
                experiment_id VARCHAR NOT NULL,
                resolution_id VARCHAR NOT NULL,
                model_id VARCHAR NOT NULL,
                sku_id VARCHAR NOT NULL,
                backend VARCHAR NOT NULL,
                variant_id VARCHAR NOT NULL,
                source VARCHAR NOT NULL,
                support_status VARCHAR NOT NULL,
                support_reason VARCHAR NOT NULL,
                data_mode VARCHAR NOT NULL,
                deployment_fidelity VARCHAR NOT NULL,
                selected BOOLEAN NOT NULL,
                selection_reason VARCHAR NOT NULL,
                support_cache_key VARCHAR NOT NULL,
                resolution_json JSON NOT NULL,
                created_at VARCHAR NOT NULL,
                PRIMARY KEY (experiment_id, resolution_id)
            );
            """
        )
        self.connection.execute(
            """
            ALTER TABLE evaluations
            ADD COLUMN IF NOT EXISTS evaluation_cache_key VARCHAR
            """
        )
        self.connection.execute(
            """
            ALTER TABLE matrix_cells
            ADD COLUMN IF NOT EXISTS model_variant_id VARCHAR
            """
        )
        self.connection.execute(
            """
            UPDATE matrix_cells
            SET model_variant_id = COALESCE(
                json_extract_string(cell_json, '$.model_variant.id'),
                model_id || '__legacy_implicit'
            )
            WHERE model_variant_id IS NULL
            """
        )
        self.connection.execute(
            """
            ALTER TABLE evaluations
            ADD COLUMN IF NOT EXISTS cache_hit BOOLEAN DEFAULT FALSE
            """
        )
        self.connection.execute(
            """
            ALTER TABLE resource_usage
            ADD COLUMN IF NOT EXISTS goodput_input_tokens DOUBLE DEFAULT 0
            """
        )
        for column in ("source_url", "source_date", "basis", "notes"):
            self.connection.execute(
                f"ALTER TABLE price_catalog ADD COLUMN IF NOT EXISTS {column} VARCHAR"
            )

    @_synchronized
    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "event_type": event_type,
            "timestamp": _utc_now(),
            "payload": jsonable(payload),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    @_synchronized
    def start_experiment(
        self, config: MatrixExperimentConfig, catalogs: CatalogBundle
    ) -> str:
        experiment_id = f"exp_{fingerprint({'config': config, 'artifacts': catalogs.artifacts})[:20]}"
        now = _utc_now()
        self.append_event(
            "experiment_started",
            {"experiment_id": experiment_id, "config": config},
        )
        self.connection.execute(
            """
            INSERT INTO experiments VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (experiment_id) DO UPDATE SET
                config_json = excluded.config_json,
                updated_at = excluded.updated_at,
                status = excluded.status
            """,
            [experiment_id, canonical_json(config), now, now, "running"],
        )
        for artifact in catalogs.artifacts:
            self.connection.execute(
                """
                INSERT INTO input_artifacts VALUES (?, ?, ?, ?)
                ON CONFLICT (experiment_id, role, path) DO UPDATE SET
                    sha256 = excluded.sha256
                """,
                [experiment_id, artifact.role, artifact.path, artifact.sha256],
            )
        return experiment_id

    @_synchronized
    def finish_experiment(self, experiment_id: str, status: str) -> None:
        self.append_event(
            "experiment_finished", {"experiment_id": experiment_id, "status": status}
        )
        self.connection.execute(
            "UPDATE experiments SET status = ?, updated_at = ? WHERE experiment_id = ?",
            [status, _utc_now(), experiment_id],
        )

    @_synchronized
    def record_input_artifact(
        self, experiment_id: str, *, role: str, path: str, sha256: str
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO input_artifacts VALUES (?, ?, ?, ?)
            ON CONFLICT (experiment_id, role, path) DO UPDATE SET sha256 = excluded.sha256
            """,
            [experiment_id, role, path, sha256],
        )

    @_synchronized
    def variant_support_cache_snapshot(self) -> dict[str, VariantSupportResult]:
        rows = self.connection.execute(
            """
            SELECT variant_support_cache_key, status, reason,
                   failed_operators_json, backend_version, data_source_json
            FROM variant_support_cache
            """
        ).fetchall()
        return {
            str(key): VariantSupportResult(
                status=status,
                reason=reason,
                failed_operators=json.loads(failed),
                backend_version=backend_version,
                data_source=json.loads(data_source),
                cache_key=str(key),
            )
            for key, status, reason, failed, backend_version, data_source in rows
        }

    @_synchronized
    def record_variant_resolution(
        self, experiment_id: str, record: VariantResolutionRecord
    ) -> None:
        now = _utc_now()
        support = record.support
        self.connection.execute(
            """
            INSERT INTO variant_support_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (variant_support_cache_key) DO UPDATE SET
                status = excluded.status,
                reason = excluded.reason,
                failed_operators_json = excluded.failed_operators_json,
                backend_version = excluded.backend_version,
                data_source_json = excluded.data_source_json,
                last_used_at = excluded.last_used_at,
                hit_count = variant_support_cache.hit_count + ?
            """,
            [
                support.cache_key,
                support.status,
                support.reason,
                canonical_json(support.failed_operators),
                support.backend_version,
                canonical_json(support.data_source),
                now,
                now,
                1 if support.cache_hit else 0,
                1 if support.cache_hit else 0,
            ],
        )
        self.connection.execute(
            """
            INSERT INTO variant_resolution VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT (experiment_id, resolution_id) DO UPDATE SET
                support_status = excluded.support_status,
                support_reason = excluded.support_reason,
                selected = excluded.selected,
                selection_reason = excluded.selection_reason,
                resolution_json = excluded.resolution_json
            """,
            [
                experiment_id,
                record.id,
                record.model_id,
                record.sku_id,
                record.backend,
                record.variant.id,
                record.source,
                support.status,
                support.reason,
                record.data_mode,
                record.variant.deployment_fidelity,
                record.selected,
                record.selection_reason,
                support.cache_key,
                canonical_json(record),
                now,
            ],
        )
        self.append_event(
            "variant_resolution",
            {"experiment_id": experiment_id, "record": record},
        )

    @_synchronized
    def import_variant_support_cache(self, paths: list[str | Path]) -> dict[str, Any]:
        imported = 0
        sources: list[str] = []
        skipped: list[str] = []
        for raw_path in paths:
            resolved = Path(raw_path).expanduser().resolve()
            database = (
                resolved if resolved.suffix == ".duckdb" else resolved / "sweep.duckdb"
            )
            if database == self.database_path.resolve():
                continue
            if not database.is_file():
                raise FileNotFoundError(
                    f"variant support cache source does not exist: {database}"
                )
            source = _connect_database(database, read_only=True)
            try:
                exists = source.execute(
                    """
                    SELECT count(*) FROM information_schema.tables
                    WHERE table_name = 'variant_support_cache'
                    """
                ).fetchone()[0]
                if not exists:
                    skipped.append(str(database))
                    continue
                rows = source.execute("SELECT * FROM variant_support_cache").fetchall()
            finally:
                source.close()
            sources.append(str(database))
            for row in rows:
                before = self.connection.execute(
                    "SELECT 1 FROM variant_support_cache WHERE variant_support_cache_key = ?",
                    [row[0]],
                ).fetchone()
                self.connection.execute(
                    """
                    INSERT INTO variant_support_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (variant_support_cache_key) DO NOTHING
                    """,
                    list(row),
                )
                if before is None:
                    imported += 1
        return {
            "imported": imported,
            "sources": sources,
            "skipped_legacy_sources": skipped,
        }

    @_synchronized
    def record_cell(
        self,
        experiment_id: str,
        cell: MatrixCell,
        *,
        status: str,
        detail: str = "",
    ) -> None:
        self.append_event(
            "cell_status",
            {
                "experiment_id": experiment_id,
                "cell_id": cell.id,
                "status": status,
                "detail": detail,
            },
        )
        self.connection.execute(
            """
            INSERT INTO matrix_cells (
                cell_id, experiment_id, fingerprint, model_id,
                model_variant_id, sku_id, workload_id, sla_id, support_tier,
                status, detail, cell_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (cell_id) DO UPDATE SET
                experiment_id = excluded.experiment_id,
                status = excluded.status,
                detail = excluded.detail,
                cell_json = excluded.cell_json,
                updated_at = excluded.updated_at
            """,
            [
                cell.id,
                experiment_id,
                cell.fingerprint,
                cell.model.id,
                cell.model_variant.id,
                cell.sku.id,
                cell.workload.id,
                cell.sla.id,
                cell.model_variant.support_tier,
                status,
                detail,
                canonical_json(cell),
                _utc_now(),
            ],
        )

    @_synchronized
    def has_successful_evaluation(self, cell_id: str, evaluation_key: str) -> bool:
        row = self.connection.execute(
            """
            SELECT 1 FROM evaluations
            WHERE cell_id = ? AND evaluation_key = ? AND outcome = 'feasible'
            LIMIT 1
            """,
            [cell_id, evaluation_key],
        ).fetchone()
        return row is not None

    @_synchronized
    def record_evaluation(
        self,
        *,
        cell_id: str,
        evaluation_key: str,
        stage: str,
        outcome: str,
        reason: str,
        selection: dict[str, Any],
        metrics: dict[str, float],
        sla_pass: bool | None,
        candidate: dict[str, Any] | None,
        resources: dict[str, float] | None = None,
        candidate_context: str | None = None,
        evaluation_cache_key: str | None = None,
        cache_hit: bool = False,
    ) -> str | None:
        evaluation_id = f"eval_{fingerprint({'cell_id': cell_id, 'stage': stage, 'key': evaluation_key})[:24]}"
        self.append_event(
            "evaluation",
            {
                "evaluation_id": evaluation_id,
                "evaluation_key": evaluation_key,
                "evaluation_cache_key": evaluation_cache_key,
                "cell_id": cell_id,
                "stage": stage,
                "outcome": outcome,
                "cache_hit": cache_hit,
                "reason": reason,
                "selection": selection,
                "metrics": metrics,
                "sla_pass": sla_pass,
                "candidate": candidate,
                "candidate_context": candidate_context,
                "resources": resources,
            },
        )
        self.connection.execute(
            """
            INSERT INTO evaluations (
                evaluation_id, evaluation_key, evaluation_cache_key, cell_id,
                stage, outcome, cache_hit, reason, selection_json,
                metrics_json, sla_pass, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (evaluation_id) DO UPDATE SET
                evaluation_cache_key = excluded.evaluation_cache_key,
                outcome = excluded.outcome,
                cache_hit = excluded.cache_hit,
                reason = excluded.reason,
                selection_json = excluded.selection_json,
                metrics_json = excluded.metrics_json,
                sla_pass = excluded.sla_pass,
                created_at = excluded.created_at
            """,
            [
                evaluation_id,
                evaluation_key,
                evaluation_cache_key,
                cell_id,
                stage,
                outcome,
                cache_hit,
                reason,
                canonical_json(selection),
                canonical_json(metrics),
                sla_pass,
                _utc_now(),
            ],
        )
        if candidate is None:
            return None

        config = dict(candidate["config"])
        candidate_id = f"cand_{fingerprint({'cell_id': cell_id, 'stage': stage, 'context': candidate_context, 'config': config})[:24]}"
        compute_config = {
            key: value
            for key, value in config.items()
            if key
            not in {
                "num_g2_blocks",
                "kv_bytes_per_token",
                "kvbm_footprint_contract",
                "kvbm_physical_bytes_per_token_estimate",
                "bandwidth_g1_to_g2_gbps",
                "bandwidth_g2_to_g1_gbps",
                "offload_batch_size",
                "g2_capacity_gib",
                "g2_capacity_scope",
                "g2_effective_capacity_gib_per_rank",
                "g2_effective_total_capacity_gib",
                "g2_target_hit_rate",
                "kvbm_deployability",
                "kvbm_profile_id",
                "host_cache_hit_weight",
                "disk_cache_hit_weight",
                "capacity_axis",
                "capacity_value",
                "capacity_search_status",
                "capacity_search_right_censored",
                "capacity_search_feasible_lower_bound",
                "capacity_search_infeasible_bound",
                "max_sla_rps",
                "max_sla_arrival_speedup_ratio",
                "max_sla_effective_rps",
            }
        }
        self.connection.execute(
            """
            INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (candidate_id) DO UPDATE SET
                score = excluded.score,
                objectives_json = excluded.objectives_json,
                config_json = excluded.config_json
            """,
            [
                candidate_id,
                evaluation_id,
                cell_id,
                float(candidate["score"]),
                int(candidate["used_gpus"]),
                canonical_json(candidate.get("objectives"))
                if candidate.get("objectives") is not None
                else None,
                canonical_json(config),
                fingerprint(compute_config),
                config.get("pd_profile_id"),
                config.get("kvbm_profile_id"),
            ],
        )
        usage = resources or {}
        self.connection.execute(
            """
            INSERT INTO resource_usage (
                candidate_id, gpu_hours, g2_gib_hours, network_gb,
                input_tokens, output_tokens, goodput_input_tokens,
                goodput_output_tokens, duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (candidate_id) DO UPDATE SET
                gpu_hours = excluded.gpu_hours,
                g2_gib_hours = excluded.g2_gib_hours,
                network_gb = excluded.network_gb,
                input_tokens = excluded.input_tokens,
                output_tokens = excluded.output_tokens,
                goodput_input_tokens = excluded.goodput_input_tokens,
                goodput_output_tokens = excluded.goodput_output_tokens,
                duration_ms = excluded.duration_ms
            """,
            [
                candidate_id,
                float(usage.get("gpu_hours", 0.0)),
                float(usage.get("g2_gib_hours", 0.0)),
                float(usage.get("network_gb", 0.0)),
                float(usage.get("input_tokens", 0.0)),
                float(usage.get("output_tokens", 0.0)),
                float(usage.get("goodput_input_tokens", 0.0)),
                float(usage.get("goodput_output_tokens", 0.0)),
                float(usage.get("duration_ms", 0.0)),
            ],
        )
        return candidate_id

    @_synchronized
    def replay_cache_snapshot(
        self, cache_namespace_key: str
    ) -> dict[str, dict[str, float]]:
        rows = self.connection.execute(
            """
            SELECT evaluation_cache_key, metrics_json
            FROM replay_evaluation_cache
            WHERE cache_namespace_key = ?
            """,
            [cache_namespace_key],
        ).fetchall()
        return {
            str(key): {
                str(metric): float(value)
                for metric, value in json.loads(raw_metrics).items()
            }
            for key, raw_metrics in rows
        }

    @_synchronized
    def store_replay_cache_entry(
        self,
        *,
        evaluation_cache_key: str,
        cache_namespace_key: str,
        semantics: dict[str, Any],
        metrics: dict[str, float],
        source_cell_id: str,
        source_stage: str,
        source_database: str | None = None,
    ) -> None:
        now = _utc_now()
        self.connection.execute(
            """
            INSERT INTO replay_evaluation_cache (
                evaluation_cache_key, cache_namespace_key, semantics_json,
                metrics_json, source_cell_id, source_stage, source_database,
                created_at, last_used_at, hit_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT (evaluation_cache_key) DO UPDATE SET
                cache_namespace_key = excluded.cache_namespace_key,
                semantics_json = excluded.semantics_json,
                metrics_json = excluded.metrics_json,
                last_used_at = excluded.last_used_at
            """,
            [
                evaluation_cache_key,
                cache_namespace_key,
                canonical_json(semantics),
                canonical_json(metrics),
                source_cell_id,
                source_stage,
                source_database,
                now,
                now,
            ],
        )

    @_synchronized
    def touch_replay_cache_entry(self, evaluation_cache_key: str) -> None:
        self.connection.execute(
            """
            UPDATE replay_evaluation_cache
            SET last_used_at = ?, hit_count = hit_count + 1
            WHERE evaluation_cache_key = ?
            """,
            [_utc_now(), evaluation_cache_key],
        )

    @_synchronized
    def import_replay_cache(self, paths: list[str | Path]) -> dict[str, Any]:
        """Copy exact replay entries from other result databases.

        Older databases without ``replay_evaluation_cache`` are reported but are
        not heuristically backfilled: their stored rows lack a complete evaluator
        contract, so guessing would violate exact-match reuse.
        """
        imported = 0
        skipped_sources: list[str] = []
        sources: list[str] = []
        for raw_path in paths:
            resolved = Path(raw_path).expanduser().resolve()
            database = (
                resolved if resolved.suffix == ".duckdb" else resolved / "sweep.duckdb"
            )
            if database == self.database_path.resolve():
                continue
            if not database.is_file():
                raise FileNotFoundError(
                    f"replay cache source does not exist: {database}"
                )
            source = _connect_database(database, read_only=True)
            try:
                exists = source.execute(
                    """
                    SELECT count(*) FROM information_schema.tables
                    WHERE table_name = 'replay_evaluation_cache'
                    """
                ).fetchone()[0]
                if not exists:
                    skipped_sources.append(str(database))
                    continue
                rows = source.execute(
                    """
                    SELECT evaluation_cache_key, cache_namespace_key,
                           semantics_json, metrics_json, source_cell_id,
                           source_stage, created_at, last_used_at, hit_count
                    FROM replay_evaluation_cache
                    """
                ).fetchall()
            finally:
                source.close()
            sources.append(str(database))
            for row in rows:
                before = self.connection.execute(
                    """
                    SELECT 1 FROM replay_evaluation_cache
                    WHERE evaluation_cache_key = ?
                    """,
                    [row[0]],
                ).fetchone()
                self.connection.execute(
                    """
                    INSERT INTO replay_evaluation_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (evaluation_cache_key) DO NOTHING
                    """,
                    [*row[:6], str(database), *row[6:]],
                )
                if before is None:
                    imported += 1
        return {
            "imported": imported,
            "sources": sources,
            "skipped_legacy_sources": skipped_sources,
        }

    @_synchronized
    def mark_frontier(
        self, candidate_ids: list[str], *, frontier_type: str = "strict_pareto"
    ) -> None:
        for rank, candidate_id in enumerate(candidate_ids, start=1):
            self.connection.execute(
                """
                INSERT INTO frontier_membership VALUES (?, ?, ?)
                ON CONFLICT (candidate_id, frontier_type) DO UPDATE SET rank = excluded.rank
                """,
                [candidate_id, frontier_type, rank],
            )

    @_synchronized
    def candidate_ids_for_cell(self, cell_id: str, *, stage: str) -> list[str]:
        rows = self.connection.execute(
            """
            SELECT c.candidate_id
            FROM candidates c
            JOIN evaluations e USING (evaluation_id)
            WHERE c.cell_id = ? AND e.stage = ? AND e.outcome = 'feasible'
              AND COALESCE(e.sla_pass, TRUE)
            ORDER BY c.score DESC, c.candidate_id
            """,
            [cell_id, stage],
        ).fetchall()
        return [str(row[0]) for row in rows]

    @_synchronized
    def candidates_for_cell(
        self, cell_id: str, *, stage: str
    ) -> list[tuple[str, dict[str, Any]]]:
        """Return replayable candidate payloads, including resumed studies."""

        rows = self.connection.execute(
            """
            SELECT c.candidate_id, c.config_json, c.used_gpus, c.score,
                   c.objectives_json, e.metrics_json
            FROM candidates c
            JOIN evaluations e USING (evaluation_id)
            WHERE c.cell_id = ? AND e.stage = ? AND e.outcome = 'feasible'
              AND COALESCE(e.sla_pass, TRUE)
            ORDER BY c.score DESC, c.candidate_id
            """,
            [cell_id, stage],
        ).fetchall()
        result: list[tuple[str, dict[str, Any]]] = []
        for candidate_id, config, used_gpus, score, objectives, metrics in rows:
            result.append(
                (
                    str(candidate_id),
                    {
                        "config": json.loads(config),
                        "used_gpus": int(used_gpus),
                        "score": float(score),
                        "objectives": None
                        if objectives is None
                        else json.loads(objectives),
                        "metrics": json.loads(metrics),
                    },
                )
            )
        return result

    @_synchronized
    def selected_operating_point_loads(
        self, cell_id: str, *, mode: str
    ) -> list[float]:
        """Resume AIC load discovery for an unchanged MatrixCell."""

        rows = self.connection.execute(
            """
            SELECT DISTINCT CAST(json_extract(metrics_json, '$.concurrency') AS DOUBLE)
            FROM evaluations
            WHERE cell_id = ? AND stage = 'aic_operating_point'
              AND outcome = 'selected'
              AND json_extract_string(selection_json, '$.mode') = ?
            ORDER BY 1
            """,
            [cell_id, mode],
        ).fetchall()
        return [float(row[0]) for row in rows if row[0] is not None]

    @_synchronized
    def mark_cell_frontiers(
        self,
        cell_id: str,
        candidate_ids: list[str],
        *,
        epsilon: float,
        frontier_prefix: str | None = None,
    ) -> dict[str, list[str]]:
        """Recompute strict and epsilon fronts after capacity/robustness studies."""
        unique_ids = list(dict.fromkeys(candidate_ids))
        if not unique_ids:
            return {"strict_pareto": [], "epsilon_pareto": []}
        placeholders = ", ".join("?" for _ in unique_ids)
        rows = self.connection.execute(
            f"""
            SELECT candidate_id, score, objectives_json
            FROM candidates
            WHERE cell_id = ? AND candidate_id IN ({placeholders})
            ORDER BY score DESC, candidate_id
            """,
            [cell_id, *unique_ids],
        ).fetchall()
        parsed: list[tuple[str, float, dict[str, float]]] = []
        for candidate_id, score, raw_objectives in rows:
            objectives_payload = json.loads(raw_objectives)
            if not isinstance(objectives_payload, dict) or not objectives_payload:
                continue
            objectives = {
                str(key): float(value) for key, value in objectives_payload.items()
            }
            parsed.append((str(candidate_id), float(score), objectives))

        def dominated(
            candidate: tuple[str, float, dict[str, float]], *, tolerance: float
        ) -> bool:
            _, _, objectives = candidate
            for other in parsed:
                if other[0] == candidate[0] or other[2].keys() != objectives.keys():
                    continue
                thresholds = {
                    key: value * (1.0 + tolerance) for key, value in objectives.items()
                }
                if all(
                    other[2][key] >= threshold for key, threshold in thresholds.items()
                ) and any(
                    other[2][key] > threshold for key, threshold in thresholds.items()
                ):
                    return True
            return False

        strict_type = (
            "strict_pareto"
            if frontier_prefix is None
            else f"{frontier_prefix}:strict_pareto"
        )
        epsilon_type = (
            "epsilon_pareto"
            if frontier_prefix is None
            else f"{frontier_prefix}:epsilon_pareto"
        )
        fronts = {
            strict_type: [
                item[0] for item in parsed if not dominated(item, tolerance=0.0)
            ],
            epsilon_type: [
                item[0] for item in parsed if not dominated(item, tolerance=epsilon)
            ],
        }
        for frontier_type, members in fronts.items():
            self.connection.execute(
                """
                DELETE FROM frontier_membership
                WHERE frontier_type = ? AND candidate_id IN (
                    SELECT candidate_id FROM candidates WHERE cell_id = ?
                )
                """,
                [frontier_type, cell_id],
            )
            self.mark_frontier(members, frontier_type=frontier_type)
        return fronts

    @_synchronized
    def install_prices(self, prices) -> None:
        for raw_price in prices:
            price = (
                raw_price
                if isinstance(raw_price, PriceSpec)
                else PriceSpec.model_validate(raw_price)
            )
            self.connection.execute(
                """
                INSERT INTO price_catalog (
                    price_id, sku_id, currency, gpu_hour, g2_gib_hour,
                    network_gb, all_in_multiplier, source_url, source_date,
                    basis, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (price_id) DO UPDATE SET
                    sku_id = excluded.sku_id,
                    currency = excluded.currency,
                    gpu_hour = excluded.gpu_hour,
                    g2_gib_hour = excluded.g2_gib_hour,
                    network_gb = excluded.network_gb,
                    all_in_multiplier = excluded.all_in_multiplier,
                    source_url = excluded.source_url,
                    source_date = excluded.source_date,
                    basis = excluded.basis,
                    notes = excluded.notes
                """,
                [
                    price.id,
                    price.sku_id,
                    price.currency,
                    price.gpu_hour,
                    price.g2_gib_hour,
                    price.network_gb,
                    price.all_in_multiplier,
                    price.source_url,
                    price.source_date,
                    price.basis,
                    price.notes,
                ],
            )

    @_synchronized
    def replace_prices(self, prices) -> None:
        """Replace report-time pricing without changing simulation records."""
        self.connection.execute("DELETE FROM price_catalog")
        self.install_prices(prices)

    @_synchronized
    def create_analysis_views(self, topk: int = 20) -> None:
        self.connection.execute(
            f"""
            CREATE OR REPLACE VIEW candidate_costs AS
            SELECT
                c.candidate_id,
                c.cell_id,
                mc.model_id,
                mc.model_variant_id,
                mc.sku_id,
                mc.workload_id,
                COALESCE(json_extract_string(mc.cell_json,
                    '$.model_variant.deployment_fidelity'), 'deployable')
                    AS deployment_fidelity,
                json_extract_string(mc.cell_json,
                    '$.variant_resolution.source') AS variant_resolution_source,
                json_extract_string(mc.cell_json,
                    '$.variant_resolution.support.status') AS variant_support_status,
                c.score,
                c.used_gpus,
                c.pd_profile_id,
                json_extract_string(c.config_json, '$.pd_bandwidth_domain')
                    AS pd_bandwidth_domain,
                CAST(json_extract(c.config_json, '$.pd_group_gpus') AS INTEGER)
                    AS pd_group_gpus,
                CAST(json_extract(c.config_json,
                    '$.pd_transfer_bandwidth_gbytes_per_s') AS DOUBLE)
                    AS pd_transfer_bandwidth_gbytes_per_s,
                c.kvbm_profile_id,
                e.stage,
                p.price_id,
                p.currency,
                r.* EXCLUDE (candidate_id),
                r.gpu_hours * p.gpu_hour AS gpu_only_cost,
                (r.gpu_hours * p.gpu_hour
                  + r.g2_gib_hours * p.g2_gib_hour
                  + r.network_gb * p.network_gb) * p.all_in_multiplier AS all_in_cost,
                CASE WHEN r.goodput_input_tokens + r.goodput_output_tokens > 0 THEN
                    1000000.0 * r.gpu_hours * p.gpu_hour
                    / (r.goodput_input_tokens + r.goodput_output_tokens) END
                    AS gpu_only_cost_per_million_total_tokens,
                CASE WHEN r.goodput_input_tokens > 0 THEN
                    1000000.0 * r.gpu_hours * p.gpu_hour
                    / r.goodput_input_tokens END
                    AS gpu_only_cost_per_million_input_tokens,
                CASE WHEN r.goodput_output_tokens > 0 THEN
                    1000000.0 * r.gpu_hours * p.gpu_hour
                    / r.goodput_output_tokens END
                    AS gpu_only_cost_per_million_output_tokens,
                CASE WHEN r.goodput_input_tokens + r.goodput_output_tokens > 0 THEN
                    1000000.0 * (r.gpu_hours * p.gpu_hour
                      + r.g2_gib_hours * p.g2_gib_hour
                      + r.network_gb * p.network_gb) * p.all_in_multiplier
                    / (r.goodput_input_tokens + r.goodput_output_tokens) END
                    AS all_in_cost_per_million_total_tokens,
                CASE WHEN r.goodput_input_tokens > 0 THEN
                    1000000.0 * (r.gpu_hours * p.gpu_hour
                      + r.g2_gib_hours * p.g2_gib_hour
                      + r.network_gb * p.network_gb) * p.all_in_multiplier
                    / r.goodput_input_tokens END
                    AS all_in_cost_per_million_input_tokens,
                CASE WHEN r.goodput_output_tokens > 0 THEN
                    1000000.0 * (r.gpu_hours * p.gpu_hour
                      + r.g2_gib_hours * p.g2_gib_hour
                      + r.network_gb * p.network_gb) * p.all_in_multiplier
                    / r.goodput_output_tokens END
                    AS all_in_cost_per_million_output_tokens,
                e.metrics_json,
                c.config_json,
                c.compute_config_key
            FROM candidates c
            JOIN evaluations e ON e.evaluation_id = c.evaluation_id
            JOIN matrix_cells mc ON mc.cell_id = c.cell_id
            JOIN resource_usage r ON r.candidate_id = c.candidate_id
            JOIN price_catalog p ON p.sku_id = mc.sku_id
            WHERE e.outcome = 'feasible' AND COALESCE(e.sla_pass, TRUE)
              AND (
                e.stage = 'capacity'
                OR (
                  e.stage = 'spica'
                  AND NOT EXISTS (
                    SELECT 1
                    FROM candidates c2
                    JOIN evaluations e2 ON e2.evaluation_id = c2.evaluation_id
                    WHERE c2.cell_id = c.cell_id
                      AND e2.stage = 'capacity'
                      AND e2.outcome = 'feasible'
                      AND COALESCE(e2.sla_pass, TRUE)
                  )
                )
              );

            CREATE OR REPLACE VIEW deployable_candidate_costs AS
            SELECT * FROM candidate_costs
            WHERE deployment_fidelity = 'deployable';

            CREATE OR REPLACE VIEW what_if_candidate_costs AS
            SELECT * FROM candidate_costs
            WHERE deployment_fidelity = 'hypothetical';

            CREATE OR REPLACE VIEW topk_cost AS
            SELECT * EXCLUDE (position) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY cell_id, price_id
                    ORDER BY all_in_cost_per_million_total_tokens NULLS LAST, used_gpus
                ) AS position
                FROM deployable_candidate_costs
            ) WHERE position <= {int(topk)};

            CREATE OR REPLACE VIEW topk_score AS
            SELECT * EXCLUDE (position) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY cell_id, price_id ORDER BY score DESC, used_gpus
                ) AS position
                FROM deployable_candidate_costs
            ) WHERE position <= {int(topk)};

            CREATE OR REPLACE VIEW candidate_dimensions AS
            SELECT *,
                COALESCE(
                    CAST(json_extract(config_json, '$.max_sla_rps') AS DOUBLE),
                    CAST(json_extract(config_json, '$.max_sla_effective_rps') AS DOUBLE)
                ) AS sla_rps,
                CAST(json_extract(metrics_json,
                    '$.mean_output_token_throughput_per_user') AS DOUBLE) AS tps_per_user,
                CAST(json_extract(metrics_json, '$.mean_ttft_ms') AS DOUBLE) AS ttft_ms,
                CAST(json_extract(metrics_json, '$.mean_tpot_ms') AS DOUBLE) AS tpot_ms,
                COALESCE(
                    CAST(json_extract(config_json,
                        '$.g2_effective_total_capacity_gib') AS DOUBLE),
                    CAST(json_extract(config_json, '$.g2_capacity_gib') AS DOUBLE),
                    0
                )
                    AS g2_capacity_gib,
                GREATEST(
                    COALESCE(CAST(json_extract(config_json,
                        '$.bandwidth_g1_to_g2_gbps') AS DOUBLE), 0),
                    COALESCE(CAST(json_extract(config_json,
                        '$.bandwidth_g2_to_g1_gbps') AS DOUBLE), 0)
                ) AS g2_bandwidth_gbytes_per_s
            FROM deployable_candidate_costs;

            CREATE OR REPLACE VIEW topk_rps AS
            SELECT * EXCLUDE (position) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY cell_id, price_id ORDER BY sla_rps DESC NULLS LAST, used_gpus
                ) AS position FROM candidate_dimensions
            ) WHERE position <= {int(topk)};

            CREATE OR REPLACE VIEW topk_tpsu AS
            SELECT * EXCLUDE (position) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY cell_id, price_id ORDER BY tps_per_user DESC NULLS LAST, used_gpus
                ) AS position FROM candidate_dimensions
            ) WHERE position <= {int(topk)};

            CREATE OR REPLACE VIEW topk_ttft AS
            SELECT * EXCLUDE (position) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY cell_id, price_id ORDER BY ttft_ms NULLS LAST, used_gpus
                ) AS position FROM candidate_dimensions
            ) WHERE position <= {int(topk)};

            CREATE OR REPLACE VIEW topk_tpot AS
            SELECT * EXCLUDE (position) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY cell_id, price_id ORDER BY tpot_ms NULLS LAST, used_gpus
                ) AS position FROM candidate_dimensions
            ) WHERE position <= {int(topk)};

            CREATE OR REPLACE VIEW topk_gpus AS
            SELECT * EXCLUDE (position) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY cell_id, price_id ORDER BY used_gpus, score DESC
                ) AS position FROM candidate_dimensions
            ) WHERE position <= {int(topk)};

            CREATE OR REPLACE VIEW topk_g2_capacity AS
            SELECT * EXCLUDE (position) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY cell_id, price_id ORDER BY g2_capacity_gib, score DESC
                ) AS position FROM candidate_dimensions
            ) WHERE position <= {int(topk)};

            CREATE OR REPLACE VIEW topk_g2_bandwidth AS
            SELECT * EXCLUDE (position) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY cell_id, price_id
                    ORDER BY g2_bandwidth_gbytes_per_s, score DESC
                ) AS position FROM candidate_dimensions
            ) WHERE position <= {int(topk)};

            CREATE OR REPLACE VIEW candidate_topk AS
                SELECT 'cost' AS dimension, * FROM topk_cost
                UNION ALL SELECT 'score', * FROM topk_score
                UNION ALL SELECT 'rps', * EXCLUDE (
                    sla_rps, tps_per_user, ttft_ms, tpot_ms,
                    g2_capacity_gib, g2_bandwidth_gbytes_per_s
                ) FROM topk_rps
                UNION ALL SELECT 'tps_per_user', * EXCLUDE (
                    sla_rps, tps_per_user, ttft_ms, tpot_ms,
                    g2_capacity_gib, g2_bandwidth_gbytes_per_s
                ) FROM topk_tpsu
                UNION ALL SELECT 'ttft', * EXCLUDE (
                    sla_rps, tps_per_user, ttft_ms, tpot_ms,
                    g2_capacity_gib, g2_bandwidth_gbytes_per_s
                ) FROM topk_ttft
                UNION ALL SELECT 'tpot', * EXCLUDE (
                    sla_rps, tps_per_user, ttft_ms, tpot_ms,
                    g2_capacity_gib, g2_bandwidth_gbytes_per_s
                ) FROM topk_tpot
                UNION ALL SELECT 'gpu_count', * EXCLUDE (
                    sla_rps, tps_per_user, ttft_ms, tpot_ms,
                    g2_capacity_gib, g2_bandwidth_gbytes_per_s
                ) FROM topk_gpus
                UNION ALL SELECT 'g2_capacity', * EXCLUDE (
                    sla_rps, tps_per_user, ttft_ms, tpot_ms,
                    g2_capacity_gib, g2_bandwidth_gbytes_per_s
                ) FROM topk_g2_capacity
                UNION ALL SELECT 'g2_bandwidth', * EXCLUDE (
                    sla_rps, tps_per_user, ttft_ms, tpot_ms,
                    g2_capacity_gib, g2_bandwidth_gbytes_per_s
                ) FROM topk_g2_bandwidth;

            CREATE OR REPLACE VIEW kvbm_uplift AS
            WITH baseline AS (
                SELECT cell_id, compute_config_key, price_id,
                       min(all_in_cost_per_million_total_tokens) AS baseline_cost,
                       max(sla_rps) AS baseline_rps
                FROM candidate_dimensions
                WHERE COALESCE(CAST(json_extract(config_json,
                        '$.num_g2_blocks') AS BIGINT), 0) = 0
                GROUP BY cell_id, compute_config_key, price_id
            ), system_baseline AS (
                SELECT cell_id, price_id,
                       min(all_in_cost_per_million_total_tokens) AS system_baseline_cost,
                       max(sla_rps) AS system_baseline_rps
                FROM candidate_dimensions
                WHERE COALESCE(CAST(json_extract(config_json,
                        '$.num_g2_blocks') AS BIGINT), 0) = 0
                GROUP BY cell_id, price_id
            )
            SELECT c.*,
                   b.baseline_cost,
                   b.baseline_rps,
                   sb.system_baseline_cost,
                   sb.system_baseline_rps,
                   b.baseline_cost - c.all_in_cost_per_million_total_tokens
                       AS matched_config_cost_reduction,
                   (b.baseline_cost - c.all_in_cost_per_million_total_tokens)
                       / NULLIF(b.baseline_cost, 0) AS matched_config_cost_reduction_ratio,
                   sb.system_baseline_cost - c.all_in_cost_per_million_total_tokens
                       AS system_optimal_cost_reduction,
                   (sb.system_baseline_cost - c.all_in_cost_per_million_total_tokens)
                       / NULLIF(sb.system_baseline_cost, 0)
                       AS system_optimal_cost_reduction_ratio,
                   c.sla_rps - b.baseline_rps AS matched_config_rps_increase,
                   (c.sla_rps - b.baseline_rps) / NULLIF(b.baseline_rps, 0)
                       AS matched_config_rps_increase_ratio,
                   c.sla_rps - sb.system_baseline_rps AS system_optimal_rps_increase,
                   (c.sla_rps - sb.system_baseline_rps) / NULLIF(sb.system_baseline_rps, 0)
                       AS system_optimal_rps_increase_ratio
            FROM candidate_dimensions c
            LEFT JOIN baseline b USING (cell_id, compute_config_key, price_id)
            LEFT JOIN system_baseline sb USING (cell_id, price_id)
            WHERE COALESCE(CAST(json_extract(c.config_json,
                    '$.num_g2_blocks') AS BIGINT), 0) > 0;
            """
        )

    @_synchronized
    def summary(self) -> dict[str, Any]:
        counts = self.connection.execute(
            """
            SELECT
              (SELECT count(*) FROM matrix_cells) AS cells,
              (SELECT count(*) FROM evaluations) AS evaluations,
              (SELECT count(*) FROM candidates) AS candidates,
              (SELECT count(*) FROM frontier_membership WHERE frontier_type='strict_pareto') AS pareto,
              (SELECT count(*) FROM matrix_cells WHERE status='complete') AS complete_cells,
              (SELECT count(*) FROM matrix_cells WHERE status='failed') AS failed_cells,
              (SELECT count(*) FROM matrix_cells WHERE status='unsupported') AS unsupported_cells,
              (SELECT count(*) FROM replay_evaluation_cache) AS replay_cache_entries,
              (SELECT COALESCE(sum(hit_count), 0) FROM replay_evaluation_cache) AS replay_cache_hits,
              (SELECT count(*) FROM variant_resolution) AS variant_resolutions,
              (SELECT count(*) FROM variant_support_cache) AS variant_support_cache_entries,
              (SELECT COALESCE(sum(hit_count), 0) FROM variant_support_cache) AS variant_support_cache_hits
            """
        ).fetchone()
        if counts is None:
            return {}
        return {
            "cells": counts[0],
            "evaluations": counts[1],
            "candidates": counts[2],
            "pareto": counts[3],
            "complete_cells": counts[4],
            "failed_cells": counts[5],
            "unsupported_cells": counts[6],
            "replay_cache_entries": counts[7],
            "replay_cache_hits": counts[8],
            "variant_resolutions": counts[9],
            "variant_support_cache_entries": counts[10],
            "variant_support_cache_hits": counts[11],
        }

    @_synchronized
    def export_parquet(self) -> None:
        exports = {
            "evaluations.parquet": "SELECT * FROM evaluations",
            "candidates.parquet": "SELECT * FROM candidates",
            "pareto.parquet": """
                SELECT c.*, f.frontier_type, f.rank
                FROM candidates c
                JOIN frontier_membership f USING (candidate_id)
                WHERE f.frontier_type = 'strict_pareto'
            """,
            "topk.parquet": "SELECT * FROM candidate_topk",
            "resource_usage.parquet": "SELECT * FROM resource_usage",
            "replay_evaluation_cache.parquet": "SELECT * FROM replay_evaluation_cache",
            "variant_resolution.parquet": "SELECT * FROM variant_resolution",
            "variant_support_cache.parquet": "SELECT * FROM variant_support_cache",
        }
        for filename, query in exports.items():
            path = str((self.output_dir / filename).resolve()).replace("'", "''")
            self.connection.execute(
                f"COPY ({query}) TO '{path}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)"
            )
