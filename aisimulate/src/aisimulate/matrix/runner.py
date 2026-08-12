# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matrix orchestration over conservative pruning and per-cell Spica studies."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from aisimulate.spica import (
    Candidate,
    OptimizationGoal,
    OptimizationTarget,
    SearchEvaluation,
    SearchSpace,
    SLATarget,
    SmartSearchConfig,
    SweepConfig,
    Workload,
    run_smart_search_with_history,
)
from aisimulate.spica.deploy import build_deployment
from aisimulate.spica.score import make_candidate
from aisimulate.spica.score import pareto_front

from ._utils import fingerprint
from .catalog import load_catalogs
from .config import (
    CatalogBundle,
    KVBMProfile,
    MatrixExperimentConfig,
    PDProfile,
    SLAProfile,
)
from .expand import (
    MatrixCell,
    expand_matrix,
    resolve_matrix_variants,
    validate_experiment,
)
from .operating_points import (
    AICOperatingPointResolution,
    resolve_aic_operating_points,
)
from .provenance import capture_runtime_provenance
from .prune import StaticPruneResult, prune_parallel_configs
from .replay_cache import (
    CacheAwareReplayEvaluator,
    CachedReplayResult,
    ReplayCacheDescription,
)
from .store import ResultStore
from .variant import VariantResolution, support_accepted
from .workload import PreparedWorkload, materialize_workload


class MatrixRunner:
    """Single-writer experiment runner; Spica owns per-cell process parallelism."""

    def run(
        self,
        config: MatrixExperimentConfig,
        result_store: ResultStore,
        resume: bool = True,
        reuse_from: list[str | Path] | None = None,
    ) -> dict[str, Any]:
        report = validate_experiment(config)
        if not report.valid:
            messages = "; ".join(issue.message for issue in report.issues)
            raise ValueError(f"matrix experiment validation failed: {messages}")
        catalogs = load_catalogs(config)
        experiment_id = result_store.start_experiment(config, catalogs)
        try:
            reuse_paths = reuse_from or []
            variant_cache_import = result_store.import_variant_support_cache(
                reuse_paths
            )
            resolution = resolve_matrix_variants(
                config,
                catalogs,
                support_cache=result_store.variant_support_cache_snapshot(),
            )
            for resolution_record in resolution.records:
                result_store.record_variant_resolution(experiment_id, resolution_record)
            cells = expand_matrix(config, catalogs, resolution=resolution)
            cache_import = result_store.import_replay_cache(reuse_paths)
            for cell in cells:
                if (
                    cell.workload.source_path is not None
                    and cell.workload.source_sha256 is not None
                ):
                    result_store.record_input_artifact(
                        experiment_id,
                        role=f"workload:{cell.workload.id}",
                        path=cell.workload.source_path,
                        sha256=cell.workload.source_sha256,
                    )
            completed = 0
            unsupported = 0
            failed = 0
            run_cell = partial(
                self._run_cell,
                experiment_id,
                store=result_store,
                resume=resume,
                simulator_revision=config.run.simulator_revision,
            )
            if config.run.cell_workers == 1:
                outcomes = map(run_cell, cells)
            else:
                with ThreadPoolExecutor(
                    max_workers=config.run.cell_workers,
                    thread_name_prefix="matrix-cell",
                ) as executor:
                    outcomes = list(executor.map(run_cell, cells))
            for outcome in outcomes:
                if outcome == "complete":
                    completed += 1
                elif outcome == "unsupported":
                    unsupported += 1
                else:
                    failed += 1
            status = "complete" if failed == 0 and unsupported == 0 else "partial"
            result_store.create_analysis_views(topk=config.run.topk)
            result_store.finish_experiment(experiment_id, status)
            self._write_manifest(
                result_store,
                config,
                catalogs,
                cells,
                experiment_id,
                status,
                cache_import,
                variant_cache_import,
                resolution,
            )
            result_store.export_parquet()
        except Exception:
            result_store.finish_experiment(experiment_id, "failed")
            raise
        return {
            "experiment_id": experiment_id,
            "status": status,
            "cells": len(cells),
            "completed": completed,
            "unsupported": unsupported,
            "failed": failed,
            **result_store.summary(),
            "replay_cache_import": cache_import,
            "variant_support_cache_import": variant_cache_import,
        }

    def _run_cell(
        self,
        experiment_id: str,
        cell: MatrixCell,
        store: ResultStore,
        *,
        resume: bool,
        simulator_revision: str | None,
    ) -> str:
        support = cell.variant_resolution.support
        if not support_accepted(support, cell.variant_resolution.data_mode):
            detail = f"variant support status={support.status}: {support.reason}"
            store.record_cell(
                experiment_id,
                cell,
                status="unsupported",
                detail=detail,
            )
            return "unsupported"
        if cell.model_variant.support_tier == "unsupported":
            store.record_cell(
                experiment_id,
                cell,
                status="unsupported",
                detail="model variant catalog support_tier=unsupported",
            )
            return "unsupported"

        prepared = materialize_workload(
            cell.workload, store.output_dir / "materialized_workloads"
        )
        store.record_cell(experiment_id, cell, status="running")
        final_candidate_ids: list[str] = []
        study_failures: list[str] = []
        for mode in cell.search_policy.spica.deployment_mode:
            if mode == "agg" and not prepared.capabilities.aggregate:
                continue
            if mode == "disagg" and not prepared.capabilities.disaggregate:
                continue
            try:
                load_points = self._load_points_for_mode(
                    store,
                    cell,
                    prepared,
                    mode,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
                study_failures.append(reason)
                store.record_evaluation(
                    cell_id=cell.id,
                    evaluation_key=f"operating-points:{mode}:{fingerprint(reason)}",
                    stage="aic_operating_point",
                    outcome="unsupported",
                    reason=reason,
                    selection={"mode": mode},
                    metrics={},
                    sla_pass=None,
                    candidate=None,
                )
                continue
            pd_profiles: list[PDProfile | None] = (
                cell.search_policy.pd_profiles if mode == "disagg" else [None]
            )
            for pd_profile in pd_profiles:
                for kvbm_profile in cell.search_policy.kvbm_profiles:
                    for load_name, load_value in load_points:
                        study_key = fingerprint(
                            {
                                "cell": cell.fingerprint,
                                "mode": mode,
                                "pd_profile": pd_profile,
                                "kvbm_profile": kvbm_profile,
                                "load_name": load_name,
                                "load_value": load_value,
                            }
                        )
                        if resume and store.has_successful_evaluation(
                            cell.id, f"substudy:{study_key}"
                        ):
                            continue
                        try:
                            search_config = self._smart_search_config(
                                cell,
                                prepared,
                                mode,
                                pd_profile,
                                kvbm_profile,
                                load_name,
                                load_value,
                            )
                            prune_result = prune_parallel_configs(
                                search_config,
                                cell.model,
                                cell.search_policy.static_prune,
                            )
                            self._record_prune(store, cell, study_key, prune_result)
                            cache_evaluator = self._cache_evaluator(
                                store,
                                cell,
                                prepared,
                                prune_result.config,
                                simulator_revision,
                            )
                            run_result = run_smart_search_with_history(
                                prune_result.config,
                                evaluator=cache_evaluator,
                                show_progress=False,
                            )
                            stored = self._record_search_history(
                                store,
                                cell,
                                study_key,
                                run_result.evaluations,
                                kvbm_profile,
                                prune_result.config,
                                cache_evaluator,
                            )
                            capacity_ids: list[str] = []
                            if (
                                cell.search_policy.capacity_search.enabled
                                and cell.search_policy.capacity_search.stage
                                == "per_substudy"
                            ):
                                capacity_ids = self._capacity_and_robustness(
                                    store,
                                    cell,
                                    prepared,
                                    prune_result.config,
                                    run_result.candidates,
                                    study_key,
                                    kvbm_profile,
                                    cache_evaluator,
                                )
                            final_candidate_ids.extend(
                                capacity_ids
                                if cell.search_policy.capacity_search.enabled
                                and cell.search_policy.capacity_search.stage
                                == "per_substudy"
                                else stored
                            )
                            store.record_evaluation(
                                cell_id=cell.id,
                                evaluation_key=f"substudy:{study_key}",
                                stage="substudy",
                                outcome="feasible",
                                reason="complete",
                                selection={
                                    "mode": mode,
                                    "pd_profile_id": None
                                    if pd_profile is None
                                    else pd_profile.id,
                                    "kvbm_profile_id": kvbm_profile.id,
                                    "load_name": load_name,
                                    "load_value": load_value,
                                },
                                metrics={
                                    "generated_static_configs": float(
                                        prune_result.generated
                                    ),
                                    "retained_static_configs": float(
                                        prune_result.retained
                                    ),
                                },
                                sla_pass=None,
                                candidate=None,
                            )
                        except (OSError, RuntimeError, ValueError) as exc:
                            reason = f"{type(exc).__name__}: {exc}"
                            study_failures.append(reason)
                            store.record_evaluation(
                                cell_id=cell.id,
                                evaluation_key=f"substudy:{study_key}",
                                stage="substudy",
                                outcome="unsupported",
                                reason=reason,
                                selection={
                                    "mode": mode,
                                    "pd_profile_id": None
                                    if pd_profile is None
                                    else pd_profile.id,
                                    "kvbm_profile_id": kvbm_profile.id,
                                },
                                metrics={},
                                sla_pass=None,
                                candidate=None,
                            )

        self._mark_discovery_frontiers(store, cell)
        if (
            cell.search_policy.capacity_search.enabled
            and cell.search_policy.capacity_search.stage == "finalists"
        ):
            final_candidate_ids = self._certify_finalists(
                store,
                cell,
                prepared,
                simulator_revision,
                resume=resume,
            )
        final_stage = "capacity" if cell.search_policy.capacity_search.enabled else "spica"
        if resume:
            final_candidate_ids.extend(
                store.candidate_ids_for_cell(cell.id, stage=final_stage)
            )
        if final_candidate_ids:
            store.mark_cell_frontiers(
                cell.id,
                final_candidate_ids,
                epsilon=cell.search_policy.epsilon_pareto,
            )
            store.record_cell(experiment_id, cell, status="complete")
            return "complete"
        detail = " | ".join(study_failures[:3]) or "no SLA-feasible candidate"
        status = "unsupported" if study_failures else "failed"
        store.record_cell(experiment_id, cell, status=status, detail=detail)
        return status

    @staticmethod
    def _load_points(
        prepared: PreparedWorkload,
    ) -> list[tuple[str | None, float | None]]:
        if prepared.load_axis is None:
            return [(None, None)]
        if prepared.load_axis.source != "explicit":
            raise ValueError(
                "dynamic load axes must be resolved with _load_points_for_mode"
            )
        return [(prepared.load_axis.name, value) for value in prepared.load_axis.values]

    def _load_points_for_mode(
        self,
        store: ResultStore,
        cell: MatrixCell,
        prepared: PreparedWorkload,
        mode: str,
    ) -> list[tuple[str | None, float | None]]:
        axis = prepared.load_axis
        if axis is None or axis.source == "explicit":
            return self._load_points(prepared)
        completion_key = (
            f"aic-operating-points-complete:{mode}:"
            f"{fingerprint({'axis': axis, 'cell': cell.fingerprint})}"
        )
        resumed = (
            store.selected_operating_point_loads(cell.id, mode=mode)
            if store.has_successful_evaluation(cell.id, completion_key)
            else []
        )
        if resumed:
            return [("concurrency", value) for value in resumed]

        pd_profile = (
            cell.search_policy.pd_profiles[0] if mode == "disagg" else None
        )
        disabled = next(
            (
                profile
                for profile in cell.search_policy.kvbm_profiles
                if self._kvbm_profile_is_disabled(profile)
            ),
            cell.search_policy.kvbm_profiles[0],
        )
        seed = float(axis.min_value or 1.0)
        seed_config = self._smart_search_config(
            cell,
            prepared,
            mode,
            pd_profile,
            disabled,
            "concurrency",
            seed,
        )
        prune_result = prune_parallel_configs(
            seed_config,
            cell.model,
            cell.search_policy.static_prune,
        )
        resolution = resolve_aic_operating_points(
            cell,
            prepared,
            mode,
            prune_result,
            axis,
        )
        self._record_operating_points(store, cell, resolution)
        store.record_evaluation(
            cell_id=cell.id,
            evaluation_key=completion_key,
            stage="aic_operating_point",
            outcome="feasible",
            reason="AIC operating-point resolution complete",
            selection={
                "mode": mode,
                "selected_concurrencies": resolution.selected_concurrencies,
                "load_axis": axis.model_dump(mode="python"),
                "compression_applied": resolution.compression_applied,
            },
            metrics={
                "selected_point_count": float(
                    len(resolution.selected_concurrencies)
                ),
                "raw_point_count": float(len(resolution.points)),
            },
            sla_pass=None,
            candidate=None,
        )
        return [("concurrency", float(value)) for value in resolution.selected_concurrencies]

    @staticmethod
    def _record_operating_points(
        store: ResultStore,
        cell: MatrixCell,
        resolution: AICOperatingPointResolution,
    ) -> None:
        for index, point in enumerate(resolution.points):
            store.record_evaluation(
                cell_id=cell.id,
                evaluation_key=(
                    f"aic-operating-point:{resolution.mode}:{index}:"
                    f"{fingerprint(point.config)}"
                ),
                stage="aic_operating_point",
                outcome="selected" if point.selected else "not_selected",
                reason=",".join(point.selection_reasons) or "AIC SLA-feasible point",
                selection={
                    "mode": resolution.mode,
                    "concurrency": point.concurrency,
                    "parallel_key": point.parallel_key,
                    "selected": point.selected,
                    "selection_reasons": point.selection_reasons,
                    "aic_config": point.config,
                    "compression_applied": resolution.compression_applied,
                },
                metrics={
                    "concurrency": float(point.concurrency),
                    "request_rate": point.request_rate,
                    "tokens_per_second_per_gpu": point.tokens_per_second_per_gpu,
                    "tokens_per_second_per_user": point.tokens_per_second_per_user,
                    "num_total_gpus": float(point.num_total_gpus),
                },
                sla_pass=None,
                candidate=None,
            )

    @staticmethod
    def _candidate_load(candidate: Candidate) -> tuple[str, float]:
        config = candidate.config
        for name in ("concurrency", "request_rate", "arrival_speedup_ratio"):
            value = config.get(name)
            if isinstance(value, (int, float)) and value > 0:
                return name, float(value)
        return "unconditioned", 0.0

    def _mark_discovery_frontiers(self, store: ResultStore, cell: MatrixCell) -> None:
        rows = store.candidates_for_cell(cell.id, stage="spica")
        if not rows:
            return
        grouped: dict[tuple[str, float], list[str]] = {}
        for candidate_id, payload in rows:
            candidate = Candidate.model_validate(payload)
            grouped.setdefault(self._candidate_load(candidate), []).append(candidate_id)
        for (axis, value), candidate_ids in grouped.items():
            store.mark_cell_frontiers(
                cell.id,
                candidate_ids,
                epsilon=cell.search_policy.epsilon_pareto,
                frontier_prefix=f"load:{axis}:{value:.12g}",
            )
        store.mark_cell_frontiers(
            cell.id,
            [candidate_id for candidate_id, _ in rows],
            epsilon=cell.search_policy.epsilon_pareto,
            frontier_prefix="global_operating_envelope",
        )

    @staticmethod
    def _qualification_key(candidate: Candidate) -> str:
        ignored = {
            "concurrency",
            "request_rate",
            "arrival_speedup_ratio",
            "kv_load_ratio",
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
        return fingerprint(
            {key: value for key, value in candidate.config.items() if key not in ignored}
        )

    def _select_finalists(
        self, cell: MatrixCell, rows: list[tuple[str, dict[str, Any]]]
    ) -> list[Candidate]:
        policy = cell.search_policy.capacity_search
        grouped: dict[tuple[str, float], list[Candidate]] = {}
        for _, payload in rows:
            candidate = Candidate.model_validate(payload)
            grouped.setdefault(self._candidate_load(candidate), []).append(candidate)

        selected: list[Candidate] = []
        objectives = [
            OptimizationTarget.GOODPUT_PER_GPU,
            OptimizationTarget.THROUGHPUT_PER_USER,
        ]
        for load in sorted(grouped):
            group = grouped[load]
            front = pareto_front(group, objectives)
            ordered = sorted(
                front,
                key=lambda candidate: (-candidate.score, candidate.used_gpus),
            )
            if len(ordered) < policy.finalists_per_load:
                seen = {self._qualification_key(candidate) for candidate in ordered}
                supplements = sorted(
                    group,
                    key=lambda candidate: (-candidate.score, candidate.used_gpus),
                )
                for candidate in supplements:
                    key = self._qualification_key(candidate)
                    if key in seen:
                        continue
                    ordered.append(candidate)
                    seen.add(key)
                    if len(ordered) >= policy.finalists_per_load:
                        break
            selected.extend(ordered[: policy.finalists_per_load])

        deduplicated: dict[str, Candidate] = {}
        for candidate in selected:
            key = self._qualification_key(candidate)
            current = deduplicated.get(key)
            if current is None or candidate.score > current.score:
                deduplicated[key] = candidate
        return sorted(
            deduplicated.values(),
            key=lambda candidate: (-candidate.score, candidate.used_gpus),
        )[: policy.max_finalists_per_cell]

    @staticmethod
    def _initial_open_loop_rps(candidate: Candidate) -> float:
        throughput = float(candidate.metrics.get("request_throughput_rps", 0.0))
        if throughput > 0:
            return throughput
        completed = float(candidate.metrics.get("completed_requests", 0.0))
        duration_ms = float(candidate.metrics.get("duration_ms", 0.0))
        if completed > 0 and duration_ms > 0:
            return completed / (duration_ms / 1000.0)
        raise ValueError("closed-loop finalist has no positive achieved request rate")

    def _certify_finalists(
        self,
        store: ResultStore,
        cell: MatrixCell,
        prepared: PreparedWorkload,
        simulator_revision: str | None,
        *,
        resume: bool = True,
    ) -> list[str]:
        rows = store.candidates_for_cell(cell.id, stage="spica")
        finalists = self._select_finalists(cell, rows)
        retained: list[str] = []
        for index, raw_candidate in enumerate(finalists):
            mode = str(raw_candidate.config["deployment_mode"])
            pd_profile_id = raw_candidate.config.get("pd_profile_id")
            pd_profile = next(
                (
                    profile
                    for profile in cell.search_policy.pd_profiles
                    if profile.id == pd_profile_id
                ),
                None,
            )
            kvbm_profile_id = raw_candidate.config.get("kvbm_profile_id")
            kvbm_profile = next(
                (
                    profile
                    for profile in cell.search_policy.kvbm_profiles
                    if profile.id == kvbm_profile_id
                ),
                None,
            )
            if kvbm_profile is None:
                raise ValueError(
                    f"finalist references unknown KVBM profile {kvbm_profile_id!r}"
                )
            source_axis, source_load = self._candidate_load(raw_candidate)
            context = f"qualification:{self._qualification_key(raw_candidate)}"
            if resume and store.has_successful_evaluation(
                cell.id, f"qualification:{context}"
            ):
                continue
            search_config = self._smart_search_config(
                cell,
                prepared,
                mode,
                pd_profile,
                kvbm_profile,
                source_axis if source_axis != "unconditioned" else None,
                source_load if source_axis != "unconditioned" else None,
            )
            initial_rps = self._initial_open_loop_rps(raw_candidate)
            open_loop_workload = search_config.workload.model_copy(
                update={
                    "concurrency": None,
                    "kv_load_ratio": None,
                    "request_rate": initial_rps,
                }
            )
            search_config = search_config.model_copy(
                update={"workload": open_loop_workload}
            )
            source_config = dict(raw_candidate.config)
            for field in (
                "concurrency",
                "request_rate",
                "arrival_speedup_ratio",
                "kv_load_ratio",
            ):
                source_config.pop(field, None)
            source_config["discovery_load_axis"] = source_axis
            source_config["discovery_load_value"] = source_load
            source = raw_candidate.model_copy(update={"config": source_config})
            cache_evaluator = self._cache_evaluator(
                store,
                cell,
                prepared,
                search_config,
                simulator_revision,
            )
            ids = self._capacity_and_robustness(
                store,
                cell,
                prepared,
                search_config,
                [source],
                context,
                kvbm_profile,
                cache_evaluator,
            )
            retained.extend(ids)
            store.record_evaluation(
                cell_id=cell.id,
                evaluation_key=f"qualification:{context}",
                stage="qualification",
                outcome="feasible" if ids else "infeasible",
                reason=(
                    "open-loop SLA boundary certification complete"
                    if ids
                    else "finalist did not pass open-loop SLA boundary certification"
                ),
                selection={
                    "source_candidate": raw_candidate.config,
                    "source_load_axis": source_axis,
                    "source_load_value": source_load,
                    "initial_request_rate": initial_rps,
                },
                metrics={"certified_candidates": float(len(ids))},
                sla_pass=bool(ids),
                candidate=None,
            )
        retained.extend(store.candidate_ids_for_cell(cell.id, stage="capacity"))
        return list(dict.fromkeys(retained))

    @staticmethod
    def _pd_bandwidth(
        cell: MatrixCell, profile: PDProfile | None, *, reverse: bool = False
    ) -> float | None:
        if profile is None:
            return None
        directional = (
            profile.bandwidth_decode_to_prefill_gbps
            if reverse
            else profile.bandwidth_prefill_to_decode_gbps
        )
        if directional is not None:
            return directional
        if reverse and profile.bandwidth_prefill_to_decode_gbps is not None:
            return profile.bandwidth_prefill_to_decode_gbps
        if profile.bandwidth_source == "topology":
            return None
        if profile.bandwidth_source == "compute_nic":
            return cell.sku.compute_nic_gbytes_per_s_per_gpu
        if profile.bandwidth_source == "local_fabric":
            topology = cell.sku.pd_transfer_topology
            return (
                topology.intra_rack_gbytes_per_s_per_gpu
                if topology.gpus_per_rack is not None
                else topology.intra_node_gbytes_per_s_per_gpu
            )
        if profile.bandwidth_gbps is None:
            raise ValueError(
                f"PD profile {profile.id!r} uses explicit bandwidth without bandwidth_gbps"
            )
        return profile.bandwidth_gbps

    @staticmethod
    def _kvbm_fields(
        cell: MatrixCell, profile: KVBMProfile, block_size: int
    ) -> dict[str, Any]:
        del block_size
        num_blocks = profile.num_g2_blocks
        to_ratio = profile.bandwidth_to_g2_ratio
        from_ratio = profile.bandwidth_from_g2_ratio
        return {
            "num_g2_blocks": num_blocks,
            # The native AIC footprint is candidate-shape dependent. Spica
            # resolves this scalar and any GiB->block conversion after TP/EP
            # have been selected.
            "kv_bytes_per_token": None,
            "bandwidth_g1_to_g2_gbps": profile.bandwidth_to_g2_gbps
            if profile.bandwidth_to_g2_gbps is not None
            else (
                None
                if to_ratio is None
                else to_ratio * cell.sku.compute_nic_gbytes_per_s_per_gpu
            ),
            "bandwidth_g2_to_g1_gbps": profile.bandwidth_from_g2_gbps
            if profile.bandwidth_from_g2_gbps is not None
            else (
                None
                if from_ratio is None
                else from_ratio * cell.sku.compute_nic_gbytes_per_s_per_gpu
            ),
            "offload_batch_size": profile.offload_batch_size,
            "g2_capacity_gib": profile.capacity_gib,
            "g2_capacity_scope": profile.capacity_scope,
            "g2_target_hit_rate": profile.target_hit_rate,
            "kvbm_profile_id": profile.id,
            "kvbm_deployability": profile.deployability,
        }

    def _smart_search_config(
        self,
        cell: MatrixCell,
        prepared: PreparedWorkload,
        mode: str,
        pd_profile: PDProfile | None,
        kvbm_profile: KVBMProfile,
        load_name: str | None,
        load_value: float | None,
    ) -> SmartSearchConfig:
        kvbm_enabled = kvbm_profile.num_g2_blocks > 0 or bool(
            kvbm_profile.capacity_gib is not None and kvbm_profile.capacity_gib > 0
        )
        footprint_required = mode == "disagg" or kvbm_enabled
        footprint = (
            None
            if cell.model.kv_footprint is None
            else cell.model.kv_footprint.model_copy(
                update={"dtype": cell.model_variant.quantization.kv_cache}
            )
        )
        if footprint_required and (
            footprint is None or footprint.layout == "legacy_unspecified"
        ):
            raise ValueError(
                f"model {cell.model.id!r} needs an auditable structured kv_footprint "
                "for P/D handoff or KVBM sizing"
            )
        workload_args = dict(prepared.replay_args)
        if load_name is not None and load_value is not None:
            # A dynamic axis has a seed value in replay_args so the prepared
            # workload remains independently replayable. Replacing the axis
            # must clear every other synthetic load mode.
            if prepared.load_axis is not None:
                for other_axis in ("request_rate", "concurrency", "kv_load_ratio"):
                    if other_axis != load_name:
                        workload_args[other_axis] = None
            if load_name == "concurrency":
                workload_args[load_name] = round(load_value)
            else:
                workload_args[load_name] = load_value
        context_length = prepared.summary.get(
            "max_context_length"
        ) or prepared.summary.get("context_length")
        if (
            context_length is not None
            and cell.model.context_length is not None
            and float(context_length) > cell.model.context_length
        ):
            raise ValueError(
                f"workload context {context_length} exceeds model limit {cell.model.context_length}"
            )

        policy = cell.search_policy.spica
        search_data = dict(policy.search_space)
        block_size = int(search_data.get("prefill_block_size", 64))
        pd_forward_bandwidth = self._pd_bandwidth(cell, pd_profile)
        pd_reverse_bandwidth = self._pd_bandwidth(cell, pd_profile, reverse=True)
        if (
            pd_forward_bandwidth is not None
            and pd_reverse_bandwidth is not None
            and not math.isclose(pd_forward_bandwidth, pd_reverse_bandwidth)
        ):
            raise ValueError(
                "DynoSim currently exposes one scalar PD transfer bandwidth; "
                f"profile {pd_profile.id!r} requests asymmetric bandwidths "
                f"{pd_forward_bandwidth} and {pd_reverse_bandwidth} GB/s"
            )
        search_data.update(
            {
                "deployment_mode": [mode],
                "backend": [cell.model_variant.backend],
                "model_name": cell.model_variant.model_path,
                "model_id": cell.model.id,
                "model_variant_id": cell.model_variant.id,
                "hardware_sku": cell.sku.aic_system,
                "gpu_budget": int(search_data.get("gpu_budget", 32)),
                "kv_footprint": None
                if footprint is None
                else footprint.model_dump(mode="python"),
                "kv_footprint_policy": cell.search_policy.kv_footprint.model_dump(
                    mode="python"
                ),
                "context_length": int(context_length)
                if context_length is not None
                else cell.model.context_length,
                "planner_scaling_policy": ["disabled"],
                "pd_profile_id": None if pd_profile is None else pd_profile.id,
                "pd_placement": None if pd_profile is None else pd_profile.placement,
                "pd_bandwidth_source": None
                if pd_profile is None
                else pd_profile.bandwidth_source,
                "pd_transfer_topology": None
                if pd_profile is None
                else cell.sku.pd_transfer_topology,
                "kv_transfer_bandwidth": pd_forward_bandwidth,
                "pd_prefill_to_decode_bandwidth_gbps": pd_forward_bandwidth,
                "pd_decode_to_prefill_bandwidth_gbps": pd_reverse_bandwidth,
                "kv_transfer_timing_mode": "full_prompt"
                if pd_profile is None
                else pd_profile.timing_mode,
                "pd_deployability": None
                if pd_profile is None
                else pd_profile.deployability,
                **self._kvbm_fields(cell, kvbm_profile, block_size),
                **cell.model_variant.quantization.as_aic_fields(),
            }
        )
        search_space = SearchSpace.model_validate(search_data)
        workload = Workload.model_validate(workload_args)
        goal = OptimizationGoal(
            target=OptimizationTarget.PARETO,
            pareto_objectives=[
                OptimizationTarget.GOODPUT_PER_GPU,
                OptimizationTarget.THROUGHPUT_PER_USER,
            ],
            sla=SLATarget(ttft_ms=cell.sla.ttft_ms, itl_ms=cell.sla.tpot_ms),
        )
        sweep = SweepConfig(
            max_rounds=policy.max_rounds,
            parallel_evals=policy.parallel_evals,
            candidates_per_round=policy.candidates_per_round,
            max_eval_seconds=policy.max_eval_seconds,
            sampler_algorithm=policy.sampler_algorithm,
            search_seed=policy.search_seed,
        )
        return SmartSearchConfig(
            search_space=search_space,
            workload=workload,
            goal=goal,
            sweep=sweep,
        )

    @staticmethod
    def _record_prune(
        store: ResultStore,
        cell: MatrixCell,
        study_key: str,
        result: StaticPruneResult,
    ) -> None:
        for index, decision in enumerate(result.decisions):
            store.record_evaluation(
                cell_id=cell.id,
                evaluation_key=f"prune:{study_key}:{index}:{fingerprint(decision.config)}",
                stage="static_prune",
                outcome="retained" if decision.retained else "pruned",
                reason=decision.reason,
                selection=decision.config,
                metrics={
                    **decision.objectives,
                    "static_pareto": float(decision.static_pareto),
                    **(
                        {"static_rank": float(decision.static_rank)}
                        if decision.static_rank is not None
                        else {}
                    ),
                },
                sla_pass=None,
                candidate=None,
            )

    def _record_search_history(
        self,
        store: ResultStore,
        cell: MatrixCell,
        study_key: str,
        evaluations: list[SearchEvaluation],
        kvbm_profile: KVBMProfile,
        search_config: SmartSearchConfig,
        cache_evaluator: CacheAwareReplayEvaluator,
    ) -> list[str]:
        candidate_ids: list[str] = []
        for evaluation in evaluations:
            candidate = evaluation.candidate
            metrics = {} if candidate is None else candidate.metrics
            sla_pass = None if candidate is None else self._sla_pass(metrics, cell.sla)
            candidate_payload = None
            resources = None
            if candidate is not None:
                candidate_payload = candidate.model_dump(mode="python")
                resources = self._resource_usage(
                    candidate,
                    kvbm_profile,
                    cell,
                )
            cache_description: ReplayCacheDescription | None = None
            cache_hit = evaluation.outcome == "cache_hit"
            if candidate is not None and cache_evaluator.cache_enabled:
                plan = build_deployment(
                    candidate.config,
                    backend_version=str(candidate.config["backend_version"]),
                    optimization_target=search_config.goal.target.planner_optimization_target,
                    planner_sla=search_config.goal.sla,
                )
                concurrency = candidate.config.get("concurrency")
                cache_description = cache_evaluator.describe(
                    plan,
                    concurrency_override=None
                    if concurrency is None
                    else int(concurrency),
                )
                self._persist_replay_metrics(
                    store,
                    cell,
                    "spica",
                    cache_evaluator,
                    cache_description,
                    metrics,
                    cache_hit=cache_hit,
                )
            evaluation_key = fingerprint(
                {
                    "study": study_key,
                    "trial_index": evaluation.index,
                    "branch": evaluation.branch,
                    "selection": evaluation.selection,
                    "parallel_config": evaluation.parallel_config,
                    "candidate_config": None if candidate is None else candidate.config,
                }
            )
            candidate_id = store.record_evaluation(
                cell_id=cell.id,
                evaluation_key=f"spica:{evaluation_key}",
                stage="spica",
                outcome="feasible" if cache_hit else evaluation.outcome,
                reason=evaluation.reason,
                selection={
                    **evaluation.selection,
                    "parallel_config": evaluation.parallel_config,
                },
                metrics=metrics,
                sla_pass=sla_pass,
                candidate=candidate_payload,
                resources=resources,
                candidate_context=study_key,
                evaluation_cache_key=None
                if cache_description is None
                else cache_description.evaluation_cache_key,
                cache_hit=cache_hit,
            )
            if (
                candidate_id is not None
                and evaluation.outcome in {"feasible", "cache_hit"}
                and sla_pass
            ):
                candidate_ids.append(candidate_id)
        return candidate_ids

    @staticmethod
    def _cache_evaluator(
        store: ResultStore,
        cell: MatrixCell,
        prepared: PreparedWorkload,
        search_config: SmartSearchConfig,
        simulator_revision: str | None,
    ) -> CacheAwareReplayEvaluator:
        evaluator = CacheAwareReplayEvaluator(
            search_config.workload,
            search_config.goal,
            workload_content_key=fingerprint(
                {
                    "loader": prepared.loader,
                    "source_sha256": prepared.source_sha256,
                    "normalization": prepared.normalization,
                }
            ),
            simulator_revision=simulator_revision,
            model_id=cell.model.id,
            model_variant_id=cell.model_variant.id,
            sku_id=cell.sku.id,
        )
        if evaluator.cache_enabled:
            evaluator.load_entries(
                store.replay_cache_snapshot(evaluator.cache_namespace_key)
            )
        return evaluator

    @staticmethod
    def _persist_replay_metrics(
        store: ResultStore,
        cell: MatrixCell,
        source_stage: str,
        evaluator: CacheAwareReplayEvaluator,
        description: ReplayCacheDescription,
        metrics: dict[str, float],
        *,
        cache_hit: bool,
    ) -> None:
        if not evaluator.cache_enabled:
            return
        if cache_hit:
            store.touch_replay_cache_entry(description.evaluation_cache_key)
        else:
            store.store_replay_cache_entry(
                evaluation_cache_key=description.evaluation_cache_key,
                cache_namespace_key=description.cache_namespace_key,
                semantics=description.semantics,
                metrics=metrics,
                source_cell_id=cell.id,
                source_stage=source_stage,
            )
        evaluator.remember(description, metrics)

    def _evaluate_with_cache(
        self,
        store: ResultStore,
        cell: MatrixCell,
        source_stage: str,
        evaluator: CacheAwareReplayEvaluator,
        plan: Any,
        *,
        concurrency_override: int | None = None,
    ) -> CachedReplayResult:
        result = evaluator.evaluate_with_cache(
            plan, concurrency_override=concurrency_override
        )
        self._persist_replay_metrics(
            store,
            cell,
            source_stage,
            evaluator,
            result.description,
            result.metrics,
            cache_hit=result.cache_hit,
        )
        return result

    @staticmethod
    def _sla_pass(metrics: dict[str, float], sla: SLAProfile) -> bool:
        num_requests = metrics.get("num_requests", 0.0)
        completed = metrics.get("completed_requests", 0.0)
        completion_rate = completed / num_requests if num_requests > 0 else 0.0
        goodput = metrics.get("goodput_completed_requests", 0.0)
        goodput_ratio = goodput / completed if completed > 0 else 0.0
        checks = [
            metrics.get("mean_ttft_ms", math.inf) <= sla.ttft_ms,
            metrics.get("mean_tpot_ms", math.inf) <= sla.tpot_ms,
            metrics.get("mean_output_token_throughput_per_user", 0.0)
            >= sla.tps_per_user,
            completion_rate >= sla.min_completion_rate,
            goodput_ratio >= sla.min_goodput_ratio,
        ]
        if sla.p95_ttft_ms is not None:
            checks.append(metrics.get("p95_ttft_ms", math.inf) <= sla.p95_ttft_ms)
        if sla.p95_tpot_ms is not None:
            checks.append(metrics.get("p95_tpot_ms", math.inf) <= sla.p95_tpot_ms)
        if sla.p10_tps_per_user is not None:
            checks.append(
                metrics.get("p10_output_token_throughput_per_user", 0.0)
                >= sla.p10_tps_per_user
            )
        return all(checks)

    @staticmethod
    def _resource_usage(
        candidate: Candidate,
        kvbm_profile: KVBMProfile,
        cell: MatrixCell,
    ) -> dict[str, float]:
        metrics = candidate.metrics
        duration_ms = metrics.get("duration_ms", 0.0)
        duration_hours = duration_ms / 3_600_000.0
        input_tokens = metrics.get("total_input_tokens", 0.0)
        output_tokens = metrics.get("total_output_tokens", 0.0)
        completed = metrics.get("completed_requests", 0.0)
        goodput_completed = metrics.get("goodput_completed_requests", 0.0)
        goodput_ratio = goodput_completed / completed if completed > 0 else 0.0
        capacity_gib = float(
            candidate.config.get("g2_effective_total_capacity_gib")
            or candidate.config.get("g2_capacity_gib")
            or 0.0
        )
        network_gb = 0.0
        if candidate.config.get("deployment_mode") == "disagg":
            pd_bytes = float(
                candidate.config.get(
                    "pd_transfer_physical_bytes_per_token_estimate", 0.0
                )
            )
            network_gb += input_tokens * pd_bytes / 1_000_000_000.0
        if int(candidate.config.get("num_g2_blocks") or 0) > 0:
            reused_ratio = metrics.get("prefix_cache_reused_ratio", 0.0)
            kvbm_bytes = float(
                candidate.config.get("kvbm_physical_bytes_per_token_estimate", 0.0)
            )
            network_gb += input_tokens * reused_ratio * kvbm_bytes / 1_000_000_000.0
        return {
            "gpu_hours": metrics.get("gpu_hours", 0.0),
            "g2_gib_hours": capacity_gib * duration_hours,
            "network_gb": network_gb,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "goodput_input_tokens": input_tokens * goodput_ratio,
            "goodput_output_tokens": output_tokens * goodput_ratio,
            "duration_ms": duration_ms,
        }

    def _capacity_and_robustness(
        self,
        store: ResultStore,
        cell: MatrixCell,
        prepared: PreparedWorkload,
        search_config: SmartSearchConfig,
        candidates: list[Candidate],
        study_key: str,
        kvbm_profile: KVBMProfile,
        cache_evaluator: CacheAwareReplayEvaluator,
    ) -> list[str]:
        retained_ids: list[str] = []
        disabled_kvbm_profile = next(
            (
                profile
                for profile in cell.search_policy.kvbm_profiles
                if self._kvbm_profile_is_disabled(profile)
            ),
            None,
        )
        for index, candidate in enumerate(candidates):
            source_candidate_config = dict(candidate.config)

            def record_probe(
                probe_index: int,
                phase: str,
                axis: str,
                load: float,
                metrics: dict[str, float],
                passed: bool,
                cache_description: ReplayCacheDescription,
                cache_hit: bool,
                source_candidate_config: dict[str, Any] = source_candidate_config,
            ) -> None:
                store.record_evaluation(
                    cell_id=cell.id,
                    evaluation_key=(
                        "capacity_probe:"
                        + fingerprint(
                            {
                                "study": study_key,
                                "candidate": source_candidate_config,
                                "probe_index": probe_index,
                                "phase": phase,
                                "axis": axis,
                                "load": load,
                            }
                        )
                    ),
                    stage="capacity_probe",
                    outcome="feasible" if passed else "infeasible",
                    reason=f"{phase}: {axis}={load:.12g}",
                    selection={
                        "source_candidate": source_candidate_config,
                        "capacity_axis": axis,
                        "capacity_value": load,
                        "phase": phase,
                    },
                    metrics=metrics,
                    sla_pass=passed,
                    candidate=None,
                    evaluation_cache_key=cache_description.evaluation_cache_key,
                    cache_hit=cache_hit,
                )

            capacity_candidate, capacity_report = self._capacity_search(
                cell,
                prepared,
                search_config,
                candidate,
                store=store,
                cache_evaluator=cache_evaluator,
                on_probe=record_probe,
            )
            if capacity_candidate is None or capacity_report is None:
                continue
            if not self._robustness_passes(
                store,
                cell,
                search_config,
                capacity_candidate,
                study_key,
                index,
                cache_evaluator,
            ):
                continue
            evaluation_key = fingerprint(
                {
                    "study": study_key,
                    "capacity": capacity_candidate.config,
                }
            )
            candidate_id = store.record_evaluation(
                cell_id=cell.id,
                evaluation_key=f"capacity:{evaluation_key}",
                stage="capacity",
                outcome="feasible",
                reason=(
                    "maximum SLA load bracketed within configured tolerance"
                    if not capacity_candidate.config.get(
                        "capacity_search_right_censored"
                    )
                    else "highest tested SLA-feasible load; capacity is right-censored by the configured search bound"
                ),
                selection={
                    "source_candidate": candidate.config,
                    "capacity_axis": capacity_candidate.config.get("capacity_axis"),
                    "capacity_value": capacity_candidate.config.get("capacity_value"),
                },
                metrics=capacity_report,
                sla_pass=True,
                candidate=capacity_candidate.model_dump(mode="python"),
                resources=self._resource_usage(capacity_candidate, kvbm_profile, cell),
                candidate_context=study_key,
            )
            if candidate_id is not None:
                retained_ids.append(candidate_id)
            if (
                candidate_id is not None
                and disabled_kvbm_profile is not None
                and not self._kvbm_profile_is_disabled(kvbm_profile)
            ):
                retained_ids.extend(
                    self._matched_disabled_kvbm_baseline(
                        store,
                        cell,
                        prepared,
                        search_config,
                        candidate,
                        study_key,
                        index,
                        kvbm_profile,
                        disabled_kvbm_profile,
                        cache_evaluator,
                    )
                )
        return retained_ids

    @staticmethod
    def _kvbm_profile_is_disabled(profile: KVBMProfile) -> bool:
        return profile.num_g2_blocks == 0 and not (
            profile.capacity_gib is not None and profile.capacity_gib > 0
        )

    def _matched_disabled_kvbm_baseline(
        self,
        store: ResultStore,
        cell: MatrixCell,
        prepared: PreparedWorkload,
        search_config: SmartSearchConfig,
        source_candidate: Candidate,
        study_key: str,
        candidate_index: int,
        source_profile: KVBMProfile,
        disabled_profile: KVBMProfile,
        cache_evaluator: CacheAwareReplayEvaluator,
    ) -> list[str]:
        """Capacity-test the exact same compute/router config with KVBM disabled."""
        baseline_config = dict(source_candidate.config)
        for key in (
            "kv_bytes_per_token",
            "kvbm_footprint_contract",
            "kvbm_physical_bytes_per_token_estimate",
            "g2_effective_capacity_gib_per_rank",
            "g2_effective_total_capacity_gib",
        ):
            baseline_config.pop(key, None)
        baseline_config.update(self._kvbm_fields(cell, disabled_profile, block_size=64))
        baseline_source = source_candidate.model_copy(
            update={"config": baseline_config}
        )
        baseline_context = f"{study_key}:matched_disabled_kvbm_for:{source_profile.id}"

        def record_probe(
            probe_index: int,
            phase: str,
            axis: str,
            load: float,
            metrics: dict[str, float],
            passed: bool,
            cache_description: ReplayCacheDescription,
            cache_hit: bool,
        ) -> None:
            store.record_evaluation(
                cell_id=cell.id,
                evaluation_key=(
                    "capacity_probe:"
                    + fingerprint(
                        {
                            "study": baseline_context,
                            "candidate": baseline_config,
                            "probe_index": probe_index,
                            "phase": phase,
                            "axis": axis,
                            "load": load,
                        }
                    )
                ),
                stage="capacity_probe",
                outcome="feasible" if passed else "infeasible",
                reason=f"matched KVBM-off {phase}: {axis}={load:.12g}",
                selection={
                    "source_candidate": source_candidate.config,
                    "matched_baseline_for_profile": source_profile.id,
                    "capacity_axis": axis,
                    "capacity_value": load,
                    "phase": phase,
                },
                metrics=metrics,
                sla_pass=passed,
                candidate=None,
                evaluation_cache_key=cache_description.evaluation_cache_key,
                cache_hit=cache_hit,
            )

        capacity_candidate, capacity_report = self._capacity_search(
            cell,
            prepared,
            search_config,
            baseline_source,
            store=store,
            cache_evaluator=cache_evaluator,
            on_probe=record_probe,
        )
        if capacity_candidate is None or capacity_report is None:
            return []
        if not self._robustness_passes(
            store,
            cell,
            search_config,
            capacity_candidate,
            baseline_context,
            candidate_index,
            cache_evaluator,
        ):
            return []
        evaluation_key = fingerprint(
            {
                "study": baseline_context,
                "capacity": capacity_candidate.config,
            }
        )
        candidate_id = store.record_evaluation(
            cell_id=cell.id,
            evaluation_key=f"capacity:{evaluation_key}",
            stage="capacity",
            outcome="feasible",
            reason=(
                "matched KVBM-off baseline; maximum SLA load bracketed within "
                "configured tolerance"
                if not capacity_candidate.config.get("capacity_search_right_censored")
                else "matched KVBM-off baseline; highest tested SLA-feasible load "
                "is right-censored by the configured search bound"
            ),
            selection={
                "source_candidate": source_candidate.config,
                "matched_baseline_for_profile": source_profile.id,
                "capacity_axis": capacity_candidate.config.get("capacity_axis"),
                "capacity_value": capacity_candidate.config.get("capacity_value"),
            },
            metrics=capacity_report,
            sla_pass=True,
            candidate=capacity_candidate.model_dump(mode="python"),
            resources=self._resource_usage(
                capacity_candidate,
                disabled_profile,
                cell,
            ),
            candidate_context=baseline_context,
        )
        return [] if candidate_id is None else [candidate_id]

    def _capacity_search(
        self,
        cell: MatrixCell,
        prepared: PreparedWorkload,
        search_config: SmartSearchConfig,
        candidate: Candidate,
        *,
        store: ResultStore,
        cache_evaluator: CacheAwareReplayEvaluator,
        on_probe: Callable[
            [
                int,
                str,
                str,
                float,
                dict[str, float],
                bool,
                ReplayCacheDescription,
                bool,
            ],
            None,
        ]
        | None = None,
    ) -> tuple[Candidate | None, dict[str, float] | None]:
        policy = cell.search_policy.capacity_search
        if not policy.enabled:
            return candidate, candidate.metrics
        workload = search_config.workload
        if workload.trace_path is not None:
            axis = "arrival_speedup_ratio"
            initial = workload.arrival_speedup_ratio
        elif workload.request_rate is not None:
            axis = "request_rate"
            initial = workload.request_rate
        else:
            return candidate, candidate.metrics

        plan = build_deployment(
            candidate.config,
            backend_version=str(candidate.config["backend_version"]),
            optimization_target=search_config.goal.target.planner_optimization_target,
            planner_sla=search_config.goal.sla,
        )

        probe_index = 0

        def evaluate(load: float, phase: str) -> tuple[dict[str, float], bool]:
            nonlocal probe_index
            updated = workload.model_copy(update={axis: load})
            load_evaluator = cache_evaluator.fork(updated)
            cached = self._evaluate_with_cache(
                store,
                cell,
                "capacity_probe",
                load_evaluator,
                plan,
            )
            report = cached.metrics
            passed = self._sla_pass(report, cell.sla)
            if on_probe is not None:
                on_probe(
                    probe_index,
                    phase,
                    axis,
                    load,
                    report,
                    passed,
                    cached.description,
                    cached.cache_hit,
                )
            probe_index += 1
            return report, passed

        initial = max(initial, policy.min_load)
        if policy.max_load is not None:
            initial = min(initial, policy.max_load)
        initial_report, initial_passed = evaluate(initial, "initial")
        failed_high: float | None = None
        if initial_passed:
            low, low_report = initial, initial_report
            high = low
            for _ in range(policy.max_expansions):
                proposed = high * policy.expansion_factor
                if policy.max_load is not None:
                    proposed = min(proposed, policy.max_load)
                if proposed <= high:
                    break
                report, passed = evaluate(proposed, "expansion")
                if passed:
                    low, low_report = proposed, report
                    high = proposed
                    continue
                failed_high = proposed
                break
        else:
            high = initial
            failed_high = initial
            low: float | None = None
            low_report: dict[str, float] | None = None
            for _ in range(policy.max_expansions):
                proposed = max(high / policy.expansion_factor, policy.min_load)
                if proposed >= high:
                    break
                report, passed = evaluate(proposed, "downward_expansion")
                if passed:
                    low, low_report = proposed, report
                    break
                high = proposed
                failed_high = proposed
            if low is None or low_report is None:
                return None, None
        if failed_high is not None:
            high = failed_high
            for _ in range(policy.max_binary_steps):
                if (high - low) / max(low, 1e-9) <= policy.tolerance:
                    break
                middle = (low + high) / 2.0
                report, passed = evaluate(middle, "binary")
                if passed:
                    low, low_report = middle, report
                else:
                    high = middle
        if policy.require_failed_upper and failed_high is None:
            return None, None
        config = dict(candidate.config)
        config["capacity_axis"] = axis
        config["capacity_value"] = low
        config["capacity_search_status"] = (
            "feasible_at_max" if failed_high is None else "bracketed"
        )
        config["capacity_search_right_censored"] = failed_high is None
        config["capacity_search_feasible_lower_bound"] = low
        config["capacity_search_infeasible_bound"] = (
            None if failed_high is None else high
        )
        if axis == "arrival_speedup_ratio":
            config["max_sla_arrival_speedup_ratio"] = low
            duration = prepared.summary.get("end_timestamp")
            start = prepared.summary.get("start_timestamp")
            requests = prepared.summary.get("request_count")
            if (
                isinstance(duration, (int, float))
                and isinstance(start, (int, float))
                and isinstance(requests, (int, float))
                and duration > start
            ):
                config["max_sla_effective_rps"] = (
                    requests / ((duration - start) / 1000.0) * low
                )
        else:
            config["max_sla_rps"] = low
        capacity_candidate = make_candidate(
            config,
            low_report,
            search_config.goal.target,
            pareto_objectives=search_config.goal.resolved_pareto_objectives,
        )
        return capacity_candidate, low_report

    def _robustness_passes(
        self,
        store: ResultStore,
        cell: MatrixCell,
        search_config: SmartSearchConfig,
        candidate: Candidate,
        study_key: str,
        candidate_index: int,
        cache_evaluator: CacheAwareReplayEvaluator,
    ) -> bool:
        policy = cell.search_policy.robustness
        if not policy.enabled:
            return True
        all_passed = True
        for seed in policy.seeds:
            for multiplier in policy.bandwidth_multipliers:
                candidate_config = dict(candidate.config)
                for field in (
                    "kv_transfer_bandwidth",
                    "bandwidth_g1_to_g2_gbps",
                    "bandwidth_g2_to_g1_gbps",
                ):
                    value = candidate_config.get(field)
                    if value is not None:
                        candidate_config[field] = float(value) * multiplier
                workload_update: dict[str, Any] = {"arrival_seed": seed}
                capacity_axis = candidate.config.get("capacity_axis")
                capacity_value = candidate.config.get("capacity_value")
                if capacity_axis is not None and capacity_value is not None:
                    workload_update[str(capacity_axis)] = capacity_value
                workload = search_config.workload.model_copy(update=workload_update)
                plan = build_deployment(
                    candidate_config,
                    backend_version=str(candidate_config["backend_version"]),
                    optimization_target=search_config.goal.target.planner_optimization_target,
                    planner_sla=search_config.goal.sla,
                )
                robust_evaluator = cache_evaluator.fork(workload)
                cached = self._evaluate_with_cache(
                    store,
                    cell,
                    "robustness",
                    robust_evaluator,
                    plan,
                )
                metrics = cached.metrics
                passed = self._sla_pass(metrics, cell.sla)
                store.record_evaluation(
                    cell_id=cell.id,
                    evaluation_key=f"robust:{fingerprint({'study': study_key, 'candidate': candidate_index, 'seed': seed, 'bandwidth': multiplier})}",
                    stage="robustness",
                    outcome="feasible" if passed else "infeasible",
                    reason=f"seed={seed}, bandwidth_multiplier={multiplier}",
                    selection={
                        "candidate": candidate.config,
                        "seed": seed,
                        "bandwidth_multiplier": multiplier,
                    },
                    metrics=metrics,
                    sla_pass=passed,
                    candidate=None,
                    evaluation_cache_key=cached.description.evaluation_cache_key,
                    cache_hit=cached.cache_hit,
                )
                if not passed:
                    all_passed = False
        return all_passed

    @staticmethod
    def _write_manifest(
        store: ResultStore,
        config: MatrixExperimentConfig,
        catalogs: CatalogBundle,
        cells: list[MatrixCell],
        experiment_id: str,
        status: str,
        cache_import: dict[str, Any],
        variant_cache_import: dict[str, Any],
        resolution: VariantResolution,
    ) -> None:
        manifest = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "status": status,
            "experiment_config": str(config.source_path),
            "experiment_sha256": config.source_sha256,
            "simulator_revision": config.run.simulator_revision,
            "replay_cache_import": cache_import,
            "variant_support_cache_import": variant_cache_import,
            "variant_resolution": [
                record.model_dump(mode="python") for record in resolution.records
            ],
            "runtime_provenance": capture_runtime_provenance(catalogs),
            "artifacts": [
                artifact.model_dump(mode="python") for artifact in catalogs.artifacts
            ],
            "cells": [
                {
                    "id": cell.id,
                    "fingerprint": cell.fingerprint,
                    "model": cell.model.id,
                    "model_variant": cell.model_variant.id,
                    "model_path": cell.model_variant.model_path,
                    "weight_precision": cell.model_variant.weight_precision,
                    "aic_quantization": cell.model_variant.quantization.model_dump(
                        mode="python"
                    ),
                    "sku": cell.sku.id,
                    "workload": cell.workload.id,
                    "workload_source": cell.workload.source_path,
                    "workload_sha256": cell.workload.source_sha256,
                    "support_tier": cell.model_variant.support_tier,
                    "deployment_fidelity": cell.model_variant.deployment_fidelity,
                    "variant_resolution_id": cell.variant_resolution.id,
                    "variant_resolution_source": cell.variant_resolution.source,
                    "variant_support_status": cell.variant_resolution.support.status,
                }
                for cell in cells
            ],
            "summary": store.summary(),
        }
        (store.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
