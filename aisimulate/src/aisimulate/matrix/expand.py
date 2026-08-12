# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic matrix expansion and non-mutating experiment validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from pydantic import BaseModel, ConfigDict

from ._utils import fingerprint
from .catalog import load_catalogs
from .config import (
    CatalogBundle,
    MatrixExperimentConfig,
    ModelSpec,
    ModelVariantSpec,
    SearchPolicyConfig,
    Selector,
    SkuSpec,
    SLAProfile,
    ValidationIssue,
    ValidationReport,
)
from .variant import (
    VariantResolution,
    VariantResolutionRecord,
    VariantSupportProbe,
    VariantSupportResult,
    resolve_variants,
    structural_resolution,
)
from .workload import LoaderContext, PreparedWorkload, prepare_workload

_ItemT = TypeVar("_ItemT", bound=BaseModel)


class MatrixCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    fingerprint: str
    model: ModelSpec
    model_variant: ModelVariantSpec
    sku: SkuSpec
    workload: PreparedWorkload
    sla: SLAProfile
    search_policy: SearchPolicyConfig
    variant_resolution: VariantResolutionRecord


def _select(
    entries: Mapping[str, _ItemT], selector: Selector, *, label: str
) -> list[_ItemT]:
    if selector.ids is not None:
        unknown = sorted(set(selector.ids) - set(entries))
        if unknown:
            raise ValueError(f"{label} selector references unknown ids {unknown}")
        selected_ids = list(dict.fromkeys(selector.ids))
    else:
        selected_ids = sorted(entries)

    required_tags = set(selector.tags)
    excluded = set(selector.exclude)
    selected = []
    for entry_id in selected_ids:
        if entry_id in excluded:
            continue
        entry = entries[entry_id]
        entry_tags = set(entry.tags)
        if required_tags and not required_tags.issubset(entry_tags):
            continue
        selected.append(entry)
    if not selected:
        raise ValueError(f"{label} selector matched no entries")
    return selected


def _sku_fingerprint_contract(sku: SkuSpec) -> dict[str, object]:
    """Return the semantic hardware identity that affects replay planning.

    P/D topology participates explicitly: changing a domain size or bandwidth
    must create new Cells, while report-time pricing remains outside this hash.
    """

    return {
        "id": sku.id,
        "aic_system": sku.aic_system,
        "compute_nic_gbytes_per_s_per_gpu": sku.compute_nic_gbytes_per_s_per_gpu,
        "pd_transfer_topology": sku.pd_transfer_topology,
        "tags": sku.tags,
    }


def _variant_fingerprint_contract(
    variant: ModelVariantSpec,
) -> dict[str, object]:
    """Hash concrete replay semantics, excluding resolver labels and origin.

    Defaults that were implicit before first-class variants are omitted so old
    explicit vLLM Cell IDs remain stable. A non-default backend, pinned backend
    version, or concrete kernel family is part of execution semantics.
    """

    contract: dict[str, object] = {
        "id": variant.id,
        "model_id": variant.model_id,
        "model_path": variant.model_path,
        "weight_precision": variant.weight_precision,
        "quantization": variant.quantization,
        "support_tier": variant.support_tier,
        "deployment_fidelity": variant.deployment_fidelity,
        "quality_profile": variant.quality_profile,
        "tags": variant.tags,
    }
    if variant.backend != "vllm":
        contract["backend"] = variant.backend
    if variant.backend_version is not None:
        contract["backend_version"] = variant.backend_version
    if variant.kernel_family != "auto":
        contract["kernel_family"] = variant.kernel_family
    return contract


def _search_policy_fingerprint_contract(
    policy: SearchPolicyConfig,
) -> dict[str, object]:
    """Normalize values that were historically implicit in Matrix v1."""

    payload = policy.model_dump(mode="python")
    search_space = dict(payload["spica"]["search_space"])
    if search_space.get("backend") == ["vllm"]:
        search_space.pop("backend")
    payload["spica"]["search_space"] = search_space
    return payload


def resolve_matrix_variants(
    config: MatrixExperimentConfig,
    catalogs: CatalogBundle | None = None,
    *,
    support_probe: VariantSupportProbe | None = None,
    support_cache: Mapping[str, VariantSupportResult] | None = None,
) -> VariantResolution:
    """Resolve selected model/SKU variants without preparing workloads."""

    bundle = catalogs or load_catalogs(config)
    models = _select(bundle.models, config.matrix.model_selector, label="model")
    skus = _select(bundle.skus, config.matrix.sku_selector, label="sku")
    return resolve_variants(
        config,
        bundle,
        models=models,
        skus=skus,
        probe=support_probe,
        support_cache=support_cache,
    )


def expand_matrix(
    config: MatrixExperimentConfig,
    catalogs: CatalogBundle | None = None,
    *,
    resolution: VariantResolution | None = None,
    support_probe: VariantSupportProbe | None = None,
    support_cache: Mapping[str, VariantSupportResult] | None = None,
) -> list[MatrixCell]:
    """Resolve and expand stable model × concrete-variant × SKU × workload Cells."""
    bundle = catalogs or load_catalogs(config)
    models = _select(bundle.models, config.matrix.model_selector, label="model")
    selected_model_ids = {model.id for model in models}
    skus = _select(bundle.skus, config.matrix.sku_selector, label="sku")
    workload_specs = _select(
        bundle.workloads, config.matrix.workload_selector, label="workload"
    )
    sla = bundle.slas.get(config.matrix.sla_profile)
    if sla is None:
        raise ValueError(f"unknown SLA profile {config.matrix.sla_profile!r}")

    prepared_workloads = [
        prepare_workload(spec, LoaderContext(base_dir=config.base_dir))
        for spec in workload_specs
    ]
    active_resolution = resolution or resolve_matrix_variants(
        config,
        bundle,
        support_probe=support_probe,
        support_cache=support_cache,
    )
    selected_records = active_resolution.selected_records
    if not selected_records:
        raise ValueError("variant resolver selected no model/SKU records")
    invalid_pairs = sorted(
        {
            (record.model_id, record.sku_id)
            for record in selected_records
            if record.model_id not in selected_model_ids
            or record.sku_id not in {sku.id for sku in skus}
        }
    )
    if invalid_pairs:
        raise ValueError(
            f"variant resolution contains records outside selected matrix: {invalid_pairs}"
        )
    cells: list[MatrixCell] = []
    models_by_id = {model.id: model for model in models}
    skus_by_id = {sku.id: sku for sku in skus}
    for record in selected_records:
        model_variant = record.variant
        model = models_by_id[record.model_id]
        sku = skus_by_id[record.sku_id]
        for workload in prepared_workloads:
            payload = {
                "schema_version": config.schema_version,
                "model": model,
                "model_variant": _variant_fingerprint_contract(model_variant),
                "sku": _sku_fingerprint_contract(sku),
                "workload_fingerprint": workload.fingerprint,
                "sla": sla,
                "search_policy": _search_policy_fingerprint_contract(
                    bundle.search_policy
                ),
                "simulator_revision": config.run.simulator_revision,
            }
            cell_fingerprint = fingerprint(payload)
            cells.append(
                MatrixCell(
                    id=f"cell_{cell_fingerprint[:20]}",
                    fingerprint=cell_fingerprint,
                    model=model,
                    model_variant=model_variant,
                    sku=sku,
                    workload=workload,
                    sla=sla,
                    search_policy=bundle.search_policy,
                    variant_resolution=record,
                )
            )
    return sorted(cells, key=lambda cell: cell.id)


def validate_experiment(config: MatrixExperimentConfig) -> ValidationReport:
    """Validate all discoverable inputs without running AIC or replay."""
    issues: list[ValidationIssue] = []
    try:
        bundle = load_catalogs(config)
        models = _select(bundle.models, config.matrix.model_selector, label="model")
        skus = _select(bundle.skus, config.matrix.sku_selector, label="sku")
        resolution = structural_resolution(config, bundle, models=models, skus=skus)
        cells = expand_matrix(config, bundle, resolution=resolution)
    except (OSError, TypeError, ValueError) as exc:
        return ValidationReport(
            valid=False,
            issues=[
                ValidationIssue(
                    severity="error", code="invalid_input", message=str(exc)
                )
            ],
        )

    for cell in cells:
        if cell.model_variant.support_tier == "unsupported":
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="catalog_unsupported",
                    item_id=cell.id,
                    message=(
                        f"model variant {cell.model_variant.id!r} is cataloged "
                        "as unsupported"
                    ),
                )
            )
        modes = set(cell.search_policy.spica.deployment_mode)
        caps = cell.workload.capabilities
        kvbm_enabled = any(
            profile.num_g2_blocks > 0
            or bool(profile.capacity_gib is not None and profile.capacity_gib > 0)
            for profile in cell.search_policy.kvbm_profiles
        )
        footprint_required = "disagg" in modes or kvbm_enabled
        if cell.model.kv_footprint is None and footprint_required:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_kv_footprint",
                    item_id=cell.id,
                    message=(
                        f"model {cell.model.id!r} needs kv_footprint for P/D "
                        "handoff or KVBM sizing"
                    ),
                )
            )
        elif (
            footprint_required
            and cell.model.kv_footprint is not None
            and cell.model.kv_footprint.layout == "legacy_unspecified"
        ):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="ambiguous_legacy_kv_footprint",
                    item_id=cell.id,
                    message=(
                        f"model {cell.model.id!r} uses legacy kv_bytes_per_token "
                        "without an auditable layout/dtype/scope contract"
                    ),
                )
            )
        if (
            cell.model.kv_footprint is not None
            and cell.model.kv_footprint.dtype is not None
            and cell.model.kv_footprint.dtype
            != cell.model_variant.quantization.kv_cache
        ):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="variant_kv_dtype_mismatch",
                    item_id=cell.id,
                    message=(
                        f"model {cell.model.id!r} KV footprint dtype "
                        f"{cell.model.kv_footprint.dtype!r} disagrees with variant "
                        f"{cell.model_variant.id!r} KV-cache precision "
                        f"{cell.model_variant.quantization.kv_cache!r}"
                    ),
                )
            )
        if "agg" in modes and not caps.aggregate:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="workload_mode_mismatch",
                    item_id=cell.id,
                    message=f"workload {cell.workload.id!r} does not support aggregate replay",
                )
            )
        if "disagg" in modes and not caps.disaggregate:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="workload_mode_mismatch",
                    item_id=cell.id,
                    message=f"workload {cell.workload.id!r} does not support disaggregated replay",
                )
            )
    return ValidationReport(
        valid=not any(issue.severity == "error" for issue in issues),
        issues=issues,
        cell_count=len(cells),
    )
