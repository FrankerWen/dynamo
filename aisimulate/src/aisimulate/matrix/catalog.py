# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned, deterministic catalog loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from ._utils import read_document, resolve_path, sha256_file
from .config import (
    SCHEMA_VERSION,
    CatalogBundle,
    InputArtifact,
    MatrixExperimentConfig,
    ModelSpec,
    ModelVariantSpec,
    SearchPolicyConfig,
    SkuSpec,
    SLAProfile,
    WorkloadSpec,
)

_SpecT = TypeVar("_SpecT", bound=BaseModel)


def _validate_document(path: Path, payload: Any, expected_kind: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"catalog {path} must contain one object")
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"catalog {path} has schema_version={version!r}; expected {SCHEMA_VERSION}"
        )
    kind = payload.get("kind")
    if kind != expected_kind:
        raise ValueError(
            f"catalog {path} has kind={kind!r}; expected {expected_kind!r}"
        )
    return payload


def _load_entries(
    base_dir: Path,
    raw_paths: list[str],
    *,
    kind: str,
    entry_key: str,
    model: type[_SpecT],
) -> tuple[dict[str, _SpecT], list[InputArtifact]]:
    entries: dict[str, _SpecT] = {}
    artifacts: list[InputArtifact] = []
    for raw_path in raw_paths:
        path = resolve_path(base_dir, raw_path)
        payload = _validate_document(path, read_document(path), kind)
        raw_entries = payload.get(entry_key)
        if not isinstance(raw_entries, list):
            raise TypeError(f"catalog {path} field {entry_key!r} must be a list")
        for raw_entry in raw_entries:
            entry = model.model_validate(raw_entry)
            entry_id = str(entry.id)
            if entry_id in entries:
                raise ValueError(
                    f"duplicate {kind} id {entry_id!r} while loading {path}"
                )
            entries[entry_id] = entry
        artifacts.append(
            InputArtifact(
                role=f"catalog:{kind}", path=str(path), sha256=sha256_file(path)
            )
        )
    return dict(sorted(entries.items())), artifacts


def load_catalogs(config: MatrixExperimentConfig) -> CatalogBundle:
    """Load every catalog referenced by an experiment, failing on ambiguity."""
    base_dir = config.base_dir
    models, model_artifacts = _load_entries(
        base_dir,
        config.catalogs.models,
        kind="models",
        entry_key="models",
        model=ModelSpec,
    )
    model_variants, model_variant_artifacts = _load_entries(
        base_dir,
        config.catalogs.model_variants,
        kind="model_variants",
        entry_key="model_variants",
        model=ModelVariantSpec,
    )
    unknown_variant_models = sorted(
        {
            variant.model_id
            for variant in model_variants.values()
            if variant.model_id not in models
        }
    )
    if unknown_variant_models:
        raise ValueError(
            "model variant catalog references unknown model ids "
            f"{unknown_variant_models}"
        )
    skus, sku_artifacts = _load_entries(
        base_dir,
        config.catalogs.skus,
        kind="skus",
        entry_key="skus",
        model=SkuSpec,
    )
    workloads, workload_artifacts = _load_entries(
        base_dir,
        config.catalogs.workloads,
        kind="workloads",
        entry_key="workloads",
        model=WorkloadSpec,
    )
    slas, sla_artifacts = _load_entries(
        base_dir,
        config.catalogs.slas,
        kind="slas",
        entry_key="slas",
        model=SLAProfile,
    )

    policy_path = resolve_path(base_dir, config.catalogs.search_policy)
    policy_payload = _validate_document(
        policy_path, read_document(policy_path), "search_policy"
    )
    policy_data = dict(policy_payload)
    policy_data.pop("kind")
    search_policy = SearchPolicyConfig.model_validate(policy_data)
    policy_artifact = InputArtifact(
        role="catalog:search_policy",
        path=str(policy_path),
        sha256=sha256_file(policy_path),
    )

    artifacts = [
        InputArtifact(
            role="experiment",
            path=str(config.source_path),
            sha256=config.source_sha256,
        ),
        *model_artifacts,
        *model_variant_artifacts,
        *sku_artifacts,
        *workload_artifacts,
        *sla_artifacts,
        policy_artifact,
    ]
    return CatalogBundle(
        models=models,
        model_variants=model_variants,
        skus=skus,
        workloads=workloads,
        slas=slas,
        search_policy=search_policy,
        artifacts=artifacts,
    )
