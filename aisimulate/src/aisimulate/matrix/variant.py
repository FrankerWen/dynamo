# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic model-precision variant resolution before Matrix expansion."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import MissingSystemFlopsError
from aiconfigurator.sdk.models import (
    attention_op_keys,
    check_is_moe,
    get_model,
    get_model_family,
)
from aiconfigurator.sdk.perf_database import (
    get_database_view,
    get_latest_database_version,
    get_systems_paths,
    is_blackwell_system,
)
from aiconfigurator.sdk.task_v2 import Task
from aiconfigurator.sdk.utils import (
    HuggingFaceDownloadError,
    get_model_config_from_model_path,
)
from pydantic import BaseModel, ConfigDict, Field

from ._utils import fingerprint, sha256_file
from .config import (
    AICQuantizationSpec,
    CatalogBundle,
    MatrixExperimentConfig,
    ModelSpec,
    ModelVariantSpec,
    Selector,
    SkuSpec,
    VariantResolutionPolicy,
)

RESOLVER_VERSION = 1
Operator = Literal["gemm", "moe", "fmha", "kv_cache", "comm"]
SupportStatus = Literal[
    "supported_exact",
    "supported_empirical",
    "hardware_precision_unsupported",
    "kernel_quant_unsupported",
    "missing_perf_data",
    "checkpoint_incompatible",
    "model_unsupported",
    "invalid_variant",
    "unprobed",
]
ResolutionSource = Literal["explicit", "native", "catalog_fallback", "aic_enumerated"]

_OPERATORS: tuple[Operator, ...] = ("gemm", "moe", "fmha", "kv_cache", "comm")
_ENUMS: dict[Operator, type] = {
    "gemm": common.GEMMQuantMode,
    "moe": common.MoEQuantMode,
    "fmha": common.FMHAQuantMode,
    "kv_cache": common.KVCacheQuantMode,
    "comm": common.CommQuantMode,
}

# (weight bits, activation bits). Storage-only modes use the same value twice.
# Unknown modes deliberately have no entry and are explicit-only.
_PRECISION_SEMANTICS: dict[str, tuple[int, int]] = {
    "bfloat16": (16, 16),
    "float16": (16, 16),
    "half": (16, 16),
    "fp8": (8, 8),
    "fp8_block": (8, 8),
    "fp8_static": (8, 8),
    "int8_wo": (8, 16),
    "int4_wo": (4, 16),
    "nvfp4": (4, 4),
    "w4a8_mxfp4_mxfp8": (4, 8),
    "w4a8_mxfp4_mxfp8_trtllm": (4, 8),
    "w4a16_mxfp4": (4, 16),
    "w4a16_mxfp4_cutlass": (4, 16),
}
_BLACKWELL_KERNEL_MODES = frozenset(
    {"nvfp4", "w4a8_mxfp4_mxfp8", "w4a8_mxfp4_mxfp8_trtllm"}
)
_VALIDATION_ALIASES = {"w4a16_mxfp4_cutlass": "w4a16_mxfp4"}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VariantSupportResult(_StrictModel):
    status: SupportStatus
    reason: str
    failed_operators: list[Operator] = Field(default_factory=list)
    backend_version: str | None = None
    data_source: dict[str, Any] = Field(default_factory=dict)
    cache_key: str
    cache_hit: bool = False

    @property
    def supported(self) -> bool:
        return self.status in {"supported_exact", "supported_empirical"}


class VariantResolutionRecord(_StrictModel):
    id: str
    model_id: str
    sku_id: str
    backend: str
    variant: ModelVariantSpec
    source: ResolutionSource
    support: VariantSupportResult
    data_mode: Literal["silicon", "empirical"]
    selected: bool
    selection_reason: str


class VariantResolution(_StrictModel):
    records: list[VariantResolutionRecord]

    @property
    def selected_records(self) -> list[VariantResolutionRecord]:
        return [record for record in self.records if record.selected]


class VariantSupportProbe(Protocol):
    """Injectable AIC support boundary used by the resolver and unit tests."""

    def cache_key(
        self,
        variant: ModelVariantSpec,
        sku: SkuSpec,
        policy: VariantResolutionPolicy,
    ) -> str: ...

    def probe(
        self,
        variant: ModelVariantSpec,
        sku: SkuSpec,
        policy: VariantResolutionPolicy,
    ) -> VariantSupportResult: ...

    def available_modes(
        self,
        variant: ModelVariantSpec,
        sku: SkuSpec,
        operator: Operator,
        policy: VariantResolutionPolicy,
    ) -> list[str]: ...


def _hash_tree(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    files: list[tuple[str, Path]] = []
    for root in paths:
        root_label = "/".join(root.parts[-4:])
        if root.is_file():
            files.append((root_label, root))
        elif root.is_dir():
            files.extend(
                (f"{root_label}/{item.relative_to(root).as_posix()}", item)
                for item in root.rglob("*")
                if item.is_file()
            )
    for relative, path in sorted(files):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


class AICVariantSupportProbe:
    """Fail-closed AIC support probe with exact input identity."""

    def __init__(self) -> None:
        self._identity_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    @staticmethod
    def _backend_version(variant: ModelVariantSpec, sku: SkuSpec) -> str | None:
        return variant.backend_version or get_latest_database_version(
            system=sku.aic_system, backend=variant.backend
        )

    @staticmethod
    def _checkpoint_identity(variant: ModelVariantSpec) -> dict[str, Any]:
        try:
            info = get_model_config_from_model_path(variant.model_path)
        except (
            HuggingFaceDownloadError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return {
                "model_path": variant.model_path,
                "config_unavailable": f"{type(exc).__name__}: {exc}",
            }
        return {
            "model_path": variant.model_path,
            "architecture": info.get("architecture"),
            "raw_config": info.get("raw_config", {}),
        }

    def _database_identity(
        self, variant: ModelVariantSpec, sku: SkuSpec, version: str | None
    ) -> dict[str, Any]:
        key = (sku.aic_system, variant.backend, version or "")
        if key in self._identity_cache:
            return self._identity_cache[key]
        selected_paths: list[Path] = []
        for root_text in get_systems_paths():
            root = Path(root_text)
            system_yaml = root / f"{sku.aic_system}.yaml"
            if not system_yaml.is_file():
                continue
            selected_paths.append(system_yaml)
            try:
                payload = json.loads(
                    json.dumps(
                        yaml.safe_load(system_yaml.read_text(encoding="utf-8")) or {}
                    )
                )
            except (OSError, TypeError, ValueError):
                continue
            data_dir = payload.get("data_dir")
            if not isinstance(data_dir, str):
                continue
            system_data = root / data_dir
            if version:
                selected_paths.extend(
                    system_data.glob(f"**/{variant.backend}/{version}")
                )
            else:
                selected_paths.append(system_data)
        identity = {
            "system": sku.aic_system,
            "backend": variant.backend,
            "backend_version": version,
            "paths": [str(path.resolve()) for path in selected_paths],
            "sha256": _hash_tree(selected_paths),
        }
        self._identity_cache[key] = identity
        return identity

    def cache_key(
        self,
        variant: ModelVariantSpec,
        sku: SkuSpec,
        policy: VariantResolutionPolicy,
    ) -> str:
        version = self._backend_version(variant, sku)
        return fingerprint(
            {
                "resolver_version": RESOLVER_VERSION,
                "checkpoint": self._checkpoint_identity(variant),
                "quantization": variant.quantization,
                "kernel_family": variant.kernel_family,
                "sku": sku.aic_system,
                "backend": variant.backend,
                "backend_version": version,
                "aic_database": self._database_identity(variant, sku, version),
                "data_mode": policy.data_mode,
            }
        )

    @staticmethod
    def _result(
        *,
        status: SupportStatus,
        reason: str,
        cache_key: str,
        version: str | None,
        failed: Sequence[Operator] = (),
        data_source: Mapping[str, Any] | None = None,
    ) -> VariantSupportResult:
        return VariantSupportResult(
            status=status,
            reason=reason,
            failed_operators=list(dict.fromkeys(failed)),
            backend_version=version,
            data_source=dict(data_source or {}),
            cache_key=cache_key,
        )

    @staticmethod
    def _quant_enum(variant: ModelVariantSpec, operator: Operator) -> Any:
        return _ENUMS[operator][getattr(variant.quantization, operator)]

    @staticmethod
    def _exact_requirements(
        task: Task, variant: ModelVariantSpec
    ) -> dict[Operator, tuple[str, str]]:
        ctx_op, gen_op = task._attention_op_keys("agg")
        required: dict[Operator, tuple[str, str]] = {
            "gemm": ("gemm", variant.quantization.gemm),
            "fmha": (ctx_op, variant.quantization.fmha),
            "kv_cache": (gen_op, variant.quantization.kv_cache),
        }
        if task.is_moe:
            required["moe"] = ("moe", variant.quantization.moe)
        return required

    def probe(
        self,
        variant: ModelVariantSpec,
        sku: SkuSpec,
        policy: VariantResolutionPolicy,
    ) -> VariantSupportResult:
        cache_key = self.cache_key(variant, sku, policy)
        version = self._backend_version(variant, sku)
        database_identity = {
            **self._database_identity(variant, sku, version),
            "checkpoint_config_sha256": fingerprint(self._checkpoint_identity(variant)),
        }
        kernel_error = _kernel_contract_error(variant)
        if kernel_error is not None:
            return self._result(
                status="invalid_variant",
                reason=kernel_error,
                cache_key=cache_key,
                version=version,
                failed=variant.fallback_for,
                data_source=database_identity,
            )
        if version is None:
            return self._result(
                status="missing_perf_data",
                reason="AIC has no backend version for this system/backend",
                cache_key=cache_key,
                version=None,
                data_source=database_identity,
            )
        if not is_blackwell_system(sku.aic_system):
            failed = [
                operator
                for operator in ("gemm", "moe")
                if getattr(variant.quantization, operator) in _BLACKWELL_KERNEL_MODES
            ]
            if failed:
                return self._result(
                    status="kernel_quant_unsupported",
                    reason=(
                        "variant requires a Blackwell-only FP4 kernel path; "
                        "FP8 tensor-core support alone is insufficient"
                    ),
                    cache_key=cache_key,
                    version=version,
                    failed=failed,
                    data_source=database_identity,
                )
        database_mode = (
            common.DatabaseMode.SILICON.name
            if policy.data_mode == "silicon"
            else common.DatabaseMode.HYBRID.name
        )
        try:
            database = get_database_view(
                sku.aic_system,
                variant.backend,
                version,
                allow_missing_data=policy.data_mode == "empirical",
                database_mode=database_mode,
                # Resolver silicon means the selected backend/version's own
                # collected rows. AIC's serving predictor may use its shared
                # sibling layer later, but that cannot upgrade compatibility
                # to supported_exact here.
                shared_layer=False if policy.data_mode == "silicon" else None,
            )
            if database is None:
                return self._result(
                    status="missing_perf_data",
                    reason="AIC performance database is unavailable",
                    cache_key=cache_key,
                    version=version,
                    data_source=database_identity,
                )
            compute_operators: tuple[Operator, ...] = (
                ("gemm", "moe", "fmha")
                if check_is_moe(variant.model_path)
                else ("gemm", "fmha")
            )
            for operator in compute_operators:
                common.get_quant_tc_flops(
                    database.system_spec, self._quant_enum(variant, operator)
                )
            task = Task(
                serving_mode="agg",
                isl=128,
                osl=8,
                pareto_sweep=False,
                model_path=variant.model_path,
                system_name=sku.aic_system,
                backend_name=variant.backend,
                backend_version=version,
                database_mode=database_mode,
                gemm_quant_mode=self._quant_enum(variant, "gemm"),
                moe_quant_mode=self._quant_enum(variant, "moe"),
                fmha_quant_mode=self._quant_enum(variant, "fmha"),
                kvcache_quant_mode=self._quant_enum(variant, "kv_cache"),
                comm_quant_mode=self._quant_enum(variant, "comm"),
            )
            task.validate()
            model_config = dataclasses.replace(
                task.build_model_config(role="agg"),
                tp_size=1,
                pp_size=1,
                attention_dp_size=1,
                moe_tp_size=1,
                moe_ep_size=1,
                cp_size=1,
            )
            get_model(
                model_path=variant.model_path,
                model_config=model_config,
                backend_name=variant.backend,
            )
            exact_database = database
            if policy.data_mode == "empirical":
                exact_database = get_database_view(
                    sku.aic_system,
                    variant.backend,
                    version,
                    allow_missing_data=False,
                    database_mode=common.DatabaseMode.SILICON.name,
                    shared_layer=False,
                )
            missing: list[Operator] = []
            supported = (
                {}
                if exact_database is None
                else getattr(exact_database, "supported_quant_mode", {}) or {}
            )
            for operator, (op_key, mode) in self._exact_requirements(
                task, variant
            ).items():
                names = list(supported.get(op_key, []) or [])
                accepted = mode in names or _VALIDATION_ALIASES.get(mode) in names
                if not accepted:
                    missing.append(operator)
            if missing and policy.data_mode == "silicon":
                return self._result(
                    status="missing_perf_data",
                    reason=(
                        "exact silicon quant tables are missing for operators "
                        + ", ".join(missing)
                    ),
                    cache_key=cache_key,
                    version=version,
                    failed=missing,
                    data_source=database_identity,
                )
            status: SupportStatus = (
                "supported_empirical" if missing else "supported_exact"
            )
            return self._result(
                status=status,
                reason=(
                    "AIC static build and exact silicon tables passed"
                    if status == "supported_exact"
                    else "AIC static build passed with empirical/hybrid data allowed"
                ),
                cache_key=cache_key,
                version=version,
                data_source={
                    **database_identity,
                    "database_mode": database_mode,
                    "shared_layer": False
                    if policy.data_mode == "silicon"
                    else "AIC default hybrid transfer",
                    "missing_exact_operators": missing,
                },
            )
        except MissingSystemFlopsError as exc:
            failed = _operators_from_message(str(exc), variant)
            return self._result(
                status="hardware_precision_unsupported",
                reason=f"{type(exc).__name__}: {exc}",
                cache_key=cache_key,
                version=version,
                failed=failed,
                data_source=database_identity,
            )
        except (
            HuggingFaceDownloadError,
            KeyError,
            NotImplementedError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            message = str(exc)
            failed = _operators_from_message(message, variant)
            lowered = message.lower()
            if isinstance(exc, HuggingFaceDownloadError):
                status = "checkpoint_incompatible"
            elif "native fp4" in lowered and "hopper" in lowered:
                status = "hardware_precision_unsupported"
                failed = ["moe"]
            elif "unsupported" in lowered and "quant mode" in lowered:
                status = "missing_perf_data"
            elif "checkpoint" in lowered or "model path" in lowered:
                status = "checkpoint_incompatible"
            elif "unknown model" in lowered or "model family" in lowered:
                status = "model_unsupported"
            else:
                status = "invalid_variant"
            return self._result(
                status=status,
                reason=f"{type(exc).__name__}: {exc}",
                cache_key=cache_key,
                version=version,
                failed=failed,
                data_source=database_identity,
            )

    def available_modes(
        self,
        variant: ModelVariantSpec,
        sku: SkuSpec,
        operator: Operator,
        policy: VariantResolutionPolicy,
    ) -> list[str]:
        version = self._backend_version(variant, sku)
        if version is None:
            return []
        database = get_database_view(
            sku.aic_system,
            variant.backend,
            version,
            allow_missing_data=policy.data_mode == "empirical",
            database_mode=(
                common.DatabaseMode.SILICON.name
                if policy.data_mode == "silicon"
                else common.DatabaseMode.HYBRID.name
            ),
            shared_layer=False if policy.data_mode == "silicon" else None,
        )
        if database is None:
            return []
        if operator == "comm":
            raw_modes = list(_ENUMS[operator].__members__)
        else:
            op_key = operator
            if operator in {"fmha", "kv_cache"}:
                family = get_model_family(variant.model_path)
                ctx_op, gen_op = attention_op_keys(family, variant.backend)
                op_key = ctx_op if operator == "fmha" else gen_op
            raw_modes = list(
                (getattr(database, "supported_quant_mode", {}) or {}).get(op_key, [])
                or []
            )
            if policy.data_mode == "empirical":
                raw_modes.extend(_ENUMS[operator].__members__)
        return sorted(
            {
                mode
                for mode in raw_modes
                if mode in _ENUMS[operator].__members__ and mode in _PRECISION_SEMANTICS
            }
        )


def _operators_from_message(message: str, variant: ModelVariantSpec) -> list[Operator]:
    lowered = message.lower()
    found: list[Operator] = []
    aliases: dict[Operator, tuple[str, ...]] = {
        "gemm": ("gemm",),
        "moe": ("moe", "expert"),
        "fmha": ("fmha", "context", "attention"),
        "kv_cache": ("kvcache", "kv cache", "generation"),
        "comm": ("comm", "nccl"),
    }
    for operator, terms in aliases.items():
        mode = getattr(variant.quantization, operator).lower()
        if mode in lowered or any(term in lowered for term in terms):
            found.append(operator)
    return found or ["gemm", "moe", "fmha"]


def _kernel_contract_error(variant: ModelVariantSpec) -> str | None:
    """Reject kernel labels that AIC replay cannot distinguish concretely."""

    family = variant.kernel_family.lower()
    if family in {"auto", "aic_enumerated"}:
        return None
    if (
        family in {"cutlass", "cutlass_or_marlin"}
        and "cutlass" in variant.quantization.moe
    ):
        return None
    if (
        family == "vllm_fp8_moe"
        and variant.backend == "vllm"
        and variant.quantization.moe in {"fp8", "fp8_block"}
    ):
        return None
    if family == "trtllm_gen" and variant.quantization.moe.endswith("_trtllm"):
        return None
    return (
        f"kernel_family={variant.kernel_family!r} is not encoded by the concrete "
        "AIC backend/quantization contract; use kernel_family='auto' or an "
        "AIC quant mode with a distinguishable kernel implementation"
    )


def precision_semantics(mode: str) -> tuple[int, int] | None:
    """Return centralized W/A precision semantics for automatic resolution."""

    return _PRECISION_SEMANTICS.get(mode)


def is_non_degrading_mode(candidate: str, native: str) -> bool:
    candidate_precision = precision_semantics(candidate)
    native_precision = precision_semantics(native)
    return bool(
        candidate_precision is not None
        and native_precision is not None
        and candidate_precision[0] >= native_precision[0]
        and candidate_precision[1] >= native_precision[1]
    )


def _selector_entries(
    entries: Mapping[str, ModelVariantSpec], selector: Selector
) -> list[ModelVariantSpec]:
    if selector.ids is not None:
        unknown = sorted(set(selector.ids) - set(entries))
        if unknown:
            raise ValueError(f"model_variant selector references unknown ids {unknown}")
        ids = list(dict.fromkeys(selector.ids))
    else:
        ids = sorted(entries)
    required_tags = set(selector.tags)
    excluded = set(selector.exclude)
    selected = [
        entries[entry_id]
        for entry_id in ids
        if entry_id not in excluded
        and (not required_tags or required_tags.issubset(entries[entry_id].tags))
    ]
    if not selected:
        raise ValueError("model_variant selector matched no entries")
    return selected


def configured_backend(catalogs: CatalogBundle) -> str:
    raw = catalogs.search_policy.spica.search_space.get("backend", ["vllm"])
    backends = [raw] if isinstance(raw, str) else list(raw)
    if len(backends) != 1:
        raise ValueError(
            "variant resolver requires exactly one policy backend; split multi-backend searches into separate experiments"
        )
    backend = str(backends[0])
    if backend not in {"vllm", "sglang", "trtllm"}:
        raise ValueError(f"unsupported policy backend {backend!r}")
    return backend


def validate_variant_catalog(
    config: MatrixExperimentConfig,
    catalogs: CatalogBundle,
    selected_model_ids: set[str],
) -> None:
    backend = configured_backend(catalogs)
    associated = [
        variant
        for variant in catalogs.model_variants.values()
        if variant.model_id in selected_model_ids and variant.backend == backend
    ]
    natives_by_model: dict[str, ModelVariantSpec] = {}
    for model_id in sorted(selected_model_ids):
        natives = [
            variant
            for variant in associated
            if variant.model_id == model_id and variant.role == "native"
        ]
        if len(natives) != 1:
            raise ValueError(
                f"model {model_id!r}/backend {backend!r} must have exactly one native variant; found {len(natives)}"
            )
        natives_by_model[model_id] = natives[0]
    if config.matrix.model_variant_selector is not None:
        explicit = _selector_entries(
            catalogs.model_variants, config.matrix.model_variant_selector
        )
        wrong_backend = sorted(
            variant.id
            for variant in explicit
            if variant.model_id in selected_model_ids and variant.backend != backend
        )
        if wrong_backend:
            raise ValueError(
                f"explicit model variants do not match policy backend {backend!r}: {wrong_backend}"
            )
        missing = sorted(
            selected_model_ids
            - {variant.model_id for variant in explicit if variant.backend == backend}
        )
        if missing:
            raise ValueError(
                f"selected models have no matching explicit variants: {missing}"
            )
        return
    for model_id in sorted(selected_model_ids):
        native = natives_by_model[model_id]
        unknown_native = sorted(
            operator
            for operator in _OPERATORS
            if precision_semantics(getattr(native.quantization, operator)) is None
        )
        if unknown_native:
            raise ValueError(
                f"native variant {native.id!r} uses precision modes without automatic semantics for {unknown_native}; select variants explicitly"
            )
        for fallback in (
            variant
            for variant in associated
            if variant.model_id == model_id and variant.role == "fallback"
        ):
            valid, reason = _fallback_recipe_valid(
                native, fallback, set(fallback.fallback_for)
            )
            if not valid:
                raise ValueError(f"invalid fallback variant {fallback.id!r}: {reason}")


def _record(
    *,
    model_id: str,
    sku: SkuSpec,
    variant: ModelVariantSpec,
    source: ResolutionSource,
    support: VariantSupportResult,
    policy: VariantResolutionPolicy,
    selected: bool,
    reason: str,
) -> VariantResolutionRecord:
    payload = {
        "model": model_id,
        "sku": sku.id,
        "backend": variant.backend,
        "variant": variant.model_dump(mode="python"),
        "source": source,
        "support_cache_key": support.cache_key,
    }
    return VariantResolutionRecord(
        id=f"variant_resolution_{fingerprint(payload)[:24]}",
        model_id=model_id,
        sku_id=sku.id,
        backend=variant.backend,
        variant=variant,
        source=source,
        support=support,
        data_mode=policy.data_mode,
        selected=selected,
        selection_reason=reason,
    )


def _probe_cached(
    probe: VariantSupportProbe,
    variant: ModelVariantSpec,
    sku: SkuSpec,
    policy: VariantResolutionPolicy,
    cache: Mapping[str, VariantSupportResult],
) -> VariantSupportResult:
    cache_key = probe.cache_key(variant, sku, policy)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached.model_copy(update={"cache_hit": True})
    result = probe.probe(variant, sku, policy)
    if result.cache_key != cache_key:
        raise ValueError("variant support probe returned a mismatched cache key")
    return result


def support_accepted(
    support: VariantSupportResult,
    policy: VariantResolutionPolicy | Literal["silicon", "empirical"],
) -> bool:
    data_mode = (
        policy.data_mode if isinstance(policy, VariantResolutionPolicy) else policy
    )
    return support.status == "supported_exact" or (
        data_mode == "empirical" and support.status == "supported_empirical"
    )


def _fallback_recipe_valid(
    native: ModelVariantSpec,
    fallback: ModelVariantSpec,
    failed: set[Operator],
) -> tuple[bool, str]:
    changed = {
        operator
        for operator in _OPERATORS
        if getattr(native.quantization, operator)
        != getattr(fallback.quantization, operator)
    }
    declared = set(fallback.fallback_for)
    if not changed:
        return False, "fallback recipe does not change any quantized operator"
    if not changed.issubset(declared):
        return (
            False,
            f"changed operators {sorted(changed)} exceed fallback_for {sorted(declared)}",
        )
    if not changed.issubset(failed):
        return (
            False,
            f"recipe changes operators that did not fail native probing: {sorted(changed - failed)}",
        )
    for operator in changed:
        candidate = getattr(fallback.quantization, operator)
        native_mode = getattr(native.quantization, operator)
        if not is_non_degrading_mode(candidate, native_mode):
            return (
                False,
                f"automatic fallback would lower or use unknown precision for {operator}: {native_mode} -> {candidate}",
            )
    return (
        True,
        "catalog fallback changes only failed operators without lowering precision",
    )


def _enumerated_variants(
    model: ModelSpec,
    native: ModelVariantSpec,
    sku: SkuSpec,
    failed: list[Operator],
    policy: VariantResolutionPolicy,
    probe: VariantSupportProbe,
) -> list[ModelVariantSpec]:
    choices: list[list[str]] = []
    for operator in failed:
        native_mode = getattr(native.quantization, operator)
        candidates = [
            mode
            for mode in probe.available_modes(native, sku, operator, policy)
            if mode != native_mode and is_non_degrading_mode(mode, native_mode)
        ]
        if not candidates:
            return []
        choices.append(sorted(set(candidates)))
    candidate_count = 1
    for modes in choices:
        candidate_count *= len(modes)
    if candidate_count > policy.max_auto_variants_per_model_sku:
        raise ValueError(
            f"automatic variant enumeration for model={model.id!r}, sku={sku.id!r} produced {candidate_count} candidates, exceeding max_auto_variants_per_model_sku={policy.max_auto_variants_per_model_sku}; add model-level fallback recipes or select variants explicitly"
        )
    variants: list[ModelVariantSpec] = []
    for values in itertools.product(*choices):
        updates = dict(native.quantization.model_dump(mode="python"))
        updates.update(dict(zip(failed, values, strict=True)))
        quantization = AICQuantizationSpec.model_validate(updates)
        suffix = fingerprint(
            {
                "model": model.id,
                "sku": sku.id,
                "backend": native.backend,
                "quantization": quantization,
            }
        )[:12]
        variants.append(
            native.model_copy(
                update={
                    "id": f"{model.id}_aic_auto_{suffix}",
                    "role": "fallback",
                    "kernel_family": "aic_enumerated",
                    "weight_precision": (
                        f"auto:{quantization.gemm}/{quantization.moe}"
                    ),
                    "quantization": quantization,
                    "fallback_for": list(failed),
                    "deployment_fidelity": "hypothetical",
                    "tags": sorted(set(native.tags) | {"aic-enumerated", "what-if"}),
                }
            )
        )
    return variants


def resolve_variants(
    config: MatrixExperimentConfig,
    catalogs: CatalogBundle,
    *,
    models: Sequence[ModelSpec],
    skus: Sequence[SkuSpec],
    probe: VariantSupportProbe | None = None,
    support_cache: Mapping[str, VariantSupportResult] | None = None,
) -> VariantResolution:
    """Resolve exactly which concrete variants each model/SKU pair expands."""

    active_probe = probe or AICVariantSupportProbe()
    cache = support_cache or {}
    selected_model_ids = {model.id for model in models}
    validate_variant_catalog(config, catalogs, selected_model_ids)
    backend = configured_backend(catalogs)
    policy = config.matrix.variant_resolution
    records: list[VariantResolutionRecord] = []

    if config.matrix.model_variant_selector is not None:
        explicit = [
            variant
            for variant in _selector_entries(
                catalogs.model_variants, config.matrix.model_variant_selector
            )
            if variant.model_id in selected_model_ids and variant.backend == backend
        ]
        for variant in explicit:
            for sku in skus:
                support = _probe_cached(active_probe, variant, sku, policy, cache)
                records.append(
                    _record(
                        model_id=variant.model_id,
                        sku=sku,
                        variant=variant,
                        source="explicit",
                        support=support,
                        policy=policy,
                        selected=True,
                        reason="explicit selector: no automatic replacement",
                    )
                )
        return VariantResolution(records=sorted(records, key=lambda item: item.id))

    variants = list(catalogs.model_variants.values())
    for model in models:
        native = next(
            variant
            for variant in variants
            if variant.model_id == model.id
            and variant.backend == backend
            and variant.role == "native"
        )
        recipes = [
            variant
            for variant in variants
            if variant.model_id == model.id
            and variant.backend == backend
            and variant.role == "fallback"
        ]
        for sku in skus:
            native_support = _probe_cached(active_probe, native, sku, policy, cache)
            if support_accepted(native_support, policy):
                records.append(
                    _record(
                        model_id=model.id,
                        sku=sku,
                        variant=native,
                        source="native",
                        support=native_support,
                        policy=policy,
                        selected=True,
                        reason="native variant supported; automatic resolution short-circuited",
                    )
                )
                continue

            trigger = native_support.status in {
                "hardware_precision_unsupported",
                "kernel_quant_unsupported",
            }
            records.append(
                _record(
                    model_id=model.id,
                    sku=sku,
                    variant=native,
                    source="native",
                    support=native_support,
                    policy=policy,
                    selected=not trigger,
                    reason=(
                        "native failure does not permit precision fallback"
                        if not trigger
                        else "native precision/kernel unsupported; evaluating fallbacks"
                    ),
                )
            )
            if not trigger:
                continue

            failed = set(native_support.failed_operators)
            if not failed:
                raise ValueError(
                    f"native variant {native.id!r} triggered fallback without failed operator attribution"
                )
            selected_fallbacks = 0
            if recipes:
                if len(recipes) > policy.max_auto_variants_per_model_sku:
                    raise ValueError(
                        f"model-level fallback recipes for model={model.id!r}, sku={sku.id!r} contain {len(recipes)} candidates, exceeding max_auto_variants_per_model_sku={policy.max_auto_variants_per_model_sku}; select variants explicitly"
                    )
                fallback_variants = recipes
                source: ResolutionSource = "catalog_fallback"
            else:
                fallback_variants = _enumerated_variants(
                    model,
                    native,
                    sku,
                    sorted(failed),
                    policy,
                    active_probe,
                )
                source = "aic_enumerated"
            for fallback in fallback_variants:
                valid, reason = _fallback_recipe_valid(native, fallback, failed)
                if not valid:
                    support = VariantSupportResult(
                        status="invalid_variant",
                        reason=reason,
                        failed_operators=list(fallback.fallback_for),
                        cache_key=active_probe.cache_key(fallback, sku, policy),
                    )
                    selected = False
                else:
                    support = _probe_cached(active_probe, fallback, sku, policy, cache)
                    selected = support_accepted(support, policy)
                    if selected:
                        selected_fallbacks += 1
                records.append(
                    _record(
                        model_id=model.id,
                        sku=sku,
                        variant=fallback,
                        source=source,
                        support=support,
                        policy=policy,
                        selected=selected,
                        reason=reason
                        if not selected
                        else "supported fallback selected",
                    )
                )
            if selected_fallbacks == 0:
                # Preserve an auditable unsupported Cell instead of silently
                # dropping the model/SKU pair from the experiment.
                for index in range(len(records) - 1, -1, -1):
                    record = records[index]
                    if (
                        record.model_id == model.id
                        and record.sku_id == sku.id
                        and record.source == "native"
                    ):
                        records[index] = record.model_copy(
                            update={
                                "selected": True,
                                "selection_reason": "no supported non-degrading fallback; retain unsupported native Cell",
                            }
                        )
                        break
    return VariantResolution(records=sorted(records, key=lambda item: item.id))


def structural_resolution(
    config: MatrixExperimentConfig,
    catalogs: CatalogBundle,
    *,
    models: Sequence[ModelSpec],
    skus: Sequence[SkuSpec],
) -> VariantResolution:
    """Produce non-AIC placeholder records for validation only."""

    selected_model_ids = {model.id for model in models}
    validate_variant_catalog(config, catalogs, selected_model_ids)
    backend = configured_backend(catalogs)
    if config.matrix.model_variant_selector is None:
        variants = [
            variant
            for variant in catalogs.model_variants.values()
            if variant.model_id in selected_model_ids
            and variant.backend == backend
            and variant.role == "native"
        ]
        source: ResolutionSource = "native"
    else:
        variants = [
            variant
            for variant in _selector_entries(
                catalogs.model_variants, config.matrix.model_variant_selector
            )
            if variant.model_id in selected_model_ids and variant.backend == backend
        ]
        source = "explicit"
    records = []
    for variant in variants:
        for sku in skus:
            cache_key = fingerprint(
                {
                    "resolver_version": RESOLVER_VERSION,
                    "unprobed": True,
                    "model_path": variant.model_path,
                    "quantization": variant.quantization,
                    "sku": sku.aic_system,
                    "backend": backend,
                    "data_mode": config.matrix.variant_resolution.data_mode,
                }
            )
            support = VariantSupportResult(
                status="unprobed",
                reason="structural validation does not load AIC",
                cache_key=cache_key,
            )
            records.append(
                _record(
                    model_id=variant.model_id,
                    sku=sku,
                    variant=variant,
                    source=source,
                    support=support,
                    policy=config.matrix.variant_resolution,
                    selected=True,
                    reason="structural placeholder",
                )
            )
    return VariantResolution(records=sorted(records, key=lambda item: item.id))
