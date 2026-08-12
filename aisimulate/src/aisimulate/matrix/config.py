# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned public configuration models for reusable matrix searches."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

from aiconfigurator_core.sdk.common import (
    CommQuantMode,
    FMHAQuantMode,
    GEMMQuantMode,
    KVCacheQuantMode,
    MoEQuantMode,
)
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from aisimulate.spica.config import (
    KVFootprintPolicy,
    KVFootprintSpec,
    PDTransferTopology,
)

from ._utils import read_document, sha256_file

SCHEMA_VERSION = 1


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CatalogRefs(_StrictModel):
    models: list[str]
    model_variants: list[str]
    skus: list[str]
    workloads: list[str]
    slas: list[str]
    search_policy: str

    @field_validator(
        "models", "model_variants", "skus", "workloads", "slas", mode="before"
    )
    @classmethod
    def _one_or_many(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value


class Selector(_StrictModel):
    ids: list[str] | None = None
    tags: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _reject_empty_ids(self) -> Selector:
        if self.ids is not None and not self.ids:
            raise ValueError("selector.ids must be omitted or contain at least one id")
        return self


class VariantResolutionPolicy(_StrictModel):
    """Controls the pre-Matrix precision compatibility resolver."""

    data_mode: Literal["silicon", "empirical"] = "silicon"
    max_auto_variants_per_model_sku: int = Field(default=4, ge=1)


class MatrixSelection(_StrictModel):
    model_selector: Selector = Field(default_factory=Selector)
    # Presence is intentional: omitted means automatic native/fallback
    # resolution, while any supplied selector means "run exactly these".
    model_variant_selector: Selector | None = None
    sku_selector: Selector = Field(default_factory=Selector)
    workload_selector: Selector = Field(default_factory=Selector)
    sla_profile: str
    variant_resolution: VariantResolutionPolicy = Field(
        default_factory=VariantResolutionPolicy
    )


class RunSettings(_StrictModel):
    cell_workers: int = Field(default=1, ge=1)
    topk: int = Field(default=20, ge=1)
    simulator_revision: str | None = None


class MatrixExperimentConfig(_StrictModel):
    """Top-level experiment config. Relative paths are anchored to its file."""

    schema_version: Literal[1] = SCHEMA_VERSION
    catalogs: CatalogRefs
    matrix: MatrixSelection
    run: RunSettings = Field(default_factory=RunSettings)
    _source_path: Path = PrivateAttr()
    _source_sha256: str = PrivateAttr()

    @classmethod
    def from_file(cls, path: str | Path) -> MatrixExperimentConfig:
        source_path = Path(path).expanduser().resolve()
        config = cls.model_validate(read_document(source_path))
        config._source_path = source_path
        config._source_sha256 = sha256_file(source_path)
        return config

    @property
    def source_path(self) -> Path:
        if not hasattr(self, "_source_path"):
            raise RuntimeError("MatrixExperimentConfig must be loaded with from_file()")
        return self._source_path

    @property
    def base_dir(self) -> Path:
        return self.source_path.parent

    @property
    def source_sha256(self) -> str:
        if not hasattr(self, "_source_sha256"):
            raise RuntimeError("MatrixExperimentConfig must be loaded with from_file()")
        return self._source_sha256


class ParallelConstraint(_StrictModel):
    tp: list[int] | None = None
    attention_dp: list[int] | None = None
    moe_tp: list[int] | None = None
    moe_ep: list[int] | None = None
    max_gpus_per_worker: int | None = Field(default=None, gt=0)


class ModelSpec(_StrictModel):
    id: str
    # Deprecated schema-v1 compatibility fields. Explicit model-variant
    # catalogs are the deployment source of truth and MatrixCell never reads
    # these values.
    model_path: str | None = None
    support_tier: Literal["formal", "experimental", "unsupported"] | None = None
    kv_footprint: KVFootprintSpec | None = None
    context_length: int | None = Field(default=None, gt=0)
    tags: list[str] = Field(default_factory=list)
    parallel: ParallelConstraint = Field(default_factory=ParallelConstraint)

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_kv_scalar(cls, value: Any) -> Any:
        """Keep schema-v1 catalogs readable while marking their scope provenance."""

        if not isinstance(value, dict) or "kv_bytes_per_token" not in value:
            return value
        data = dict(value)
        scalar = data.pop("kv_bytes_per_token")
        if data.get("kv_footprint") is not None:
            raise ValueError(
                "model catalog cannot set both kv_footprint and legacy kv_bytes_per_token"
            )
        data["kv_footprint"] = {
            "source": "explicit_linear",
            "layout": "legacy_unspecified",
            "scope": "per_rank",
            "explicit_bytes_per_token": scalar,
        }
        return data


class AICQuantizationSpec(_StrictModel):
    """Complete AIC precision contract for one deployable model variant.

    Weight precision is not one scalar for modern MoE models: dense GEMMs and
    expert GEMMs can use different formats. Attention, KV-cache storage, and
    collectives are also modeled independently and therefore remain explicit.
    """

    gemm: str
    moe: str
    fmha: str
    kv_cache: str
    comm: str

    @model_validator(mode="after")
    def _validate_against_installed_aic(self) -> AICQuantizationSpec:
        # Validate against the linked AIC version rather than freezing today's
        # enum into Matrix. A future AIC mode is usable by adding a catalog
        # variant only; misspellings still fail before matrix expansion.
        enums = {
            "gemm": GEMMQuantMode,
            "moe": MoEQuantMode,
            "fmha": FMHAQuantMode,
            "kv_cache": KVCacheQuantMode,
            "comm": CommQuantMode,
        }
        for field_name, enum_type in enums.items():
            value = getattr(self, field_name)
            if value not in enum_type.__members__:
                raise ValueError(
                    f"unknown AIC {field_name} quantization mode {value!r}; "
                    f"installed choices are {sorted(enum_type.__members__)}"
                )
        return self

    def as_aic_fields(self) -> dict[str, str]:
        """Return the exact field names consumed by DynoSim's AIC binding."""

        return {
            "aic_gemm_dtype": self.gemm,
            "aic_moe_dtype": self.moe,
            "aic_fmha_dtype": self.fmha,
            "aic_kv_cache_dtype": self.kv_cache,
            "aic_comm_dtype": self.comm,
        }


class ModelVariantSpec(_StrictModel):
    """A concrete checkpoint plus its complete deployment precision profile."""

    id: str
    model_id: str
    role: Literal["native", "fallback"] = "native"
    model_path: str
    backend: Literal["vllm", "sglang", "trtllm"] = "vllm"
    backend_version: str | None = None
    kernel_family: str = "auto"
    weight_precision: str
    quantization: AICQuantizationSpec
    fallback_for: list[Literal["gemm", "moe", "fmha", "kv_cache", "comm"]] = Field(
        default_factory=list
    )
    support_tier: Literal["formal", "experimental", "unsupported"] = "formal"
    deployment_fidelity: Literal["deployable", "hypothetical"] = "deployable"
    quality_profile: str | None = None
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _warn_legacy_role(cls, value: Any) -> Any:
        if isinstance(value, dict) and "role" not in value:
            warnings.warn(
                "model variant entries without role are deprecated and are "
                "temporarily interpreted as role='native'",
                DeprecationWarning,
                stacklevel=3,
            )
        return value

    @field_validator("weight_precision")
    @classmethod
    def _nonempty_weight_precision(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("weight_precision must not be empty")
        return value

    @model_validator(mode="after")
    def _validate_variant_role(self) -> ModelVariantSpec:
        if not self.kernel_family.strip():
            raise ValueError("kernel_family must not be empty")
        if self.role == "native" and self.fallback_for:
            raise ValueError("a native model variant cannot set fallback_for")
        if self.role == "fallback" and not self.fallback_for:
            raise ValueError("a fallback model variant must set fallback_for")
        if len(self.fallback_for) != len(set(self.fallback_for)):
            raise ValueError("fallback_for contains duplicate operators")
        return self


class SkuSpec(_StrictModel):
    id: str
    aic_system: str
    compute_nic_gbytes_per_s_per_gpu: float = Field(
        gt=0,
        validation_alias=AliasChoices(
            "compute_nic_gbytes_per_s_per_gpu",
            "compute_nic_gbps_per_gpu",
        ),
        description=(
            "Decimal gigabytes per second (1 GB/s = 1e9 byte/s), per GPU, "
            "single direction. The legacy *_gbps_per_gpu input name is accepted "
            "for schema-v1 catalog compatibility."
        ),
    )
    pd_transfer_topology: PDTransferTopology
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_flat_topology(cls, value: Any) -> Any:
        """Accept the schema-v1 flat local/compute fields as an upgrade path.

        New catalogs use the explicit domain ladder.  The legacy spelling could
        only represent one local and one remote bandwidth, so both rack-local
        tiers inherit the local value and the remote tier inherits compute NIC.
        """

        if not isinstance(value, dict) or "pd_transfer_topology" in value:
            return value
        data = dict(value)
        local = data.pop(
            "local_fabric_gbytes_per_s_per_gpu",
            data.pop("local_fabric_gbps_per_gpu", None),
        )
        gpus_per_node = data.pop("gpus_per_node", None)
        gpus_per_rack = data.pop("gpus_per_rack", None)
        if local is None or gpus_per_node is None:
            return value
        compute = data.get(
            "compute_nic_gbytes_per_s_per_gpu",
            data.get("compute_nic_gbps_per_gpu"),
        )
        if compute is None:
            return value
        topology: dict[str, Any] = {
            "gpus_per_node": gpus_per_node,
            "intra_node_gbytes_per_s_per_gpu": local,
        }
        if gpus_per_rack is None:
            topology["inter_node_gbytes_per_s_per_gpu"] = compute
        else:
            topology.update(
                {
                    "gpus_per_rack": gpus_per_rack,
                    "intra_rack_gbytes_per_s_per_gpu": local,
                    "inter_rack_gbytes_per_s_per_gpu": compute,
                }
            )
        data["pd_transfer_topology"] = topology
        return data


class WorkloadSpec(_StrictModel):
    id: str
    loader: str
    tags: list[str] = Field(default_factory=list)
    config: dict[str, Any]


class SLAProfile(_StrictModel):
    id: str
    ttft_ms: float = Field(gt=0)
    tpot_ms: float = Field(gt=0)
    tps_per_user: float = Field(gt=0)
    p95_ttft_ms: float | None = Field(default=None, gt=0)
    p95_tpot_ms: float | None = Field(default=None, gt=0)
    p10_tps_per_user: float | None = Field(default=None, gt=0)
    min_goodput_ratio: float = Field(default=0.99, ge=0, le=1)
    min_completion_rate: float = Field(default=1.0, ge=0, le=1)


class PriceSpec(_StrictModel):
    id: str
    sku_id: str
    currency: str = "USD"
    gpu_hour: float = Field(ge=0)
    g2_gib_hour: float = Field(default=0.0, ge=0)
    network_gb: float = Field(default=0.0, ge=0)
    all_in_multiplier: float = Field(default=1.0, ge=1.0)
    source_url: str | None = None
    source_date: str | None = None
    basis: str | None = None
    notes: str | None = None


class StaticPrunePolicy(_StrictModel):
    max_candidates: int = Field(default=128, ge=1)
    topk_per_strategy: int = Field(default=8, ge=1)


class SpicaPolicy(_StrictModel):
    max_rounds: int = Field(default=20, ge=1)
    parallel_evals: int = Field(default=8, ge=1)
    candidates_per_round: int | None = Field(default=None, ge=1)
    max_eval_seconds: float | None = Field(default=600.0, gt=0)
    sampler_algorithm: (
        Literal["DEFAULT", "RANDOM_SEARCH", "QUASI_RANDOM_SEARCH"] | None
    ) = None
    search_seed: int | None = Field(default=None, ge=0)
    deployment_mode: list[Literal["agg", "disagg"]] = Field(
        default_factory=lambda: ["disagg", "agg"]
    )
    search_space: dict[str, Any] = Field(default_factory=dict)


class PDProfile(_StrictModel):
    id: str
    placement: str
    timing_mode: Literal["full_prompt", "destination_missing"] = "full_prompt"
    bandwidth_gbps: float | None = Field(default=None, gt=0)
    bandwidth_prefill_to_decode_gbps: float | None = Field(default=None, gt=0)
    bandwidth_decode_to_prefill_gbps: float | None = Field(default=None, gt=0)
    bandwidth_source: Literal["topology", "compute_nic", "local_fabric", "explicit"] = (
        "compute_nic"
    )
    deployability: Literal["deployable", "hypothetical"] = "deployable"

    @model_validator(mode="after")
    def _explicit_bandwidth_is_resolved(self) -> PDProfile:
        if (
            self.bandwidth_source == "explicit"
            and self.bandwidth_gbps is None
            and self.bandwidth_prefill_to_decode_gbps is None
        ):
            raise ValueError(
                "an explicit PD profile needs bandwidth_gbps or bandwidth_prefill_to_decode_gbps"
            )
        if self.bandwidth_source == "topology" and any(
            value is not None
            for value in (
                self.bandwidth_gbps,
                self.bandwidth_prefill_to_decode_gbps,
                self.bandwidth_decode_to_prefill_gbps,
            )
        ):
            raise ValueError(
                "a topology-resolved PD profile cannot also set an explicit bandwidth"
            )
        if self.bandwidth_source == "topology" and self.placement != "compact":
            raise ValueError(
                "a topology-resolved PD profile currently requires placement='compact'"
            )
        return self


class KVBMProfile(_StrictModel):
    id: str
    num_g2_blocks: int = Field(default=0, ge=0)
    capacity_gib: float | None = Field(default=None, ge=0)
    capacity_scope: Literal["deployment_total", "per_rank"] = "deployment_total"
    target_hit_rate: float | None = Field(default=None, ge=0, le=1)
    bandwidth_to_g2_ratio: float | None = Field(default=None, gt=0)
    bandwidth_from_g2_ratio: float | None = Field(default=None, gt=0)
    bandwidth_to_g2_gbps: float | None = Field(default=None, gt=0)
    bandwidth_from_g2_gbps: float | None = Field(default=None, gt=0)
    offload_batch_size: int | None = Field(default=None, gt=0)
    deployability: Literal["deployable", "hypothetical"] = "deployable"

    @model_validator(mode="after")
    def _disabled_profile_has_no_transfer(self) -> KVBMProfile:
        if (
            self.bandwidth_to_g2_ratio is not None
            and self.bandwidth_to_g2_gbps is not None
        ) or (
            self.bandwidth_from_g2_ratio is not None
            and self.bandwidth_from_g2_gbps is not None
        ):
            raise ValueError(
                "KVBM bandwidth must use either an absolute GB/s value or a compute-NIC ratio per direction"
            )
        capacity_enabled = self.capacity_gib is not None and self.capacity_gib > 0
        if (
            self.num_g2_blocks == 0
            and not capacity_enabled
            and any(
                value is not None
                for value in (
                    self.bandwidth_to_g2_ratio,
                    self.bandwidth_from_g2_ratio,
                    self.bandwidth_to_g2_gbps,
                    self.bandwidth_from_g2_gbps,
                    self.offload_batch_size,
                )
            )
        ):
            raise ValueError("a disabled G2 profile cannot set transfer parameters")
        return self


class RobustnessPolicy(_StrictModel):
    enabled: bool = False
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    bandwidth_multipliers: list[float] = Field(default_factory=lambda: [0.8, 1.0, 1.2])


class CapacitySearchPolicy(_StrictModel):
    enabled: bool = True
    stage: Literal["per_substudy", "finalists"] = "per_substudy"
    expansion_factor: float = Field(default=1.5, gt=1)
    tolerance: float = Field(default=0.03, gt=0, lt=1)
    max_expansions: int = Field(default=8, ge=1)
    max_binary_steps: int = Field(default=12, ge=1)
    min_load: float = Field(default=0.01, gt=0)
    max_load: float | None = Field(default=None, gt=0)
    require_failed_upper: bool = False
    finalists_per_load: int = Field(default=4, ge=1)
    max_finalists_per_cell: int = Field(default=32, ge=1)

    @model_validator(mode="after")
    def _ordered_load_bounds(self) -> CapacitySearchPolicy:
        if self.max_load is not None and self.min_load >= self.max_load:
            raise ValueError("capacity_search.min_load must be below max_load")
        return self


class SearchPolicyConfig(_StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    epsilon_pareto: float = Field(default=0.03, ge=0, lt=1)
    static_prune: StaticPrunePolicy = Field(default_factory=StaticPrunePolicy)
    spica: SpicaPolicy = Field(default_factory=SpicaPolicy)
    pd_profiles: list[PDProfile] = Field(
        default_factory=lambda: [
            PDProfile(
                id="topology_aware",
                placement="compact",
                bandwidth_source="topology",
            )
        ]
    )
    kvbm_profiles: list[KVBMProfile] = Field(
        default_factory=lambda: [KVBMProfile(id="g2_off", num_g2_blocks=0)]
    )
    robustness: RobustnessPolicy = Field(default_factory=RobustnessPolicy)
    capacity_search: CapacitySearchPolicy = Field(default_factory=CapacitySearchPolicy)
    kv_footprint: KVFootprintPolicy = Field(default_factory=KVFootprintPolicy)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_scalar_g2(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "kvbm_profiles" in value:
            return value
        spica = value.get("spica")
        if not isinstance(spica, dict):
            return value
        search_space = spica.get("search_space")
        if not isinstance(search_space, dict):
            return value
        scalar_keys = {
            "num_g2_blocks",
            "bandwidth_g1_to_g2_gbps",
            "bandwidth_g2_to_g1_gbps",
            "offload_batch_size",
        }
        if not scalar_keys.intersection(search_space):
            return value
        normalized = dict(value)
        normalized["kvbm_profiles"] = [
            {
                "id": "legacy_scalar",
                "num_g2_blocks": int(search_space.get("num_g2_blocks", 0)),
                "bandwidth_to_g2_gbps": search_space.get("bandwidth_g1_to_g2_gbps"),
                "bandwidth_from_g2_gbps": search_space.get("bandwidth_g2_to_g1_gbps"),
                "offload_batch_size": search_space.get("offload_batch_size"),
            }
        ]
        return normalized

    @model_validator(mode="after")
    def _unique_profile_ids(self) -> SearchPolicyConfig:
        for label, profiles in (
            ("pd_profiles", self.pd_profiles),
            ("kvbm_profiles", self.kvbm_profiles),
        ):
            ids = [profile.id for profile in profiles]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{label} contains duplicate ids")
        return self


class InputArtifact(_StrictModel):
    role: str
    path: str
    sha256: str


class CatalogBundle(_StrictModel):
    models: dict[str, ModelSpec]
    model_variants: dict[str, ModelVariantSpec]
    skus: dict[str, SkuSpec]
    workloads: dict[str, WorkloadSpec]
    slas: dict[str, SLAProfile]
    search_policy: SearchPolicyConfig
    artifacts: list[InputArtifact]


class ValidationIssue(_StrictModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    item_id: str | None = None


class ValidationReport(_StrictModel):
    valid: bool
    issues: list[ValidationIssue]
    cell_count: int = 0
