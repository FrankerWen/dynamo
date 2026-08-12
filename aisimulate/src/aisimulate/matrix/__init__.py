# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable catalog-driven matrix search for DynoSim + AIConfigurator."""

from aisimulate.spica.config import KVFootprintPolicy, KVFootprintSpec

from .catalog import load_catalogs
from .config import (
    AICQuantizationSpec,
    CatalogBundle,
    MatrixExperimentConfig,
    ModelSpec,
    ModelVariantSpec,
    PriceSpec,
    SearchPolicyConfig,
    SkuSpec,
    SLAProfile,
    ValidationReport,
    VariantResolutionPolicy,
    WorkloadSpec,
)
from .expand import (
    MatrixCell,
    expand_matrix,
    resolve_matrix_variants,
    validate_experiment,
)
from .operating_points import (
    AICOperatingPoint,
    AICOperatingPointResolution,
    resolve_aic_operating_points,
)
from .report import generate_report
from .runner import MatrixRunner
from .store import ResultStore
from .variant import (
    AICVariantSupportProbe,
    VariantResolution,
    VariantResolutionRecord,
    VariantSupportProbe,
    VariantSupportResult,
    is_non_degrading_mode,
    precision_semantics,
    resolve_variants,
)
from .workload import (
    PreparedWorkload,
    WorkloadLoader,
    register_workload_loader,
)

__all__ = [
    "AICQuantizationSpec",
    "AICOperatingPoint",
    "AICOperatingPointResolution",
    "AICVariantSupportProbe",
    "CatalogBundle",
    "KVFootprintPolicy",
    "KVFootprintSpec",
    "MatrixCell",
    "MatrixExperimentConfig",
    "MatrixRunner",
    "ModelSpec",
    "ModelVariantSpec",
    "PreparedWorkload",
    "PriceSpec",
    "ResultStore",
    "SLAProfile",
    "SearchPolicyConfig",
    "SkuSpec",
    "ValidationReport",
    "VariantResolution",
    "VariantResolutionPolicy",
    "VariantResolutionRecord",
    "VariantSupportProbe",
    "VariantSupportResult",
    "WorkloadLoader",
    "WorkloadSpec",
    "expand_matrix",
    "generate_report",
    "is_non_degrading_mode",
    "load_catalogs",
    "precision_semantics",
    "register_workload_loader",
    "resolve_matrix_variants",
    "resolve_aic_operating_points",
    "resolve_variants",
    "validate_experiment",
]
