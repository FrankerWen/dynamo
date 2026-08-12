# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from aisimulate.matrix import (
    MatrixExperimentConfig,
    PriceSpec,
    ResultStore,
    expand_matrix,
    generate_report,
    load_catalogs,
)
from aisimulate.matrix._utils import fingerprint
from aisimulate.matrix.config import ModelVariantSpec, Selector
from aisimulate.matrix.expand import resolve_matrix_variants
from aisimulate.matrix.variant import (
    Operator,
    VariantResolution,
    VariantSupportResult,
    is_non_degrading_mode,
)

EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[3] / "examples" / "aisimulate" / "matrix"
)


class FakeProbe:
    def __init__(
        self,
        statuses: Mapping[tuple[str, str], tuple[str, str, list[Operator]]],
        *,
        modes: Mapping[Operator, list[str]] | None = None,
    ) -> None:
        self.statuses = dict(statuses)
        self.modes = dict(modes or {})
        self.probe_calls: list[tuple[str, str]] = []
        self.available_calls: list[Operator] = []

    def cache_key(self, variant, sku, policy) -> str:
        return fingerprint(
            {
                "variant": variant.model_path,
                "quantization": variant.quantization,
                "kernel_family": variant.kernel_family,
                "sku": sku.aic_system,
                "backend": variant.backend,
                "data_mode": policy.data_mode,
            }
        )

    def probe(self, variant, sku, policy) -> VariantSupportResult:
        self.probe_calls.append((variant.id, sku.id))
        status, reason, failed = self.statuses.get(
            (variant.id, sku.id), ("supported_exact", "supported", [])
        )
        return VariantSupportResult(
            status=status,
            reason=reason,
            failed_operators=failed,
            backend_version="test-version",
            data_source={"kind": "fake"},
            cache_key=self.cache_key(variant, sku, policy),
        )

    def available_modes(self, variant, sku, operator, policy) -> list[str]:
        del variant, sku, policy
        self.available_calls.append(operator)
        return list(self.modes.get(operator, []))


def _base():
    config = MatrixExperimentConfig.from_file(EXAMPLE_ROOT / "experiment.yaml")
    catalogs = load_catalogs(config)
    catalogs = catalogs.model_copy(
        update={
            "model_variants": {
                variant_id: variant
                for variant_id, variant in catalogs.model_variants.items()
                if variant.role == "native"
            }
        }
    )
    return config, catalogs


def _config(
    config: MatrixExperimentConfig,
    *,
    model_id: str = "deepseek_v4_pro",
    sku_ids: list[str] | None = None,
    variant_selector: Selector | None = None,
    data_mode: str = "silicon",
    max_auto: int = 8,
) -> MatrixExperimentConfig:
    matrix = config.matrix.model_copy(
        update={
            "model_selector": Selector(ids=[model_id]),
            "sku_selector": Selector(ids=sku_ids or ["gb300"]),
            "workload_selector": Selector(ids=["synthetic_4k"]),
            "model_variant_selector": variant_selector,
            "variant_resolution": config.matrix.variant_resolution.model_copy(
                update={
                    "data_mode": data_mode,
                    "max_auto_variants_per_model_sku": max_auto,
                }
            ),
        }
    )
    return config.model_copy(update={"matrix": matrix})


def _fallback(
    native: ModelVariantSpec,
    variant_id: str,
    *,
    moe: str,
) -> ModelVariantSpec:
    return ModelVariantSpec.model_validate(
        {
            **native.model_dump(mode="python"),
            "id": variant_id,
            "role": "fallback",
            "kernel_family": "test_kernel",
            "weight_precision": f"fallback-{moe}",
            "quantization": {
                **native.quantization.model_dump(mode="python"),
                "moe": moe,
            },
            "fallback_for": ["moe"],
        }
    )


def _with_variants(catalogs, *variants: ModelVariantSpec):
    merged = dict(catalogs.model_variants)
    merged.update({variant.id: variant for variant in variants})
    return catalogs.model_copy(update={"model_variants": merged})


def test_explicit_variants_are_exact_and_unsupported_does_not_fallback():
    base_config, catalogs = _base()
    native = catalogs.model_variants["deepseek_v4_pro_native_vllm"]
    fallback = _fallback(native, "deepseek_v4_pro_w4a16", moe="w4a16_mxfp4_cutlass")
    catalogs = _with_variants(catalogs, fallback)
    config = _config(
        base_config,
        variant_selector=Selector(ids=[native.id, fallback.id]),
    )
    probe = FakeProbe(
        {(fallback.id, "gb300"): ("kernel_quant_unsupported", "no kernel", ["moe"])}
    )

    resolution = resolve_matrix_variants(config, catalogs, support_probe=probe)

    assert {record.variant.id for record in resolution.selected_records} == {
        native.id,
        fallback.id,
    }
    assert all(record.source == "explicit" for record in resolution.records)
    assert probe.available_calls == []


def test_native_supported_short_circuits_all_fallbacks():
    base_config, catalogs = _base()
    native = catalogs.model_variants["deepseek_v4_pro_native_vllm"]
    fallback = _fallback(native, "deepseek_v4_pro_w4a16", moe="w4a16_mxfp4_cutlass")
    catalogs = _with_variants(catalogs, fallback)
    config = _config(base_config)
    probe = FakeProbe({})

    resolution = resolve_matrix_variants(config, catalogs, support_probe=probe)

    assert [record.variant.id for record in resolution.selected_records] == [native.id]
    assert probe.probe_calls == [(native.id, "gb300")]
    assert probe.available_calls == []


def test_native_precision_failure_runs_every_catalog_recipe_without_enumeration():
    base_config, catalogs = _base()
    native = catalogs.model_variants["deepseek_v4_pro_native_vllm"]
    w4a16 = _fallback(native, "deepseek_v4_pro_w4a16", moe="w4a16_mxfp4_cutlass")
    w8a8 = _fallback(native, "deepseek_v4_pro_w8a8", moe="fp8")
    catalogs = _with_variants(catalogs, w4a16, w8a8)
    config = _config(base_config)
    probe = FakeProbe(
        {(native.id, "gb300"): ("kernel_quant_unsupported", "native kernel", ["moe"])}
    )

    resolution = resolve_matrix_variants(config, catalogs, support_probe=probe)

    assert {record.variant.id for record in resolution.selected_records} == {
        w4a16.id,
        w8a8.id,
    }
    assert probe.available_calls == []
    assert all(
        record.source == "catalog_fallback" for record in resolution.selected_records
    )


def test_aic_enumeration_changes_only_failed_operator_and_never_lowers_precision():
    base_config, catalogs = _base()
    native = catalogs.model_variants["deepseek_v4_pro_native_vllm"]
    config = _config(base_config)
    probe = FakeProbe(
        {(native.id, "gb300"): ("hardware_precision_unsupported", "no FP4", ["moe"])},
        modes={
            "moe": [
                "nvfp4",
                "w4a16_mxfp4_cutlass",
                "fp8",
                "bfloat16",
            ]
        },
    )

    resolution = resolve_matrix_variants(config, catalogs, support_probe=probe)
    selected = resolution.selected_records

    assert len(selected) == 3
    assert all(record.source == "aic_enumerated" for record in selected)
    assert all(
        record.variant.deployment_fidelity == "hypothetical" for record in selected
    )
    for record in selected:
        assert record.variant.quantization.gemm == native.quantization.gemm
        assert record.variant.quantization.fmha == native.quantization.fmha
        assert is_non_degrading_mode(
            record.variant.quantization.moe, native.quantization.moe
        )
    assert "nvfp4" not in {record.variant.quantization.moe for record in selected}


@pytest.mark.parametrize(
    "status",
    ["missing_perf_data", "checkpoint_incompatible", "invalid_variant"],
)
def test_non_precision_failures_do_not_trigger_fallback(status):
    base_config, catalogs = _base()
    native = catalogs.model_variants["deepseek_v4_pro_native_vllm"]
    config = _config(base_config)
    probe = FakeProbe({(native.id, "gb300"): (status, "not precision", [])})

    resolution = resolve_matrix_variants(config, catalogs, support_probe=probe)

    assert [record.variant.id for record in resolution.selected_records] == [native.id]
    assert probe.available_calls == []


def test_silicon_rejects_empirical_while_empirical_mode_accepts_it():
    base_config, catalogs = _base()
    native = catalogs.model_variants["deepseek_v4_pro_native_vllm"]
    probe = FakeProbe({(native.id, "gb300"): ("supported_empirical", "hybrid", [])})

    silicon = resolve_matrix_variants(
        _config(base_config, data_mode="silicon"), catalogs, support_probe=probe
    )
    empirical = resolve_matrix_variants(
        _config(base_config, data_mode="empirical"), catalogs, support_probe=probe
    )

    assert silicon.selected_records[0].support.status == "supported_empirical"
    assert empirical.selected_records[0].support.status == "supported_empirical"
    # Both retain a Cell for audit; only the empirical one is runnable.
    assert silicon.selected_records[0].selection_reason.startswith("native failure")
    assert empirical.selected_records[0].selection_reason.startswith(
        "native variant supported"
    )


def test_h100_blackwell_native_rejection_can_select_w4a16_recipe():
    base_config, catalogs = _base()
    native = catalogs.model_variants["deepseek_v4_pro_native_vllm"]
    fallback = _fallback(native, "deepseek_v4_pro_w4a16", moe="w4a16_mxfp4_cutlass")
    h100 = catalogs.skus["gb300"].model_copy(
        update={"id": "h100_sxm", "aic_system": "h100_sxm"}
    )
    catalogs = _with_variants(catalogs, fallback).model_copy(
        update={"skus": {**catalogs.skus, h100.id: h100}}
    )
    config = _config(base_config, sku_ids=[h100.id])
    probe = FakeProbe(
        {
            (native.id, h100.id): (
                "kernel_quant_unsupported",
                "Blackwell FP4 kernel",
                ["moe"],
            )
        }
    )

    resolution = resolve_matrix_variants(config, catalogs, support_probe=probe)

    assert [record.variant.id for record in resolution.selected_records] == [
        fallback.id
    ]
    assert (
        resolution.selected_records[0].variant.quantization.moe == "w4a16_mxfp4_cutlass"
    )


def test_auto_enumeration_over_limit_fails_closed():
    base_config, catalogs = _base()
    native = catalogs.model_variants["deepseek_v4_pro_native_vllm"].model_copy(
        update={
            "quantization": catalogs.model_variants[
                "deepseek_v4_pro_native_vllm"
            ].quantization.model_copy(update={"gemm": "nvfp4", "moe": "nvfp4"})
        }
    )
    variants = dict(catalogs.model_variants)
    variants[native.id] = native
    catalogs = catalogs.model_copy(update={"model_variants": variants})
    config = _config(base_config, max_auto=8)
    probe = FakeProbe(
        {
            (native.id, "gb300"): (
                "hardware_precision_unsupported",
                "no FP4",
                ["gemm", "moe"],
            )
        },
        modes={
            "gemm": ["int4_wo", "fp8", "bfloat16"],
            "moe": ["w4a8_mxfp4_mxfp8", "w4a16_mxfp4", "fp8", "bfloat16"],
        },
    )

    with pytest.raises(ValueError, match="exceeding max_auto_variants"):
        resolve_matrix_variants(config, catalogs, support_probe=probe)


def test_resolver_source_does_not_change_cell_id():
    base_config, catalogs = _base()
    config = _config(
        base_config,
        variant_selector=Selector(ids=["deepseek_v4_pro_native_vllm"]),
    )
    resolution = resolve_matrix_variants(config, catalogs, support_probe=FakeProbe({}))
    explicit_cells = expand_matrix(config, catalogs, resolution=resolution)
    native_resolution = VariantResolution(
        records=[
            record.model_copy(update={"source": "native"})
            for record in resolution.records
        ]
    )
    native_cells = expand_matrix(config, catalogs, resolution=native_resolution)

    assert [cell.id for cell in explicit_cells] == [cell.id for cell in native_cells]


def test_variant_support_cache_round_trip_skips_probe(tmp_path):
    base_config, catalogs = _base()
    config = _config(base_config)
    probe = FakeProbe({})
    first = resolve_matrix_variants(config, catalogs, support_probe=probe)
    source_dir = tmp_path / "primary"
    with ResultStore.open(source_dir) as source:
        experiment_id = source.start_experiment(config, catalogs)
        for record in first.records:
            source.record_variant_resolution(experiment_id, record)
        assert source.summary()["variant_support_cache_entries"] == 1

    with ResultStore.open(tmp_path / "formal") as target:
        imported = target.import_variant_support_cache([source_dir])
        assert imported["imported"] == 1
        cached = target.variant_support_cache_snapshot()
        second_probe = FakeProbe({})
        second = resolve_matrix_variants(
            config,
            catalogs,
            support_probe=second_probe,
            support_cache=cached,
        )
        assert second_probe.probe_calls == []
        assert second.records[0].support.cache_hit
        assert target.summary()["variant_support_cache_entries"] == 1


def test_default_reports_exclude_hypothetical_and_emit_what_if(tmp_path):
    base_config, catalogs = _base()
    config = _config(
        base_config,
        variant_selector=Selector(ids=["deepseek_v4_pro_native_vllm"]),
    )
    resolution = resolve_matrix_variants(config, catalogs, support_probe=FakeProbe({}))
    deployable = expand_matrix(config, catalogs, resolution=resolution)[0]
    hypothetical_variant = deployable.model_variant.model_copy(
        update={"id": "deepseek_v4_pro_what_if", "deployment_fidelity": "hypothetical"}
    )
    hypothetical_resolution = deployable.variant_resolution.model_copy(
        update={
            "id": "variant_resolution_what_if",
            "variant": hypothetical_variant,
            "source": "aic_enumerated",
        }
    )
    hypothetical = deployable.model_copy(
        update={
            "id": "cell_what_if",
            "fingerprint": "what-if-fingerprint",
            "model_variant": hypothetical_variant,
            "variant_resolution": hypothetical_resolution,
        }
    )

    with ResultStore.open(tmp_path / "report") as store:
        experiment_id = store.start_experiment(config, catalogs)
        store.install_prices(
            [PriceSpec(id="test", sku_id=deployable.sku.id, gpu_hour=1.0)]
        )
        for cell in (deployable, hypothetical):
            store.record_cell(experiment_id, cell, status="complete")
            candidate_id = store.record_evaluation(
                cell_id=cell.id,
                evaluation_key="final",
                stage="spica",
                outcome="feasible",
                reason="",
                selection={},
                metrics={"mean_output_token_throughput_per_user": 40.0},
                sla_pass=True,
                candidate={
                    "config": {
                        "used_gpus": 2,
                        "deployment_mode": "agg",
                        "kvbm_profile_id": "g2_off",
                    },
                    "used_gpus": 2,
                    "score": 1.0,
                    "metrics": {},
                    "objectives": {"goodput_per_gpu": 1.0},
                },
                resources={
                    "gpu_hours": 1.0,
                    "input_tokens": 500_000,
                    "output_tokens": 500_000,
                    "goodput_input_tokens": 500_000,
                    "goodput_output_tokens": 500_000,
                    "duration_ms": 3_600_000,
                },
            )
            assert candidate_id is not None
            store.mark_cell_frontiers(cell.id, [candidate_id], epsilon=0.03)
        generate_report(store)
        assert store.connection.execute(
            "SELECT count(*) FROM deployable_candidate_costs"
        ).fetchone() == (1,)
        assert store.connection.execute(
            "SELECT count(*) FROM what_if_candidate_costs"
        ).fetchone() == (1,)
        assert store.connection.execute(
            "SELECT count(*) FROM candidate_dimensions"
        ).fetchone() == (1,)

    assert len((tmp_path / "report" / "pareto_front.csv").read_text().splitlines()) == 2
    assert (
        len((tmp_path / "report" / "what_if_pareto_front.csv").read_text().splitlines())
        == 2
    )
