# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from aisimulate.matrix import (
    MatrixExperimentConfig,
    MatrixRunner,
    ModelSpec,
    PriceSpec,
    ResultStore,
    SearchPolicyConfig,
    SkuSpec,
    expand_matrix,
    generate_report,
    load_catalogs,
    validate_experiment,
)
from aisimulate.matrix.prune import StaticPruneResult
from aisimulate.matrix.operating_points import _compress
from aisimulate.matrix.replay_cache import (
    CachedReplayResult,
    ReplayCacheDescription,
)
from aisimulate.matrix.workload import (
    LoaderContext,
    WorkloadSpec,
    materialize_workload,
    prepare_workload,
)
from aisimulate.spica import Candidate, SearchEvaluation, SmartSearchRunResult

EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[3] / "examples" / "aisimulate" / "matrix"
)


def _copy_example(tmp_path: Path) -> Path:
    target = tmp_path / "matrix"
    shutil.copytree(EXAMPLE_ROOT, target)
    return target / "experiment.yaml"


def test_example_catalogs_expand_deterministically():
    config = MatrixExperimentConfig.from_file(EXAMPLE_ROOT / "experiment.yaml")
    catalogs = load_catalogs(config)
    report = validate_experiment(config)
    first = expand_matrix(config, catalogs)
    second = expand_matrix(config, catalogs)

    assert report.valid
    assert report.cell_count == 20
    assert [cell.id for cell in first] == [cell.id for cell in second]
    assert len({cell.fingerprint for cell in first}) == 20
    assert not hasattr(config.catalogs, "prices")
    assert all(not hasattr(sku, "max_gpus") for sku in catalogs.skus.values())
    assert all(not hasattr(sku, "price_ref") for sku in catalogs.skus.values())


def test_sku_bandwidth_domains_make_gbytes_per_second_explicit():
    config = MatrixExperimentConfig.from_file(EXAMPLE_ROOT / "experiment.yaml")
    sku = load_catalogs(config).skus["gb300"]

    assert sku.compute_nic_gbytes_per_s_per_gpu == 100
    assert sku.pd_transfer_topology.gpus_per_node == 4
    assert sku.pd_transfer_topology.gpus_per_rack == 72
    assert sku.pd_transfer_topology.intra_rack_gbytes_per_s_per_gpu == 900
    assert sku.pd_transfer_topology.inter_rack_gbytes_per_s_per_gpu == 100
    serialized = sku.model_dump(mode="python")
    assert serialized["compute_nic_gbytes_per_s_per_gpu"] == 100
    assert serialized["pd_transfer_topology"]["gpus_per_rack"] == 72
    assert "compute_nic_gbps_per_gpu" not in serialized
    assert "local_fabric_gbps_per_gpu" not in serialized

    legacy = SkuSpec.model_validate(
        {
            "id": "legacy",
            "aic_system": "legacy",
            "compute_nic_gbps_per_gpu": 25,
            "local_fabric_gbps_per_gpu": 450,
            "gpus_per_node": 8,
        }
    )
    assert legacy.compute_nic_gbytes_per_s_per_gpu == 25
    assert legacy.pd_transfer_topology.gpus_per_node == 8
    assert legacy.pd_transfer_topology.intra_node_gbytes_per_s_per_gpu == 450
    assert legacy.pd_transfer_topology.inter_node_gbytes_per_s_per_gpu == 25
    assert set(legacy.model_dump()) >= {
        "compute_nic_gbytes_per_s_per_gpu",
        "pd_transfer_topology",
    }


def test_changing_rack_topology_only_invalidates_that_sku_cells(tmp_path):
    config_path = _copy_example(tmp_path)
    before = expand_matrix(MatrixExperimentConfig.from_file(config_path))
    sku_path = config_path.parent / "catalogs" / "skus.yaml"
    payload = yaml.safe_load(sku_path.read_text(encoding="utf-8"))
    gb300 = next(sku for sku in payload["skus"] if sku["id"] == "gb300")
    gb300["pd_transfer_topology"]["gpus_per_rack"] = 64
    sku_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    after = expand_matrix(MatrixExperimentConfig.from_file(config_path))

    def ids(cells, sku_id):
        return {
            (cell.model.id, cell.workload.id): cell.id
            for cell in cells
            if cell.sku.id == sku_id
        }

    assert ids(before, "b300_sxm") == ids(after, "b300_sxm")
    assert ids(before, "gb300").keys() == ids(after, "gb300").keys()
    assert all(
        old != ids(after, "gb300")[key] for key, old in ids(before, "gb300").items()
    )


def test_formal_and_preliminary_gpu_limits_live_in_policies():
    formal = load_catalogs(
        MatrixExperimentConfig.from_file(EXAMPLE_ROOT / "experiment.yaml")
    ).search_policy
    preliminary = load_catalogs(
        MatrixExperimentConfig.from_file(EXAMPLE_ROOT / "experiment_preliminary.yaml")
    ).search_policy

    assert formal.spica.search_space["gpu_budget"] == 72
    assert preliminary.spica.search_space["gpu_budget"] == 16
    assert formal.spica.sampler_algorithm == "QUASI_RANDOM_SEARCH"
    assert formal.spica.search_seed == 20260810


def test_full_parallel_variant_is_isolated_and_preserves_trial_budget():
    config = MatrixExperimentConfig.from_file(EXAMPLE_ROOT / "experiment_parallel.yaml")
    policy = load_catalogs(config).search_policy

    assert config.run.cell_workers == 4
    assert policy.spica.parallel_evals == 16
    assert policy.spica.candidates_per_round == 16
    assert policy.spica.max_rounds == 10
    assert policy.spica.max_rounds * policy.spica.candidates_per_round == 160
    assert len(expand_matrix(config)) == 20


def test_parallel_focus_keeps_sample_budget_and_selects_one_cell():
    config = MatrixExperimentConfig.from_file(
        EXAMPLE_ROOT / "experiment_parallel_focus.yaml"
    )
    catalogs = load_catalogs(config)
    cells = expand_matrix(config, catalogs)

    assert len(cells) == 1
    # Topology-aware P/D placement is replay semantics and therefore forks the Cell.
    assert cells[0].id == "cell_e106fb9d9dc2e12b4af8"
    assert cells[0].model.id == "deepseek_v4_pro"
    assert cells[0].model_variant.id == "deepseek_v4_pro_native_vllm"
    assert cells[0].sku.id == "gb300"
    assert cells[0].workload.summary == {
        "isl": 8192,
        "osl": 1024,
        "prefix_length": 0,
        "context_length": 9216,
    }
    assert cells[0].workload.load_axis.source == "aic_operating_points"
    assert cells[0].workload.load_axis.name == "concurrency"
    assert cells[0].workload.load_axis.values == []
    assert cells[0].workload.load_axis.max_points == 8
    assert config.run.cell_workers == 1
    assert catalogs.search_policy.spica.max_rounds == 10
    assert catalogs.search_policy.spica.parallel_evals == 16
    assert catalogs.search_policy.spica.candidates_per_round == 16
    assert (
        catalogs.search_policy.spica.max_rounds
        * catalogs.search_policy.spica.candidates_per_round
        == 160
    )
    assert catalogs.search_policy.capacity_search.stage == "finalists"


def test_synthetic_aic_concurrency_sweep_rejects_manual_load_mix():
    spec = WorkloadSpec(
        id="aic-concurrency",
        loader="synthetic",
        config={
            "isl": 8192,
            "osl": 1024,
            "request_rate": 1,
            "concurrency_sweep": {
                "source": "aic_operating_points",
                "max_points": 8,
            },
        },
    )
    with pytest.raises(ValueError, match="exactly one"):
        prepare_workload(spec, LoaderContext(base_dir=EXAMPLE_ROOT))


def test_finalist_selection_is_load_conditioned_then_deduplicated():
    config = MatrixExperimentConfig.from_file(
        EXAMPLE_ROOT / "experiment_parallel_focus.yaml"
    )
    cell = expand_matrix(config)[0]

    def candidate(concurrency: int, goodput: float, tpsu: float, router: str):
        return Candidate(
            config={
                "deployment_mode": "agg",
                "concurrency": concurrency,
                "router_mode": router,
            },
            used_gpus=8,
            score=goodput,
            metrics={},
            objectives={
                "goodput_per_gpu": goodput,
                "throughput_per_user": tpsu,
            },
        )

    candidates = [
        candidate(4, 10, 50, "round_robin"),
        candidate(4, 8, 60, "kv_router"),
        candidate(8, 20, 40, "round_robin"),
        candidate(8, 15, 55, "kv_router"),
    ]
    rows = [
        (f"cand-{index}", item.model_dump(mode="python"))
        for index, item in enumerate(candidates)
    ]
    selected = MatrixRunner()._select_finalists(cell, rows)

    # The same two concrete configurations appear at both loads, so finalist
    # qualification replays only the best representative of each config.
    assert len(selected) == 2
    assert {item.config["router_mode"] for item in selected} == {
        "round_robin",
        "kv_router",
    }


def test_aic_operating_point_compression_preserves_both_boundaries():
    values = [4, 8, 12, 20, 36, 64, 108, 192, 320, 640]
    selected = _compress(values, 8)

    assert len(selected) == 8
    assert selected[0] == 4
    assert selected[-1] == 640
    assert selected == sorted(set(selected))


def test_replay_worker_import_does_not_eagerly_load_duckdb():
    """Spawned Spica workers must not create DuckDB's host-sized thread pool."""

    env = dict(os.environ)
    source_root = str(EXAMPLE_ROOT.parents[2] / "aisimulate" / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import aisimulate.matrix.store; "
                "print('duckdb' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.stdout.strip() == "False"


def test_adding_model_and_variant_catalog_entries_needs_no_code_change(tmp_path):
    config_path = _copy_example(tmp_path)
    models_path = config_path.parent / "catalogs" / "models.yaml"
    payload = models_path.read_text(encoding="utf-8")
    payload += (
        "\n  - id: future_model\n"
        "    kv_bytes_per_token: 65536\n"
        "    tags: [formal-or-experimental]\n"
    )
    models_path.write_text(payload, encoding="utf-8")
    variants_path = config_path.parent / "catalogs" / "model_variants.yaml"
    variant_payload = yaml.safe_load(variants_path.read_text(encoding="utf-8"))
    variant_payload["model_variants"].append(
        {
            "id": "future_model_native_vllm",
            "model_id": "future_model",
            "model_path": "example/Future-Model",
            "weight_precision": "bfloat16",
            "quantization": {
                "gemm": "bfloat16",
                "moe": "bfloat16",
                "fmha": "bfloat16",
                "kv_cache": "bfloat16",
                "comm": "half",
            },
            "support_tier": "experimental",
            "tags": ["native"],
        }
    )
    variants_path.write_text(
        yaml.safe_dump(variant_payload, sort_keys=False), encoding="utf-8"
    )

    config = MatrixExperimentConfig.from_file(config_path)
    cells = expand_matrix(config)

    assert len(cells) == 24
    assert any(cell.model.id == "future_model" for cell in cells)
    assert any(cell.model_variant.id == "future_model_native_vllm" for cell in cells)


def test_model_catalog_uses_structured_aic_native_kv_contract():
    config = MatrixExperimentConfig.from_file(EXAMPLE_ROOT / "experiment.yaml")
    models = load_catalogs(config).models

    assert models["deepseek_v4_flash"].kv_footprint.source == "aic_native"
    assert models["deepseek_v4_flash"].kv_footprint.layout == "compressed_sparse_mla"
    assert models["kimi_k3"].kv_footprint.layout == "hybrid_mla_kda_fixed_state"
    assert all(model.kv_footprint.scope == "per_rank" for model in models.values())


def test_model_variants_pin_complete_aic_quantization_profiles():
    config = MatrixExperimentConfig.from_file(EXAMPLE_ROOT / "experiment.yaml")
    variants = load_catalogs(config).model_variants

    deepseek = variants["deepseek_v4_pro_native_vllm"]
    assert deepseek.model_id == "deepseek_v4_pro"
    assert deepseek.quantization.model_dump() == {
        "gemm": "fp8_block",
        "moe": "w4a8_mxfp4_mxfp8",
        "fmha": "bfloat16",
        "kv_cache": "fp8",
        "comm": "half",
    }
    assert variants["glm_5_2_native_vllm"].weight_precision == "nvfp4"


def test_matrix_search_config_pins_variant_into_every_aic_path():
    config = MatrixExperimentConfig.from_file(
        EXAMPLE_ROOT / "experiment_parallel_focus.yaml"
    )
    cell = expand_matrix(config)[0]
    kvbm_profile = cell.search_policy.kvbm_profiles[0]

    smart = MatrixRunner()._smart_search_config(
        cell,
        cell.workload,
        "agg",
        None,
        kvbm_profile,
        "concurrency",
        4,
    )

    search = smart.search_space
    assert search.model_id == "deepseek_v4_pro"
    assert search.model_variant_id == "deepseek_v4_pro_native_vllm"
    assert search.model_name == "deepseek-ai/DeepSeek-V4-Pro"
    assert search.aic_gemm_dtype == "fp8_block"
    assert search.aic_moe_dtype == "w4a8_mxfp4_mxfp8"
    assert search.aic_fmha_dtype == "bfloat16"
    assert search.aic_kv_cache_dtype == "fp8"
    assert search.aic_comm_dtype == "half"
    assert search.kv_footprint.dtype == "fp8"


def test_changing_one_variant_only_invalidates_its_cells(tmp_path):
    config_path = _copy_example(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "  model_selector: {tags: [formal-or-experimental]}\n",
            "  model_selector: {tags: [formal-or-experimental]}\n"
            "  model_variant_selector: {tags: [native]}\n",
        ),
        encoding="utf-8",
    )
    before = expand_matrix(MatrixExperimentConfig.from_file(config_path))
    variants_path = config_path.parent / "catalogs" / "model_variants.yaml"
    payload = yaml.safe_load(variants_path.read_text(encoding="utf-8"))
    target = next(
        variant
        for variant in payload["model_variants"]
        if variant["id"] == "deepseek_v4_pro_native_vllm"
    )
    target["quantization"]["comm"] = "fp8"
    variants_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    after = expand_matrix(MatrixExperimentConfig.from_file(config_path))

    def ids(cells):
        return {
            (cell.model_variant.id, cell.sku.id, cell.workload.id): cell.id
            for cell in cells
        }

    before_ids = ids(before)
    after_ids = ids(after)
    changed = [key for key in before_ids if before_ids[key] != after_ids[key]]
    assert changed
    assert {key[0] for key in changed} == {"deepseek_v4_pro_native_vllm"}


def test_second_variant_adds_cells_only_for_its_model(tmp_path):
    config_path = _copy_example(tmp_path)
    variants_path = config_path.parent / "catalogs" / "model_variants.yaml"
    payload = yaml.safe_load(variants_path.read_text(encoding="utf-8"))
    source = next(
        variant
        for variant in payload["model_variants"]
        if variant["id"] == "deepseek_v4_pro_native_vllm"
    )
    second = json.loads(json.dumps(source))
    second["id"] = "deepseek_v4_pro_fp8_comm_vllm"
    second["role"] = "fallback"
    second["fallback_for"] = ["comm"]
    second["quantization"]["comm"] = "fp8"
    payload["model_variants"].append(second)
    variants_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "  model_selector: {tags: [formal-or-experimental]}\n",
            "  model_selector: {tags: [formal-or-experimental]}\n"
            "  model_variant_selector: {tags: [native]}\n",
        ),
        encoding="utf-8",
    )

    cells = expand_matrix(MatrixExperimentConfig.from_file(config_path))

    assert len(cells) == 24
    new_cells = [
        cell
        for cell in cells
        if cell.model_variant.id == "deepseek_v4_pro_fp8_comm_vllm"
    ]
    assert len(new_cells) == 4
    assert {cell.model.id for cell in new_cells} == {"deepseek_v4_pro"}


def test_legacy_model_scalar_upgrades_to_explicit_per_rank_contract():
    model = ModelSpec.model_validate(
        {
            "id": "legacy",
            "model_path": "example/Legacy",
            "kv_bytes_per_token": 65_536,
        }
    )

    assert model.kv_footprint.source == "explicit_linear"
    assert model.kv_footprint.scope == "per_rank"
    assert model.kv_footprint.layout == "legacy_unspecified"
    assert model.kv_footprint.explicit_bytes_per_token == 65_536
    assert "kv_bytes_per_token" not in model.model_dump()


def test_validation_rejects_missing_kv_footprint_for_pd_or_kvbm(tmp_path):
    config_path = _copy_example(tmp_path)
    models_path = config_path.parent / "catalogs" / "models.yaml"
    payload = yaml.safe_load(models_path.read_text(encoding="utf-8"))
    payload["models"][0].pop("kv_footprint")
    models_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = validate_experiment(MatrixExperimentConfig.from_file(config_path))

    assert not report.valid
    assert any(issue.code == "missing_kv_footprint" for issue in report.issues)


def test_validation_rejects_ambiguous_legacy_scalar_for_pd_or_kvbm(tmp_path):
    config_path = _copy_example(tmp_path)
    models_path = config_path.parent / "catalogs" / "models.yaml"
    payload = yaml.safe_load(models_path.read_text(encoding="utf-8"))
    payload["models"][0].pop("kv_footprint")
    payload["models"][0]["kv_bytes_per_token"] = 88_064
    models_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = validate_experiment(MatrixExperimentConfig.from_file(config_path))

    assert not report.valid
    assert any(issue.code == "ambiguous_legacy_kv_footprint" for issue in report.issues)


def test_yaml_and_json_catalogs_merge_via_relative_paths(tmp_path):
    config_path = _copy_example(tmp_path)
    extra_path = config_path.parent / "catalogs" / "models-extra.json"
    extra_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "models",
                "models": [
                    {
                        "id": "json_model",
                        "kv_bytes_per_token": 65536,
                        "tags": ["formal-or-experimental"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    variant_path = config_path.parent / "catalogs" / "model-variants-extra.json"
    variant_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "model_variants",
                "model_variants": [
                    {
                        "id": "json_model_native_vllm",
                        "model_id": "json_model",
                        "model_path": "example/Json-Model",
                        "weight_precision": "bfloat16",
                        "quantization": {
                            "gemm": "bfloat16",
                            "moe": "bfloat16",
                            "fmha": "bfloat16",
                            "kv_cache": "bfloat16",
                            "comm": "half",
                        },
                        "support_tier": "experimental",
                        "tags": ["native"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        .replace(
            "models: [catalogs/models.yaml]",
            "models: [catalogs/models.yaml, catalogs/models-extra.json]",
        )
        .replace(
            "model_variants: [catalogs/model_variants.yaml]",
            "model_variants: [catalogs/model_variants.yaml, catalogs/model-variants-extra.json]",
        ),
        encoding="utf-8",
    )

    cells = expand_matrix(MatrixExperimentConfig.from_file(config_path))

    assert len(cells) == 24
    assert any(cell.model.id == "json_model" for cell in cells)
    assert any(cell.model_variant.id == "json_model_native_vllm" for cell in cells)


def test_duplicate_ids_fail_closed(tmp_path):
    config_path = _copy_example(tmp_path)
    config_text = config_path.read_text(encoding="utf-8").replace(
        "models: [catalogs/models.yaml]",
        "models: [catalogs/models.yaml, catalogs/models.yaml]",
    )
    config_path.write_text(config_text, encoding="utf-8")

    config = MatrixExperimentConfig.from_file(config_path)
    with pytest.raises(ValueError, match="duplicate models id"):
        load_catalogs(config)


def test_model_variant_unknown_model_reference_fails_closed(tmp_path):
    config_path = _copy_example(tmp_path)
    variants_path = config_path.parent / "catalogs" / "model_variants.yaml"
    payload = yaml.safe_load(variants_path.read_text(encoding="utf-8"))
    payload["model_variants"][0]["model_id"] = "unknown_model"
    variants_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = validate_experiment(MatrixExperimentConfig.from_file(config_path))

    assert not report.valid
    assert "references unknown model ids" in report.issues[0].message


@pytest.mark.parametrize(
    ("target", "replacement", "message"),
    [
        (
            "sku_selector: {ids: [b300_sxm, gb300]}",
            "sku_selector: {ids: [unknown_sku]}",
            "sku selector references unknown ids",
        ),
        (
            "sla_profile: default_sla",
            "sla_profile: unknown_sla",
            "unknown SLA profile",
        ),
    ],
)
def test_unknown_matrix_references_fail_validation(
    tmp_path, target, replacement, message
):
    config_path = _copy_example(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(target, replacement),
        encoding="utf-8",
    )

    report = validate_experiment(MatrixExperimentConfig.from_file(config_path))

    assert not report.valid
    assert message in report.issues[0].message


def test_catalog_schema_version_fails_validation(tmp_path):
    config_path = _copy_example(tmp_path)
    models_path = config_path.parent / "catalogs" / "models.yaml"
    models_path.write_text(
        models_path.read_text(encoding="utf-8").replace(
            "schema_version: 1", "schema_version: 2", 1
        ),
        encoding="utf-8",
    )

    report = validate_experiment(MatrixExperimentConfig.from_file(config_path))

    assert not report.valid
    assert "schema_version=2" in report.issues[0].message


def test_trace_content_changes_only_workload_fingerprint(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"timestamp": 0, "input_length": 8, "output_length": 2}\n',
        encoding="utf-8",
    )
    spec = WorkloadSpec(
        id="trace",
        loader="mooncake_trace",
        config={"path": "trace.jsonl"},
    )
    context = LoaderContext(base_dir=tmp_path)
    first = prepare_workload(spec, context)
    trace.write_text(
        '{"timestamp": 0, "input_length": 9, "output_length": 2}\n',
        encoding="utf-8",
    )
    second = prepare_workload(spec, context)

    assert first.fingerprint != second.fingerprint
    assert first.source_sha256 != second.source_sha256


def test_simulator_revision_invalidates_cells_but_price_changes_do_not(tmp_path):
    config_path = _copy_example(tmp_path)
    original = MatrixExperimentConfig.from_file(config_path)
    original_ids = [cell.id for cell in expand_matrix(original)]

    prices_path = config_path.parent / "catalogs" / "prices.yaml"
    prices_path.write_text(
        prices_path.read_text(encoding="utf-8").replace(
            "gpu_hour: 1.0", "gpu_hour: 9.0"
        ),
        encoding="utf-8",
    )
    repriced_ids = [
        cell.id for cell in expand_matrix(MatrixExperimentConfig.from_file(config_path))
    ]
    assert repriced_ids == original_ids

    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "  simulator_revision: formal-matrix-2026-08-11-model-variant-v3",
            "  simulator_revision: next-simulator-build",
        ),
        encoding="utf-8",
    )
    revised_ids = [
        cell.id for cell in expand_matrix(MatrixExperimentConfig.from_file(config_path))
    ]
    assert set(revised_ids).isdisjoint(original_ids)


def test_applied_compute_agentic_requires_closed_loop_concurrency(tmp_path):
    spec = WorkloadSpec(
        id="applied-agentic",
        loader="applied_compute_agentic_trace",
        config={"path": "not-read-before-capability-gate.jsonl"},
    )

    with pytest.raises(ValueError, match="requires replay_concurrency"):
        prepare_workload(spec, LoaderContext(base_dir=tmp_path))


def test_synthetic_prefix_length_drives_replay_locality():
    spec = WorkloadSpec(
        id="synthetic-prefix",
        loader="synthetic",
        config={
            "isl": 1024,
            "osl": 128,
            "prefix_length": 768,
            "request_rate": 1,
        },
    )
    prepared = prepare_workload(spec, LoaderContext(base_dir=EXAMPLE_ROOT))

    assert prepared.replay_args["shared_prefix_ratio"] == pytest.approx(0.75)
    assert prepared.replay_args["num_prefix_groups"] == 1


def test_legacy_scalar_g2_normalizes_to_one_profile():
    policy = SearchPolicyConfig.model_validate(
        {
            "schema_version": 1,
            "spica": {
                "search_space": {
                    "num_g2_blocks": 4096,
                    "bandwidth_g1_to_g2_gbps": 25,
                    "bandwidth_g2_to_g1_gbps": 50,
                    "offload_batch_size": 32,
                }
            },
        }
    )

    assert len(policy.kvbm_profiles) == 1
    assert policy.kvbm_profiles[0].id == "legacy_scalar"
    assert policy.kvbm_profiles[0].num_g2_blocks == 4096
    assert policy.kvbm_profiles[0].bandwidth_from_g2_gbps == 50


@pytest.mark.parametrize("file_format", ["csv", "jsonl"])
def test_tabular_loader_materializes_mooncake(tmp_path, file_format):
    source = tmp_path / f"requests.{file_format}"
    if file_format == "csv":
        source.write_text("ts,in,out\n0,8,2\n10,9,3\n", encoding="utf-8")
    else:
        source.write_text(
            '{"ts": 0, "in": 8, "out": 2}\n{"ts": 10, "in": 9, "out": 3}\n',
            encoding="utf-8",
        )
    spec = WorkloadSpec(
        id="table",
        loader="tabular_trace",
        config={
            "path": source.name,
            "columns": {
                "timestamp": "ts",
                "input_length": "in",
                "output_length": "out",
            },
        },
    )
    prepared = prepare_workload(spec, LoaderContext(base_dir=tmp_path))
    materialized = materialize_workload(prepared, tmp_path / "normalized")
    normalized_path = Path(materialized.replay_args["trace_path"])
    rows = [json.loads(line) for line in normalized_path.read_text().splitlines()]

    assert prepared.summary["row_count"] == 2
    assert rows == [
        {"input_length": 8, "output_length": 2, "timestamp": 0.0},
        {"input_length": 9, "output_length": 3, "timestamp": 10.0},
    ]


def test_result_store_resume_repricing_and_exports(tmp_path):
    config = MatrixExperimentConfig.from_file(EXAMPLE_ROOT / "experiment.yaml")
    catalogs = load_catalogs(config)
    cell = expand_matrix(config, catalogs)[0]
    with ResultStore.open(tmp_path / "run") as store:
        experiment_id = store.start_experiment(config, catalogs)
        assert store.connection.execute(
            "SELECT count(*) FROM price_catalog"
        ).fetchone() == (0,)
        assert store.connection.execute(
            "SELECT count(*) FROM input_artifacts WHERE role = 'catalog:prices'"
        ).fetchone() == (0,)
        store.install_prices(
            [PriceSpec(id="test-price", sku_id=cell.sku.id, gpu_hour=1.0)]
        )
        store.record_cell(experiment_id, cell, status="running")
        evaluation_key = "stable-evaluation"
        candidate_id = store.record_evaluation(
            cell_id=cell.id,
            evaluation_key=evaluation_key,
            stage="spica",
            outcome="feasible",
            reason="",
            selection={"router_mode": "round_robin"},
            metrics={
                "gpu_hours": 2.0,
                "duration_ms": 3_600_000.0,
                "mean_output_token_throughput_per_user": 40.0,
            },
            sla_pass=True,
            candidate={
                "config": {
                    "used_gpus": 2,
                    "deployment_mode": "agg",
                    "kvbm_profile_id": "g2_off",
                },
                "used_gpus": 2,
                "score": 10.0,
                "metrics": {},
                "objectives": {"goodput_per_gpu": 10.0},
            },
            resources={
                "gpu_hours": 2.0,
                "g2_gib_hours": 0.0,
                "network_gb": 0.0,
                "input_tokens": 500_000,
                "output_tokens": 500_000,
                "goodput_input_tokens": 500_000,
                "goodput_output_tokens": 500_000,
                "duration_ms": 3_600_000.0,
            },
        )
        assert candidate_id is not None
        assert store.has_successful_evaluation(cell.id, evaluation_key)
        fronts = store.mark_cell_frontiers(cell.id, [candidate_id], epsilon=0.03)
        assert fronts == {
            "strict_pareto": [candidate_id],
            "epsilon_pareto": [candidate_id],
        }
        assert store.candidate_ids_for_cell(cell.id, stage="spica") == [candidate_id]
        store.create_analysis_views(topk=5)
        cost = store.connection.execute(
            "SELECT gpu_only_cost_per_million_total_tokens FROM candidate_costs"
        ).fetchone()
        assert cost is not None and cost[0] == pytest.approx(2.0)
        first_report = generate_report(store, topk=5)
        second_report = generate_report(store, topk=5)
        assert first_report == second_report
        store.export_parquet()

    assert (tmp_path / "run" / "events.jsonl").is_file()
    assert (tmp_path / "run" / "evaluations.parquet").is_file()
    assert (tmp_path / "run" / "pareto.parquet").is_file()
    assert (tmp_path / "run" / "pareto_front.csv").is_file()
    assert (tmp_path / "run" / "cost_by_model_sku.csv").is_file()
    assert (tmp_path / "run" / "kv_bandwidth_impact.csv").is_file()
    assert (tmp_path / "run" / "kv_footprint_audit.csv").is_file()
    assert (tmp_path / "run" / "what_if_pareto_front.csv").is_file()
    assert (tmp_path / "run" / "variant_resolution.csv").is_file()
    assert (tmp_path / "run" / "variant_resolution.md").is_file()
    assert (tmp_path / "run" / "report.json").is_file()
    figures = sorted((tmp_path / "run" / "figures").glob("*.png"))
    assert len(figures) == 7
    assert all(figure.stat().st_size > 1_000 for figure in figures)
    markdown = (tmp_path / "run" / "preliminary_findings.md").read_text(
        encoding="utf-8"
    )
    assert "# DynoSim + AIC 矩阵搜索报告" in markdown
    assert "## Pareto 曲线（中间结果）" in markdown
    assert "## 模型 Variant 与 AIC 精度契约" in markdown
    assert "figures/00_pareto_front.png" in markdown
    assert "## 5. KV 存储带宽与 compute NIC 带宽比例的收益" in markdown
    assert "figures/05_kv_bandwidth_uplift.png" in markdown
    assert "## KV 字节量口径" in markdown
    report_payload = json.loads(
        (tmp_path / "run" / "report.json").read_text(encoding="utf-8")
    )
    assert report_payload["markdown_report"] == "preliminary_findings.md"
    assert len(report_payload["figures"]) == 7
    with (tmp_path / "run" / "cost_by_model_sku.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert "model_variant_id" in next(csv.reader(handle))


def test_report_fails_closed_without_report_time_price(tmp_path):
    config = MatrixExperimentConfig.from_file(EXAMPLE_ROOT / "experiment.yaml")
    catalogs = load_catalogs(config)
    cell = expand_matrix(config, catalogs)[0]

    with ResultStore.open(tmp_path / "missing-price") as store:
        experiment_id = store.start_experiment(config, catalogs)
        store.record_cell(experiment_id, cell, status="complete")
        with pytest.raises(ValueError, match="does not cover run SKU"):
            generate_report(store)


def test_formal_runner_script_dry_run(tmp_path):
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "run_matrix_experiment.sh"
    )
    completed = subprocess.run(
        [
            str(script),
            "--smoke",
            "--dry-run",
            "--skip-linkage-audit",
            "--output-dir",
            str(tmp_path / "dry-run"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "MATRIX_PYTHON": sys.executable},
    )

    assert "dry-run 完成" in completed.stdout
    assert (tmp_path / "dry-run" / "validation.json").is_file()
    assert (tmp_path / "dry-run" / "variant_resolution.json").is_file()
    assert (tmp_path / "dry-run" / "expansion.json").is_file()


def test_kvbm_matched_config_and_price_provenance_are_reportable(tmp_path):
    config = MatrixExperimentConfig.from_file(EXAMPLE_ROOT / "experiment.yaml")
    catalogs = load_catalogs(config)
    cell = expand_matrix(config, catalogs)[0]
    metrics = {
        "mean_output_token_throughput_per_user": 40.0,
        "mean_ttft_ms": 100.0,
        "mean_tpot_ms": 10.0,
    }
    resources = {
        "gpu_hours": 1.0,
        "g2_gib_hours": 0.0,
        "network_gb": 0.0,
        "input_tokens": 500_000,
        "output_tokens": 500_000,
        "goodput_input_tokens": 500_000,
        "goodput_output_tokens": 500_000,
        "duration_ms": 3_600_000.0,
    }
    common_config = {
        "backend": "vllm",
        "backend_version": "test",
        "deployment_mode": "agg",
        "used_gpus": 2,
        "router_mode": "round_robin",
        "max_sla_rps": 2.0,
    }

    with ResultStore.open(tmp_path / "matched") as store:
        experiment_id = store.start_experiment(config, catalogs)
        store.record_cell(experiment_id, cell, status="complete")
        store.install_prices(
            [
                PriceSpec(
                    id="public-provenance",
                    sku_id=cell.sku.id,
                    gpu_hour=1.0,
                    source_url="https://example.invalid/public-price.csv",
                    source_date="2026-08-07",
                    basis="public per-GPU anchor",
                    notes="test provenance",
                )
            ]
        )
        candidate_ids = []
        for label, kvbm_fields, gpu_hours in (
            (
                "off",
                {
                    "kvbm_profile_id": "custom_disabled_name",
                    "num_g2_blocks": 0,
                    "g2_capacity_gib": None,
                    "bandwidth_g1_to_g2_gbps": None,
                    "bandwidth_g2_to_g1_gbps": None,
                    "offload_batch_size": None,
                },
                1.0,
            ),
            (
                "on",
                {
                    "kvbm_profile_id": "custom_enabled_name",
                    "num_g2_blocks": 1024,
                    "g2_capacity_gib": 64.0,
                    "bandwidth_g1_to_g2_gbps": 25.0,
                    "bandwidth_g2_to_g1_gbps": 25.0,
                    "offload_batch_size": 32,
                },
                0.8,
            ),
        ):
            candidate_id = store.record_evaluation(
                cell_id=cell.id,
                evaluation_key=f"matched-{label}",
                stage="capacity",
                outcome="feasible",
                reason="",
                selection={},
                metrics=metrics,
                sla_pass=True,
                candidate={
                    "config": {**common_config, **kvbm_fields},
                    "used_gpus": 2,
                    "score": 10.0,
                    "metrics": metrics,
                    "objectives": {"goodput_per_gpu": 10.0},
                },
                resources={**resources, "gpu_hours": gpu_hours},
                candidate_context=label,
            )
            assert candidate_id is not None
            candidate_ids.append(candidate_id)

        keys = store.connection.execute(
            "SELECT count(DISTINCT compute_config_key) FROM candidates"
        ).fetchone()
        assert keys == (1,)
        store.create_analysis_views()
        matched = store.connection.execute(
            """
            SELECT baseline_cost, matched_config_cost_reduction
            FROM kvbm_uplift
            WHERE kvbm_profile_id = 'custom_enabled_name'
              AND price_id = 'public-provenance'
            """
        ).fetchone()
        assert matched == pytest.approx((1.0, 0.2))
        provenance = store.connection.execute(
            """
            SELECT source_url, source_date, basis, notes
            FROM price_catalog WHERE price_id = 'public-provenance'
            """
        ).fetchone()
        assert provenance == (
            "https://example.invalid/public-price.csv",
            "2026-08-07",
            "public per-GPU anchor",
            "test provenance",
        )

        # Reports must remain readable for databases created before the schema-v1
        # spelling clarification. New CSV columns always use explicit GB/s names.
        raw_cell = store.connection.execute(
            "SELECT cell_json FROM matrix_cells WHERE cell_id = ?", [cell.id]
        ).fetchone()[0]
        legacy_cell = json.loads(raw_cell)
        legacy_sku = legacy_cell["sku"]
        legacy_sku["compute_nic_gbps_per_gpu"] = legacy_sku.pop(
            "compute_nic_gbytes_per_s_per_gpu"
        )
        store.connection.execute(
            "UPDATE matrix_cells SET cell_json = ? WHERE cell_id = ?",
            [json.dumps(legacy_cell), cell.id],
        )
        generate_report(store)
        with (tmp_path / "matched" / "kv_bandwidth_impact.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]["storage_bandwidth_gbytes_per_s"] == "25.0"
        assert rows[0]["compute_nic_gbytes_per_s"] == "100.0"
        assert "pd_bandwidth_domain" in rows[0]
        assert "pd_transfer_bandwidth_gbytes_per_s" in rows[0]
        assert "storage_bandwidth_gbps" not in rows[0]
        assert "compute_nic_gbps" not in rows[0]


def test_capacity_search_records_all_probes_and_right_censoring(
    tmp_path,
    monkeypatch,
):
    config = MatrixExperimentConfig.from_file(
        EXAMPLE_ROOT / "experiment_preliminary_smoke.yaml"
    )
    cell = expand_matrix(config)[0]
    runner = MatrixRunner()
    kvbm_profile = cell.search_policy.kvbm_profiles[0]
    search_config = runner._smart_search_config(
        cell,
        cell.workload,
        "agg",
        None,
        kvbm_profile,
        "request_rate",
        0.25,
    )
    source = Candidate(
        config={
            "backend_version": "test",
            "deployment_mode": "agg",
            "used_gpus": 2,
        },
        used_gpus=2,
        score=1.0,
        metrics={},
        objectives={"goodput_per_gpu": 1.0},
    )

    def fake_cached_evaluate(*args, **kwargs):
        del args, kwargs
        return CachedReplayResult(
            metrics={
                "num_requests": 10.0,
                "completed_requests": 10.0,
                "goodput_completed_requests": 10.0,
                "mean_ttft_ms": 100.0,
                "mean_tpot_ms": 10.0,
                "mean_output_token_throughput_per_user": 50.0,
                "p95_ttft_ms": 150.0,
                "p95_tpot_ms": 15.0,
                "p10_output_token_throughput_per_user": 40.0,
                "duration_ms": 1000.0,
                "gpu_hours": 1.0,
                "total_input_tokens": 1000.0,
                "total_output_tokens": 100.0,
            },
            description=ReplayCacheDescription(
                evaluation_cache_key="test-cache-key",
                cache_namespace_key="test-namespace",
                semantics={"test": True},
            ),
            cache_hit=False,
        )

    def fake_candidate(config, report, target, *, pareto_objectives):
        del target, pareto_objectives
        return Candidate(
            config=config,
            used_gpus=2,
            score=1.0,
            metrics=report,
            objectives={"goodput_per_gpu": 1.0},
        )

    monkeypatch.setattr(runner, "_evaluate_with_cache", fake_cached_evaluate)
    monkeypatch.setattr(
        "aisimulate.matrix.runner.build_deployment", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr("aisimulate.matrix.runner.make_candidate", fake_candidate)
    probes = []
    with ResultStore.open(tmp_path / "capacity") as store:
        cache_evaluator = runner._cache_evaluator(
            store,
            cell,
            cell.workload,
            search_config,
            "test-revision",
        )
        candidate, _ = runner._capacity_search(
            cell,
            cell.workload,
            search_config,
            source,
            store=store,
            cache_evaluator=cache_evaluator,
            on_probe=lambda index, phase, axis, load, metrics, passed, *_: (
                probes.append((index, phase, axis, load, passed))
            ),
        )

    assert candidate is not None
    assert candidate.config["max_sla_rps"] == 4.0
    assert candidate.config["capacity_search_right_censored"] is True
    assert candidate.config["capacity_search_infeasible_bound"] is None
    assert probes == [
        (0, "initial", "request_rate", 0.25, True),
        (1, "expansion", "request_rate", 0.5, True),
        (2, "expansion", "request_rate", 1.0, True),
        (3, "expansion", "request_rate", 2.0, True),
        (4, "expansion", "request_rate", 4.0, True),
    ]


def test_capacity_search_brackets_downward_when_initial_rps_fails(
    tmp_path,
    monkeypatch,
):
    config = MatrixExperimentConfig.from_file(
        EXAMPLE_ROOT / "experiment_parallel_focus.yaml"
    )
    cell = expand_matrix(config)[0]
    runner = MatrixRunner()
    kvbm_profile = cell.search_policy.kvbm_profiles[0]
    search_config = runner._smart_search_config(
        cell,
        cell.workload,
        "agg",
        None,
        kvbm_profile,
        "request_rate",
        4.0,
    )
    source = Candidate(
        config={
            "backend_version": "test",
            "deployment_mode": "agg",
            "used_gpus": 2,
        },
        used_gpus=2,
        score=1.0,
        metrics={},
        objectives={"goodput_per_gpu": 1.0, "throughput_per_user": 1.0},
    )
    observed_loads = []

    def fake_cached_evaluate(store, cell, stage, evaluator, plan):
        del store, cell, stage, plan
        load = float(evaluator.workload.request_rate)
        observed_loads.append(load)
        passed = load <= 1.0
        return CachedReplayResult(
            metrics={
                "num_requests": 100.0,
                "completed_requests": 100.0,
                "goodput_completed_requests": 100.0 if passed else 0.0,
                "mean_ttft_ms": 100.0 if passed else 10_000.0,
                "mean_tpot_ms": 10.0 if passed else 100.0,
                "mean_output_token_throughput_per_user": 50.0 if passed else 1.0,
                "p95_ttft_ms": 150.0 if passed else 10_000.0,
                "p95_tpot_ms": 15.0 if passed else 100.0,
                "p10_output_token_throughput_per_user": 40.0 if passed else 1.0,
                "duration_ms": 1000.0,
                "gpu_hours": 1.0,
            },
            description=ReplayCacheDescription(
                evaluation_cache_key=f"load-{load}",
                cache_namespace_key="test",
                semantics={},
            ),
            cache_hit=False,
        )

    monkeypatch.setattr(runner, "_evaluate_with_cache", fake_cached_evaluate)
    monkeypatch.setattr(
        "aisimulate.matrix.runner.build_deployment", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        "aisimulate.matrix.runner.make_candidate",
        lambda config, report, target, *, pareto_objectives: source.model_copy(
            update={"config": config, "metrics": report}
        ),
    )
    with ResultStore.open(tmp_path / "downward") as store:
        evaluator = runner._cache_evaluator(
            store, cell, cell.workload, search_config, "test-revision"
        )
        candidate, _ = runner._capacity_search(
            cell,
            cell.workload,
            search_config,
            source,
            store=store,
            cache_evaluator=evaluator,
        )

    assert candidate is not None
    assert observed_loads[:3] == [4.0, 2.0, 1.0]
    assert candidate.config["capacity_search_status"] == "bracketed"
    assert candidate.config["capacity_search_right_censored"] is False
    assert 0.97 <= candidate.config["max_sla_rps"] <= 1.0
    assert 1.0 < candidate.config["capacity_search_infeasible_bound"] <= 1.04


def test_explicit_cell_workers_keep_result_store_single_writer(tmp_path):
    config_path = _copy_example(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        .replace(
            "model_selector: {tags: [formal-or-experimental]}",
            "model_selector: {ids: [deepseek_v4_flash]}",
        )
        .replace("cell_workers: 1", "cell_workers: 2"),
        encoding="utf-8",
    )
    variants_path = config_path.parent / "catalogs" / "model_variants.yaml"
    variants_path.write_text(
        variants_path.read_text(encoding="utf-8").replace(
            "support_tier: formal", "support_tier: unsupported", 1
        ),
        encoding="utf-8",
    )
    config = MatrixExperimentConfig.from_file(config_path)

    with ResultStore.open(tmp_path / "parallel-run") as store:
        summary = MatrixRunner().run(config, store)

    assert summary["cells"] == 4
    assert summary["status"] == "partial"
    assert summary["unsupported"] == 4
    assert summary["failed"] == 0
    manifest = json.loads(
        (tmp_path / "parallel-run" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["runtime_provenance"]["binding"]["origin"]
    assert set(
        manifest["runtime_provenance"]["aiconfigurator"]["selected_systems"]
    ) == {"b300_sxm", "gb300"}


def test_matrix_runner_file_and_synthetic_resume_end_to_end(tmp_path, monkeypatch):
    config_path = _copy_example(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        .replace(
            "model_selector: {tags: [formal-or-experimental]}",
            "model_selector: {ids: [deepseek_v4_flash, glm_5_2]}",
        )
        .replace(
            "workload_selector: {tags: [production-study]}",
            "workload_selector: {ids: [synthetic_4k, agentic_from_file]}",
        )
        .replace(
            "simulator_revision: formal-matrix-2026-08-11-model-variant-v3",
            "simulator_revision: null",
        ),
        encoding="utf-8",
    )
    workloads_path = config_path.parent / "catalogs" / "workloads.yaml"
    workload_catalog = yaml.safe_load(workloads_path.read_text(encoding="utf-8"))
    for workload in workload_catalog["workloads"]:
        if workload["id"] == "synthetic_4k":
            workload["config"]["request_rates"] = [1]
        elif workload["id"] == "agentic_from_file":
            workload["config"]["arrival_speedup_ratios"] = [1]
    workloads_path.write_text(
        yaml.safe_dump(workload_catalog, sort_keys=False), encoding="utf-8"
    )
    trace_path = config_path.parent / "traces" / "agentic.jsonl"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text(
        '{"timestamp": 0, "input_length": 128, "output_length": 16}\n',
        encoding="utf-8",
    )
    policy_path = config_path.parent / "policies" / "vllm_pd_g2.yaml"
    policy_path.write_text(
        policy_path.read_text(encoding="utf-8")
        .replace("deployment_mode: [disagg, agg]", "deployment_mode: [agg]")
        .replace(
            "capacity_search:\n  enabled: true", "capacity_search:\n  enabled: false"
        )
        .replace("robustness:\n  enabled: true", "robustness:\n  enabled: false"),
        encoding="utf-8",
    )
    calls = 0

    def fake_prune(config, model, policy):
        del model, policy
        return StaticPruneResult(
            config=config,
            generated=1,
            retained=1,
            decisions=(),
        )

    def fake_search(config, *, evaluator, show_progress):
        nonlocal calls
        del evaluator, show_progress
        calls += 1
        candidate = Candidate(
            config={
                "deployment_mode": "agg",
                "backend_version": "test",
                "used_gpus": 2,
                "kvbm_profile_id": config.search_space.kvbm_profile_id,
                "g2_capacity_gib": config.search_space.g2_capacity_gib,
            },
            used_gpus=2,
            score=10.0,
            metrics={
                "num_requests": 10.0,
                "completed_requests": 10.0,
                "goodput_completed_requests": 10.0,
                "mean_ttft_ms": 100.0,
                "mean_tpot_ms": 10.0,
                "mean_output_token_throughput_per_user": 50.0,
                "p95_ttft_ms": 150.0,
                "p95_tpot_ms": 15.0,
                "p10_output_token_throughput_per_user": 40.0,
                "duration_ms": 1000.0,
                "gpu_hours": 2 / 3600,
                "total_input_tokens": 1280.0,
                "total_output_tokens": 160.0,
                "prefix_cache_reused_ratio": 0.5,
            },
            objectives={"goodput_per_gpu": 10.0, "throughput_per_user": 50.0},
        )
        evaluation = SearchEvaluation(
            index=1,
            branch="agg",
            outcome="feasible",
            selection={"deployment_mode": "agg"},
            parallel_config={"tp": 2},
            candidate=candidate,
        )
        return SmartSearchRunResult(
            candidates=[candidate],
            evaluations=[evaluation],
            tally={"feasible": 1},
        )

    monkeypatch.setattr("aisimulate.matrix.runner.prune_parallel_configs", fake_prune)
    monkeypatch.setattr(
        "aisimulate.matrix.runner.run_smart_search_with_history", fake_search
    )
    config = MatrixExperimentConfig.from_file(config_path)
    with ResultStore.open(tmp_path / "e2e-run") as store:
        first = MatrixRunner().run(config, store)
        calls_after_first = calls
        second = MatrixRunner().run(config, store, resume=True)

    assert first["cells"] == 8
    assert first["completed"] == 8
    assert second["completed"] == 8
    assert calls_after_first == 40
    assert calls == calls_after_first
