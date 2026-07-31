# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from benchmarks.mocker.dynosim_hisim_alignment import (
    convert_workload,
    deterministic_uuid,
    discover_points,
    engine_args,
    load_and_validate_point,
)


pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


def _write_point(root: Path, name: str = "sweep_b_mixed_rps_point_8") -> Path:
    point = root / name
    (point / "raw_sglang").mkdir(parents=True)
    (point / "acc_bench" / "compare").mkdir(parents=True)
    (point / "acc_bench" / "hisim_aic").mkdir(parents=True)
    rows = [
        {
            "rid": "r0",
            "timestamp": 0.25,
            "input_ids": [11, 12, 13],
            "input_length": 3,
            "output_length": 2,
        },
        {
            "rid": "r1",
            "timestamp": 0.5,
            "input_ids": [11, 12, 99],
            "input_length": 3,
            "output_length": 2,
        },
    ]
    (point / "bench_serving_dataset.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    raw = {
        "completed": 2,
        "rids": ["r0", "r1"],
        "input_lens": [3, 3],
        "output_lens": [2, 2],
        "ttfts": [0.1, 0.2],
        "e2e_latencies": [0.2, 0.3],
        "server_info": {
            "ep_dispatch_algorithm": "fake",
            "tp_size": 4,
            "ep_size": 4,
            "dp_size": 1,
            "max_total_tokens": 655360,
            "max_prefill_tokens": 16384,
            "chunked_prefill_size": 8192,
            "page_size": 1,
            "disable_cuda_graph": False,
            "disable_cuda_graph_padding": False,
            "cuda_graph_bs": [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64],
        },
    }
    (point / "raw_sglang" / "client_metrics.jsonl").write_text(json.dumps(raw) + "\n")
    metrics = {
        key: {"raw": 10.0, "mape_percent": 1.0}
        for key in (
            "mean_ttft_ms",
            "mean_tpot_ms",
            "mean_itl_ms",
            "mean_e2e_latency_ms",
            "input_throughput",
        )
    }
    mape = {
        "completed": {"raw": 2, "hisim_aic": 2},
        "ingress_envelope": {"ingress_ms": 3.0},
        "metrics": metrics,
    }
    (point / "acc_bench" / "compare" / "acc_bench_mape.json").write_text(
        json.dumps(mape)
    )
    sim_config = {
        "predictor": {"workload_distribution": "balanced", "device_name": "h100_pcie"},
        "scheduler": {
            "backend_name": "sglang",
            "backend_version": "0.5.9",
            "tp_size": 4,
            "ep_size": 4,
            "dp_size": 1,
        },
    }
    (point / "acc_bench" / "hisim_aic" / "sim_config.json").write_text(
        json.dumps(sim_config)
    )
    return point


def test_converter_preserves_seconds_rids_tokens_prefix_and_hashes(tmp_path):
    point = load_and_validate_point(_write_point(tmp_path))
    trace = tmp_path / "out" / "trace.jsonl"
    rid_map = tmp_path / "out" / "rid_map.json"
    manifest = convert_workload(point, trace, rid_map)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]

    assert rows[0]["timestamp"] == 250.0
    assert rows[1]["timestamp"] == 500.0
    assert rows[0]["request_id"] == "r0"
    assert rows[0]["hash_ids"] == [11, 12, 13]
    assert rows[1]["hash_ids"][:2] == rows[0]["hash_ids"][:2]
    assert manifest["request_count"] == 2
    assert len(manifest["trace_sha256"]) == 64


def test_deterministic_uuid_matches_replay_bench_line_ordinal():
    assert deterministic_uuid(0) == "00000000-0000-0000-0000-000000000001"
    assert deterministic_uuid(31) == "00000000-0000-0000-0000-000000000020"


def test_discovery_excludes_single_batch_and_backup_directories(tmp_path):
    for scenario in ("a", "b", "b_mixed", "c", "d"):
        for point in (1, 2, 4, 8, 16, 24, 32):
            (tmp_path / f"sweep_{scenario}_rps_point_{point}").mkdir()
    (tmp_path / "sweep_a_single_batch_point_1").mkdir()
    (tmp_path / "sweep_a_rps_point_1_backup").mkdir()

    discovered = discover_points(tmp_path)

    assert len(discovered) == 35
    assert all("_single_batch_" not in path.name and "backup" not in path.name for path in discovered)


def test_engine_args_fix_sglang_graph_and_tp4_ep4_balanced_mapping():
    args = engine_args()
    assert args["engine_type"] == "sglang"
    assert args["aic_tp_size"] * args["aic_attention_dp_size"] == (
        args["aic_moe_tp_size"] * args["aic_moe_ep_size"]
    )
    assert args["aic_workload_distribution"] == "balanced"
    assert args["aic_kv_cache_dtype"] == "bfloat16"
    assert args["aic_mla_module_kv_cache_dtype"] == "fp8"
    assert args["aic_fmha_dtype"] == "bfloat16"
    assert args["sglang"]["cuda_graph_padding_enabled"] is True
    assert args["sglang"]["cuda_graph_batch_sizes"][-1] == 64


def test_engine_args_reuses_per_point_formal_cuda_graph_contract(tmp_path):
    point_path = _write_point(tmp_path, "sweep_d_rps_point_32")
    sim_config_path = point_path / "acc_bench" / "hisim_aic" / "sim_config.json"
    sim_config = json.loads(sim_config_path.read_text())
    sim_config["scheduler"]["cuda_graph_batch_sizes"] = [1, 2, 4, 8, 16, 32, 64, 128]
    sim_config_path.write_text(json.dumps(sim_config))

    args = engine_args(load_and_validate_point(point_path))

    assert args["sglang"]["cuda_graph_batch_sizes"] == [1, 2, 4, 8, 16, 32, 64, 128]
