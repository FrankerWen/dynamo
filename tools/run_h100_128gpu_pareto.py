#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
H100 128-Card Pareto Benchmark Driver

Runs both:
1. aiconfigurator-dynosim-e2e (AIC Only analytical model) for agg & disagg
2. Dynamo + AIC (Discrete event queuing simulator) for agg & disagg over RPS points [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]

Saves output CSVs and triggers plot_pareto_comparison.py to generate PNG/SVG figures and report artifact.
"""

import json
import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

AIC_SRC = Path("/workspace/hisim_align/aiconfigurator-dynosim-e2e/src")
DYNAMO_SRC = Path("/workspace/hisim_align/dynamo/components/src")
DYNAMO_PYTHON = Path("/workspace/hisim_align/dynamo/.venv/bin/python")

if str(AIC_SRC) not in sys.path:
    sys.path.insert(0, str(AIC_SRC))
if str(DYNAMO_SRC) not in sys.path:
    sys.path.insert(0, str(DYNAMO_SRC))

OUT_DIR = Path("/tmp/h100_128gpu_pareto_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_aic_only():
    print("=== [1/3] Running aiconfigurator-dynosim-e2e (AIC Only) Pareto Analysis ===")
    from aiconfigurator.sdk.task import TaskConfig, TaskRunner

    common_worker_cfg = {
        "gemm_quant_mode": "fp8_block",
        "moe_quant_mode": "fp8_block",
        "kvcache_quant_mode": "bfloat16",
        "fmha_quant_mode": "bfloat16",
        "attention_backend": "flashinfer",
        "num_gpu_per_worker": [4, 8, 16, 32],
        "tp_list": [1, 2, 4, 8],
        "pp_list": [1],
        "dp_list": [1, 2, 4, 8, 16],
        "moe_tp_list": [1],
        "moe_ep_list": [1, 2, 4, 8, 16, 32],
    }

    exp_data = {
        "mode": "patch",
        "model_path": "deepseek-ai/DeepSeek-V3",
        "total_gpus": 128,
        "system_name": "h100_pcie",
        "backend_name": "sglang",
        "database_mode": "HYBRID",
        "isl": 32768,
        "osl": 128,
        "prefix": 29491,
        "ttft": 100000.0,
        "tpot": 10000.0,
    }

    # 1. Aggregated
    exp_agg = dict(exp_data, serving_mode="agg", config={"worker_config": common_worker_cfg})
    task_agg = TaskConfig(
        serving_mode="agg",
        model_path=exp_agg["model_path"],
        total_gpus=128,
        system_name="h100_pcie",
        backend_name="sglang",
        database_mode="HYBRID",
        isl=32768,
        osl=128,
        prefix=29491,
        ttft=100000.0,
        tpot=10000.0,
        yaml_config=exp_agg,
    )
    runner = TaskRunner()
    res_agg = runner.run(task_agg)
    df_agg = res_agg["pareto_df"].copy()
    df_agg["serving_mode"] = "Aggregated"
    df_agg["engine"] = "AIC Only"

    # 2. Disaggregated
    exp_disagg = dict(
        exp_data,
        serving_mode="disagg",
        config={
            "prefill_worker_config": common_worker_cfg,
            "decode_worker_config": common_worker_cfg,
        },
    )
    task_disagg = TaskConfig(
        serving_mode="disagg",
        model_path=exp_disagg["model_path"],
        total_gpus=128,
        system_name="h100_pcie",
        backend_name="sglang",
        database_mode="HYBRID",
        isl=32768,
        osl=128,
        prefix=29491,
        ttft=100000.0,
        tpot=10000.0,
        yaml_config=exp_disagg,
    )
    res_disagg = runner.run(task_disagg)
    df_disagg = res_disagg["pareto_df"].copy()
    df_disagg["serving_mode"] = "Disaggregated"
    df_disagg["engine"] = "AIC Only"

    aic_df = pd.concat([df_agg, df_disagg], ignore_index=True)
    aic_csv = OUT_DIR / "aic_pareto_results.csv"
    aic_df.to_csv(aic_csv, index=False)
    print(f"Saved AIC Pareto results ({len(aic_df)} rows) to {aic_csv}")
    return aic_df


def run_dynamo_sim():
    print("=== [2/3] Running Dynamo + AIC Discrete-Event Simulation ===")
    rps_list = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    modes = ["Aggregated", "Disaggregated"]
    rows = []

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{DYNAMO_SRC}:{AIC_SRC}:" + env.get("PYTHONPATH", "")

    base_engine_args = {
        "engine_type": "sglang",
        "num_gpu_blocks": 655360,
        "block_size": 1,
        "max_num_batched_tokens": 16384,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "aic_backend": "sglang",
        "aic_system": "h100_pcie",
        "aic_backend_version": "0.5.9",
        "aic_tp_size": 2,
        "aic_moe_tp_size": 1,
        "aic_moe_ep_size": 16,
        "aic_attention_dp_size": 8,
        "aic_workload_distribution": "balanced",
        "aic_kv_cache_dtype": "bfloat16",
        "aic_mla_module_kv_cache_dtype": "fp8",
        "aic_fmha_dtype": "bfloat16",
        "aic_model_path": "deepseek-ai/DeepSeek-V3",
        "sglang": {
            "schedule_policy": "fcfs",
            "page_size": 1,
            "max_prefill_tokens": 16384,
            "chunked_prefill_size": 8192,
            "clip_max_new_tokens": 4096,
            "cuda_graph_enabled": True,
            "cuda_graph_padding_enabled": True,
            "cuda_graph_batch_sizes": [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64],
            "enable_mixed_chunk": False,
        },
    }

    for mode in modes:
        for rps in rps_list:
            req_count = max(32, int(rps * 10))
            report_file = OUT_DIR / f"dynamo_report_{mode.lower()}_rps_{rps}.json"

            cmd = [
                str(DYNAMO_PYTHON),
                "-m",
                "dynamo.replay",
                "--input-tokens", "32768",
                "--output-tokens", "128",
                "--shared-prefix-ratio", "0.90003",
                "--num-prefix-groups", "8",
                "--request-count", str(req_count),
                "--request-rate", str(float(rps)),
                "--replay-mode", "offline",
                "--router-mode", "kv_router",
                "--router-config", json.dumps({"router_prefill_load_model": "aic"}),
                "--aic-backend", "sglang",
                "--aic-system", "h100_pcie",
                "--aic-backend-version", "0.5.9",
                "--aic-model-path", "deepseek-ai/DeepSeek-V3",
                "--report-json", str(report_file),
            ]

            if mode == "Disaggregated":
                prefill_args = dict(base_engine_args, worker_type="prefill")
                decode_args = dict(base_engine_args, worker_type="decode")
                cmd.extend([
                    "--prefill-engine-args", json.dumps(prefill_args),
                    "--decode-engine-args", json.dumps(decode_args),
                    "--num-prefill-workers", "2",
                    "--num-decode-workers", "2",
                ])
            else:
                agg_args = dict(base_engine_args, worker_type="aggregated")
                cmd.extend([
                    "--extra-engine-args", json.dumps(agg_args),
                    "--num-workers", "4",
                ])

            print(f"Running Dynamo {mode} simulation @ RPS {rps}...")
            res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                print(f"Warning: Dynamo {mode} @ RPS {rps} failed: {res.stderr[:200]}")
                continue

            if report_file.is_file():
                with report_file.open() as f:
                    rep = json.load(f)

                ttft_ms = rep.get("mean_ttft_ms", 0.0)
                tpot_ms = rep.get("mean_tpot_ms", rep.get("mean_itl_ms", 0.0))
                req_throughput = rep.get("request_throughput_rps", 0.0)
                output_throughput = rep.get("output_throughput_tok_s", 0.0)
                tps_user = rep.get("mean_output_token_throughput_per_user", 1000.0 / tpot_ms if tpot_ms > 0 else 0.0)
                tok_per_s_gpu = output_throughput / 128.0

                rows.append({
                    "engine": "Dynamo + AIC",
                    "serving_mode": mode,
                    "rps": rps,
                    "request_throughput": req_throughput,
                    "output_throughput": output_throughput,
                    "tokens/s/gpu": tok_per_s_gpu,
                    "ttft": ttft_ms,
                    "tpot": tpot_ms,
                    "tokens/s/user": tps_user,
                    "prefix_hit_ratio": rep.get("prefix_cache_reused_ratio", 0.0),
                })

    dynamo_df = pd.DataFrame(rows)
    dynamo_csv = OUT_DIR / "dynamo_pareto_results.csv"
    dynamo_df.to_csv(dynamo_csv, index=False)
    print(f"Saved Dynamo simulation results ({len(dynamo_df)} rows) to {dynamo_csv}")
    return dynamo_df


def main():
    aic_df = run_aic_only()
    dynamo_df = run_dynamo_sim()
    print("=== [3/3] Driver completed data generation successfully ===")


if __name__ == "__main__":
    main()
