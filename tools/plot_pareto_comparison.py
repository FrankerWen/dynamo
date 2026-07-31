#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pareto Curve Comparison Plotter

Plot 1x2 Subplots with Token Throughput (tokens/s/GPU) on the Y-AXIS (Vertical Axis):
1. Subplot A: Token Throughput (tokens/s/GPU) [Y-axis] vs TTFT (ms) [X-axis]
2. Subplot B: Token Throughput (tokens/s/GPU) [Y-axis] vs TPS_per_user (tokens/s/user) [X-axis]

Across 4 series:
- Dynamo + AIC (Aggregated)
- Dynamo + AIC (Disaggregated)
- AIC Only (Aggregated)
- AIC Only (Disaggregated)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("/tmp/h100_128gpu_pareto_data")
ARTIFACT_DIR = Path("/root/.gemini/antigravity-ide/brain/9859df2a-0f2f-41b9-8285-63e3e447b8e2")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def plot_pareto_comparison():
    aic_csv = DATA_DIR / "aic_pareto_results.csv"
    dynamo_csv = DATA_DIR / "dynamo_pareto_results.csv"

    if not aic_csv.is_file() or not dynamo_csv.is_file():
        print(f"Error: Required CSV files not found in {DATA_DIR}")
        sys.exit(1)

    df_aic = pd.read_csv(aic_csv)
    df_dyn = pd.read_csv(dynamo_csv)

    # Set up matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    fig.suptitle('H100 128-GPU DeepSeek-V3 (671B) Pareto Curve Comparison:\nDynamo + AIC (Discrete-Event Simulator) vs AIC Only (Analytical Model)', 
                 fontsize=14, fontweight='bold', y=0.98)

    # Series definitions: color, marker, linestyle
    style_map = {
        ("Dynamo + AIC", "Aggregated"): {"color": "#1f77b4", "marker": "o", "ls": "-", "label": "Dynamo + AIC (Aggregated)"},
        ("Dynamo + AIC", "Disaggregated"): {"color": "#ff7f0e", "marker": "s", "ls": "-", "label": "Dynamo + AIC (Disaggregated)"},
        ("AIC Only", "Aggregated"): {"color": "#2ca02c", "marker": "^", "ls": "--", "label": "AIC Only (Aggregated)"},
        ("AIC Only", "Disaggregated"): {"color": "#d62728", "marker": "D", "ls": "--", "label": "AIC Only (Disaggregated)"},
    }

    # Left Plot (Subplot A): Y-axis = Token Throughput (tokens/s/GPU), X-axis = TTFT (ms)
    # Right Plot (Subplot B): Y-axis = Token Throughput (tokens/s/GPU), X-axis = TPS per user (tokens/s/user)
    
    # 1. Plot Dynamo data
    for mode in ["Aggregated", "Disaggregated"]:
        sub = df_dyn[df_dyn["serving_mode"] == mode].copy()
        if not sub.empty:
            sub = sub.sort_values("ttft")
            st = style_map[("Dynamo + AIC", mode)]
            
            # Y: Throughput, X: TTFT
            ax1.plot(sub["ttft"], sub["tokens/s/gpu"], 
                     color=st["color"], marker=st["marker"], linestyle=st["ls"], 
                     linewidth=2.5, markersize=8, label=st["label"])
            
            # Y: Throughput, X: TPS per user
            sub_tps = sub.sort_values("tokens/s/user")
            ax2.plot(sub_tps["tokens/s/user"], sub_tps["tokens/s/gpu"], 
                     color=st["color"], marker=st["marker"], linestyle=st["ls"], 
                     linewidth=2.5, markersize=8, label=st["label"])

    # 2. Plot AIC data
    for mode in ["Aggregated", "Disaggregated"]:
        sub = df_aic[df_aic["serving_mode"] == mode].copy()
        if not sub.empty:
            sub_ttft = sub.sort_values("ttft")
            st = style_map[("AIC Only", mode)]
            
            ax1.plot(sub_ttft["ttft"], sub_ttft["tokens/s/gpu"], 
                     color=st["color"], marker=st["marker"], linestyle=st["ls"], 
                     linewidth=2.2, markersize=7, label=st["label"], alpha=0.85)
            
            sub_tps = sub.sort_values("tokens/s/user")
            ax2.plot(sub_tps["tokens/s/user"], sub_tps["tokens/s/gpu"], 
                     color=st["color"], marker=st["marker"], linestyle=st["ls"], 
                     linewidth=2.2, markersize=7, label=st["label"], alpha=0.85)

    # Subplot A Formatting (Y: Throughput vs X: TTFT)
    ax1.set_title("Subplot A: Token Throughput vs TTFT", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Mean TTFT (ms)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Token Throughput (tokens/s/GPU)", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=9.5)

    # Subplot B Formatting (Y: Throughput vs X: TPS per User)
    ax2.set_title("Subplot B: Token Throughput vs TPS per User", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("TPS per User (tokens/s/user)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Token Throughput (tokens/s/GPU)", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=9.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    
    png_path = ARTIFACT_DIR / "h100_128gpu_pareto_comparison.png"
    svg_path = ARTIFACT_DIR / "h100_128gpu_pareto_comparison.svg"
    
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Updated Pareto comparison plot with Token Throughput on Y-axis:\n  - {png_path}\n  - {svg_path}")


if __name__ == "__main__":
    plot_pareto_comparison()
