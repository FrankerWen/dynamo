#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
SWEEP_ROOT="$(cd -- "$REPO_ROOT/.." && pwd -P)"

CONFIG="$REPO_ROOT/examples/aisimulate/matrix/experiment_parallel_focus.yaml"
OUTPUT_DIR="$SWEEP_ROOT/results/matrix-parallel-focus"

echo "两阶段 focus：AIC concurrency → DynoSim/Spica 分负载 Pareto → 开放环 RPS SLA 认证"
echo "配置：GB300 / DeepSeek V4 Pro native / ISL 8192 / OSL 1024 / prefix 0"

exec "$SCRIPT_DIR/run_matrix_experiment.sh" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
