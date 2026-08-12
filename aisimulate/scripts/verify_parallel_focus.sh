#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
SWEEP_ROOT="$(cd -- "$REPO_ROOT/.." && pwd -P)"

if [[ -n "${MATRIX_PYTHON:-}" ]]; then
    PYTHON_BIN="$MATRIX_PYTHON"
elif [[ -x "$SWEEP_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$SWEEP_ROOT/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi

RUN_DIR="${1:-$SWEEP_ROOT/results/matrix-parallel-focus}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/verify_matrix_run.py" \
    --run-dir "$RUN_DIR" \
    --expected-cells 1 \
    --model deepseek_v4_pro \
    --model-variant deepseek_v4_pro_native_vllm \
    --sku gb300 \
    --workload synthetic_8k_1k_no_prefix \
    --require-curve-data
