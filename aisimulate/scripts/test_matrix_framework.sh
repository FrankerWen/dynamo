#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Repeatable CPU-only framework tests plus a configuration dry run.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
SWEEP_ROOT="$(cd -- "$REPO_ROOT/.." && pwd -P)"

if [[ -n "${MATRIX_PYTHON:-}" ]]; then
    PYTHON_BIN="$MATRIX_PYTHON"
elif [[ -x "$SWEEP_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$SWEEP_ROOT/.venv/bin/python"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TEMP_DIR"' EXIT

export PYTHONPATH="$REPO_ROOT/aisimulate/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m pytest -q -c /dev/null --disable-warnings \
    "$REPO_ROOT/aisimulate/tests/matrix" \
    "$REPO_ROOT/aisimulate/tests/spica"

"$SCRIPT_DIR/run_matrix_experiment.sh" \
    --smoke \
    --dry-run \
    --skip-linkage-audit \
    --output-dir "$TEMP_DIR/dry-run"

"$SCRIPT_DIR/run_matrix_experiment.sh" \
    --config "$REPO_ROOT/examples/aisimulate/matrix/experiment_parallel_focus.yaml" \
    --dry-run \
    --skip-linkage-audit \
    --output-dir "$TEMP_DIR/focus-dry-run"

echo "Matrix framework 测试、smoke 与 AIC-concurrency focus 配置展开均通过。"
