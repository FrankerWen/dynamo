#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
SWEEP_ROOT="$(cd -- "$REPO_ROOT/.." && pwd -P)"

CONFIG="$REPO_ROOT/examples/aisimulate/matrix/experiment_parallel.yaml"
OUTPUT_DIR="$SWEEP_ROOT/results/matrix-formal-parallel"

exec "$SCRIPT_DIR/run_matrix_experiment.sh" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
