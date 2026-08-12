#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Reproducible validation, execution, resume, and report workflow for matrix studies.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
SWEEP_ROOT="$(cd -- "$REPO_ROOT/.." && pwd -P)"

DEFAULT_CONFIG="$REPO_ROOT/examples/aisimulate/matrix/experiment.yaml"
SMOKE_CONFIG="$REPO_ROOT/examples/aisimulate/matrix/experiment_preliminary_smoke.yaml"
DEFAULT_PRICE_CATALOG="$REPO_ROOT/examples/aisimulate/matrix/catalogs/prices_public_preliminary.yaml"
DEFAULT_OUTPUT_DIR="$SWEEP_ROOT/results/matrix-formal"
DEFAULT_SMOKE_OUTPUT_DIR="$SWEEP_ROOT/results/matrix-formal-smoke"

CONFIG="$DEFAULT_CONFIG"
PRICE_CATALOG="$DEFAULT_PRICE_CATALOG"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
OUTPUT_EXPLICIT=0
DRY_RUN=0
REPORT_ONLY=0
SKIP_REPORT=0
SKIP_LINKAGE_AUDIT=0
RESUME=1
REUSE_FROM=()

usage() {
    cat <<'EOF'
用法：run_matrix_experiment.sh [选项]

默认执行正式 72-GPU policy：linkage audit → validate → resolve variants → expand → run/resume → report。

选项：
  --config PATH          实验配置；默认 experiment.yaml
  --output-dir PATH      结果目录；默认 ../results/matrix-formal
  --price-catalog PATH   仅在 report 阶段加载的价格 catalog
  --smoke                使用单 Cell smoke 配置
  --dry-run              只执行 validate、variant 解析和 expand
  --report-only          不运行仿真，只对既有数据库重新定价并生成报告
  --skip-report          完成仿真后不生成价格报告
  --no-resume            不复用成功 evaluation；要求输出目录中不存在 sweep.duckdb
  --reuse-from PATH      从其他 run 导入精确 replay cache；可重复指定
  --skip-linkage-audit   跳过本地 Dynamo/AIC linkage audit
  -h, --help             显示帮助

环境变量：
  MATRIX_PYTHON          Python 解释器；默认优先使用工作区 .venv
  MATRIX_LINKAGE_AUDIT   linkage audit 脚本路径
  DYN_RUNTIME_NUM_WORKER_THREADS  每个 replay 进程的 Tokio worker；默认 1
  DYN_RUNTIME_MAX_BLOCKING_THREADS  每个 replay 进程的 blocking 上限；默认 4
  DYN_LOG                Dynamo tracing 过滤；默认 error
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            OUTPUT_EXPLICIT=1
            shift 2
            ;;
        --price-catalog)
            PRICE_CATALOG="$2"
            shift 2
            ;;
        --smoke)
            CONFIG="$SMOKE_CONFIG"
            if [[ "$OUTPUT_EXPLICIT" -eq 0 ]]; then
                OUTPUT_DIR="$DEFAULT_SMOKE_OUTPUT_DIR"
            fi
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --report-only)
            REPORT_ONLY=1
            shift
            ;;
        --skip-report)
            SKIP_REPORT=1
            shift
            ;;
        --no-resume)
            RESUME=0
            shift
            ;;
        --reuse-from)
            REUSE_FROM+=("$(realpath -m -- "$2")")
            shift 2
            ;;
        --skip-linkage-audit)
            SKIP_LINKAGE_AUDIT=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数：$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "${MATRIX_PYTHON:-}" ]]; then
    PYTHON_BIN="$MATRIX_PYTHON"
elif [[ -x "$SWEEP_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$SWEEP_ROOT/.venv/bin/python"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi

CONFIG="$(realpath -m -- "$CONFIG")"
OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"
PRICE_CATALOG="$(realpath -m -- "$PRICE_CATALOG")"
export PYTHONPATH="$REPO_ROOT/aisimulate/src${PYTHONPATH:+:$PYTHONPATH}"

# Matrix replay workers are separate processes.  The Dynamo Python extension
# otherwise creates one Tokio worker per host CPU in every process, which turns
# e.g. 16 parallel evaluations on a 192-core host into thousands of mostly idle
# threads.  Keep the offline-sweep default deliberately small; callers can
# override every value when benchmarking a different host/runtime.
export DYN_RUNTIME_NUM_WORKER_THREADS="${DYN_RUNTIME_NUM_WORKER_THREADS:-1}"
export DYN_RUNTIME_MAX_BLOCKING_THREADS="${DYN_RUNTIME_MAX_BLOCKING_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-1}"
export RUST_LOG="${RUST_LOG:-error}"
export DYN_LOG="${DYN_LOG:-error}"

if [[ "$REPORT_ONLY" -eq 1 && "$DRY_RUN" -eq 1 ]]; then
    echo "--report-only 与 --dry-run 不能同时使用" >&2
    exit 2
fi
if [[ "$REPORT_ONLY" -eq 1 && "$SKIP_REPORT" -eq 1 ]]; then
    echo "--report-only 与 --skip-report 不能同时使用" >&2
    exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "实验配置不存在：$CONFIG" >&2
    exit 2
fi
if [[ "$SKIP_REPORT" -eq 0 && ! -f "$PRICE_CATALOG" ]]; then
    echo "价格 catalog 不存在：$PRICE_CATALOG" >&2
    exit 2
fi
if [[ "$RESUME" -eq 0 && -f "$OUTPUT_DIR/sweep.duckdb" ]]; then
    echo "--no-resume 要求使用新的输出目录；已存在：$OUTPUT_DIR/sweep.duckdb" >&2
    exit 2
fi

mkdir -p -- "$OUTPUT_DIR"

echo "Python:       $PYTHON_BIN"
echo "Config:       $CONFIG"
echo "Output:       $OUTPUT_DIR"
echo "Price report: $PRICE_CATALOG"
echo "Replay runtime: Tokio workers=$DYN_RUNTIME_NUM_WORKER_THREADS, blocking=$DYN_RUNTIME_MAX_BLOCKING_THREADS, OMP=$OMP_NUM_THREADS"
if [[ "$PRICE_CATALOG" == "$DEFAULT_PRICE_CATALOG" && "$SKIP_REPORT" -eq 0 ]]; then
    echo "提示：默认价格 catalog 中 GB300 GPU 单价是 B300 的规划代理值，不是正式报价。"
fi

if [[ "$REPORT_ONLY" -eq 0 ]]; then
    "$PYTHON_BIN" -m aisimulate.matrix validate --config "$CONFIG" \
        | tee "$OUTPUT_DIR/validation.json"
    "$PYTHON_BIN" -m aisimulate.matrix resolve-variants --config "$CONFIG" \
        | tee "$OUTPUT_DIR/variant_resolution.json"
    "$PYTHON_BIN" -m aisimulate.matrix expand --config "$CONFIG" \
        | tee "$OUTPUT_DIR/expansion.json"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "dry-run 完成；未启动仿真。"
        exit 0
    fi

    if [[ "$SKIP_LINKAGE_AUDIT" -eq 0 ]]; then
        AUDIT_SCRIPT="${MATRIX_LINKAGE_AUDIT:-$SWEEP_ROOT/dynosim-aic-dev/scripts/audit_local_linkage.py}"
        if [[ ! -f "$AUDIT_SCRIPT" ]]; then
            echo "找不到 linkage audit：$AUDIT_SCRIPT" >&2
            echo "可显式传入 MATRIX_LINKAGE_AUDIT，或使用 --skip-linkage-audit。" >&2
            exit 2
        fi
        "$PYTHON_BIN" "$AUDIT_SCRIPT" --root "$SWEEP_ROOT" --json \
            | tee "$OUTPUT_DIR/linkage_audit.json"
    fi

    RESUME_FLAG="--resume"
    if [[ "$RESUME" -eq 0 ]]; then
        RESUME_FLAG="--no-resume"
    fi
    "$PYTHON_BIN" -m aisimulate.matrix run \
        --config "$CONFIG" \
        --output-dir "$OUTPUT_DIR" \
        "${REUSE_FROM[@]/#/--reuse-from=}" \
        "$RESUME_FLAG" 2>&1 | tee -a "$OUTPUT_DIR/run.log"
fi

if [[ "$SKIP_REPORT" -eq 0 ]]; then
    if [[ ! -f "$OUTPUT_DIR/sweep.duckdb" ]]; then
        echo "报告所需数据库不存在：$OUTPUT_DIR/sweep.duckdb" >&2
        exit 2
    fi
    "$PYTHON_BIN" -m aisimulate.matrix report \
        --run-dir "$OUTPUT_DIR" \
        --price-catalog "$PRICE_CATALOG" \
        2>&1 | tee "$OUTPUT_DIR/report_summary.json"
fi

echo "完成。结果目录：$OUTPUT_DIR"
