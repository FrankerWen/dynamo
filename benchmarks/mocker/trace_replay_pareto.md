Searched for "arrival-speedup-ratio"
Viewed main.py:750-920

DynoSim 在 `/workspace/hisim_align/dynamo` 中提供了完整的命令行 CLI（`python -m dynamo.replay`）、底层引擎配置项，以及 Python 优化扫描 API（`dynamo.profiler.utils.replay_optimize`）。

以下是 DynoSim 对 **Trace 回放** 与 **扫描（Sweep）** 的核心支持参数分类与详细解释：

---

### 一、 Trace 回放命令行参数 (`python -m dynamo.replay`)

命令行入口支持直接回放 Trace 文件或使用合成负载。主要参数分为以下几大类：

#### 1. Trace 文件与输入格式参数

| 参数名 | 类型 / 默认值 | 说明 |
|---|---|---|
| `trace_files` (位置参数) | 字符串 (文件路径) | 输入的 Trace 文件路径。 |
| `--trace-format` | `mooncake` \| `mooncake-delta` \| `agentic_mooncake` \| `applied_compute_agentic` \| `dynamo` (默认 `mooncake`) | Trace 文件格式。<br>• `mooncake`: 标准 Mooncake 格式。<br>• `dynamo`: 真实运行采样的原生格式（`dynamo.request.trace.v1`）。<br>• `agentic_mooncake`: 支持依赖图/分支的工作流格式。 |
| `--trace-block-size` | 整数 (默认 `512`) | **Trace 数据集粒度**：指定 Trace 文件中每个 `hash_id` 代表多少 Token（与引擎内部的 `block_size` 解耦）。 |

> **合成负载参数（不传入 Trace 文件时启用）**：
> - `--input-tokens`, `--output-tokens`, `--request-count`: 设定固定的输入/输出 Token 数与请求数（若 `--turns-per-session > 1` 则代表 Session 数量）。
> - `--request-rate`: 泊松开环请求到达速率（Req/s）。
> - `--turns-per-session`: 每 Session 的轮数（默认 1，大于 1 时启用多轮）。
> - `--shared-prefix-ratio` / `--num-prefix-groups`: 设定跨 Session 前缀共享比例及 Group 数量。
> - `--inter-turn-delay-ms`: 设定合成多轮会话内轮与轮之间的固定等待时间（ms）。

#### 2. 负载速率与时间轴控制参数

| 参数名 | 类型 / 默认值 | 说明 |
|---|---|---|
| `--arrival-speedup-ratio` | 浮点数 (默认 `1.0`) | **时间轴加速因子**：用于按比例压缩/拉伸 Session 首轮到达时间。在多轮 Trace 中，**仅按比例压缩 Session 间的首轮到达间隔，保持 Session 内部的 `delay_ms` 不变**。 |
| `--replay-concurrency` | 整数 (可选) | **闭环固定并发度模式**：忽略 Trace 第一轮的时间戳，强制保持固定数量的动态 Request/Session 在途运行。 |
| `--max-sim-time-seconds` | 浮点数 (可选) | **仿真限时终止**：设定虚拟时钟上限（秒），超时后立即停止仿真并计算统计报告。 |

#### 3. 运行模式与路由架构参数

| 参数名 | 类型 / 默认值 | 说明 |
|---|---|---|
| `--replay-mode` | `offline` \| `online` (默认 `offline`) | • `offline`: 纯逻辑时钟快仿真（无需 GPU、NATS 或 etcd）。<br>• `online`: 挂载真实 Mock Worker 执行网络/事件交互。 |
| `--router-mode` | `round_robin` \| `kv_router` (默认 `round_robin`) | • `round_robin`: 轮询路由。<br>• `kv_router`: 使能基于 KV Cache 意识的路由。 |
| `--num-workers` | 整数 (默认 `1`) | 聚合部署（Aggregated）模式下的 Worker 数量。 |
| `--num-prefill-workers` | 整数 (默认 `1`) | 解耦部署（Disaggregated）模式下的 Prefill Worker 节点数。 |
| `--num-decode-workers` | 整数 (默认 `1`) | 解耦部署（Disaggregated）模式下的 Decode Worker 节点数。 |

#### 4. 引擎配置 JSON (`--extra-engine-args` / `--prefill-engine-args` / `--decode-engine-args`)

通过传入 JSON 字符串定义 Simulated Engine 行为及 AIC 硬件模型（如 `--extra-engine-args '{"engine_type":"sglang", ...}'`）：

```json
{
  "engine_type": "sglang",       // 引擎类型: sglang / vllm / trtllm
  "block_size": 64,              // 引擎 KV 页粒度
  "enable_prefix_caching": true, // 是否使能 Radix Tree 前缀缓存
  "enable_chunked_prefill": true,// 是否开启分块 Prefill
  
  // === AIC 硬件性能模型配置 ===
  "aic_backend": "sglang",        // 使能 AIC, 需与 engine_type 一致
  "aic_system": "h100_sxm",       // 硬件 SKU: h100_pcie, h100_sxm, h200_sxm 等
  "aic_model_path": "meta-llama/Llama-3.1-70B-Instruct", // 模型名称/路径
  "aic_tp_size": 4,               // 张量并行 (TP) 规格
  "aic_moe_tp_size": 1,           // MoE TP 规格
  "aic_moe_ep_size": 4,           // MoE 专家并行 (EP) 规格
  "aic_attention_dp_size": 1,     // Attention DP 规格
  
  // === SGLang 专用调度参数 ===
  "sglang": {
    "schedule_policy": "fcfs",     // 调度策略: fcfs 或 lpm (Longest Prefix Match)
    "page_size": 64,               // 保持与 block_size 一致
    "max_prefill_tokens": 16384,   // 单 Batch 最大 Prefill Token 预算
    "chunked_prefill_size": 8192   // Chunk 大小
  }
}
```

#### 5. 报告输出与 Goodput SLA 参数

| 参数名 | 类型 / 默认值 | 说明 |
|---|---|---|
| `--report-json` | 字符串 (路径) | 指定生成的全量统计 JSON 报告输出路径。 |
| `--report-jsonl` | 字符串 (路径) | **逐请求明细**：逐行导出每个 Request 的详细时间戳、Prefix 命中 Token 数、ITL 序列及分布 Worker 索引（非常适合后续画 Scatter 散点图或分析 Worker 负载）。 |
| `--sla-ttft-ms` | 浮点数 | Goodput SLA：最大首 Token 延迟阈值（ms）。 |
| `--sla-itl-ms` | 浮点数 | Goodput SLA：最大平均 Token 间延迟阈值（ms）。 |
| `--sla-e2e-ms` | 浮点数 | Goodput SLA：最大端到端延迟阈值（ms）。设定 SLA 后报告会自动包含 `goodput_*` 指标。 |

---

### 二、 自动扫描与优化探索参数 (`ReplayOptimizeSpec`)

对于拓扑与资源配置扫频，Python API (`dynamo.profiler.utils.replay_optimize`) 接收 `ReplayOptimizeSpec` 结构体，支持在给定的 GPU 预算和 SLA 约束下自动探索设计空间：

```python
from dynamo.profiler.utils.replay_optimize import (
    ReplayOptimizeSpec, EngineSpec, HardwareSpec, WorkloadSpec, SLASpec, RouterSpec
)

spec = ReplayOptimizeSpec(
    # 1. 基础引擎定义
    engine=EngineSpec(
        model_path="Qwen/Qwen3-32B",
        backend="vllm",
        base_engine_args={"block_size": 512, "enable_prefix_caching": True}
    ),
    
    # 2. 硬件预算约束 (全自动扫频 TP/Worker 组合)
    hardware=HardwareSpec(
        gpu_sku="h200_sxm",
        total_gpus=16  # 设定总 GPU 预算上限
    ),
    
    # 3. 负载描述 (指定 Trace 文件或合成参数)
    workload=WorkloadSpec(
        trace_file="/path/to/trace.jsonl",
        arrival_speedup_ratio=1.0
    ),
    
    # 4. SLA 门限过滤 (剔除不符合 SLA 的非劣解)
    sla=SLASpec(
        ttft=500.0,       # Mean TTFT 门限 (ms)
        itl=20.0,         # Mean ITL 门限 (ms)
        e2e_latency=5000.0# 端到端延迟门限 (ms)
    ),
    
    # 5. Router 参数扫描空间
    router=RouterSpec(
        mode="kv_router",
        overlap_credits=[0.0, 0.5, 1.0],      # 扫描 KV 路由 Overlap 权重
        prefill_load_scales=[0.5, 1.0, 2.0]   # 扫描 Prefill 负载衰减因子
    ),
    
    # 6. 标的优化目标
    objective="throughput", # 优化目标: throughput / mean_ttft_ms / mean_e2e_latency_ms
    max_parallel_evals=4    # 并发调优任务数
)
```

扫描运行完成后，返回 `DenseReplayOptimizationResult` 包含：
- `result.best_feasible`: 满足所有 SLA 约束和 GPU 预算的最优配置。
- `result.evaluated_df`: 评估过的所有配置试算表（包含 TP 形状、Worker 划分配比、Throughput、TTFT、ITL 等全量数据，可直接用于绘制 Pareto 前沿图）。