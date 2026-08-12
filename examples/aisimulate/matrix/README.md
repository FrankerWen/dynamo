<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Catalog 驱动的矩阵搜索

硬件事实统一保存在 `catalogs/skus.yaml`。B300 SXM 和 GB300 的 AIC system、
compute NIC 以及 P/D 传输拓扑域只定义一次；候选可使用的 GPU 上限由 search
policy 决定。拓扑感知的 `compact` placement 把一个候选的全部 P/D worker 组成
一个 routing group，并按 group GPU 数自动解析带宽域：

```text
group <= gpus_per_node                         -> intra_node
有 rack 且 group <= gpus_per_rack              -> intra_rack
无 rack 且 group > gpus_per_node               -> inter_node
有 rack 且 group > gpus_per_rack               -> inter_rack
```

解析后的 domain、P/D GPU 数和实际 GB/s 保存在每个候选配置中。这个 compact
规则对允许任意 P worker 路由到任意 D worker 的 routing group 是保守的；它不会
把多个 rack 上彼此隔离、只做 rack-local handoff 的多组副本误判成同一个本地域。

带宽字段统一使用 `*_gbytes_per_s_*` 命名，单位是十进制 GB/s
（`1 GB/s = 1e9 byte/s`）、每 GPU、单方向。schema v1 仍接受旧的
`*_gbps_per_gpu` 输入名，但解析后的 catalog、manifest 和新报告只输出明确的新名称。
DynoSim binding 中历史遗留的 `_gbps` 参数名实际同样按 GB/s 计算。旧的显式
`compute_nic`、`local_fabric` 和 `explicit` PD profile 仍可解析；正式示例使用单个
`topology_aware_full_prompt` profile，避免把同一候选重复当作两种手工带宽场景。

`catalogs/models.yaml` 只保存模型结构事实；`catalogs/model_variants.yaml` 保存可部署
checkpoint 和完整的 AIC 精度 profile。每个 Cell 固定一个 variant，矩阵维度因此是
`workload × model × model_variant × SKU × SLA`。variant 显式给出 dense GEMM、MoE、
FMHA、KV cache 和通信五种精度；这些字段同时进入 AIC 时延、原生内存/KV 容量、KV
footprint 和 DynoSim replay。新增 FP4/FP8 variant 必须增加真实 checkpoint/profile
catalog 条目，不会从权重精度标签静默推断或回退。

`model_variant_selector` 出现时是显式模式：只运行声明项，不做任何替换；省略时由
`VariantResolver` 自动选择。每个模型/backend 必须恰有一个 `role: native`。原生配方
在 SKU 上受支持时立即短路，不比较其他精度；只有硬件精度或具体 kernel 不支持才检查
`role: fallback` 配方。显存不足、SLA 失败和普通 shape 数据缺失都不会触发换精度。
自动候选必须在 weight、activation、KV 三个维度均不降低精度，最多 8 项，超限直接
报错。`silicon` 模式只把所选 backend/version 自身的精确表认作 exact；`empirical`
允许 AIC hybrid/transfer，但会保留低可信来源标记。无法证明 checkpoint/runtime 匹配
的 AIC 枚举配方只进入 `hypothetical` what-if Pareto，不进入默认推荐。

模型 catalog 的 KV 布局事实不再保存一个无作用域的 `kv_bytes_per_token`，KV dtype 则
由 model variant 提供。当前模型均使用 `kv_footprint.source: aic_native`，Matrix 会针对
候选的 TP/attention-DP/EP、backend、variant KV dtype 和工作负载长度调用 AIC
`get_kvcache_bytes_per_sequence()`。解析结果的作用域是
`per_rank`：KVBM 的 `kv_bytes_per_token` 只表示 rank-local block 大小；P/D handoff
另用 `kv_transfer_bytes_per_token`。当 P/D 的 TP 不同时，它按 decode 端需要物化的
总字节、prefill 发送 rank 数和 decode 单 rank 接收量计算理想均衡 reshard 的最忙链路，
不会盲乘 TP，也不会漏掉 MLA replication 的广播 fan-out。物理网络用量单独按 decode
端实际物化总字节计量；共享 fabric 争用仍不在当前单带宽 replay 合约内。
schema v1 的裸 `kv_bytes_per_token` 仍可被解析以便迁移，但会标记为
`legacy_unspecified`；只要实验包含 P/D 或 G2，校验就会 fail closed，必须改成带
layout 和 scope 的结构化 footprint，并在 variant 中显式指定 KV dtype。

DeepSeek V4 的压缩/滑窗/indexer 曲线和 Kimi K3 的固定 KDA state 不能由一个全局线性
scalar 精确表达。对固定 ISL 的 Synthetic workload，`fixed_synthetic_average` 会保留
该长度下每请求的 AIC 原生总字节，并在候选中标注 KVBM per-block timing 为近似；对
异构 trace 或 `destination_missing` 非线性传输仍 fail closed。`capacity_gib` 默认是
整个候选 deployment 的物理 G2 总容量，运行时会根据 TP、实例数、block size 和原生
per-rank footprint 转为每个 DP-rank scheduler 的逻辑 block 数；物理池数量是
`TP × attention-DP × replicas`，不会漏算 attention-DP 复制出来的 rank-local 池。

- `policies/vllm_pd_g2_preliminary.yaml`：16 GPU 初步搜索。
- `policies/vllm_pd_g2.yaml`：72 GPU 正式搜索。
- `policies/vllm_pd_g2_parallel16.yaml`：72 GPU、每轮 16 个并行 trial 的
  focused 并发验证。

`experiment_parallel.yaml` 在完整矩阵上使用 `cell_workers: 4` 和 parallel16 policy；
它写入独立的 `matrix-formal-parallel` 结果目录，不会混入或覆盖串行 formal 数据库。

价格不属于实验输入，也不进入 Cell fingerprint。仿真结束后，`report` 命令按
`sku_id` 加载价格 catalog；因此更换价格不需要重跑仿真。

## 可重复正式运行

默认脚本执行本地 linkage audit、配置校验、Variant 解析、矩阵展开、断点续跑和最终报告：

```bash
aisimulate/scripts/run_matrix_experiment.sh \
  --output-dir /workspace/dynosim_aic_sweep/results/matrix-formal \
  --price-catalog examples/aisimulate/matrix/catalogs/prices_public_preliminary.yaml
```

正式和初步 policy 都固定使用 `QUASI_RANDOM_SEARCH` 与 `search_seed=20260810`。
seed、policy 内容和 simulator revision 都进入 Cell fingerprint。同一份代码、AIC
数据库、binding、输入和配置可产生相同的候选建议序列；运行 manifest 记录这些来源。

跨实验只复用完全相同的 replay 计算。使用 `--reuse-from` 导入其他数据库的
`evaluation_cache_key`；该 key 不包含搜索轮数、GPU budget 或价格，但包含 workload
内容、负载/seed、SLA、解析后的部署计划以及 simulator revision。旧数据库若没有
精确 cache 表会被报告为 legacy source，不会做不安全的猜测式复用。
model variant 修正后的示例 revision 是 `*-2026-08-11-model-variant-v3`，cache contract
为 v3；resolver 来源、native/fallback 标签和 variant ID 不进入 replay key，完整
checkpoint、AIC 精度、backend/system 和最终 deployment plan 才是计算语义。旧
revision 数据库仍可保留和查询，但 resume/`--reuse-from` 不会把 checkpoint 默认精度
或旧 KV 标量语义当成 v3 命中。

Variant 支持探测另有 `variant_support_cache_key`，包含 checkpoint/config、具体 quant
profile、SKU/AIC system、backend/version、AIC 数据库哈希、data mode 和 resolver
版本；价格、SLA 和 Spica 轮数不参与。`--reuse-from` 会同时导入匹配的支持探测缓存和
replay 缓存。

常用脚本模式：

```bash
# 只校验并估算搜索规模
aisimulate/scripts/run_matrix_experiment.sh --dry-run

# 单 Cell 冒烟
aisimulate/scripts/run_matrix_experiment.sh --smoke

# 从另一轮导入精确 replay cache
aisimulate/scripts/run_matrix_experiment.sh \
  --reuse-from /workspace/dynosim_aic_sweep/results/another-run

# 对既有资源量重新定价和生成五类图表/中文报告
aisimulate/scripts/run_matrix_experiment.sh \
  --report-only \
  --output-dir /workspace/dynosim_aic_sweep/results/matrix-formal \
  --price-catalog /path/to/versioned-prices.yaml

# CPU 测试和 smoke 配置展开
aisimulate/scripts/test_matrix_framework.sh

# GB300 × DeepSeek V4 Pro × ISL 8K/OSL 1K、无 prefix：
# AIC 并发点发现 → 分负载 Spica Pareto → 入围者开放环 RPS 认证
aisimulate/scripts/run_parallel_focus.sh

# 搜索结束后校验数据库完整性、Pareto/五维 CSV、七张图和中文报告
aisimulate/scripts/verify_parallel_focus.sh

# 完整 20-Cell、4 个 Cell 并行、每 Cell 16-trial 并发；可从既有 formal 数据库导入精确 replay cache
aisimulate/scripts/run_matrix_parallel.sh \
  --reuse-from /workspace/dynosim_aic_sweep/results/matrix-formal
```

`--no-resume` 只允许用于没有 `sweep.duckdb` 的新输出目录，防止将两次完整运行
写进同一个数据库。默认使用 `--resume`，只跳过 key 完全匹配的成功 evaluation。

## 直接使用 CLI

不运行 replay，只校验并查看矩阵：

```bash
python -m aisimulate.matrix validate \
  --config examples/aisimulate/matrix/experiment.yaml
python -m aisimulate.matrix resolve-variants \
  --config examples/aisimulate/matrix/experiment.yaml
python -m aisimulate.matrix expand \
  --config examples/aisimulate/matrix/experiment.yaml
```

运行或恢复实验，然后对保存的资源量定价：

```bash
python -m aisimulate.matrix run \
  --config examples/aisimulate/matrix/experiment.yaml \
  --output-dir results/matrix-study \
  --resume
python -m aisimulate.matrix report \
  --run-dir results/matrix-study \
  --price-catalog examples/aisimulate/matrix/catalogs/prices.yaml
```

仓库内的 `prices.yaml` 使用相对单位；生成采购或预算结论前，应替换为版本化 TCO
catalog。价格 catalog 必须覆盖运行数据库中的全部 SKU，否则报告阶段 fail closed。

`workloads_parallel_focus.yaml` 没有手写 `[0.5, 2, 8, 32]` 一类 RPS 锚点。
`concurrency_sweep.source: aic_operating_points` 先调用 AIC 的
`parallel × batch/context` 静态 sweep，在 SLA 和 GPU 上限内抽取静态 Pareto 与每种
并行配置 TopK 的 concurrency。每个选中 concurrency 是独立的闭环 Spica 研究，结果
同时保存分负载 Pareto 和跨负载 global operating envelope；搜索结束后只对每个负载
前沿的去重入围配置做开放环 RPS 双向扩张/二分、正式 SLA 和稳健性复验。AIC 点只负责
确定发现负载，正式候选仍由 DynoSim replay 判定。

`expand` 输出确定性的 Cell 清单，以及 substudy、静态候选、Spica trial 和稳健性
复验的规模上限。动态 AIC 并发点在 `expand` 中显示的是 `max_points` 配置上限，实际点数
会写入 `aic_operating_points.csv`。正式运行前应先检查该估算。Matrix Cell 默认串行，只有宿主机能承受
Cell 线程与 Cell 内 trial 进程两层并发时，才提高 `run.cell_workers`。

总 trial 上限是 `max_rounds × candidates_per_round`，不是只由 `max_rounds`
决定。正式 policy 的 `20 × 8` 和 focused policy 的 `10 × 16` 都是每个
substudy 160 个 trial；后者用于隔离验证并发提升，不会因减少样本数而人为缩短耗时。
只有一个 Cell 时，`cell_workers > 1` 不产生加速。多 Cell 运行的近似进程上限为
`cell_workers × parallel_evals`，还应为主进程、AIC 内部线程和结果提交保留余量。

矩阵脚本默认设置 `DYN_RUNTIME_NUM_WORKER_THREADS=1`、
`DYN_RUNTIME_MAX_BLOCKING_THREADS=4` 以及常见数值库单线程环境变量。这样可避免每个
spawn replay worker 都创建宿主机核数规模的 Tokio/DuckDB 线程池；所有值都可通过
同名环境变量显式覆盖。DuckDB 只在拥有结果库的主进程中按需导入。脚本还默认设置
`DYN_LOG=error`；Dynamo 使用该变量而不是 `RUST_LOG` 控制 tracing 过滤，正式 sweep
由此避免每请求 INFO 日志占满磁盘。

结果目录采用 append-first `events.jsonl`，并包含 `sweep.duckdb`、来源
`manifest.json`、原始/历史 Parquet、严格与 epsilon Pareto、独立 TopK、Pareto CSV、
`aic_operating_points.csv`、`load_conditioned_pareto.csv`、
`global_operating_envelope.csv`、`capacity_certification.csv`、
五类趋势 CSV、一张最终 Pareto 图、一张分负载 Pareto 图、五张趋势图和中文 Markdown 报告。默认报告只使用
`deployable` 候选，另输出 `what_if_pareto_front.csv`；精度选择、淘汰原因和可信度写入
`variant_resolution.csv` 与中文 `variant_resolution.md`。

文件 workload 按内容计算哈希。`agentic_mooncake_trace` 保留依赖和 think time，并
搜索 `arrival_speedup_ratio`；`applied_compute_agentic_trace` 要求
`replay_concurrency`。未知 AIC 模型、系统或不支持的 replay 组合会 fail closed。
