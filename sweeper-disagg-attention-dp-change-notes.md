# Sweeper 简化版 Disaggregated Attention-DP 修改说明

## 1. 文档目的

本文说明 `sweeper-disagg-attention-dp.patch` 中的实现范围、设计选择、配置方式、验证结果和已知限制。

这不是 Dynamo 对 disaggregated attention-DP 的完整通用支持，而是面向离线 Sweeper/Mocker 的受限实现：

- Prefill 和 Decode 可以使用不同的 attention-DP 大小；
- P、D 两个资源池分别独立 round-robin；
- worker 数量固定，不支持 Planner 动态 scaling；
- 不支持 KV Router 或 Router adapter；
- 不扩展取消竞态语义；
- 沿用现有的每请求标量 P→D KV 传输延迟模型。

## 2. Patch 信息和基线

- Patch：`sweeper-disagg-attention-dp.patch`
- Patch 大小：约 52 KiB
- 修改范围：17 个文件，595 行新增、119 行删除
- SHA256：`52b396f6a9cf7b27e5a72885926c53712eaa213c9b328152f88d9ac54eca18d9`
- 生成时仓库 HEAD：`f4625cde5c16c976fc09626b67674bff6a788446`

该 patch 是在当前工作区已有的 Sweeper/AIC attention-DP 基础上增量生成的，并主动排除了 ANALYTICAL、load sweep、checkpoint 和 Pareto 等无关改动。因此：

- 当前工作树已经包含这些修改，不应再次应用；
- patch 适用于“本功能修改前、但已包含当前既有 Sweeper/AIC 基础改动”的相同基线；
- 若直接应用到纯净官方 HEAD，Sweeper 部分可能因缺少先行的 `aic_attention_dp_size` 接线而需要手工 rebase。

应用前应先检查：

```bash
git apply --check sweeper-disagg-attention-dp.patch
git apply sweeper-disagg-attention-dp.patch
```

## 3. 支持边界

| 能力 | 当前行为 |
| --- | --- |
| P/D attention-DP 大小不同 | 支持，例如 P=2、D=4 |
| P/D replica 数不同 | 沿用现有 disagg 配置，支持 |
| P/D 调度 | 两个池分别独立 round-robin |
| 动态 scaling | attention-DP 模式明确拒绝 |
| Planner hook | attention-DP 模式明确拒绝 |
| KV Router/Router hook | attention-DP 模式明确拒绝 |
| P→D KV 传输 | 沿用每请求标量延迟 |
| 链路共享和拥塞 | 不支持 |
| NIXL/RDMA/拓扑 | 不支持 |
| 取消竞态扩展 | 不在本次范围内 |
| 后端 | 沿用现有 disagg 限制：vLLM 或 SGLang，P/D 后端必须一致 |
| TRT-LLM disagg | 仍不支持 |

只有当 Prefill 或 Decode 任一侧的 attention-DP 大于 1 时，新增的受限模式门禁才生效。P=1、D=1 的既有 disagg 行为保持不变。

## 4. 核心设计

### 4.1 每个资源池独立展开 DP rank

原有 `EngineComponent::new_ranked` 已能把一个逻辑 worker 展开为多个 rank scheduler。本次让 disagg runtime 分别使用 Prefill 和 Decode 的 engine args 构造两个 ranked engine：

```text
Prefill schedulers = num_prefill_workers × prefill_dp_size
Decode schedulers  = num_decode_workers  × decode_dp_size
```

因此 P、D 的 attention-DP 不需要相等。两个 pool 分别维护自己的 active topology 和 round-robin 游标。

新增 `new_ranked_with_capture` 是为了在 ranked engine 下继续保留 disagg 既有的 pool-specific raw-event capture 策略；它不改变调度算法。

### 4.2 handoff 保存完整 endpoint 身份

每个请求在 Prefill 和 Decode 两侧分别保存：

```text
DisaggEndpoint {
    scheduler_id,
    logical_worker_id,
    dp_rank,
}
```

三个字段的职责不同：

- `scheduler_id`：资源池内部实际 scheduler 索引，用于后续 activate、release 和 cleanup；
- `logical_worker_id`：对外报告中的 worker/replica 身份；
- `dp_rank`：该逻辑 worker 内的 attention-DP rank。

这样可以避免把 flattened scheduler ID 错当成逻辑 worker ID，也确保 P→D handoff 完成后，后续动作仍回到最初选中的 rank。

逐请求报告继续保持兼容：

- `prefill_worker_idx`、`decode_worker_idx` 仍记录逻辑 worker；
- rank 级信息记录在 `routing_history` 的 `scheduler_id` 和 `dp_rank` 中。

### 4.3 独立 round-robin

当前简单模式不引入跨 P/D 的联合放置。一个请求的路径为：

```text
请求到达
  ├─ Prefill pool round-robin 选择 P scheduler/rank
  ├─ Prefill 完成并计算标量 P→D 延迟
  ├─ Decode pool round-robin 独立选择 D scheduler/rank
  └─ 在原 D scheduler/rank 激活并执行 Decode
```

例如一个逻辑 P worker、P-DP=2，以及一个逻辑 D worker、D-DP=4 时，连续请求的 rank 序列为：

```text
P: 0, 1, 0, 1, 0, 1, ...
D: 0, 1, 2, 3, 0, 1, ...
```

P rank 和 D rank 没有绑定关系，这是本次“最简单算法”的预期行为。

### 4.4 fail-closed 支持门禁

Mocker 公共离线 disagg entrypoint 在任一侧 DP>1 时执行以下检查：

- `router_mode` 必须为 `round_robin`；
- `scaling_policy` 必须为空。

Sweeper 的 Dynamo runner 进一步拒绝：

- `dynamo.planner/scaling_policy` hook；
- `dynamo.router/placement_policy` hook。

因此不支持的 Planner/Router 组合会在运行前报错，而不是静默退化为未经验证的行为。

## 5. Sweeper 配置和参数传递

### 5.1 非对称 P/D attention-DP 示例

可通过 pinned `parallel_configs` 明确给出不同的 P/D shape：

```yaml
search_space:
  deployment_mode: [disagg]
  backend: [sglang]

  parallel_configs:
    - prefill:
        tp: 1
        attention_dp: 2
        moe_tp: 1
        moe_ep: 2
        replicas: 1
      decode:
        tp: 1
        attention_dp: 4
        moe_tp: 1
        moe_ep: 4
        replicas: 1

  model_name: example/model
  hardware_sku: b200_sxm
  gpu_budget: 6
```

Sweeper 的传递链路为：

```text
parallel_configs
  → unrolled sample: prefill_attention_dp/decode_attention_dp
  → BackendDeploymentSpec
  → P/D MockEngineArgs.aic_attention_dp_size
  → binding topology reconciliation
  → Mocker P/D dp_size
```

### 5.2 P→D 标量传输配置

新增的 `pd_transfer` 是固定部署假设，不是 optimizer 搜索维度：

```yaml
search_space:
  deployment_mode: [disagg]

  pd_transfer:
    kv_bytes_per_token: 131072
    bandwidth_gb_s: 64.0
    timing_mode: full_prompt
```

字段含义：

| 字段 | 约束 | 说明 |
| --- | --- | --- |
| `kv_bytes_per_token` | 正整数 | 每个被计费 token 的 KV 字节数 |
| `bandwidth_gb_s` | 正数 | 每请求有效传输带宽，单位 GB/s |
| `timing_mode` | `full_prompt` 或 `destination_missing` | 被计费 token 的选择方式 |

延迟公式为：

```text
delay_ms = charged_tokens × kv_bytes_per_token
           / (bandwidth_gb_s × 1e9) × 1000
```

- `full_prompt`：按完整 prompt token 数计费；
- `destination_missing`：只按 Decode 目标侧缺失的 prompt footprint 计费；
- 未配置 `pd_transfer`：不增加 P→D 传输延迟。

Sweeper 会把相同的三个底层字段写入 P/D engine args：

```text
kv_bytes_per_token
kv_transfer_bandwidth
kv_transfer_timing_mode
```

当前 handoff 延迟仍由既有标量模型计算。配置到两侧是为了部署描述一致，不表示同时计费两次；验证确认每请求只产生一次 P→D 延迟。

### 5.3 模型精度边界

该带宽是每请求可独占的有效线速。当前模型不表达：

- 多请求共享链路后的 processor sharing；
- P/D GPU 数或 rank 数对总带宽的拆分；
- NIC 数量、NVLink、PCIe、RDMA、NIXL 或跨节点拓扑；
- 排队、拥塞、重传或流控；
- 多路径和不同 P-rank/D-rank 之间的带宽差异。

对于 MLA 或注意力稀疏模型，`kv_bytes_per_token` 应按目标序列长度和实际 KV layout 校准，不能直接套用普通 GQA 的理论值。

## 6. 文件级修改

### 6.1 Mocker/Rust runtime

- `lib/mocker/src/replay/offline/components/engine.rs`
  - 新增带显式 capture 策略的 ranked engine 构造函数。
- `lib/mocker/src/replay/offline/state.rs`
  - 新增 `DisaggEndpoint`，请求状态保存 P/D 完整 endpoint。
- `lib/mocker/src/replay/offline/disagg.rs`
  - P/D 分别构造 ranked engine；handoff 操作按 endpoint scheduler 执行。
- `lib/mocker/src/replay/validate.rs`
  - 放开非对称 DP；DP>1 时仅允许 round-robin 且拒绝 scaling。
- `lib/mocker/src/replay/entrypoints.rs`
  - 所有离线 disagg entrypoint 把 scaling 是否启用传入验证层。
- `lib/mocker/src/replay/offline/disagg_tests.rs`
  - 增加 vLLM/SGLang DP2/DP4、rank identity 和传输延迟测试。

### 6.2 Sweeper/Python

- `aisimulate/src/aisimulate/sweeper/config.py`
  - 新增 `PdTransferConfig`、正值校验和 disagg-only 校验。
- `aisimulate/src/aisimulate/sweeper/sample.py`
  - 将固定 `pd_transfer` 放入 disagg unrolled sample。
- `aisimulate/src/aisimulate/sweeper/deploy.py`
  - 将标量传输字段注入 P/D engine args。
- `aisimulate/src/aisimulate/sweeper/__init__.py`
  - 导出 `PdTransferConfig`。
- `components/src/dynamo/replay/simulation.py`
  - 对 disagg attention-DP 增加 fixed round-robin/no-hook 门禁。

### 6.3 测试和文档

- `aisimulate/tests/sweeper/test_config.py`
- `aisimulate/tests/sweeper/test_sample.py`
- `aisimulate/tests/sweeper/test_deploy.py`
- `components/src/dynamo/replay/tests/test_simulation.py`
- `aisimulate/docs/sweeper/configuration.md`
- `docs/fern/pages/developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/configuration.md`

## 7. 验证结果

### 7.1 Rust

- `cargo test -p dynamo-mocker --lib`
  - 728 passed，0 failed；
- `cargo clippy -p dynamo-mocker --lib --tests -- -D warnings`
  - 通过；
- vLLM/SGLang 非对称 attention-DP 定向测试
  - 两个后端均通过；
- 既有 disagg 测试组
  - 40 passed。

### 7.2 Python/Sweeper

- Sweeper config/sample/deploy：97 passed；
- Dynamo replay runner：11 passed；
- 合计：108 passed。

当前环境的项目级 pytest 配置引用了未安装的 `pytest_benchmark`，因此上述 Python 测试通过隔离 pytest 配置运行；没有测试失败。

### 7.3 原生 Python→Rust 回放

重建 PyO3 binding 后执行 8 请求、P-DP=2、D-DP=4 的真实离线回放，结果为：

```text
completed: 8/8
prefill_dp_ranks: [0, 1, 0, 1, 0, 1, 0, 1]
decode_dp_ranks:  [0, 1, 2, 3, 0, 1, 2, 3]
transfer_ms:      [64, 64, 64, 64, 64, 64, 64, 64]
```

测试参数为 64 prompt tokens、`kv_bytes_per_token=1,000,000`、`bandwidth_gb_s=1.0`，因此理论值和仿真值均为 64 ms。

### 7.4 Patch 完整性

已执行：

```bash
git apply --reverse --check sweeper-disagg-attention-dp.patch
```

检查通过，说明该 patch 与当前工作树中已实现的功能差异相匹配。

## 8. 已知限制和后续扩展

本次没有实现以下能力：

1. Planner 动态增减 P/D worker 或 DP rank；
2. KV-aware、load-aware 或 topology-aware 的 P/D 放置；
3. cancellation 与 scaling/handoff 并发时的新状态机语义；
4. 共享链路拥塞和带宽竞争；
5. rank-to-rank 物理网络拓扑；
6. 不同 P/D endpoint 的异构带宽；
7. 在线 replay 或真实 Dynamo serving 路径的 disagg attention-DP 认证。

如果后续需要完整支持，建议按以下顺序扩展：

1. 先定义 rank-aware placement 和稳定 endpoint 生命周期接口；
2. 再定义 scaling 时 logical worker/rank 的 drain、handoff 和 cleanup 语义；
3. 补齐 cancellation 与 scaling 交错的状态机测试；
4. 最后引入共享链路或 topology-aware transfer model，并将其与 KVBM tier bandwidth 明确分层。

在完成这些工作前，本 patch 应继续被描述为“离线 Sweeper 的固定规模、独立 round-robin、标量传输延迟版 disagg attention-DP”。
