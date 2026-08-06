# DynoSim 与 HiSim+AIC 对齐修复记录

本文按“修复动机、修改内容、验证方法、修复效果”记录 DynoSim 的 prefix admission、P0、P1、P2 改进。

测试范围为 TP4+EP4、H100 PCIe、DeepSeek-V3 20-layer。所有结果均复用既有 SGLang/HiSim artifacts，通过 CPU/offline replay 得到，没有重新采集 GPU 数据。本文中的 MAPE 是 DynoSim 离线诊断指标，不是新的 HiSim `acc_bench/compare` formal ACC；DynoSim formal ACC 为 `N/A`。

P3 按约定暂缓。

## 1. 修复前的 AIC 调用机制判断

HiSim 和 DynoSim 都按调度 batch 查询 AIC，并将 batch 内请求长度归约为平均值：

- HiSim Decode 使用 `mean(past_kv_length)`；Prefill 使用 `mean(past_kv_length)` 和 `mean(input_length)`。
- DynoSim Prefill 使用 admission batch 的 `mean_isl`、`mean_prefix`；Decode 使用 running batch 的平均 materialized context。
- 两者都将 logical decode batch 向上 padding 到最小可容纳的 CUDA graph capture size。

因此，“batch-level 查询”和“平均长度”不是旧 decode 偏差的根因。真正需要处理的是 prefix/chunk 顺序、AIC database provenance、dtype/MLA graph/layer mix contract，以及首 token/KV 生命周期。

`workload_distribution=balanced` 只是 AIC workload assumption，不能作为 DynoSim 执行了 SGLang `EP_DISPATCH_ALGORITHM=fake` TopK 的证据。

## 2. 前置修复：完整 prefix 必须在 chunk budget 前匹配

### 2.1 修复动机

B 和 B-mixed 的 prompt 长 32,768 tokens，共享前缀长 29,491 tokens，chunked prefill size 为 8,192。

旧 `get_new_batch_prefill` 先把请求截断到当前 8,192-token chunk，再将截断序列交给 radix cache 做 prefix match。这样首次 admission 最多只能看到 8,192-token 前缀，后续本应命中的 21,299 个 prefix tokens 被重复当作 fresh prefill，造成：

- prefill/KV budget 被错误扣减；
- 单 aggregated worker 形成不真实的长队列；
- B/B-mixed TTFT 和 E2E 大幅膨胀；
- chunk continuation 重复发 admission，并覆盖首次 reuse 结果。

这个问题发生在 scheduler admission 层，不能通过校准 AIC decode latency 解决。

### 2.2 修改内容

主要修改位于 `lib/mocker/src/scheduler/sglang/prefill.rs`：

1. 首次 admission 先对完整 `current_sequence` 做 radix-prefix match。
2. 只对 `sequence_len - prefix_len` 的 fresh suffix 应用 chunk、KV 和 prefill budget。
3. 在 eviction/allocation 期间锁住已经匹配的 radix path，防止匹配结果失效。
4. continuation 沿现有 KV lease 扩展，不重新做逻辑 admission。
5. `tokens_computed`、`prefix_tokens` 和 budget 扣减使用完整-prefix match 的实际结果。
6. 保留 page/KV capacity 规则，没有通过放宽 `output_reserve` 凑低 MAPE。

同时新增回归测试：

- 完整共享前缀必须在 chunk 前命中；
- chunk budget 只扣 fresh suffix；
- chunk continuation 不得重复发 admission。

### 2.3 验证方法

B-mixed 每点有 8 个 prefix group，每组 12 个请求。7 个 RPS 点均满足：

- 8 个 group leader 复用 0 token；
- 88 个 warm request 各复用 29,491 tokens；
- 首次 admission reuse ratio 为 `0.8249944051106771`；
- 7/7 点、672/672 请求完成；
- 无 NaN、负 latency 或 AIC database miss。

B-mixed RPS2 的白盒时间线也验证了 queue 变化来自 prefix admission，而不是 ingress 或 AIC 缺表。

### 2.4 修复效果

独立 prefix 修复结果：

| 范围 | 修复前 DynoSim | 修复后 DynoSim | MAPE 变化 |
|---|---:|---:|---:|
| B 35 cells | 877.076257% | 94.396026% | -782.680231 pp |
| B-mixed 35 cells | 447.532434% | 74.881012% | -372.651422 pp |
| 合并 175 cells | 287.445373% | 56.379043% | -231.066331 pp |

B-mixed RPS2：

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 平均 scheduler queue | 14,720.956 ms | 267.517 ms |
| TTFT | 15,628.007 ms | 535.218 ms |
| TTFT MAPE | 2,640.100309% | 6.158878% |

修复后全局最大误差转移到 B-RPS4 `mean_itl_ms=335.790934%`。这说明 prefix/chunk 根因已经关闭，下一层瓶颈是 decode service time、CUDA graph contract 和 scheduler lifecycle。

## 3. P0：AIC 数据库与 provenance 修复

### 3.1 修复动机

prefix 修复后，旧分析把约 50 ms 的 A-32 decode 预测归因于“DynoSim 没有使用 HiSim recorded overlay”。在修改数据库或算子模型前，必须先确认两边底层数据是否一致。

同时存在 provenance 缺口：

- DynoSim artifact 没有完整记录 resolved systems root、system YAML 和逐文件 hash。
- 历史 HiSim formal artifact 只记录 `database_path=null`。
- HiSim debug replay `events.jsonl` 不能用于填充 formal ACC 未记录字段。

### 3.2 修改内容

主要修改位于 `benchmarks/mocker/dynosim_hisim_alignment.py` 和报告脚本：

- 新增 `aic_database_manifest()`，记录 resolved systems root、system YAML、全部数据文件 SHA256 和 aggregate SHA256。
- manifest 纳入 `h100_pcie.yaml`。
- provenance schema 升级为 `dynosim_hisim_aic_provenance.v2`。
- 严格拆分 DynoSim resolved contract、HiSim formal ACC `sim_config.json` 和 HiSim debug replay events。
- debug events 标记为 `participates_in_formal_match=false`。
- formal artifact 未记录的字段显式写为 `unrecorded`。
- 为 35 个 RPS 点逐点比较 system、backend、TP/EP/DP、workload distribution、dtype 和 CUDA graph contract。
- 生成 `aic_database_equivalence.json`，全量比较 canonical txt 与 DynoSim parquet。
- 单独审计 recorded overlay 的 base rows 与新增 distribution rows。

### 3.3 验证方法

两个数据根：

```text
DynoSim:  /workspace/hisim_align/aiconfigurator-dynosim-e2e/src/aiconfigurator/systems
Canonical: /workspace/hisim_align/aiconfigurator/src/aiconfigurator/systems
```

H100 PCIe 全量审计：

| 检查项 | 结果 |
|---|---:|
| 数据表数量 | 17 |
| 总行数 | 222,721 |
| schema、行数、行顺序 | 一致 |
| 数值 | 等价 |
| 最大绝对差 | `1.4210854715202004e-14` |
| 最大相对差 | `3.8915720948265776e-14` |
| `h100_pcie.yaml` | byte-identical |
| YAML SHA256 | `694f83ef10cee8dfc8d9d78957757371daef75cd0bc1e3917b74f04082d1a6fe` |

最终 DynoSim manifest：

```text
file_count=19
aggregate_sha256=b93c0ac775ef1122441e5777c20c174223a2c542d3434ae21367e92142f833bb
```

Recorded overlay 验证：

- 76,869 条 base MoE rows 全部原样保留；
- 只新增 48 条 recorded-only rows；
- `balanced` replay 不消费新增 rows；
- 35 个 RPS 点的 formal 已记录 contract 全部匹配，结果为 35/35。

### 3.4 修复效果

P0 是可比性和 provenance 修复，不改变预测 latency，因此不为它单独宣称 MAPE 收益。其效果是：

- 排除了“缺少 recorded overlay 导致旧 50 ms decode”的错误根因。
- 证明 canonical txt 与 DynoSim parquet 数据内容数值等价。
- 每个最终 RPS 点都能追溯到唯一的 DynoSim resolved root 和 aggregate hash。
- formal ACC config 与 debug replay 证据不再混用。

证据边界：历史 HiSim artifact 没有记录 resolved systems root 或逐文件 hash，因此只能确认数据内容可比，不能声称历史 HiSim 与当前 DynoSim 使用了完全相同的物理路径或文件 hash。

## 4. P1：AIC dtype、MLA graph 和 layer mix 修复

### 4.1 修复动机

P0 排除数据库差异后，A-32 decode step 仍显示明确的 AIC graph/config mismatch：

| Decode batch | 修复前 DynoSim | HiSim 参考 |
|---|---:|---:|
| batch 19 | 约 50.09 ms | 约 21.35 ms |
| batch 13 | 约 34.37 ms | 约 15.09 ms |

白盒检查发现：

1. DynoSim global KV cache 使用 FP8，HiSim formal config 使用 BF16。
2. archived MLA module 表需要独立 FP8 KV dtype，旧接口不能同时表达 global BF16 与 module FP8。
3. DeepSeek SGLang graph 没有使用一致的 WideEP MLA primary。
4. cropped model 丢失 `first_k_dense_replace` / `moe_layer_freq`，不能正确表达 3 dense + 17 MoE。
5. replay 使用固定 CUDA graph capture set，D-32 formal config 实际捕获到 batch 128。

### 4.2 修改内容

#### Dynamo 配置与 bridge

- 在 `MockEngineArgs` 新增 `aic_mla_module_kv_cache_dtype`。
- 将字段贯通 Python config、PyO3 binding、Rust replay、KV estimator 和 `RustAicCallback`。
- 将 module dtype 纳入 compiled-engine cache identity。
- alignment contract 改为 global KV `bfloat16`、MLA module KV `fp8`、FMHA `bfloat16`。

#### Aiconfigurator graph

- DeepSeek SGLang context primary 使用 `WideEPContextMLA`。
- generation primary 使用 `WideEPGenerationMLA`。
- module primary 使用 FP8 KV，global/fallback 路径保留 BF16 KV。
- HF config parser 保留 `first_k_dense_replace` 和 `moe_layer_freq`。
- graph 按 3 dense + 17 MoE 分别缩放 dense FFN、shared expert、router、dispatch 和 routed MoE op count。

#### CUDA graph contract

- `engine_args(point)` 从每个点的 formal HiSim `sim_config.json` 读取 capture sizes。
- 普通点当前通常到 64；D-32 使用到 128。

### 4.3 验证方法

白盒构图直接检查实际模型和 op primary：

```text
model_class=DeepSeekModel
num_layers=20
dense_layers=3
moe_layers=17
context_primary=WideEPContextMLA
generation_primary=WideEPGenerationMLA
context_module_kv=fp8
generation_module_kv=fp8
global_kv=bfloat16
fmha=bfloat16
```

同时确认：

- `MockEngineArgs` serde 保留 module KV dtype；
- D-32 capture set 最大值为 128；
- `cargo check -p dynamo-mocker --features aic-forward-pass` 通过；
- AIC focused pytest 86/86 通过。

### 4.4 修复效果

| Decode batch | 修复前 DynoSim | P1 后 DynoSim | HiSim 参考 |
|---|---:|---:|---:|
| batch 19 | 约 50.09 ms | 约 22.53 ms | 约 21.35 ms |
| batch 13 | 约 34.37 ms | 约 14.49 ms | 约 15.09 ms |

P1 将算子预测层拉回与 HiSim 接近的量级，证明旧 50/34 ms 的主要问题是 dtype、MLA graph 和 layer mix contract，而不是数据库 overlay。

P1 不代表请求级 scheduler 已经对齐。TTFT、throughput 和 E2E 仍受 admission、prefill 顺序、首 token 可见性和 client/server time domain 影响，因此这里以白盒 decode step 作为 P1 的独立效果，不把后续 P2 收益归给 P1。

## 5. P2：首 token 与 KV materialization 生命周期修复

### 5.1 修复动机

non-mixed prefill 完成后，首 token 是本次 forward 的 logits 输出。它已对客户端可见，但尚未作为下一次 forward 的输入，因此还没有对应 KV。

旧实现发布首 token 时立即预留 KV slot。在 KV 满载场景下，P2 最终修复前的 35 点 RPS replay 共出现 557 次：

`Failed to reserve first-token KV slots after prefill`

旧生命周期还可能：

- 为首 token 额外收取一次 decode duration；
- 在新 prefill boundary 发布首 token时推进已有 decode 请求；
- 将最终未被下一次 forward 消费的输出 token 当作 materialized KV 缓存。

### 5.2 修改内容

主要修改位于 `core.rs`、`decode.rs`、`request.rs` 和 `tests.rs`：

1. non-mixed prefill boundary 完成时立即发布该批请求首 token。
2. 发布首 token 不额外调用 AIC decode，也不推进其他 running decode 请求。
3. 首 token 通过 `append_unmaterialized_output_token()` 加入输出序列，但不增加 `materialized_tokens`。
4. 下一次 decode forward 消费该 token时才申请 KV slot并 materialize。
5. prompt KV 在首 token 发布时仍可进入 radix cache。
6. 请求完成时只缓存实际 materialized KV，不缓存最终未消费 token。
7. decode capacity、平均 context 和 active KV token 统一基于 `materialized_tokens`。
8. MTP、zero-output、retract 和 router event 生命周期同步调整。

### 5.3 验证方法

测试覆盖：

- prefill boundary 阶梯发布首 token；
- 不推进已有 decode 请求；
- 不额外收取 decode duration；
- KV pool 已满时仍能发布首 token；
- 下一次 decode 才 materialize KV；
- 完成请求不缓存最终未消费 token；
- MTP、zero-output、retract 和 router events 保持一致。

验证结果：

- SGLang scheduler Rust tests：62/62 通过；
- 最终 70 点 first-token KV reservation warning：0；
- 非有限/负请求指标：0；
- AIC database/interpolation miss、OOM：0。

### 5.4 修复效果

| 项目 | P2 最终修复前 | P2 最终修复后 |
|---|---:|---:|
| First-token KV reservation warning | 557 | 0 |
| RPS diagnostic MAPE | 41.553459% | 39.618066% |
| 变化 |  | -1.935392 pp |

更重要的效果是客户端可见 token 与 GPU 已 materialize KV 不再被当成同一状态。满载时首 token 不再被错误 KV 预留阻塞，cache/router 也只发布实际存在的 KV block。

P2 没有消除全部 TTFT 误差。正确生命周期移除了过去的偶然误差抵消，使长输入 prefill/admission/client-envelope 差异更清楚地暴露出来。

## 6. P0/P1/P2 联合效果

同口径基线为旧 `prefix_fixed_full/accuracy_report.json`，最终结果为 `dynosim_p012_final/accuracy_report.json`。两边都位于 prefix 修复之后，并复用相同真实 artifacts。

### 6.1 总体结果

| Scope | 修复前 | P0/P1/P2 后 | 变化 | HiSim 既有参考 |
|---|---:|---:|---:|---:|
| 70 points / 350 cells | 41.296005% | 34.891637% | -6.404368 pp | 9.481926% |
| RPS / 175 cells | 56.379043% | 39.618066% | -16.760977 pp | 13.164734% |
| Single batch / 175 cells | 26.212967% | 30.165208% | +3.952241 pp | 5.799118% |

### 6.2 RPS 指标分解

| Metric | 修复前 | 修复后 | 变化 |
|---|---:|---:|---:|
| TTFT | 27.643578% | 62.004307% | +34.360729 pp |
| TPOT | 85.945602% | 32.212124% | -53.733478 pp |
| ITL | 85.935878% | 32.217589% | -53.718289 pp |
| E2E | 61.273996% | 39.430739% | -21.843257 pp |
| Input throughput | 21.096161% | 32.225573% | +11.129412 pp |

### 6.3 Single-batch 指标分解

| Metric | 修复前 | 修复后 | 变化 |
|---|---:|---:|---:|
| TTFT | 37.376340% | 43.426569% | +6.050229 pp |
| TPOT | 26.533098% | 17.746373% | -8.786724 pp |
| ITL | 26.533098% | 17.746373% | -8.786724 pp |
| E2E | 23.728025% | 28.185237% | +4.457212 pp |
| Input throughput | 16.894273% | 43.721487% | +26.827213 pp |

最大改善为 B-RPS4 ITL：335.790934% -> 68.884767%，下降 266.906167 pp。最大回退为 B-RPS16 TTFT：17.224378% -> 95.252701%，上升 78.028323 pp。

联合结果说明：P1/P2 对 decode/TPOT/ITL 的方向正确，但 TTFT 和 throughput 仍存在系统性偏差。旧结果中的部分低 MAPE 来自错误生命周期或服务时间的偶然抵消，不应使用全局 scale 或修改 AIC 表值掩盖剩余 scheduler time-domain 问题。

## 7. P1 补充验证：compiled engine 与 Python op-walk 配对 A/B

这部分是 P1 的调用模式验证，不是 P3，也没有把实验性 Python callback 合入正式 DynoSim。

### 7.1 验证动机

P0 已证明 HiSim CSV 与 DynoSim Parquet 的底层表值等价，但“数据库相同”仍不等于“完整查询函数相同”。当前正式 mocker 直接构造 pure-Rust `RustAicCallback`，而 HiSim 使用 Python model/op query 路径。

另外，当前正式代码中的 `DYNAMO_AIC_DISABLE_COMPILED_ENGINE=1` 只由 Python `AicSession` 读取；mocker 已经取消 Python fallback，因此只设置该变量仍会打印 `pure-Rust RustAicCallback`，不能形成有效 A/B。

### 7.2 验证内容

在隔离 worktree 中恢复一个仅受 `DYNAMO_AIC_DISABLE_COMPILED_ENGINE=1` 控制的 `PyAicCallback`，完整透传 TP/EP/DP、dtype 和 workload distribution。主 worktree 未改这个执行路径。

两组都复用 `prefix_fixed_full/run1`：

- 35 个 RPS + 35 个 single-batch，共 70 点、3,867 请求；
- 每点 trace 在 source/compiled/op-walk 三方 hash 一致；
- 两组 engine args hash 一致；35 个 RPS 点还与历史 source args 逐点一致；
- 历史 single-batch 目录没有保存 engine args，两组统一使用历史 35 个 RPS 点中唯一的一套共同参数；
- 两组使用相同 AIC manifest，`aggregate_sha256=b93c0ac775ef1122441e5777c20c174223a2c542d3434ae21367e92142f833bb`；
- 70/70 compiled 日志命中 pure-Rust marker，70/70 实验日志命中 Python op-walk marker。

历史 `prefix_fixed_full` 与本次 A/B 的 Dynamo/AIC revision 不同，因此历史列只作上下文。调用模式的因果效果只比较本次 current compiled control 与 current Python op-walk。

### 7.3 验证效果

| Scope | 历史 prefix_fixed | current compiled | Python op-walk | HiSim 既有参考 | op-walk - compiled |
|---|---:|---:|---:|---:|---:|
| 70 points / 350 cells | 41.296005% | 34.898321% | 11.486348% | 9.481926% | -23.411973 pp |
| RPS / 175 cells | 56.379043% | 39.631435% | 14.275396% | 13.164734% | -25.356038 pp |
| Single batch / 175 cells | 26.212967% | 30.165208% | 8.697300% | 5.799118% | -21.467907 pp |

Python op-walk 使总体 MAPE 相对下降 67.086244%；323/350 个 metric cells 改善，27 个回退。分场景结果：

| 场景 | current compiled | Python op-walk | HiSim | 变化 |
|---|---:|---:|---:|---:|
| A | 42.574442% | 5.117086% | 1.725423% | -37.457356 pp |
| B | 42.295176% | 10.909814% | 6.072986% | -31.385362 pp |
| B-mixed | 43.430685% | 7.642436% | 3.618379% | -35.788249 pp |
| C | 30.941717% | 23.543424% | 25.029007% | -7.398293 pp |
| D | 15.249588% | 10.218983% | 10.963837% | -5.030605 pp |

这证明 P0/P1/P2 后剩余差异中，compiled AIC 查询实现仍占主要部分，不能把 34.9% 全部归因于 Rust scheduler。

### 7.4 白盒根因

相同 prefill shape 的查询结果：

| `(bs, new_tokens, prefix)` | compiled | Python op-walk | op-walk / compiled |
|---|---:|---:|---:|
| `(1, 8192, 0)` | 108.765 ms | 132.103 ms | 1.21x |
| `(1, 8192, 8192)` | 108.233 ms | 170.708 ms | 1.58x |
| `(1, 8192, 16384)` | 106.974 ms | 203.984 ms | 1.91x |
| `(1, 8192, 24576)` | 106.142 ms | 236.207 ms | 2.23x |

差值集中在 `context_mla_block`。compiled graph 将 WideEP MLA 的 `num_heads` 序列化为 TP size=4；Python `WideEPContextMLA` 则按 `128 / TP = 32` 查询 per-rank heads。只在内存中把 compiled spec 的 4 改成 32 后：

- `(1,8192,0)` 和 `(1,8192,8192)` prefill 与 Python op-walk 完全一致；
- 表内 decode 探针恢复完全或近似一致；
- 更长 prefix 仍有 8.54%–20.39% 残差，来自 Rust/Python 插值、外推实现差异。

因此这不是底层表值不同，而是同一表之上的 compiled graph 参数和插值执行语义不等价。

完整报告：`dynosim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_h100pcie_gsp_mixed_seed11_full/aic_mode_ab/aic_call_mode_diagnostic_report.md`。

## 8. 最终验证与 artifacts

最终目录：`/workspace/hisim_align/dynamo/dynosim_p012_final`

| 验证项 | 结果 |
|---|---:|
| 总点数 / cells | 70 / 350 |
| RPS 点 / requests | 35 / 3432 |
| Single-batch 点 / requests | 35 / 435 |
| RPS formal contract match | 35/35 |
| First-token KV reservation warning | 0 |
| Database/interpolation miss | 0 |
| OOM | 0 |
| 非有限或负请求指标 | 0 |
| Rust SGLang scheduler tests | 62 passed |
| Alignment pytest | 5 passed |
| AIC focused pytest | 86 passed |
| AIC feature cargo check | passed |
| `py_compile` / `git diff --check` | passed |

主要报告：

- `dynosim_p012_final/sweep_acc_report.md`：同口径 before/after 和全部点表格。
- `dynosim_p012_final/sweep_diagnostic_report.md`：P0/P1/P2 provenance 和极值归因。
- `dynosim_p012_final/aic_database_equivalence.json`：17 表数据库审计。

## 9. 能否在不大改的情况下达到 HiSim 高精度

1. AIC predictor 层可以通过相对局部的改动复用 HiSim+AIC graph/config/database contract。P0/P1 已将 A-32 decode step 从约 50/34 ms 修正到约 22.5/14.5 ms。
2. Request lifecycle 层需要 P2 这种 scheduler 语义修复，已经超出纯参数校准，但改动仍集中在 token/KV/cache 生命周期。
3. 调用模式配对 A/B 将总体/RPS MAPE 从 34.90%/39.63% 降到 11.49%/14.28%，已接近 HiSim 的 9.48%/13.16%。这说明达到接近 HiSim 的精度不需要重写 Rust scheduler，但需要修正 compiled WideEP MLA graph 参数及长 prefix 插值/外推，不能只调整数据库值。
4. op-walk 并未消除全部调度差异：C 场景仍为 23.54%，C-RPS24 单点为 61.44%。P3 继续保持 deferred；先把 compiled engine 对 Python op-walk 的 fixed-shape parity 修到位，再用逐 forward trace 区分剩余 scheduler/admission/time-domain 误差。

结论：`hisim+aic` 可以作为 DynoSim 的高精度算子预测底座。当前最大的可局部修复问题是 compiled AIC 查询与 Python op-walk 不等价；修正后仍需要 P3 处理 C 等场景的 scheduler/time-domain 残差。P3 当前保持 deferred。

文档更新时间：2026-08-05
