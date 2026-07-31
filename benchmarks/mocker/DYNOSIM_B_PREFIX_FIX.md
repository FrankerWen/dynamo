# DynoSim SGLang B 场景 prefix 修复与 MAPE 对比

> 本文保留首次 B-only 修复验收的边界与数据。B-mixed 7 点完成同一
> prefix 修复的正式 run1 后，35 点正式结论已更新到
> `DYNOSIM_PREFIX_FIX_FINAL.md`。

## 结论

本次只重跑正式 B 场景的 7 个 RPS 点：
`sweep_b_rps_point_{1,2,4,8,16,24,32}`。未运行 single-batch 或
B-mixed，未启动 SGLang，也未重新采集实机数据。

35 个 point-metric cell 的 DynoSim 平均 MAPE 从 **877.076257%**
降至 **94.396026%**，改善 **782.680231 个百分点**。同一批 HiSim
ACC 平均 MAPE 为 **6.783453%**。修复解决了 B-4 的主要 scheduler
异常，但没有使用经验缩放掩盖剩余的 decode/AIC 建模误差。

| 指标 | 修复前 DynoSim | 修复后 DynoSim | HiSim | 改善百分点 |
|---|---:|---:|---:|---:|
| `mean_ttft_ms` | 2726.928690% | 24.545347% | 12.954146% | +2702.383344 |
| `mean_tpot_ms` | 445.535807% | 148.232123% | 7.095138% | +297.303684 |
| `mean_itl_ms` | 445.546354% | 148.234598% | 7.096176% | +297.311756 |
| `mean_e2e_latency_ms` | 702.927846% | 122.082815% | 5.201970% | +580.845031 |
| `input_throughput` | 64.442589% | 28.885249% | 1.569834% | +35.557340 |

| 点位 | 修复前 DynoSim | 修复后 DynoSim | HiSim | 改善百分点 |
|---|---:|---:|---:|---:|
| `sweep_b_rps_point_1` | 477.241892% | 44.445575% | 5.493322% | +432.796317 |
| `sweep_b_rps_point_2` | 1784.491743% | 130.293058% | 9.463893% | +1654.198685 |
| `sweep_b_rps_point_4` | 2420.086530% | 208.547346% | 14.582579% | +2211.539184 |
| `sweep_b_rps_point_8` | 708.786899% | 73.243469% | 8.048399% | +635.543430 |
| `sweep_b_rps_point_16` | 274.633880% | 65.554290% | 2.430855% | +209.079590 |
| `sweep_b_rps_point_24` | 245.447636% | 71.538005% | 5.017274% | +173.909631 |
| `sweep_b_rps_point_32` | 228.845221% | 67.150441% | 2.447849% | +161.694781 |

## 根本原因

B 的每个 prompt 长 32,768 tokens，120-request 点的完整共享前缀为
29,491 tokens，每请求真实独有后缀只有 3,277 tokens；SGLang chunk
size 为 8,192。

旧 `get_new_batch_prefill` 先按 chunk 截断请求，再将截断序列交给
radix cache 做 prefix match。这把首次 admission 能看见的共享前缀
人为限制在首个 8,192-token chunk。后续 21,299 个本应命中的共享
prefix tokens 被分三轮当作 fresh prefill 计时并占用 chunk/KV budget。
续 chunk 又重复发出 `AdmissionEvent`，使 per-request
`reused_input_tokens` 最终被覆盖成 24,576，不能代表首次 admission。

B-4 的运行证据形成了闭环：

- 首次 admission prefix ratio：`0.247917` → `0.892494`。前者精确对应
  119 个 warm request 每个只命中 8,192 tokens；后者对应每个命中完整
  29,491-token LCP。
- 平均 scheduler queue：`35155.622 ms` → `145.168 ms`，下降
  `99.587%`。
- 正式 TTFT（复用同一 `64.761877 ms` ingress）：
  `36072.383 ms` → `474.202 ms`；TTFT MAPE：
  `9970.276894%` → `32.382273%`。
- input throughput：`36645.920` → `94617.020 tok/s`；MAPE：
  `71.128152%` → `25.455053%`。
- 正式 run1 无 AIC database miss，ingress 只占旧 TTFT 的极小部分，
  且 scheduler arrival 未改变。

因此，旧 B-4 的 35 秒 TTFT 不是 AIC 缺表、ingress 或 fake/balanced
语义造成的，而是 DynoSim SGLang path 的完整 prefix match 与 chunk
顺序错误造成的单 worker prefill 队列。

## 修复

`lib/mocker/src/scheduler/sglang/prefill.rs` 现在：

1. 首次 admission 先对完整 `current_sequence` 做 radix-prefix match；
2. 只对 `sequence_len - prefix_len` 的 fresh tokens 应用 chunk 和
   prefill/KV budget；
3. 在 eviction/allocation 之间锁住已经匹配的 radix path，防止读到的
   prefix 被并发淘汰后改变；
4. 续 chunk 沿现有 KV lease 扩展，不重新做逻辑 admission，也不覆盖
   first-admission reuse；
5. FPM/AIC 的 `tokens_computed`、`prefix_tokens` 与 budget 扣减都使用
   实际完整-prefix admission 结果。

同时移除了不再适用的 `SglangRequest::extend_input_len`，并新增三项
Rust 回归测试：

- 完整共享前缀必须在 chunk 前命中；
- chunk budget 只扣 fresh suffix，因此同一 batch 可以接纳多个 warm
  request；
- chunk continuation 不得再次发 admission。

SGLang scheduler 定向测试结果为 **60 passed, 0 failed**。

## 修复后的剩余误差

修复后 27/35 个 cell 改善，8/35 个 cell 回退。8 个回退全部是
RPS 8/16/24/32 的 TPOT/ITL：

- 旧错误把 prefill 变成约 `1.118 req/s` 的节流瓶颈，decode TPOT
  固定在约 `194.439 ms`；
- 修复后这些高负载点进入约 `3.145 req/s` 的 decode 饱和状态，TPOT
  固定在约 `209.237 ms`，而实机为 `99.040–111.053 ms`。

从相同的高负载 duration/request-throughput/TPOT 平台推断，prefix
修复移除了错误的 prefill throttling，暴露了单 aggregated worker、
decode batch/CUDA graph padding 与 AIC decode service-time 的剩余差异。
这 8 个回退不是 prefix 修复应当回滚的证据。

修复后最大的单 cell MAPE 是 B-4 `mean_itl_ms=335.790934%`；
B-4 `mean_e2e_latency_ms` 仍为 `313.319534%`。这说明 TTFT 根因已经
修正，但 B-4 的 decode/ITL 残差仍是下一层准确度瓶颈。

## 数据来源与验收

本次 replay 使用原 accepted `run1` 中保存的 canonical
`trace.mooncake.jsonl`、RID map、ingress envelope、raw metric values、
HiSim ACC MAPE 和原 workload/raw/HiSim/sim-config SHA256。原实机 source
root 在本次执行时已不在工作区，因此没有用其他目录冒充源数据；报告
明确记录为 preserved offline-replay provenance。

实机 baseline 的路由语义仍是 `ep_dispatch_algorithm=fake`；
DynoSim 算子计时使用 AIC `workload_distribution=balanced`。DynoSim
不执行 fake TopK，本文不做该声明。

单次正式 B replay 的验收结果：

- run1 完成 7/7 点、690/690 requests；
- canonical trace 与保存的 source manifest SHA256 一致；
- 无 NaN、负延迟或 AIC database miss；
- H100 PCIe、SGLang 0.5.9、TP4/attention-DP1/MoE-TP1/EP4 映射合法；
- preserved before artifacts 未修改；
- ACC report 仅由 run1 生成，不进行跨 run 平均。

复跑入口为 `benchmarks/mocker/dynosim_b_prefix_fix_compare.py`。完整机器可读
结果见输出根目录的 `before_after_mape.json` 和 `acceptance.json`，可读
报告见 `before_after_mape.md`。
