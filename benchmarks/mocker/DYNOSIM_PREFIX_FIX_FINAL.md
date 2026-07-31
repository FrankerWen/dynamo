# DynoSim SGLang prefix 修复后的正式结论

## 结论

完整 prefix 在 chunked prefill 之前匹配的修复，现已分别应用到正式
B 与 B-mixed 的全部 7 个 RPS 点。正式 ACC 只保留 run1；A/C/D
沿用原 accepted run1，合并后的 35 点、175 个正式 point-metric
cell 结果如下：

| 汇总 | 修复前 DynoSim | 修复后 DynoSim | HiSim |
|---|---:|---:|---:|
| 175-cell 平均 MAPE | 287.445373% | 56.379043% | 13.164734% |
| 赢得的 cell | 41 | 44 | 131（修复后） |

DynoSim 总平均 MAPE 改善 **231.066331 个百分点**，但仍比 HiSim 高
**43.214309 个百分点**。因此正式结论是：prefix/chunk 顺序是原
B/B-mixed 巨大 TTFT 和 queue 误差的主因；修复显著提高了 DynoSim
准确度，但未使其整体优于 HiSim，也没有使用经验缩放。

修复后的全局最大误差不再是 TTFT，而是
`sweep_b_rps_point_4 / mean_itl_ms`：

| 值 | 毫秒或 MAPE |
|---|---:|
| 实机 `mean_itl_ms` | 35.265130 ms |
| DynoSim `mean_itl_ms` | 153.682240 ms |
| DynoSim MAPE | 335.790934% |
| HiSim MAPE | 19.651731% |

这意味着完整-prefix admission 根因已经关闭，当前最大的剩余瓶颈已
转移到 decode/ITL service-time 建模，而不是继续归因于 ingress、
prefix cache 或 AIC database miss。

## 分场景结果

| 场景 | 修复前 DynoSim | 修复后 DynoSim | HiSim |
|---|---:|---:|---:|
| A | 54.917425% | 54.917425% | 1.275083% |
| B | 877.076257% | 94.396026% | 6.783453% |
| B-mixed | 447.532434% | 74.881012% | 3.937068% |
| C | 29.890803% | 29.890803% | 40.384559% |
| D | 27.809948% | 27.809948% | 13.443510% |

B 的 35-cell 平均 MAPE 改善 **782.680231 个百分点**；B-mixed
改善 **372.651422 个百分点**。修复后 DynoSim 在 B-mixed 的 2/35
个 cell 优于 HiSim，整个 175-cell 集合的胜负为 DynoSim 44、
HiSim 131、tie 0。

## B-mixed 根因闭环

B-mixed 每点包含 8 个 prefix group，每组 12 个请求。修复后的精确
reuse 分布是 8 个 group leader 复用 0 token、88 个 warm 请求各复用
29,491 token；run1 的 7 个点均满足该分布。

最大的原 B-mixed 异常 `sweep_b_mixed_rps_point_2` 变化如下：

| 证据 | 修复前 | 修复后 |
|---|---:|---:|
| 首次 admission prefix ratio | 0.229167 | 0.824994 |
| 平均 scheduler queue | 14720.956 ms | 267.517 ms |
| 正式 TTFT | 15628.007 ms | 535.218 ms |
| TTFT MAPE | 2640.100309% | 6.158878% |

queue 减少解释了 TTFT 降幅的 **95.763871%**。旧实现先把输入截断为
8,192-token chunk，再做 radix-prefix match，使 group 内本应命中的
完整 29,491-token 前缀被重复当作 fresh prefill；单 aggregated
worker 因而形成错误队列。修复改为先对完整 sequence 做 prefix
match，只对 fresh suffix 应用 chunk 和 KV/prefill budget，并避免续
chunk 重复 admission 或覆盖首次 reuse。

修复后 B-mixed 最大 cell 是
`sweep_b_mixed_rps_point_2 / mean_tpot_ms=200.060736%`。它与全局
B-4 ITL 最大值共同指向 decode batch/CUDA graph padding、单
aggregated worker 调度及 AIC decode service-time 的剩余偏差。当前
证据不能把该残差归为真实 fake TopK 路由，因为 DynoSim 仅使用 AIC
`workload_distribution=balanced`，不执行真实 expert dispatcher。

## 验收与数据边界

- 仅重放正式 `sweep_b_mixed_rps_point_{1,2,4,8,16,24,32}`，未运行
  single-batch。
- run1 完成 7/7 点、672/672 请求；ACC report 仅由 run1 生成。
- 每点均满足 8-group token-level prefix 合同；无 NaN、负延迟或
  AIC database miss；H100 PCIe、SGLang 0.5.9、TP4/attention-DP1、
  MoE-TP1/EP4 映射合法。
- 未启动 SGLang、未重新采集实机数据、未做 GPU preflight；复用原
  accepted run1 保存的 trace、RID map、raw/HiSim ACC、ingress 和
  source SHA256，并确认 preserved before artifacts 未修改。
- 实机 baseline 的路由语义为 `ep_dispatch_algorithm=fake`；DynoSim
  AIC 算子计时为 `balanced`。报告不声称 DynoSim 内部执行 fake
  TopK，也不添加经验 scale。

## 产物

- 最终 35 点结果根：
  `/workspace/hisim_align/dynosim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_h100pcie_gsp_mixed_seed11_full`
- 修复后自包含运行：最终结果根的 `prefix_fixed_full/run1`；原
  `run1` 保留为修复前基线。
- 最终 ACC 报告：最终结果根的 `final_accuracy_report.json` 与
  `final_accuracy_report.md`
- 最终验收：最终结果根的 `final_acceptance.json`
- B-mixed 复跑入口：
  `benchmarks/mocker/dynosim_bmixed_prefix_fix_compare.py`
- 完整运行提升入口：
  `benchmarks/mocker/dynosim_promote_prefix_fixed_full.py`
- B-only 修复报告：`benchmarks/mocker/DYNOSIM_B_PREFIX_FIX.md`
- B-mixed 输出根：
  `/workspace/hisim_align/dynosim_sglang_compare_deep_gemm_acc_layer20_32768_tp4ep4_h100pcie_gsp_mixed_seed11_full_bmixed_prefix_fix`
- 机器可读正式汇总：输出根的 `merged_accuracy_report.json`
- 可读正式汇总：输出根的 `merged_accuracy_report.md`
- B-mixed 前后对比：输出根的 `before_after_mape.json` 与
  `before_after_mape.md`
- 验收与 prefix 合同：输出根的 `acceptance.json` 与
  `prefix_acceptance.json`
