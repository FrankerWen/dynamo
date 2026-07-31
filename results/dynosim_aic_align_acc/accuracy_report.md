# DynoSim SGLang vs HiSim 离线重放诊断

本报告复用 TP4+EP4 真实 artifacts，属于 CPU/offline replay diagnostic；不是新的 HiSim `acc_bench/compare` formal ACC，DynoSim formal ACC 为 `N/A`。

## 最大单点/单指标差异

- 点位：`sweep_b_rps_point_16`；指标：`mean_ttft_ms`；DynoSim MAPE 95.252701%，HiSim MAPE 6.655653%，差值 +88.597048 个百分点。

### 最大差异归因

- 调度：平均排队 104.587 ms，占 scheduler TTFT 55.888%；admit 后到首 token 平均 82.551 ms。DynoSim input throughput 303550.867 tok/s，实机 177183.506 tok/s。
- Prefix/cache：完整 token/block=1 链路实际复用比例 0.892494，每请求最终复用 0–29491 / 32768 tokens；复用存在，但未阻止单 worker 队列增长。
- CUDA graph：只对 decode padding；B-4 TPOT 为 DynoSim 68.585 ms、实机 106.389 ms，因此可通过 decode contention 间接放大 TTFT，但不是 prefill TTFT 的直接 padding 项。
- AIC 数据：`h100_pcie` / SGLang `0.5.9` / `balanced`，KV `bfloat16`、FMHA `bfloat16`；无 database miss、无经验缩放。
- Ingress：复用固定 47.986 ms，只占 DynoSim TTFT 20.409%，不能解释 35 秒量级的 scheduler queue。

## 总体结果

- 70 个点、5 项指标、350 个 cell；DynoSim 总均值 34.891637%，HiSim 总均值 9.481926%。
- 胜负：DynoSim 51，HiSim 299，平局 0。

| 指标 | DynoSim MAPE 均值 | HiSim MAPE 均值 | 差值百分点 | DynoSim/HiSim/平 |
|---|---:|---:|---:|---:|
| `mean_ttft_ms` | 52.715438% | 24.489560% | +28.225878 | 22/48/0 |
| `mean_tpot_ms` | 24.979248% | 6.287195% | +18.692053 | 9/61/0 |
| `mean_itl_ms` | 24.981981% | 6.285935% | +18.696046 | 9/61/0 |
| `mean_e2e_latency_ms` | 33.807988% | 6.066841% | +27.741147 | 6/64/0 |
| `input_throughput` | 37.973530% | 4.280101% | +33.693429 | 5/65/0 |

## 归因边界

- 调度：FCFS admission、batching 和 queueing 差异主要进入 throughput/TTFT。
- Prefix/cache：B 与 B-mixed 使用 block size 1 的完整 token 序列，保留 token 级共享前缀。
- CUDA graph：仅 decode 计时 batch 上取整到最小可容纳 capture size；prefill 不 padding。
- AIC：数据库 miss、负延迟或非有限值均为硬失败；未添加经验缩放。
- Ingress：逐点复用原 HiSim ACC 的固定 envelope，只加到 DynoSim TTFT/E2E。
- fake+balanced：实机 baseline 使用 `ep_dispatch_algorithm=fake`；DynoSim 仅使用 AIC `workload_distribution=balanced`，不声称内部执行 fake TopK 路由。
