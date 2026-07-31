# DynoSim P0/P1/P2 Diagnostic Report

This report covers CPU/offline replay only. It does not replace HiSim formal ACC. P3 is deferred.

## P0: AIC Database Provenance

- DynoSim resolved root: `/workspace/hisim_align/aiconfigurator-dynosim-e2e/src/aiconfigurator/systems`. Audited 17 tables / 222721 rows; numeric equivalence is `True`.
- Maximum txt/parquet conversion delta: absolute 1.421e-14, relative 3.892e-14; H100 PCIe YAML is byte-identical: `True`.
- Historical HiSim artifacts record `database_path=null` but not the resolved systems root/hash. Data content is comparable; physical root/hash identity is not proven.
- Recorded overlay preserves all 76869 base MoE rows and adds 48 recorded-only rows. Balanced replay does not consume those rows, so the overlay is not the old 50 ms decode root cause.

## P1/P2 Outcome

- P1 separates BF16 global KV from FP8 MLA-module KV, uses the SGLang WideEP MLA graph, and preserves the 3 dense + 17 MoE cropped-model structure.
- P2 emits non-mixed first tokens at each prefill boundary without charging a decode duration or reserving KV before the token is consumed by a later forward pass.
- CUDA graph capture sizes are read per point from the formal HiSim sim config; D-32 uses captures through batch 128.

## Before / After Extremes

- Largest improvement: `sweep_b_rps_point_4` / `mean_itl_ms`, 335.790934% -> 68.884767% (-266.906167 pp).
- Largest regression: `sweep_b_rps_point_16` / `mean_ttft_ms`, 17.224378% -> 95.252701% (+78.028323 pp).

## Remaining Accuracy Gap

- Decode/ITL improves materially, but long-input TTFT and throughput remain inaccurate. The remaining gap is scheduler prefill admission/visibility and client-envelope behavior, not evidence for changing AIC table values.
- `workload_distribution=balanced` is an AIC workload assumption; it is not proof that DynoSim executes SGLang fake TopK routing.
- P3 remains deferred until the P0/P1/P2 results are accepted.
