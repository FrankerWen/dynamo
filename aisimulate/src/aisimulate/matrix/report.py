# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reprice stored resource usage and emit reusable analysis tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._utils import read_document, sha256_file
from .config import SCHEMA_VERSION, PriceSpec
from .store import ResultStore
from .visualize import generate_visual_report


def _load_prices(path: Path) -> list[PriceSpec]:
    payload = read_document(path)
    if not isinstance(payload, dict):
        raise TypeError(f"price catalog {path} must contain one object")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("kind") != "prices"
    ):
        raise ValueError(
            f"price catalog {path} must have schema_version={SCHEMA_VERSION} and kind='prices'"
        )
    prices = payload.get("prices")
    if not isinstance(prices, list):
        raise TypeError(f"price catalog {path} field 'prices' must be a list")
    return [PriceSpec.model_validate(price) for price in prices]


def _validate_price_coverage(store: ResultStore) -> None:
    run_skus = {
        row[0]
        for row in store.connection.execute(
            "SELECT DISTINCT sku_id FROM matrix_cells"
        ).fetchall()
    }
    priced_skus = {
        row[0]
        for row in store.connection.execute(
            "SELECT DISTINCT sku_id FROM price_catalog"
        ).fetchall()
    }
    missing = sorted(run_skus - priced_skus)
    if missing:
        raise ValueError(
            "report-time price catalog does not cover run SKU(s): " + ", ".join(missing)
        )


def _write_variant_resolution_report(store: ResultStore) -> Path:
    """Write the human-facing resolver explanation in Chinese."""

    rows = store.connection.execute(
        """
        SELECT model_id, sku_id, variant_id, source, support_status,
               data_mode, deployment_fidelity, selected, selection_reason,
               support_reason
        FROM variant_resolution
        WHERE experiment_id = (
            SELECT experiment_id FROM experiments
            ORDER BY updated_at DESC LIMIT 1
        )
        ORDER BY model_id, sku_id, source, variant_id
        """
    ).fetchall()
    lines = [
        "# 精度 Variant 解析报告",
        "",
        "默认候选推荐仅纳入 `deployable`；`hypothetical` 只进入 what-if Pareto。",
        "自动模式仅在原生精度因硬件或 kernel 不支持时触发备用方案，且不会自动降低精度。",
        "",
        "| 模型 | SKU | Variant | 来源 | 支持状态 | 数据模式 | 部署属性 | 展开 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(value).replace("|", "\\|")
                for value in (*row[:7], "是" if row[7] else "否")
            )
            + " |"
        )
    rejected = [
        row for row in rows if not row[7] or not str(row[4]).startswith("supported_")
    ]
    if rejected:
        lines.extend(["", "## 未采用或不受支持项", ""])
        for row in rejected:
            lines.append(f"- `{row[0]} / {row[1]} / {row[2]}`：{row[8]}；AIC：{row[9]}")
    if not rows:
        lines.extend(["", "本数据库没有 Variant 解析记录（可能是旧版运行结果）。"])
    path = store.output_dir / "variant_resolution.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_report(
    store: ResultStore,
    *,
    price_catalog: str | Path | None = None,
    topk: int = 20,
) -> dict[str, Any]:
    """Refresh price-dependent views and emit the five requested analysis axes."""
    resolved_price_catalog: Path | None = None
    if price_catalog is not None:
        resolved_price_catalog = Path(price_catalog).expanduser().resolve()
        store.replace_prices(_load_prices(resolved_price_catalog))
    _validate_price_coverage(store)
    store.create_analysis_views(topk=topk)

    queries = {
        "pareto_front.csv": """
            WITH final_candidates AS (
                SELECT c.candidate_id, c.cell_id, c.score, c.used_gpus,
                       c.pd_profile_id, c.kvbm_profile_id, c.objectives_json,
                       c.config_json, e.stage, e.metrics_json, e.sla_pass,
                       mc.model_id, mc.model_variant_id, mc.sku_id,
                       mc.workload_id
                FROM candidates c
                JOIN evaluations e ON e.evaluation_id = c.evaluation_id
                JOIN matrix_cells mc ON mc.cell_id = c.cell_id
                WHERE e.outcome = 'feasible'
                  AND COALESCE(json_extract_string(mc.cell_json,
                    '$.model_variant.deployment_fidelity'), 'deployable') = 'deployable'
                  AND (
                    e.stage = 'capacity'
                    OR (
                      e.stage = 'spica'
                      AND NOT EXISTS (
                        SELECT 1
                        FROM candidates c2
                        JOIN evaluations e2
                          ON e2.evaluation_id = c2.evaluation_id
                        WHERE c2.cell_id = c.cell_id
                          AND e2.stage = 'capacity'
                          AND e2.outcome = 'feasible'
                          AND COALESCE(e2.sla_pass, TRUE)
                      )
                    )
                  )
            )
            SELECT fc.model_id, fc.model_variant_id, fc.sku_id, fc.workload_id,
                   fc.candidate_id,
                   fc.stage, fc.used_gpus, fc.score, fc.pd_profile_id,
                   fc.kvbm_profile_id, COALESCE(fc.sla_pass, FALSE) AS sla_pass,
                   CAST(json_extract(fc.objectives_json,
                     '$.goodput_per_gpu') AS DOUBLE) AS goodput_per_gpu,
                   COALESCE(
                     CAST(json_extract(fc.metrics_json,
                       '$.output_throughput_tok_s') AS DOUBLE)
                       / NULLIF(fc.used_gpus, 0),
                     CAST(json_extract(fc.objectives_json,
                       '$.goodput_per_gpu') AS DOUBLE)
                   ) AS raw_throughput_per_gpu,
                   CAST(json_extract(fc.objectives_json,
                     '$.throughput_per_user') AS DOUBLE) AS throughput_per_user,
                   CAST(json_extract(fc.metrics_json,
                     '$.mean_ttft_ms') AS DOUBLE) AS mean_ttft_ms,
                   CAST(json_extract(fc.metrics_json,
                     '$.mean_tpot_ms') AS DOUBLE) AS mean_tpot_ms,
                   json_extract_string(fc.config_json,
                     '$.pd_bandwidth_domain') AS pd_bandwidth_domain,
                   CAST(json_extract(fc.config_json,
                     '$.pd_group_gpus') AS INTEGER) AS pd_group_gpus,
                   CAST(json_extract(fc.config_json,
                     '$.pd_transfer_bandwidth_gbytes_per_s') AS DOUBLE)
                     AS pd_transfer_bandwidth_gbytes_per_s,
                   strict_front.candidate_id IS NOT NULL AS strict_pareto,
                   epsilon_front.candidate_id IS NOT NULL AS epsilon_pareto,
                   fc.config_json
            FROM final_candidates fc
            LEFT JOIN frontier_membership strict_front
              ON strict_front.candidate_id = fc.candidate_id
             AND strict_front.frontier_type = 'strict_pareto'
            LEFT JOIN frontier_membership epsilon_front
              ON epsilon_front.candidate_id = fc.candidate_id
             AND epsilon_front.frontier_type = 'epsilon_pareto'
            ORDER BY fc.model_id, fc.model_variant_id, fc.workload_id, fc.sku_id,
                     throughput_per_user, goodput_per_gpu
        """,
        "aic_operating_points.csv": """
            SELECT mc.model_id, mc.model_variant_id, mc.sku_id, mc.workload_id,
                   json_extract_string(e.selection_json, '$.mode') AS mode,
                   CAST(json_extract(e.metrics_json, '$.concurrency') AS BIGINT)
                     AS concurrency,
                   CAST(json_extract(e.metrics_json, '$.request_rate') AS DOUBLE)
                     AS aic_request_rate,
                   CAST(json_extract(e.metrics_json,
                     '$.tokens_per_second_per_gpu') AS DOUBLE)
                     AS aic_tokens_per_second_per_gpu,
                   CAST(json_extract(e.metrics_json,
                     '$.tokens_per_second_per_user') AS DOUBLE)
                     AS aic_tokens_per_second_per_user,
                   CAST(json_extract(e.metrics_json, '$.num_total_gpus') AS BIGINT)
                     AS aic_num_total_gpus,
                   e.outcome = 'selected' AS selected,
                   e.reason AS selection_reasons,
                   json_extract_string(e.selection_json, '$.parallel_key')
                     AS parallel_key,
                   json_extract(e.selection_json, '$.aic_config') AS aic_config
            FROM evaluations e
            JOIN matrix_cells mc ON mc.cell_id = e.cell_id
            WHERE e.stage = 'aic_operating_point'
              AND e.outcome IN ('selected', 'not_selected')
            ORDER BY mc.model_id, mc.sku_id, mc.workload_id, mode, concurrency
        """,
        "load_conditioned_pareto.csv": """
            SELECT mc.model_id, mc.model_variant_id, mc.sku_id, mc.workload_id,
                   c.candidate_id, e.stage, c.used_gpus, c.score,
                   CAST(json_extract(c.config_json, '$.concurrency') AS DOUBLE)
                     AS concurrency,
                   CAST(json_extract(c.objectives_json,
                     '$.goodput_per_gpu') AS DOUBLE) AS goodput_per_gpu,
                   CAST(json_extract(c.objectives_json,
                     '$.throughput_per_user') AS DOUBLE) AS throughput_per_user,
                   f.frontier_type, f.rank, e.metrics_json, c.config_json
            FROM frontier_membership f
            JOIN candidates c USING (candidate_id)
            JOIN evaluations e USING (evaluation_id)
            JOIN matrix_cells mc ON mc.cell_id = c.cell_id
            WHERE f.frontier_type LIKE 'load:%:strict_pareto'
            ORDER BY mc.model_id, mc.sku_id, mc.workload_id,
                     concurrency, f.rank
        """,
        "global_operating_envelope.csv": """
            SELECT mc.model_id, mc.model_variant_id, mc.sku_id, mc.workload_id,
                   c.candidate_id, e.stage, c.used_gpus, c.score,
                   CAST(json_extract(c.config_json, '$.concurrency') AS DOUBLE)
                     AS discovery_concurrency,
                   CAST(json_extract(c.objectives_json,
                     '$.goodput_per_gpu') AS DOUBLE) AS goodput_per_gpu,
                   CAST(json_extract(c.objectives_json,
                     '$.throughput_per_user') AS DOUBLE) AS throughput_per_user,
                   f.rank, e.metrics_json, c.config_json
            FROM frontier_membership f
            JOIN candidates c USING (candidate_id)
            JOIN evaluations e USING (evaluation_id)
            JOIN matrix_cells mc ON mc.cell_id = c.cell_id
            WHERE f.frontier_type = 'global_operating_envelope:strict_pareto'
            ORDER BY mc.model_id, mc.sku_id, mc.workload_id, f.rank
        """,
        "capacity_certification.csv": """
            SELECT mc.model_id, mc.model_variant_id, mc.sku_id, mc.workload_id,
                   c.candidate_id, c.used_gpus, c.pd_profile_id, c.kvbm_profile_id,
                   json_extract_string(c.config_json, '$.capacity_search_status')
                     AS capacity_search_status,
                   CAST(json_extract(c.config_json,
                     '$.capacity_search_feasible_lower_bound') AS DOUBLE)
                     AS feasible_rps_lower_bound,
                   CAST(json_extract(c.config_json,
                     '$.capacity_search_infeasible_bound') AS DOUBLE)
                     AS infeasible_rps_upper_bound,
                   CAST(json_extract(c.config_json,
                     '$.capacity_search_right_censored') AS BOOLEAN)
                     AS right_censored,
                   CAST(json_extract(c.config_json,
                     '$.discovery_load_value') AS DOUBLE) AS discovery_concurrency,
                   e.metrics_json, c.config_json
            FROM candidates c
            JOIN evaluations e USING (evaluation_id)
            JOIN matrix_cells mc ON mc.cell_id = c.cell_id
            WHERE e.stage = 'capacity' AND e.outcome = 'feasible'
            ORDER BY mc.model_id, mc.sku_id, mc.workload_id,
                     feasible_rps_lower_bound DESC, c.used_gpus
        """,
        "cost_by_model_sku.csv": """
            WITH ranked AS (
                SELECT cc.*,
                       CAST(json_extract(mc.cell_json,
                         '$.workload.summary.context_length') AS DOUBLE)
                         AS context_length,
                       COALESCE(
                         CAST(json_extract(cc.config_json, '$.max_sla_rps') AS DOUBLE),
                         CAST(json_extract(cc.config_json,
                           '$.max_sla_effective_rps') AS DOUBLE)
                       ) AS sla_rps,
                       CAST(json_extract(cc.metrics_json,
                         '$.mean_output_token_throughput_per_user') AS DOUBLE)
                         AS achieved_tps_per_user,
                       row_number() OVER (
                         PARTITION BY cc.model_id, cc.model_variant_id, cc.sku_id,
                                      cc.workload_id, cc.price_id
                         ORDER BY cc.all_in_cost_per_million_total_tokens NULLS LAST,
                                  cc.used_gpus, cc.candidate_id
                       ) AS cost_rank,
                       count(*) OVER (
                         PARTITION BY cc.model_id, cc.model_variant_id, cc.sku_id,
                                      cc.workload_id, cc.price_id
                       ) AS candidate_count
                FROM deployable_candidate_costs cc
                JOIN matrix_cells mc USING (cell_id)
            )
            SELECT model_id, model_variant_id, sku_id, workload_id,
                   context_length, price_id,
                   currency, candidate_id, stage, used_gpus, score, sla_rps,
                   achieved_tps_per_user,
                   pd_bandwidth_domain, pd_group_gpus,
                   pd_transfer_bandwidth_gbytes_per_s,
                   gpu_only_cost_per_million_input_tokens,
                   gpu_only_cost_per_million_output_tokens,
                   gpu_only_cost_per_million_total_tokens,
                   all_in_cost_per_million_input_tokens,
                   all_in_cost_per_million_output_tokens,
                   all_in_cost_per_million_total_tokens,
                   candidate_count, config_json
            FROM ranked
            WHERE cost_rank = 1
            ORDER BY model_id, model_variant_id, workload_id, sku_id, price_id
        """,
        "cost_by_tpsu_requirement.csv": """
            WITH requirements AS (
                SELECT DISTINCT model_id, model_variant_id, workload_id,
                       tps_per_user AS required_tps_per_user
                FROM candidate_dimensions
                WHERE tps_per_user > 0
                UNION
                SELECT DISTINCT model_id, model_variant_id, workload_id,
                       CAST(json_extract(cell_json,
                         '$.sla.tps_per_user') AS DOUBLE)
                         AS required_tps_per_user
                FROM matrix_cells
            ), eligible AS (
                SELECT r.required_tps_per_user, c.*,
                       CAST(json_extract(mc.cell_json,
                         '$.sla.tps_per_user') AS DOUBLE) AS selected_sla_tps_floor,
                       row_number() OVER (
                         PARTITION BY c.model_id, c.model_variant_id, c.sku_id,
                                      c.workload_id, c.price_id,
                                      r.required_tps_per_user
                         ORDER BY c.all_in_cost_per_million_total_tokens NULLS LAST,
                                  c.used_gpus, c.candidate_id
                       ) AS cost_rank
                FROM requirements r
                JOIN candidate_dimensions c
                  ON c.model_id = r.model_id
                 AND c.model_variant_id = r.model_variant_id
                 AND c.workload_id = r.workload_id
                 AND c.tps_per_user >= r.required_tps_per_user
                JOIN matrix_cells mc USING (cell_id)
            )
            SELECT model_id, model_variant_id, sku_id, workload_id, price_id,
                   currency,
                   required_tps_per_user, selected_sla_tps_floor,
                   tps_per_user AS achieved_tps_per_user,
                   all_in_cost_per_million_total_tokens, sla_rps, used_gpus,
                   pd_bandwidth_domain, pd_group_gpus,
                   pd_transfer_bandwidth_gbytes_per_s,
                   candidate_id, stage, config_json
            FROM eligible
            WHERE cost_rank = 1
            ORDER BY model_id, model_variant_id, workload_id,
                     required_tps_per_user, sku_id
        """,
        "cost_by_context_sku.csv": """
            WITH ranked AS (
                SELECT cc.*,
                       CAST(json_extract(mc.cell_json,
                         '$.workload.summary.context_length') AS DOUBLE)
                         AS context_length,
                       COALESCE(
                         CAST(json_extract(cc.config_json, '$.max_sla_rps') AS DOUBLE),
                         CAST(json_extract(cc.config_json,
                           '$.max_sla_effective_rps') AS DOUBLE)
                       ) AS sla_rps,
                       row_number() OVER (
                         PARTITION BY cc.model_id, cc.model_variant_id, cc.sku_id,
                                      cc.workload_id, cc.price_id
                         ORDER BY cc.all_in_cost_per_million_total_tokens NULLS LAST,
                                  cc.used_gpus, cc.candidate_id
                       ) AS cost_rank
                FROM deployable_candidate_costs cc
                JOIN matrix_cells mc USING (cell_id)
            )
            SELECT model_id, model_variant_id, sku_id, workload_id, price_id,
                   currency,
                   context_length, sla_rps,
                   all_in_cost_per_million_total_tokens AS best_cost_per_million,
                   gpu_only_cost_per_million_total_tokens,
                   pd_bandwidth_domain, pd_group_gpus,
                   pd_transfer_bandwidth_gbytes_per_s,
                   candidate_id, stage, used_gpus, config_json
            FROM ranked
            WHERE cost_rank = 1
            ORDER BY model_id, model_variant_id, context_length, sku_id, price_id
        """,
        "kv_capacity_impact.csv": """
            SELECT k.model_id, k.model_variant_id, k.sku_id, k.workload_id,
                   k.price_id,
                   k.candidate_id, k.kvbm_profile_id, k.stage, k.used_gpus,
                   k.sla_rps,
                   k.pd_bandwidth_domain, k.pd_group_gpus,
                   k.pd_transfer_bandwidth_gbytes_per_s,
                   CAST(json_extract(config_json, '$.g2_capacity_gib') AS DOUBLE)
                     AS g2_capacity_gib,
                   CAST(json_extract(config_json, '$.g2_target_hit_rate') AS DOUBLE)
                     AS target_hit_rate,
                   CAST(json_extract(metrics_json, '$.prefix_cache_reused_ratio') AS DOUBLE)
                     AS measured_hit_rate,
                   k.g2_bandwidth_gbytes_per_s AS storage_bandwidth_gbytes_per_s,
                   COALESCE(
                     CAST(json_extract(mc.cell_json,
                       '$.sku.compute_nic_gbytes_per_s_per_gpu') AS DOUBLE),
                     CAST(json_extract(mc.cell_json,
                       '$.sku.compute_nic_gbps_per_gpu') AS DOUBLE)
                   ) AS compute_nic_gbytes_per_s,
                   storage_bandwidth_gbytes_per_s
                     / NULLIF(compute_nic_gbytes_per_s, 0)
                     AS storage_to_compute_nic_ratio,
                   k.matched_config_cost_reduction,
                   k.matched_config_cost_reduction_ratio,
                   k.system_optimal_cost_reduction,
                   k.system_optimal_cost_reduction_ratio,
                   k.matched_config_rps_increase,
                   k.matched_config_rps_increase_ratio,
                   k.system_optimal_rps_increase,
                   k.system_optimal_rps_increase_ratio,
                   k.config_json
            FROM kvbm_uplift k
            JOIN matrix_cells mc USING (cell_id)
            ORDER BY k.model_id, k.model_variant_id, k.sku_id, k.workload_id,
                     g2_capacity_gib,
                     storage_to_compute_nic_ratio
        """,
        "kv_bandwidth_impact.csv": """
            SELECT k.model_id, k.model_variant_id, k.sku_id, k.workload_id,
                   k.price_id, k.candidate_id,
                   k.kvbm_profile_id, k.stage, k.used_gpus, k.sla_rps,
                   k.pd_bandwidth_domain, k.pd_group_gpus,
                   k.pd_transfer_bandwidth_gbytes_per_s,
                   CAST(json_extract(k.config_json,
                     '$.bandwidth_g1_to_g2_gbps') AS DOUBLE)
                     AS storage_bandwidth_gbytes_per_s,
                   COALESCE(
                     CAST(json_extract(mc.cell_json,
                       '$.sku.compute_nic_gbytes_per_s_per_gpu') AS DOUBLE),
                     CAST(json_extract(mc.cell_json,
                       '$.sku.compute_nic_gbps_per_gpu') AS DOUBLE)
                   ) AS compute_nic_gbytes_per_s,
                   storage_bandwidth_gbytes_per_s
                     / NULLIF(compute_nic_gbytes_per_s, 0)
                     AS storage_to_compute_nic_ratio,
                   CAST(json_extract(k.config_json,
                     '$.g2_capacity_gib') AS DOUBLE) AS g2_capacity_gib,
                   CAST(json_extract(k.config_json,
                     '$.g2_target_hit_rate') AS DOUBLE) AS target_hit_rate,
                   CAST(json_extract(k.metrics_json,
                     '$.prefix_cache_reused_ratio') AS DOUBLE) AS measured_hit_rate,
                   k.matched_config_cost_reduction,
                   k.matched_config_cost_reduction_ratio,
                   k.system_optimal_cost_reduction,
                   k.system_optimal_cost_reduction_ratio,
                   k.matched_config_rps_increase,
                   k.matched_config_rps_increase_ratio,
                   k.system_optimal_rps_increase,
                   k.system_optimal_rps_increase_ratio,
                   k.config_json
            FROM kvbm_uplift k
            JOIN matrix_cells mc USING (cell_id)
            ORDER BY k.model_id, k.model_variant_id, k.sku_id, k.workload_id,
                     storage_to_compute_nic_ratio
        """,
        "kv_footprint_audit.csv": """
            WITH base AS (
                SELECT mc.model_id, mc.model_variant_id, mc.sku_id,
                       mc.workload_id, c.candidate_id,
                       e.stage, c.config_json
                FROM candidates c
                JOIN evaluations e ON e.evaluation_id = c.evaluation_id
                JOIN matrix_cells mc ON mc.cell_id = c.cell_id
                WHERE e.outcome = 'feasible' AND e.stage IN ('spica', 'capacity')
            ), roles AS (
                SELECT *, 'agg' AS role, '$.agg_kv_footprint' AS footprint_path
                FROM base
                UNION ALL
                SELECT *, 'prefill', '$.prefill_kv_footprint' FROM base
                UNION ALL
                SELECT *, 'decode', '$.decode_kv_footprint' FROM base
            )
            SELECT model_id, model_variant_id, sku_id, workload_id, candidate_id,
                   stage, role,
                   json_extract_string(config_json,
                     footprint_path || '.source') AS source,
                   json_extract_string(config_json,
                     footprint_path || '.layout') AS layout,
                   json_extract_string(config_json,
                     footprint_path || '.scope') AS scope,
                   json_extract_string(config_json,
                     footprint_path || '.dtype') AS dtype,
                   json_extract_string(config_json,
                     footprint_path || '.model_class') AS aic_model_class,
                   json_extract_string(config_json,
                     footprint_path || '.growth_kind') AS growth_kind,
                   CAST(json_extract(config_json,
                     footprint_path || '.tp') AS INTEGER) AS tp,
                   CAST(json_extract(config_json,
                     footprint_path || '.reference_length') AS BIGINT)
                     AS reference_length,
                   CAST(json_extract(config_json,
                     footprint_path || '.bytes_at_zero') AS BIGINT)
                     AS fixed_bytes_per_sequence_per_rank,
                   CAST(json_extract(config_json,
                     footprint_path || '.bytes_at_reference') AS BIGINT)
                     AS reference_bytes_per_sequence_per_rank,
                   CAST(json_extract(config_json,
                     footprint_path || '.marginal_bytes_per_token') AS BIGINT)
                     AS initial_marginal_bytes_per_token_per_rank,
                   CAST(json_extract(config_json,
                     '$.kv_bytes_per_token') AS BIGINT) AS kvbm_bytes_per_token_per_rank,
                   CAST(json_extract(config_json,
                     '$.kv_transfer_bytes_per_token') AS BIGINT)
                     AS pd_transfer_bytes_per_token_critical_link,
                   json_extract_string(config_json,
                     '$.kvbm_footprint_contract') AS kvbm_footprint_contract,
                   json_extract_string(config_json,
                     '$.pd_transfer_footprint_contract') AS pd_transfer_footprint_contract,
                   json_extract_string(config_json,
                     '$.pd_transfer_physical_bytes_contract')
                     AS pd_transfer_physical_bytes_contract
            FROM roles
            WHERE json_extract(config_json, footprint_path) IS NOT NULL
            ORDER BY model_id, model_variant_id, sku_id, workload_id,
                     candidate_id, role
        """,
        "cell_status.csv": """
            SELECT model_id, model_variant_id,
                   json_extract_string(cell_json,
                     '$.model_variant.weight_precision') AS weight_precision,
                   json_extract_string(cell_json,
                     '$.model_variant.quantization.gemm') AS aic_gemm_dtype,
                   json_extract_string(cell_json,
                     '$.model_variant.quantization.moe') AS aic_moe_dtype,
                   json_extract_string(cell_json,
                     '$.model_variant.quantization.fmha') AS aic_fmha_dtype,
                   json_extract_string(cell_json,
                     '$.model_variant.quantization.kv_cache') AS aic_kv_cache_dtype,
                   json_extract_string(cell_json,
                     '$.model_variant.quantization.comm') AS aic_comm_dtype,
                   json_extract_string(cell_json,
                     '$.model_variant.deployment_fidelity') AS deployment_fidelity,
                   json_extract_string(cell_json,
                     '$.variant_resolution.source') AS variant_resolution_source,
                   json_extract_string(cell_json,
                     '$.variant_resolution.support.status') AS variant_support_status,
                   sku_id, workload_id, sla_id, support_tier,
                   status, detail, cell_id, fingerprint
            FROM matrix_cells
            ORDER BY model_id, model_variant_id, workload_id, sku_id
        """,
        "what_if_pareto_front.csv": """
            SELECT w.model_id, w.model_variant_id, w.sku_id, w.workload_id,
                   w.candidate_id, w.stage, w.used_gpus, w.score,
                   w.pd_profile_id, w.kvbm_profile_id,
                   w.all_in_cost_per_million_total_tokens,
                   strict_front.candidate_id IS NOT NULL AS strict_pareto,
                   epsilon_front.candidate_id IS NOT NULL AS epsilon_pareto,
                   w.metrics_json, w.config_json
            FROM what_if_candidate_costs w
            LEFT JOIN frontier_membership strict_front
              ON strict_front.candidate_id = w.candidate_id
             AND strict_front.frontier_type = 'strict_pareto'
            LEFT JOIN frontier_membership epsilon_front
              ON epsilon_front.candidate_id = w.candidate_id
             AND epsilon_front.frontier_type = 'epsilon_pareto'
            ORDER BY w.model_id, w.model_variant_id, w.workload_id,
                     w.sku_id, w.score DESC
        """,
        "variant_resolution.csv": """
            SELECT model_id, sku_id, variant_id, backend, source,
                   support_status, support_reason, data_mode,
                   deployment_fidelity, selected, selection_reason,
                   support_cache_key, experiment_id, resolution_id
            FROM variant_resolution
            WHERE experiment_id = (
                SELECT experiment_id FROM experiments
                ORDER BY updated_at DESC LIMIT 1
            )
            ORDER BY model_id, sku_id, source, variant_id
        """,
    }
    for filename, query in queries.items():
        output_path = str((store.output_dir / filename).resolve()).replace("'", "''")
        store.connection.execute(
            f"COPY ({query}) TO '{output_path}' (HEADER, DELIMITER ',')"
        )
    price_output = str((store.output_dir / "price_catalog_used.csv").resolve()).replace(
        "'", "''"
    )
    store.connection.execute(
        f"COPY (SELECT * FROM price_catalog ORDER BY sku_id, price_id) "
        f"TO '{price_output}' (HEADER, DELIMITER ',')"
    )
    store.export_parquet()
    summary = store.summary()
    summary["price_catalog"] = (
        None if resolved_price_catalog is None else str(resolved_price_catalog)
    )
    summary["price_catalog_sha256"] = (
        None if resolved_price_catalog is None else sha256_file(resolved_price_catalog)
    )
    summary["variant_resolution_report"] = str(_write_variant_resolution_report(store))
    summary.update(generate_visual_report(store.output_dir, summary))
    (store.output_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
