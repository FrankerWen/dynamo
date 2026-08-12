# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve P/D KV-transfer bandwidth from a candidate's compact GPU placement."""

from __future__ import annotations

from dataclasses import dataclass

from .config import PDTransferTopology


@dataclass(frozen=True)
class PDPlacementResolution:
    """Resolved transfer path for one disaggregated candidate."""

    domain: str
    bandwidth_gbytes_per_s_per_gpu: float
    prefill_gpus: int
    decode_gpus: int
    nodes_spanned: int
    racks_spanned: int | None

    @property
    def group_gpus(self) -> int:
        return self.prefill_gpus + self.decode_gpus

    def sample_fields(self) -> dict[str, int | float | str]:
        bandwidth = self.bandwidth_gbytes_per_s_per_gpu
        return {
            "pd_placement_policy": "compact",
            "pd_bandwidth_domain": self.domain,
            "pd_prefill_gpus": self.prefill_gpus,
            "pd_decode_gpus": self.decode_gpus,
            "pd_group_gpus": self.group_gpus,
            "pd_nodes_spanned": self.nodes_spanned,
            "pd_prefill_gpu_slot_start": 0,
            "pd_prefill_gpu_slot_end": self.prefill_gpus - 1,
            "pd_decode_gpu_slot_start": self.prefill_gpus,
            "pd_decode_gpu_slot_end": self.group_gpus - 1,
            "pd_transfer_bandwidth_gbytes_per_s": bandwidth,
            # Dynamo/Spica compatibility fields historically say ``gbps`` but
            # their runtime contract is decimal GB/s.
            "kv_transfer_bandwidth": bandwidth,
            "pd_prefill_to_decode_bandwidth_gbps": bandwidth,
            "pd_decode_to_prefill_bandwidth_gbps": bandwidth,
            **(
                {}
                if self.racks_spanned is None
                else {"pd_racks_spanned": self.racks_spanned}
            ),
        }


def resolve_compact_pd_placement(
    topology: PDTransferTopology,
    *,
    prefill_worker_gpus: int,
    prefill_replicas: int,
    decode_worker_gpus: int,
    decode_replicas: int,
) -> PDPlacementResolution:
    """Pack one P/D group contiguously and resolve its narrowest topology tier.

    The group contains every prefill and decode replica in the candidate.  This
    is deliberately conservative for an any-to-any router: local bandwidth is
    selected only when the whole routing group fits in one node or rack.  A
    future placement implementation can replace this resolver with explicit
    endpoint coordinates without changing the replay boundary.
    """

    values = {
        "prefill_worker_gpus": prefill_worker_gpus,
        "prefill_replicas": prefill_replicas,
        "decode_worker_gpus": decode_worker_gpus,
        "decode_replicas": decode_replicas,
    }
    invalid = {name: value for name, value in values.items() if value <= 0}
    if invalid:
        raise ValueError(f"P/D placement counts must be positive, got {invalid}")

    prefill_gpus = prefill_worker_gpus * prefill_replicas
    decode_gpus = decode_worker_gpus * decode_replicas
    group_gpus = prefill_gpus + decode_gpus
    if group_gpus <= topology.gpus_per_node:
        domain = "intra_node"
        bandwidth = topology.intra_node_gbytes_per_s_per_gpu
    elif topology.gpus_per_rack is None:
        domain = "inter_node"
        bandwidth = topology.inter_node_gbytes_per_s_per_gpu
    elif group_gpus <= topology.gpus_per_rack:
        domain = "intra_rack"
        bandwidth = topology.intra_rack_gbytes_per_s_per_gpu
    else:
        domain = "inter_rack"
        bandwidth = topology.inter_rack_gbytes_per_s_per_gpu
    if bandwidth is None:  # protected by PDTransferTopology validation
        raise ValueError(
            f"P/D topology has no bandwidth for resolved domain {domain!r}"
        )
    return PDPlacementResolution(
        domain=domain,
        bandwidth_gbytes_per_s_per_gpu=bandwidth,
        prefill_gpus=prefill_gpus,
        decode_gpus=decode_gpus,
        nodes_spanned=(group_gpus + topology.gpus_per_node - 1)
        // topology.gpus_per_node,
        racks_spanned=None
        if topology.gpus_per_rack is None
        else (group_gpus + topology.gpus_per_rack - 1) // topology.gpus_per_rack,
    )
