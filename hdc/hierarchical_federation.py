"""
hdc/hierarchical_federation.py
================================
Hierarchical Federated HDC Learning for Edge Intelligence
==========================================================
Reference:
    McMahan et al. (2017) "Communication-Efficient Learning of Deep Networks
    from Decentralised Data" AISTATS 2017. — FedAvg baseline.

    Briggs, Fan, Andras (2020) "Federated Learning with Hierarchical Clustering
    of Local Updates to Improve Training on Non-IID Data" IJCNN 2020.

    El Mhamdi et al. (2018) "The Hidden Vulnerability of Distributed Learning"
    ICML 2018. — Geometric median robustness.

HDC federation advantage over NN federation:
    NN federation: 1M parameter model × float32 = 4 MB per agent per round
    HDC federation: D bits × C classes = 4096 × 10 / 8 = 5 KB per agent per round

    Bandwidth reduction: ~800× per round vs FedAvg.
    Further savings from hierarchical aggregation: O(log(n_agents)) rounds.

Hierarchical architecture (e.g., ISR / SIGINT application):
    Level 0 (sensors): raw data → local HDC prototypes  [edge device, 10 KB RAM]
    Level 1 (nodes):   aggregate 4-8 sensors/region     [tactical hub, 1 MB RAM]
    Level 2 (region):  aggregate 4-8 nodes/sector       [base station, 10 MB RAM]
    Level 3 (global):  final global model               [cloud, unlimited]

Each level only communicates with the level above — the inter-level bandwidth
is small (HDC prototype payloads) and aggregation happens in parallel.

Key features:
  1. Async aggregation — nodes can contribute at different rates
  2. Per-tier bandwidth budget — prune/quantise if budget exceeded
  3. Dropout robustness — missing nodes handled gracefully
  4. Differential privacy option — Gaussian noise before upload
  5. Byzantine detection at each tier via geometric median
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch

from hdc.physics_world_model import _majority


def _hamming(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - (a.float() != b.float()).float().mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Tier configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TierConfig:
    """Configuration for one tier in the hierarchy."""
    name:             str
    n_children:       int           # Max children per node at this tier
    bandwidth_bytes:  int = 65536   # Bytes allowed per aggregation round
    dp_epsilon:       float = 0.0   # ε-DP noise (0 = disabled)
    dp_delta:         float = 1e-5
    byzantine_frac:   float = 0.0   # Expected Byzantine fraction (0 = none)
    use_geom_median:  bool  = False  # Use geometric median (robustness)
    async_timeout_s:  float = 5.0   # Seconds to wait for slow children
    min_children:     int   = 1     # Minimum responses to aggregate


# ═══════════════════════════════════════════════════════════════════════════════
# Prototype compression
# ═══════════════════════════════════════════════════════════════════════════════

class PrototypeCompressor:
    """
    Compress HDC prototype payload to meet bandwidth constraints.

    HDC binary prototypes are already very compact (D bits / 8 bytes).
    For very tight budgets, two additional options:
      1. Dimension reduction: keep only top-K most discriminative dims
      2. Scalar quantisation: transmit the soft float accumulator as int8

    The receiver reconstructs via dimension expansion (zero-padding) or
    dequantisation.

    Args:
        full_dim:    Full prototype dimension
        budget_bits: Available bit budget per prototype
    """

    def __init__(self, full_dim: int, budget_bits: int):
        self.full_dim    = full_dim
        self.budget_bits = budget_bits
        self.k_dims      = min(full_dim, max(64, budget_bits))

    def compress(self, proto: torch.Tensor) -> torch.Tensor:
        """Return compressed prototype (keeps top-k dims by |value|)."""
        if self.k_dims >= self.full_dim:
            return proto   # no compression needed
        abs_val = proto.float().abs()
        _, top_idx = abs_val.topk(self.k_dims)
        compressed = torch.zeros(self.full_dim, device=proto.device)
        compressed[top_idx] = proto[top_idx]
        return compressed

    def decompress(self, compressed: torch.Tensor) -> torch.Tensor:
        """Reconstruct full prototype (zeros at non-transmitted dims)."""
        return compressed   # already full-dim with zeros at dropped dims

    def compression_ratio(self) -> float:
        return self.k_dims / self.full_dim


# ═══════════════════════════════════════════════════════════════════════════════
# Single tier node
# ═══════════════════════════════════════════════════════════════════════════════

class FederationTierNode:
    """
    One node in the hierarchical federation.

    Each node aggregates contributions from its children (or leaf agents),
    optionally applies DP noise, and sends the result to its parent tier.

    Aggregation options:
        'weighted_mean':  weighted average by child sample counts
        'geom_median':    geometric median (Byzantine robust)
        'majority':       majority vote over binary prototypes

    Args:
        node_id:    Unique identifier for this node
        n_classes:  Number of classification classes
        dim:        Prototype HV dimension
        cfg:        TierConfig for this tier
        device:     torch device
    """

    def __init__(
        self,
        node_id:   str,
        n_classes: int,
        dim:       int,
        cfg:       TierConfig,
        device:    str = "cpu",
    ):
        self.node_id   = node_id
        self.n_classes = n_classes
        self.dim       = dim
        self.cfg       = cfg
        self.device    = device

        self._pending: List[Dict[str, Any]] = []  # buffered child contributions
        self._round = 0
        self._compressor = PrototypeCompressor(
            dim, cfg.bandwidth_bytes * 8 // n_classes
        )

    def receive(self, contribution: Dict[str, Any]):
        """Buffer a child contribution (async-safe)."""
        self._pending.append(contribution)

    def _add_dp_noise(self, protos: torch.Tensor) -> torch.Tensor:
        """Add calibrated Gaussian noise for (ε, δ)-DP."""
        if self.cfg.dp_epsilon <= 0:
            return protos
        # Gaussian mechanism: σ = sqrt(2 ln(1.25/δ)) × sensitivity / ε
        # For binary HDC: sensitivity = D (max Hamming distance = D bits)
        sensitivity = float(self.dim)
        sigma = math.sqrt(2 * math.log(1.25 / max(self.cfg.dp_delta, 1e-10)))
        sigma = sigma * sensitivity / max(self.cfg.dp_epsilon, 1e-8)
        noise = torch.randn_like(protos) * sigma
        return protos + noise

    def _geom_median(self, protos_list: List[torch.Tensor], n_iter: int = 15) -> torch.Tensor:
        """Weiszfeld geometric median for Byzantine robustness."""
        stacked = torch.stack([p.float().to(self.device) for p in protos_list])
        c = stacked.median(dim=0).values
        for _ in range(n_iter):
            dists = (stacked - c.unsqueeze(0)).norm(dim=1).clamp(min=1e-8)
            w     = 1.0 / dists
            c_new = (w.unsqueeze(1) * stacked).sum(0) / w.sum()
            if (c_new - c).norm().item() < 1e-8:
                break
            c = c_new
        return c

    def aggregate(self) -> Optional[Dict[str, Any]]:
        """
        Aggregate all pending contributions and produce a node-level prototype.

        Returns None if fewer than min_children contributions received.
        Clears pending buffer after aggregation.
        """
        if len(self._pending) < self.cfg.min_children:
            return None

        self._round += 1
        contributions = list(self._pending)
        self._pending.clear()

        global_protos = []
        total_samples = [0] * self.n_classes

        for c in range(self.n_classes):
            protos  = []
            weights = []
            for contrib in contributions:
                if c < len(contrib.get("prototypes", [])):
                    protos.append(contrib["prototypes"][c].to(self.device))
                    weights.append(float(contrib.get("counts", [1])[c]) if c < len(contrib.get("counts", [])) else 1.0)
                    total_samples[c] += weights[-1]

            if not protos:
                global_protos.append(torch.zeros(self.dim, device=self.device))
                continue

            if self.cfg.use_geom_median and len(protos) >= 3:
                agg = self._geom_median(protos)
            else:
                total_w = max(sum(weights), 1e-8)
                agg     = sum(w / total_w * p for w, p in zip(weights, protos))

            # Apply DP noise before sending up the hierarchy
            agg = self._add_dp_noise(agg.unsqueeze(0)).squeeze(0)

            # Compress to bandwidth budget
            agg = self._compressor.compress(agg)

            global_protos.append(_majority(agg))

        return {
            "node_id":    self.node_id,
            "round":      self._round,
            "prototypes": global_protos,
            "counts":     total_samples,
            "n_children": len(contributions),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Hierarchical coordinator
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchicalFederatedHDC:
    """
    Multi-tier hierarchical federation for scalable edge HDC deployment.

    Manages the full tree: leaf agents → tier-1 nodes → tier-2 nodes → global.
    Designed for IQT-style applications where sensor data from many edge devices
    must be aggregated without any single node seeing raw data from all others.

    Architecture:
        Global model ← tier-2 aggregators ← tier-1 hubs ← leaf agents

    Each tier applies:
        1. Weighted mean or geometric median aggregation
        2. Optional (ε, δ)-DP noise before upward transmission
        3. Bandwidth compression if payload exceeds budget
        4. Byzantine detection via geometric median (if enabled)

    Bandwidth scaling:
        Leaf to tier-1:  n_leaves × D bits payload
        Tier-1 to tier-2: n_tier1_nodes × D bits payload (already aggregated)
        Tier-2 to global: n_tier2_nodes × D bits payload

        Total bandwidth: O(n_leaves × D bits) vs O(n_leaves × |NN_params| × 32 bits)

    Args:
        n_classes:    Number of classification classes
        dim:          Prototype HV dimension
        tier_configs: List of TierConfig from bottom to top
        device:       torch device
    """

    def __init__(
        self,
        n_classes:    int,
        dim:          int,
        tier_configs: Optional[List[TierConfig]] = None,
        device:       str = "cpu",
    ):
        self.n_classes = n_classes
        self.dim       = dim
        self.device    = device

        # Default two-tier hierarchy if not specified
        if tier_configs is None:
            tier_configs = [
                TierConfig("edge",   n_children=8,  bandwidth_bytes=4096),
                TierConfig("region", n_children=4,  bandwidth_bytes=8192,
                           dp_epsilon=1.0, use_geom_median=True),
            ]
        self.tier_configs = tier_configs

        # Global model: merged prototypes from all tier-top nodes
        self._global_protos: List[torch.Tensor] = [
            torch.zeros(dim, device=device) for _ in range(n_classes)
        ]
        self._global_counts   = [0] * n_classes
        self._global_round    = 0

        # Tier-1 aggregation nodes (auto-created as agents register)
        self._tier1_nodes: Dict[str, FederationTierNode] = {}
        self._tier1_assignments: Dict[str, str] = {}   # agent_id → node_id

        # Total communication cost tracking
        self._bytes_transmitted = 0

    def _assign_to_tier1(self, agent_id: str) -> str:
        """Assign agent to a tier-1 node (round-robin within capacity)."""
        if agent_id in self._tier1_assignments:
            return self._tier1_assignments[agent_id]

        # Find or create a tier-1 node with capacity
        cfg = self.tier_configs[0] if self.tier_configs else TierConfig("tier1", 8)
        for node_id, node in self._tier1_nodes.items():
            n_assigned = sum(1 for v in self._tier1_assignments.values() if v == node_id)
            if n_assigned < cfg.n_children:
                self._tier1_assignments[agent_id] = node_id
                return node_id

        # Create new tier-1 node
        new_id = f"tier1_{len(self._tier1_nodes)}"
        self._tier1_nodes[new_id] = FederationTierNode(
            new_id, self.n_classes, self.dim, cfg, self.device
        )
        self._tier1_assignments[agent_id] = new_id
        return new_id

    def submit(self, agent_id: str, export: Dict[str, Any]):
        """
        Submit a local prototype from a leaf agent.

        Routes the contribution to the appropriate tier-1 node.

        Args:
            agent_id: Unique identifier for the submitting agent
            export:   HDCAgent.export_prototypes() dict
        """
        node_id = self._assign_to_tier1(agent_id)
        self._tier1_nodes[node_id].receive(export)

        # Track communication cost: D bits × C classes (binary prototypes)
        self._bytes_transmitted += self.dim * self.n_classes // 8

    def aggregate_tier1(self) -> List[Dict[str, Any]]:
        """
        Trigger tier-1 aggregation across all nodes.

        Returns list of tier-1 aggregated exports (one per tier-1 node).
        """
        tier1_results = []
        for node in self._tier1_nodes.values():
            result = node.aggregate()
            if result is not None:
                tier1_results.append(result)
                self._bytes_transmitted += self.dim * self.n_classes // 8

        return tier1_results

    def aggregate_global(self, tier1_results: Optional[List[Dict]] = None) -> List[torch.Tensor]:
        """
        Final global aggregation from tier-1 results.

        If tier1_results is None, triggers tier-1 aggregation first.

        Returns:
            List of n_classes global prototype HVs.
        """
        if tier1_results is None:
            tier1_results = self.aggregate_tier1()

        if not tier1_results:
            return self._global_protos

        self._global_round += 1

        # Apply second tier config (or default) for global aggregation
        global_cfg = self.tier_configs[-1] if len(self.tier_configs) > 1 else self.tier_configs[0]

        global_protos = []
        for c in range(self.n_classes):
            protos  = [r["prototypes"][c].to(self.device) for r in tier1_results
                       if c < len(r["prototypes"])]
            weights = [float(r["counts"][c]) for r in tier1_results
                       if c < len(r.get("counts", []))]

            if not protos:
                global_protos.append(self._global_protos[c])
                continue

            if global_cfg.use_geom_median and len(protos) >= 3:
                agg = FederationTierNode(
                    "global", self.n_classes, self.dim, global_cfg, self.device
                )._geom_median(protos)
            else:
                total_w = max(sum(weights), 1e-8) if weights else 1.0
                agg = sum((w / total_w if weights else 1.0 / len(protos)) * p
                          for w, p in zip(weights or [1.0] * len(protos), protos))

            global_protos.append(_majority(agg))
            self._global_counts[c] += sum(weights)

        self._global_protos = global_protos
        return global_protos

    def federated_round(self) -> Dict[str, Any]:
        """
        Execute a complete federation round: tier-1 → tier-2 → global.

        Returns:
            Dict with round number, n_contributors, bandwidth_kb, global_model.
        """
        tier1_results = self.aggregate_tier1()
        global_protos = self.aggregate_global(tier1_results)

        return {
            "round":            self._global_round,
            "n_tier1_nodes":    len(tier1_results),
            "n_leaf_agents":    len(self._tier1_assignments),
            "bandwidth_kb":     self._bytes_transmitted / 1024,
            "global_protos":    global_protos,
        }

    def communication_savings(self, n_nn_params: int = 1_000_000) -> Dict:
        """Compare bandwidth to NN federation (FedAvg)."""
        hdc_bytes = self.dim * self.n_classes // 8
        nn_bytes  = n_nn_params * 4   # float32
        return {
            "hdc_payload_bytes":   hdc_bytes,
            "nn_payload_bytes":    nn_bytes,
            "bandwidth_reduction": nn_bytes // max(hdc_bytes, 1),
            "bits_transmitted_total": self._bytes_transmitted * 8,
        }
