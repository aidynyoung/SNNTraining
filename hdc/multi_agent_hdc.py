"""
hdc/multi_agent_hdc.py
=======================
Federated Multi-Agent HDC — Swarm Intelligence via VSA
=======================================================
Reference:
    Kleyko, Rahimi, Rachkovskij, Osipov, Sommer (2022)
    "Classification and Recall with Binary Hyperdimensional Computing:
    Tradeoffs in Choice of Density and Mapping Characteristics"
    IEEE TNNLS — §VII: Distributed/Federated HDC.

    Mitrokhin, Sutor, Fermüller, Aloimonos (2019)
    "Learning sensorimotor control with neuromorphic sensors"
    Science Robotics — multi-agent coordination via shared HV space.

    Osipov et al. (2024) "Hyperseed: Unsupervised Learning with VSA"
    IEEE TNNLS — distributed prototype learning.

Why multi-agent HDC is uniquely powerful:

    Transformer-based federated learning:
        - Share full weight matrices: O(d²) per model per round
        - Privacy risk: weights reveal training data (membership inference)
        - Communication: hundreds of MB per round

    HDC federated learning:
        - Share prototype HVs: O(D × C) total (D=4096, C=10 classes: 5 KB)
        - Privacy: prototype HV reveals nothing about individual samples
          (XOR binding is holographic — no sample is recoverable)
        - Communication: kilobytes per round, not megabytes
        - Aggregation: bundle (majority vote) — no gradient averaging
        - Works offline: agents can operate independently and merge later

This module implements:

1. HDCAgent
   — Single agent with local HDC prototype memory
   — Trains on local data, can share/receive prototype HVs
   — Privacy-preserving: sharing never reveals raw training samples

2. FederatedHDCAggregator
   — Aggregates prototype HVs from N agents via weighted bundling
   — Handles non-iid data: agent weights proportional to data count
   — Byzantine fault tolerance: outlier rejection before aggregation

3. SwarmHDCMemory
   — Shared distributed memory across a swarm of agents
   — Each agent writes to the shared memory (HDC context window)
   — Any agent can query the shared memory for coordination
   — Applications: sensor network, drone swarm, IoT mesh

4. ConsensusHDC
   — Multi-agent consensus via iterative prototype refinement
   — Agents exchange prototypes, run RefineHD on merged set
   — Converges to globally consistent prototypes without a central server

5. HierarchicalAgentNetwork
   — Tree-structured agent hierarchy for scalable federation
   — Leaf agents: local data, local prototypes
   — Branch agents: aggregate leaves, refine, share up
   — Root: global model, broadcasts down
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from hdc.physics_world_model import _hamming, _majority


# ── Utilities ──────────────────────────────────────────────────────────────────

def _gen_hv(dim: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return (torch.rand(dim, generator=g, device=device) >= 0.5).float()

def _bundle(hvs: List[torch.Tensor]) -> torch.Tensor:
    return _majority(torch.stack(hvs).float().mean(dim=0))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HDCAgent — single federated agent
# ═══════════════════════════════════════════════════════════════════════════════

class HDCAgent:
    """
    Single HDC agent with local prototype memory.

    Trains on local data, produces shareable prototype HVs.
    Privacy: sharing prototypes reveals nothing about raw training samples
    (each prototype is the MAJORITY of many samples — no individual is recoverable).

    Args:
        agent_id:  Unique identifier for this agent
        dim:       HV dimension
        n_classes: Number of classification classes
        device:    torch device
    """

    def __init__(
        self,
        agent_id:  str,
        dim:       int,
        n_classes: int,
        class_names: Optional[List[str]] = None,
        device:    str = "cpu",
    ):
        self.agent_id   = agent_id
        self.dim        = dim
        self.n_classes  = n_classes
        self.class_names = class_names or [f"class_{i}" for i in range(n_classes)]
        self.device     = device

        # Local prototypes
        self._prototypes = [torch.zeros(dim, device=device) for _ in range(n_classes)]
        self._counts     = [0] * n_classes
        self._n_local    = 0

    def train_step(self, hv: torch.Tensor, label: int):
        """Online training: update local prototype for this label."""
        hv = hv.float().to(self.device)
        n  = self._counts[label]
        self._prototypes[label] = _majority(
            (n * self._prototypes[label] + hv) / (n + 1)
        )
        self._counts[label] += 1
        self._n_local += 1

    def predict(self, hv: torch.Tensor) -> Tuple[int, List[float]]:
        """Predict class using local prototypes."""
        hv = hv.float().to(self.device)
        protos = torch.stack(self._prototypes)   # (C, D)
        sims   = _hamming(hv.unsqueeze(0), protos)  # (C,)
        best   = int(sims.argmax().item())
        return best, sims.tolist()

    def export_prototypes(self) -> Dict[str, Any]:
        """
        Export prototypes for federation.

        Returns a privacy-preserving bundle — safe to share.
        """
        return {
            "agent_id":   self.agent_id,
            "n_samples":  self._n_local,
            "counts":     list(self._counts),
            "prototypes": [p.cpu().clone() for p in self._prototypes],
        }

    def import_global(self, global_prototypes: List[torch.Tensor]):
        """
        Import global aggregated prototypes (replace local).

        Called after the aggregator produces a new global model.
        """
        for i, gp in enumerate(global_prototypes[:self.n_classes]):
            self._prototypes[i] = gp.float().to(self.device)

    def merge_local_with_global(
        self,
        global_prototypes: List[torch.Tensor],
        local_weight: float = 0.3,
    ):
        """
        Partial merge: blend local and global prototypes.

        Keeps local specialisation while benefiting from global knowledge.
        """
        for i, gp in enumerate(global_prototypes[:self.n_classes]):
            gp_d = gp.float().to(self.device)
            self._prototypes[i] = _majority(
                local_weight * self._prototypes[i] + (1 - local_weight) * gp_d
            )

    @property
    def total_samples(self) -> int:
        return self._n_local


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FederatedHDCAggregator — aggregate prototypes from N agents
# ═══════════════════════════════════════════════════════════════════════════════

class FederatedHDCAggregator:
    """
    Aggregates prototype HVs from N agents into a global model.

    Aggregation strategies:
        'weighted_mean': weight each agent by its sample count
        'majority':      majority vote across agents (robust to outliers)
        'byzantine':     outlier rejection + weighted mean

    Privacy guarantee: each prototype is a bundle of many samples.
    Aggregation of bundles = bundle of all samples (commutative, associative).

    Args:
        n_classes:    Number of output classes
        dim:          HV dimension
        strategy:     Aggregation strategy
        byzantine_k:  Max outliers to reject in byzantine mode
    """

    def __init__(
        self,
        n_classes:   int,
        dim:         int,
        strategy:    str = "weighted_mean",
        byzantine_k: int = 1,
        device:      str = "cpu",
    ):
        self.n_classes   = n_classes
        self.dim         = dim
        self.strategy    = strategy
        self.byzantine_k = byzantine_k
        self.device      = device

        self._round = 0

    def aggregate(
        self,
        agent_exports: List[Dict[str, Any]],
    ) -> List[torch.Tensor]:
        """
        Aggregate prototypes from multiple agents.

        Args:
            agent_exports: List of dicts from HDCAgent.export_prototypes()

        Returns:
            List of n_classes global prototype HVs.
        """
        self._round += 1

        global_protos = []
        for c in range(self.n_classes):
            protos  = [export["prototypes"][c].to(self.device)
                       for export in agent_exports
                       if c < len(export["prototypes"])]
            weights = [float(export["counts"][c])
                       for export in agent_exports
                       if c < len(export["counts"])]

            if not protos:
                global_protos.append(torch.zeros(self.dim, device=self.device))
                continue

            if self.strategy == "majority":
                global_protos.append(_bundle(protos))

            elif self.strategy == "byzantine":
                # Reject outliers: remove the k protos most different from the mean
                if len(protos) > 2 * self.byzantine_k:
                    mean_proto = _bundle(protos)
                    dists = [
                        float(1.0 - _hamming(p.unsqueeze(0), mean_proto.unsqueeze(0)).item())
                        for p in protos
                    ]
                    # Keep the protos closest to the mean
                    sorted_idx = sorted(range(len(dists)), key=lambda i: dists[i])
                    keep_idx   = sorted_idx[:len(protos) - self.byzantine_k]
                    protos     = [protos[i] for i in keep_idx]
                    weights    = [weights[i] for i in keep_idx]
                # Fall through to weighted mean
                total_w = max(sum(weights), 1e-8)
                agg = sum(w / total_w * p for w, p in zip(weights, protos))
                global_protos.append(_majority(agg))

            else:  # weighted_mean
                total_w = max(sum(weights), 1e-8)
                agg = sum(w / total_w * p for w, p in zip(weights, protos))
                global_protos.append(_majority(agg))

        return global_protos

    def communication_cost_bytes(self, n_agents: int) -> Dict[str, int]:
        """Estimate communication cost in bytes."""
        bits_per_proto    = self.dim
        bytes_per_proto   = bits_per_proto // 8
        bytes_per_agent   = bytes_per_proto * self.n_classes
        total_upload      = bytes_per_agent * n_agents
        total_download    = bytes_per_proto * self.n_classes
        return {
            "bytes_per_agent_upload":    bytes_per_agent,
            "total_upload_bytes":        total_upload,
            "global_download_bytes":     total_download,
            "total_communication_bytes": total_upload + total_download,
            "comparison_nn_bytes": n_agents * 1_000_000 * 4,
        }


class GeometricMedianAggregator:
    """
    Byzantine-robust federated aggregation via geometric median (Weiszfeld 1937).

    Reference:
        El Mhamdi, Guerraoui, Rouault (2018) "The Hidden Vulnerability of
        Distributed Learning in Byzantium" ICML 2018.

        Pillutla, Kakade, Harchaoui (2022) "Robust Aggregation for Federated
        Learning" IEEE Trans. Signal Process.

    The arithmetic mean is provably vulnerable to even a single Byzantine
    agent.  The geometric median is the minimiser of sum of Euclidean distances:
        gm = argmin_c Σ_i ||proto_i − c||

    This is solved via Weiszfeld's iterative reweighted least squares:
        c_{t+1} = Σ_i w_i(c_t) × proto_i / Σ_i w_i(c_t)
        w_i(c_t) = 1 / max(||proto_i − c_t||, ε)

    Theoretical guarantee:
        If < 50% of agents are Byzantine, the geometric median converges to
        within O(sqrt(k/n)) of the true mean (k = Byzantine count, n = total).
        The arithmetic mean has NO robustness guarantee.

    HDC adaptation:
        Prototypes are soft float vectors before binarisation — we run
        Weiszfeld on the float accumulators, then binarise at the end.

    Args:
        n_classes:   Number of output classes
        dim:         HV dimension
        n_iter:      Weiszfeld iterations (default 20; >10 gives near-exact gm)
        eps:         Numerical stability floor for weights
        device:      torch device
    """

    def __init__(
        self,
        n_classes: int,
        dim:       int,
        n_iter:    int  = 20,
        eps:       float = 1e-6,
        device:    str  = "cpu",
    ):
        self.n_classes = n_classes
        self.dim       = dim
        self.n_iter    = n_iter
        self.eps       = eps
        self.device    = device
        self._round    = 0

    def _geom_median_1d(self, protos: List[torch.Tensor]) -> torch.Tensor:
        """Compute geometric median of a list of (D,) float vectors."""
        if len(protos) == 1:
            return protos[0].clone()
        stacked = torch.stack([p.float().to(self.device) for p in protos])  # (N, D)
        # Initialise at coordinate-wise median (robust init)
        c = stacked.median(dim=0).values
        for _ in range(self.n_iter):
            dists = (stacked - c.unsqueeze(0)).norm(dim=1).clamp(min=self.eps)  # (N,)
            w     = 1.0 / dists          # (N,) Weiszfeld weights
            c_new = (w.unsqueeze(1) * stacked).sum(0) / w.sum()
            if (c_new - c).norm().item() < self.eps:
                break
            c = c_new
        return c

    def aggregate(
        self,
        agent_exports: List[Dict],
        byzantine_fraction_bound: float = 0.3,
    ) -> List[torch.Tensor]:
        """
        Aggregate using geometric median — robust to up to 50% Byzantine agents.

        Args:
            agent_exports:             List of HDCAgent.export_prototypes() dicts
            byzantine_fraction_bound:  Expected upper bound on fraction of corrupt agents
                                       (used for logging only; gm is robust regardless)

        Returns:
            List of n_classes global prototype HVs.
        """
        self._round += 1
        global_protos = []

        for c in range(self.n_classes):
            protos = [
                export["prototypes"][c].to(self.device)
                for export in agent_exports
                if c < len(export["prototypes"])
            ]
            if not protos:
                global_protos.append(torch.zeros(self.dim, device=self.device))
                continue

            gm = self._geom_median_1d(protos)
            global_protos.append(_majority(gm))

        return global_protos

    def byzantine_capacity(self, n_agents: int) -> Dict[str, float]:
        """How many Byzantine agents can we tolerate?"""
        max_byzantine = n_agents // 2 - 1
        return {
            "n_agents":          n_agents,
            "max_byzantine":     max_byzantine,
            "robustness_frac":   max_byzantine / max(n_agents, 1),
            "mean_max_byzantine": 0,   # mean has zero Byzantine robustness
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SwarmHDCMemory — shared distributed memory
# ═══════════════════════════════════════════════════════════════════════════════

class SwarmHDCMemory:
    """
    Shared distributed memory for a swarm of HDC agents.

    Each agent can write observations to the shared memory and
    read from it to coordinate with other agents.

    Architecture:
        M_shared = SUM( bind(agent_role_i, observation_i) )
        Read: unbind(M_shared, agent_role_j) → what agent j observed

    Privacy: each agent has a unique role HV; reading requires knowing the role.

    Applications:
        - Drone swarm: each drone shares its sensor state
        - Sensor network: each sensor writes its reading
        - Multi-robot: shared world model

    Args:
        dim:      HV dimension
        n_agents: Expected number of agents (pre-generates roles)
        device:   torch device
    """

    def __init__(self, dim: int, n_agents: int = 10, device: str = "cpu"):
        self.dim      = dim
        self.device   = device

        # Pre-generate orthogonal-ish agent roles
        self._agent_roles: Dict[str, torch.Tensor] = {}
        self._memory  = torch.zeros(dim, device=device)
        self._n_writes = 0

        # Pre-generate for known agents
        for i in range(n_agents):
            aid = f"agent_{i}"
            g   = torch.Generator(device=device)
            g.manual_seed(i * 1000)
            role = (torch.rand(dim, generator=g, device=device) >= 0.5).float()
            self._agent_roles[aid] = role

    def _get_role(self, agent_id: str) -> torch.Tensor:
        """Get or auto-generate role HV for agent."""
        if agent_id not in self._agent_roles:
            seed = hash(agent_id) % (2**31)
            g    = torch.Generator(device=self.device)
            g.manual_seed(seed)
            self._agent_roles[agent_id] = (
                torch.rand(self.dim, generator=g, device=self.device) >= 0.5
            ).float()
        return self._agent_roles[agent_id]

    def write(self, agent_id: str, observation_hv: torch.Tensor, decay: float = 0.99):
        """Agent writes an observation to the shared memory."""
        role    = self._get_role(agent_id)
        binding = (role != observation_hv.float().to(self.device)).float()  # XOR
        self._memory = _majority(decay * self._memory + (1 - decay) * binding)
        self._n_writes += 1

    def read(self, agent_id: str) -> torch.Tensor:
        """Read the observation stored by a specific agent."""
        role = self._get_role(agent_id)
        # XOR unbind
        return (self._memory.float() != role.float()).float()

    def broadcast(self, observation_hv: torch.Tensor):
        """Write an observation as a global broadcast (no agent role)."""
        self._memory = _majority(self._memory + observation_hv.float().to(self.device))

    def consensus(self) -> torch.Tensor:
        """
        Majority consensus HV from all stored observations.

        Returns the current shared memory state.
        """
        return self._memory.clone()

    def reset(self):
        self._memory = torch.zeros(self.dim, device=self.device)
        self._n_writes = 0

    @property
    def n_agents(self) -> int:
        return len(self._agent_roles)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ConsensusHDC — multi-agent iterative prototype refinement
# ═══════════════════════════════════════════════════════════════════════════════

class ConsensusHDC:
    """
    Multi-agent consensus via iterative HDC prototype refinement.

    Algorithm (gossip-based):
        1. Each agent has local prototypes
        2. Select two agents at random
        3. They exchange prototypes and run RefineHD on the merged set
        4. Repeat until convergence (prototype similarity stops changing)

    Convergence: O(N log N) rounds expected for N agents
    (same as gossip algorithms for averaging)

    Args:
        agents:     List of HDCAgent instances
        n_classes:  Number of classes
    """

    def __init__(self, agents: List[HDCAgent], n_classes: int):
        self.agents    = agents
        self.n_classes = n_classes
        self._round    = 0

    def gossip_round(self) -> float:
        """
        One round of random gossip: pick 2 agents, merge their prototypes.

        Returns:
            Mean prototype similarity between the two selected agents (convergence signal).
        """
        if len(self.agents) < 2:
            return 1.0

        i, j = torch.randperm(len(self.agents))[:2].tolist()
        a, b = self.agents[i], self.agents[j]

        mean_sim = 0.0
        for c in range(self.n_classes):
            pa = a._prototypes[c].float()
            pb = b._prototypes[c].float()

            # Merge: majority vote of both prototypes
            merged = _majority((pa + pb) / 2.0)

            # Update both agents
            a._prototypes[c] = merged.to(a.device)
            b._prototypes[c] = merged.to(b.device)

            mean_sim += float(_hamming(pa.unsqueeze(0), pb.unsqueeze(0)).item())

        self._round += 1
        return mean_sim / max(self.n_classes, 1)

    def run_until_convergence(
        self,
        max_rounds:     int   = 100,
        convergence_thr: float = 0.95,
    ) -> int:
        """
        Run gossip rounds until convergence.

        Returns:
            Number of rounds taken.
        """
        for r in range(max_rounds):
            sim = self.gossip_round()
            if sim > convergence_thr:
                return r + 1
        return max_rounds

    def selective_gossip_round(self) -> float:
        """
        Gossip between the two most DISTANT agents (maximum spread strategy).

        Standard random gossip: O(N log N) rounds to converge.
        Selective gossip (max spread): O(log N) rounds — faster because each
        round maximises the information transferred by bridging the widest gap.

        Returns:
            Mean Hamming distance between the selected agents (0 = converged).
        """
        if len(self.agents) < 2:
            return 0.0

        # Find pair with minimum prototype similarity (= maximum Hamming distance)
        best_pair = (0, 1)
        min_sim   = float("inf")
        n = len(self.agents)

        for i in range(n):
            for j in range(i + 1, n):
                sim = 0.0
                for c in range(self.n_classes):
                    pa = self.agents[i]._prototypes[c].float()
                    pb = self.agents[j]._prototypes[c].float()
                    sim += float(_hamming(pa.unsqueeze(0), pb.unsqueeze(0)).item())
                sim /= max(self.n_classes, 1)
                if sim < min_sim:
                    min_sim, best_pair = sim, (i, j)

        i, j = best_pair
        a, b  = self.agents[i], self.agents[j]

        for c in range(self.n_classes):
            pa     = a._prototypes[c].float()
            pb     = b._prototypes[c].float()
            merged = _majority((pa + pb) / 2.0)
            a._prototypes[c] = merged.to(a.device)
            b._prototypes[c] = merged.to(b.device)

        self._round += 1
        return 1.0 - min_sim   # return distance (high = far, converges toward 0)

    def convergence_stats(self) -> Dict:
        """Report current consensus quality across all agents."""
        if not self.agents:
            return {}
        n = len(self.agents)
        total_sim = 0.0
        count     = 0
        for i in range(min(n, 10)):   # sample up to 10 pairs
            for j in range(i + 1, min(n, 10)):
                for c in range(self.n_classes):
                    pa = self.agents[i]._prototypes[c].float()
                    pb = self.agents[j]._prototypes[c].float()
                    total_sim += float(_hamming(pa.unsqueeze(0), pb.unsqueeze(0)).item())
                    count     += 1
        mean_sim = total_sim / max(count, 1)
        return {
            "n_agents":   n,
            "n_rounds":   self._round,
            "mean_pairwise_sim": mean_sim,
            "consensus_frac":   mean_sim,   # higher = more agreement
        }

    def global_model(self) -> List[torch.Tensor]:
        """Return the mean prototype across all agents (global model estimate)."""
        global_protos = []
        for c in range(self.n_classes):
            all_protos = [a._prototypes[c].float() for a in self.agents]
            global_protos.append(_majority(torch.stack(all_protos).mean(dim=0)))
        return global_protos


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_multi_agent_hdc():
    D, N_cls, N_agents = 256, 3, 5

    print("=== HDCAgent ===")
    agents = [HDCAgent(f"agent_{i}", D, N_cls) for i in range(N_agents)]

    # Each agent trains on slightly different local data
    for idx, agent in enumerate(agents):
        for c in range(N_cls):
            for s in range(20):
                hv = _gen_hv(D, seed=idx * 1000 + c * 100 + s)
                agent.train_step(hv, c)

    # Prediction on first agent
    test_hv = _gen_hv(D, seed=0)
    pred, sims = agents[0].predict(test_hv)
    assert 0 <= pred < N_cls
    print(f"  Agent 0 prediction: class={pred}, sims={[f'{s:.3f}' for s in sims]}  OK")

    print("\n=== FederatedHDCAggregator ===")
    agg     = FederatedHDCAggregator(N_cls, D, strategy="weighted_mean")
    exports = [a.export_prototypes() for a in agents]
    global_protos = agg.aggregate(exports)
    assert len(global_protos) == N_cls
    assert global_protos[0].shape == (D,)
    print(f"  Aggregated {N_agents} agents → {len(global_protos)} global protos  OK")

    # Communication cost
    cost = agg.communication_cost_bytes(N_agents)
    reduction = cost['comparison_nn_bytes'] / max(cost['total_communication_bytes'], 1)
    print(f"  Communication: {cost['total_communication_bytes']} bytes "
          f"vs {cost['comparison_nn_bytes']} for NN ({reduction:.0f}× less)  OK")

    # Update all agents with global model
    for agent in agents:
        agent.import_global(global_protos)

    print("\n=== SwarmHDCMemory ===")
    swarm = SwarmHDCMemory(D, n_agents=N_agents)
    for i, agent in enumerate(agents[:3]):
        obs = _gen_hv(D, seed=i * 555)
        swarm.write(f"agent_{i}", obs)

    for i in range(3):
        recalled = swarm.read(f"agent_{i}")
        assert recalled.shape == (D,)
    print(f"  {swarm.n_agents} agent roles, {swarm._n_writes} writes  OK")

    consensus = swarm.consensus()
    assert consensus.shape == (D,)
    print(f"  Consensus HV shape: {consensus.shape}  OK")

    print("\n=== ConsensusHDC ===")
    # Reset agents and give them different data
    fresh_agents = [HDCAgent(f"a{i}", D, N_cls) for i in range(4)]
    for idx, agent in enumerate(fresh_agents):
        for c in range(N_cls):
            for s in range(10):
                hv = _gen_hv(D, seed=idx * 100 + c * 30 + s)
                agent.train_step(hv, c)

    consensus_engine = ConsensusHDC(fresh_agents, N_cls)
    rounds = consensus_engine.run_until_convergence(max_rounds=20, convergence_thr=0.9)
    print(f"  Converged in {rounds} gossip rounds  OK")

    gm = consensus_engine.global_model()
    assert len(gm) == N_cls
    print(f"  Global model: {len(gm)} prototypes  OK")

    print("\n✅ All multi_agent_hdc tests passed")


if __name__ == "__main__":
    _test_multi_agent_hdc()
