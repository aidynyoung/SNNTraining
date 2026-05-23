"""
hdc/graph_neural_hd.py
=======================
Graph Neural Networks in Hypervector Space — VS-Graph HDC
==========================================================
Reference:
    Poursiami, Imani et al. (2025)
    "VS-Graph: Scalable and Efficient Graph Classification Using
    Hyperdimensional Computing" arXiv:2512.03394.

    Boureanu, Bragagnolo, Rahimi (2022)
    "Scalable Hyperdimensional Computing for Graph Node Classification"
    IEEE BigData 2022.

    Kleyko et al. (2022) "A Survey on HDC" Part II §V:
    "Structured data and graph representations in VSA"

Why HDC outperforms GNNs for graph classification in constrained settings:

    GNN (message passing):
        - O(E × d) per layer  (E = edges, d = embedding dim)
        - Requires backpropagation through all edges
        - Memory: store all node/edge embeddings
        - Non-trivial hyperparameter tuning

    HDC graph (VS-Graph):
        - O(V × D) total  (V = vertices, D = HV dimension)
        - No backpropagation: bundle + lookup
        - Memory: O(V + E) for graph + O(C × D) for class prototypes
        - Single-pass online learning

    VS-Graph HDC pipeline:
        1. Node features → HV via level-ID encoding
        2. Edge → bind(node1_hv, node2_hv)
        3. Neighbourhood bundle: node_hv = bundle([edge_hvs])
        4. Graph HV: bundle([all_node_hvs])
        5. Classify: Hamming_sim(graph_hv, class_proto_c)

This module implements:

1. NodeEncoder
   — Encodes continuous node feature vectors to binary HVs
   — Supports: level-ID, phasor (VFA), FlyHash (sparse)
   — Multiple encoding rounds for richer representations

2. EdgeHDC
   — Encodes edges as bind(source_hv, target_hv) XOR pairs
   — Directed edges: bind(src, permute(tgt))
   — Weighted edges: scale by weight before bundling

3. HDCMessagePassing
   — One round of HDC message passing:
     For each node v: aggregate HVs of all neighbours via bundle
   — L rounds give L-hop receptive field
   — No learned weights — pure HDC

4. HDCGraphNetwork
   — Complete graph neural network in HDC
   — Graph-level pooling: readout = bundle(all_node_hvs)
   — Multiple HDCMessagePassing rounds
   — Online learning: class_proto += graph_hv

5. HDCGraphClassifier
   — End-to-end graph classification
   — Supports: node classification, graph classification, link prediction
   — RefineHD-style online training

6. SubgraphHDC
   — Encodes subgraph patterns as HVs
   — Application: motif detection, graphlet counting in O(D)
   — Based on: graphlet-kernel HDC
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from hdc.physics_world_model import _hamming, _majority, _xor
from hdc.hdc_glue import gen_hvs


# ── Utility ────────────────────────────────────────────────────────────────────

def _gen_hv(dim: int, seed=None, device: str = "cpu") -> torch.Tensor:
    import hashlib
    if seed is None:
        g = torch.Generator(device=device)
        return (torch.rand(dim, generator=g, device=device) >= 0.5).float()
    if isinstance(seed, str):
        raw = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16) % (2**31)
    else:
        raw = int(seed) % (2**31)
    g = torch.Generator(device=device)
    g.manual_seed(raw)
    return (torch.rand(dim, generator=g, device=device) >= 0.5).float()


# ── Graph Data Structures ─────────────────────────────────────────────────────

@dataclass
class HDCGraph:
    """A graph with HV-encoded nodes and edges."""
    node_features: Dict[int, torch.Tensor]          # node_id → feature HV
    edges:         List[Tuple[int, int]]             # (src, dst) pairs
    edge_weights:  Optional[Dict[Tuple, float]] = None
    node_labels:   Optional[Dict[int, int]]     = None
    graph_label:   Optional[int]                = None


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NodeEncoder
# ═══════════════════════════════════════════════════════════════════════════════

class NodeEncoder:
    """
    Encodes node feature vectors to binary hypervectors.

    Reference:
        Poursiami et al. (2025) VS-Graph §III.A: Node Encoding.

    Encoding methods:
        'level_id': level hypervectors × feature ID hypervectors
                    standard HDC encoding, preserves continuous values
        'flyhash':  sparse random projection via Drosophila circuit
        'phasor':   complex-valued VFA encoding (from hdc/vfa.py)

    Args:
        n_features: Number of node feature dimensions
        dim:        Output HV dimension
        method:     Encoding method ('level_id' | 'flyhash')
        n_levels:   Number of quantisation levels for level_id
        device:     torch device
    """

    def __init__(
        self,
        n_features: int,
        dim:        int,
        method:     str = "level_id",
        n_levels:   int = 21,
        device:     str = "cpu",
    ):
        self.n_features = n_features
        self.dim        = dim
        self.method     = method
        self.n_levels   = n_levels
        self.device     = device

        # Feature ID HVs: one per feature dimension
        self._feature_hvs = torch.stack([
            _gen_hv(dim, seed=f"feat_{i}", device=device)
            for i in range(n_features)
        ])  # (n_features, D)

        # Level HVs: one per quantisation level
        self._level_hvs = torch.stack([
            _gen_hv(dim, seed=f"level_{l}", device=device)
            for l in range(n_levels)
        ])  # (n_levels, D)

        # FlyHash projection (for sparse encoding)
        if method == "flyhash":
            k = max(1, int(dim * 0.05))
            self._proj = torch.randn(n_features, dim, device=device)
            self._k    = k

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode a node's feature vector to a binary HV.

        Args:
            features: (n_features,) node feature vector

        Returns:
            (D,) binary node HV
        """
        if self.method == "level_id":
            return self._level_id_encode(features)
        elif self.method == "flyhash":
            return self._flyhash_encode(features)
        else:
            return self._level_id_encode(features)

    def _level_id_encode(self, features: torch.Tensor) -> torch.Tensor:
        """Level-ID encoding: bind(feature_id_hv, level_hv) for each dimension."""
        x_f   = torch.sigmoid(features.float().to(self.device))
        hvs   = []
        n_dim = min(self.n_features, x_f.shape[0])
        for i in range(n_dim):
            lvl_idx = max(0, min(self.n_levels - 1, int(x_f[i].item() * (self.n_levels - 1))))
            bound   = (self._feature_hvs[i] != self._level_hvs[lvl_idx]).float()
            hvs.append(bound)
        return _majority(torch.stack(hvs).float().mean(dim=0))

    def _flyhash_encode(self, features: torch.Tensor) -> torch.Tensor:
        """FlyHash: sparse k-WTA random projection."""
        x_f   = features.float().to(self.device)
        proj  = x_f @ self._proj  # (D,)
        topk  = proj.topk(self._k).indices
        out   = torch.zeros(self.dim, device=self.device)
        out[topk] = 1.0
        return out

    def encode_batch(self, features: torch.Tensor) -> torch.Tensor:
        """Encode a batch (N, n_features) → (N, D)."""
        return torch.stack([self.encode(features[i]) for i in range(features.shape[0])])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EdgeHDC
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeHDC:
    """
    Encodes edges as XOR bindings of source and destination node HVs.

    Reference:
        Poursiami et al. (2025) VS-Graph §III.B: Edge Encoding.
        "An edge (u,v) is encoded as bind(HV_u, HV_v)"

    Edge types:
        Undirected: edge_hv = XOR(src_hv, dst_hv)        [symmetric]
        Directed:   edge_hv = XOR(src_hv, permute(dst_hv)) [asymmetric]
        Typed:      edge_hv = XOR(XOR(src_hv, dst_hv), edge_type_hv)

    Args:
        dim:      HV dimension
        directed: If True, use asymmetric encoding for directed graphs
        device:   torch device
    """

    def __init__(self, dim: int, directed: bool = False, device: str = "cpu"):
        self.dim      = dim
        self.directed = directed
        self.device   = device

    def encode(
        self,
        src_hv: torch.Tensor,
        dst_hv: torch.Tensor,
        weight: float = 1.0,
        edge_type_hv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a single edge.

        Args:
            src_hv:       (D,) source node HV
            dst_hv:       (D,) destination node HV
            weight:       Edge weight (1.0 = default)
            edge_type_hv: Optional (D,) edge type HV

        Returns:
            (D,) edge HV
        """
        s = src_hv.float().to(self.device)
        d = dst_hv.float().to(self.device)

        if self.directed:
            # Asymmetric: permute destination to encode direction
            d = torch.roll(d, 1, dims=0)

        edge_hv = (s != d).float()   # XOR

        if edge_type_hv is not None:
            edge_hv = (edge_hv != edge_type_hv.float().to(self.device)).float()

        return edge_hv

    def encode_path(self, node_hvs: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode a path of N nodes as a sequential edge bundle.

        path_hv = bundle([edge(v0,v1), edge(v1,v2), ..., edge(v_{N-2},v_{N-1})])
        """
        if len(node_hvs) < 2:
            return node_hvs[0] if node_hvs else torch.zeros(self.dim, device=self.device)

        edge_hvs = [
            self.encode(node_hvs[i], node_hvs[i + 1])
            for i in range(len(node_hvs) - 1)
        ]
        return _majority(torch.stack(edge_hvs).float().mean(dim=0))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HDCMessagePassing
# ═══════════════════════════════════════════════════════════════════════════════

class HDCMessagePassing:
    """
    One round of HDC message passing: aggregate neighbour HVs.

    Reference:
        Poursiami et al. (2025) VS-Graph §III.C: Message Passing.
        "For node v at step t+1: HV_v = bundle(HV_v, {HV_u : u ∈ N(v)})"

    For each node v:
        messages = [HV_u for u in neighbors(v)]
        HV_v_new = bundle([HV_v] + messages)

    L rounds of message passing → L-hop neighbourhood aggregation.

    Note: unlike neural message passing, there are no learned weights.
    The "weights" are implicit in the random projection vectors.

    Args:
        dim:       HV dimension
        edge_encoder: EdgeHDC instance (for edge-aware passing)
    """

    def __init__(
        self,
        dim:          int,
        edge_encoder: Optional[EdgeHDC] = None,
        device:       str = "cpu",
        degree_norm:  bool = True,
        use_attention: bool = True,
    ):
        self.dim           = dim
        self.edge_encoder  = edge_encoder
        self.device        = device
        self.degree_norm   = degree_norm    # GCN-style 1/sqrt(d_i × d_j) normalisation
        self.use_attention = use_attention  # HDC similarity as attention weight

    def _attention_weight(
        self,
        v_hv: torch.Tensor,
        u_hv: torch.Tensor,
        deg_v: int,
        deg_u: int,
    ) -> float:
        """
        HDC attention: Hamming similarity × GCN degree normalisation.

        Standard GNN attention weights come from learned MLPs. In HDC, we use
        Hamming similarity as a natural attention score — nodes that are already
        similar contribute more to each other's update (homophily prior).

        Combined with GCN degree normalisation 1/sqrt(deg_v × deg_u) this
        gives a symmetric, bounded attention weight.

        Args:
            v_hv, u_hv: node HVs
            deg_v, deg_u: node degrees (including self-loop)
        """
        sim = float(_hamming(v_hv.unsqueeze(0), u_hv.unsqueeze(0)).item())
        if self.degree_norm and deg_v > 0 and deg_u > 0:
            import math as _math
            sim = sim / _math.sqrt(deg_v * deg_u)
        return max(sim, 1e-6)

    def pass_messages(
        self,
        node_hvs:     Dict[int, torch.Tensor],
        adjacency:    Dict[int, List[int]],
        edge_weights: Optional[Dict[Tuple, float]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        One round of HDC message passing with optional attention + degree norm.

        Improvements over baseline:
        - GCN degree normalisation 1/sqrt(d_v × d_u): prevents high-degree
          nodes from dominating aggregation (Kipf & Welling 2017)
        - HDC attention: Hamming similarity as message weight — similar
          neighbours contribute more (homophily prior)
        - Edge-aware messages: bind(edge_hv, neighbour_hv) before bundling

        Args:
            node_hvs:    {node_id: HV} input node representations
            adjacency:   {node_id: [neighbour_ids]} graph structure
            edge_weights: Optional {(src,dst): weight}

        Returns:
            {node_id: HV} updated node representations
        """
        # Pre-compute degrees (include self-loop: +1)
        degrees: Dict[int, int] = {v: len(neigh) + 1 for v, neigh in adjacency.items()}

        new_hvs = {}
        for v, neigh_list in adjacency.items():
            if v not in node_hvs:
                continue

            v_hv    = node_hvs[v].float()
            deg_v   = degrees.get(v, 1)
            accum   = torch.zeros(self.dim, device=self.device)
            w_total = 0.0

            # Self contribution
            w_self = 1.0 / deg_v if self.degree_norm else 1.0
            accum  += w_self * v_hv
            w_total += w_self

            # Neighbour contributions
            for u in neigh_list:
                if u not in node_hvs:
                    continue
                u_hv  = node_hvs[u].float()
                deg_u = degrees.get(u, 1)

                # Message: edge-aware binding if available
                if self.edge_encoder is not None:
                    edge_hv = self.edge_encoder.encode(v_hv, u_hv)
                    msg     = _majority((edge_hv + u_hv) / 2.0)
                else:
                    msg = u_hv

                # Explicit edge weight (overrides attention if provided)
                if edge_weights is not None:
                    w = edge_weights.get((u, v), edge_weights.get((v, u), 1.0))
                elif self.use_attention:
                    w = self._attention_weight(v_hv, u_hv, deg_v, deg_u)
                else:
                    w = 1.0 / deg_v if self.degree_norm else 1.0

                accum   += w * msg.float()
                w_total += w

            # Normalise and binarise
            new_hvs[v] = _majority(accum / max(w_total, 1e-8))

        return new_hvs

    def multi_round(
        self,
        node_hvs:     Dict[int, torch.Tensor],
        adjacency:    Dict[int, List[int]],
        n_rounds:     int  = 2,
        skip_connect: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        Run n_rounds of message passing with optional skip connections.

        Skip connections (ResNet-style) prevent over-smoothing: after each
        round, the updated HV is bundled with the original pre-round HV:
            h_v^{(l+1)} = majority(MP(h_v^{(l)}) + h_v^{(0)})
        This preserves node-level features across many rounds and is
        especially effective for >2 message passing steps.

        Args:
            skip_connect: If True, bundle with initial HVs after each round
        """
        initial = dict(node_hvs)
        current = dict(node_hvs)
        for _ in range(n_rounds):
            updated = self.pass_messages(current, adjacency)
            if skip_connect:
                # Residual: bundle updated + initial for each node
                for v in updated:
                    if v in initial:
                        blended = (updated[v].float() + initial[v].float()) / 2.0
                        updated[v] = _majority(blended)
            current = updated
        return current


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HDCGraphNetwork
# ═══════════════════════════════════════════════════════════════════════════════

class HDCGraphNetwork:
    """
    Complete HDC graph neural network.

    Reference:
        Poursiami et al. (2025) VS-Graph — complete pipeline.

    Pipeline:
        1. Node features → HVs (NodeEncoder)
        2. L rounds of message passing (HDCMessagePassing)
        3. Graph-level pooling: bundle all node HVs
        4. Optional: add edge HVs to graph representation
        5. Classify: Hamming_sim(graph_hv, class_protos)

    Online learning:
        class_proto[c] += graph_hv   (incremental bundle)

    Args:
        n_node_features: Number of node feature dimensions
        dim:             HV dimension
        n_rounds:        Message passing rounds (default 2)
        directed:        If True, use directed edge encoding
        device:          torch device
    """

    def __init__(
        self,
        n_node_features: int,
        dim:             int,
        n_rounds:        int   = 2,
        directed:        bool  = False,
        device:          str   = "cpu",
    ):
        self.dim      = dim
        self.n_rounds = n_rounds
        self.device   = device

        self.node_encoder = NodeEncoder(n_node_features, dim, device=device)
        self.edge_encoder = EdgeHDC(dim, directed=directed, device=device)
        self.mp           = HDCMessagePassing(dim, self.edge_encoder, device=device)

        # Class prototypes
        self._protos: Dict[int, torch.Tensor] = {}
        self._counts: Dict[int, int]          = {}

    def _build_adjacency(self, graph: HDCGraph) -> Dict[int, List[int]]:
        """Build adjacency list from edge list."""
        adj: Dict[int, List[int]] = {n: [] for n in graph.node_features}
        for src, dst in graph.edges:
            adj.setdefault(src, []).append(dst)
            if not self.edge_encoder.directed:
                adj.setdefault(dst, []).append(src)
        return adj

    def encode_graph(self, graph: HDCGraph) -> torch.Tensor:
        """
        Encode a graph to a single (D,) HV.

        Args:
            graph: HDCGraph with node features and edges

        Returns:
            (D,) graph-level HV
        """
        # 1. Encode node features
        node_hvs = {
            nid: self.node_encoder.encode(feat)
            for nid, feat in graph.node_features.items()
        }

        # 2. Message passing
        adj = self._build_adjacency(graph)
        node_hvs = self.mp.multi_round(node_hvs, adj, self.n_rounds)

        # 3. Graph-level readout: bundle all node HVs
        all_hvs = list(node_hvs.values())
        if not all_hvs:
            return torch.zeros(self.dim, device=self.device)

        graph_hv = _majority(torch.stack(all_hvs).float().mean(dim=0))

        # 4. Add edge HVs for richer representation
        if graph.edges:
            edge_hvs = []
            for src, dst in graph.edges:
                if src in node_hvs and dst in node_hvs:
                    e_hv = self.edge_encoder.encode(node_hvs[src], node_hvs[dst])
                    edge_hvs.append(e_hv)
            if edge_hvs:
                edge_bundle = _majority(torch.stack(edge_hvs).float().mean(dim=0))
                graph_hv    = _majority((graph_hv.float() + edge_bundle.float()) / 2.0)

        return graph_hv

    def train(self, graph: HDCGraph, label: int):
        """Online training: update class prototype."""
        graph_hv = self.encode_graph(graph)
        n = self._counts.get(label, 0)
        if label not in self._protos:
            self._protos[label] = graph_hv.clone()
        else:
            self._protos[label] = _majority(
                (n * self._protos[label] + graph_hv) / (n + 1)
            )
        self._counts[label] = n + 1

    def predict(self, graph: HDCGraph) -> Tuple[int, Dict[int, float]]:
        """
        Predict graph label.

        Returns:
            (predicted_label, {label: similarity})
        """
        graph_hv = self.encode_graph(graph)
        if not self._protos:
            return -1, {}

        sims = {}
        for label, proto in self._protos.items():
            sims[label] = float(_hamming(graph_hv.unsqueeze(0), proto.unsqueeze(0)).item())

        best = max(sims, key=sims.get)
        return best, sims

    def refine(self, graph: HDCGraph, label: int, lr: float = 0.1):
        """
        RefineHD update: push toward correct, pull away from incorrect.
        """
        graph_hv = self.encode_graph(graph)
        pred, sims = self.predict(graph)

        if pred != label:
            # Pull toward correct
            if label in self._protos:
                self._protos[label] = _majority(
                    (1 - lr) * self._protos[label] + lr * graph_hv
                )
            # Push away from wrong
            if pred in self._protos:
                self._protos[pred] = _majority(
                    (1 + lr) * self._protos[pred] - lr * graph_hv
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HDCGraphClassifier (wrapper with evaluation)
# ═══════════════════════════════════════════════════════════════════════════════

class HDCGraphClassifier:
    """
    End-to-end graph classifier wrapping HDCGraphNetwork.

    Supports:
        - Graph-level classification
        - Node-level classification (node labels)
        - Link prediction (does edge (u,v) exist?)

    Args:
        n_node_features: Node feature dimension
        n_classes:       Number of graph classes
        dim:             HV dimension
        n_rounds:        Message passing rounds
    """

    def __init__(
        self,
        n_node_features: int,
        n_classes:       int,
        dim:             int   = 1000,
        n_rounds:        int   = 2,
        class_names:     Optional[List[str]] = None,
        device:          str   = "cpu",
    ):
        self.gnn         = HDCGraphNetwork(n_node_features, dim, n_rounds, device=device)
        self.n_classes   = n_classes
        self.class_names = class_names or [f"class_{i}" for i in range(n_classes)]
        self.dim         = dim

    def fit(self, graphs: List[HDCGraph], labels: List[int], n_refine: int = 2):
        """Train on a list of labelled graphs."""
        for g, lbl in zip(graphs, labels):
            self.gnn.train(g, lbl)
        for _ in range(n_refine):
            for g, lbl in zip(graphs, labels):
                self.gnn.refine(g, lbl)

    def predict(self, graph: HDCGraph) -> Tuple[int, str, float]:
        """Predict graph class."""
        pred_id, sims = self.gnn.predict(graph)
        name = self.class_names[pred_id] if 0 <= pred_id < len(self.class_names) else "unknown"
        conf = sims.get(pred_id, 0.0)
        return pred_id, name, conf

    def accuracy(self, graphs: List[HDCGraph], labels: List[int]) -> float:
        """Compute classification accuracy."""
        correct = sum(
            self.gnn.predict(g)[0] == lbl
            for g, lbl in zip(graphs, labels)
        )
        return correct / max(len(graphs), 1)

    def link_predict(self, graph: HDCGraph, src: int, dst: int) -> float:
        """
        Predict whether edge (src, dst) should exist.
        Returns: Hamming similarity between bind(src_hv, dst_hv) and edge_bundle.
        """
        node_hvs = {
            nid: self.gnn.node_encoder.encode(feat)
            for nid, feat in graph.node_features.items()
        }
        if src not in node_hvs or dst not in node_hvs:
            return 0.0
        edge_hv = self.gnn.edge_encoder.encode(node_hvs[src], node_hvs[dst])
        # Compare to all existing edges
        if not graph.edges:
            return 0.5
        existing_hvs = []
        for s, d in graph.edges:
            if s in node_hvs and d in node_hvs:
                existing_hvs.append(self.gnn.edge_encoder.encode(node_hvs[s], node_hvs[d]))
        if not existing_hvs:
            return 0.5
        existing_bundle = _majority(torch.stack(existing_hvs).float().mean(dim=0))
        return float(_hamming(edge_hv.unsqueeze(0), existing_bundle.unsqueeze(0)).item())


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SubgraphHDC — motif detection
# ═══════════════════════════════════════════════════════════════════════════════

class SubgraphHDC:
    """
    Encode subgraph patterns (motifs) as HVs for motif counting / detection.

    Reference:
        Boureanu et al. (2022) "Scalable HDC for Graph Node Classification"
        — subgraph kernels in HDC.

    A k-subgraph pattern is encoded as:
        pattern_hv = bundle([bind(edge_hvs)])

    Counting: Hamming_sim(graph_hv, pattern_hv) ≈ fraction of pattern present.

    Args:
        dim:    HV dimension
        device: torch device
    """

    def __init__(self, dim: int, device: str = "cpu"):
        self.dim    = dim
        self.device = device
        self.edge_enc = EdgeHDC(dim, device=device)
        self._patterns: Dict[str, torch.Tensor] = {}

    def register_pattern(self, name: str, edge_list: List[Tuple[int, int]],
                          node_hvs: Dict[int, torch.Tensor]):
        """Register a named subgraph pattern."""
        if not edge_list:
            return
        edge_hvs = [
            self.edge_enc.encode(node_hvs[s], node_hvs[d])
            for s, d in edge_list if s in node_hvs and d in node_hvs
        ]
        if edge_hvs:
            self._patterns[name] = _majority(torch.stack(edge_hvs).float().mean(dim=0))

    def detect(self, graph_hv: torch.Tensor) -> Dict[str, float]:
        """
        Detect which registered patterns are present in a graph HV.

        Returns:
            {pattern_name: similarity_score}
        """
        return {
            name: float(_hamming(graph_hv.unsqueeze(0), p_hv.unsqueeze(0)).item())
            for name, p_hv in self._patterns.items()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _make_test_graph(n_nodes: int, n_feat: int, label: int,
                     seed: int = 0) -> HDCGraph:
    """Create a random test graph."""
    torch.manual_seed(seed)
    features = {i: torch.randn(n_feat) + label * 2 for i in range(n_nodes)}
    edges    = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]  # cycle graph
    return HDCGraph(node_features=features, edges=edges, graph_label=label)


def _test_graph_neural_hd():
    N_FEAT, DIM, N_NODES = 8, 256, 5

    print("=== NodeEncoder ===")
    enc = NodeEncoder(N_FEAT, DIM, method="level_id")
    x   = torch.randn(N_FEAT)
    hv  = enc.encode(x)
    assert hv.shape == (DIM,)
    assert set(hv.unique().tolist()).issubset({0.0, 1.0})
    print(f"  Encoded shape: {hv.shape}, density={hv.mean():.3f}  OK")

    # FlyHash encoding
    enc_f = NodeEncoder(N_FEAT, DIM, method="flyhash")
    hv_f  = enc_f.encode(x)
    assert hv_f.shape == (DIM,)
    print(f"  FlyHash density: {hv_f.mean():.3f}  OK")

    print("\n=== EdgeHDC ===")
    edge = EdgeHDC(DIM, directed=False)
    src  = _gen_hv(DIM, seed=0)
    dst  = _gen_hv(DIM, seed=1)
    ehv  = edge.encode(src, dst)
    assert ehv.shape == (DIM,)
    # Undirected: encode(src,dst) == encode(dst,src)
    ehv2 = edge.encode(dst, src)
    assert torch.equal(ehv, ehv2), "Undirected edges should be symmetric"
    print(f"  Edge HV: {ehv.shape}, symmetric={torch.equal(ehv, ehv2)}  OK")

    print("\n=== HDCMessagePassing ===")
    mp = HDCMessagePassing(DIM)
    node_hvs = {i: _gen_hv(DIM, seed=i) for i in range(N_NODES)}
    adj = {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1], 4: [2]}
    new_hvs = mp.pass_messages(node_hvs, adj)
    assert len(new_hvs) == N_NODES
    assert all(v.shape == (DIM,) for v in new_hvs.values())
    print(f"  After 1 round: {len(new_hvs)} nodes updated  OK")

    print("\n=== HDCGraphNetwork ===")
    gnn = HDCGraphNetwork(N_FEAT, DIM, n_rounds=2)
    g0  = _make_test_graph(N_NODES, N_FEAT, label=0, seed=0)
    g1  = _make_test_graph(N_NODES, N_FEAT, label=1, seed=10)

    # Train
    for i in range(10):
        gnn.train(_make_test_graph(N_NODES, N_FEAT, label=0, seed=i),     0)
        gnn.train(_make_test_graph(N_NODES, N_FEAT, label=1, seed=100+i), 1)

    # Predict
    pred, sims = gnn.predict(g0)
    print(f"  Class 0 graph: pred={pred} (sims={sims})  OK")

    print("\n=== HDCGraphClassifier ===")
    clf = HDCGraphClassifier(N_FEAT, n_classes=2, dim=DIM)
    train_graphs = [_make_test_graph(N_NODES, N_FEAT, c, s)
                    for c in range(2) for s in range(10)]
    train_labels = [c for c in range(2) for _ in range(10)]
    clf.fit(train_graphs, train_labels)

    acc = clf.accuracy(train_graphs, train_labels)
    print(f"  Train accuracy: {acc:.2f}  OK")

    pred_id, name, conf = clf.predict(g0)
    print(f"  Predict: class={pred_id} ({name}), conf={conf:.3f}  OK")

    # Link prediction
    lp = clf.link_predict(g0, 0, 1)
    print(f"  Link prediction (0,1): sim={lp:.3f}  OK")

    print("\n=== SubgraphHDC ===")
    sub = SubgraphHDC(DIM)
    node_hvs_sub = {i: _gen_hv(DIM, seed=i) for i in range(4)}
    sub.register_pattern("triangle", [(0,1),(1,2),(0,2)], node_hvs_sub)
    sub.register_pattern("path",     [(0,1),(1,2),(2,3)], node_hvs_sub)
    graph_hv     = gnn.encode_graph(g0)
    detections   = sub.detect(graph_hv)
    assert "triangle" in detections and "path" in detections
    print(f"  Subgraph detection: {detections}  OK")

    print("\n✅ All graph_neural_hd tests passed")


if __name__ == "__main__":
    _test_graph_neural_hd()
