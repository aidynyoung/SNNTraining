"""
VS-Graph: Graph Classification with Hyperdimensional Computing
==============================================================
Poursiami, Snyder, Cong, Potok & Parsa (2025)
"VS-Graph: Scalable and Efficient Graph Classification Using
Hyperdimensional Computing"
arXiv:2512.03394 — George Mason University / Oak Ridge National Laboratory

Bridges the accuracy gap between HDC-based graph methods and GNNs by
introducing message-passing in HV space — matching or exceeding GCN, GAT
accuracy on 5 benchmarks while being 200-500× faster.

Algorithm (§III of paper):

1. Node encoding (initial HVs):
     For each node i: compute PageRank score → discretize to rank bin r_i
     h_i^{(0)} = item_memory[r_i]      [look up from random HV codebook]

2. HDC message passing (L layers, Eq. 2-3):
     Aggregate: m_i^{(l)} = OR_{j∈N(i)} h_j^{(l)}    [bitwise OR of neighbors]
     Update:    h_i^{(l+1)} = α·h_i^{(l)} + (1-α)·m_i^{(l)} → binarize
     Key: OR is idempotent — repeated neighbors don't accumulate (no normalization needed)

3. Graph readout (Eq. 4):
     For each edge (u,v): edge_hv = XOR(h_u^{(L)}, h_v^{(L)})   [bind endpoints]
     G = MAJORITY_SUM(all edge HVs)     [bundle = single graph HV]

4. Prototype classification:
     Train: bundle graph HVs per class → class prototype
     Infer: nearest prototype by Hamming similarity

Results (Table I, Fig. 1):
  MUTAG:    88.47% (vs GraphHD 83.99%, GCN 78.88%)
  PROTEINS: 73.29% (vs GraphHD 71.97%, GAT 58.52%)
  DD:       76.46% (vs GraphHD 71.62%)
  Training: 0.142ms/graph vs 61ms for GCN (430× faster)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from hdc.hdc_glue import hv_batch_sim, hv_majority, gen_hvs


# ═══════════════════════════════════════════════════════════════════════════════
# Graph data structure
# ═══════════════════════════════════════════════════════════════════════════════

class Graph:
    """
    Lightweight graph representation for VS-Graph.

    Stores adjacency list and optional node features / labels.
    """

    def __init__(
        self,
        n_nodes: int,
        edges: List[Tuple[int, int]],
        node_features: Optional[torch.Tensor] = None,
        label: Optional[int] = None,
    ):
        self.n_nodes = n_nodes
        self.edges = edges
        self.node_features = node_features  # (n_nodes, feat_dim) or None
        self.label = label

        # Adjacency list
        self.adj: List[List[int]] = [[] for _ in range(n_nodes)]
        for u, v in edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

    def degree(self, node: int) -> int:
        return len(self.adj[node])

    @property
    def n_edges(self) -> int:
        return len(self.edges)


# ═══════════════════════════════════════════════════════════════════════════════
# Node encoding: PageRank → rank bins → HV lookup
# ═══════════════════════════════════════════════════════════════════════════════

def pagerank(graph: Graph, damping: float = 0.85, n_iter: int = 20) -> torch.Tensor:
    """
    Compute PageRank scores for all nodes (Eq. 1-like, §III-A).

    Uses degree-normalized adjacency for the power iteration.

    Returns:
        (n_nodes,) float tensor of PageRank scores
    """
    N = graph.n_nodes
    pr = torch.ones(N) / N

    for _ in range(n_iter):
        new_pr = torch.ones(N) * (1 - damping) / N
        for u in range(N):
            d_u = max(graph.degree(u), 1)
            for v in graph.adj[u]:
                new_pr[v] += damping * pr[u] / d_u
        pr = new_pr

    return pr


def degree_scores(graph: Graph) -> torch.Tensor:
    """
    Fallback to degree-based scores when PageRank is too slow.

    Normalised degree in [0, 1].
    """
    degs = torch.tensor([graph.degree(i) for i in range(graph.n_nodes)], dtype=torch.float)
    max_deg = degs.max().clamp(min=1)
    return degs / max_deg


# ═══════════════════════════════════════════════════════════════════════════════
# VS-Graph encoder
# ═══════════════════════════════════════════════════════════════════════════════

class VSGraph(object):
    """
    VS-Graph: message-passing HDC for graph classification.

    Encodes graphs as single hypervectors using:
      1. PageRank/degree → rank-bin → node HV (from item memory)
      2. L rounds of OR-message-passing with residual
      3. Edge-binding (XOR) + graph-level bundling (majority)
      4. Prototype-based HDC classification

    Advantages over GraphHD (the previous HDC approach):
      - Message passing propagates structural context (multi-hop neighbourhood)
      - OR aggregation is idempotent → no normalisation required
      - Residual connection preserves node identity across layers
      - Matches GCN/GAT accuracy at 200-500× lower training cost

    Args:
        hd_dim: Hypervector dimensionality
        n_rank_bins: Number of rank bins for node encoding (8 recommended)
        n_layers: Message-passing layers L (2-3 recommended)
        alpha: Residual weight in update rule (0.5 default)
        use_pagerank: Use PageRank (True) or degree (False) for node scores
        seed: Random seed
    """

    def __init__(
        self,
        hd_dim: int = 4096,
        n_rank_bins: int = 8,
        n_layers: int = 2,
        alpha: float = 0.5,
        use_pagerank: bool = False,
        seed: int = 42,
    ):
        self.hd_dim = hd_dim
        self.n_rank_bins = n_rank_bins
        self.n_layers = n_layers
        self.alpha = alpha
        self.use_pagerank = use_pagerank

        # Item memory: one random HV per rank bin
        self._rank_hvs = gen_hvs(n_rank_bins, hd_dim, seed=seed)   # (B, D)

        # Node-position HVs: break symmetry when multiple nodes share the same rank.
        # XOR(rank_hv, pos_hv_i) ensures h_i ≠ h_j even if rank_i = rank_j.
        # We use a fixed pool of 64 position HVs cycled mod 64.
        self._n_pos = 64
        self._pos_hvs = gen_hvs(self._n_pos, hd_dim, seed=seed + 999)  # (64, D)

        # Prototype accumulator per class
        self._prototypes: Dict[int, torch.Tensor] = {}
        self._counts: Dict[int, int] = {}

    def _score_to_bin(self, score: float) -> int:
        """Map a node score in [0,1] to a rank bin index."""
        return min(int(score * self.n_rank_bins), self.n_rank_bins - 1)

    def _encode_node_initial(self, graph: Graph) -> torch.Tensor:
        """
        Encode each node to initial HV using rank-bin lookup (§III-A).

        Returns:
            (n_nodes, D) initial node HV matrix
        """
        if self.use_pagerank and graph.n_nodes > 1:
            scores = pagerank(graph)
        else:
            scores = degree_scores(graph)

        node_hvs = torch.zeros(graph.n_nodes, self.hd_dim)
        for i, sc in enumerate(scores.tolist()):
            bin_idx = self._score_to_bin(sc)
            # XOR rank HV with positional HV to break symmetry
            rank_hv = self._rank_hvs[bin_idx]
            pos_hv  = self._pos_hvs[i % self._n_pos]
            node_hvs[i] = (rank_hv != pos_hv).float()  # XOR

        return node_hvs  # (N, D) binary

    def _message_pass(self, graph: Graph, node_hvs: torch.Tensor) -> torch.Tensor:
        """
        L rounds of OR-message-passing with residual (Eq. 2-3).

        Aggregate:  m_i = OR_{j∈N(i)} h_j   [bitwise OR of neighbor HVs]
        Update:     h_i = binarize(α·h_i + (1-α)·m_i)

        OR is idempotent: adding the same neighbour twice has no effect.
        This keeps representations bounded without normalisation.

        Args:
            graph: Graph structure
            node_hvs: (N, D) current node HVs (binary)

        Returns:
            (N, D) updated node HVs after L layers
        """
        h = node_hvs.float().clone()

        for _ in range(self.n_layers):
            m = torch.zeros_like(h)
            for i in range(graph.n_nodes):
                if not graph.adj[i]:
                    m[i] = torch.zeros(self.hd_dim)
                    continue
                # OR-aggregate neighbour HVs (idempotent)
                neighbours = torch.stack([h[j] for j in graph.adj[i]])  # (deg, D)
                m[i] = (neighbours.sum(dim=0) > 0).float()              # OR = any > 0

            # Convex combination + binarize (Eq. 3)
            h = (self.alpha * h + (1 - self.alpha) * m)
            h = (h >= 0.5).float()

        return h

    def encode_graph(self, graph: Graph) -> torch.Tensor:
        """
        Encode a graph as a single HV (§III-B + §III-C).

        Steps:
          1. Initial node HVs via rank-bin lookup
          2. L-layer OR-message-passing with residual
          3. Edge HVs = XOR(h_u, h_v) for all edges
          4. Graph HV = majority-bundle of all edge HVs

        Returns:
            (D,) binary graph HV
        """
        if graph.n_nodes == 0:
            return torch.zeros(self.hd_dim)

        # Steps 1-2
        h0 = self._encode_node_initial(graph)
        h_final = self._message_pass(graph, h0)

        # Step 3-4: edge binding + graph bundling
        if not graph.edges:
            # No edges: bundle initial node HVs
            n = h0.shape[0]
            return (h0.sum(dim=0) * 2 > n).float()

        # Use INITIAL node HVs for edge binding.
        # Why: after OR message-passing, node HVs saturate toward all-1s,
        # making XOR ≈ 0. Using h^{(0)} preserves the positional diversity.
        # The message-passed h^{(L)} contributes structurally via its influence
        # on the final binding target, not the source.
        # Approach: edge_hv = XOR(h0_u ⊗ h_final_u, h0_v ⊗ h_final_v)
        # Simplification: edge_hv = XOR(h0_u, h0_v) ⊕ XOR(h_final_u, h_final_v)
        edge_hvs = []
        for u, v in graph.edges:
            # Bind initial + message-passed representations
            init_edge   = (h0[u]      != h0[v]).float()       # XOR of initial
            struct_edge = (h_final[u] != h_final[v]).float()  # XOR of structural
            # Bundle both signals: positions where either differs
            combined = ((init_edge + struct_edge) > 0).float()
            edge_hvs.append(combined)

        stacked = torch.stack(edge_hvs)           # (E, D)
        n_edge = stacked.shape[0]
        # Proper majority: count how many edges have a 1 in each dimension
        return (stacked.sum(dim=0) * 2 > n_edge).float()

    # ── Training and inference ────────────────────────────────────────────────

    def train(self, graphs: List[Graph]):
        """One-shot prototype accumulation across all training graphs."""
        for g in graphs:
            if g.label is None:
                continue
            hv = self.encode_graph(g)
            c = g.label
            if c not in self._prototypes:
                self._prototypes[c] = hv.float()
                self._counts[c] = 1
            else:
                self._prototypes[c] += hv.float()
                self._counts[c] += 1

        # Binarise
        for c in self._prototypes:
            n = self._counts[c]
            self._prototypes[c] = (self._prototypes[c] / n >= 0.5).float()

    def predict(self, graph: Graph) -> Tuple[int, float]:
        """Predict class of a graph via nearest prototype."""
        hv = self.encode_graph(graph)
        best_class, best_sim = 0, -1.0
        for c, proto in self._prototypes.items():
            sim = float(hv_batch_sim(hv, proto.unsqueeze(0))[0])
            if sim > best_sim:
                best_sim = sim
                best_class = c
        return best_class, best_sim

    def accuracy(self, graphs: List[Graph]) -> float:
        correct = sum(
            1 for g in graphs
            if g.label is not None and self.predict(g)[0] == g.label
        )
        labeled = sum(1 for g in graphs if g.label is not None)
        return correct / max(labeled, 1)

    def online_train(self, graph: Graph, lr: float = 0.1):
        """
        Online one-shot update: add one labelled graph without full retraining.

        Args:
            graph: Labelled graph (graph.label must be set)
            lr:    Blending rate for prototype update
        """
        if graph.label is None:
            return
        hv = self.encode(graph)
        c  = graph.label
        if c not in self._prototypes:
            self._prototypes[c] = hv.float().clone()
            self._counts[c] = 1
        else:
            n = self._counts[c]
            alpha = lr
            self._prototypes[c] = _majority(
                (1 - alpha) * self._prototypes[c].float() + alpha * hv.float()
            )
            self._counts[c] = n + 1

    def anomaly_score(self, graph: Graph) -> float:
        """
        Anomaly score: 1 - max_similarity_to_any_prototype.

        High score = graph does not resemble any known class structure.
        Useful for: novel topology detection, adversarial graph detection.

        Returns:
            Anomaly score ∈ [0, 1]; > 0.4 suggests a structurally novel graph.
        """
        if not self._prototypes:
            return 0.5   # unknown without prototypes
        hv    = self.encode(graph)
        sims  = [float(_hamming(hv.unsqueeze(0), p.unsqueeze(0)).item())
                 for p in self._prototypes.values()]
        return float(1.0 - max(sims))

    @property
    def n_classes(self) -> int:
        return len(self._prototypes)


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities: build graphs from common formats
# ═══════════════════════════════════════════════════════════════════════════════

def graph_from_adjacency(adj_matrix: torch.Tensor, label: Optional[int] = None) -> Graph:
    """Build a Graph from an adjacency matrix (N×N)."""
    N = adj_matrix.shape[0]
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)
             if adj_matrix[i, j].item() > 0]
    return Graph(N, edges, label=label)


def graph_from_edge_list(
    n_nodes: int,
    edge_index: torch.Tensor,
    label: Optional[int] = None,
) -> Graph:
    """Build a Graph from a (2, E) edge index tensor (PyG format)."""
    edges = [(int(edge_index[0, i]), int(edge_index[1, i]))
             for i in range(edge_index.shape[1])
             if edge_index[0, i] < edge_index[1, i]]
    return Graph(n_nodes, edges, label=label)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_vs_graph():
    print("=" * 60)
    print("Testing VS-Graph (Poursiami et al. 2025, arXiv:2512.03394)")
    print("=" * 60)

    torch.manual_seed(42)
    model = VSGraph(hd_dim=2000, n_rank_bins=8, n_layers=2, alpha=0.5, seed=0)

    # Create synthetic graphs: class 0 = ring, class 1 = star, class 2 = complete
    def ring(n, c):
        edges = [(i, (i + 1) % n) for i in range(n)]
        return Graph(n, edges, label=c)

    def star(n, c):
        edges = [(0, i) for i in range(1, n)]
        return Graph(n, edges, label=c)

    def complete(n, c):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return Graph(n, edges, label=c)

    # Build dataset
    train_graphs = (
        [ring(6, 0) for _ in range(10)] +
        [star(6, 1) for _ in range(10)] +
        [complete(5, 2) for _ in range(10)]
    )

    model.train(train_graphs)
    print(f"  Trained on {len(train_graphs)} graphs, {model.n_classes} classes")

    acc = model.accuracy(train_graphs)
    print(f"  Training accuracy: {acc:.1%}")
    assert acc > 0.7, f"Training accuracy too low: {acc:.1%}"

    # Test: different-size versions of the same topology
    test_graphs = [ring(7, 0), star(8, 1), complete(4, 2)]
    for g in test_graphs:
        pred, conf = model.predict(g)
        print(f"  {['ring','star','complete'][g.label]}(n={g.n_nodes}): "
              f"pred={pred} ({'✓' if pred==g.label else '✗'}) conf={conf:.4f}")

    # VS-Graph vs GraphHD (no message passing, L=0)
    graphhd = VSGraph(hd_dim=2000, n_rank_bins=8, n_layers=0, seed=0)
    graphhd.train(train_graphs)
    acc_vs  = model.accuracy(train_graphs)
    acc_ghd = graphhd.accuracy(train_graphs)
    print(f"  VS-Graph (L=2): {acc_vs:.1%}  GraphHD (L=0): {acc_ghd:.1%}")

    # Graph encoding speed
    import time
    t0 = time.perf_counter()
    for _ in range(100):
        model.encode_graph(ring(10, 0))
    ms = (time.perf_counter()-t0)/100*1000
    print(f"  Encoding speed: {ms:.3f}ms/graph  (paper reports 0.142ms at D=4096)")

    print("  ✅ VS-Graph OK")


if __name__ == "__main__":
    test_vs_graph()
    print("\n=== VS-Graph tests passed ===")
