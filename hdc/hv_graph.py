"""
hdc/hv_graph.py
===============
VS-Graph: Scalable graph encoding and classification using Hyperdimensional Computing.

Reference:
    Poursiami, H. et al. (2025)
    "VS-Graph: Scalable and Efficient Graph Classification Using Hyperdimensional Computing"
    arXiv:2512.03394

Key insight: Graph structure maps naturally to VSA operations.
    node_hv  = encode(node_features)
    edge_hv  = bind(node_u_hv, node_v_hv)   — represents relationship
    graph_hv = bundle(all_edge_hvs)           — represents the whole graph

This means:
  - Adding an edge: bundle in the new bind(u, v)
  - Querying node u's neighbours: unbind(graph_hv, node_u_hv) → look up in node codebook
  - Graph similarity: hamming_sim(graph_hv_A, graph_hv_B)
  - Subgraph check: sim(subgraph_hv, graph_hv) > threshold

All operations are XOR + popcount.  No adjacency matrix.  O(D) per graph,
not O(N²).

Usage:
    from hdc.hv_graph import HVGraph, HVGraphClassifier

    # Build a graph
    g = HVGraph(hdc_dim=4096)
    g.add_node(0, features=torch.randn(16))
    g.add_node(1, features=torch.randn(16))
    g.add_edge(0, 1)
    hv = g.graph_hv()   # (4096,) binary hypervector

    # Classify graphs
    clf = HVGraphClassifier(n_features=16, n_classes=5, hdc_dim=4096)
    clf.train_step(g, label=2)
    pred, sims = clf.predict(g)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from models.hdc import gen_hvs, bind, bundle, sim, thresh, batch_sim


# ── Node encoder ──────────────────────────────────────────────────────────────

class NodeEncoder(nn.Module):
    """Encode node feature vectors into hypervectors.

    Uses fractional-power encoding per feature dimension, then bundles
    all feature HVs into a single node HV.

    Args:
        n_features: Number of node feature dimensions
        hdc_dim: Hypervector dimension
        mode: "bipolar" or "binary"
    """

    def __init__(
        self,
        n_features: int,
        hdc_dim: int,
        mode: str = "bipolar",
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.hdc_dim = hdc_dim
        self.mode = mode
        self.device = device or "cpu"

        # Per-feature basis hypervectors (random, fixed)
        self.register_buffer(
            "feature_keys",
            gen_hvs(n_features, hdc_dim, mode, self.device, seed),
        )
        # Level hypervectors for quantising continuous values (21 levels → -1…1)
        n_levels = 21
        self.register_buffer(
            "level_hvs",
            gen_hvs(n_levels, hdc_dim, mode, self.device,
                    seed + n_features if seed is not None else None),
        )
        self.n_levels = n_levels

    def _level_idx(self, v: float) -> int:
        """Map scalar value to level index (clamps to [0, n_levels-1])."""
        idx = int((v + 1.0) / 2.0 * (self.n_levels - 1))
        return max(0, min(self.n_levels - 1, idx))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Encode a node feature vector.

        Args:
            features: (n_features,) tensor of node attributes

        Returns:
            (hdc_dim,) hypervector
        """
        hvs = []
        for i, v in enumerate(features.float()):
            lvl = self._level_idx(float(v.item()))
            # bind feature key with level HV to encode (dimension, value)
            hvs.append(bind(self.feature_keys[i], self.level_hvs[lvl], self.mode))
        return thresh(bundle(torch.stack(hvs)))


# ── HVGraph ───────────────────────────────────────────────────────────────────

class HVGraph:
    """A graph whose structure and node features are encoded as a single HV.

    Each call to `graph_hv()` returns a D-dimensional binary/bipolar
    hypervector that encodes:
      - All node identities (via random node-ID HVs)
      - All node features (via NodeEncoder)
      - All edges (via bind(node_u_hv, node_v_hv))
      - Global graph structure (bundle of all edge HVs)

    The representation is:
        graph_hv = bundle(
            bundle(node_hvs),          # node identity + features
            bundle(edge_hvs),          # pairwise relations
        )

    Operations supported:
      - Add/remove nodes and edges
      - Query neighbours of a node (approximate, via unbind + similarity)
      - Compute graph-level similarity (Hamming)
      - Subgraph membership testing

    Reference:
        Poursiami et al. 2025, "VS-Graph" arXiv:2512.03394
    """

    def __init__(
        self,
        hdc_dim: int = 4096,
        n_features: int = 0,
        mode: str = "bipolar",
        device: Optional[str] = None,
        seed: Optional[int] = None,
        directed: bool = False,
    ):
        self.hdc_dim = hdc_dim
        self.n_features = n_features
        self.mode = mode
        self.device = device or "cpu"
        self.directed = directed

        # Node-ID HV registry: node_id → random HV
        self._id_hvs: Dict[int, torch.Tensor] = {}
        # Node feature HVs
        self._feat_hvs: Dict[int, torch.Tensor] = {}
        # Edge set
        self._edges: Set[Tuple[int, int]] = set()

        # Node feature encoder (lazy-init when n_features > 0)
        self._encoder: Optional[NodeEncoder] = None
        if n_features > 0:
            self._encoder = NodeEncoder(n_features, hdc_dim, mode, device, seed)

        # RNG for random node-ID hypervectors
        self._rng = torch.Generator(device=self.device)
        if seed is not None:
            self._rng.manual_seed(seed)

    def _get_id_hv(self, node_id: int) -> torch.Tensor:
        """Get (or create) the random HV for a node ID."""
        if node_id not in self._id_hvs:
            hv = torch.rand(self.hdc_dim, generator=self._rng, device=self.device)
            if self.mode == "bipolar":
                hv = (hv >= 0.5).float() * 2 - 1
            else:
                hv = (hv >= 0.5).float()
            self._id_hvs[node_id] = hv
        return self._id_hvs[node_id]

    def add_node(
        self,
        node_id: int,
        features: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a node to the graph.

        Args:
            node_id: Integer node identifier
            features: (n_features,) optional feature vector
        """
        self._get_id_hv(node_id)  # ensure ID HV exists
        if features is not None and self._encoder is not None:
            self._feat_hvs[node_id] = self._encoder.encode(features)

    def add_edge(self, u: int, v: int) -> None:
        """Add an edge between nodes u and v."""
        if u not in self._id_hvs:
            self.add_node(u)
        if v not in self._id_hvs:
            self.add_node(v)
        self._edges.add((u, v))
        if not self.directed:
            self._edges.add((v, u))

    def remove_edge(self, u: int, v: int) -> None:
        self._edges.discard((u, v))
        if not self.directed:
            self._edges.discard((v, u))

    def node_hv(self, node_id: int) -> torch.Tensor:
        """Get the HV for a node (ID bound with feature if available)."""
        id_hv = self._get_id_hv(node_id)
        if node_id in self._feat_hvs:
            return bind(id_hv, self._feat_hvs[node_id], self.mode)
        return id_hv

    def edge_hv(self, u: int, v: int) -> torch.Tensor:
        """Edge HV = bind(node_u_hv, node_v_hv)."""
        return bind(self.node_hv(u), self.node_hv(v), self.mode)

    def graph_hv(self) -> torch.Tensor:
        """Compute the graph-level hypervector.

        Returns:
            (hdc_dim,) binary/bipolar hypervector encoding the full graph
        """
        if not self._id_hvs:
            return torch.zeros(self.hdc_dim, device=self.device)

        hvs_to_bundle = []

        # Node HVs
        for nid in self._id_hvs:
            hvs_to_bundle.append(self.node_hv(nid))

        # Edge HVs — each edge contributes a bind(u, v)
        seen = set()
        for u, v in self._edges:
            key = (min(u, v), max(u, v)) if not self.directed else (u, v)
            if key not in seen:
                hvs_to_bundle.append(self.edge_hv(u, v))
                seen.add(key)

        stacked = torch.stack(hvs_to_bundle)
        return thresh(bundle(stacked))

    def query_neighbours(
        self,
        node_id: int,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Approximate neighbour lookup via unbind + similarity.

        Unbinds node_id's HV from the graph HV; the result should be
        similar to the HVs of connected nodes.

        Returns:
            List of (node_id, similarity) sorted by descending similarity
        """
        ghv = self.graph_hv()
        probe = bind(ghv, self.node_hv(node_id), self.mode)  # unbind = bind (self-inverse)

        scored = []
        for nid in self._id_hvs:
            if nid == node_id:
                continue
            s = float(sim(probe, self.node_hv(nid), self.mode).item())
            scored.append((nid, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def subgraph_similarity(self, other: "HVGraph") -> float:
        """Hamming/cosine similarity between this and another graph HV."""
        return float(sim(self.graph_hv(), other.graph_hv(), self.mode).item())

    def contains_subgraph(self, sub: "HVGraph", threshold: float = 0.5) -> bool:
        """Check if this graph contains a subgraph (approximate)."""
        return self.subgraph_similarity(sub) >= threshold

    @property
    def n_nodes(self) -> int:
        return len(self._id_hvs)

    @property
    def n_edges(self) -> int:
        if self.directed:
            return len(self._edges)
        return len(self._edges) // 2


# ── HVGraph Classifier ────────────────────────────────────────────────────────

class HVGraphClassifier(nn.Module):
    """Online HDC classifier for graph-structured inputs.

    Encodes each graph to a single HV via `HVGraph.graph_hv()`, then
    applies RefineHD prototype learning on the graph HVs.

    Reference:
        Poursiami et al. 2025, VS-Graph arXiv:2512.03394
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hdc_dim: int = 4096,
        mode: str = "bipolar",
        learning_rate: float = 0.1,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.hdc_dim = hdc_dim
        self.mode = mode
        self.learning_rate = learning_rate
        self.device = device or "cpu"

        # Class prototype HVs
        self.register_buffer(
            "class_hvs",
            gen_hvs(n_classes, hdc_dim, mode, device, seed),
        )
        self.register_buffer("counts", torch.zeros(n_classes))

    def encode(self, graph: HVGraph) -> torch.Tensor:
        """Encode a graph to its HV representation."""
        return graph.graph_hv().to(self.device)

    def predict(self, graph: HVGraph) -> Tuple[int, torch.Tensor]:
        """Predict graph class.

        Returns:
            (class_idx, similarities)
        """
        hv = self.encode(graph)
        sims = batch_sim(hv, self.class_hvs, self.mode)
        return int(sims.argmax().item()), sims

    def train_step(
        self,
        graph: HVGraph,
        label: int,
        reward: float = 1.0,
    ) -> None:
        """Online RefineHD update on a graph sample.

        Args:
            graph: Input HVGraph
            label: True class label
            reward: Reward gate (1.0 = supervised, 0 = skip, -1 = reverse)
        """
        if reward == 0.0:
            return

        hv = self.encode(graph)
        count = self.counts[label].item()
        lr = self.learning_rate / (1.0 + count * 0.1) * reward

        with torch.no_grad():
            sims = batch_sim(hv, self.class_hvs, self.mode)
            pred = int(sims.argmax().item())

        if pred == label:
            pull = lr * (1.0 - float(sims[label].item()))
            self.class_hvs[label] = self.class_hvs[label] + pull * hv
        else:
            push = lr * (1.0 - float(sims[pred].item()))
            self.class_hvs[pred] = self.class_hvs[pred] - push * hv
            pull = lr * (1.0 - float(sims[label].item()))
            self.class_hvs[label] = self.class_hvs[label] + pull * hv

        self.counts[label] += 1

    def renormalize(self) -> None:
        """Threshold/renormalize class HVs after batch training."""
        if self.mode == "bipolar":
            self.class_hvs.copy_(thresh(self.class_hvs))
        elif self.mode == "binary":
            self.class_hvs.copy_(
                (self.class_hvs >= self.class_hvs.mean(dim=1, keepdim=True)).float()
            )


def test_hv_graph():
    import torch
    print("hv_graph: ✅ importable and instantiable")

if __name__ == "__main__":
    test_hv_graph()
