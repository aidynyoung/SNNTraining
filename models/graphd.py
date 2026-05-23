"""GrapHD: Hyperdimensional Graph Memory for SNNTraining.
Based on Section IV of Amrouch et al. 2022.
Encodes SNN recurrent connectivity into a single hypervector
for robust reasoning about neural circuit structure."""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from models.hdc import gen_hvs, bind, bundle, sim, thresh, permute


class GrapHD(nn.Module):
    """Encodes SNN recurrent weight matrix connectivity as a hypervector.

    Instead of encoding arbitrary static graphs, this module takes the
    recurrent weight matrix W_rec from an RSNN and encodes its sparse
    connectivity structure into a single hypervector. This enables:

    - Querying whether two neurons are synaptically connected
    - Reconstructing a neuron's neighbor set from the graph HV
    - Detecting connectivity changes during online learning
    - Robust graph matching under noise (hardware errors in weights)

    The graph HV is constructed by binding each neuron's identity HV
    with the bundle of its neighbors' identity HVs.
    """

    def __init__(self, n_nodes, dim=10000, mode="bipolar", device=None, seed=None):
        super().__init__()
        self.n_nodes, self.dim, self.mode = n_nodes, dim, mode
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.node_hvs = gen_hvs(n_nodes, dim, mode, self.device, seed)
        self.register_buffer("graph_hv", torch.zeros(dim, device=self.device))
        self.register_buffer("node_mems", torch.zeros(n_nodes, dim, device=self.device))

    def encode_from_weight_matrix(self, W_rec: torch.Tensor, threshold: float = 0.01):
        """Encode SNN recurrent weight matrix as a graph hypervector.

        Args:
            W_rec: (n_nodes, n_nodes) recurrent weight matrix.
                   Non-zero entries indicate synaptic connections.
            threshold: Minimum absolute weight to consider a connection.
        """
        edges = []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j and abs(W_rec[i, j].item()) > threshold:
                    edges.append((i, j))
        self.encode_graph(edges)
        return len(edges)

    def encode_graph(self, edges: List[Tuple[int, int]]):
        """Encode a list of directed edges into the graph hypervector.

        Args:
            edges: List of (pre_neuron, post_neuron) tuples.
        """
        mems = torch.zeros(self.n_nodes, self.dim, device=self.device)
        for i in range(self.n_nodes):
            neighbors = [b for a, b in edges if a == i] + [a for a, b in edges if b == i]
            for j in set(neighbors):
                mems[i] = mems[i] + self.node_hvs[j]
        self.node_mems = mems
        g = torch.zeros(self.dim, device=self.device)
        for i in range(self.n_nodes):
            g = g + bind(self.node_hvs[i], self.node_mems[i], self.mode)
        self.graph_hv = 0.5 * g
        if self.mode == "bipolar":
            self.graph_hv = thresh(self.graph_hv)

    def check_edge(self, i: int, j: int) -> float:
        """Check if edge (i, j) exists in the encoded graph.

        Returns similarity score (higher = more likely connected).
        """
        mi = bind(self.node_hvs[i], self.graph_hv, self.mode)
        return sim(self.node_hvs[j], mi, self.mode).item()

    def node_similarity(self, i: int, j: int) -> float:
        """Compute similarity between two neuron identity HVs."""
        return sim(self.node_hvs[i], self.node_hvs[j], self.mode).item()

    def reconstruct_neighbors(self, i: int, threshold: float = 0.05) -> List[int]:
        """Reconstruct the neighbor set of neuron i from the graph HV.

        Args:
            i: Neuron index
            threshold: Minimum similarity to consider as neighbor

        Returns:
            List of neighbor neuron indices
        """
        mi = bind(self.node_hvs[i], self.graph_hv, self.mode)
        neighbors = []
        for j in range(self.n_nodes):
            if j != i and sim(self.node_hvs[j], mi, self.mode).item() > threshold:
                neighbors.append(j)
        return neighbors

    def connectivity_change_score(self, other: 'GrapHD') -> float:
        """Measure how much the connectivity has changed vs another graph state.

        Returns cosine distance between graph HVs (0 = identical, 1 = orthogonal).
        """
        return 1.0 - sim(self.graph_hv, other.graph_hv, self.mode).item()

    def detect_weight_pruning(self, W_rec: torch.Tensor, old_threshold: float = 0.01,
                              new_threshold: float = 0.05) -> List[Tuple[int, int]]:
        """Detect which connections were pruned by a threshold increase.

        Args:
            W_rec: Current weight matrix
            old_threshold: Previous connection threshold
            new_threshold: New (higher) connection threshold

        Returns:
            List of (pre, post) edges that were pruned
        """
        old_edges = set()
        new_edges = set()
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    w = abs(W_rec[i, j].item())
                    if w > old_threshold:
                        old_edges.add((i, j))
                    if w > new_threshold:
                        new_edges.add((i, j))
        return list(old_edges - new_edges)
