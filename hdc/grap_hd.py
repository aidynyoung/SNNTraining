"""
GrapHD: Hyperdimensional Graph Memory
====================================
Implements GrapHD from Section IV of:
"Brain-Inspired Hyperdimensional Computing for Ultra-Efficient Edge AI"
(NSF purl/10392362)

Provides:
- Hyperdimensional graph encoding (node hypervectors)
- Memory node construction
- Graph memory bundling
- Cognitive operations: retrieval, reconstruction, matching

Based on the paper's approach:
- Graph encoded as single hypervector (holographic)
- Nodes = random hypervectors (nearly orthogonal)
- Edges = bundled neighbor hypervectors
- Information distributed across all dimensions
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass
import numpy as np


@dataclass
class GrapHDConfig:
    """Configuration for GrapHD."""
    dim: int = 1000  # Hypervector dimension
    n_nodes: int = 10  # Expected number of nodes
    n_edges: int = 20  # Expected number of edges
    encoding: str = "binary"  # "binary", "bipolar"
    bundle_method: str = "mean"  # "mean", "sum"


def generate_random_hypervector(
    dim: int,
    encoding: str = "bipolar",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate random hypervector.
    
    Args:
        dim: Dimension
        encoding: "binary" (0/1) or "bipolar" (+1/-1)
        seed: Optional random seed
    
    Returns:
        Random hypervector (dim,)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if encoding == "bipolar":
        hv = torch.randint(0, 2, (dim,)) * 2 - 1
        hv = hv.float()
        hv[hv == 0] = 1  # Map 0 to +1
    else:
        hv = torch.rand(dim)
    
    return hv


def node_hypervectors(
    n_nodes: int,
    dim: int,
    encoding: str = "bipolar",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate node hypervectors.
    
    Nearly orthogonal: δ(H_k, H_l) ≈ 0 for k ≠ l
    
    Args:
        n_nodes: Number of nodes
        dim: Dimension
        encoding: Encoding type
        seed: Random seed
    
    Returns:
        Node hypervectors (n_nodes, dim)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    hvs = torch.randn(n_nodes, dim) if encoding == "bipolar" else torch.rand(n_nodes, dim)
    
    if encoding == "bipolar":
        hvs = torch.sign(hvs)
        hvs[hvs == 0] = 1
    
    return hvs


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity."""
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8)


class GrapHDEncoder:
    """
    GrapHD: Hyperdimensional Graph Encoding.
    
    Encodes graph as single hypervector using:
    1. Node hypervectors (random signatures)
    2. Memory nodes (bundled neighbors)
    3. Graph memory (bundled node+memory pairs)
    
    Attributes:
        config: GrapHDConfig
        node_hvs: Node hypervectors
        memory_nodes: Node memories
        graph_memory: Full graph hypervector
    """
    
    def __init__(
        self,
        config: Optional[GrapHDConfig] = None,
    ):
        self.config = config or GrapHDConfig()
        self.node_hvs: Optional[torch.Tensor] = None
        self.memory_nodes: Optional[torch.Tensor] = None
        self.graph_memory: Optional[torch.Tensor] = None
        self.node_list: List[int] = []
        self.edge_list: List[Tuple[int, int]] = []
    
    def encode(self, adjacency: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        """Encode a graph from an adjacency matrix and node feature matrix.

        Args:
            adjacency: (n_nodes, n_nodes) adjacency matrix
            node_features: (n_nodes, feat_dim) node feature matrix

        Returns:
            (dim,) graph hypervector
        """
        n_nodes = adjacency.shape[0]
        rows, cols = torch.where(adjacency > 0)
        edges = list(zip(rows.tolist(), cols.tolist()))
        return self.encode_graph(edges, n_nodes=n_nodes)

    def encode_graph(
        self,
        edges: List[Tuple[int, int]],
        n_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode graph as hypervector.
        
        Args:
            edges: List of (source, target) edges
            n_nodes: Number of nodes (infer if not provided)
        
        Returns:
            Graph hypervector (dim,)
        """
        if n_nodes is None:
            n_nodes = max(max(e) for e in edges) + 1 if edges else 0
        
        self.n_nodes = n_nodes
        self.edge_list = edges
        
        # Generate node hypervectors
        self.node_hvs = node_hypervectors(n_nodes, self.config.dim)
        
        # Create memory nodes (bundled neighbors)
        self.memory_nodes = torch.zeros(n_nodes, self.config.dim)
        
        for i, j in edges:
            # Add neighbor hypervector to memory
            self.memory_nodes[i] += self.node_hvs[j]
            self.memory_nodes[j] += self.node_hvs[i]  # Undirected
        
        # Bundling: normalize
        if self.config.bundle_method == "mean":
            n_neighbors = torch.zeros(n_nodes)
            for i, j in edges:
                n_neighbors[i] += 1
                n_neighbors[j] += 1
            n_neighbors[n_neighbors == 0] = 1  # Avoid div by zero
            self.memory_nodes = self.memory_nodes / n_neighbors.unsqueeze(1)
        
        # Create graph memory: bundle of (node ⊕ memory)
        self.graph_memory = torch.zeros(self.config.dim)
        
        for i in range(n_nodes):
            # Binding: XOR (here using element-wise multiplication for bipolar)
            bound = self.node_hvs[i] * self.memory_nodes[i]
            self.graph_memory += bound
        
        # Normalize
        self.graph_memory = torch.sign(self.graph_memory)
        self.graph_memory[self.graph_memory == 0] = 1
        
        self.node_list = list(range(n_nodes))
        
        return self.graph_memory
    
    def reconstruct_node_memory(self, node_id: int) -> torch.Tensor:
        """
        Reconstruct memory for a node.
        
        Args:
            node_id: Node index
        
        Returns:
            Reconstructed memory (dim,)
        """
        if self.graph_memory is None or self.node_hvs is None:
            raise ValueError("Graph not encoded")
        
        # Unbinding: (graph ⊕ node_hv) ≈ memory
        reconstructed = self.graph_memory * self.node_hvs[node_id]
        
        return reconstructed
    
    def check_edge(self, node_i: int, node_j: int) -> float:
        """
        Check if edge exists between nodes.
        
        Returns:
            Decision score (high if edge exists)
        """
        if self.node_hvs is None:
            raise ValueError("Graph not encoded")
        
        memory_i = self.reconstruct_node_memory(node_i)
        
        # Similarity to neighbor hypervector
        similarity = cosine_similarity(memory_i, self.node_hvs[node_j])
        
        return similarity
    
    def get_neighbors(self, node_id: int) -> List[int]:
        """Get neighbors of node."""
        neighbors = []
        for i, j in self.edge_list:
            if i == node_id:
                neighbors.append(j)
            elif j == node_id:
                neighbors.append(i)
        return list(set(neighbors))
    
    def graph_similarity(self, other: "GrapHDEncoder") -> float:
        """
        Compute graph similarity.

        Args:
            other: Another GrapHDEncoder

        Returns:
            Similarity score (0-1)
        """
        if self.graph_memory is None or other.graph_memory is None:
            raise ValueError("Graphs not encoded")

        return cosine_similarity(self.graph_memory, other.graph_memory)

    def subgraph_similarity(
        self,
        other: "GrapHDEncoder",
        node_subset: List[int],
    ) -> float:
        """
        Compute similarity between a subgraph of this encoder and another graph.

        Extracts the subgraph induced by `node_subset`, re-encodes it,
        and computes similarity to `other.graph_memory`.

        Useful for: substructure search, motif detection, partial matching.

        Args:
            other:       Another GrapHDEncoder (the template to match against)
            node_subset: List of node indices forming the query subgraph

        Returns:
            Cosine similarity of subgraph vs other graph ∈ [-1, 1]
        """
        if other.graph_memory is None:
            raise ValueError("Other graph not encoded")
        if self.node_memory is None:
            raise ValueError("This graph not encoded — call encode_graph first")

        # Build subgraph HV from the node subset
        node_subset_set = set(node_subset)
        sub_hvs = [self.node_memory[n] for n in node_subset
                   if n < len(self.node_memory)]
        if not sub_hvs:
            return 0.0

        # Bundle subset nodes
        sub_hv = torch.stack(sub_hvs).mean(dim=0)
        return cosine_similarity(sub_hv, other.graph_memory)

    def graph_health(self) -> Dict:
        """
        One-call graph memory status: encoded nodes, edge density, dimension.

        edge_density = edges / max_possible_edges (clique ratio).
        """
        if not hasattr(self, "node_hvs") or self.node_hvs is None:
            return {"status": "not_encoded", "dim": self.config.dim}
        n_nodes = getattr(self, "n_nodes", 0)
        edge_list = getattr(self, "edge_list", [])
        n_edges = len(edge_list)
        max_edges = n_nodes * (n_nodes - 1) // 2
        density = n_edges / max(max_edges, 1)
        return {
            "dim":          self.config.dim,
            "n_nodes":      n_nodes,
            "n_edges":      n_edges,
            "edge_density": round(density, 4),
            "encoded":      self.graph_memory is not None,
        }


def grap_hd_operations():
    """
    Demonstrates GrapHD cognitive operations.
    """
    print("Testing GrapHD operations...")
    
    # Create graph: triangle (0-1-2-0) + edge to 3
    edges = [(0, 1), (1, 2), (2, 0), (0, 3)]
    
    encoder = GrapHDEncoder(GrapHDConfig(dim=100))
    graph_hv = encoder.encode_graph(edges, n_nodes=4)
    
    print(f"Graph hypervector: {graph_hv[:10]}...")
    print(f"Graph dimension: {graph_hv.shape}")
    
    # Reconstruct node memory
    mem_0 = encoder.reconstruct_node_memory(0)
    print(f"Memory for node 0: {mem_0[:10]}...")
    
    # Check edges
    for i in range(4):
        for j in range(i+1, 4):
            score = encoder.check_edge(i, j)
            is_edge = "YES" if score > 0.1 else "NO"
            print(f"Edge ({i},{j}): {score:.3f} -> {is_edge}")
    
    # Get neighbors
    print(f"Neighbors of 0: {encoder.get_neighbors(0)}")
    print(f"Neighbors of 1: {encoder.get_neighbors(1)}")
    
    # Create second graph for similarity
    edges2 = [(0, 1), (1, 2), (2, 0), (0, 4)]
    encoder2 = GrapHDEncoder(GrapHDConfig(dim=100))
    encoder2.encode_graph(edges2, n_nodes=5)
    
    sim = encoder.graph_similarity(encoder2)
    print(f"\nGraph similarity (triangles): {sim:.3f}")
    
    print("\nGrapHD tests complete!")


def test_grap_hd_reconstruction():
    """Test graph reconstruction accuracy."""
    print("Testing GrapHD reconstruction...")
    
    torch.manual_seed(42)
    
    for dim in [1000, 2000, 4000, 6000]:
        edges = [(i, (i+1)%10) for i in range(10)]  # Ring
        edges += [(i, (i+2)%10) for i in range(10)]  # Skip connections
        
        encoder = GrapHDEncoder(GrapHDConfig(dim=dim))
        encoder.encode_graph(edges, n_nodes=10)
        
        # Reconstruct and check
        mismatches = 0
        for i in range(10):
            mem = encoder.reconstruct_node_memory(i)
            neighbors = set(encoder.get_neighbors(i))
            
            # Check reconstructed neighbors
            for j in range(10):
                if j in neighbors:
                    score = encoder.check_edge(i, j)
                    if score < 0.1:
                        mismatches += 1
                else:
                    score = encoder.check_edge(i, j)
                    if score > 0.1:
                        mismatches += 1
        
        accuracy = 1 - mismatches / 100
        print(f"D={dim}: reconstruction accuracy = {accuracy:.1%}")
    
    print("\nReconstruction tests complete!")


if __name__ == "__main__":
    grap_hd_operations()
    print()
    test_grap_hd_reconstruction()