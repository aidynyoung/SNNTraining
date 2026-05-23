"""GrapHD Graph Memory Experiment: encode SNN layer connectivity as hypervector.
Demonstrates graph encoding, edge recovery, and noise tolerance.
Based on Amrouch et al. Section IV."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.graphd import GrapHD

def run_graph_experiment(n_nodes=50, dim=4000):
    edges = [(i, (i+1)%n_nodes) for i in range(n_nodes)]
    edges += [(i, (i+3)%n_nodes) for i in range(n_nodes)]
    graph = GrapHD(n_nodes, dim=dim, mode="bipolar", seed=0)
    graph.encode_graph(edges)

    correct = 0
    total = 0
    for i, j in edges:
        score = graph.check_edge(i, j)
        correct += score > 0.05
        total += 1
    print(f"Edge recall: {correct}/{total} = {correct/total:.2%}")

    for noise in [0, 0.05, 0.10, 0.20]:
        import torch
        noisy_hv = graph.graph_hv.clone()
        mask = torch.rand(dim) < noise
        noisy_hv[mask] = 0
        graph.graph_hv = noisy_hv
        c = sum(1 for i, j in edges if graph.check_edge(i, j) > 0.05)
        print(f"  Noise {noise:.0%}: recall {c}/{total} = {c/total:.2%}")
        graph.encode_graph(edges)

if __name__ == "__main__":
    run_graph_experiment()
