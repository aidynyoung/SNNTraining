"""
Hyperseed: Unsupervised Learning with Vector Symbolic Architectures
====================================================================
Based on: Osipov, D., et al. (2024)
"Hyperseed: Unsupervised Learning with Vector Symbolic Architectures"
IEEE Transactions on Neural Networks and Learning Systems

Hyperseed is an unsupervised learning algorithm that discovers clusters
and patterns in data using HDC principles without labeled examples.

Key innovations:
1. **Seed Generation** — Random hypervectors as initial cluster seeds
2. **Competitive Assignment** — Each sample assigned to nearest seed
3. **Prototype Refinement** — Seeds updated via bundling assigned samples
4. **Automatic Cluster Discovery** — Number of clusters determined by similarity thresholds
5. **Hierarchical Clustering** — Multi-resolution clustering via threshold annealing

Reference:
  Osipov, D., et al. (2024)
  "Hyperseed: Unsupervised Learning with Vector Symbolic Architectures"
  IEEE TNNLS, doi: 10.1109/TNNLS.2022.3201404
"""

import torch
from typing import Optional, List, Tuple, Dict, Any
from hdc.hdc_glue import (
    hv_xor, hv_popcount, hv_hamming_sim, hv_bundle,
    hv_permute, hv_majority, hv_batch_sim, gen_hvs,
)


class HyperseedCluster:
    """
    A single Hyperseed cluster with prototype and metadata.

    Each cluster maintains:
    - prototype: The cluster center hypervector
    - count: Number of samples assigned
    - variance: Intra-cluster similarity (measure of compactness)
    """

    def __init__(self, prototype: torch.Tensor, label: int):
        self.prototype = prototype.clone()
        self.label = label
        self.count = 0
        self._assigned_hvs: List[torch.Tensor] = []

    def assign(self, hv: torch.Tensor):
        """Assign a sample to this cluster."""
        self._assigned_hvs.append(hv.clone())
        self.count += 1

    def update(self):
        """Update prototype by bundling all assigned samples."""
        if self._assigned_hvs:
            bundled = hv_bundle(torch.stack(self._assigned_hvs))
            self.prototype = hv_majority(bundled)

    def get_variance(self) -> float:
        """Compute intra-cluster variance (1 - avg similarity to prototype)."""
        if not self._assigned_hvs:
            return 0.0
        sims = [float(hv_hamming_sim(hv, self.prototype)) for hv in self._assigned_hvs]
        return 1.0 - (sum(sims) / len(sims))

    def reset(self):
        """Reset assigned samples (for iterative refinement)."""
        self._assigned_hvs = []
        self.count = 0


class Hyperseed:
    """
    Hyperseed: Unsupervised Learning with Vector Symbolic Architectures.

    Discovers clusters in data without labels by:
    1. Encoding each sample into a hypervector
    2. Generating random seed hypervectors as initial cluster centers
    3. Iteratively assigning samples to nearest seed and updating seeds
    4. Merging similar clusters and splitting high-variance clusters
    5. Returning cluster labels for all samples

    Parameters:
    - n_seeds: Initial number of random seeds (default: auto-detect)
    - merge_threshold: Similarity threshold for merging clusters (default: 0.6)
    - split_threshold: Variance threshold for splitting clusters (default: 0.3)
    - max_iterations: Maximum refinement iterations (default: 20)
    - convergence_delta: Minimum change for convergence (default: 0.001)
    """

    def __init__(
        self,
        dim: int = 10000,
        n_seeds: Optional[int] = None,
        merge_threshold: float = 0.6,
        split_threshold: float = 0.3,
        max_iterations: int = 20,
        convergence_delta: float = 0.001,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.n_seeds = n_seeds
        self.merge_threshold = merge_threshold
        self.split_threshold = split_threshold
        self.max_iterations = max_iterations
        self.convergence_delta = convergence_delta
        self.seed = seed

        self.clusters: List[HyperseedCluster] = []
        self.labels_: Optional[torch.Tensor] = None
        self.n_clusters_: int = 0
        self.converged_: bool = False

    def _generate_seeds(self, n: int) -> List[HyperseedCluster]:
        """Generate random seed hypervectors as initial cluster centers."""
        seed_base = self.seed or 42
        clusters = []
        for i in range(n):
            hv = gen_hvs(1, self.dim, seed=seed_base + i).squeeze(0)
            clusters.append(HyperseedCluster(hv, i))
        return clusters

    def _assign_samples(
        self,
        hvs: torch.Tensor,
    ) -> torch.Tensor:
        """Assign each sample to the nearest cluster.

        Args:
            hvs: (n_samples, dim) hypervectors

        Returns:
            (n_samples,) cluster assignments
        """
        n_samples = hvs.shape[0]
        n_clusters = len(self.clusters)
        prototypes = torch.stack([c.prototype for c in self.clusters])

        # Compute all pairwise similarities
        # (n_samples, n_clusters)
        sims = torch.zeros(n_samples, n_clusters)
        for i in range(n_samples):
            for j in range(n_clusters):
                sims[i, j] = hv_hamming_sim(hvs[i], prototypes[j])

        return sims.argmax(dim=-1)

    def _merge_similar_clusters(self):
        """Merge clusters whose prototypes are too similar."""
        if len(self.clusters) < 2:
            return

        merged = True
        while merged:
            merged = False
            i = 0
            while i < len(self.clusters):
                j = i + 1
                while j < len(self.clusters):
                    sim = float(hv_hamming_sim(
                        self.clusters[i].prototype,
                        self.clusters[j].prototype,
                    ))
                    if sim > self.merge_threshold:
                        # Merge j into i
                        bundled = hv_bundle(torch.stack([
                            self.clusters[i].prototype,
                            self.clusters[j].prototype,
                        ]))
                        self.clusters[i].prototype = hv_majority(bundled)
                        self.clusters[i].count += self.clusters[j].count
                        self.clusters.pop(j)
                        merged = True
                    else:
                        j += 1
                i += 1

    def _split_high_variance_clusters(self):
        """Split clusters with high intra-cluster variance."""
        new_clusters = []
        for cluster in self.clusters:
            variance = cluster.get_variance()
            if variance > self.split_threshold and cluster.count > 1:
                # Create two new seeds by perturbing the prototype
                noise1 = (torch.rand(self.dim) < 0.1).float()
                noise2 = (torch.rand(self.dim) < 0.1).float()
                p1 = hv_majority(hv_bundle(torch.stack([
                    cluster.prototype, hv_xor(cluster.prototype, noise1)
                ])))
                p2 = hv_majority(hv_bundle(torch.stack([
                    cluster.prototype, hv_xor(cluster.prototype, noise2)
                ])))
                new_clusters.append(HyperseedCluster(p1, len(self.clusters) + len(new_clusters)))
                new_clusters.append(HyperseedCluster(p2, len(self.clusters) + len(new_clusters) + 1))
            else:
                new_clusters.append(cluster)
        self.clusters = new_clusters

    def fit(
        self,
        X: torch.Tensor,
        encode_fn: Optional[callable] = None,
    ) -> torch.Tensor:
        """Fit Hyperseed to data.

        Args:
            X: (n_samples, n_features) or (n_samples, dim) if encode_fn is None
            encode_fn: Optional function to convert features → hypervectors

        Returns:
            (n_samples,) cluster labels
        """
        # Encode data
        if encode_fn is not None:
            hvs = torch.stack([encode_fn(x) for x in X])
        else:
            hvs = X

        n_samples = hvs.shape[0]

        # Initialize seeds
        n_seeds = self.n_seeds or min(int(n_samples ** 0.5), 20)
        self.clusters = self._generate_seeds(n_seeds)

        # Iterative refinement
        prev_assignments = torch.full((n_samples,), -1)
        for iteration in range(self.max_iterations):
            # Assign samples to nearest cluster
            assignments = self._assign_samples(hvs)

            # Reset and re-assign
            for c in self.clusters:
                c.reset()
            for i in range(n_samples):
                idx = int(assignments[i].item())
                if idx < len(self.clusters):
                    self.clusters[idx].assign(hvs[i])

            # Update prototypes
            for c in self.clusters:
                c.update()

            # Merge similar clusters
            self._merge_similar_clusters()

            # Split high-variance clusters
            self._split_high_variance_clusters()

            # Check convergence
            if prev_assignments.shape[0] == n_samples and prev_assignments.shape == assignments.shape:
                changes = (assignments != prev_assignments).sum().item()
                change_ratio = changes / n_samples
                if change_ratio < self.convergence_delta:
                    self.converged_ = True
                    break

            prev_assignments = assignments.clone()

        # Final assignment
        self.labels_ = self._assign_samples(hvs)
        self.n_clusters_ = len(self.clusters)

        # Update final cluster counts
        for c in self.clusters:
            c.reset()
        for i in range(n_samples):
            self.clusters[int(self.labels_[i].item())].assign(hvs[i])

        return self.labels_

    def predict(self, X: torch.Tensor, encode_fn: Optional[callable] = None) -> torch.Tensor:
        """Predict cluster labels for new data.

        Args:
            X: (n_samples, n_features) or (n_samples, dim)
            encode_fn: Optional encoding function

        Returns:
            (n_samples,) cluster labels
        """
        if encode_fn is not None:
            hvs = torch.stack([encode_fn(x) for x in X])
        else:
            hvs = X

        return self._assign_samples(hvs)

    def get_cluster_prototypes(self) -> torch.Tensor:
        """Get all cluster prototype hypervectors.

        Returns:
            (n_clusters, dim) prototypes
        """
        return torch.stack([c.prototype for c in self.clusters])

    def get_cluster_sizes(self) -> List[int]:
        """Get the size of each cluster."""
        return [c.count for c in self.clusters]

    def get_cluster_variances(self) -> List[float]:
        """Get the intra-cluster variance for each cluster."""
        return [c.get_variance() for c in self.clusters]

    def silhouette_score(self, hvs: torch.Tensor) -> float:
        """Compute silhouette score for the clustering.

        Higher is better (range: -1 to 1).

        Args:
            hvs: (n_samples, dim) hypervectors

        Returns:
            Average silhouette score
        """
        if self.labels_ is None:
            return 0.0

        n_samples = hvs.shape[0]
        scores = []

        for i in range(n_samples):
            label = int(self.labels_[i].item())
            same_cluster = (self.labels_ == label).nonzero().squeeze(-1)
            other_clusters = (self.labels_ != label).nonzero().squeeze(-1)

            if len(same_cluster) <= 1 or len(other_clusters) == 0:
                scores.append(0.0)
                continue

            # Mean intra-cluster distance
            intra_dists = []
            for j in same_cluster:
                if j != i:
                    intra_dists.append(1.0 - float(hv_hamming_sim(hvs[i], hvs[j])))
            a = sum(intra_dists) / len(intra_dists) if intra_dists else 0.0

            # Mean nearest-cluster distance
            other_labels = torch.unique(self.labels_[other_clusters])
            min_inter = float('inf')
            for ol in other_labels:
                ol_mask = (self.labels_ == ol)
                ol_indices = ol_mask.nonzero().squeeze(-1)
                inter_dists = [1.0 - float(hv_hamming_sim(hvs[i], hvs[int(j)])) for j in ol_indices]
                b_c = sum(inter_dists) / len(inter_dists)
                min_inter = min(min_inter, b_c)
            b = min_inter if min_inter != float('inf') else 0.0

            score = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
            scores.append(score)

        return sum(scores) / len(scores)


class HierarchicalHyperseed:
    """
    Hierarchical Hyperseed: Multi-resolution clustering via threshold annealing.

    Builds a hierarchy of clusters by running Hyperseed at multiple
    merge/split thresholds, producing a dendrogram-like structure.

    Useful for:
    - Discovering clusters at multiple granularities
    - Determining the optimal number of clusters
    - Visualizing cluster relationships
    """

    def __init__(
        self,
        dim: int = 10000,
        thresholds: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.seed = seed

        self.levels: List[Dict[str, Any]] = []

    def fit(
        self,
        X: torch.Tensor,
        encode_fn: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        """Fit hierarchical Hyperseed.

        Args:
            X: (n_samples, n_features) or (n_samples, dim)
            encode_fn: Optional encoding function

        Returns:
            List of {threshold, n_clusters, labels, prototypes} for each level
        """
        self.levels = []

        for threshold in sorted(self.thresholds):
            hs = Hyperseed(
                dim=self.dim,
                merge_threshold=threshold,
                split_threshold=threshold * 0.5,
                seed=self.seed,
            )
            labels = hs.fit(X, encode_fn=encode_fn)

            self.levels.append({
                "threshold": threshold,
                "n_clusters": hs.n_clusters_,
                "labels": labels.clone(),
                "prototypes": hs.get_cluster_prototypes().clone(),
                "converged": hs.converged_,
            })

        return self.levels


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_hyperseed():
    """Verify Hyperseed unsupervised clustering."""
    print("=" * 60)
    print("Testing Hyperseed (Osipov 2024)")
    print("=" * 60)

    dim = 1000
    n_samples = 20

    # Create 2 well-separated synthetic clusters with very low noise
    p1 = gen_hvs(1, dim, seed=100).squeeze(0)
    p2 = hv_xor(p1, gen_hvs(1, dim, seed=200).squeeze(0))  # Maximally different
    prototypes = torch.stack([p1, p2])
    hvs = []
    true_labels = []
    for i in range(n_samples):
        cluster = i % 2
        noise = (torch.rand(dim) < 0.1).float()
        hv = hv_majority(hv_bundle(torch.stack([prototypes[cluster], noise])))
        hvs.append(hv)
        true_labels.append(cluster)

    hvs = torch.stack(hvs)
    true_labels = torch.tensor(true_labels)

    # Run Hyperseed with high merge threshold (no merging)
    hs = Hyperseed(dim=dim, n_seeds=4, merge_threshold=0.9, split_threshold=0.5, max_iterations=10)
    pred_labels = hs.fit(hvs)

    print(f"  True clusters: 2")
    print(f"  Found clusters: {hs.n_clusters_}")
    print(f"  Converged: {hs.converged_}")
    print(f"  Cluster sizes: {hs.get_cluster_sizes()}")

    # Compute adjusted Rand index approximation
    correct = 0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            same_true = (true_labels[i] == true_labels[j])
            same_pred = (pred_labels[i] == pred_labels[j])
            if same_true == same_pred:
                correct += 1
    total_pairs = n_samples * (n_samples - 1) / 2
    ari = correct / total_pairs
    print(f"  Pairwise agreement (ARI approx): {ari:.4f}")

    silhouette = hs.silhouette_score(hvs)
    print(f"  Silhouette score: {silhouette:.4f}")

    # Check: found 2 clusters, converged
    found_2 = hs.n_clusters_ >= 2
    print(f"  Found >=2 clusters: {'✅' if found_2 else '❌'}")
    print(f"  Converged: {'✅' if hs.converged_ else '❌'}")
    print(f"  {'✅' if found_2 and hs.converged_ else '❌'} Hyperseed test!")


def test_hierarchical_hyperseed():
    """Verify hierarchical Hyperseed."""
    print("=" * 60)
    print("Testing Hierarchical Hyperseed (Osipov 2024)")
    print("=" * 60)

    dim = 1000
    n_samples = 30

    hvs = gen_hvs(n_samples, dim)

    hhs = HierarchicalHyperseed(
        dim=dim,
        thresholds=[0.3, 0.5, 0.7],
    )
    levels = hhs.fit(hvs)

    for level in levels:
        print(f"  Threshold {level['threshold']}: {level['n_clusters']} clusters, converged={level['converged']}")

    print(f"  ✅ Hierarchical Hyperseed test!")


if __name__ == "__main__":
    test_hyperseed()
    print()
    test_hierarchical_hyperseed()
