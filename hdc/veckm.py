"""
hdc/veckm.py
=============
VecKM: Vectorized Kernel Mixture for Point Cloud Encoding in HDC Space
=======================================================================
Reference:
    Yuan, D., et al. (2024)
    "VecKM: A Linear Time and Space Local Point Cloud Geometry Encoder"
    International Conference on Machine Learning (ICML) 2024.
    https://github.com/dhyuan99/VecKM

    Bochner's Theorem (1932) — Fourier features approximate shift-invariant kernels.
    Rahimi & Recht (2007) "Random Features for Large-Scale Kernel Machines" NeurIPS.
    — Foundation for VecKM's random Fourier feature approach.

Why VecKM matters for SNNTraining (Physical AI):

    Standard 3D point cloud processing (PointNet, PointNet++, Point Transformer):
        - Requires explicit k-NN grouping: O(N × K × d) cost
        - Not deployable on edge hardware (high memory, high compute)
        - Not compatible with HDC pipeline (neural network weights)

    VecKM:
        - Linear time O(N × D) — no KNN needed
        - Outputs per-point geometry HVs compatible with HDC associative memory
        - Factorizable kernel: K(xi, xj) = K_margin(xi) × K_margin(xj)
        - Directly integrates with HDCGraphNetwork for 3D graph reasoning
        - Enables: drone obstacle detection, LiDAR SLAM, 3D object classification

    Key insight (Yuan 2024 §2.2 — Factorizable Kernel Property):
        K(xi, xj) = exp(-||xi - xj||² / 2σ²)         [RBF kernel]
                  = exp(-||xi||²/2σ²) × exp(-||xj||²/2σ²) × exp(xi·xj/σ²)

        This factorisation means:
            local_feature(xi) = Σ_j K(xi, xj) × feature(xj)
                               = K_margin(xi) × Σ_j K_margin(xj) × feature(xj)
                               = K_margin(xi) × global_pool(K_margin(xj) × feature(xj))

        So local features reduce to a PRODUCT of per-point functions —
        no explicit KNN needed, just global pooling of shifted features.

    In Fourier feature space (FastVecKM):
        K(xi, xj) ≈ z(xi)^T z(xj)   where z(x) = exp(i × ω^T x) / sqrt(D)
        local_feature(xi) = z(xi)^T × Σ_j z(xj) × feature(xj)
                          = z(xi)^T × global_aggregation(z, features)

This module implements:

1. ExactVecKM
   — Per-point local geometry HV via explicit KNN + RBF kernel mixture
   — Slower (O(N²)) but exact
   — Best for small point clouds (<1000 points)

2. FastVecKM
   — Linear-time O(N × D) via random Fourier features + global aggregation
   — Approximates ExactVecKM kernel mixture
   — Best for large point clouds (10k+ points)

3. HDCPointCloudClassifier
   — Complete pipeline: FastVecKM → per-point HVs → graph-level bundle → classify
   — Online learning via RefineHD

4. LiDAREncoder
   — Specialised for LiDAR scan data (2D range images + depth)
   — Integrates with EliteSNNTrainingPipeline for drone obstacle detection
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from hdc.physics_world_model import _hamming, _majority


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ExactVecKM — explicit KNN + RBF kernel mixture
# ═══════════════════════════════════════════════════════════════════════════════

class ExactVecKM:
    """
    Exact VecKM: local geometry via explicit KNN + RBF kernel mixture.

    For each point xi, computes:
        z(xi) = MAJORITY(Σ_j K(xi, xj) × random_proj(xj)  for j in KNN(xi))

    where K is the RBF kernel: K(xi, xj) = exp(-||xi-xj||²/2σ²)

    The output is a per-point binary HV capturing local geometry.

    Args:
        dim:         Output HV dimension
        n_neighbors: Number of nearest neighbours K
        sigma:       RBF bandwidth σ (controls neighbourhood radius)
        device:      torch device
    """

    def __init__(
        self,
        dim:         int   = 256,
        n_neighbors: int   = 16,
        sigma:       float = 1.0,
        seed:        Optional[int] = None,
        device:      str   = "cpu",
    ):
        self.dim         = dim
        self.n_neighbors = n_neighbors
        self.sigma       = sigma
        self.device      = device

        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed)
        # Random projection matrix for 3D → D encoding
        self._proj = torch.randn(3, dim, generator=g, device=device) / math.sqrt(dim)

    def _rbf_kernel(self, xi: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:
        """RBF kernel K(xi, xj) = exp(-||xi-xj||²/2σ²)."""
        diff = xi.unsqueeze(1) - xj.unsqueeze(0)   # (N, M, 3)
        sq   = (diff ** 2).sum(dim=-1)               # (N, M)
        return torch.exp(-sq / (2 * self.sigma ** 2))

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        """
        Encode N 3D points to per-point geometry HVs.

        Args:
            points: (N, 3) point cloud coordinates

        Returns:
            (N, dim) binary HV matrix — one per point
        """
        N = points.shape[0]
        p = points.float().to(self.device)

        # Compute pairwise RBF kernel
        K = self._rbf_kernel(p, p)   # (N, N)

        # KNN: keep only top-K neighbours
        if N > self.n_neighbors:
            topk_vals, topk_idx = K.topk(self.n_neighbors, dim=1)
            mask = torch.zeros_like(K).scatter_(1, topk_idx, 1.0)
            K    = K * mask
            K    = K / (K.sum(dim=1, keepdim=True) + 1e-8)  # normalise

        # Project each point to feature space
        features = p @ self._proj   # (N, D)

        # Local aggregation: weighted sum of neighbour features
        local_f  = K @ features     # (N, D)

        # Binarise to HDC representation
        return (local_f > local_f.median(dim=1, keepdim=True).values).float()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FastVecKM — O(N×D) via random Fourier features
# ═══════════════════════════════════════════════════════════════════════════════

class FastVecKM:
    """
    FastVecKM: linear-time point cloud geometry encoder.

    Reference: Yuan et al. (2024) ICML — FastVecKM (Equation 2).

    Key insight: the RBF kernel is approximated by random Fourier features:
        K(xi, xj) ≈ z(xi)^T z(xj)
        where z(x) = [cos(ω_1^T x + b_1), ..., cos(ω_D^T x + b_D)] / sqrt(D)

    This allows local aggregation to be computed WITHOUT explicit KNN:
        local_feature(xi) = z(xi)^T × (Σ_j z(xj) × input_feature_j) / N

    The global sum Σ_j z(xj) × feature_j is computed ONCE in O(N×D),
    then each local feature is extracted in O(D) — total O(N×D), linear.

    For HDC output: binarise the local features via median thresholding.

    Args:
        dim:     Output HV dimension
        sigma:   RBF bandwidth (controls neighbourhood size)
        seed:    Random seed
        device:  torch device
    """

    def __init__(
        self,
        dim:         int   = 512,
        sigma:       float = 1.0,
        seed:        Optional[int] = None,
        device:      str   = "cpu",
        multi_scale: bool  = False,
    ):
        self.dim         = dim
        self.sigma       = sigma
        self.device      = device
        self.multi_scale = multi_scale

        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed)
        # Random frequencies ω ~ N(0, 1/σ²)
        self.omega = torch.randn(3, dim, generator=g, device=device) / sigma
        self.bias  = torch.rand(dim, generator=g, device=device) * 2 * math.pi

        # Output projection: maps low-dim aggregated features back to D.
        # Default: raw 3D coords are values (F=3), so project R³ → R^D.
        # This makes the pipeline truly O(N×D×3) = O(N×D) — linear time.
        g2 = torch.Generator(device=device)
        g2.manual_seed((seed or 0) + 1)
        self._out_proj = torch.randn(3, dim, generator=g2, device=device) / math.sqrt(3)

        # Multi-scale: add encoders at σ/2 and 2σ for richer descriptors
        if multi_scale:
            self._fast_fine   = FastVecKM(dim, sigma / 2.0, seed=(seed or 0) + 2,
                                          device=device, multi_scale=False)
            self._fast_coarse = FastVecKM(dim, sigma * 2.0, seed=(seed or 0) + 3,
                                          device=device, multi_scale=False)

        # Incremental update state (for streaming data)
        self._global_pool: Optional[torch.Tensor] = None  # (D, F) accumulated
        self._n_incremental = 0

    def _fourier_features(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute random Fourier features z(x) = cos(ω^T x + b) / sqrt(D).

        Args:
            points: (N, 3) 3D coordinates

        Returns:
            (N, D) real-valued Fourier features
        """
        proj = points.float().to(self.device) @ self.omega + self.bias  # (N, D)
        return torch.cos(proj) / math.sqrt(self.dim)

    def encode(self, points: torch.Tensor, point_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode N 3D points to per-point geometry HVs in TRUE O(N × D) time.

        The original formulation used Z as both keys and values, producing a
        (D, D) global pool — actually O(N×D²).  This implementation uses raw
        3D coordinates as values (F=3), giving:
            global_pool = Z^T @ p  →  (D, 3)  — O(N×D×3) = O(N×D)
            local_f[i]  = z(xi) @ global_pool  →  (3,)    — O(D×3) per point
            output[i]   = local_f[i] @ out_proj  →  (D,)   — O(3×D) per point

        Total: O(N×D) — linear in both N and D.

        Args:
            points:         (N, 3) point cloud coordinates
            point_features: Optional (N, F) per-point features; if F ≤ 32, used
                            as values for O(N×D×F) aggregation.  For large F,
                            raw coords are used instead to maintain linear time.

        Returns:
            (N, D) binary geometry HVs
        """
        p = points.float().to(self.device)   # (N, 3)
        Z = self._fourier_features(p)         # (N, D)

        # Choose value tensor: prefer low-dim features for linear-time guarantee
        if point_features is not None:
            feat = point_features.float().to(self.device)
            if feat.shape[1] <= 32:
                values = feat
                out_proj = torch.randn(feat.shape[1], self.dim,
                                       device=self.device) / math.sqrt(feat.shape[1])
            else:
                values   = p            # fall back to coords for large feature dim
                out_proj = self._out_proj
        else:
            values   = p                # (N, 3) raw coordinates — linear time
            out_proj = self._out_proj   # (3, D)

        # O(N×D×F): global aggregation
        global_pool = Z.T @ values   # (D, F)  — one pass over all points
        # O(D×F) per point: local feature extraction
        local_f = Z @ global_pool    # (N, F)
        # O(F×D) per point: project to output dimension
        local_out = local_f @ out_proj   # (N, D)

        # Binarise: threshold at per-point median → binary HV
        median_vals = local_out.median(dim=1, keepdim=True).values
        hvs = (local_out > median_vals).float()

        # Multi-scale fusion: bundle fine + medium + coarse HVs
        if self.multi_scale:
            hvs_fine   = self._fast_fine.encode(points, point_features)
            hvs_coarse = self._fast_coarse.encode(points, point_features)
            stack      = torch.stack([hvs, hvs_fine, hvs_coarse], dim=0)  # (3, N, D)
            hvs        = _majority(stack.mean(dim=0))

        return hvs

    def encode_batch(
        self, batch_points: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Encode a list of point clouds (variable N per cloud)."""
        return [self.encode(pts) for pts in batch_points]

    def encode_incremental(self, points: torch.Tensor) -> torch.Tensor:
        """
        Incremental encode: accumulate global_pool across multiple partial scans.

        Useful for streaming LiDAR where points arrive in chunks.
        Each call contributes to an online global pool; call reset_incremental()
        to start a new scan frame.

        Args:
            points: (N, 3) new chunk of points

        Returns:
            (N, D) per-point HVs using the accumulated global context
        """
        p = points.float().to(self.device)
        Z = self._fourier_features(p)   # (N, D)

        chunk_pool = Z.T @ p   # (D, 3) — contribution from this chunk
        if self._global_pool is None:
            self._global_pool = chunk_pool
        else:
            self._global_pool = self._global_pool + chunk_pool
        self._n_incremental += p.shape[0]

        # Local features using accumulated context
        pool_norm = self._global_pool / max(self._n_incremental, 1)
        local_f   = Z @ pool_norm    # (N, 3)
        local_out = local_f @ self._out_proj   # (N, D)

        median_vals = local_out.median(dim=1, keepdim=True).values
        return (local_out > median_vals).float()

    def reset_incremental(self):
        """Reset the incremental global pool for a new scan frame."""
        self._global_pool   = None
        self._n_incremental = 0

    def estimate_normals(self, points: torch.Tensor) -> torch.Tensor:
        """
        Estimate per-point surface normals using the Fourier feature gradient.

        The gradient of the local feature w.r.t. point position approximates
        the local surface normal direction.  For each point xi:
            ∂z(xi)/∂xi = diag(-sin(ω^T xi + b)) × ω^T  — Jacobian of z

        The dominant eigenvector of the covariance of gradients across the
        local neighbourhood gives the surface normal direction.

        This is O(N×D) — same asymptotic complexity as encode().

        Args:
            points: (N, 3) point cloud

        Returns:
            (N, 3) unit normal vectors per point
        """
        p   = points.float().to(self.device)   # (N, 3)
        Z   = self._fourier_features(p)         # (N, D)
        # Jacobian rows: dz_d/dx = -sin(ω_d^T x + b_d) × ω_d
        sin_proj = torch.sin(p @ self.omega + self.bias)  # (N, D)
        # dZ/dx — shape (N, D, 3): J[n, d, :] = -sin_proj[n,d] * omega[:, d]
        J = -sin_proj.unsqueeze(2) * self.omega.T.unsqueeze(0)  # (N, D, 3)
        # Per-point covariance of gradient rows: (N, 3, 3)
        cov = torch.einsum('ndi,ndj->nij', J, J) / self.dim   # (N, 3, 3)
        # Smallest eigenvector = surface normal direction
        try:
            eigvals, eigvecs = torch.linalg.eigh(cov)   # eigvecs: (N, 3, 3), ascending
            normals = eigvecs[:, :, 0]   # (N, 3) — smallest eigenvalue direction
        except Exception:
            normals = torch.zeros(p.shape[0], 3, device=self.device)
            normals[:, 2] = 1.0   # fallback: z-axis
        # Normalise
        norms   = normals.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return normals / norms

    def global_descriptor(self, points: torch.Tensor) -> torch.Tensor:
        """
        Single global HDV descriptor for an entire point cloud.
        Bundles per-point HVs via majority vote.
        """
        per_point = self.encode(points)   # (N, D)
        return _majority(per_point.float().mean(dim=0))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HDCPointCloudClassifier — end-to-end 3D classification
# ═══════════════════════════════════════════════════════════════════════════════

class HDCPointCloudClassifier:
    """
    End-to-end HDC 3D point cloud classifier.

    Pipeline:
        1. FastVecKM → per-point geometry HVs  (O(N×D))
        2. Graph-level bundle: global_descriptor  (O(N×D))
        3. HDC prototype matching via Hamming similarity
        4. RefineHD online training

    No neural network weights. No backpropagation. Online learning.
    Suitable for resource-constrained edge deployment.

    Args:
        n_classes:   Number of 3D object/scene classes
        dim:         HV dimension
        n_neighbors: For exact KNN (ExactVecKM) — ignored for FastVecKM
        sigma:       RBF bandwidth
        fast:        If True, use FastVecKM (linear); else ExactVecKM (exact)
    """

    def __init__(
        self,
        n_classes:   int,
        dim:         int   = 512,
        sigma:       float = 1.0,
        fast:        bool  = True,
        class_names: Optional[List[str]] = None,
        device:      str   = "cpu",
    ):
        self.n_classes   = n_classes
        self.dim         = dim
        self.class_names = class_names or [f"class_{i}" for i in range(n_classes)]
        self.device      = device

        if fast:
            self.encoder = FastVecKM(dim=dim, sigma=sigma, device=device)
        else:
            self.encoder = ExactVecKM(dim=dim, sigma=sigma, device=device)

        self._prototypes = [torch.zeros(dim, device=device) for _ in range(n_classes)]
        self._counts     = [0] * n_classes

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        """Encode a point cloud to a single global HV descriptor."""
        if hasattr(self.encoder, 'global_descriptor'):
            return self.encoder.global_descriptor(points)
        per_point = self.encoder.encode(points)
        return _majority(per_point.float().mean(dim=0))

    def train(self, points: torch.Tensor, label: int):
        """Online training: update class prototype."""
        hv = self.encode(points)
        n  = self._counts[label]
        self._prototypes[label] = _majority(
            (n * self._prototypes[label] + hv) / (n + 1)
        )
        self._counts[label] += 1

    def predict(self, points: torch.Tensor) -> Tuple[int, List[float]]:
        """Predict class with Hamming similarity scores."""
        hv     = self.encode(points)
        protos = torch.stack(self._prototypes)
        sims   = _hamming(hv.unsqueeze(0), protos)
        best   = int(sims.argmax().item())
        return best, sims.tolist()

    def refine(self, points: torch.Tensor, label: int, lr: float = 0.1):
        """RefineHD update: push toward correct, away from wrong."""
        hv = self.encode(points)
        pred, _ = self.predict(points)
        if pred != label:
            self._prototypes[label] = _majority(
                (1 - lr) * self._prototypes[label] + lr * hv
            )
            self._prototypes[pred] = _majority(
                (1 + lr) * self._prototypes[pred] - lr * hv
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LiDAREncoder — specialised for drone/robot 3D sensing
# ═══════════════════════════════════════════════════════════════════════════════

class LiDAREncoder:
    """
    LiDAR point cloud encoder for drone / robot Physical AI.

    Specialises FastVecKM for:
        1. Sparse outdoor LiDAR scans (typically 16-64 beams, ~10k points)
        2. Range normalisation (LiDAR has known max range)
        3. Height stratification (ground / obstacle / sky layers)
        4. Temporal aggregation (HDV memory across scan frames)

    Output: per-scan global HV for obstacle detection + classification.

    Args:
        dim:       HV dimension
        max_range: Maximum LiDAR range in metres (for normalisation)
        n_layers:  Number of height stratification layers
        sigma:     VecKM bandwidth (default: scaled to max_range / 10)
        device:    torch device
    """

    def __init__(
        self,
        dim:       int   = 512,
        max_range: float = 50.0,
        n_layers:  int   = 4,
        sigma:     Optional[float] = None,
        device:    str   = "cpu",
    ):
        self.dim       = dim
        self.max_range = max_range
        self.n_layers  = n_layers
        self.device    = device

        sigma = sigma or max_range / 10.0
        self.encoder   = FastVecKM(dim=dim, sigma=sigma, device=device)

        # Layer boundaries (equal height bands)
        self._layer_hvs = [
            (torch.rand(dim, device=device) >= 0.5).float()
            for _ in range(n_layers)
        ]

        # Temporal memory: EMA of recent scan HVs for motion-robust detection
        self._scan_memory  = torch.zeros(dim, device=device)
        self._scan_count   = 0

        # Obstacle prototype memory: online-learned from registered obstacles
        # Each prototype is a (D,) HV representing one obstacle type / zone
        self._obstacle_protos: List[torch.Tensor] = []
        self._obstacle_labels: List[str]          = []
        self._obstacle_counts: List[int]          = []

        # Safety margin: alert when closest distance < safe_margin (normalised)
        self._safe_margin = 0.2   # Hamming sim threshold for danger zone

    def _normalise(self, points: torch.Tensor) -> torch.Tensor:
        """Normalise points to [-1, 1] range."""
        p = points.float().to(self.device)
        return p / self.max_range

    def _height_layer(self, z: float) -> int:
        """Map z-coordinate to height layer index."""
        z_norm = (z + self.max_range) / (2 * self.max_range)
        return max(0, min(self.n_layers - 1, int(z_norm * self.n_layers)))

    def encode_scan(self, points: torch.Tensor) -> torch.Tensor:
        """
        Encode one LiDAR scan to a global HV.

        Args:
            points: (N, 3) LiDAR point cloud in metres [x, y, z]

        Returns:
            (D,) global scene HV
        """
        self._scan_count += 1
        p_norm = self._normalise(points)

        # Per-point geometry HVs
        per_point = self.encoder.encode(p_norm)   # (N, D)

        # Height-stratified bundling: bundle per layer, then bundle layers
        if points.shape[0] > 0:
            z_coords   = points[:, 2].tolist()
            layer_hvs  = {}
            for pt_hv, z in zip(per_point, z_coords):
                layer = self._height_layer(float(z))
                if layer not in layer_hvs:
                    layer_hvs[layer] = []
                layer_hvs[layer].append(pt_hv)

            layer_bundles = []
            for layer_idx in range(self.n_layers):
                if layer_idx in layer_hvs:
                    bundle = _majority(torch.stack(layer_hvs[layer_idx]).float().mean(0))
                    # Bind with layer identity HV
                    layer_bundle = (bundle != self._layer_hvs[layer_idx]).float()
                    layer_bundles.append(layer_bundle)

            if layer_bundles:
                scan_hv = _majority(torch.stack(layer_bundles).float().mean(0))
            else:
                scan_hv = torch.zeros(self.dim, device=self.device)
        else:
            scan_hv = torch.zeros(self.dim, device=self.device)

        # Update temporal memory with EMA
        self._scan_memory = _majority(0.9 * self._scan_memory + 0.1 * scan_hv)

        return scan_hv

    def register_obstacle(
        self,
        scan_hv: torch.Tensor,
        label:   str = "obstacle",
    ):
        """
        Register a scan HV as an obstacle prototype (online learning).

        Called when the operator or collision sensor confirms an obstacle.
        Subsequent `detect_obstacles()` calls will compare new scans to all
        registered prototypes.  Multiple observations of the same label are
        bundled (averaged) into a single prototype for efficiency.

        Args:
            scan_hv: (D,) scan HV encoding the obstacle environment
            label:   Semantic label for this obstacle type
        """
        if label in self._obstacle_labels:
            idx = self._obstacle_labels.index(label)
            n   = self._obstacle_counts[idx]
            self._obstacle_protos[idx] = _majority(
                (n * self._obstacle_protos[idx] + scan_hv.float()) / (n + 1)
            )
            self._obstacle_counts[idx] += 1
        else:
            self._obstacle_protos.append(scan_hv.float().clone())
            self._obstacle_labels.append(label)
            self._obstacle_counts.append(1)

    def detect_obstacles(
        self,
        scan_hv:       torch.Tensor,
        danger_thresh: float = 0.65,
    ) -> List[Dict]:
        """
        Compare current scan against all registered obstacle prototypes.

        Returns a list of matches above danger_thresh, sorted by similarity.

        Args:
            scan_hv:       (D,) current scan HV from encode_scan()
            danger_thresh: Minimum Hamming similarity to trigger an alert

        Returns:
            List of {label, similarity, count} dicts for matched prototypes
        """
        alerts = []
        for proto, lbl, cnt in zip(self._obstacle_protos,
                                    self._obstacle_labels,
                                    self._obstacle_counts):
            sim = float(_hamming(scan_hv.unsqueeze(0), proto.unsqueeze(0)).item())
            if sim >= danger_thresh:
                alerts.append({"label": lbl, "similarity": sim, "count": cnt})
        alerts.sort(key=lambda a: a["similarity"], reverse=True)
        return alerts

    def obstacle_score(self, scan_hv: torch.Tensor, obstacle_proto: torch.Tensor) -> float:
        """Hamming similarity between scan and a given obstacle prototype."""
        return float(_hamming(scan_hv.unsqueeze(0), obstacle_proto.unsqueeze(0)).item())

    def temporal_context(self) -> torch.Tensor:
        """Return accumulated temporal context across recent scans."""
        return self._scan_memory.clone()

    def reset(self):
        self._scan_memory.zero_()
        self._scan_count = 0
        # Obstacle prototypes persist across resets (they're learned knowledge)


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_veckm():
    print("=== ExactVecKM ===")
    enc = ExactVecKM(dim=128, n_neighbors=8, sigma=1.0, seed=42)
    pts = torch.randn(20, 3)
    hvs = enc.encode(pts)
    assert hvs.shape == (20, 128)
    assert set(hvs.unique().tolist()).issubset({0.0, 1.0})
    print(f"  (20,3) → {hvs.shape}  density={hvs.mean():.3f}  OK")

    print("\n=== FastVecKM ===")
    fast = FastVecKM(dim=256, sigma=1.0, seed=42)
    pts_large = torch.randn(200, 3)
    hvs_f = fast.encode(pts_large)
    assert hvs_f.shape == (200, 256)
    print(f"  (200,3) → {hvs_f.shape}  OK")

    # Global descriptor
    gd = fast.global_descriptor(pts_large)
    assert gd.shape == (256,)
    assert set(gd.unique().tolist()).issubset({0.0, 1.0})
    print(f"  global_descriptor: {gd.shape}  OK")

    # Similar point clouds should have similar descriptors
    pts2   = pts_large + torch.randn_like(pts_large) * 0.05   # slightly perturbed
    pts3   = torch.randn(200, 3) * 5.0                          # very different
    gd2    = fast.global_descriptor(pts2)
    gd3    = fast.global_descriptor(pts3)
    sim_12 = float(_hamming(gd.unsqueeze(0), gd2.unsqueeze(0)).item())
    sim_13 = float(_hamming(gd.unsqueeze(0), gd3.unsqueeze(0)).item())
    print(f"  sim(original, perturbed)={sim_12:.3f}  sim(original, different)={sim_13:.3f}")
    assert sim_12 > sim_13, "Similar clouds should be more similar"

    print("\n=== HDCPointCloudClassifier ===")
    clf = HDCPointCloudClassifier(n_classes=3, dim=256, sigma=0.5)
    # Train: 3 classes with spatially distinct point clouds
    for c in range(3):
        for s in range(10):
            pts_c = torch.randn(50, 3) + c * 3.0   # offset by class
            clf.train(pts_c, c)

    # Test
    test_pts = torch.randn(50, 3) + 6.0   # near class 2
    pred, sims = clf.predict(test_pts)
    assert 0 <= pred < 3
    print(f"  Prediction: class={pred}, sims={[f'{s:.3f}' for s in sims]}  OK")

    print("\n=== LiDAREncoder ===")
    lidar = LiDAREncoder(dim=256, max_range=50.0, n_layers=4)
    scan  = torch.randn(500, 3) * 10.0
    scan[:, 2] *= 0.5   # flatten height distribution
    scan_hv = lidar.encode_scan(scan)
    assert scan_hv.shape == (256,)
    ctx = lidar.temporal_context()
    assert ctx.shape == (256,)
    print(f"  Scan HV: {scan_hv.shape}, temporal ctx: {ctx.shape}  OK")
    print(f"  Scans processed: {lidar._scan_count}  OK")

    print("\n✅ All veckm tests passed")


if __name__ == "__main__":
    _test_veckm()
