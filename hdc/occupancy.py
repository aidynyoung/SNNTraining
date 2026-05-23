"""
VSA Occupancy Grid Maps for RL-Based Navigation
=================================================
Snyder, Shea, Capodieci, Gorsich & Parsa (2025)
"Generalizable Reinforcement Learning with Biologically Inspired
Hyperdimensional Occupancy Grid Maps for Exploration and Goal-Directed
Path Planning"
arXiv:2502.09393 — George Mason University / Neya Robotics / US Army

Replaces Bayesian Hilbert Maps (BHM) — the classical probabilistic
occupancy grid approach — with a VSA-based alternative (VSA-OGM) that:
  1. Encodes spatial positions as HVs using random Fourier features (VFA)
  2. Accumulates occupied positions into a single occupancy HV
  3. Queries occupancy probability as Hamming/cosine similarity
  4. Feeds the occupancy HV directly to an RL policy network

This connects three Arthedain modules:
  - SpatialHDCEncoder (hdc/vfa.py)  — position encoding
  - KernelHDCRegressor (hdc/vfa.py) — occupancy function regression
  - SelfImprovementLoop (hdc/planner.py) — online map update

Algorithm (§III of paper):
  For each LiDAR point cloud observation:
    1. Transform polar → Cartesian coordinates
    2. For each ray-cast point (x, y, label):
         z(x,y) = SpatialHDCEncoder.encode(x, y)
    3. Accumulate: OGM_HV += label × z(x,y)    [KernelHDCRegressor]
    4. Query:  P(occupied at p) ≈ <z(p), OGM_HV> / n    [dot product]

Key results:
  - Achieves comparable RL performance to BHM on exploration + F1-Tenth tasks
  - Trains 3-5× faster than BHM (no hyperparameter tuning)
  - Generalises to unseen environments (BHM overfits to training maps)
  - 0.1ms query time (vs >10ms for Gaussian process methods)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from hdc.vfa import SpatialHDCEncoder, KernelHDCRegressor, KernelEncoder


# ═══════════════════════════════════════════════════════════════════════════════
# 1. VSA-OGM — core occupancy grid map
# ═══════════════════════════════════════════════════════════════════════════════

class VSAOGM:
    """
    VSA Occupancy Grid Map (Snyder et al. 2025, §III).

    Represents a 2D occupancy map as a single hypervector by accumulating
    the VFA-encoded positions of all observed points, weighted by their
    occupancy labels (1=occupied, -1=free).

    Training (map update):
        For each observation (x, y, label) where label ∈ {-1, +1}:
            z = spatial_encoder.encode(x, y)       [position HV]
            OGM_HV += label × z                    [accumulate]

    Inference (occupancy query):
        P(occupied at p) ≈ dot(z(p), OGM_HV) / n_observations

    The inner product recovers the kernel regression:
        <z(p), OGM_HV> / n ≈ Σ_i label_i × K(p, p_i) / n
    which is the kernel-weighted average of nearby labels — exactly
    the Nadaraya-Watson kernel regression estimate.

    Integration with RL:
        The OGM_HV is used directly as the observation for the RL policy.
        Alternatively, query a fixed grid of positions to produce a
        feature map for CNN/MLP policy networks.

    Args:
        hd_dim: HV dimensionality
        spatial_bandwidth: σ for the 2D Gaussian kernel (controls smoothing)
        kernel: 'gaussian' | 'laplacian' | 'periodic'
        seed: Random seed
    """

    def __init__(
        self,
        hd_dim: int = 4096,
        spatial_bandwidth: float = 0.5,
        kernel: str = "gaussian",
        seed: int = 42,
    ):
        self.hd_dim = hd_dim

        # 2D spatial encoder: encodes (x, y) → complex HV
        self.spatial_enc = SpatialHDCEncoder(
            hd_dim=hd_dim,
            dims=2,
            kernel=kernel,
            bandwidth=spatial_bandwidth,
            seed=seed,
        )

        # Accumulator for the occupancy function HV
        self._ogm_hv = torch.zeros(hd_dim)   # real-valued accumulator
        self._n = 0

        # Normalisation constant
        self._ogm_norm: Optional[float] = None

        # Temporal decay: for dynamic environments where occupancy changes over time
        self._decay_factor: float = 1.0   # 1.0 = no decay (static map)

    def update(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Update the occupancy map with new observations.

        Args:
            points: (N, 2) Cartesian coordinates (x, y)
            labels: (N,) occupancy labels: +1 = occupied, -1 = free
        """
        # Encode all positions as complex HVs (N, D)
        Z = self.spatial_enc.encode_batch(points)   # (N, D) complex

        # Temporal decay: fade old observations for dynamic environments
        if self._decay_factor < 1.0:
            self._ogm_hv *= self._decay_factor

        # Accumulate: OGM_HV += Σ label_i × z(x_i, y_i)
        y = labels.float()                          # (N,)
        weighted_real = (Z.real * y.unsqueeze(-1)).sum(dim=0)   # (D,)
        self._ogm_hv += weighted_real
        self._n += len(points)
        self._ogm_norm = None   # invalidate cache

    def set_temporal_decay(self, decay: float = 0.99):
        """
        Enable temporal decay for dynamic environments.

        Each time update() is called, old observations are down-weighted
        by `decay`, allowing recent observations to dominate.  Use:
            decay=1.0 (default) → static map (no decay)
            decay=0.99          → slow decay (~100 updates to halve old info)
            decay=0.90          → fast decay (~7 updates to halve old info)

        Args:
            decay: Multiplicative decay factor ∈ (0, 1]
        """
        self._decay_factor = float(max(0.01, min(1.0, decay)))

    def query(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Query occupancy probability at a set of positions.

        P(occupied at p) ≈ Re(<z(p), OGM_HV>) / n_observations

        Args:
            positions: (M, 2) or (2,) query positions

        Returns:
            (M,) or scalar occupancy probabilities ∈ [-1, 1]
            (positive = likely occupied, negative = likely free)
        """
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        Z_q = self.spatial_enc.encode_batch(positions)      # (M, D) complex
        ogm_c = self._ogm_hv.unsqueeze(0)                   # (1, D) real as complex proxy

        # Inner product: Re(Z_q^* · OGM_HV)
        probs = (Z_q.real * self._ogm_hv.unsqueeze(0)).mean(dim=-1)  # (M,)

        # Normalise by number of observations
        if self._n > 0:
            probs = probs / math.sqrt(self._n)

        return probs.squeeze(0) if squeeze else probs

    def binary_map(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        resolution: float = 0.1,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """
        Render a binary occupancy grid.

        Args:
            x_range: (x_min, x_max) world coordinates
            y_range: (y_min, y_max) world coordinates
            resolution: Grid cell size in world units
            threshold: P(occupied) threshold for occupied/free decision

        Returns:
            (nx, ny) binary occupancy grid (1=occupied, 0=free)
        """
        xs = torch.arange(x_range[0], x_range[1], resolution)
        ys = torch.arange(y_range[0], y_range[1], resolution)
        nx, ny = len(xs), len(ys)

        grid_points = torch.stack([
            xs.repeat(ny),
            ys.repeat_interleave(nx)
        ], dim=-1)  # (nx*ny, 2)

        probs = self.query(grid_points)        # (nx*ny,)
        occupied = (probs > threshold).float()
        return occupied.view(ny, nx)

    def ogm_feature_vector(
        self,
        query_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Produce a fixed-size feature vector by querying a grid.

        For RL policy networks that need a fixed-size input regardless
        of the sensor resolution.

        Args:
            query_grid: (K, 2) fixed set of query positions (e.g. 16×16 grid)

        Returns:
            (K,) occupancy feature vector suitable for RL policy input
        """
        return self.query(query_grid)

    @property
    def ogm_hv(self) -> torch.Tensor:
        """The raw OGM hypervector (for direct use as RL observation)."""
        return self._ogm_hv.clone()

    def reset(self):
        """Clear the map."""
        self._ogm_hv.zero_()
        self._n = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LiDAR → Cartesian converter
# ═══════════════════════════════════════════════════════════════════════════════

def polar_to_cartesian(
    ranges: torch.Tensor,
    angles: torch.Tensor,
    pose: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convert polar LiDAR scan to Cartesian point cloud (§III-A, Fig. 1).

    Args:
        ranges: (N,) range measurements in metres
        angles: (N,) angle measurements in radians
        pose: Optional (3,) agent pose (x, y, θ) for global frame transform

    Returns:
        (N, 2) Cartesian coordinates in the agent or global frame
    """
    x = ranges * torch.cos(angles)
    y = ranges * torch.sin(angles)
    points = torch.stack([x, y], dim=-1)   # (N, 2)

    if pose is not None:
        # Transform to global frame
        px, py, theta = float(pose[0]), float(pose[1]), float(pose[2])
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        R = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]])
        t = torch.tensor([px, py])
        points = points @ R.T + t

    return points


def raytrace_labels(
    points: torch.Tensor,
    n_free_per_ray: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate (occupied, free) training points from a point cloud.

    For each LiDAR hit point (occupied), generate n_free_per_ray
    intermediate points along the ray from origin to hit (free space).

    Args:
        points: (N, 2) hit positions (occupied)
        n_free_per_ray: Number of free-space samples per ray

    Returns:
        all_points: (N*(n_free+1), 2) all points
        labels: (N*(n_free+1),) ∈ {+1 occupied, -1 free}
    """
    all_pts, all_labels = [], []

    origin = torch.zeros(2)
    for pt in points:
        # Occupied: the hit point
        all_pts.append(pt)
        all_labels.append(1.0)

        # Free: intermediate points along the ray
        for k in range(1, n_free_per_ray + 1):
            frac = k / (n_free_per_ray + 1)
            free_pt = origin + frac * (pt - origin)
            all_pts.append(free_pt)
            all_labels.append(-1.0)

    return torch.stack(all_pts), torch.tensor(all_labels)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RL-ready wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class VSAOGMAgent:
    """
    RL agent interface using VSA-OGM as the perception backbone.

    Wraps VSAOGM to produce fixed-size observations for standard RL
    policy networks (PPO, SAC, DQN, etc.).

    Observation modes:
      'hv':    Return the raw OGM HV (hd_dim,) — large but information-rich
      'grid':  Return occupancy values at a fixed query grid (k×k,) — compact

    The OGM is updated incrementally on each step from LiDAR observations.
    The policy receives the current map state as its observation.

    Args:
        hd_dim: OGM HV dimension
        obs_mode: 'hv' or 'grid'
        grid_size: K for K×K query grid (only used if obs_mode='grid')
        world_size: (x_range, y_range) for grid placement
        spatial_bandwidth: Kernel smoothing bandwidth
    """

    def __init__(
        self,
        hd_dim: int = 2048,
        obs_mode: str = "grid",
        grid_size: int = 16,
        world_size: float = 10.0,
        spatial_bandwidth: float = 0.5,
        seed: int = 42,
    ):
        self.hd_dim = hd_dim
        self.obs_mode = obs_mode
        self.grid_size = grid_size

        self.ogm = VSAOGM(hd_dim=hd_dim, spatial_bandwidth=spatial_bandwidth, seed=seed)

        # Fixed query grid for 'grid' mode
        g = torch.linspace(-world_size/2, world_size/2, grid_size)
        xs, ys = torch.meshgrid(g, g, indexing='ij')
        self._query_grid = torch.stack([xs.flatten(), ys.flatten()], dim=-1)  # (K², 2)

    def reset(self):
        """Reset the OGM at the start of an episode."""
        self.ogm.reset()
        return self._observation()

    def step(self, lidar_ranges: torch.Tensor, lidar_angles: torch.Tensor,
             pose: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Update OGM from LiDAR scan and return new observation.

        Args:
            lidar_ranges: (N,) range measurements
            lidar_angles: (N,) angle measurements
            pose: Optional (3,) agent pose for global frame

        Returns:
            Observation tensor
        """
        points = polar_to_cartesian(lidar_ranges, lidar_angles, pose)
        all_pts, labels = raytrace_labels(points, n_free_per_ray=2)
        self.ogm.update(all_pts, labels)
        return self._observation()

    def _observation(self) -> torch.Tensor:
        if self.obs_mode == "hv":
            return self.ogm.ogm_hv
        else:
            return self.ogm.ogm_feature_vector(self._query_grid)

    @property
    def obs_dim(self) -> int:
        if self.obs_mode == "hv":
            return self.hd_dim
        return self.grid_size ** 2


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_vsaogm():
    print("=" * 60)
    print("Testing VSAOGM (Snyder et al. 2025, arXiv:2502.09393)")
    print("=" * 60)

    torch.manual_seed(42)
    ogm = VSAOGM(hd_dim=2000, spatial_bandwidth=0.5, seed=0)

    # Simulate a wall at x=2 with free space before it
    occupied = torch.tensor([[2.0, y] for y in torch.linspace(-1, 1, 20)])
    free = torch.tensor([[x, 0.0] for x in torch.linspace(0, 1.8, 20)])

    occ_labels = torch.ones(len(occupied))
    free_labels = -torch.ones(len(free))

    ogm.update(occupied, occ_labels)
    ogm.update(free, free_labels)

    # Query: points near the wall should have high occupancy
    near_wall  = torch.tensor([[2.0, 0.0]])
    free_space = torch.tensor([[1.0, 0.0]])
    empty      = torch.tensor([[4.0, 0.0]])

    p_wall  = float(ogm.query(near_wall)[0])
    p_free  = float(ogm.query(free_space)[0])
    p_empty = float(ogm.query(empty)[0])

    print(f"  P(occupied near wall x=2): {p_wall:.4f}  (want high)")
    print(f"  P(occupied free space x=1): {p_free:.4f}  (want low/negative)")
    print(f"  P(occupied empty x=4):  {p_empty:.4f}")
    assert p_wall > p_free, "Wall should be more occupied than free space"

    # Binary map
    bmap = ogm.binary_map((-3, 5), (-2, 2), resolution=0.2)
    print(f"  Binary map shape: {bmap.shape}, fraction occupied: {bmap.mean():.3f}")
    assert bmap.shape[0] > 0 and bmap.shape[1] > 0

    print("  ✅ VSAOGM OK")


def test_lidar_conversion():
    print("=" * 60)
    print("Testing LiDAR → Cartesian conversion")
    print("=" * 60)

    # 360° scan with uniform range 5m
    angles = torch.linspace(0, 2*math.pi, 36)
    ranges = torch.ones(36) * 5.0
    pts = polar_to_cartesian(ranges, angles)

    # Should form a circle of radius 5
    radii = pts.norm(dim=-1)
    print(f"  Radii: mean={radii.mean():.4f}, std={radii.std():.4f}  (want ≈5.0, 0.0)")
    assert abs(radii.mean() - 5.0) < 0.1

    # With pose transform
    pose = torch.tensor([1.0, 2.0, math.pi/4])  # at (1,2), facing 45°
    pts_global = polar_to_cartesian(ranges, angles, pose)
    print(f"  Global frame: centred at ≈({pts_global[:,0].mean():.2f}, {pts_global[:,1].mean():.2f})")

    # Raytrace labels
    all_pts, labels = raytrace_labels(pts[:5], n_free_per_ray=2)
    n_occ  = (labels > 0).sum().item()
    n_free = (labels < 0).sum().item()
    print(f"  5 hits → {n_occ} occupied + {n_free} free labels")
    assert n_occ == 5 and n_free == 10

    print("  ✅ LiDAR conversion OK")


def test_vsaogm_agent():
    print("=" * 60)
    print("Testing VSAOGMAgent (RL interface)")
    print("=" * 60)

    torch.manual_seed(99)
    agent = VSAOGMAgent(hd_dim=1000, obs_mode="grid", grid_size=8, seed=0)
    print(f"  Observation dimension: {agent.obs_dim}")
    assert agent.obs_dim == 64  # 8×8

    # Reset
    obs0 = agent.reset()
    print(f"  Initial obs shape: {obs0.shape}  (all zeros: {(obs0==0).all().item()})")
    assert obs0.shape == (64,)

    # Step with synthetic LiDAR
    angles = torch.linspace(0, math.pi, 18)
    ranges = torch.ones(18) * 3.0
    obs1 = agent.step(ranges, angles)
    print(f"  After LiDAR step: obs range [{obs1.min():.4f}, {obs1.max():.4f}]")
    assert obs1.shape == (64,)

    # HV mode
    agent_hv = VSAOGMAgent(hd_dim=1000, obs_mode="hv", seed=0)
    agent_hv.reset()
    obs_hv = agent_hv.step(ranges, angles)
    print(f"  HV obs shape: {obs_hv.shape}  (obs_dim={agent_hv.obs_dim})")
    assert obs_hv.shape == (1000,)

    print("  ✅ VSAOGMAgent OK")


if __name__ == "__main__":
    test_vsaogm()
    print()
    test_lidar_conversion()
    print()
    test_vsaogm_agent()
    print()
    print("=== All VSA-OGM tests passed ===")
