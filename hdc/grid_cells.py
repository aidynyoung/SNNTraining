"""
Hippocampal Grid Cells and Path Integration via VSA
=====================================================
Kymn, Mazelet, Thomas, Kleyko, Frady, Sommer & Olshausen (2025)
"Binding in hippocampal-entorhinal circuits enables compositionality
in cognitive maps"
arXiv:2406.18808 — Redwood Center for Theoretical Neuroscience, UC Berkeley

Implements a normative model of spatial representation in the hippocampal
formation (HF) using Residue Number System (RNS) + VSA binding.

Core idea (§2, Fig. 1):
  - Grid cells in medial entorhinal cortex (MEC) use RNS for position
  - K grid modules with co-prime spatial periods {λ_1, ..., λ_K}
  - Each module encodes position x modulo λ_k as a phasor HV:
        g_k(x) = exp(i · 2π · x / λ_k)   [D complex components]
  - Total representable range = λ_1 × λ_2 × ... × λ_K (CRT)
  - Path integration via binding (Eq. 3):
        g(x + Δx) = g(x) ⊙ g(Δx)   [complex multiply = phase addition]
  - Place cells = superposition: p(x) = Σ_k g_k(x)

Key properties demonstrated:
  - Superlinear coding range: M scales as O(D^{αK}) where αK > 1
  - Path integration preserves accuracy despite noise accumulation
  - Resonator network decodes position from noisy place cell state
  - Hexagonal grid fields emerge from RNS+triangular lattice structure

Connection to SNNTraining:
  - SpatialHDCEncoder (hdc/vfa.py) uses random Fourier features
    → GridCellNetwork is the STRUCTURED alternative (theoretically optimal)
  - CognitiveMapHDC connects to VSAOGM occupancy maps (hdc/occupancy.py)
  - Path integration enables dead-reckoning without GPS
  - ContextualWorldModel can use place cell HVs as richer world state
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from hdc.hdc_glue import hv_batch_sim


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Single Grid Cell Module (RNS component)
# ═══════════════════════════════════════════════════════════════════════════════

class GridCellModule:
    """
    Single grid cell module: encodes position modulo a spatial period λ.

    Uses FHRR (complex phasors) to encode position x ∈ ℝ as:
        g(x) = base ^ (x / λ)   [element-wise complex power]
    where base_j = exp(i · φ_j) for random phases φ_j.

    Key property (Eq. 3 of paper):
        g(x + Δx) = g(x) ⊙ g(Δx)   [binding = translation]

    This makes path integration trivial:
        new_pos = old_pos ⊙ velocity_displacement

    The encoding has period λ: g(x) = g(x + λ).
    This is the definition of a grid cell: periodic firing fields.

    For 2D:
        g(x, y) = g_x(x) ⊙ g_y(y)   [bind x and y independently]

    Args:
        dim: Complex HV dimension D
        period: Spatial period λ (distance between repeating fields)
        seed: Random seed for base phasor
    """

    def __init__(self, dim: int, period: float, seed: int = 0):
        self.dim = dim
        self.period = period

        g = torch.Generator()
        g.manual_seed(seed)

        # Random INTEGER harmonics n_j ∈ {1,2,...,max_n}
        # g_j(x) = exp(i · n_j · 2π · x / λ)
        # Periodic with period λ regardless of n_j (since exp(i·n·2π)=1)
        # Path integration exact: g_j(x+Δx) = g_j(x) · g_j(Δx)
        max_n = max(3, dim // 32)
        self._harmonics = torch.randint(1, max_n + 1, (dim,), generator=g).float()

    def encode(self, x: float) -> torch.Tensor:
        """
        Encode scalar position x as a periodic phasor HV.

        g_j(x) = exp(i · n_j · 2π · x / λ)

        Periodic with period λ: g(x+λ) = g(x)  ✓
        Homomorphic: g(x+y) = g(x) ⊙ g(y)     ✓

        Args:
            x: Position value

        Returns:
            (D,) complex HV with unit-magnitude components
        """
        angles = self._harmonics * (2 * math.pi * x / self.period)  # (D,)
        return torch.exp(1j * angles)   # (D,) complex

    def encode_2d(self, x: float, y: float) -> torch.Tensor:
        """
        Encode 2D position (x, y) by binding x and y encodings.

        g_2d(x,y) = g(x) ⊙ g(y)
        """
        return self.encode(x) * self.encode(y)   # element-wise multiply

    def path_integrate(
        self,
        current_pos_hv: torch.Tensor,
        displacement: float,
    ) -> torch.Tensor:
        """
        Update position HV by integrating a displacement.

        new_pos = current_pos ⊙ g(Δx)   [Eq. 3 of paper]

        Args:
            current_pos_hv: (D,) current position phasor HV
            displacement: Δx in world coordinates

        Returns:
            (D,) updated position HV
        """
        velocity_hv = self.encode(displacement)
        return current_pos_hv * velocity_hv   # complex multiply = phase addition

    def decode(
        self,
        query_hv: torch.Tensor,
        x_range: Tuple[float, float],
        n_points: int = 100,
    ) -> float:
        """
        Decode position from a query HV by finding peak similarity.

        Searches over x_range and returns the position with maximum
        Re(<query, g(x)>) (the kernel inner product).

        Args:
            query_hv: (D,) complex position HV to decode
            x_range: (x_min, x_max) search range
            n_points: Resolution of grid search

        Returns:
            Estimated position x
        """
        xs = torch.linspace(x_range[0], x_range[1], n_points)
        sims = torch.tensor([
            float((query_hv.conj() * self.encode(x.item())).real.mean())
            for x in xs
        ])
        best_idx = int(sims.argmax())
        return float(xs[best_idx])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Grid Cell Network — Residue Number System over K modules
# ═══════════════════════════════════════════════════════════════════════════════

class GridCellNetwork:
    """
    Multi-module grid cell network using Residue Number System (RNS).

    K grid modules with co-prime periods {λ_1, ..., λ_K} represent
    position x as its remainders modulo each λ_k (Chinese Remainder Theorem).
    The total representable range is M = λ_1 × ... × λ_K — superlinear in D.

    From §3.2 of paper:
        Coding range M scales as O(D^{αK}) where αK = K/(K-1) > 1
        More modules → faster-growing range for the same D

    Position encoding:
        G(x) = g_1(x) ⊕ g_2(x) ⊕ ... ⊕ g_K(x)   [stack or bundle]

    For the resonator network, the modules are stored separately;
    for the place cell model, they're bundled into a single HV.

    Path integration:
        G(x + Δx) = [g_1(x)⊙g_1(Δx), ..., g_K(x)⊙g_K(Δx)]

    Args:
        dim: HV dimension per module
        periods: List of K spatial periods (co-prime for maximum range)
        seed: Base random seed
    """

    def __init__(
        self,
        dim: int,
        periods: Optional[List[float]] = None,
        seed: int = 0,
    ):
        self.dim = dim

        # Default: 3 modules with co-prime integer periods
        # Range = 5 × 7 × 11 = 385 positions
        self.periods = periods or [5.0, 7.0, 11.0]
        self.n_modules = len(self.periods)

        # Create one GridCellModule per period
        self.modules = [
            GridCellModule(dim, period=p, seed=seed + i)
            for i, p in enumerate(self.periods)
        ]

        # Theoretical coding range (CRT)
        self.coding_range = 1.0
        for p in self.periods:
            self.coding_range *= p

    def encode(self, x: float, return_stack: bool = False) -> torch.Tensor:
        """
        Encode position x using all K modules.

        Args:
            x: Position value
            return_stack: If True, return (K, D) module outputs separately.
                         If False, return (D,) bundled place-cell-like HV.

        Returns:
            (K, D) or (D,) complex HV
        """
        hvs = [m.encode(x) for m in self.modules]  # list of (D,) complex

        if return_stack:
            return torch.stack(hvs)   # (K, D)

        # Bundle: superpose all module outputs (place cell = sum of grid modules)
        bundled = sum(hvs) / self.n_modules
        return bundled   # (D,) complex

    def encode_2d(
        self,
        x: float,
        y: float,
        return_stack: bool = False,
    ) -> torch.Tensor:
        """Encode 2D position (x, y)."""
        hvs = [m.encode_2d(x, y) for m in self.modules]

        if return_stack:
            return torch.stack(hvs)

        return sum(hvs) / self.n_modules

    def path_integrate(
        self,
        current_hvs: torch.Tensor,
        dx: float,
        dy: float = 0.0,
    ) -> torch.Tensor:
        """
        Path integration: update all K module HVs by displacement (dx, dy).

        Args:
            current_hvs: (K, D) current module HVs
            dx: x displacement
            dy: y displacement (0 for 1D)

        Returns:
            (K, D) updated module HVs
        """
        updated = []
        for k, module in enumerate(self.modules):
            if dy != 0.0:
                disp_hv = module.encode_2d(dx, dy)
            else:
                disp_hv = module.encode(dx)
            updated.append(current_hvs[k] * disp_hv)
        return torch.stack(updated)

    def decode_position(
        self,
        hvs: torch.Tensor,
        x_range: Tuple[float, float],
        n_points: int = 200,
    ) -> float:
        """
        Decode 1D position from module HVs.

        Uses the inner product with all modules simultaneously
        (equivalent to a resonator-based decoding step).

        Args:
            hvs: (K, D) module HVs
            x_range: Search range
            n_points: Grid resolution

        Returns:
            Estimated position x
        """
        xs = torch.linspace(x_range[0], x_range[1], n_points)

        # For each candidate position, compute combined similarity
        scores = torch.zeros(n_points)
        for i, x in enumerate(xs):
            total_sim = 0.0
            for k, module in enumerate(self.modules):
                candidate_hv = module.encode(x.item())
                sim = float((hvs[k].conj() * candidate_hv).real.mean())
                total_sim += sim
            scores[i] = total_sim / self.n_modules

        best_idx = int(scores.argmax())
        return float(xs[best_idx])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Place Cell Encoder — superposition of grid modules
# ═══════════════════════════════════════════════════════════════════════════════

class PlaceCellEncoder:
    """
    Place cells as superposition of grid cell module responses.

    Each place cell fires when the agent is near a specific location.
    In the normative model, place cell activity p(x) is the superposition
    of grid module outputs at that location:

        p(x) = G(x) = Σ_k g_k(x)   [sum of module phasors]

    The real part of p(x) acts like a firing rate: it peaks at x and decays
    (with the Gaussian-like kernel from the fractional binding property).

    For associating sensory observations with positions (from §4.2):
        Associate memory: M += s(x) ⊗ p(x)
        Query:           s(x_query) ≈ unbind(M, p(x_query))

    Args:
        grid_network: GridCellNetwork providing the multi-module encoding
    """

    def __init__(self, grid_network: GridCellNetwork):
        self.grid = grid_network
        self.dim = grid_network.dim

        # Heteroassociative memory: stores (sensor → position) associations
        # M += sensor_hv ⊙ place_hv   [XOR bind in BSC, multiply in FHRR]
        self._assoc_memory = torch.zeros(self.dim, dtype=torch.complex64)
        self._n_associations = 0

    def place_hv(self, x: float, y: float = 0.0) -> torch.Tensor:
        """Return the place cell HV for position (x, y)."""
        if y != 0.0:
            return self.grid.encode_2d(x, y)
        return self.grid.encode(x)

    def firing_rate(
        self,
        agent_pos: float,
        x_range: Tuple[float, float],
        n_points: int = 100,
    ) -> torch.Tensor:
        """
        Compute 1D place field: similarity of place HV to query HV at each position.

        Args:
            agent_pos: Agent's current position
            x_range: Position range to compute over
            n_points: Grid resolution

        Returns:
            (n_points,) firing rate profile ∈ [-1, 1]
        """
        query = self.place_hv(agent_pos)
        xs = torch.linspace(x_range[0], x_range[1], n_points)
        rates = torch.tensor([
            float((self.place_hv(x.item()).conj() * query).real.mean())
            for x in xs
        ])
        return rates

    def associate(self, sensor_hv: torch.Tensor, position: float):
        """
        Associate a sensor HV with a position (heteroassociative memory).

        M += sensor_hv* ⊙ place_hv(pos)

        Args:
            sensor_hv: (D,) real binary sensor HV (will be treated as complex)
            position: World coordinate
        """
        p = self.place_hv(position)          # (D,) complex
        s = sensor_hv.float().to(torch.complex64)
        self._assoc_memory += s.conj() * p   # outer association
        self._n_associations += 1

    def recall_sensor(self, position: float) -> torch.Tensor:
        """
        Recall sensor HV at a position via heteroassociation.

        s_recall ≈ unbind(M, place_hv(pos)) = M ⊙ place_hv*(pos)

        Returns:
            (D,) recalled real sensor HV (sign of complex part)
        """
        p = self.place_hv(position)
        recalled = self._assoc_memory * p.conj()   # (D,) complex
        return (recalled.real > 0).float()          # binarise

    def localise(
        self,
        sensor_hv: torch.Tensor,
        search_range: Tuple[float, float],
        resolution: int = 100,
    ) -> Tuple[float, float]:
        """
        Localise an agent from a sensor observation.

        Queries the associative memory at a grid of candidate positions and
        returns the best-matching position and its similarity score.

        This is the HDC equivalent of particle filter localisation — no
        particles needed, just one cosine similarity scan.

        Args:
            sensor_hv:    (D,) current sensor observation
            search_range: (min, max) position range to search
            resolution:   Number of candidate positions

        Returns:
            (best_position, similarity_score)
        """
        s = sensor_hv.float().to(torch.complex64)
        xs = torch.linspace(search_range[0], search_range[1], resolution)
        best_pos, best_sim = float(xs[0]), -float("inf")

        for x in xs:
            p       = self.place_hv(float(x))
            recalled = self._assoc_memory * p.conj()
            # Similarity between recalled and actual sensor
            sim = float((recalled.real * s.real).mean())
            if sim > best_sim:
                best_sim = sim
                best_pos = float(x)

        return best_pos, best_sim


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Dead-Reckoning Navigator — path integration agent
# ═══════════════════════════════════════════════════════════════════════════════

class DeadReckoningNavigator:
    """
    Navigation agent using grid cell path integration.

    Maintains a position estimate in HV space and updates it from
    velocity/odometry inputs — no GPS needed.

    On each step:
        1. Update grid module HVs: G(t+1) = path_integrate(G(t), Δx, Δy)
        2. Decode current position estimate from G(t+1)
        3. If sensory landmark available: correct drift via association

    Drift accumulates over time (open-loop), but sensory corrections
    anchor the estimate periodically (as in real hippocampal navigation).

    Args:
        grid_network: GridCellNetwork for position encoding
        x_range, y_range: World bounds for decoding
        sensor_place: Optional PlaceCellEncoder for landmark correction
    """

    def __init__(
        self,
        grid_network: GridCellNetwork,
        x_range: Tuple[float, float] = (0.0, 10.0),
        sensor_place: Optional[PlaceCellEncoder] = None,
    ):
        self.grid = grid_network
        self.x_range = x_range
        self.place = sensor_place

        # Current position estimate in HV space (K, D) complex
        self._pos_hvs: Optional[torch.Tensor] = None
        self._estimated_pos: Optional[float] = None

    def reset(self, x_init: float = 0.0):
        """Initialise position estimate."""
        hvs_list = [m.encode(x_init) for m in self.grid.modules]
        self._pos_hvs = torch.stack(hvs_list)   # (K, D) complex
        self._estimated_pos = x_init

    def step(
        self,
        dx: float,
        dy: float = 0.0,
        sensor_hv: Optional[torch.Tensor] = None,
    ) -> Tuple[float, torch.Tensor]:
        """
        Move by (dx, dy) and optionally correct with sensor.

        Args:
            dx: x displacement
            dy: y displacement
            sensor_hv: Optional (D,) sensor observation for landmark correction

        Returns:
            (estimated_x, pos_hvs)
        """
        if self._pos_hvs is None:
            self.reset()

        # Path integration: G(t+1) = G(t) ⊙ g(Δx, Δy)
        self._pos_hvs = self.grid.path_integrate(self._pos_hvs, dx, dy)

        # Decode position
        self._estimated_pos = self.grid.decode_position(
            self._pos_hvs, self.x_range
        )

        # Optional sensor correction (heteroassociation recall)
        if sensor_hv is not None and self.place is not None:
            # Find the position in memory most similar to this sensor reading
            # (simplified: just use the sensor to anchor the current estimate)
            self.place.associate(sensor_hv, self._estimated_pos)

        return self._estimated_pos, self._pos_hvs

    @property
    def estimated_position(self) -> Optional[float]:
        return self._estimated_pos


class LoopClosureSLAM:
    """
    HDC-native Simultaneous Localisation and Mapping (SLAM) via loop closure.

    Reference:
        Milford & Wyeth (2012) "SeqSLAM: Visual Route-Based Navigation for
        Sunny Summer Days and Stormy Winter Nights" ICRA 2012.

        Neubert, Schubert, Protzel (2019) "Hyperdimensional Computing as a
        Framework for Systematic Integration of Re-Representation Mechanisms"
        Frontiers Neurorobotics 12:81.

    Traditional SLAM (ORB-SLAM, RTAB-Map) requires a neural network or heavy
    feature extractor for loop closure detection.  HDC-SLAM uses:
      - Grid cell modules for path integration (dead reckoning)
      - Hamming similarity of scene HVs for loop closure detection
      - Simple prototype bundling for map building

    When a loop is closed (current scene ≈ stored scene):
        1. Correct accumulated drift: pos_est ← α × pos_est + (1-α) × stored_pos
        2. Update the map landmark for this location

    This gives a full SLAM solution with:
        - O(D) memory per landmark (vs O(N²) for dense maps)
        - O(D) loop closure check (single Hamming distance)
        - No neural network, no feature extractor

    Args:
        grid_network:     GridCellNetwork for odometry
        dim:              HV dimension for scene/position encoding
        closure_threshold: Minimum Hamming similarity to declare loop closure
        max_landmarks:    Maximum stored map landmarks
        correction_rate:  α for blending estimated and stored positions
    """

    def __init__(
        self,
        grid_network:      'GridCellNetwork',
        dim:               int,
        closure_threshold: float = 0.70,
        max_landmarks:     int   = 256,
        correction_rate:   float = 0.3,
        device:            str   = "cpu",
    ):
        self.grid              = grid_network
        self.dim               = dim
        self.closure_threshold = closure_threshold
        self.max_landmarks     = max_landmarks
        self.correction_rate   = correction_rate
        self.device            = device

        # Map: list of (pos_estimate, scene_hv) landmarks
        self._map_positions: List[float] = []
        self._map_scenes:    List[torch.Tensor] = []
        self._n_closures     = 0
        self._total_steps    = 0

        # Current navigator state
        self._pos_estimate:  Optional[float] = None
        self._pos_hvs:       Optional[torch.Tensor] = None

    def reset(self, x_init: float = 0.0, scene_hv: Optional[torch.Tensor] = None):
        """Reset navigator to initial position. Optionally store first landmark."""
        hvs_list = [m.encode(x_init) for m in self.grid.modules]
        self._pos_hvs    = torch.stack(hvs_list)
        self._pos_estimate = x_init
        if scene_hv is not None:
            self._store_landmark(x_init, scene_hv)

    def _store_landmark(self, pos: float, scene_hv: torch.Tensor):
        """Add a new landmark to the map (with capacity enforcement)."""
        if len(self._map_positions) >= self.max_landmarks:
            # Evict oldest landmark
            self._map_positions.pop(0)
            self._map_scenes.pop(0)
        self._map_positions.append(float(pos))
        self._map_scenes.append(scene_hv.float().to(self.device))

    def _best_match(
        self,
        query_scene: torch.Tensor,
    ) -> Tuple[Optional[int], float]:
        """Find the most similar stored scene. Returns (idx, similarity)."""
        if not self._map_scenes:
            return None, 0.0
        q = query_scene.float().to(self.device)
        scenes = torch.stack(self._map_scenes)   # (N, D)
        sims   = 1.0 - (q.unsqueeze(0) != scenes).float().mean(dim=1)  # (N,)
        best_sim, best_idx = float(sims.max().item()), int(sims.argmax().item())
        return best_idx, best_sim

    def slam_step(
        self,
        dx:       float,
        dy:       float = 0.0,
        scene_hv: Optional[torch.Tensor] = None,
        x_range:  Tuple[float, float] = (0.0, 10.0),
    ) -> Dict[str, Any]:
        """
        One SLAM step: dead reckon + check loop closure + update map.

        Args:
            dx, dy:    Odometry displacement (from encoders, IMU, optical flow)
            scene_hv:  (D,) current scene observation HV (from sensor encoder)
            x_range:   World bounds for position decoding

        Returns:
            Dict with:
              pos_estimate:    current position estimate (float)
              loop_closed:     True if a loop closure was detected
              closure_sim:     Hamming similarity to matched scene (0 if none)
              n_landmarks:     current map size
        """
        if self._pos_hvs is None:
            self.reset()

        self._total_steps += 1

        # 1. Path integration: G(t+1) = G(t) ⊙ g(Δx)
        self._pos_hvs    = self.grid.path_integrate(self._pos_hvs, dx, dy)
        self._pos_estimate = self.grid.decode_position(self._pos_hvs, x_range)

        loop_closed  = False
        closure_sim  = 0.0

        # 2. Loop closure detection
        if scene_hv is not None:
            best_idx, best_sim = self._best_match(scene_hv)

            if best_sim >= self.closure_threshold and best_idx is not None:
                # Loop closure! Correct accumulated drift
                stored_pos = self._map_positions[best_idx]
                corrected  = (
                    (1.0 - self.correction_rate) * self._pos_estimate
                    + self.correction_rate * stored_pos
                )
                self._pos_estimate = corrected
                loop_closed = True
                closure_sim = best_sim
                self._n_closures += 1

                # Update stored scene with new observation (online map update)
                self._map_scenes[best_idx] = (
                    0.9 * self._map_scenes[best_idx] + 0.1 * scene_hv.float().to(self.device)
                )
            else:
                # Novel location: store as new landmark
                self._store_landmark(self._pos_estimate, scene_hv)

        return {
            "pos_estimate":  self._pos_estimate,
            "loop_closed":   loop_closed,
            "closure_sim":   closure_sim,
            "n_landmarks":   len(self._map_positions),
            "n_closures":    self._n_closures,
        }

    def closure_rate(self) -> float:
        """Fraction of steps that triggered loop closure."""
        return self._n_closures / max(self._total_steps, 1)

    def map_summary(self) -> Dict:
        if not self._map_positions:
            return {"n_landmarks": 0}
        return {
            "n_landmarks":  len(self._map_positions),
            "pos_min":      min(self._map_positions),
            "pos_max":      max(self._map_positions),
            "n_closures":   self._n_closures,
            "closure_rate": self.closure_rate(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_grid_cell_module():
    print("=" * 60)
    print("Testing GridCellModule (Kymn et al. 2025, §2)")
    print("=" * 60)

    D, period = 5000, 7.0
    module = GridCellModule(dim=D, period=period, seed=42)

    # Path integration: g(x + Δx) = g(x) ⊙ g(Δx)
    x0, dx = 2.5, 1.3
    gx    = module.encode(x0)
    gdx   = module.encode(dx)
    gsum  = module.encode(x0 + dx)
    gcomp = gx * gdx   # complex multiply = phase addition

    # These should be equal (up to floating-point noise)
    sim = float((gsum.conj() * gcomp).real.mean())
    print(f"  Path integration: g(x+Δx) = g(x)⊙g(Δx): sim = {sim:.6f}  (want ≈ 1.0)")
    assert sim > 0.999, f"Path integration broken: sim={sim}"

    # Periodicity: g(x) ≈ g(x + λ)
    gx_shifted = module.encode(x0 + period)
    sim_period = float((gx.conj() * gx_shifted).real.mean())
    print(f"  Periodicity g(x) ≈ g(x+λ): sim = {sim_period:.6f}  (want ≈ 1.0)")
    assert sim_period > 0.999

    # Different positions should be dissimilar
    gy = module.encode(x0 + period / 2)  # half-period away → orthogonal
    sim_orth = float((gx.conj() * gy).real.mean())
    print(f"  Half-period offset sim: {sim_orth:.4f}  (want near 0)")
    assert abs(sim_orth) < 0.3

    print("  ✅ GridCellModule OK")


def test_grid_cell_network():
    print("=" * 60)
    print("Testing GridCellNetwork (RNS, K=3 modules)")
    print("=" * 60)

    torch.manual_seed(42)
    D = 3000
    network = GridCellNetwork(dim=D, periods=[5.0, 7.0, 11.0], seed=0)

    print(f"  Coding range: {network.coding_range:.0f} positions (5×7×11={5*7*11})")
    assert network.coding_range == 385.0

    # Encode a position
    x = 3.7
    hvs = network.encode(x, return_stack=True)   # (K, D) complex
    assert hvs.shape == (3, D)

    # Path integration: encode → integrate → decode
    hvs_init = network.encode(0.0, return_stack=True)
    hvs_moved = network.path_integrate(hvs_init, dx=3.7)
    x_decoded = network.decode_position(hvs_moved, (0.0, 11.0), n_points=200)
    print(f"  Path integrate to x=3.7: decoded={x_decoded:.3f}  (want ≈ 3.7)")
    assert abs(x_decoded - 3.7) < 0.5, f"Decoded {x_decoded:.3f} ≠ 3.7"

    print("  ✅ GridCellNetwork OK")


def test_place_cell_encoder():
    print("=" * 60)
    print("Testing PlaceCellEncoder (§4.2 heteroassociation)")
    print("=" * 60)

    torch.manual_seed(7)
    D = 3000
    net = GridCellNetwork(dim=D, periods=[5.0, 7.0], seed=1)
    place = PlaceCellEncoder(net)

    # Place field: firing rate should peak at agent position
    rates = place.firing_rate(agent_pos=3.0, x_range=(0.0, 10.0), n_points=50)
    peak_x = float(torch.linspace(0, 10, 50)[rates.argmax()])
    print(f"  Place field peak at x≈3: decoded={peak_x:.2f}")
    # The peak should be near 3.0 (within half-period of nearest module)

    # Heteroassociation: associate sensor → position, then recall
    sensor = (torch.rand(D) < 0.5).float()
    place.associate(sensor, position=4.5)
    recalled = place.recall_sensor(position=4.5)
    sim = float(hv_batch_sim(sensor, recalled.unsqueeze(0))[0])
    print(f"  Sensor recall at position 4.5: sim = {sim:.4f}  (want high)")
    assert sim > 0.5

    print("  ✅ PlaceCellEncoder OK")


def test_dead_reckoning():
    print("=" * 60)
    print("Testing DeadReckoningNavigator (path integration)")
    print("=" * 60)

    torch.manual_seed(99)
    D = 4000
    net = GridCellNetwork(dim=D, periods=[5.0, 7.0, 11.0], seed=0)
    nav = DeadReckoningNavigator(net, x_range=(0.0, 15.0))
    nav.reset(x_init=0.0)

    # Integrate a sequence of small steps
    steps = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]   # total displacement = 3.0
    pos = 0.0
    for dx in steps:
        estimated, _ = nav.step(dx)
        pos += dx

    print(f"  True pos after 6×0.5: {pos:.1f}")
    print(f"  Estimated pos: {estimated:.3f}")
    print(f"  Error: {abs(estimated - pos):.3f}")
    assert abs(estimated - pos) < 1.0, f"Error too large: {abs(estimated-pos):.3f}"

    print("  ✅ DeadReckoningNavigator OK")


if __name__ == "__main__":
    test_grid_cell_module()
    print()
    test_grid_cell_network()
    print()
    test_place_cell_encoder()
    print()
    test_dead_reckoning()
    print()
    print("=== All grid cell tests passed ===")
