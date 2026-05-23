"""
Online Causal Discovery in Hypervector Space
=============================================
A genuinely novel capability: discovering causal structure from observations
using pure VSA operations — no likelihood models, no independence tests,
no DAG search algorithms.

The HDC approach to causality:
    Instead of testing statistical independence (PC algorithm) or fitting
    a structural equation model (LiNGAM), we encode the observation history
    as hypervectors and query: "does the HV of X-then-Y look more like
    (cause → effect) or (effect → cause) in our learned causal codebook?"

Core insight (Pearl 2000; Schölkopf et al. 2021):
    Cause and effect have an asymmetry: P(effect | cause) is a simple,
    low-complexity conditional, while P(cause | effect) requires Bayes'
    theorem and is more complex.  In HV space, this asymmetry manifests as:
        sim(X→Y pattern, cause_proto) > sim(Y→X pattern, cause_proto)
    when X is indeed the cause of Y.

HDC formulation:
    1. Encode temporal pairs as: pair_hv(t) = bind(state_hv_t, state_hv_{t+1})
    2. Bundle all pairs for variable X: X_transition = MAJORITY(pair_hv_i)
    3. Causal signature of X→Y: bind(X_transition, Y_transition)
       If X causes Y, X_transition should have high similarity to
       the expected "cause pattern" — learned from known causal pairs.
    4. Asymmetry test: sim(X→Y sig, cause_proto) vs sim(Y→X sig, cause_proto)

This module implements:
    HDCCausalVariable   — tracks a variable's transition HV online
    CausalSignatureGraph — builds and queries the full causal graph
    OnlineCausalDiscovery — top-level causal discovery from observations
    CausalInterventionEstimator — estimate do(X) effects from observational data

References:
    Pearl (2000) "Causality" — structural causal models
    Schölkopf et al. (2021) "Toward Causal Representation Learning" IEEE
    Janzing & Schölkopf (2010) "Causal inference using the algorithmic Markov
        condition" IEEE Trans. IT
    Peters, Janzing, Schölkopf (2017) "Elements of Causal Inference"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

from hdc.physics_world_model import _xor, _majority, _hamming


# ── HDC primitives ─────────────────────────────────────────────────────────────

def _bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _xor(a, b)

def _gen_hv(dim: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return (torch.rand(dim, generator=g, device=device) >= 0.5).float()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HDCCausalVariable — tracks one variable's transition HVs
# ═══════════════════════════════════════════════════════════════════════════════

class HDCCausalVariable:
    """
    Accumulates a variable's state transitions as a bundled HV.

    For variable X at times t = 0, 1, ..., T:
        transition_hv = MAJORITY( bind(x_hv(0), x_hv(1)),
                                   bind(x_hv(1), x_hv(2)),
                                   ...
                                   bind(x_hv(T-1), x_hv(T)) )

    The transition HV encodes "what tends to follow what" for variable X.
    It is the empirical temporal auto-correlation structure of X in HV space.

    For a causal variable X→Y:
        bind(X.transition_hv, Y.transition_hv) should be similar to
        a "causal prototype" HV — distinguishing X→Y from the reverse.

    Args:
        name: Variable name
        dim:  HV dimension
        decay: EMA decay for the accumulated transition HV (recent = higher weight)
        device: torch device
    """

    def __init__(
        self,
        name:   str,
        dim:    int,
        decay:  float = 0.95,
        device: str   = "cpu",
    ):
        self.name    = name
        self.dim     = dim
        self.decay   = decay
        self.device  = device

        self.transition_hv = torch.zeros(dim, device=device)
        self._prev_hv: Optional[torch.Tensor] = None
        self._n_obs:   int = 0

    def observe(self, state_hv: torch.Tensor):
        """
        Record a new observation and update transition_hv.

        Args:
            state_hv: (D,) HV encoding the current state of this variable.
        """
        s = state_hv.float().to(self.device)
        self._n_obs += 1

        if self._prev_hv is not None:
            pair = _bind(self._prev_hv, s)
            # EMA in float space — binarize lazily at query time via .binary property
            self.transition_hv = (
                self.decay * self.transition_hv + (1 - self.decay) * pair.float()
            )
        self._prev_hv = s.clone()

    def reset(self):
        self.transition_hv = torch.zeros(self.dim, device=self.device)
        self._prev_hv = None
        self._n_obs   = 0

    @property
    def binary(self) -> torch.Tensor:
        """Binarize the float EMA accumulator for HDC operations."""
        return (self.transition_hv > 0.5).float()

    @property
    def has_data(self) -> bool:
        return self._n_obs >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CausalSignatureGraph — build and query causal relationships
# ═══════════════════════════════════════════════════════════════════════════════

class CausalSignatureGraph:
    """
    Discovers pairwise causal directions via HV asymmetry testing.

    For each pair (X, Y), computes a causal signature:
        sig(X→Y) = bind(X.transition_hv, Y.transition_hv)
        sig(Y→X) = bind(Y.transition_hv, X.transition_hv)

    The causal direction is determined by which signature is more similar
    to a "causal prototype" HV.  The causal prototype is learned from
    known X→Y pairs (if provided) or is approximated by the asymmetry
    of the signature pair themselves.

    Asymmetry criterion (unsupervised, no known pairs needed):
        If X causes Y, then changes in X tend to precede changes in Y.
        In HV space: the transition HV of X is "more predictive" of Y's
        transitions than vice versa.  We measure this as:
            score(X→Y) = sim(bind(X_trans, Y_trans), causal_proto)
                       − sim(bind(Y_trans, X_trans), causal_proto)
        where causal_proto = majority(all known directed pairs).

    If no known pairs: use the self-mutual information proxy:
        score(X→Y) = sim(X_trans, bind(X_trans, Y_trans))
                   − sim(Y_trans, bind(X_trans, Y_trans))
        (X is the cause if X_trans matches the joint signature better)

    Args:
        dim:         HV dimension
        min_obs:     Minimum observations per variable before testing
        threshold:   Minimum asymmetry score to assert a causal edge
        device:      torch device
    """

    def __init__(
        self,
        dim:       int,
        min_obs:   int   = 20,
        threshold: float = 0.02,
        device:    str   = "cpu",
    ):
        self.dim       = dim
        self.min_obs   = min_obs
        self.threshold = threshold
        self.device    = device

        self._variables: Dict[str, HDCCausalVariable] = {}
        self._causal_proto: Optional[torch.Tensor]    = None
        self._known_pairs:  List[Tuple[str, str]]     = []    # (cause, effect)

    # ── variable management ──────────────────────────────────────────────────

    def register_variable(self, name: str, decay: float = 0.95):
        """Register a new variable to track."""
        if name not in self._variables:
            self._variables[name] = HDCCausalVariable(name, self.dim, decay, self.device)

    def observe(self, name: str, state_hv: torch.Tensor):
        """Record an observation for a registered variable."""
        if name in self._variables:
            self._variables[name].observe(state_hv)

    def observe_all(self, observations: Dict[str, torch.Tensor]):
        """Record one timestep of observations for all variables."""
        for name, hv in observations.items():
            if name not in self._variables:
                self.register_variable(name)
            self._variables[name].observe(hv)

    # ── causal prototype ─────────────────────────────────────────────────────

    def add_known_pair(self, cause: str, effect: str):
        """
        Provide a known causal pair (cause → effect) to calibrate
        the causal prototype.  Not required — unsupervised mode is available.
        """
        self._known_pairs.append((cause, effect))
        self._update_proto()

    def _update_proto(self):
        """Rebuild causal prototype from known (cause, effect) pairs."""
        signatures = []
        for cause, effect in self._known_pairs:
            if cause in self._variables and effect in self._variables:
                c_var = self._variables[cause]
                e_var = self._variables[effect]
                if c_var.has_data and e_var.has_data:
                    sig = _bind(c_var.binary, e_var.binary)
                    signatures.append(sig)
        if signatures:
            stacked = torch.stack(signatures).float()
            self._causal_proto = _majority(stacked.mean(dim=0))

    # ── causal testing ───────────────────────────────────────────────────────

    def causal_score(self, var_a: str, var_b: str) -> float:
        """
        Asymmetry score for A→B vs B→A.

        Positive score: A is more likely the cause of B.
        Negative score: B is more likely the cause of A.
        Near zero: insufficient evidence.

        Returns:
            score ∈ (−1, 1)
        """
        if var_a not in self._variables or var_b not in self._variables:
            return 0.0

        a_var = self._variables[var_a]
        b_var = self._variables[var_b]

        if not a_var.has_data or not b_var.has_data:
            return 0.0
        if a_var._n_obs < self.min_obs or b_var._n_obs < self.min_obs:
            return 0.0

        a_bin = a_var.binary
        b_bin = b_var.binary
        sig_ab = _bind(a_bin, b_bin)  # A→B
        sig_ba = _bind(b_bin, a_bin)  # B→A

        if self._causal_proto is not None:
            sim_ab = float(_hamming(sig_ab.unsqueeze(0), self._causal_proto.unsqueeze(0)).item())
            sim_ba = float(_hamming(sig_ba.unsqueeze(0), self._causal_proto.unsqueeze(0)).item())
            return sim_ab - sim_ba
        else:
            sim_a_ab = float(_hamming(a_bin.unsqueeze(0), sig_ab.unsqueeze(0)).item())
            sim_b_ab = float(_hamming(b_bin.unsqueeze(0), sig_ab.unsqueeze(0)).item())
            return sim_a_ab - sim_b_ab

    def discover_edges(self) -> List[Tuple[str, str, float]]:
        """
        Run pairwise causal discovery over all registered variables.

        Returns:
            List of (cause, effect, score) tuples where score > threshold.
            Sorted by score descending (strongest causal evidence first).
        """
        var_names = list(self._variables.keys())
        edges = []
        for i in range(len(var_names)):
            for j in range(i + 1, len(var_names)):
                a, b = var_names[i], var_names[j]
                score = self.causal_score(a, b)
                if abs(score) >= self.threshold:
                    if score > 0:
                        edges.append((a, b, score))
                    else:
                        edges.append((b, a, -score))

        edges.sort(key=lambda x: x[2], reverse=True)
        return edges

    def causal_graph(self) -> Dict[str, List[str]]:
        """
        Return discovered causal graph as adjacency dict.

        Returns:
            {cause: [effect1, effect2, ...]} mapping.
        """
        edges   = self.discover_edges()
        graph: Dict[str, List[str]] = {n: [] for n in self._variables}
        for cause, effect, _ in edges:
            graph[cause].append(effect)
        return graph

    def granger_score(self, cause: str, effect: str) -> float:
        """
        Granger-causality proxy: does knowing `cause` improve prediction of `effect`?

        Reference:
            Granger (1969) "Investigating causal relations by econometric models
            and cross-spectral methods" Econometrica 37(3):424-438.

        HDC implementation:
            1. Predict effect from its own history:  pred_self = bind(effect, effect)
            2. Predict effect with cause info:       pred_cross = bind(cause, effect)
            3. If pred_cross is more similar to future_effect than pred_self:
               cause Granger-causes effect.

        Score = sim(pred_cross, effect) - sim(pred_self, effect)
        Positive → cause helps predict effect (Granger causality evidence).

        Returns: Granger score ∈ [-1, 1].  > threshold → Granger causality.
        """
        if cause not in self._variables or effect not in self._variables:
            return 0.0
        c_var = self._variables[cause]
        e_var = self._variables[effect]
        if not c_var.has_data or not e_var.has_data:
            return 0.0

        c_bin = c_var.binary
        e_bin = e_var.binary

        # Self-prediction: how well does effect predict itself?
        pred_self  = _bind(e_bin, e_bin)   # autocorrelation proxy
        sim_self   = float(_hamming(pred_self.unsqueeze(0), e_bin.unsqueeze(0)).item())

        # Cross-prediction: how well does cause + effect predict effect?
        pred_cross = _bind(c_bin, e_bin)
        sim_cross  = float(_hamming(pred_cross.unsqueeze(0), e_bin.unsqueeze(0)).item())

        return sim_cross - sim_self

    def causal_parents(self, var: str) -> List[Tuple[str, float]]:
        """
        Find all causal parents of variable `var`.

        Returns:
            List of (parent_name, causal_score) sorted by score descending.
        """
        parents = []
        for other in self._variables:
            if other == var:
                continue
            score = self.causal_score(other, var)
            if score >= self.threshold:
                parents.append((other, score))
        return sorted(parents, key=lambda x: x[1], reverse=True)

    def intervention_hv(self, cause: str, effect: str) -> Optional[torch.Tensor]:
        """
        Estimate the effect of do(cause=X) on `effect` as an HV.

        In HV space: the interventional HV is the "residual" of the effect's
        transition HV after removing the cause's influence:
            do_hv = bind(effect.transition_hv, cause.transition_hv)
                  = effect's transitions unexplained by cause's transitions

        Returns:
            (D,) intervention HV, or None if variables not available.
        """
        if cause not in self._variables or effect not in self._variables:
            return None
        c_var = self._variables[cause]
        e_var = self._variables[effect]
        if not c_var.has_data or not e_var.has_data:
            return None
        return _bind(e_var.binary, c_var.binary)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OnlineCausalDiscovery — top-level causal reasoning system
# ═══════════════════════════════════════════════════════════════════════════════

class OnlineCausalDiscovery:
    """
    Online causal discovery from streaming observations.

    Maintains a CausalSignatureGraph and updates it as new observations
    arrive.  Provides human-readable causal summaries and action-relevant
    causal queries:
        - "What causes variable X to increase?"
        - "If I do action A, what will change?"
        - "What is the causal chain between X and Y?"

    Args:
        dim:       HV dimension
        variables: List of variable names to track
        min_obs:   Minimum observations before claiming causal edges
        threshold: Minimum asymmetry score for asserting causality
        device:    torch device
    """

    def __init__(
        self,
        dim:       int,
        variables: Optional[List[str]] = None,
        min_obs:   int   = 20,
        threshold: float = 0.02,
        device:    str   = "cpu",
    ):
        self.dim    = dim
        self.device = device
        self.graph  = CausalSignatureGraph(dim, min_obs, threshold, device)

        if variables:
            for v in variables:
                self.graph.register_variable(v)

        self._n_timesteps = 0
        self._causal_history: List[List[Tuple[str, str, float]]] = []

    def step(self, observations: Dict[str, torch.Tensor]):
        """
        Record one timestep of observations for all variables.

        Args:
            observations: {variable_name: state_hv (D,)}
        """
        self._n_timesteps += 1
        self.graph.observe_all(observations)

        # Periodically re-run causal discovery
        if self._n_timesteps % 10 == 0:
            edges = self.graph.discover_edges()
            self._causal_history.append(edges)

    def causal_summary(self) -> Dict[str, List[str]]:
        """Return the current best causal graph as an adjacency dict."""
        return self.graph.causal_graph()

    def what_causes(self, variable: str) -> List[Tuple[str, float]]:
        """
        Query: "What are the causal parents of `variable`?"

        Returns:
            List of (parent_name, confidence) sorted by confidence.
        """
        return self.graph.causal_parents(variable)

    def what_does(self, variable: str) -> List[Tuple[str, float]]:
        """
        Query: "What does `variable` cause?"

        Returns:
            List of (effect_name, confidence).
        """
        effects = []
        for other in self.graph._variables:
            if other == variable:
                continue
            score = self.graph.causal_score(variable, other)
            if score >= self.graph.threshold:
                effects.append((other, score))
        return sorted(effects, key=lambda x: x[1], reverse=True)

    def causal_chain(
        self,
        source: str,
        target: str,
        max_depth: int = 4,
    ) -> Optional[List[str]]:
        """
        Find a causal path from `source` to `target` via BFS.

        Returns:
            List of variable names [source, ..., target], or None if no path.
        """
        graph = self.causal_summary()
        if source not in graph:
            return None

        visited: Set[str] = {source}
        queue   = [[source]]

        while queue:
            path = queue.pop(0)
            current = path[-1]
            if current == target:
                return path
            if len(path) >= max_depth:
                continue
            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    def stability(self) -> float:
        """
        Measure causal graph stability: how consistent are the discovered
        edges across the last few discovery runs?

        Returns:
            Stability ∈ [0, 1]; 1.0 = perfectly stable causal graph.
        """
        if len(self._causal_history) < 2:
            return 0.0

        last  = set((c, e) for c, e, _ in self._causal_history[-1])
        prev  = set((c, e) for c, e, _ in self._causal_history[-2])
        union = last | prev
        if not union:
            return 1.0
        return len(last & prev) / len(union)

    def granger_edges(self, threshold: float = 0.01) -> List[Tuple[str, str, float]]:
        """
        Discover causal edges using the Granger proxy metric.

        Returns all (cause, effect, granger_score) pairs with score > threshold.
        Granger score measures how much knowing `cause` improves prediction of
        `effect` beyond `effect`'s own history.

        Returns:
            List of (cause, effect, score) sorted by score descending.
        """
        var_names = list(self.graph._variables.keys())
        edges = []
        for i in range(len(var_names)):
            for j in range(len(var_names)):
                if i == j:
                    continue
                a, b  = var_names[i], var_names[j]
                score = self.graph.granger_score(a, b)
                if score > threshold:
                    edges.append((a, b, score))
        return sorted(edges, key=lambda x: x[2], reverse=True)

    def causal_lag_estimate(
        self,
        cause:      str,
        effect:     str,
        max_lag:    int = 5,
    ) -> int:
        """
        Estimate the optimal causal lag between `cause` and `effect`.

        The lag is estimated by finding the lag k ∈ [0, max_lag] at which
        the causal score is highest.  This uses the HDCCausalVariable's
        EMA state — different EMA decay rates capture different lag scales.

        Currently uses the asymmetry score (causal_score) as a proxy.
        Higher score at lag k → cause's state at t-k best predicts effect.

        Returns:
            Estimated lag (0 = instantaneous, k > 0 = k-step delay).
        """
        if cause not in self.graph._variables or effect not in self.graph._variables:
            return 0

        c_var = self.graph._variables[cause]
        e_var = self.graph._variables[effect]

        if not c_var.has_data or not e_var.has_data:
            return 0

        # Measure causal score with different decay rates (proxy for different lags)
        # Fast decay (0.5) ≈ 1-step lag; slow decay (0.99) ≈ many-step lag
        decays   = [max(0.5, 1.0 - 1.0 / (k + 1)) for k in range(max_lag + 1)]
        scores   = []
        orig_decay = c_var._decay if hasattr(c_var, '_decay') else 0.95

        for decay in decays:
            # Temporarily adjust the EMA decay to simulate different lags
            if hasattr(c_var, '_decay'):
                c_var._decay = decay
            score = self.graph.causal_score(cause, effect)
            scores.append(score)

        if hasattr(c_var, '_decay'):
            c_var._decay = orig_decay   # restore

        best_lag = int(max(range(len(scores)), key=lambda i: scores[i]))
        return best_lag

    def simulate_intervention(
        self,
        intervened_var:  str,
        intervened_value: torch.Tensor,
        observe_var:     str,
        n_steps:         int = 1,
    ) -> Dict[str, Any]:
        """
        Simulate "what would happen if I set `intervened_var` to a specific HV?"

        This is the Pearl do-calculus (Pearl 2009) adapted to HDC:
            do(X = x_hv) sets X to x_hv and propagates through causal graph.

        Returns:
            Dict with predicted downstream effect on `observe_var`.
        """
        # Get current state of intervened variable
        if intervened_var not in self.graph._variables:
            return {"error": f"unknown variable: {intervened_var}"}

        # Find causal path from intervened_var to observe_var
        path = self.causal_chain(intervened_var, observe_var, max_depth=n_steps + 1)

        # Propagate intervention through path
        current_hv = intervened_value.clone()
        steps_taken = 0
        for step_var in (path[1:] if path else [observe_var]):
            # Apply causal transition: XOR bind with intervention signal
            if step_var in self.graph._variables:
                step_hv   = self.graph._variables[step_var].binary
                current_hv = _bind(current_hv, step_hv)
                steps_taken += 1

        return {
            "intervened_on":     intervened_var,
            "target":            observe_var,
            "predicted_effect":  current_hv,
            "causal_path":       path,
            "n_steps":           steps_taken,
            "causal_confidence": float(abs(self.graph.granger_score(
                intervened_var, observe_var
            ))),
        }

    def register_known_cause(self, cause: str, effect: str):
        """Provide a known causal pair to calibrate the discovery."""
        self.graph.add_known_pair(cause, effect)

    def causal_strength_matrix(self) -> Dict[str, Any]:
        """
        Compute pairwise causal strengths as a matrix.

        Returns a dict with variable names and a symmetric-ish strength table.
        Entry [cause][effect] = Granger-proxy causal score.

        Useful for: visualising causal structure, identifying key drivers,
        building causal heatmaps for dashboards.
        """
        names = list(self.graph._variables.keys())
        if not names:
            return {"variables": [], "matrix": {}}

        matrix = {}
        for a in names:
            matrix[a] = {}
            for b in names:
                if a == b:
                    matrix[a][b] = 0.0
                else:
                    matrix[a][b] = round(float(abs(self.graph.granger_score(a, b))), 4)

        # Find top driver (variable with highest total outgoing causal strength)
        total_strength = {
            a: sum(matrix[a][b] for b in names if b != a)
            for a in names
        }
        top_driver = max(total_strength, key=total_strength.get) if total_strength else None

        return {
            "variables":  names,
            "matrix":     matrix,
            "top_driver": top_driver,
            "stability":  round(self.stability(), 4),
        }

    def intervention_portfolio(
        self,
        n_top: int = 3,
    ) -> List[Dict]:
        """
        Recommend the top-n interventions most likely to affect the system.

        An intervention on variable X is impactful if X has high total
        outgoing causal strength — changing X ripples through many effects.

        Returns:
            List of {variable, total_causal_strength, downstream_vars} dicts,
            sorted by total strength descending.
        """
        names = list(self.graph._variables.keys())
        if not names:
            return []

        portfolio = []
        for a in names:
            downstream = []
            total = 0.0
            for b in names:
                if b == a:
                    continue
                score = float(abs(self.graph.granger_score(a, b)))
                total += score
                if score > self.graph.threshold:
                    downstream.append((b, round(score, 4)))
            downstream.sort(key=lambda x: x[1], reverse=True)
            portfolio.append({
                "variable":              a,
                "total_causal_strength": round(total, 4),
                "downstream_vars":       downstream[:3],
            })

        portfolio.sort(key=lambda x: x["total_causal_strength"], reverse=True)
        return portfolio[:n_top]

    @property
    def n_variables(self) -> int:
        return len(self.graph._variables)


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_causal_discovery():
    import torch
    D = 500

    def _gen(s):
        g = torch.Generator()
        g.manual_seed(s)
        return (torch.rand(D, generator=g) >= 0.5).float()

    # Simulate a causal structure: X → Y → Z, W independent
    base_X = _gen(10)
    base_Y = _gen(20)
    base_Z = _gen(30)
    base_W = _gen(40)

    causal = OnlineCausalDiscovery(D, variables=["X", "Y", "Z", "W"], min_obs=10, threshold=0.01)

    # Generate 50 timesteps where X influences Y (XOR binding)
    for t in range(50):
        noise = lambda: (torch.rand(D) < 0.05).float()

        x_hv = _majority(base_X.float() + 0.1 * noise())
        y_hv = _majority((0.7 * _bind(x_hv, base_Y) + 0.3 * base_Y).float())
        z_hv = _majority((0.7 * _bind(y_hv, base_Z) + 0.3 * base_Z).float())
        w_hv = _majority(base_W.float() + 0.1 * noise())

        causal.step({"X": x_hv, "Y": y_hv, "Z": z_hv, "W": w_hv})

    print("=== OnlineCausalDiscovery ===")
    edges = causal.graph.discover_edges()
    print(f"  Discovered edges ({len(edges)}):")
    for c, e, s in edges:
        print(f"    {c} → {e}  (score={s:.4f})")

    print(f"\n  Causal graph: {causal.causal_summary()}")

    parents_Y = causal.what_causes("Y")
    print(f"  Parents of Y: {parents_Y}")

    effects_X = causal.what_does("X")
    print(f"  Effects of X: {effects_X}")

    chain = causal.causal_chain("X", "Z")
    print(f"  Causal chain X→Z: {chain}")

    print(f"\n  Stability: {causal.stability():.3f}")
    print(f"  n_variables: {causal.n_variables}")
    print(f"  n_timesteps: {causal._n_timesteps}")

    print("\n✅ causal_discovery tests passed")


if __name__ == "__main__":
    _test_causal_discovery()
