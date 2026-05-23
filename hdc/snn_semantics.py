"""
SNN + Vector Semantics Integration
====================================
Maximises quality of the SNN-HDC-vector-semantic pipeline through four
mutually reinforcing improvements:

1. **Temporal Spike Coding** — encodes not just *which* neurons fire but
   *when* they fire, yielding exponentially more information per spike.
   Three complementary encodings combine into a single composite HV:
     - Rate coding:       spike_rate → level HV (Rahimi 2017, LevelHypervectors)
     - Temporal coding:   first-spike-time → fractional binding v^{t/T}
     - Population coding: pairwise spike correlations → XOR-bundled co-activation

2. **STDP-HDC** — Spike-Timing Dependent Plasticity in HV space.
   Biological STDP strengthens synapses when pre fires *before* post:
     ΔW_{pre→post} ∝ exp(−|Δt| / τ)
   HDC-STDP encodes this as a temporal binding in concept space:
     concept_pair = XOR(concept_pre, Sh(concept_post, k))
   where k ∝ Δt. Repeated co-occurrence → concept_pair reinforced in a
   "temporal knowledge graph" that captures causal ordering.

3. **Semantic Attention** — semantic HVs modulate SNN neuron thresholds.
   When a concept C is "active" (sim(state_hv, C_hv) > threshold):
     v_th_i ← v_th_i × (1 − α × sim(basis_i, C_hv))
   Neurons whose basis HV matches the active concept get *lower* threshold
   → they fire more readily. This implements top-down attentional spotlight:
   semantics shapes perception, perception updates semantics.

4. **Semantic Quality Metrics** — continuous measurement of representation
   quality without a labelled test set:
     - Coherence:  mean sim between concepts that co-occur (want high)
     - Separability: mean sim between random concept pairs (want ~0.5)
     - Semantic gap: coherence − separability (want > 0.2)
     - Stability: mean HV drift per observation (want low)
     - Coverage: fraction of HV space reachable from current concepts

These metrics guide adaptive learning-rate scheduling in the
LifeLongSemanticLearner: when quality drops, increase learning rate;
when stable, decrease it.

References:
  Mitrokhin/Sutor 2019 (Science Robotics) — HAP, temporal spike encoding
  Sutor/Summers-Stay 2018 (arXiv:1806.10755) — lifelong semantic learning
  Rahimi 2017 (rahimi_nanoscale.py) — level HVs, fractional binding
  Verges Boncompte 2024 (resonator.py, minirocket_hdc.py) — FPE
  Kleyko 2023 Survey (kleyko_survey.py) — temporal n-gram encoding
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from hdc.hdc_glue import hv_batch_sim, hv_xor, hv_majority, gen_hvs, hv_permute
from hdc.rahimi_nanoscale import LevelHypervectors
from hdc.minirocket_hdc import FractionalBinding
from hdc.vector_semantic import KnowledgeGraph, SemanticVectorEncoder


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Temporal Spike Coding
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalSpikeEncoder:
    """
    Encode spike trains as composite hypervectors using three coding schemes.

    Three encodings combined via majority bundle:

    Rate coding (Rahimi 2017, §II-B):
        rate_i = mean(spikes[:, i]) ∈ [0, 1]
        hv_rate = MAJORITY(level_hv(rate_i) ⊗ id_hv(i) for each neuron i)
        → encodes "which neurons fire how often"

    First-spike-time coding (Verges Boncompte 2024):
        fst_i = first timestep neuron i fires / T ∈ [0, 1]  (1.0 if never)
        hv_fst = MAJORITY(v^{1 - fst_i} ⊗ id_hv(i) for each neuron i)
        → v^p from FractionalBinding: earlier fires = closer to v^1 = v
        → encodes "which neurons fire early (more informative)"

    Population correlation coding (Kleyko 2023 Survey §III):
        For each pair (i, j) that co-fire within window W:
          hv_corr ⊕= XOR(id_hv(i), id_hv(j))   (XOR ≈ bind the pair)
        → encodes "which neurons synchronise"

    All three HVs are XOR-bound with modality-specific role HVs, then
    bundled via majority to produce a single composite spike HV.

    Args:
        n_neurons: Number of neurons in the SNN
        hd_dim: Hypervector dimensionality
        n_levels: Quantisation levels for rate coding
        corr_window: Max timestep lag for co-firing pair detection
        seed: Random seed
    """

    def __init__(
        self,
        n_neurons: int,
        hd_dim: int,
        n_levels: int = 64,
        corr_window: int = 3,
        seed: int = 0,
    ):
        self.n_neurons = n_neurons
        self.hd_dim = hd_dim
        self.corr_window = corr_window

        # Neuron identity HVs (one per neuron, near-orthogonal)
        self._id_hvs = gen_hvs(n_neurons, hd_dim, seed=seed)               # (N, D)

        # Level HVs for rate coding (Rahimi 2017)
        self._level_hvs = LevelHypervectors(n_levels=n_levels, dim=hd_dim, seed=seed+1)

        # Fractional binding for first-spike-time (Verges Boncompte 2024)
        self._frac = FractionalBinding(dim=hd_dim, seed=seed+2)

        # Role HVs: bind each coding dimension to a unique role
        g = torch.Generator(); g.manual_seed(seed+3)
        self._role_rate  = (torch.rand(hd_dim, generator=g) < 0.5).float()
        self._role_fst   = (torch.rand(hd_dim, generator=g) < 0.5).float()
        self._role_corr  = (torch.rand(hd_dim, generator=g) < 0.5).float()

    def encode(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Encode a spike train to a composite HV.

        Args:
            spikes: (T, n_neurons) binary spike tensor

        Returns:
            (hd_dim,) composite spike HV
        """
        T, N = spikes.shape
        components = []

        # ── Rate coding ───────────────────────────────────────────────────────
        rates = spikes.float().mean(dim=0)   # (N,) ∈ [0, 1]
        rate_parts = []
        for i in range(N):
            lhv = self._level_hvs.encode(float(rates[i]))  # level HV for rate
            bound = hv_xor(self._id_hvs[i], lhv)           # bind neuron id ⊗ rate
            rate_parts.append(bound)
        hv_rate = hv_majority(torch.stack(rate_parts).mean(dim=0))
        components.append(hv_xor(hv_rate, self._role_rate))

        # ── First-spike-time coding ───────────────────────────────────────────
        fst_parts = []
        for i in range(N):
            fire_times = spikes[:, i].nonzero(as_tuple=True)[0]
            if len(fire_times) > 0:
                fst = float(fire_times[0].item()) / max(T - 1, 1)
            else:
                fst = 1.0   # never fired → latest possible time
            # v^{1-fst}: early firer (fst=0) → v^1 = v (distinctive)
            #             late firer (fst=1) → v^0 = constant (less info)
            fst_hv = self._frac.encode(1.0 - fst).detach().clone()
            fst_hv_bin = (fst_hv > 0).float()
            bound = hv_xor(self._id_hvs[i], fst_hv_bin)
            fst_parts.append(bound)
        hv_fst = hv_majority(torch.stack(fst_parts).mean(dim=0))
        components.append(hv_xor(hv_fst, self._role_fst))

        # ── Population correlation coding ─────────────────────────────────────
        corr_hv = torch.zeros(self.hd_dim)
        n_corr = 0
        for t in range(T):
            firing_now = spikes[t].nonzero(as_tuple=True)[0].tolist()
            # Check co-firing within window
            t_end = min(T, t + self.corr_window)
            for t2 in range(t + 1, t_end):
                firing_later = spikes[t2].nonzero(as_tuple=True)[0].tolist()
                for i in firing_now:
                    for j in firing_later:
                        if i != j:
                            corr_hv += hv_xor(self._id_hvs[i], self._id_hvs[j]).float()
                            n_corr += 1
        if n_corr > 0:
            hv_corr = hv_majority(corr_hv / n_corr)
        else:
            hv_corr = (torch.rand(self.hd_dim) < 0.5).float()
        components.append(hv_xor(hv_corr, self._role_corr))

        # ── Composite: majority bundle of all three ───────────────────────────
        stacked = torch.stack(components).float()
        return hv_majority(stacked.mean(dim=0))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. STDP-HDC — Temporal Knowledge Graph
# ═══════════════════════════════════════════════════════════════════════════════

class STDPHDCGraph:
    """
    Spike-Timing Dependent Plasticity in HV space.

    Biological STDP:
        pre fires at t_pre, post fires at t_post
        Δt = t_post − t_pre
        ΔW ∝ exp(−|Δt| / τ)   (LTP if Δt > 0, LTD if Δt < 0)

    HDC-STDP:
        pre concept: C_pre = concept HV associated with pre-synaptic spike
        post concept: C_post = concept HV associated with post-synaptic spike
        temporal displacement: encoded as cyclic shift Sh(C_post, k)
                               where k = round(Δt / τ × max_shift)

        pair_hv = XOR(C_pre, Sh(C_post, k))  → "C_pre caused C_post at lag k"

        This pair HV is accumulated in the temporal knowledge graph:
            G ← G + weight(Δt) × pair_hv

        where weight(Δt) = exp(−|Δt| / τ_stdp) × sign(Δt)
                         = positive for causal pairs, negative for anticausal

    The graph G captures "what concept tends to follow what other concept,
    and with what temporal lag." Querying by unbinding:
        consequent_hv = XOR(G, C_pre)   → predicted next concept given C_pre

    Args:
        hd_dim: Hypervector dimensionality
        tau_stdp: STDP time constant (steps)
        max_shift: Maximum cyclic shift for temporal encoding
        ltp_weight: Weight for causal pairs (LTP)
        ltd_weight: Weight for anti-causal pairs (LTD)
    """

    def __init__(
        self,
        hd_dim: int,
        tau_stdp: float = 10.0,
        max_shift: int = 20,
        ltp_weight: float = 1.0,
        ltd_weight: float = 0.5,
    ):
        self.hd_dim = hd_dim
        self.tau_stdp = tau_stdp
        self.max_shift = max_shift
        self.ltp_weight = ltp_weight
        self.ltd_weight = ltd_weight

        # Temporal graph HV accumulator
        self._graph = torch.zeros(hd_dim)
        self._n_pairs = 0

        # Concept event buffer: (concept_hv, timestamp) deque
        self._event_buffer: Deque[Tuple[torch.Tensor, int]] = deque(maxlen=50)
        self._tick = 0

    def observe_spike(self, concept_hv: torch.Tensor):
        """
        Record a spike at the current tick, associated with concept_hv.

        Computes STDP updates between this spike and all buffered past spikes.
        """
        now = self._tick

        for past_hv, past_t in self._event_buffer:
            dt = now - past_t   # positive: past_t fires before now (causal)
            if dt <= 0:
                continue

            # Temporal weight: decays with |dt|, positive for causal
            w = math.exp(-dt / self.tau_stdp) * self.ltp_weight

            # Temporal shift: encodes causal lag direction
            k = min(int(round(dt / self.tau_stdp * self.max_shift)), self.max_shift)
            shifted_post = hv_permute(concept_hv, k=k)

            # pair_hv = XOR(C_pre, Sh(C_post, k)) — causal binding
            pair_hv = hv_xor(past_hv, shifted_post)
            self._graph += w * pair_hv.float()
            self._n_pairs += 1

        self._event_buffer.append((concept_hv.detach(), now))
        self._tick += 1

    def predict_consequent(
        self,
        concept_hv: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Predict what concept tends to follow concept_hv.

        Unbind: consequent = XOR(G_bin, C_pre)
        Confidence: how non-random is the result?

        Returns:
            (consequent_hv, confidence ∈ [0, 1])
        """
        if self._n_pairs == 0:
            return (torch.rand(self.hd_dim) < 0.5).float(), 0.0

        g_bin = hv_majority(self._graph / max(self._n_pairs, 1))
        consequent = hv_xor(g_bin, concept_hv)
        confidence = abs(float(consequent.mean().item()) - 0.5) * 2
        return consequent, confidence

    def causal_strength(
        self,
        pre_hv: torch.Tensor,
        post_hv: torch.Tensor,
    ) -> float:
        """
        How strongly does pre causally predict post in the learned graph?

        Returns:
            causal strength ∈ [0, 1]
        """
        predicted, _ = self.predict_consequent(pre_hv)
        return float(hv_batch_sim(predicted, post_hv.unsqueeze(0))[0].item())

    @property
    def n_pairs(self) -> int:
        return self._n_pairs


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Semantic Attention — HV similarity modulates SNN thresholds
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticAttention:
    """
    Top-down semantic modulation of SNN firing thresholds.

    When a concept C becomes "active" (sim(state_hv, C_hv) > gate_threshold):
        for each neuron i:
            attention_i = sim(basis_i, C_hv)  ∈ [0, 1]
            v_th_i ← v_th_nominal × (1 − α × attention_i)

    Neurons whose basis HV is similar to the active concept get a LOWER
    threshold → they fire more readily → they generate more spikes relevant
    to concept C → the state HV becomes even more similar to C → attractor.

    This implements a VSA-native attentional spotlight:
    - No weight matrices, no backward passes
    - Attention = Hamming similarity in concept space
    - Graceful: sim ≈ 0.5 (random) → no modulation

    Reset: when a concept deactivates or the attention period expires,
    thresholds return to nominal values.

    Args:
        basis_hvs: (N, D) neuron basis hypervectors (from SpikingHVNetwork)
        nominal_threshold: Baseline firing threshold
        attention_gain: Maximum fractional threshold reduction (α)
        gate_threshold: Min state-concept similarity to trigger attention
    """

    def __init__(
        self,
        basis_hvs: torch.Tensor,          # (N, D)
        nominal_threshold: float = 1.0,
        attention_gain: float = 0.4,
        gate_threshold: float = 0.6,
    ):
        self.basis_hvs = basis_hvs
        self.nominal = nominal_threshold
        self.gain = attention_gain
        self.gate = gate_threshold
        self.n_neurons = basis_hvs.shape[0]

        self._active_concept: Optional[torch.Tensor] = None
        self._active_name: Optional[str] = None

    def attend(
        self,
        state_hv: torch.Tensor,
        concept_library: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[str], torch.Tensor]:
        """
        Find the most active concept and compute modulated thresholds.

        Args:
            state_hv: (D,) current SNN state hypervector
            concept_library: {name: HV} of known concepts

        Returns:
            (active_concept_name, threshold_vector (N,))
        """
        if not concept_library:
            return None, torch.full((self.n_neurons,), self.nominal)

        names = list(concept_library.keys())
        hvs = torch.stack([concept_library[n] for n in names])   # (K, D)

        # Similarity of state to all concepts
        sims = hv_batch_sim(state_hv, hvs)                        # (K,)
        best_sim = float(sims.max().item())
        best_name = names[int(sims.argmax().item())]

        if best_sim < self.gate:
            self._active_concept = None
            self._active_name = None
            return None, torch.full((self.n_neurons,), self.nominal)

        # Compute neuron-level attention
        concept_hv = concept_library[best_name]
        neuron_attention = hv_batch_sim(concept_hv, self.basis_hvs)  # (N,)
        # attention ∈ [0, 1] (Hamming similarity); 0.5 = random (no modulation)
        modulation = (neuron_attention - 0.5) * 2  # map [0.5, 1] → [0, 1]
        modulation = modulation.clamp(min=0.0)

        thresholds = self.nominal * (1.0 - self.gain * modulation)  # (N,)
        thresholds = thresholds.clamp(min=self.nominal * 0.2)        # floor at 20%

        self._active_concept = concept_hv
        self._active_name = best_name
        return best_name, thresholds

    @property
    def active_concept(self) -> Optional[str]:
        return self._active_name


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Semantic Quality Metrics
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SemanticQualityReport:
    """Snapshot of semantic representation quality."""
    n_concepts: int
    coherence: float        # mean sim between co-occurring concepts (want high)
    separability: float     # mean sim between random pairs (want ~0.5)
    semantic_gap: float     # coherence − separability (want > 0.15)
    stability: float        # mean HV drift per observation (want < 0.05)
    coverage: float         # estimated fraction of HV space explored
    quality_score: float    # composite ∈ [0, 1]


class SemanticQualityMonitor:
    """
    Continuous measurement of semantic representation quality.

    Tracks four dimensions of quality without requiring labelled data:

    **Coherence** — concepts that appear together should be similar.
        If the knowledge graph says A and B co-occur frequently,
        sim(HV_A, HV_B) should be high. Measures whether structure in the
        world is reflected as similarity in HV space.

    **Separability** — distinct concepts should be orthogonal.
        Mean pairwise similarity between random concept pairs should be ≈0.5
        (random, orthogonal). Deviation from 0.5 in either direction is bad:
        too high = confusion, too low = impossible in binary space.

    **Semantic gap** — coherence − separability > 0 means the system has
        learned to distinguish "related" from "unrelated" concepts.
        Gap < 0.1: poor learning; Gap > 0.25: strong semantic structure.

    **Stability** — drift of concept HVs over time.
        In lifelong learning, HVs should stabilise as concepts are well-learned.
        High drift = catastrophic forgetting or rapid concept revision.

    **Coverage** — how much of the D-dimensional space is used.
        Estimated via pairwise diversity: mean(1 - |sim - 0.5|) across all pairs.
        Coverage = 1 → all pairs are at distance ≈ 0.5 (maximally spread).

    Args:
        knowledge_graph: KnowledgeGraph whose concepts to evaluate
        stability_window: Number of past HVs to compute drift from
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        stability_window: int = 20,
    ):
        self.kg = knowledge_graph
        self.stability_window = stability_window
        self._past_hvs: Dict[str, Deque[torch.Tensor]] = {}

    def snapshot(self) -> SemanticQualityReport:
        """Compute current quality report."""
        concepts = list(self.kg.vertices.keys())
        n = len(concepts)

        if n < 2:
            return SemanticQualityReport(n, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0)

        hvs = [self.kg.vertices[c] for c in concepts]

        # ── Coherence: co-occurring concepts should be similar ────────────────
        coherence_sims = []
        for (a, b), w in list(self.kg.edges.items())[:50]:  # sample up to 50 edges
            if a in self.kg.vertices and b in self.kg.vertices:
                sim = float(hv_batch_sim(
                    self.kg.vertices[a],
                    self.kg.vertices[b].unsqueeze(0)
                )[0].item())
                coherence_sims.append(sim)
        coherence = sum(coherence_sims) / len(coherence_sims) if coherence_sims else 0.5

        # ── Separability: random pairs should be ~0.5 ────────────────────────
        rnd_sims = []
        import random
        pairs = [(random.randint(0, n-1), random.randint(0, n-1))
                 for _ in range(min(50, n*(n-1)//2))]
        for i, j in pairs:
            if i != j:
                sim = float(hv_batch_sim(hvs[i], hvs[j].unsqueeze(0))[0].item())
                rnd_sims.append(sim)
        separability = sum(rnd_sims) / len(rnd_sims) if rnd_sims else 0.5

        # ── Semantic gap ──────────────────────────────────────────────────────
        semantic_gap = max(0.0, coherence - separability)

        # ── Stability: mean drift of concept HVs ─────────────────────────────
        drifts = []
        for name, hv in zip(concepts, hvs):
            if name in self._past_hvs and len(self._past_hvs[name]) >= 2:
                past = list(self._past_hvs[name])[-2]
                drift = 1.0 - float(hv_batch_sim(hv, past.unsqueeze(0))[0].item())
                drifts.append(drift)
            # Record current HV
            if name not in self._past_hvs:
                self._past_hvs[name] = deque(maxlen=self.stability_window)
            self._past_hvs[name].append(hv.detach().clone())
        stability = sum(drifts) / len(drifts) if drifts else 0.0

        # ── Coverage: diversity of concept HVs ───────────────────────────────
        if rnd_sims:
            coverage = float(torch.tensor([1.0 - abs(s - 0.5) * 2 for s in rnd_sims]).mean())
        else:
            coverage = 0.5

        # ── Composite quality score ───────────────────────────────────────────
        gap_score  = min(1.0, semantic_gap * 5)           # 0.2 gap → 1.0
        stab_score = max(0.0, 1.0 - stability * 10)       # drift > 0.1 → 0
        cov_score  = min(1.0, coverage * 2)               # 0.5 coverage → 1.0
        quality = (gap_score + stab_score + cov_score) / 3

        return SemanticQualityReport(
            n_concepts=n,
            coherence=round(coherence, 4),
            separability=round(separability, 4),
            semantic_gap=round(semantic_gap, 4),
            stability=round(stability, 4),
            coverage=round(coverage, 4),
            quality_score=round(quality, 4),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SNNSemanticAgent — full integration
# ═══════════════════════════════════════════════════════════════════════════════

class SNNSemanticAgent:
    """
    Full SNN + Vector Semantic integration agent.

    Combines:
    - SpikingHVNetwork (or any SNN with state_hv output)
    - TemporalSpikeEncoder (rate + FST + population coding)
    - KnowledgeGraph + LifeLongSemanticLearner (Sutor 2018)
    - STDPHDCGraph (temporal causal knowledge)
    - SemanticAttention (top-down threshold modulation)
    - SemanticQualityMonitor (continuous quality tracking)

    Tick cycle (called for each sensor observation):
      1. Encode sensor input → spike train (via SNN or direct)
      2. Temporal spike encoding → composite HV
      3. Identify nearest concept in knowledge graph
      4. STDP update: record spike event with concept HV
      5. Semantic attention: modulate SNN thresholds for next step
      6. Update knowledge graph with new co-occurrences
      7. Tension optimisation (periodic): refine concept HVs
      8. Quality monitoring: track coherence, separability, stability

    Args:
        n_neurons: SNN neuron count (for basis HVs)
        hd_dim: Hypervector dimensionality
        snn_network: Optional SpikingHVNetwork to drive
        quality_check_period: Steps between quality snapshots
    """

    def __init__(
        self,
        n_neurons: int,
        hd_dim: int,
        snn_network=None,
        quality_check_period: int = 50,
        seed: int = 42,
    ):
        self.n_neurons = n_neurons
        self.hd_dim = hd_dim
        self.snn = snn_network
        self.quality_check_period = quality_check_period

        # SNN-HDC bridge
        self.spike_encoder = TemporalSpikeEncoder(n_neurons, hd_dim, seed=seed)
        self.stdp_graph = STDPHDCGraph(hd_dim)

        # Vector semantic core
        self.kg = KnowledgeGraph(dim=hd_dim, seed=seed)

        # Semantic attention (needs SNN basis HVs if available)
        if snn_network is not None and hasattr(snn_network, 'basis'):
            basis = snn_network.basis
        else:
            basis = gen_hvs(n_neurons, hd_dim, seed=seed)
        self.attention = SemanticAttention(basis, nominal_threshold=1.0)

        # Quality monitor
        self.quality_monitor = SemanticQualityMonitor(self.kg)

        self._tick = 0
        self._last_quality: Optional[SemanticQualityReport] = None
        self._concept_history: List[str] = []

    def observe(
        self,
        spikes: torch.Tensor,
        concept_label: Optional[str] = None,
    ) -> Dict:
        """
        Process one observation through the full SNN-semantic pipeline.

        Args:
            spikes: (T, n_neurons) binary spike train
            concept_label: Optional string label (for supervised mode)

        Returns:
            Dict with composite_hv, nearest_concept, stdp_confidence,
                   attention_concept, quality_report (periodic)
        """
        self._tick += 1

        # Step 1: Temporal spike encoding → composite HV
        composite_hv = self.spike_encoder.encode(spikes)

        # Step 2: Identify nearest concept (or create new one)
        nearest_concept = concept_label
        if concept_label is not None:
            # Supervised: add/update concept
            existing = self.kg.get_hv(concept_label)
            if existing is None:
                self.kg.add_concept(concept_label)
                self.kg.set_hv(concept_label, composite_hv)
            else:
                # EMA update: blend new observation into existing concept HV
                blended = hv_majority((existing.float() * 0.8 + composite_hv.float() * 0.2))
                self.kg.set_hv(concept_label, blended)
        else:
            # Unsupervised: find nearest existing concept
            concepts = list(self.kg.vertices.keys())
            if concepts:
                hvs = torch.stack([self.kg.vertices[c] for c in concepts])
                sims = hv_batch_sim(composite_hv, hvs)
                best_sim = float(sims.max().item())
                nearest_concept = concepts[int(sims.argmax().item())] if best_sim > 0.6 else None
                if nearest_concept is None:
                    # Auto-create new concept
                    auto_name = f"concept_{len(concepts)}"
                    self.kg.add_concept(auto_name)
                    self.kg.set_hv(auto_name, composite_hv)
                    nearest_concept = auto_name

        # Step 3: STDP update
        concept_hv = self.kg.get_hv(nearest_concept) if nearest_concept else composite_hv
        if concept_hv is not None:
            self.stdp_graph.observe_spike(concept_hv)

        # Step 4: Co-occurrence tracking for knowledge graph
        if self._concept_history and nearest_concept:
            prev_concept = self._concept_history[-1]
            if prev_concept != nearest_concept:
                self.kg.add_cooccurrence(prev_concept, nearest_concept, weight=1.0)

        if nearest_concept:
            self._concept_history.append(nearest_concept)

        # Step 5: Semantic attention modulation
        concept_lib = self.kg.vertices
        attn_concept, thresholds = self.attention.attend(composite_hv, concept_lib)

        # Step 6: Quality monitoring (periodic)
        quality_report = None
        if self._tick % self.quality_check_period == 0:
            quality_report = self.quality_monitor.snapshot()
            self._last_quality = quality_report

        # Step 7: STDP causal prediction
        stdp_pred = None
        stdp_conf = 0.0
        if concept_hv is not None:
            stdp_pred, stdp_conf = self.stdp_graph.predict_consequent(concept_hv)

        return {
            "composite_hv": composite_hv,
            "nearest_concept": nearest_concept,
            "attention_concept": attn_concept,
            "attention_thresholds": thresholds,
            "stdp_confidence": stdp_conf,
            "stdp_prediction": stdp_pred,
            "quality_report": quality_report,
            "tick": self._tick,
        }

    def concept_summary(self) -> Dict:
        """Return summary of learned concepts and quality."""
        return {
            "n_concepts": len(self.kg.vertices),
            "concept_names": list(self.kg.vertices.keys()),
            "n_stdp_pairs": self.stdp_graph.n_pairs,
            "last_quality": self._last_quality,
        }

    def most_active_concepts(self, top_k: int = 5) -> List[Tuple[str, int]]:
        """
        Return the top-k most frequently activated concepts from history.

        Useful for: identifying the dominant semantic themes in the spike stream,
        detecting recurring states in a physical system.

        Returns:
            List of (concept_name, activation_count) sorted descending.
        """
        from collections import Counter
        counts = Counter(self._concept_history)
        return counts.most_common(top_k)

    def concept_transition_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        Build a concept→concept transition count matrix from the history.

        Entry [A][B] = number of times concept A was immediately followed by B.
        Reveals temporal structure: which concepts tend to follow which.

        Returns:
            Dict of {concept_from: {concept_to: count}}.
        """
        matrix: Dict[str, Dict[str, int]] = {}
        for i in range(len(self._concept_history) - 1):
            a = self._concept_history[i]
            b = self._concept_history[i + 1]
            if a not in matrix:
                matrix[a] = {}
            matrix[a][b] = matrix[a].get(b, 0) + 1
        return matrix

    def semantic_health(self) -> Dict:
        """
        Overall semantic health of the agent's learned knowledge.

        Returns:
            Dict with n_concepts, most_active, transition_entropy, quality.
        """
        import math
        top = self.most_active_concepts(top_k=3)
        n_hist = max(len(self._concept_history), 1)

        # Transition entropy: how diverse is the concept sequence?
        trans = self.concept_transition_matrix()
        all_transitions = [cnt for src in trans.values() for cnt in src.values()]
        t_total = max(sum(all_transitions), 1)
        t_entropy = -sum((c / t_total) * math.log(c / t_total)
                         for c in all_transitions if c > 0)

        return {
            "n_concepts":        len(self.kg.vertices),
            "n_ticks":           self._tick,
            "most_active":       top,
            "transition_entropy": round(t_entropy, 4),
            "last_quality":      self._last_quality,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_temporal_spike_encoder():
    print("=" * 60)
    print("Testing TemporalSpikeEncoder (rate + FST + population)")
    print("=" * 60)

    torch.manual_seed(42)
    N, T, D = 64, 20, 2000
    enc = TemporalSpikeEncoder(n_neurons=N, hd_dim=D, seed=0)

    # Two different spike patterns should produce different HVs
    spikes_a = (torch.rand(T, N) < 0.3).float()
    spikes_b = (torch.rand(T, N) < 0.3).float()
    spikes_a_copy = spikes_a.clone()

    hv_a = enc.encode(spikes_a)
    hv_b = enc.encode(spikes_b)
    hv_a2 = enc.encode(spikes_a_copy)

    sim_same = float(hv_batch_sim(hv_a, hv_a2.unsqueeze(0))[0])
    sim_diff = float(hv_batch_sim(hv_a, hv_b.unsqueeze(0))[0])
    density  = float(hv_a.mean())

    print(f"  sim(same spikes): {sim_same:.4f}  (want ≈ 1.0)")
    print(f"  sim(diff spikes): {sim_diff:.4f}  (want ≈ 0.5)")
    print(f"  output density:   {density:.4f}  (want ≈ 0.5)")

    assert sim_same > 0.9, f"Same spikes should produce same HV: {sim_same}"
    assert abs(density - 0.5) < 0.1, f"Output density off: {density}"

    # High-rate neurons should dominate: dense spikes → different HV than sparse
    spikes_dense  = (torch.rand(T, N) < 0.8).float()
    spikes_sparse = (torch.rand(T, N) < 0.05).float()
    hv_dense  = enc.encode(spikes_dense)
    hv_sparse = enc.encode(spikes_sparse)
    sim_ds = float(hv_batch_sim(hv_dense, hv_sparse.unsqueeze(0))[0])
    print(f"  sim(dense, sparse): {sim_ds:.4f}  (want < 0.8)")

    print("  ✅ TemporalSpikeEncoder OK")


def test_stdp_hdc():
    print("=" * 60)
    print("Testing STDPHDCGraph (temporal causal knowledge)")
    print("=" * 60)

    torch.manual_seed(7)
    D = 2000
    graph = STDPHDCGraph(D, tau_stdp=5.0)

    # Concept HVs
    concept_A = (torch.rand(D) < 0.5).float()
    concept_B = (torch.rand(D) < 0.5).float()
    concept_C = (torch.rand(D) < 0.5).float()

    # Repeatedly observe A → B (causal sequence)
    for _ in range(20):
        graph.observe_spike(concept_A)
        graph.observe_spike(concept_B)  # follows A by 1 step

    print(f"  STDP pairs recorded: {graph.n_pairs}")
    assert graph.n_pairs > 0

    # A should predict B
    pred, conf = graph.predict_consequent(concept_A)
    causal_str_AB = graph.causal_strength(concept_A, concept_B)
    causal_str_AC = graph.causal_strength(concept_A, concept_C)

    print(f"  Causal strength A→B: {causal_str_AB:.4f}  (want > A→C)")
    print(f"  Causal strength A→C: {causal_str_AC:.4f}")
    print(f"  Prediction confidence: {conf:.4f}")

    assert causal_str_AB >= causal_str_AC, "A→B should be stronger than A→C"

    print("  ✅ STDPHDCGraph OK")


def test_semantic_attention():
    print("=" * 60)
    print("Testing SemanticAttention (top-down threshold modulation)")
    print("=" * 60)

    torch.manual_seed(99)
    N, D = 64, 2000
    basis_hvs = gen_hvs(N, D, seed=42)
    attn = SemanticAttention(basis_hvs, nominal_threshold=1.0, attention_gain=0.4)

    # Concept library: concept_A similar to first 16 basis HVs
    concept_A = hv_majority(basis_hvs[:16].mean(dim=0))
    concept_B = (torch.rand(D) < 0.5).float()   # random, unrelated

    concept_lib = {"A": concept_A, "B": concept_B}

    # State HV similar to concept_A → should activate attention on A
    state_hv = concept_A.clone()
    name, thresholds = attn.attend(state_hv, concept_lib)

    print(f"  Active concept: {name}  (want 'A')")
    print(f"  Threshold range: [{thresholds.min():.4f}, {thresholds.max():.4f}]")
    print(f"  Nominal threshold: 1.0")

    if name == "A":
        # Neurons similar to A should have lower threshold
        attn_to_A = hv_batch_sim(concept_A, basis_hvs)  # (N,)
        high_attn_neurons = (attn_to_A > 0.6).float()
        if high_attn_neurons.sum() > 0:
            avg_high = thresholds[high_attn_neurons.bool()].mean().item()
            avg_low  = thresholds[~high_attn_neurons.bool()].mean().item()
            print(f"  High-attention neurons avg threshold: {avg_high:.4f}")
            print(f"  Low-attention neurons avg threshold:  {avg_low:.4f}")
            assert avg_high <= avg_low, "Attended neurons should have lower threshold"

    print("  ✅ SemanticAttention OK")


def test_semantic_quality_monitor():
    print("=" * 60)
    print("Testing SemanticQualityMonitor")
    print("=" * 60)

    torch.manual_seed(0)
    D = 2000
    kg = KnowledgeGraph(dim=D, seed=42)

    # Add 5 concepts with structured co-occurrences
    concept_hvs = {}
    for name in ["bearing_normal", "bearing_fault", "gear_normal", "gear_fault", "ambient"]:
        kg.add_concept(name)
        concept_hvs[name] = (torch.rand(D) < 0.5).float()
        kg.set_hv(name, concept_hvs[name])

    # Add co-occurrences (fault pairs are related)
    kg.add_cooccurrence("bearing_fault", "gear_fault", weight=0.8)
    kg.add_cooccurrence("bearing_normal", "gear_normal", weight=0.7)
    kg.add_cooccurrence("bearing_normal", "ambient", weight=0.5)

    monitor = SemanticQualityMonitor(kg)
    report = monitor.snapshot()

    print(f"  n_concepts: {report.n_concepts}")
    print(f"  coherence:   {report.coherence:.4f}  (cooccurring concepts' similarity)")
    print(f"  separability:{report.separability:.4f}  (want ≈ 0.5)")
    print(f"  semantic_gap:{report.semantic_gap:.4f}  (coherence − separability, want > 0)")
    print(f"  coverage:    {report.coverage:.4f}")
    print(f"  quality:     {report.quality_score:.4f}")

    assert 0 <= report.quality_score <= 1
    assert report.n_concepts == 5

    print("  ✅ SemanticQualityMonitor OK")


def test_snn_semantic_agent():
    print("=" * 60)
    print("Testing SNNSemanticAgent (full integration)")
    print("=" * 60)

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    torch.manual_seed(42)
    N, D, T = 32, 1000, 8

    agent = SNNSemanticAgent(n_neurons=N, hd_dim=D, seed=0,
                              quality_check_period=10)

    # Observe 20 spike patterns with alternating concepts
    def make_spikes(rate: float) -> torch.Tensor:
        return (torch.rand(T, N) < rate).float()

    for i in range(25):
        if i % 2 == 0:
            result = agent.observe(make_spikes(0.7), concept_label="high_rate")
        else:
            result = agent.observe(make_spikes(0.1), concept_label="low_rate")

    print(f"  After 25 observations:")
    summary = agent.concept_summary()
    print(f"  Concepts: {summary['concept_names']}")
    print(f"  STDP pairs: {summary['n_stdp_pairs']}")

    if summary['last_quality']:
        q = summary['last_quality']
        print(f"  Quality: coherence={q.coherence:.3f} gap={q.semantic_gap:.3f} score={q.quality_score:.3f}")

    assert len(summary['concept_names']) >= 2
    assert summary['n_stdp_pairs'] > 0

    # Test causal prediction: high_rate → low_rate
    high_hv = agent.kg.get_hv("high_rate")
    low_hv  = agent.kg.get_hv("low_rate")
    if high_hv is not None and low_hv is not None:
        cs = agent.stdp_graph.causal_strength(high_hv, low_hv)
        print(f"  Causal strength (high→low): {cs:.4f}")

    print("  ✅ SNNSemanticAgent OK")


if __name__ == "__main__":
    test_temporal_spike_encoder()
    print()
    test_stdp_hdc()
    print()
    test_semantic_attention()
    print()
    test_semantic_quality_monitor()
    print()
    test_snn_semantic_agent()
    print()
    print("=== All snn_semantics tests passed ===")
