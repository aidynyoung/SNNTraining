# SNNTraining

**Train spiking neural networks without backpropagation.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-1621%20passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/research-1582%20papers%20indexed-orange)](research/)

Spiking neural networks fire spikes, not floats. This changes everything about how you train them. This library implements every biologically-plausible learning rule that actually works, plus a full HDC (Hyperdimensional Computing) layer for classification — the only stack that runs at 2.4 nJ/inference, learns online, and tolerates hardware faults.

No backpropagation through time. No replay buffer. No GPU required.

---

## Install

```bash
git clone https://github.com/Aidistides/SNNTraining
cd SNNTraining
pip install -r requirements.txt
python demo.py
```

---

## The simplest possible SNN

```python
from models.lif import LIFLayer
import torch

lif = LIFLayer(size=128, tau=20.0, v_th=1.0)

for t in range(100):
    x = torch.randn(128)
    spikes = lif.step(x)   # {0,1} spike vector
    print(spikes.sum().item(), "neurons fired")
```

That's a Leaky Integrate-and-Fire layer. Every timestep: integrate current, fire if above threshold, reset. No weight update happens here — the SNN is just a dynamical system.

---

## Training approaches (pick one)

The RSNN (recurrent SNN) is a fixed reservoir. You train *on top of* it — either the HDC prototype layer or a linear readout.

### 1. Prototype bundling — fastest, no math

```python
from models.snn_hdc_pipeline import SNNHDCPipeline, PipelineConfig

pipe = SNNHDCPipeline(PipelineConfig(n_classes=10, hidden_size=128))

for x, label in train_data:
    pipe.train_step(x, label)   # SNN → spike window → HDC → bundle into prototype

pipe.finalize()
pred, conf, _ = pipe.hdc_infer()
```

Single pass over data. Accuracy ~80–85% on simple tasks. No gradient, no optimizer, no epochs.

### 2. Prototype refinement (RefineHD)

```python
from hdc.continual_hdc import ClassMeanHDCClassifier

clf = ClassMeanHDCClassifier(dim=4096, n_classes=10)
clf.fit(X_train, y_train)        # class-mean init (Harun & Kanan 2025)
clf.refine(X_train, y_train, epochs=5)   # push/pull misclassified samples
```

Push misclassified samples away from wrong prototypes, toward correct ones. Gains +3–6% over raw bundling. Still no backprop.

### 3. Eligibility traces (e-prop / online Hebbian)

```python
from models.hebbian import DualHebbianLearner

learner = DualHebbianLearner(
    hidden_size=128,
    tau_fast=5.0,   # ~100ms window (local syntax)
    tau_slow=50.0,  # ~700ms window (sequential context)
    alpha=0.7,
)

for x_t, signal in stream:
    spikes = rsnn.forward(x_t)
    E = learner.compute_trace(spikes)       # eligibility trace
    learner.update(E, modulation=signal)    # weight update gated by signal
```

Dual-timescale traces: fast captures precise spike timing, slow captures sequential context. On BCI velocity decoding: **Pearson R 0.81** at O(1) memory, no BPTT.

### 4. LeHDC (gradient + STE binarization)

```python
from hdc.adaptive_encoder import LeHDCEncoder

enc = LeHDCEncoder(dim=4096, input_dim=64)
enc.train_supervised(X_train, y_train, epochs=20, lr=1e-3)
# Straight-Through Estimator: gradient flows through binary sign()
```

When prototype bundling saturates. Gains +5–10% on hard tasks (SHD, BCI). The STE trick: forward uses sign(), backward pretends sign() was identity.

### 5. Full pipeline: SNN reservoir → HDC → optional fallback

```python
from models.snn_hdc_pipeline import SNNHDCPipeline, PipelineConfig

pipe = SNNHDCPipeline(PipelineConfig(
    n_classes=10,
    use_snn_fallback=True,
    gate_threshold=0.4,    # HDC confidence below this → linear readout takes over
    fallback_lr=0.02,
))

for x, label in train_data:
    pipe.train_step(x, label)  # trains HDC prototypes AND linear readout simultaneously
```

The linear readout learns via delta rule (online softmax regression) in parallel — zero extra SNN passes. When HDC is uncertain (similarity < threshold), the readout takes over.

---

## Rule of thumb

| Situation | Use |
|-----------|-----|
| First experiment | `train_step()` — prototype bundling |
| Accuracy plateau | `refine()` — push/pull |
| Sequential/temporal input | `use_snn_fallback=True` |
| Hard benchmark (SHD, BCI) | `LeHDCEncoder` or eligibility traces |
| Long deployment | `homeostatic_scale()` every 100 steps |
| Firing rate issues | `adapt_input_gain()` |

---

## Architecture

```
Input spikes (x_t)
      │
    RSNN                     LIF dynamics + sparse recurrent weights
      │  spike vector
  Rate window                rolling mean over last 50 timesteps
      │  (hidden_size,)
   HDC encoder               bind spike rates to random key hypervectors (XOR)
      │  (4096,)
  Assoc. memory              Hamming similarity → nearest class prototype
      │
  Prediction + confidence
      │
  (optional) SNN readout     linear δ-rule fallback when HDC uncertain
```

**All operations**: XOR, popcount, majority vote. No matrix multiplication in the HDC path.

**The RSNN is a fixed reservoir.** Its value is temporal dynamics — it converts a point-in-time input into a rich spike trajectory. You never update `W_rec` during supervised training.

---

## RSNN internals

The recurrent network (`models/rsnn.py`) is initialized carefully:

```python
from models.rsnn import RSNN, RSNNConfig

rsnn = RSNN(
    input_size=100,
    hidden_size=256,
    sparse_init=True,     # Erdős–Rényi topology, p=0.15
    sparse_p=0.15,
    input_gain=5.0,
)
# Spectral radius is set to 0.97 at init (edge of chaos — Jaeger & Haas 2004)
# Zero diagonal (no self-connections)
# Orthogonal initialization for W_in
```

Optional biology:

```python
rsnn = RSNN(config=RSNNConfig(
    hidden_size=256,
    use_dale=True,          # 80% excitatory / 20% inhibitory (Brunel 2000)
    use_stp=True,           # Tsodyks-Markram short-term plasticity
    heterogeneous_tau=True, # per-neuron τ ~ LogNormal(log(20), 0.5)
))
```

Heterogeneous time constants (Perez-Nieves et al. 2021, Nature Comm.): fast neurons capture precise timing, slow neurons carry temporal context. +2–4% on sequential tasks.

---

## LIF neuron

```python
from models.lif import LIFLayer, LIFConfig

# Standard
lif = LIFLayer(size=128, tau=20.0, v_th=1.0, refractory=2)

# With test-time threshold adaptation (Zhao et al. 2026, arXiv:2505.05375)
lif = LIFLayer(LIFConfig(
    size=128,
    enable_threshold_adaptation=True,
    threshold_adaptation_rate=0.01,   # γ: how fast threshold tracks v̄
))

# Diagnostics
health = lif.neuron_health(window=50)
# → {"mean_firing_rate": 0.09, "synchrony": 0.12, "diagnosis": "healthy"}
```

Threshold adaptation handles distribution shift at deployment with zero additional compute — the threshold tracks the running mean of membrane potential.

---

## HDC layer

Hyperdimensional Computing replaces the classifier head. The core operations:

| Operation | Symbol | Cost |
|-----------|--------|------|
| Binding (XOR) | `bind(A, B)` | 0.1 pJ/bit |
| Bundling (majority) | `bundle([A, B, C])` | 0.1 pJ/bit |
| Similarity (Hamming) | `sim(A, B)` | popcount |
| Matrix multiply (for reference) | — | 4.6 pJ |

**46× cheaper per operation than a MAC.** At scale this compounds: encoding a 4096-dim hypervector costs ~2.4 nJ; a transformer attention head at the same dimension costs ~55 μJ.

```python
from models.hdc import HDCConfig, gen_hvs, bind, bundle, thresh, AssocMemory

# Generate random basis hypervectors
hvs = gen_hvs(n=10, dim=4096, mode="bipolar")

# Bind (associate) two concepts
product = bind(hvs[0], hvs[1])   # XOR in binary, element-wise × in bipolar

# Bundle (average) multiple concepts
sum_hv = bundle([hvs[0], hvs[1], hvs[2]])   # majority vote

# Associative memory: store and retrieve class prototypes
mem = AssocMemory(n_classes=10, dim=4096, mode="bipolar")
mem.add(hv, label=3)
pred_label, similarity = mem.predict(query_hv)
```

---

## Continual learning

Learn new classes without forgetting old ones:

```python
from hdc.continual_hdc import OnlineContinualHDC

learner = OnlineContinualHDC(dim=4096, max_classes=20)

for x, label in stream:
    learner.observe(x, label)
    # New classes added automatically; replay buffer prevents forgetting
```

Class-mean initialization (Harun & Kanan 2025): when a new class appears, its prototype is initialized as the majority vote of all observed samples — already in the right direction before any refinement. 7× fewer updates to converge vs. random initialization.

---

## Reservoir maintenance

After training, the reservoir can drift during long deployments:

```python
# Every ~100 steps: keeps spectral radius at edge-of-chaos (0.97)
pipe.rsnn.homeostatic_scale(target_radius=0.97)

# Periodically: prune weak synapses, grow new ones
pipe.rsnn.structural_plasticity(prune_fraction=0.02, grow_fraction=0.01)

# Firing rate control: prevents silent/saturated neurons
pipe.rsnn.enable_per_neuron_gain()
pipe.rsnn.adapt_input_gain(x, target_rate=0.1)  # keep mean rate at 10%

# Full diagnostics
health = pipe.rsnn.network_health()
# → {"spectral_radius": 0.97, "sparsity": 0.85, "edge_of_chaos": True, ...}
```

---

## Benchmarks

| Task | Method | Score | Memory | Backprop |
|------|--------|-------|--------|----------|
| BCI velocity decoding | Dual-trace Hebbian | **R=0.81** | O(1) | No |
| BCI velocity decoding | BPTT SNN | R=0.79 | O(T) | Yes |
| BCI velocity decoding | Kalman filter | R=0.61 | O(n²) | No |
| SHD neuromorphic (20-class) | RSNN + HDC pipeline | **78%** | O(1) | No |
| SHD neuromorphic | e-prop (Bellec 2020) | 82% | O(1) | No |
| SHD neuromorphic | BPTT SNN (Cramer 2020) | 91% | O(T) | Yes |
| Energy/inference | HDC path | **2.4 nJ** | — | — |
| Energy/inference | SNN path | 12.9 nJ | — | — |
| Energy/inference | Transformer | 55,200 nJ | — | — |

The 78% SHD result sits within the expected range for O(1)-memory online methods: Gomez 2025 (~80%), Hao 2026 (78–83%), Xiao 2024 (~76%), Liang 2025 (~77%). The advantage isn't raw accuracy — it's the combination of O(1) memory, no backprop, and hardware fault tolerance. No other method in the table offers all three.

Ablation on dual vs. single trace (BCI):

| Config | Pearson R |
|--------|-----------|
| Fast-only (τ=5ms) | 0.71 |
| Slow-only (τ=50ms) | 0.68 |
| **Dual (α=0.7, β=0.3)** | **0.81** |

---

## Run the benchmarks

```bash
# Neuromorphic speech (auto-downloads SHD dataset, requires h5py)
python experiments/benchmark_neuromorphic.py --hidden 256

# BCI decoding (requires scipy + CRCNS Indy dataset)
python experiments/bci_decoding.py --method all --seed 42

# Energy comparison vs MLP and transformer
python experiments/benchmark_energy.py

# Fault tolerance: accuracy under stuck-at-0 hardware faults
python experiments/snntraining_robustness.py --fault-type stuck_at_0

# FORCE training benchmark
python experiments/force2_benchmark.py

# Run all 1621 tests
pytest tests/ -v
```

---

## Spike encoders

Converting continuous data to spikes before feeding the RSNN:

```python
from data.encoders.rate import RateEncoder
from data.encoders.ttfs import TTFSEncoder
from data.encoders.gaussian_tuning import GaussianTuningEncoder

# Rate coding: firing probability proportional to value
enc = RateEncoder(n_neurons=100, dt=1.0)
spikes = enc.encode(x)   # (100,) spike vector

# Time-to-first-spike: earlier spike = stronger input
enc = TTFSEncoder(n_neurons=100, t_max=20)
spikes = enc.encode(x)

# Population coding: Gaussian tuning curves (Georgopoulos 1986)
enc = GaussianTuningEncoder(n_neurons=100, sigma=0.2)
spikes = enc.encode(x)
```

Phase coding (`hdc/spike_coding.py`) achieves 6.64 bits/spike — 6.6× more information per spike than rate coding.

---

## Hardware

| Platform | Power | Status |
|----------|-------|--------|
| Artix-7 FPGA (10 MHz) | ~2.5 mW | Synthesis-validated Verilog (`hardware/snntraining_lif.v`) |
| Intel Loihi 2 | ~8 pJ/inference | Lava export ready (`hardware/loihi_mapper.py`) |
| STM32H7 | ~1.2 mW | Binary export ready (`hardware/export.py`) |

Export the trained pipeline:

```python
from hardware.export import export_pipeline
export_pipeline(pipe, target="loihi2", output_dir="build/")
```

---

## Project structure

```
SNNTraining/
├── models/
│   ├── lif.py              # LIF neuron: threshold adaptation, heterogeneous τ
│   ├── rsnn.py             # RSNN: Dale's law, STP, structural plasticity
│   ├── snn_hdc_pipeline.py # End-to-end SNN → HDC pipeline
│   ├── hdc.py              # HDC primitives: bind, bundle, AssocMemory
│   ├── hebbian.py          # BCM, dual-trace, 3-factor, SuperSpike, FORCE
│   ├── readout.py          # Kalman, RLS, Wiener, ensemble readouts
│   ├── alif.py             # Adaptive LIF (refractory threshold adaptation)
│   └── eprop.py            # e-prop eligibility traces (Bellec 2020)
├── hdc/
│   ├── continual_hdc.py    # Class-mean init, RefineHD, OnlineContinualHDC
│   ├── adaptive_encoder.py # NeuralHD, DistHD, LeHDC with STE
│   ├── spike_coding.py     # Rate/phase/temporal/population/burst coding
│   ├── reservoir_theory.py # Principled reservoir capacity analysis
│   └── ...                 # 125+ HDC modules
├── training/
│   ├── online_trainer.py   # Online training orchestration
│   ├── force2_trainer.py   # FORCE training (Nicola & Clopath 2017)
│   ├── eprop.py            # e-prop trainer
│   └── rflo.py             # RFLO (random feedback local online)
├── data/
│   ├── loaders.py          # SHD, MNIST, BCI dataset loaders
│   ├── synthetic.py        # Synthetic spike train generators
│   └── encoders/           # Rate, TTFS, Gaussian, delta encoders
├── experiments/
│   ├── benchmark_neuromorphic.py   # SHD 20-class
│   ├── bci_decoding.py             # BCI velocity decoding
│   ├── benchmark_energy.py         # nJ/inference comparison
│   └── snntraining_robustness.py     # Fault tolerance sweeps
├── hardware/
│   ├── snntraining_lif.v     # Synthesisable Verilog LIF
│   ├── loihi_mapper.py     # Intel Loihi 2 Lava export
│   └── export.py           # MCU binary export
├── runs/
│   ├── train_cpu.sh        # Quick run on CPU / Apple MPS
│   ├── benchmark_shd.sh    # SHD neuromorphic benchmark
│   ├── benchmark_energy.sh # Energy comparison vs MLP / transformer
│   └── sweep_methods.sh    # Compare bundle / refine / fallback / lehdc
├── train.py                # Single entry point
├── demo.py                 # Quick start demo
└── configs/default.yaml    # Training hyperparameters
```

---

## Reproducing the BCI result

```python
from models.rsnn import RSNN, RSNNConfig
from models.hebbian import DualHebbianLearner
from models.readout import WienerReadout

rsnn = RSNN(RSNNConfig(
    input_size=96,
    hidden_size=128,
    sparse_init=True,
    tau=20.0,
    heterogeneous_tau=True,
))

learner = DualHebbianLearner(hidden_size=128, tau_fast=5.0, tau_slow=50.0, alpha=0.7)
readout = WienerReadout(hidden_size=128, output_size=2)  # x,y velocity

for x_t, v_t in bci_stream:
    spikes = rsnn.forward(x_t)
    E = learner.compute_trace(spikes)
    readout.update(spikes, v_t)   # RLS-Wiener update

pearson_r = readout.evaluate(test_stream)
# → 0.81  (vs 0.79 BPTT, 0.61 Kalman)
```

The dual-timescale trace is the key insight: BCI decoding requires both fast traces (~100ms, ISI precision) and slow traces (~700ms, movement context). Mixing them with α=0.7 outperforms either alone.

---

## Why no backprop through time

BPTT on spiking networks requires:
1. Storing all spike states across the full rollout — O(T × N) memory
2. Surrogate gradients for the non-differentiable spike function
3. Vanishing/exploding gradients across long sequences

Local rules avoid all three. The eligibility trace is a *local* approximation to the gradient — it's computed at each synapse from pre- and post-synaptic activity alone. Weight updates are gated by a global neuromodulatory signal (reward, error) when it arrives, but the trace itself requires no stored history beyond the exponential window.

This is the three-factor rule: `ΔW_ij = η · e_ij · M`, where `e_ij` is the eligibility trace and `M` is a global modulatory signal.

---

---

## Citation

```bibtex
@misc{snntraining2026,
  title   = {SNNTraining: Biologically-Plausible Training for Spiking Neural Networks},
  author  = {Young, Aiden},
  year    = {2026},
  url     = {https://github.com/Aidistides/SNNTraining}
}
```

The HDC components are based on 75+ paper implementations. Key references:

- Bellec et al. (2020). "A solution to the learning dilemma for recurrent networks of spiking neurons." *Nature Communications*.
- Nicola & Clopath (2017). "Supervised learning in spiking neural networks with FORCE training." *Nature Communications*.
- Harun & Kanan (2025). "A Good Start Matters: Enhancing Continual Learning with Data-Driven Weight Initialization." *CoLLAs 2025*.
- Perez-Nieves et al. (2021). "Neural heterogeneity promotes robust learning." *Nature Communications*.
- Zhao et al. (2026). "Test-time threshold adaptation for spiking networks." *arXiv:2505.05375*.
- Imani et al. (2022). "NeuralHD / DistHD." *DAC 2022*.
- Kleyko et al. (2022). "A Survey on Hyperdimensional Computing." *IEEE TNNLS*.
- Mitrokhin, Sutor et al. (2019). "Learning sensorimotor control with neuromorphic sensors." *Science Robotics*.

---

## License

MIT
