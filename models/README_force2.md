# FORCE2: Enhanced FORCE Training (Nicola & Clopath 2017)

This module implements improvements from **"Supervised learning in spiking neural networks with FORCE training"** (Nicola & Clopath, Nature Communications 2017).

## Key Improvements from the Paper

### 1. Chaotic Regime Initialization

The paper's key insight: Networks initialized in a **chaotic regime** (spectral radius > 1) have richer dynamics and can learn complex temporal patterns.

```python
from models.force_enhanced import ChaoticInitializer, ChaoticInitConfig

# Initialize with target spectral radius of 1.5 (chaotic regime)
init = ChaoticInitializer(ChaoticInitConfig(target_radius=1.5))
W_rec = init.initialize(n_neurons=1000, device='cpu')

print(f"Actual spectral radius: {init.compute_spectral_radius(W_rec):.3f}")
```

**Why it matters:**
- Spectral radius < 1: Stable, fixed-point dynamics → poor learning capacity
- Spectral radius > 1: Chaotic dynamics → rich temporal representations
- The paper recommends 1.0-1.8 depending on task complexity

### 2. Multi-Timescale Synaptic Dynamics

Biological synapses operate at multiple timescales (AMPA ~3ms, NMDA ~100ms, GABA-B ~300ms). The paper shows combining these improves learning.

```python
from models.force_enhanced import MultiTimescaleSynapses, MultiTimescaleSynapseConfig

synapses = MultiTimescaleSynapses(
    n_neurons=1000,
    cfg=MultiTimescaleSynapseConfig(
        tau_fast=3.0,      # AMPA-like (ms)
        tau_slow=100.0,    # NMDA-like (ms)
        tau_ultra=300.0,   # GABA-B-like (ms)
        alpha_fast=0.5,    # Fast contribution
        alpha_slow=0.4,   # Slow contribution
        alpha_ultra=0.1,   # Ultra-slow contribution
    ),
    device='cpu'
)

# Each step updates filtered spike trains
s_filtered = synapses.step(spikes)
```

### 3. Sparse Structured Connectivity

The paper uses sparse connectivity with only a fraction of weights trainable (typically 10%). This provides:
- Better stability during learning
- Reduced computational cost
- Reservoir-like dynamics from fixed weights

```python
from models.force_enhanced import SparseFixedConnectivity

sparse_conn = SparseFixedConnectivity(
    n_neurons=1000,
    connectivity_p=0.1,           # 10% connectivity
    trainable_fraction=0.1,        # 10% of those are trainable
    device='cpu'
)

# Total trainable: ~1% of all possible connections
print(f"Trainable: {sparse_conn.get_trainable_count()}")
print(f"Total: {sparse_conn.get_total_count()}")
```

### 4. Pattern Generation Utilities

The paper tests on various patterns - oscillators, chaotic attractors, even songs.

```python
from models.force_enhanced import PatternGenerator

# Simple oscillator (Hz)
osc = PatternGenerator.generate_oscillator(freq=2.0, n_steps=1000)

# Coupled oscillators
coupled = PatternGenerator.generate_coupled_oscillators(
    freqs=[1.0, 3.0, 5.0],
    amplitudes=[1.0, 0.5, 0.25],
    n_steps=3000
)

# Lorenz chaotic attractor (classic FORCE test)
lorenz = PatternGenerator.generate_lorenz_attractor(n_steps=5000)

# Rossler attractor
rossler = PatternGenerator.generate_rossler_attractor(n_steps=5000)

# "Ode to Joy" melody (as in paper)
melody = PatternGenerator.generate_ode_to_joy(n_steps=4000)
```

## Using FORCE2 Trainer

The `FORCE2Trainer` integrates all these improvements:

```python
from training.force2_trainer import FORCE2Trainer, FORCE2Config, make_force2_trainer_for_oscillator

# Create trainer for oscillator learning
trainer = make_force2_trainer_for_oscillator(
    freq=2.0,
    n_neurons=1000,
    device='cpu'
)

# Train on pattern
from models.force_enhanced import PatternGenerator
pattern = PatternGenerator.generate_oscillator(freq=2.0, n_steps=2000)

for t in range(len(pattern)):
    y_pred, error = trainer.train_step(
        x=torch.zeros(1),  # No input
        target=pattern[t]
    )

# Test
print(f"Stats: {trainer.get_stats()}")
```

### Full Configuration

```python
from training.force2_trainer import FORCE2Trainer, FORCE2Config
from models.force_enhanced import ChaoticInitConfig, MultiTimescaleSynapseConfig

cfg = FORCE2Config(
    n_neurons=2000,
    n_outputs=3,
    
    # Chaotic initialization
    chaotic_cfg=ChaoticInitConfig(
        target_radius=1.8,        # Higher for complex tasks
        connectivity_p=0.15,      # Sparsity
        use_dales_principle=True,  # E/I separation
    ),
    
    # Multi-timescale synapses
    multi_tau_cfg=MultiTimescaleSynapseConfig(
        tau_fast=5.0,
        tau_slow=100.0,
        tau_ultra=300.0,
        alpha_fast=0.4,
        alpha_slow=0.4,
        alpha_ultra=0.2,
    ),
    
    # RLS parameters
    alpha_rls=1.0,
    forgetting_factor=0.9995,
    
    # Training modes
    train_readout=True,       # Always train readout
    train_recurrent=False,    # Optional: train recurrent weights
    train_input=False,        # Optional: train input weights
    
    # Efficiency
    skip_below_error=0.001,  # Skip RLS when error small
)

trainer = FORCE2Trainer(cfg, device='cpu')
```

## Benchmark Results

Run the benchmark to compare against paper results:

```bash
python experiments/force2_benchmark.py --test all --save_results results/force2.json
```

Expected results (from paper):
- Simple oscillator: R² > 0.95
- Coupled oscillators: R² > 0.90
- Lorenz attractor: R² > 0.85
- Ode to Joy: R² > 0.80 per component

## Paper Reference

```bibtex
@article{nicola2017supervised,
  title={Supervised learning in spiking neural networks with FORCE training},
  author={Nicola, Wilten and Clopath, Claudia},
  journal={Nature Communications},
  volume={8},
  pages={2208},
  year={2017},
  publisher={Nature Publishing Group}
}
```

## Implementation Notes

1. **Spectral Radius Scaling**: We scale the recurrent weight matrix to achieve the target spectral radius by computing eigenvalues and rescaling.

2. **Dale's Principle**: When enabled, neurons are designated as excitatory (positive outgoing weights) or inhibitory (negative outgoing weights) based on the E/I balance ratio.

3. **RLS Updates**: The trainer uses Recursive Least Squares for readout weights, which converges faster than gradient descent.

4. **Skip Logic**: RLS updates are skipped when error is below threshold to save computation.

## Comparison to Original SNNTraining

| Feature | Original SNNTraining | FORCE2 |
|---------|-------------------|--------|
| Learning Rule | Dual-timescale Hebbian | RLS + Chaos |
| Spectral Radius | Usually < 1 | 1.0-1.8 |
| Synapses | Single timescale | Multi-timescale |
| Connectivity | Dense | Sparse, structured |
| Best For | BCI decoding | Temporal patterns |
| Memory | O(P) eligibility | O(N²) RLS |

Use **SNNTraining (Hebbian)** for online BCI decoding with O(1) memory.
Use **FORCE2** for learning complex temporal patterns (oscillators, chaos, songs).
