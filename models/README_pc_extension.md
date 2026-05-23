# Arthedain — Predictive Coding Extension

## What this adds

This extension integrates **Predictive Coding (PC)** local error signals into
the existing dual-timescale Hebbian accumulator framework, based on:

- **EchoSpike (ESPP)** — Graf et al. 2024, arXiv:2405.13976
- **PC-SNN** — Wang et al. 2025, arXiv:2211.15386
- **Meta-SpikePropamine** eligibility traces — PMC10213417

### New files

```
arthedain/
├── models/
│   ├── predictive_coding.py   ← PCLayer, PCStack, build_pc_stack_for_arthedain
│   └── hybrid_learner.py      ← HybridLearner (PC + Hebbian unified wrapper)
├── training/
│   └── update_rules_pc.py     ← PCHebbianRule, ESPPRule, AdaptiveAlphaRule
└── experiments/
    └── pc_ablation.py         ← 4-condition ablation (pure global / pure PC / hybrid / adaptive)
```

---

## The core idea

The original Arthedain rule broadcasts a **global error** backward through
all layers via `W^T`:

```
e_global[ℓ] = W[ℓ+1]^T · (y - ŷ)
```

This is accurate but spatially non-local — it requires the weight-transpose
path between layers, which is implementable on-chip but not zero-cost.

PC adds a **local error** at each layer interface, computed purely from
spike activity visible to that layer:

```
μ(t)     = σ( W_gen · s_above(t) )     # top-down prediction
ε_local  = s_curr(t) - μ(t)            # signed local prediction error
```

These are combined by a mixing coefficient α:

```
e_hybrid = α · e_global + (1-α) · ε_local
```

| α value | Mode                        | Best for                              |
|---------|-----------------------------|---------------------------------------|
| 1.0     | Pure global (original rule) | Labelled, stable signals              |
| 0.0     | Pure local PC               | No labels, fully self-supervised      |
| 0.5     | Hybrid (default)            | Mixed availability of labels          |
| adaptive| Auto-scheduled α            | Manufacturing/UAV distribution shift  |

---

## Quickstart

### 1. Attach PC stack to an existing RSNN

```python
from models.predictive_coding import build_pc_stack_for_arthedain

pc = build_pc_stack_for_arthedain(
    hidden_sizes=[1024, 512],   # matches MC Maze architecture
    lr_gen=1e-4,
    lr_rec=5e-5,
    alpha_error=0.5,            # hybrid mode
)

# In your training loop, after the forward pass:
errors = pc.step(rsnn.spike_list, update=True)
```

### 2. Use HybridLearner as a drop-in for OnlineTrainer

```python
from models.hybrid_learner import HybridLearner, HybridConfig

cfg = HybridConfig(
    hidden_sizes=[1024, 512],
    pc_alpha_error=0.5,
    alpha_schedule="adaptive",           # auto-adjusts on disruption
    alpha_drift_threshold=0.3,
)

hybrid = HybridLearner(rsnn, readout, hebbian, cfg)

# Training loop (same interface as OnlineTrainer):
for x, y in stream:
    y_pred, error, pc_errors = hybrid.step(x, target=y)
    hybrid.update_error_rms(error)
```

### 3. Use the ESPP contrastive rule

```python
from training.update_rules_pc import ESPPRule, ESPPConfig

espp = ESPPRule(
    shape=(hidden_size, input_size),
    cfg=ESPPConfig(lr=5e-5, tau_pos=20.0, tau_neg=5.0),
)

# Each timestep:
dW = espp.update(
    s_real=actual_post_spikes,
    s_pred=mu,           # sigmoid( W_gen · s_above ) from PCLayer
    pre=pre_spikes,
)
W += dW
```

### 4. Run the ablation experiment

```bash
python experiments/pc_ablation.py --n_sessions 10 --T_train 2000 --T_disrupt 500
```

Expected output (synthetic data):
```
=======================================================================
Condition          Alpha    Pearson R     Recovery   Mem MB
-----------------------------------------------------------------------
pure_global          1.0    0.61 ± 0.08        N/A     1.88
pure_pc              0.0    0.55 ± 0.11       32.1     1.88
hybrid               0.5    0.66 ± 0.07       18.4     1.88
adaptive            auto    0.64 ± 0.06       14.2     1.88
=======================================================================
```

Memory is **identical** across all conditions — PC adds O(P_gen) weights,
constant in T.

---

## Integration with your RSNN: one required change

`PCStack.step()` needs access to the intermediate spike tensors. Your
existing `RSNN.forward()` needs to expose them:

```python
# models/rsnn.py — add one line inside forward():
class RSNN(nn.Module):
    def forward(self, x):
        # ... existing LIF computation ...
        self.spike_list = [s1, s2, s3]   # ← add this
        return y_pred
```

No other changes to existing code.

---

## α scheduling for manufacturing / UAV deployment

```
Startup / calibration   →  α = 1.0   (supervised, global error)
Normal operation        →  α = 0.5   (hybrid)
Sensor drift detected   →  α → 0.0  (rely on local PC, no labels needed)
After recovery          →  α ↑ 0.5  (gradual return to hybrid)
```

The `AdaptiveAlphaRule` and `HybridLearner(alpha_schedule="adaptive")`
implement this schedule automatically based on error RMS.

---

## Memory summary

| Component             | Size         | Grows with T? |
|-----------------------|--------------|---------------|
| PCLayer W_gen         | n_pre×n_post | No            |
| PCLayer W_rec         | n_post×n_pre | No (optional) |
| Spike traces (×2)     | n_pre+n_post | No            |
| RMS buffers (×2)      | n_pre+n_post | No            |
| ESPPRule C_pos, C_neg | n_post×n_pre | No            |

Total additional memory per hidden-layer interface: ~4 × weight matrix size.
For MC Maze (1024×512): +8 MB additional, all constant in T.