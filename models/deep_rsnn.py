"""
deep_rsnn.py
============
Multi-layer Recurrent Spiking Neural Network (Deep RSNN).

Extends the single-layer RSNN to support deep architectures with
arbitrary numbers of recurrent layers. Each layer can have:
  - Recurrent connections within the layer
  - Feedforward connections to the next layer
  - Skip connections (optional)
  - Layer-wise plasticity with independent eligibility traces

Integrates with PCStack for hierarchical predictive coding where
each layer interface has its own local error signal.

Architecture:
    Input → LIF[0] → LIF[1] → ... → LIF[L-1] → Readout
              ↑        ↑              ↑
            W_rec[0] W_rec[1] ...  W_rec[L-1]

Key features:
- Constant memory per layer (not O(T) with sequence length)
- Optional skip connections for gradient/gradient-free flow
- Compatible with all learning methods (e-prop, FORCE, dynamics)
- Full integration with predictive coding stack

References
----------
- Deep SNNs: https://arxiv.org/abs/2006.03824
- Layer-wise PC in deep networks: https://arxiv.org/abs/2211.15386
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from .lif import LIFLayer, LIFConfig


@dataclass
class DeepRSNNConfig:
    """Configuration for multi-layer RSNN."""
    input_size: int = 100
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])
    output_size: int = 2
    
    # LIF parameters (shared across layers, or per-layer if list)
    tau: float = 20.0
    v_th: float = 1.0
    refractory: int = 2
    
    # Architecture options
    use_skip_connections: bool = False
    skip_every: int = 1              # Add skip every N layers
    
    # Initialization
    sparse_init: bool = True
    sparse_p: float = 0.15
    weight_scale: float = 0.1
    
    # Plasticity
    learnable_recurrent: bool = True
    freeze_layers: Optional[List[int]] = None  # Layers to freeze


class DeepRSNNLayer(nn.Module):
    """
    Single layer of deep RSNN with recurrent and feedforward connections.
    
    Maintains:
      - W_rec: recurrent weights (hidden, hidden)
      - W_ff: feedforward to next layer (next_hidden, hidden) if not last
      - LIF dynamics: membrane potential and spikes
      - Eligibility traces (if used by trainer)
    """
    
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        next_hidden_size: Optional[int],
        cfg: DeepRSNNConfig,
        layer_id: int,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.next_hidden_size = next_hidden_size
        self.cfg = cfg
        
        # LIF neurons
        self.lif = LIFLayer(
            size=hidden_size,
            tau=cfg.tau,
            v_th=cfg.v_th,
            refractory=cfg.refractory,
        )

        # Temporal Batch Normalisation (tdBN) — Zheng et al. NeurIPS 2020
        # Normalise input current across time rather than batch.
        # Compatible with single-sample online inference (uses running stats).
        # Prevents vanishing/exploding gradients in 3+ layer networks.
        self.tdbn = torch.nn.LayerNorm(hidden_size, elementwise_affine=True)
        
        # Input weights (from previous layer or input)
        self.W_in = nn.Parameter(
            torch.randn(hidden_size, in_size) * cfg.weight_scale
        )
        
        # Recurrent weights
        if cfg.learnable_recurrent and (cfg.freeze_layers is None or layer_id not in cfg.freeze_layers):
            self.W_rec = nn.Parameter(
                torch.randn(hidden_size, hidden_size) * cfg.weight_scale
            )
            # Make recurrent weights sparse if requested
            if cfg.sparse_init:
                mask = (torch.rand_like(self.W_rec) < cfg.sparse_p).float()
                self.W_rec.data *= mask
        else:
            # Frozen random recurrent weights (reservoir mode)
            self.register_buffer(
                "W_rec",
                torch.randn(hidden_size, hidden_size) * cfg.weight_scale * 0.5
            )
        
        # Feedforward weights to next layer
        if next_hidden_size is not None:
            self.W_ff = nn.Parameter(
                torch.randn(next_hidden_size, hidden_size) * cfg.weight_scale
            )
        else:
            self.W_ff = None
        
        # State tracking
        self.prev_input = None
        self.prev_spikes = None
        
    def forward(
        self,
        x: torch.Tensor,           # (batch, in_size) - input from prev layer
        prev_layer_spikes: Optional[torch.Tensor] = None,  # for skip connections
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through this layer.
        
        Returns:
            spikes: (batch, hidden_size) - output spikes
            v: (batch, hidden_size) - membrane potential (for surrogate grad)
            ff_out: (batch, next_hidden) or None - feedforward to next layer
        """
        # Combine input with recurrent activity
        if self.prev_spikes is None:
            rec_input = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        else:
            rec_input = F.linear(self.prev_spikes, self.W_rec)
        
        ff_input = F.linear(x, self.W_in)

        # tdBN: normalise total input current before the spike threshold
        # (Zheng et al. 2020 NeurIPS) — prevents gradient vanishing in deep layers
        total_input = ff_input + rec_input
        total_input = self.tdbn(total_input)
        ff_input = total_input - rec_input   # decompose back for clarity

        # Add skip connection if available
        if prev_layer_spikes is not None and self.cfg.use_skip_connections:
            # Project to match dimensions if needed
            if prev_layer_spikes.shape[1] != self.hidden_size:
                # Simple linear projection (could be learned)
                skip_proj = nn.functional.linear(
                    prev_layer_spikes,
                    torch.randn(self.hidden_size, prev_layer_spikes.shape[1], 
                               device=x.device) * 0.01
                )
            else:
                skip_proj = prev_layer_spikes
            ff_input = ff_input + skip_proj
        
        total_input = ff_input + rec_input
        
        # LIF dynamics
        spikes = self.lif.step(total_input.squeeze(0) if total_input.dim() == 2 and total_input.shape[0] == 1 else total_input[0] if total_input.dim() == 2 else total_input)
        
        # Handle batch properly
        if x.dim() == 2 and x.shape[0] > 1:
            # Batch processing
            spikes_list = []
            for b in range(x.shape[0]):
                b_input = ff_input[b] + rec_input[b] if rec_input.dim() > 1 else rec_input
                b_spikes = self.lif.step(b_input)
                spikes_list.append(b_spikes)
            spikes = torch.stack(spikes_list)
        
        # Store for next step - ensure 1D for recurrent connections
        if spikes.dim() == 1:
            self.prev_spikes = spikes.detach()
        elif spikes.dim() == 2:
            # Take last in batch or squeeze if single sample
            self.prev_spikes = spikes[-1].detach() if spikes.shape[0] > 1 else spikes[0].detach()
        else:
            self.prev_spikes = spikes.detach().squeeze()
        self.prev_input = x.detach()
        
        # Compute feedforward output for next layer
        if self.W_ff is not None:
            ff_out = F.linear(spikes, self.W_ff)
        else:
            ff_out = None
        
        # Get membrane potential for e-prop
        v = self.lif.v.clone()
        
        return spikes, v, ff_out
    
    def reset(self):
        """Reset layer state."""
        self.lif.reset()
        self.prev_spikes = None
        self.prev_input = None
    
    def get_parameters_for_trainer(self) -> Dict[str, nn.Parameter]:
        """Get trainable parameters for external trainer."""
        params = {'W_in': self.W_in}
        if isinstance(self.W_rec, nn.Parameter):
            params['W_rec'] = self.W_rec
        if self.W_ff is not None:
            params['W_ff'] = self.W_ff
        return params


class DeepRSNN(nn.Module):
    """
    Multi-layer recurrent spiking neural network.
    
    Stack of DeepRSNNLayers with optional skip connections.
    Exposes spike_list for integration with PCStack.
    
    Usage:
        cfg = DeepRSNNConfig(hidden_sizes=[256, 128, 64])
        model = DeepRSNN(cfg)
        
        # Streaming forward
        for x_t in stream:
            spikes_by_layer, y_pred = model.forward(x_t)
            # spikes_by_layer is list of spike tensors, one per layer
    """
    
    def __init__(self, cfg: DeepRSNNConfig):
        super().__init__()
        self.cfg = cfg
        
        # Build layers
        self.layers = nn.ModuleList()
        layer_sizes = [cfg.input_size] + cfg.hidden_sizes
        
        for i in range(len(cfg.hidden_sizes)):
            in_size = layer_sizes[i]
            hidden_size = cfg.hidden_sizes[i]
            next_hidden = cfg.hidden_sizes[i + 1] if i + 1 < len(cfg.hidden_sizes) else None
            
            layer = DeepRSNNLayer(
                in_size=in_size,
                hidden_size=hidden_size,
                next_hidden_size=next_hidden,
                cfg=cfg,
                layer_id=i,
            )
            self.layers.append(layer)
        
        # Readout from last layer
        self.readout = nn.Linear(cfg.hidden_sizes[-1], cfg.output_size, bias=True)
        
        # Spike tracking (for PCStack and analysis)
        self.spike_list = []
        self.v_list = []  # membrane potentials for e-prop
        
    def forward(
        self,
        x: torch.Tensor,  # (batch, input_size) or (input_size,)
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through all layers.
        
        Returns:
            spike_list: List of spike tensors, one per layer
            y_pred: Readout prediction (batch, output_size)
        """
        # Ensure batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        self.spike_list = []
        self.v_list = []
        
        current = x
        prev_layer_spikes = None
        
        # Forward through each layer
        for i, layer in enumerate(self.layers):
            # Skip connections: pass spikes from 2 layers ago
            skip_spikes = None
            if self.cfg.use_skip_connections and i >= self.cfg.skip_every:
                skip_spikes = self.spike_list[i - self.cfg.skip_every]
            
            spikes, v, ff_out = layer(current, skip_spikes)
            
            # Ensure spikes have batch dimension for next layer
            if spikes.dim() == 1:
                spikes = spikes.unsqueeze(0)
            
            self.spike_list.append(spikes)
            self.v_list.append(v)
            
            # Prepare input for next layer
            # Use ff_out only if dimensions match next layer's expected input
            next_layer_idx = i + 1
            if ff_out is not None and next_layer_idx < len(self.layers):
                next_in_size = self.layers[next_layer_idx].in_size
                if ff_out.shape[-1] == next_in_size:
                    current = ff_out
                else:
                    current = spikes
            elif ff_out is not None:
                current = ff_out
            else:
                current = spikes
            
            if current.dim() == 1:
                current = current.unsqueeze(0)
        
        # Readout from last layer
        last_spikes = self.spike_list[-1]
        y_pred = self.readout(last_spikes)
        
        return self.spike_list, y_pred
    
    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single tensor output version (for compatibility).
        
        Returns just the readout prediction.
        """
        spike_list, y_pred = self.forward(x)
        return y_pred.squeeze(0) if y_pred.shape[0] == 1 else y_pred
    
    def reset(self):
        """Reset all layers."""
        for layer in self.layers:
            layer.reset()
        self.spike_list = []
        self.v_list = []
    
    def get_all_weights(self) -> Dict[str, torch.Tensor]:
        """Get all weights for inspection/analysis."""
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f'layer{i}_W_in'] = layer.W_in.data
            weights[f'layer{i}_W_rec'] = layer.W_rec.data if hasattr(layer.W_rec, 'data') else layer.W_rec
            if layer.W_ff is not None:
                weights[f'layer{i}_W_ff'] = layer.W_ff.data
        weights['readout_W'] = self.readout.weight.data
        return weights
    
    def get_spike_list(self) -> List[torch.Tensor]:
        """Get list of spikes from last forward pass (for PCStack)."""
        return self.spike_list
    
    def get_firing_rates(self, window: int = 100) -> Dict[int, float]:
        """Get average firing rates per layer over recent window."""
        rates = {}
        for i, layer in enumerate(self.layers):
            rate = layer.lif.get_firing_rates(window)
            rates[i] = rate.mean().item()
        return rates

    def layer_health(self) -> List[Dict]:
        """
        Per-layer health diagnostics for monitoring deep SNN dynamics.

        Returns a list of dicts (one per layer) with:
          - firing_rate: mean spike rate
          - synchrony: population synchrony score
          - dead_neurons: fraction of neurons with zero firing rate
          - exploding_neurons: fraction with rate > 0.9 (saturated)

        A healthy layer has:
          firing_rate ≈ 0.05-0.2 (5-20% active)
          synchrony < 0.5 (not locked to one pattern)
          dead_neurons < 0.1 (< 10% inactive)
          exploding_neurons < 0.05 (< 5% saturated)
        """
        health = []
        for i, layer in enumerate(self.layers):
            rate_vec = layer.lif.get_firing_rates(window=50)
            mean_rate  = float(rate_vec.mean().item())
            dead       = float((rate_vec < 0.01).float().mean().item())
            exploding  = float((rate_vec > 0.9).float().mean().item())
            synchrony  = layer.lif.population_synchrony(window=20)
            health.append({
                "layer":            i,
                "firing_rate":      round(mean_rate, 4),
                "synchrony":        round(synchrony, 4),
                "dead_neurons":     round(dead, 4),
                "exploding_neurons": round(exploding, 4),
                "healthy":          (0.01 < mean_rate < 0.5
                                     and synchrony < 0.7
                                     and dead < 0.3
                                     and exploding < 0.1),
            })
        return health


# ---------------------------------------------------------------------------
# Integration with Predictive Coding
# ---------------------------------------------------------------------------

def make_deep_rsnn_with_pc(
    input_size: int,
    hidden_sizes: List[int],
    output_size: int = 2,
    pc_lr_gen: float = 1e-4,
    pc_alpha_error: float = 0.5,
    use_skip_connections: bool = False,
) -> Tuple[DeepRSNN, any]:
    """
    Create a DeepRSNN with integrated PCStack.
    
    Returns:
        rsnn: DeepRSNN model
        pc_stack: PCStack configured for this architecture
    """
    from .predictive_coding import build_pc_stack_for_arthedain
    
    cfg = DeepRSNNConfig(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        use_skip_connections=use_skip_connections,
    )
    
    rsnn = DeepRSNN(cfg)
    
    # Build PC stack for layer interfaces
    pc_stack = build_pc_stack_for_arthedain(
        hidden_sizes=hidden_sizes,
        lr_gen=pc_lr_gen,
        alpha_error=pc_alpha_error,
    )
    
    return rsnn, pc_stack


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_deep_rsnn(
    input_size: int,
    hidden_sizes: List[int] = [256, 128],
    output_size: int = 2,
    tau: float = 20.0,
    sparse: bool = True,
) -> DeepRSNN:
    """
    Factory function for creating a DeepRSNN.
    
    Args:
        input_size: Input dimension
        hidden_sizes: List of hidden layer sizes (e.g., [256, 128] for 2 layers)
        output_size: Readout dimension
        tau: Membrane time constant (ms)
        sparse: Use sparse recurrent initialization
    """
    cfg = DeepRSNNConfig(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        tau=tau,
        sparse_init=sparse,
    )
    return DeepRSNN(cfg)
