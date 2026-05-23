"""
dynamics_learning.py
====================
Fast learning through network dynamics without synaptic plasticity.

Implements "learning without weight changes" paradigm where adaptation
occurs through optimization of network initial states and dynamics,
rather than modifying synaptic weights.

Key insight: For a fixed, pre-trained recurrent network (reservoir),
fast task adaptation can be achieved by optimizing:
  - Initial network state (hidden state at t=0)
  - Readout dynamics / gain modulation
  - Attention-like gating over reservoir units

This enables:
  - Millisecond-scale task switching
  - No interference between tasks (weights frozen)
  - Rapid adaptation to new conditions

Based on: "Fast Learning Without Synaptic Plasticity in RNNs" (Nature 2024)
https://www.nature.com/articles/s41598-024-55769-0

Also incorporates ideas from:
- Reservoir Computing (echo state property)
- Liquid State Machines
- Context-dependent computation in biological networks
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import math


@dataclass
class DynamicsLearningConfig:
    """Configuration for dynamics-based fast learning."""
    # Learning rates
    lr_initial_state: float = 0.1       # LR for initial state optimization
    lr_gain: float = 0.01               # LR for gain modulation
    lr_gate: float = 0.05               # LR for activity gating
    
    # Architecture
    n_contexts: int = 4                 # Number of distinct task contexts
    use_gain_modulation: bool = True    # Use neuron gain modulation
    use_activity_gating: bool = True    # Use activity gating (soft attention)
    
    # Optimization
    n_adaptation_steps: int = 50        # Steps for fast adaptation to new task
    adaptation_lr: float = 0.1          # LR during fast adaptation phase
    
    # Regularization
    state_penalty: float = 1e-3         # Penalty on initial state magnitude
    sparsity_target: float = 0.5        # Target sparsity for gates


class GainModulation(nn.Module):
    """
    Neuron-wise gain modulation for context-dependent computation.
    
    Each context learns a multiplicative gain factor for each neuron:
        z'(t) = gain[context] * z(t)
        
    This allows the same reservoir to operate in different "modes"
    without changing recurrent weights.
    """
    
    def __init__(self, n_neurons: int, n_contexts: int):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_contexts = n_contexts
        
        # Gain factors: one per neuron per context
        # Initialize near 1.0 (no modulation)
        self.gains = nn.Parameter(
            torch.ones(n_contexts, n_neurons) + 
            torch.randn(n_contexts, n_neurons) * 0.1
        )
        
    def forward(
        self,
        activity: torch.Tensor,    # (batch, n_neurons)
        context_id: int,         # which context to use
    ) -> torch.Tensor:
        """Apply gain modulation to neural activity."""
        gains = torch.clamp(self.gains[context_id], 0.1, 10.0)  # bounded
        return activity * gains
    
    def forward_batch(
        self,
        activity: torch.Tensor,    # (batch, n_neurons)
        context_ids: torch.Tensor, # (batch,) - per-sample context
    ) -> torch.Tensor:
        """Apply different gains per sample in batch."""
        # Gather gains for each sample
        gains = self.gains[context_ids]  # (batch, n_neurons)
        gains = torch.clamp(gains, 0.1, 10.0)
        return activity * gains


class ActivityGating(nn.Module):
    """
    Soft attention gating over reservoir neurons.
    
    Learns which neurons to "attend" to for each context:
        z'(t) = gate[context] * z(t)
        
    where gate is in [0, 1] (sigmoid). This is more selective than
    gain modulation and can completely silence irrelevant neurons.
    """
    
    def __init__(self, n_neurons: int, n_contexts: int, temperature: float = 0.1):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_contexts = n_contexts
        self.temperature = temperature
        
        # Gate logits (before sigmoid)
        self.gate_logits = nn.Parameter(
            torch.zeros(n_contexts, n_neurons)
        )
        
    def forward(
        self,
        activity: torch.Tensor,    # (batch, n_neurons)
        context_id: int,
    ) -> torch.Tensor:
        """Apply activity gating."""
        gates = torch.sigmoid(self.gate_logits[context_id] / self.temperature)
        return activity * gates
    
    def get_sparsity(self, context_id: int) -> float:
        """Return fraction of gated-off neurons."""
        gates = torch.sigmoid(self.gate_logits[context_id])
        return (gates < 0.5).float().mean().item()


class InitialStateOptimizer(nn.Module):
    """
    Optimizable initial states for fast task adaptation.
    
    Instead of starting from zero initial state, learn and optimize
    task-specific initial conditions that put the network in the
    right "starting position" for each task.
    
    This is particularly powerful for sequence tasks where the
    initial context matters.
    """
    
    def __init__(self, n_neurons: int, n_contexts: int):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_contexts = n_contexts
        
        # Initial states: one per context
        self.initial_states = nn.Parameter(
            torch.zeros(n_contexts, n_neurons)
        )
        
        # Also store current runtime state (not a parameter)
        self.register_buffer("runtime_state", torch.zeros(n_neurons))
        self.current_context = 0
        
    def get_initial_state(self, context_id: Optional[int] = None) -> torch.Tensor:
        """Get initial state for a context."""
        if context_id is None:
            context_id = self.current_context
        return self.initial_states[context_id]
    
    def set_context(self, context_id: int):
        """Set current context and reset runtime state."""
        self.current_context = context_id
        self.runtime_state.zero_()
        
    def reset_runtime(self):
        """Reset runtime state to initial for current context."""
        self.runtime_state.copy_(self.get_initial_state())
        
    def fast_adapt(
        self,
        rsnn: nn.Module,
        data_stream: List[Tuple[torch.Tensor, torch.Tensor]],
        n_steps: int = 50,
        lr: float = 0.1,
        context_id: int = 0,
    ) -> float:
        """
        Fast adaptation to a new task by optimizing initial state.
        
        This is the key "fast learning" capability - given a few examples
        of a new task, optimize the initial state (keeping weights frozen).
        
        Args:
            rsnn: Frozen recurrent SNN
            data_stream: List of (input, target) pairs for adaptation
            n_steps: Number of optimization steps
            lr: Learning rate for adaptation
            context_id: Which context slot to adapt
            
        Returns:
            final_loss: Loss after adaptation
        """
        optimizer = torch.optim.Adam(
            [self.initial_states[context_id]],
            lr=lr
        )
        
        final_loss = float('inf')
        
        for step in range(n_steps):
            total_loss = 0.0
            
            for x, target in data_stream:
                optimizer.zero_grad()
                
                # Reset to current initial state
                self.reset_runtime()
                if hasattr(rsnn, 'reset'):
                    rsnn.reset()
                if hasattr(rsnn, 'lif'):
                    rsnn.lif.v = self.runtime_state.clone()
                
                # Forward
                y_pred = rsnn(x)
                
                # Loss
                loss = torch.nn.functional.mse_loss(y_pred, target)
                
                # Backprop through initial state (weights frozen!)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            final_loss = total_loss / len(data_stream)
            
            # Early stopping if converged
            if step > 10 and final_loss < 0.01:
                break
        
        return final_loss


class DynamicsLearner(nn.Module):
    """
    Complete dynamics-based learning system.
    
    Combines gain modulation, activity gating, and initial state
    optimization to enable fast learning without weight changes.
    
    Usage:
        1. Pre-train or initialize a recurrent SNN (frozen during use)
        2. Create DynamicsLearner for the SNN
        3. For each task:
           - If known task: use stored context
           - If new task: fast_adapt() to learn initial state
        4. Run inference with modulated dynamics
    """
    
    def __init__(
        self,
        rsnn: nn.Module,
        readout: nn.Module,
        cfg: DynamicsLearningConfig,
    ):
        super().__init__()
        self.rsnn = rsnn
        self.readout = readout
        self.cfg = cfg
        
        # Freeze RSNN weights
        for param in rsnn.parameters():
            param.requires_grad = False
        
        # Get dimensions
        self.hidden_size = getattr(rsnn, 'hidden_size', rsnn.W_rec.shape[0])
        self.input_size = getattr(rsnn, 'input_size', rsnn.W_in.shape[1])
        
        # Initialize modulation modules
        self.gain_mod = GainModulation(self.hidden_size, cfg.n_contexts) \
                       if cfg.use_gain_modulation else None
        
        self.activity_gate = ActivityGating(self.hidden_size, cfg.n_contexts) \
                            if cfg.use_activity_gating else None
        
        self.initial_state_opt = InitialStateOptimizer(self.hidden_size, cfg.n_contexts)
        
        # Current context tracking
        self.current_context = 0
        self._adapted_contexts = set()
        
    def forward(
        self,
        x: torch.Tensor,
        context_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with dynamics modulation.
        
        Args:
            x: Input (batch, input_size)
            context_id: Which context to use (None = current)
            
        Returns:
            y_pred: Prediction (batch, output_size)
        """
        if context_id is None:
            context_id = self.current_context
        
        # Set initial state
        initial_v = self.initial_state_opt.get_initial_state(context_id)
        if hasattr(self.rsnn, 'lif'):
            self.rsnn.lif.v = initial_v.clone()
        
        # Forward through RSNN
        spikes = self.rsnn(x)
        
        # Apply dynamics modulation
        if self.gain_mod is not None:
            spikes = self.gain_mod(spikes, context_id)
        
        if self.activity_gate is not None:
            spikes = self.activity_gate(spikes, context_id)
        
        # Readout
        y_pred = self.readout(spikes)
        
        return y_pred
    
    def fast_adapt_to_task(
        self,
        task_data: List[Tuple[torch.Tensor, torch.Tensor]],
        context_id: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Fast adaptation to a new task.
        
        Optimizes initial state (and optionally gains/gates) for the
        given task data, while keeping recurrent weights frozen.
        
        Args:
            task_data: List of (input, target) examples
            context_id: Which context slot to use (auto-assigned if None)
            verbose: Print adaptation progress
            
        Returns:
            stats: Dict with final_loss, n_steps, etc.
        """
        # Assign context if not specified
        if context_id is None:
            context_id = self._assign_new_context()
        
        # Ensure data is on right device
        device = next(self.rsnn.parameters()).device
        task_data = [
            (x.to(device), y.to(device)) for x, y in task_data
        ]
        
        # Optimize initial state
        final_loss = self.initial_state_opt.fast_adapt(
            rsnn=self.rsnn,
            data_stream=task_data,
            n_steps=self.cfg.n_adaptation_steps,
            lr=self.cfg.adaptation_lr,
            context_id=context_id,
        )
        
        self._adapted_contexts.add(context_id)
        
        stats = {
            'final_loss': final_loss,
            'context_id': context_id,
            'n_examples': len(task_data),
        }
        
        if verbose:
            print(f"Fast adaptation complete: loss={final_loss:.4f}, context={context_id}")
        
        return stats
    
    def _assign_new_context(self) -> int:
        """Find next available context slot."""
        for i in range(self.cfg.n_contexts):
            if i not in self._adapted_contexts:
                return i
        # All contexts used - reuse least recently used (simple: use 0)
        return 0
    
    def set_context(self, context_id: int):
        """Switch to a different task context."""
        self.current_context = context_id
        self.initial_state_opt.set_context(context_id)
    
    def get_context_info(self, context_id: int) -> Dict[str, any]:
        """Get information about a learned context."""
        info = {
            'context_id': context_id,
            'is_adapted': context_id in self._adapted_contexts,
            'initial_state_norm': self.initial_state_opt.initial_states[context_id].norm().item(),
        }
        
        if self.activity_gate is not None:
            info['gate_sparsity'] = self.activity_gate.get_sparsity(context_id)
        
        return info
    
    def save_contexts(self) -> Dict[str, torch.Tensor]:
        """Save all learned contexts for later restoration."""
        state = {
            'initial_states': self.initial_state_opt.initial_states.data.clone(),
            'adapted_contexts': list(self._adapted_contexts),
        }
        
        if self.gain_mod is not None:
            state['gains'] = self.gain_mod.gains.data.clone()
        
        if self.activity_gate is not None:
            state['gate_logits'] = self.activity_gate.gate_logits.data.clone()
        
        return state
    
    def load_contexts(self, state: Dict[str, torch.Tensor]):
        """Restore learned contexts."""
        self.initial_state_opt.initial_states.data.copy_(state['initial_states'])
        self._adapted_contexts = set(state['adapted_contexts'])
        
        if 'gains' in state and self.gain_mod is not None:
            self.gain_mod.gains.data.copy_(state['gains'])
        
        if 'gate_logits' in state and self.activity_gate is not None:
            self.activity_gate.gate_logits.data.copy_(state['gate_logits'])


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def make_dynamics_learner(
    rsnn: nn.Module,
    readout: nn.Module,
    n_contexts: int = 4,
    fast_adaptation_steps: int = 50,
) -> DynamicsLearner:
    """
    Factory function for dynamics-based learner.
    
    Args:
        rsnn: Pre-trained or randomly initialized recurrent SNN (will be frozen)
        readout: Readout layer (can be trained or frozen)
        n_contexts: Number of task contexts to support
        fast_adaptation_steps: Steps for fast adaptation to new tasks
    """
    cfg = DynamicsLearningConfig(
        n_contexts=n_contexts,
        n_adaptation_steps=fast_adaptation_steps,
        use_gain_modulation=True,
        use_activity_gating=True,
    )
    return DynamicsLearner(rsnn, readout, cfg)
