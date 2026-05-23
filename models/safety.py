"""
Safety, Observability, and Confidence Estimation
================================================
Output bounds checking and d_LIF confidence proxy.

Implements:
- Hard output clips on physical ranges
- Confidence proxy based on d_LIF (surrogate gradient near threshold)
- Per-timestep uncertainty quantification
"""

import torch
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class SafetyConfig:
    """Configuration for safety mechanisms."""
    # Output bounds (per-domain defaults)
    output_min: float = -2.0
    output_max: float = 2.0
    
    # d_LIF confidence parameters
    beta_surrogate: float = 10.0  # Fast sigmoid parameter from paper
    v_threshold: float = 1.0
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.7
    low_confidence_threshold: float = 0.3


class SafeOutputLayer:
    """
    Output layer with hard bounds and confidence estimation.
    
    Clips predictions to physical limits and provides confidence
    scores for downstream arbitration logic.
    """
    
    def __init__(
        self,
        output_dim: int,
        config: Optional[SafetyConfig] = None,
        device: Optional[str] = None
    ):
        self.output_dim = output_dim
        self.config = config or SafetyConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(
        self,
        raw_output: torch.Tensor,
        d_lif_values: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply output bounds and compute confidence.
        
        Args:
            raw_output: Unconstrained network output
            d_lif_values: d_LIF values from LIF layer for confidence
            
        Returns:
            (clipped_output, info_dict)
        """
        # Hard bounds clipping
        clipped = torch.clamp(
            raw_output,
            self.config.output_min,
            self.config.output_max
        )
        
        # Compute confidence if d_LIF values provided
        confidence = self._compute_confidence(d_lif_values)
        
        info = {
            'was_clipped': (raw_output != clipped).any().item(),
            'clip_magnitude': (raw_output - clipped).abs().max().item() if (raw_output != clipped).any() else 0.0,
            'confidence': confidence,
            'confidence_level': self._confidence_level(confidence),
        }
        
        return clipped, info
    
    def _compute_confidence(self, d_lif_values: Optional[torch.Tensor]) -> float:
        """
        Compute confidence score from d_LIF values.
        
        The paper's d_LIF surrogate gradient is naturally high when
        neurons are near threshold (high sensitivity = high certainty).
        When neurons are far from threshold, d_LIF is low (uncertain).
        
        Returns mean d_LIF as confidence proxy [0, 1].
        """
        if d_lif_values is None:
            return 1.0  # No confidence info available
        
        # d_LIF is already in [0, 1] range from fast sigmoid
        mean_d_lif = d_lif_values.mean().item()
        
        # Normalize to [0, 1] confidence score
        confidence = mean_d_lif
        
        return confidence
    
    def _confidence_level(self, confidence: float) -> str:
        """Classify confidence level."""
        if confidence >= self.config.high_confidence_threshold:
            return 'high'
        elif confidence >= self.config.low_confidence_threshold:
            return 'medium'
        else:
            return 'low'


class DLIFComputer:
    """
    Computes d_LIF surrogate gradient for confidence estimation.
    
    Implements the paper's fast sigmoid surrogate:
        d_LIF = 1 / (1 + beta * |v - theta|)
    
    This is not part of backward pass - evaluated locally at each
    neuron and used as postsynaptic sensitivity factor.
    """
    
    def __init__(
        self,
        beta: float = 10.0,
        v_threshold: float = 1.0,
        device: Optional[str] = None
    ):
        self.beta = beta
        self.v_threshold = v_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def compute(self, membrane_potential: torch.Tensor) -> torch.Tensor:
        """
        Compute d_LIF for given membrane potentials.
        
        Args:
            membrane_potential: Membrane potentials v (before spike generation)
            
        Returns:
            d_LIF values in [0, 1], higher = neuron near threshold
        """
        # Fast sigmoid: 1 / (1 + beta * |v - theta|)
        distance_from_threshold = (membrane_potential - self.v_threshold).abs()
        d_lif = 1.0 / (1.0 + self.beta * distance_from_threshold)
        
        return d_lif
    
    def compute_sensitivity(
        self,
        membrane_potential: torch.Tensor,
        error_signal: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute error-modulated sensitivity for three-factor rule.
        
        This is d_LIF * error, used as postsynaptic factor in Hebbian update.
        
        Args:
            membrane_potential: Current membrane potentials
            error_signal: Layer-local error signal
            
        Returns:
            Modulated sensitivity per neuron
        """
        d_lif = self.compute(membrane_potential)
        
        # Postsynaptic factor: d_LIF * |error|
        # Error signal may be multi-dimensional, take mean across output dims
        if error_signal.dim() > 1:
            error_magnitude = error_signal.abs().mean(dim=1)
        else:
            error_magnitude = error_signal.abs()
        
        sensitivity = d_lif * error_magnitude
        
        return sensitivity


class SafetyWrapper:
    """
    Complete safety wrapper for SNN deployment.
    
    Combines output bounds, confidence estimation, and health monitoring.
    """
    
    def __init__(
        self,
        model,
        output_dim: int,
        output_bounds: Optional[Tuple[float, float]] = None,
        device: Optional[str] = None
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Safety config
        config = SafetyConfig()
        if output_bounds:
            config.output_min, config.output_max = output_bounds
        
        self.safe_output = SafeOutputLayer(output_dim, config, device)
        self.d_lif_computer = DLIFComputer(device=device)
        
        # Safety statistics
        self.stats = {
            'total_steps': 0,
            'clips_triggered': 0,
            'low_confidence_steps': 0,
        }
        
    def safe_forward(
        self,
        x: torch.Tensor,
        return_confidence: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with safety checks.
        
        Args:
            x: Input tensor
            return_confidence: Whether to compute confidence
            
        Returns:
            (output, safety_info) where safety_info contains
            confidence, clip status, etc.
        """
        x = x.to(self.device)
        
        # Get raw model output
        # Store membrane potential before spike for d_LIF computation
        if hasattr(self.model, 'lif'):
            # Access LIF layer membrane potential
            v_before = self.model.lif.v.clone()
        else:
            v_before = None
        
        # Forward
        spikes = self.model.forward(x)
        
        # Compute readout
        if hasattr(self.model, 'readout'):
            raw_output = self.model.readout.forward(spikes)
        else:
            # Assume model returns output directly
            raw_output = spikes
        
        # Compute d_LIF for confidence
        d_lif = None
        if return_confidence and v_before is not None:
            d_lif = self.d_lif_computer.compute(v_before)
        
        # Apply safety
        safe_out, safety_info = self.safe_output.forward(raw_output, d_lif)
        
        # Update statistics
        self.stats['total_steps'] += 1
        if safety_info['was_clipped']:
            self.stats['clips_triggered'] += 1
        if safety_info['confidence_level'] == 'low':
            self.stats['low_confidence_steps'] += 1
        
        safety_info['stats'] = self.stats.copy()
        
        return safe_out, safety_info
    
    def check_safety_violation(self, safety_info: Dict) -> bool:
        """
        Check if safety violation requires intervention.
        
        Returns True if operator should be alerted.
        """
        # Alert on:
        # 1. Repeated clipping (suggests model divergence)
        # 2. Sustained low confidence
        
        clip_rate = self.stats['clips_triggered'] / max(1, self.stats['total_steps'])
        low_conf_rate = self.stats['low_confidence_steps'] / max(1, self.stats['total_steps'])
        
        if clip_rate > 0.1:  # >10% of outputs clipped
            return True

        if low_conf_rate > 0.5:  # >50% low confidence
            return True

        return False

    def safety_score(self) -> float:
        """
        Compute a scalar safety health score ∈ [0, 1].

        1.0 = never clipped, always high confidence (healthy)
        0.0 = always clipping, no confidence (unsafe)

        Useful for: automated monitoring dashboards, go/no-go decisions,
        deployment readiness checks.
        """
        n = max(self.stats["total_steps"], 1)
        clip_rate     = self.stats["clips_triggered"]     / n
        low_conf_rate = self.stats["low_confidence_steps"] / n
        return float(1.0 - 0.6 * clip_rate - 0.4 * low_conf_rate)

    def detailed_stats(self) -> Dict:
        """Return detailed safety statistics including derived rates."""
        n = max(self.stats["total_steps"], 1)
        return {
            **self.stats,
            "clip_rate":     round(self.stats["clips_triggered"]     / n, 4),
            "low_conf_rate": round(self.stats["low_confidence_steps"] / n, 4),
            "safety_score":  round(self.safety_score(), 4),
            "healthy":       not self.needs_recalibration(),
        }


def create_domain_specific_safety(domain: str) -> SafetyConfig:
    """
    Create safety configuration for specific deployment domain.
    
    Args:
        domain: One of 'bci', 'uav', 'manufacturing'
        
    Returns:
        SafetyConfig with appropriate bounds
    """
    config = SafetyConfig()
    
    if domain == 'bci':
        # BCI velocity decoding: ±0.5 m/s typical
        config.output_min = -0.5
        config.output_max = 0.5
    elif domain == 'uav':
        # UAV control: ±2.0 m/s velocity, ±45° attitude
        config.output_min = -2.0
        config.output_max = 2.0
    elif domain == 'manufacturing':
        # Manufacturing: torque/force limits (normalized)
        config.output_min = -1.0
        config.output_max = 1.0
    
    return config


if __name__ == "__main__":
    # Test safety components
    print("Safety Component Test")
    print("=" * 50)
    
    # Test d_LIF computation
    d_lif = DLIFComputer(beta=10.0, v_threshold=1.0)
    
    # Neurons at various distances from threshold
    v_test = torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5, 2.0])
    d_values = d_lif.compute(v_test)
    
    print("d_LIF values at different membrane potentials:")
    for v, d in zip(v_test, d_values):
        print(f"  v={v:.1f}: d_LIF={d:.4f}")
    
    # Test safe output layer
    safety = SafeOutputLayer(output_dim=2)
    
    # Normal output
    out1, info1 = safety.forward(torch.tensor([0.5, -0.3]))
    print(f"\nNormal output: {out1.tolist()}, clipped={info1['was_clipped']}")
    
    # Output requiring clipping
    out2, info2 = safety.forward(torch.tensor([5.0, -3.0]))
    print(f"Clipped output: {out2.tolist()}, clipped={info2['was_clipped']}")
    
    print("\nSafety components operational.")
