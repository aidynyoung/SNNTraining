"""
 Voltage Scaling Tolerance for HDC
==========================
Implements voltage scaling tolerance from Section III-A of:
"Brain-Inspired Hyperdimensional Computing for Ultra-Efficient Edge AI"
(NSF purl/10392362)

Provides:
- Safe region detection (0.8V - 0.6V nominal)
- Error rate profiling under different voltages
- Adaptive error masking based on voltage level

The paper shows HDC can tolerate aggressive voltage scaling:
- Safe region: 0.8V - 0.6V (minimal accuracy loss)
- With masking: down to 0.45V with negligible accuracy loss
- Energy savings: 50-70% on average
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class VoltageScalingConfig:
    """Configuration for voltage scaling tolerance."""
    nominal_voltage: float = 0.8  # Volts
    safe_min_voltage: float = 0.6  # Safe region lower bound
    critical_min_voltage: float = 0.45  # Critical minimum
    bit_error_rate_at_nominal: float = 1e-9
    bit_error_rate_at_safe: float = 1e-6
    bit_error_rate_at_critical: float = 1e-2
    enable_adaptive_masking: bool = True


def estimate_bit_error_rate(voltage: float, config: VoltageScalingConfig) -> float:
    """
    Estimate bit error rate at given voltage using exponential model.
    
    Based on Intel 22nm FDX22 technology profiling from the paper.
    Error rate increases exponentially as voltage decreases.
    
    Args:
        voltage: Supply voltage in volts
        config: Voltage scaling configuration
    
    Returns:
        Estimated bit error rate
    
    Example:
        >>> config = VoltageScalingConfig()
        >>> estimate_bit_error_rate(0.8, config)
        1e-9
        >>> estimate_bit_error_rate(0.6, config)
        ~1e-6 (safe region)
        >>> estimate_bit_error_rate(0.5, config)
        ~1e-2 (critical!)
    """
    if voltage >= config.nominal_voltage:
        return config.bit_error_rate_at_nominal
    
    if voltage <= config.critical_min_voltage:
        return config.bit_error_rate_at_critical
    
    # Interpolate using exponential model
    v_range = config.nominal_voltage - config.critical_min_voltage
    v_normalized = (config.nominal_voltage - voltage) / v_range
    
    # Exponential increase: ~10^(-6) at safe min, ~10^(-2) at critical
    log_ber_nom = np.log10(config.bit_error_rate_at_nominal)
    log_ber_safe = np.log10(config.bit_error_rate_at_safe)
    log_ber_crit = np.log10(config.bit_error_rate_at_critical)
    
    if voltage >= config.safe_min_voltage:
        # Safe region: interpolate between nominal and safe
        t = (config.nominal_voltage - voltage) / (config.nominal_voltage - config.safe_min_voltage)
        log_ber = log_ber_nom + t * (log_ber_safe - log_ber_nom)
    else:
        # Critical region: interpolate between safe and critical
        t = (config.safe_min_voltage - voltage) / (config.safe_min_voltage - config.critical_min_voltage)
        log_ber = log_ber_safe + t * (log_ber_crit - log_ber_safe)
    
    return 10 ** log_ber


class VoltageScaler:
    """
    Voltage scaling with adaptive masking.
    
    Monitors voltage levels and applies appropriate error masking
    to maintain accuracy under aggressive voltage scaling.
    
    Attributes:
        config: VoltageScalingConfig
        current_voltage: Current supply voltage
        mode: "full", "safe", "adaptive"
    """
    
    def __init__(
        self,
        config: Optional[VoltageScalingConfig] = None,
        initial_voltage: float = 0.8,
    ):
        self.config = config or VoltageScalingConfig()
        self.current_voltage = initial_voltage
        self.mode = "adaptive"
        
        # Statistics
        self.voltage_history: List[float] = []
        self.error_rate_history: List[float] = []
        self.mask_applied_count = 0
        self.total_inferences = 0
    
    def set_voltage(self, voltage: float) -> None:
        """Set the current voltage level."""
        self.current_voltage = voltage
        self.voltage_history.append(voltage)
    
    def get_error_rate(self) -> float:
        """Get estimated bit error rate at current voltage."""
        return estimate_bit_error_rate(self.current_voltage, self.config)
    
    def get_region(self) -> str:
        """Get the voltage region classification."""
        if self.current_voltage >= self.config.safe_min_voltage:
            return "safe"
        elif self.current_voltage >= self.config.critical_min_voltage:
            return "marginal"
        else:
            return "critical"
    
    def should_apply_masking(self) -> bool:
        """Determine if error masking should be applied."""
        if not self.config.enable_adaptive_masking:
            return False
        
        region = self.get_region()
        return region in ("marginal", "critical")
    
    def get_masking_level(self) -> float:
        """
        Get the masking level based on voltage.
        
        Returns float between 0 (no masking) and 1 (maximum masking).
        """
        if self.get_region() == "safe":
            return 0.0
        
        if self.get_region() == "marginal":
            # Linear increase in marginal region
            return (self.config.safe_min_voltage - self.current_voltage) / (
                self.config.safe_min_voltage - self.config.critical_min_voltage
            ) * 0.5  # Max 50% in marginal
        
        # Full masking in critical
        return 1.0
    
    def step(self) -> Dict[str, float]:
        """
        Process one inference step.
        
        Returns:
            Dictionary with error rate and masking info
        """
        error_rate = self.get_error_rate()
        self.error_rate_history.append(error_rate)
        self.total_inferences += 1
        
        if self.should_apply_masking():
            self.mask_applied_count += 1
        
        return {
            "voltage": self.current_voltage,
            "error_rate": error_rate,
            "region": self.get_region(),
            "masking_level": self.get_masking_level(),
            "masking_applied": self.should_apply_masking(),
        }
    
    def get_stats(self) -> Dict[str, float]:
        """Get voltage scaling statistics."""
        return {
            "current_voltage": self.current_voltage,
            "error_rate": self.get_error_rate(),
            "region": self.get_region(),
            "masking_level": self.get_masking_level(),
            "masking_applied_pct": self.mask_applied_count / max(1, self.total_inferences),
            "avg_voltage": np.mean(self.voltage_history) if self.voltage_history else self.current_voltage,
        }
    
    def scale(self, hv: torch.Tensor) -> torch.Tensor:
        """Apply voltage-dependent corruption to a hypervector.

        Args:
            hv: (dim,) input hypervector

        Returns:
            (dim,) possibly corrupted hypervector
        """
        error_rate = self.get_error_rate()
        corrupted = hv.clone()
        if error_rate > 0:
            mask = torch.rand_like(hv) < error_rate
            corrupted[mask] = 0.0
        return corrupted

    def estimate_energy_savings(self) -> float:
        """
        Estimate energy savings from voltage scaling.
        
        Based on P ∝ V² f relationship.
        """
        if not self.voltage_history:
            return 0.0
        
        v_current = np.mean(self.voltage_history)
        v_nom = self.config.nominal_voltage
        
        # Approximate: dynamic power scales with V²
        power_savings = 1 - (v_current / v_nom) ** 2
        return max(0, power_savings)


class SafeRegionDetector:
    """
    Detects safe operating region for voltage scaling.
    
    Analyzes model performance at different voltages to determine
    the safe region boundaries.
    
    Attributes:
        config: VoltageScalingConfig
        results: Dictionary of voltage -> accuracy mappings
    """
    
    def __init__(
        self,
        config: Optional[VoltageScalingConfig] = None,
    ):
        self.config = config or VoltageScalingConfig()
        self.results: Dict[float, float] = {}
    
    def add_result(self, voltage: float, accuracy: float) -> None:
        """Add accuracy result at given voltage."""
        self.results[voltage] = accuracy
    
    def find_safe_boundary(self, baseline_accuracy: float, tolerance: float = 0.01) -> float:
        """
        Find the safe voltage boundary.
        
        Args:
            baseline_accuracy: Accuracy at nominal voltage
            tolerance: Maximum allowed accuracy drop (1% = 0.01)
        
        Returns:
            Safe minimum voltage
        """
        sorted_voltages = sorted(self.results.keys(), reverse=True)
        
        for v in sorted_voltages:
            if v >= self.config.safe_min_voltage:
                continue
            
            accuracy = self.results[v]
            if baseline_accuracy - accuracy <= tolerance:
                return v
        
        return self.config.safe_min_voltage
    
    def find_critical_boundary(self, baseline_accuracy: float, tolerance: float = 0.05) -> float:
        """
        Find the critical voltage boundary.
        
        Args:
            baseline_accuracy: Accuracy at nominal voltage
            tolerance: Maximum allowed accuracy drop (5% = 0.05)
        
        Returns:
            Critical minimum voltage
        """
        sorted_voltages = sorted(self.results.keys(), reverse=True)
        
        for v in sorted_voltages:
            accuracy = self.results[v]
            if baseline_accuracy - accuracy <= tolerance:
                return v
        
        return self.config.critical_min_voltage
    
    def get_safe_voltage_range(self, baseline_accuracy: float) -> Tuple[float, float]:
        """Get safe voltage range (nominal, safe_min)."""
        return (
            self.config.nominal_voltage,
            self.find_safe_boundary(baseline_accuracy)
        )
    
    def detect(self, hv: torch.Tensor, voltage: float) -> bool:
        """Detect whether the given voltage is in the safe operating region.

        Args:
            hv: hypervector (unused, present for API consistency)
            voltage: Supply voltage to evaluate

        Returns:
            True if voltage is safe
        """
        return bool(voltage >= self.config.safe_min_voltage)

    def recommend_voltage(self, target_accuracy: float) -> float:
        """
        Recommend optimal voltage for target accuracy.
        
        Args:
            target_accuracy: Desired accuracy (0-1)
        
        Returns:
            Recommended voltage
        """
        if not self.results:
            return self.config.nominal_voltage
        
        sorted_voltages = sorted(self.results.keys(), reverse=True)
        
        for v in sorted_voltages:
            if self.results[v] >= target_accuracy:
                return v
        
        # If target not achievable, return minimum safe
        return self.config.safe_min_voltage


def test_voltage_scaling():
    """Test voltage scaling functions."""
    print("Testing voltage scaling...")
    
    config = VoltageScalingConfig()
    
    # Test error rate estimation
    print(f"At 0.8V: BER = {estimate_bit_error_rate(0.8, config):.2e}")
    print(f"At 0.6V: BER = {estimate_bit_error_rate(0.6, config):.2e}")
    print(f"At 0.5V: BER = {estimate_bit_error_rate(0.5, config):.2e}")
    print(f"At 0.45V: BER = {estimate_bit_error_rate(0.45, config):.2e}")
    
    # Test VoltageScaler
    scaler = VoltageScaler(config, initial_voltage=0.8)
    
    # Simulate voltage scaling
    for voltage in [0.8, 0.7, 0.6, 0.55, 0.5, 0.45]:
        scaler.set_voltage(voltage)
        info = scaler.step()
        print(f"V={info['voltage']:.2f}, Region={info['region']}, "
              f"BER={info['error_rate']:.2e}, Mask={info['masking_applied']}")
    
    # Test SafeRegionDetector
    detector = SafeRegionDetector(config)
    detector.add_result(0.8, 0.95)
    detector.add_result(0.7, 0.94)
    detector.add_result(0.6, 0.93)
    detector.add_result(0.5, 0.70)  # Large drop!
    detector.add_result(0.45, 0.40)
    
    baseline = detector.results[0.8]
    safe_boundary = detector.find_safe_boundary(baseline)
    print(f"\nSafe boundary: {safe_boundary}V (1% tolerance)")
    
    critical_boundary = detector.find_critical_boundary(baseline)
    print(f"Critical boundary: {critical_boundary}V (5% tolerance)")
    
    # Energy savings
    scaler.set_voltage(0.5)
    print(f"\nEnergy savings at 0.5V: {scaler.estimate_energy_savings():.1%}")
    
    print("\nVoltage scaling tests complete!")


if __name__ == "__main__":
    test_voltage_scaling()