"""
hdc/riscv_hdc.py
=================
RISC-V + HDC Co-Design Simulation
==================================
Based on papers from Paul R. Genssler (TUM):
  - "A 22 nm RISC-V Customized with Near-Memory Acceleration for HDC Edge Training"
  - "Domain-Specific Hyperdimensional RISC-V Processor for Edge-AI Training"
  - "Spike-RISC: Algorithm/ISA Co-Optimization for Efficient SNNs on RISC-V"
  - "TransHD: Spatial Transformer Features Extraction for HDC Synergetic Learning"

Provides:
  - RISC-V HDC instruction set simulation (custom ISA extensions)
  - Near-memory acceleration modeling for HDC operations
  - Energy/latency estimation for RISC-V HDC processors
  - Spike-RISC co-optimization for SNN + HDC on RISC-V

Usage:
    from hdc.riscv_hdc import RISC_V_HDC_Processor, HDCInstructionSet
    proc = RISC_V_HDC_Processor(dim=4096)
    cycles = proc.simulate_hdc_training(n_samples=100, n_classes=10)
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── HDC Custom ISA Extensions for RISC-V ──────────────────────────────────────

@dataclass
class HDCInstruction:
    """A single HDC custom instruction for RISC-V."""
    name: str
    opcode: str
    funct3: int
    funct7: int
    latency_cycles: int
    energy_pj: float  # picojoules per operation
    description: str


class HDCInstructionSet:
    """
    Custom RISC-V ISA extensions for hyperdimensional computing.
    
    Based on Genssler et al. "Domain-Specific Hyperdimensional RISC-V Processor"
    and "Spike-RISC: Algorithm/ISA Co-Optimization for Efficient SNNs on RISC-V".
    
    These instructions accelerate the core HDC operations:
      - hdc.bind:   XOR-based binding of two hypervectors
      - hdc.bundle: Element-wise addition/majority for bundling
      - hdc.perm:   Cyclic permutation (rotation)
      - hdc.sim:    Hamming distance / cosine similarity
      - hdc.encode: N-gram encoding of input data
      - hdc.train:  One-shot training (bundle into prototype)
    """

    def __init__(self, dim: int = 4096, word_size: int = 32):
        self.dim = dim
        self.word_size = word_size
        self.n_words = dim // word_size  # number of words per HV
        
        # Define custom instructions
        self.instructions = {
            "hdc.bind": HDCInstruction(
                name="hdc.bind",
                opcode="CUSTOM_0",
                funct3=0,
                funct7=0x01,
                latency_cycles=self._estimate_bind_latency(),
                energy_pj=self._estimate_bind_energy(),
                description="XOR binding of two hypervectors (rd = rs1 XOR rs2)",
            ),
            "hdc.bundle": HDCInstruction(
                name="hdc.bundle",
                opcode="CUSTOM_0",
                funct3=1,
                funct7=0x01,
                latency_cycles=self._estimate_bundle_latency(),
                energy_pj=self._estimate_bundle_energy(),
                description="Element-wise addition for bundling (rd = rs1 + rs2)",
            ),
            "hdc.perm": HDCInstruction(
                name="hdc.perm",
                opcode="CUSTOM_0",
                funct3=2,
                funct7=0x01,
                latency_cycles=self._estimate_perm_latency(),
                energy_pj=self._estimate_perm_energy(),
                description="Cyclic permutation (rd = rotate(rs1, imm))",
            ),
            "hdc.sim": HDCInstruction(
                name="hdc.sim",
                opcode="CUSTOM_0",
                funct3=3,
                funct7=0x01,
                latency_cycles=self._estimate_sim_latency(),
                energy_pj=self._estimate_sim_energy(),
                description="Hamming distance / cosine similarity (rd = sim(rs1, rs2))",
            ),
            "hdc.encode": HDCInstruction(
                name="hdc.encode",
                opcode="CUSTOM_0",
                funct3=4,
                funct7=0x01,
                latency_cycles=self._estimate_encode_latency(),
                energy_pj=self._estimate_encode_energy(),
                description="N-gram encoding of input data to hypervector",
            ),
            "hdc.train": HDCInstruction(
                name="hdc.train",
                opcode="CUSTOM_0",
                funct3=5,
                funct7=0x01,
                latency_cycles=self._estimate_train_latency(),
                energy_pj=self._estimate_train_energy(),
                description="One-shot training: bundle HV into class prototype",
            ),
            "hdc.mv": HDCInstruction(
                name="hdc.mv",
                opcode="CUSTOM_0",
                funct3=6,
                funct7=0x01,
                latency_cycles=self._estimate_mv_latency(),
                energy_pj=self._estimate_mv_energy(),
                description="Hypervector move between HV register file and memory",
            ),
        }

    def _estimate_bind_latency(self) -> int:
        """Estimate XOR binding latency in cycles.
        
        For 4096-bit HVs on a 32-bit RISC-V: 4096/32 = 128 word operations.
        With SIMD-style processing: ~16 cycles (8-wide SIMD).
        """
        return max(1, self.n_words // 8)

    def _estimate_bundle_latency(self) -> int:
        """Estimate bundling latency (addition + threshold)."""
        return max(1, self.n_words // 4)  # More complex: add + compare

    def _estimate_perm_latency(self) -> int:
        """Estimate cyclic permutation latency."""
        return max(1, self.n_words // 16)  # Simple: just rotate

    def _estimate_sim_latency(self) -> int:
        """Estimate similarity (Hamming distance) latency."""
        return max(1, self.n_words // 4)  # XOR + popcount

    def _estimate_encode_latency(self) -> int:
        """Estimate n-gram encoding latency."""
        return max(1, self.n_words)  # Most complex: n-gram generation

    def _estimate_train_latency(self) -> int:
        """Estimate one-shot training latency."""
        return max(1, self.n_words // 2)  # Bundle + normalize

    def _estimate_mv_latency(self) -> int:
        """Estimate HV move latency."""
        return max(1, self.n_words // 16)

    def _estimate_bind_energy(self) -> float:
        """Estimate energy for XOR binding in pJ.
        
        Based on 22nm FDX22 technology (Genssler et al.):
        - XOR: ~0.1 pJ per 32-bit word
        - 4096-bit HV: 128 words × 0.1 pJ = 12.8 pJ
        """
        return self.n_words * 0.1

    def _estimate_bundle_energy(self) -> float:
        """Estimate energy for bundling."""
        return self.n_words * 0.15  # Addition is slightly more expensive

    def _estimate_perm_energy(self) -> float:
        """Estimate energy for permutation."""
        return self.n_words * 0.05  # Simple rotation

    def _estimate_sim_energy(self) -> float:
        """Estimate energy for similarity computation."""
        return self.n_words * 0.2  # XOR + popcount tree

    def _estimate_encode_energy(self) -> float:
        """Estimate energy for n-gram encoding."""
        return self.n_words * 0.5  # Most complex operation

    def _estimate_train_energy(self) -> float:
        """Estimate energy for one-shot training."""
        return self.n_words * 0.3

    def _estimate_mv_energy(self) -> float:
        """Estimate energy for HV move."""
        return self.n_words * 0.08

    def summary(self) -> Dict:
        """Return instruction set summary."""
        return {
            name: {
                "latency_cycles": instr.latency_cycles,
                "energy_pj": instr.energy_pj,
                "description": instr.description,
            }
            for name, instr in self.instructions.items()
        }


# ── Near-Memory Acceleration ──────────────────────────────────────────────────

@dataclass
class NearMemoryAcceleratorConfig:
    """Configuration for near-memory HDC acceleration.
    
    Based on Genssler et al. "A 22 nm RISC-V Customized with 
    Near-Memory Acceleration for Hyperdimensional Edge Training":
    - Near-memory compute units placed adjacent to SRAM banks
    - Custom HDC functional units: XOR, adder tree, popcount
    - 22nm FDX22 technology node
    """
    enabled: bool = True
    n_memory_banks: int = 8
    bank_width_bits: int = 512  # Wide SRAM interface
    hd_dim: int = 4096
    technology_nm: int = 22
    voltage_v: float = 0.8
    frequency_mhz: float = 500.0
    sram_energy_pj_per_bit: float = 0.005  # pJ per bit access (22nm)
    compute_energy_pj_per_op: float = 0.1   # pJ per 32-bit operation


class NearMemoryAccelerator:
    """
    Near-memory acceleration for HDC on RISC-V.
    
    Models the performance and energy of near-memory HDC compute units
    that sit adjacent to SRAM banks, avoiding data movement to the core.
    """

    def __init__(self, config: Optional[NearMemoryAcceleratorConfig] = None):
        self.config = config or NearMemoryAcceleratorConfig()
        self.total_operations = 0
        self.total_energy_pj = 0.0
        self.total_cycles = 0

    def hd_dim_to_banks(self) -> int:
        """Number of banks needed to store one hypervector."""
        bits_per_bank = self.config.bank_width_bits
        return math.ceil(self.config.hd_dim / bits_per_bank)

    def bind_energy(self) -> float:
        """Energy for near-memory XOR binding of two HVs."""
        n_banks = self.hd_dim_to_banks()
        # Read two HVs + write result
        read_energy = 2 * n_banks * self.config.bank_width_bits * self.config.sram_energy_pj_per_bit
        compute_energy = (self.config.hd_dim // 32) * self.config.compute_energy_pj_per_op
        write_energy = n_banks * self.config.bank_width_bits * self.config.sram_energy_pj_per_bit
        return read_energy + compute_energy + write_energy

    def bundle_energy(self) -> float:
        """Energy for near-memory bundling."""
        n_banks = self.hd_dim_to_banks()
        read_energy = 2 * n_banks * self.config.bank_width_bits * self.config.sram_energy_pj_per_bit
        compute_energy = (self.config.hd_dim // 16) * self.config.compute_energy_pj_per_op
        write_energy = n_banks * self.config.bank_width_bits * self.config.sram_energy_pj_per_bit
        return read_energy + compute_energy + write_energy

    def similarity_energy(self) -> float:
        """Energy for near-memory similarity computation."""
        n_banks = self.hd_dim_to_banks()
        read_energy = 2 * n_banks * self.config.bank_width_bits * self.config.sram_energy_pj_per_bit
        # Popcount tree is the main compute cost
        compute_energy = (self.config.hd_dim // 8) * self.config.compute_energy_pj_per_op
        return read_energy + compute_energy

    def training_energy(self, n_samples: int) -> float:
        """Energy for one-shot training of n_samples."""
        # Each sample: encode + bundle into prototype
        encode_energy = self.bind_energy() * 3  # Approx: 3 binds per n-gram
        bundle_energy = self.bundle_energy()
        return n_samples * (encode_energy + bundle_energy)

    def inference_energy(self, n_samples: int, n_classes: int) -> float:
        """Energy for inference on n_samples across n_classes."""
        # Each sample: encode + compare to all class prototypes
        encode_energy = self.bind_energy() * 3
        sim_energy = self.similarity_energy() * n_classes
        return n_samples * (encode_energy + sim_energy)

    def speedup_vs_core(self) -> float:
        """Estimated speedup of near-memory vs. core execution.
        
        Near-memory avoids data movement bottleneck. For HDC:
        - Core: data must move from SRAM → core register → compute → SRAM
        - Near-memory: compute happens at the SRAM bank
        
        Typical speedup: 3-10x depending on operation.
        """
        # Data movement dominates: ~70% of energy in core execution
        # Near-memory eliminates most data movement
        return 5.0  # Conservative estimate from Genssler et al.

    def energy_savings_pct(self) -> float:
        """Energy savings percentage from near-memory acceleration."""
        return (1 - 1 / self.speedup_vs_core()) * 100

    def get_stats(self) -> Dict:
        """Return accelerator statistics."""
        return {
            "technology_nm": self.config.technology_nm,
            "frequency_mhz": self.config.frequency_mhz,
            "voltage_v": self.config.voltage_v,
            "n_memory_banks": self.config.n_memory_banks,
            "bank_width_bits": self.config.bank_width_bits,
            "hd_dim": self.config.hd_dim,
            "bind_energy_pj": self.bind_energy(),
            "bundle_energy_pj": self.bundle_energy(),
            "similarity_energy_pj": self.similarity_energy(),
            "training_energy_pj_per_sample": self.training_energy(1),
            "inference_energy_pj_per_sample_10class": self.inference_energy(1, 10),
            "speedup_vs_core": self.speedup_vs_core(),
            "energy_savings_pct": self.energy_savings_pct(),
        }

    def benchmark_vs_cpu(
        self,
        n_classes: int = 10,
        n_test_samples: int = 1000,
        cpu_inference_energy_nj: float = 50.0,   # typical ARM Cortex-M4 inference
    ) -> Dict:
        """
        Compare near-memory HDC vs CPU baseline for a deployment decision.

        Args:
            n_classes:                 Number of classification classes
            n_test_samples:            Benchmark sample count
            cpu_inference_energy_nj:   CPU baseline energy per inference (nJ)

        Returns:
            Dict suitable for investor/technical reports with:
              hdc_energy_pj, cpu_energy_nj, reduction_factor, decision
        """
        hdc_pj   = self.inference_energy(1, n_classes)
        hdc_nj   = hdc_pj / 1000.0
        factor   = cpu_inference_energy_nj / max(hdc_nj, 1e-6)

        return {
            "hdc_energy_pJ":     round(hdc_pj, 3),
            "hdc_energy_nJ":     round(hdc_nj, 6),
            "cpu_energy_nJ":     cpu_inference_energy_nj,
            "energy_reduction":  round(factor, 1),
            "throughput_ratio":  round(self.speedup_vs_core(), 1),
            "n_classes":         n_classes,
            "recommended":       factor > 10,   # 10× reduction = deploy
            "decision":          "DEPLOY_HDC" if factor > 10 else "PROFILE_MORE",
        }


# ── Full RISC-V HDC Processor Simulation ──────────────────────────────────────

@dataclass
class RISC_V_HDC_Config:
    """Configuration for RISC-V HDC processor simulation."""
    dim: int = 4096
    frequency_mhz: float = 500.0
    voltage_v: float = 0.8
    technology_nm: int = 22
    hdc_isa_enabled: bool = True
    near_memory_enabled: bool = True
    n_hv_registers: int = 32  # HV register file size
    pipeline_stages: int = 5   # Classic RISC-V pipeline
    simd_width: int = 8        # SIMD width for HDC operations


class RISC_V_HDC_Processor:
    """
    Full RISC-V HDC processor simulation.
    
    Models:
    - Custom HDC ISA extensions
    - Near-memory acceleration
    - Pipeline simulation
    - Energy and latency estimation
    
    Based on Genssler et al. papers from TUM:
    - Domain-specific RISC-V processor for HDC edge training
    - Spike-RISC co-optimization
    - Near-memory acceleration
    """

    def __init__(self, config: Optional[RISC_V_HDC_Config] = None):
        self.config = config or RISC_V_HDC_Config()
        self.isa = HDCInstructionSet(dim=self.config.dim)
        self.near_memory = NearMemoryAccelerator(
            NearMemoryAcceleratorConfig(
                hd_dim=self.config.dim,
                technology_nm=self.config.technology_nm,
                voltage_v=self.config.voltage_v,
                frequency_mhz=self.config.frequency_mhz,
            )
        ) if self.config.near_memory_enabled else None

        # Pipeline state
        self.cycle_count = 0
        self.instruction_count = 0
        self.total_energy_pj = 0.0
        self.stall_cycles = 0
        self.hv_register_file: List[Optional[torch.Tensor]] = [None] * self.config.n_hv_registers

        # Performance counters
        self.perf_counters = {
            "hdc.bind": 0,
            "hdc.bundle": 0,
            "hdc.perm": 0,
            "hdc.sim": 0,
            "hdc.encode": 0,
            "hdc.train": 0,
            "hdc.mv": 0,
            "base_ops": 0,
        }

    def execute_hdc_instruction(self, instr_name: str) -> int:
        """Execute one HDC instruction and return cycle count."""
        if instr_name not in self.isa.instructions:
            raise ValueError(f"Unknown HDC instruction: {instr_name}")

        instr = self.isa.instructions[instr_name]
        self.instruction_count += 1
        self.perf_counters[instr_name] += 1

        # Pipeline: fetch → decode → execute → memory → writeback
        # HDC instructions may have multi-cycle execute stage
        cycles = instr.latency_cycles

        # If near-memory accelerator is available, reduce cycles
        if self.near_memory and self.config.near_memory_enabled:
            cycles = max(1, cycles // 3)  # 3x speedup from near-memory

        self.cycle_count += cycles
        self.total_energy_pj += instr.energy_pj

        return cycles

    def simulate_hdc_training(
        self,
        n_samples: int,
        n_classes: int,
        n_gram_length: int = 4,
    ) -> Dict:
        """Simulate training n_samples into n_classes.
        
        Returns cycle count and energy for the full training pipeline.
        """
        cycles = 0
        energy = 0.0

        for _ in range(n_samples):
            # 1. Encode input (n-gram encoding)
            cycles += self.execute_hdc_instruction("hdc.encode")
            energy += self.isa.instructions["hdc.encode"].energy_pj

            # 2. Store encoded HV in register
            cycles += self.execute_hdc_instruction("hdc.mv")
            energy += self.isa.instructions["hdc.mv"].energy_pj

            # 3. Bundle into class prototype (one-shot training)
            cycles += self.execute_hdc_instruction("hdc.train")
            energy += self.isa.instructions["hdc.train"].energy_pj

        # Finalize: normalize all class prototypes
        for _ in range(n_classes):
            cycles += self.execute_hdc_instruction("hdc.bundle")
            energy += self.isa.instructions["hdc.bundle"].energy_pj

        return {
            "total_cycles": cycles,
            "total_energy_pj": energy,
            "total_energy_uj": energy / 1e6,
            "cycles_per_sample": cycles / n_samples,
            "energy_per_sample_pj": energy / n_samples,
            "frequency_mhz": self.config.frequency_mhz,
            "execution_time_us": cycles / self.config.frequency_mhz,
            "instruction_count": self.instruction_count,
            "perf_counters": dict(self.perf_counters),
        }

    def simulate_hdc_inference(
        self,
        n_samples: int,
        n_classes: int,
    ) -> Dict:
        """Simulate inference on n_samples across n_classes."""
        cycles = 0
        energy = 0.0

        for _ in range(n_samples):
            # 1. Encode input
            cycles += self.execute_hdc_instruction("hdc.encode")
            energy += self.isa.instructions["hdc.encode"].energy_pj

            # 2. Compare to all class prototypes
            for _ in range(n_classes):
                cycles += self.execute_hdc_instruction("hdc.sim")
                energy += self.isa.instructions["hdc.sim"].energy_pj

        return {
            "total_cycles": cycles,
            "total_energy_pj": energy,
            "total_energy_uj": energy / 1e6,
            "cycles_per_sample": cycles / n_samples,
            "energy_per_sample_pj": energy / n_samples,
            "execution_time_us": cycles / self.config.frequency_mhz,
            "throughput_samples_per_sec": (n_samples * self.config.frequency_mhz * 1e6) / cycles,
        }

    def compare_with_baseline_riscv(self) -> Dict:
        """Compare HDC-accelerated RISC-V vs baseline RISC-V.
        
        Baseline RISC-V would implement HDC operations in software
        using base integer instructions (add, xor, etc.).
        """
        # Baseline: each HV operation requires many base instructions
        # For 4096-bit HV on 32-bit RISC-V: 128 word operations
        base_cycles_per_bind = 128 * 3  # Load + XOR + Store
        hdc_cycles_per_bind = self.isa.instructions["hdc.bind"].latency_cycles

        base_cycles_per_sim = 128 * 4  # Load + XOR + popcount + store
        hdc_cycles_per_sim = self.isa.instructions["hdc.sim"].latency_cycles

        speedup_bind = base_cycles_per_bind / hdc_cycles_per_bind
        speedup_sim = base_cycles_per_sim / hdc_cycles_per_sim

        return {
            "baseline_bind_cycles": base_cycles_per_bind,
            "hdc_isa_bind_cycles": hdc_cycles_per_bind,
            "bind_speedup": speedup_bind,
            "baseline_sim_cycles": base_cycles_per_sim,
            "hdc_isa_sim_cycles": hdc_cycles_per_sim,
            "sim_speedup": speedup_sim,
            "overall_speedup_estimate": (speedup_bind + speedup_sim) / 2,
            "isa_summary": self.isa.summary(),
            "near_memory_stats": self.near_memory.get_stats() if self.near_memory else None,
        }

    def get_performance_report(self) -> Dict:
        """Get full performance report."""
        return {
            "config": {
                "dim": self.config.dim,
                "frequency_mhz": self.config.frequency_mhz,
                "voltage_v": self.config.voltage_v,
                "technology_nm": self.config.technology_nm,
                "hdc_isa_enabled": self.config.hdc_isa_enabled,
                "near_memory_enabled": self.config.near_memory_enabled,
            },
            "cycle_count": self.cycle_count,
            "instruction_count": self.instruction_count,
            "total_energy_pj": self.total_energy_pj,
            "stall_cycles": self.stall_cycles,
            "perf_counters": dict(self.perf_counters),
            "isa_summary": self.isa.summary(),
        }


# ── TransHD: Spatial Transformer + HDC Synergetic Learning ────────────────────

class TransHDEncoder(torch.nn.Module):
    """
    TransHD: Spatial Transformer Features Extraction for HDC Synergetic Learning.
    
    Based on Genssler et al. "TransHD: Spatial transformer features extraction 
    for HDC synergetic learning" (2024).
    
    Key insight: Use a lightweight spatial transformer network to extract
    features, then encode them into hyperdimensional space for classification.
    The transformer provides spatial attention, HDC provides efficient learning.
    """

    def __init__(
        self,
        input_channels: int = 1,
        spatial_dim: int = 64,
        hd_dim: int = 4096,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.spatial_dim = spatial_dim
        self.hd_dim = hd_dim

        # Spatial transformer: lightweight CNN + attention
        self.spatial_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
        )

        # Spatial attention mechanism
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(32 * 4 * 4, spatial_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(spatial_dim, 32 * 4 * 4),
            torch.nn.Sigmoid(),
        )

        # Projection to hyperdimensional space
        self.hd_projection = torch.nn.Linear(32 * 4 * 4, hd_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into hyperdimensional space.
        
        Args:
            x: (B, C, H, W) input tensor
        
        Returns:
            (B, dim) hypervectors
        """
        # Extract spatial features
        features = self.spatial_encoder(x)  # (B, 512)
        
        # Apply spatial attention
        attn_weights = self.attention(features)  # (B, 512)
        attended = features * attn_weights
        
        # Project to HD space
        hvs = self.hd_projection(attended)  # (B, dim)
        
        # Binarize to bipolar hypervectors
        hvs = torch.where(hvs > 0, 
                          torch.ones_like(hvs), 
                          -torch.ones_like(hvs))
        
        return hvs


# ── Self-test ──────────────────────────────────────────────────────────────────

def test_riscv_hdc():
    """Test RISC-V HDC processor simulation."""
    print("=" * 60)
    print("Testing RISC-V HDC Co-Design Simulation")
    print("=" * 60)

    # Test instruction set
    isa = HDCInstructionSet(dim=4096)
    summary = isa.summary()
    print("\nHDC Custom ISA Extensions:")
    for name, info in summary.items():
        print(f"  {name:20s}  {info['latency_cycles']:4d} cycles  "
              f"{info['energy_pj']:8.1f} pJ  | {info['description'][:50]}")

    # Test processor
    proc = RISC_V_HDC_Processor()
    
    # Simulate training
    print("\nSimulating HDC Training (100 samples, 10 classes):")
    train_result = proc.simulate_hdc_training(n_samples=100, n_classes=10)
    print(f"  Total cycles: {train_result['total_cycles']:,}")
    print(f"  Total energy: {train_result['total_energy_uj']:.2f} µJ")
    print(f"  Cycles/sample: {train_result['cycles_per_sample']:.0f}")
    print(f"  Energy/sample: {train_result['energy_per_sample_pj']:.1f} pJ")
    print(f"  Execution time: {train_result['execution_time_us']:.2f} µs")

    # Simulate inference
    print("\nSimulating HDC Inference (1000 samples, 10 classes):")
    inf_result = proc.simulate_hdc_inference(n_samples=1000, n_classes=10)
    print(f"  Total cycles: {inf_result['total_cycles']:,}")
    print(f"  Total energy: {inf_result['total_energy_uj']:.2f} µJ")
    print(f"  Throughput: {inf_result['throughput_samples_per_sec']:.0f} samples/s")

    # Compare with baseline
    print("\nComparison with Baseline RISC-V:")
    comparison = proc.compare_with_baseline_riscv()
    print(f"  Bind speedup: {comparison['bind_speedup']:.1f}x")
    print(f"  Similarity speedup: {comparison['sim_speedup']:.1f}x")
    print(f"  Overall speedup: {comparison['overall_speedup_estimate']:.1f}x")

    # Test TransHD
    print("\nTesting TransHD Encoder:")
    transhd = TransHDEncoder(input_channels=1, hd_dim=4096)
    dummy_input = torch.randn(4, 1, 28, 28)
    output = transhd(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output values: {output.unique().tolist()}")

    print("\n✅ RISC-V HDC test complete!")


if __name__ == "__main__":
    test_riscv_hdc()
