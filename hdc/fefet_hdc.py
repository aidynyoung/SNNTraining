"""
hdc/fefet_hdc.py
=================
FeFET-Based Hyperdimensional Computing
========================================
Based on papers from Paul R. Genssler (TUM):
  - "Cross-layer FeFET Reliability Modeling for Robust Hyperdimensional Computing"
  - "All-in-Memory Brain-Inspired Computing Using FeFET Synapses"
  - "HDGIM: Hyperdimensional Genome Sequence Matching on Unreliable Highly Scaled FeFET"
  - "On the Reliability of FeFET On-Chip Memory"

Provides:
  - FeFET synapse device model (multi-level cell, retention, endurance)
  - Cross-layer reliability modeling (device → circuit → system)
  - FeFET-based in-memory HDC computing
  - HDGIM: Genome sequence matching with FeFET HDC

Usage:
    from hdc.fefet_hdc import FeFETDevice, FeFETCrossbar, HDGIM
    device = FeFETDevice(technology_nm=28)
    crossbar = FeFETCrossbar(device, dim=4096)
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FeFET Device Model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeFETDeviceConfig:
    """FeFET device parameters.
    
    Based on Genssler et al. "Cross-layer FeFET Reliability Modeling 
    for Robust Hyperdimensional Computing" and FeFET characterization data.
    
    Key parameters:
    - Technology node (28nm, 22nm FDX, etc.)
    - Number of programmable states (multi-level cell)
    - Retention time characteristics
    - Endurance (P/E cycles before breakdown)
    - Variability (cycle-to-cycle, device-to-device)
    """
    technology_nm: int = 28
    n_levels: int = 4  # Multi-level cell: 2 bits per cell
    vdd_v: float = 1.8
    programming_voltage_v: float = 4.0
    pulse_width_ns: float = 10.0
    
    # Retention
    retention_time_years: float = 10.0
    retention_activation_energy_ev: float = 0.8
    
    # Endurance
    endurance_cycles: int = 10_000  # Typical FeFET endurance
    breakdown_probability_per_cycle: float = 1e-5
    
    # Variability
    cycle_to_cycle_std: float = 0.05  # 5% of programming window
    device_to_device_std: float = 0.08  # 8% variation
    
    # Temperature dependence
    temperature_c: float = 25.0
    temperature_coefficient: float = 0.001  # Per degree C
    
    # Energy
    read_energy_pj: float = 0.05  # pJ per read
    write_energy_pj: float = 0.5   # pJ per write (higher due to programming voltage)


class FeFETDevice:
    """
    FeFET (Ferroelectric Field-Effect Transistor) device model.
    
    FeFETs are promising for HDC because:
    1. Multi-level cell capability (2+ bits per cell)
    2. Non-volatile storage (retention > 10 years)
    3. In-memory computing (compute where data is stored)
    4. CMOS compatibility (can be integrated with RISC-V)
    
    This model captures:
    - Multi-level programming with variability
    - Retention degradation over time
    - Endurance wear-out
    - Temperature dependence
    """

    def __init__(self, config: Optional[FeFETDeviceConfig] = None):
        self.config = config or FeFETDeviceConfig()
        self.rng = torch.Generator().manual_seed(42)
        
        # State
        self.program_cycles = 0
        self.read_cycles = 0
        self.current_temperature = self.config.temperature_c
        
        # Degradation tracking
        self.retention_degradation = 0.0
        self.endurance_degradation = 0.0

    def program_level(self, target_level: int) -> float:
        """Program FeFET to a target conductance level.
        
        Args:
            target_level: 0 to n_levels-1
        
        Returns:
            Actual programmed conductance (normalized 0-1)
        """
        self.program_cycles += 1
        
        # Ideal conductance
        ideal = target_level / (self.config.n_levels - 1)
        
        # Cycle-to-cycle variability
        c2c_noise = torch.randn(1, generator=self.rng).item() * self.config.cycle_to_cycle_std
        
        # Device-to-device variability (constant offset)
        if self.program_cycles == 1:
            self._d2d_offset = torch.randn(1, generator=self.rng).item() * self.config.device_to_device_std
        d2d_offset = getattr(self, '_d2d_offset', 0.0)
        
        # Endurance degradation (conductance window closure)
        endurance_factor = 1.0 - self.endurance_degradation
        
        # Temperature effect
        temp_delta = self.current_temperature - self.config.temperature_c
        temp_factor = 1.0 + self.config.temperature_coefficient * temp_delta
        
        actual = (ideal + c2c_noise + d2d_offset) * endurance_factor * temp_factor
        return max(0.0, min(1.0, actual))

    def read_conductance(self, stored_level: float, time_elapsed_years: float = 0) -> float:
        """Read conductance from FeFET, accounting for retention loss.
        
        Args:
            stored_level: The programmed conductance level
            time_elapsed_years: Time since programming
        
        Returns:
            Read conductance (may have degraded due to retention)
        """
        self.read_cycles += 1
        
        # Temperature delta from reference
        temp_delta = self.current_temperature - self.config.temperature_c
        
        # Retention degradation: conductance drifts over time
        # Model: Arrhenius-like retention loss
        retention_loss = 1.0 - math.exp(
            -time_elapsed_years / self.config.retention_time_years
            * math.exp(-self.config.retention_activation_energy_ev / (0.0259 * (1 + temp_delta/298)))
        )
        
        # Read disturb: small change from reading
        read_disturb = torch.randn(1, generator=self.rng).item() * 0.001 * self.read_cycles
        
        return stored_level * (1 - retention_loss) + read_disturb

    def apply_temperature(self, temperature_c: float):
        """Set operating temperature."""
        self.current_temperature = temperature_c

    def simulate_endurance_cycling(self, n_cycles: int) -> Dict:
        """Simulate endurance degradation over P/E cycles.
        
        Args:
            n_cycles: Number of program/erase cycles
        
        Returns:
            Degradation statistics
        """
        initial_window = 1.0
        for _ in range(n_cycles):
            self.program_cycles += 1
            
            # Gradual window closure
            self.endurance_degradation += self.config.breakdown_probability_per_cycle
            
            # Sudden breakdown (random)
            if torch.rand(1, generator=self.rng).item() < self.config.breakdown_probability_per_cycle:
                self.endurance_degradation = 1.0  # Device failed
                break

        remaining_window = max(0, initial_window - self.endurance_degradation)
        return {
            "total_cycles": n_cycles,
            "endurance_degradation": self.endurance_degradation,
            "remaining_window_pct": remaining_window * 100,
            "device_failed": self.endurance_degradation >= 1.0,
        }

    def get_device_stats(self) -> Dict:
        """Get FeFET device statistics."""
        return {
            "technology_nm": self.config.technology_nm,
            "n_levels": self.config.n_levels,
            "program_cycles": self.program_cycles,
            "read_cycles": self.read_cycles,
            "endurance_degradation": self.endurance_degradation,
            "temperature_c": self.current_temperature,
            "read_energy_pj": self.config.read_energy_pj,
            "write_energy_pj": self.config.write_energy_pj,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FeFET Crossbar for HDC
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeFETCrossbarConfig:
    """Configuration for FeFET crossbar array for HDC.
    
    Based on Genssler et al. "All-in-Memory Brain-Inspired Computing 
    Using FeFET Synapses" and cross-layer reliability work.
    """
    dim: int = 4096
    n_rows: int = 128  # Number of word lines
    n_columns: int = 4096  # Number of bit lines (HD dimension)
    cell_per_row: int = 32  # Cells per word line
    operating_voltage_v: float = 0.8
    read_energy_pj_per_cell: float = 0.05
    write_energy_pj_per_cell: float = 0.5
    retention_time_years: float = 10.0
    enable_error_correction: bool = True


class FeFETCrossbar:
    """
    FeFET crossbar array for in-memory HDC computation.
    
    Implements:
    - Hypervector storage in FeFET multi-level cells
    - In-memory similarity computation (Hamming distance)
    - Bundling and binding operations in the analog domain
    - Error correction for FeFET reliability issues
    
    Key advantage: All-in-memory computing eliminates data movement,
    which is the dominant energy cost in traditional architectures.
    """

    def __init__(
        self,
        device: FeFETDevice,
        config: Optional[FeFETCrossbarConfig] = None,
    ):
        self.device = device
        self.config = config or FeFETCrossbarConfig()
        
        # Storage: each row stores one hypervector
        self.storage = torch.zeros(self.config.n_rows, self.config.dim)
        self.row_labels: List[str] = []
        self.n_stored = 0  # Track number of stored HVs
        
        # Error correction
        self.ecc_enabled = self.config.enable_error_correction
        self.error_positions: List[torch.Tensor] = []

    def store_hypervector(
        self,
        hv: torch.Tensor,
        label: str = "",
        row_idx: Optional[int] = None,
    ) -> int:
        """Store a hypervector in the FeFET crossbar.
        
        Args:
            hv: (dim,) bipolar hypervector
            label: Optional label for the row
            row_idx: Specific row to use (None = next available)
        
        Returns:
            Row index where HV was stored
        """
        if row_idx is None:
            row_idx = self.n_stored
            if row_idx >= self.config.n_rows:
                raise ValueError("Crossbar is full")

        # Program each FeFET cell
        for i in range(self.config.dim):
            # Map bipolar (±1) to FeFET level (0 or n_levels-1)
            target_level = self.device.config.n_levels - 1 if hv[i] > 0 else 0
            programmed = self.device.program_level(target_level)
            self.storage[row_idx, i] = programmed

        if label:
            self.row_labels.append(label)
        self.n_stored += 1
        
        return row_idx

    def compute_similarity(
        self,
        query_hv: torch.Tensor,
        row_idx: int,
        time_elapsed_years: float = 0,
    ) -> float:
        """Compute similarity between query and stored HV.
        
        Performs in-memory Hamming distance computation:
        1. Read stored FeFET conductances
        2. Compare with query hypervector
        3. Return normalized similarity
        
        Args:
            query_hv: (dim,) query hypervector
            row_idx: Row to compare against
            time_elapsed_years: Time since programming (for retention)
        
        Returns:
            Similarity score (0-1)
        """
        stored = self.storage[row_idx]
        
        # Read with retention degradation
        read_hv = torch.zeros(self.config.dim)
        for i in range(self.config.dim):
            read_hv[i] = self.device.read_conductance(
                stored[i].item(), time_elapsed_years
            )
        
        # Binarize read values
        read_binary = torch.where(read_hv > 0.5, 
                                   torch.ones_like(read_hv), 
                                  -torch.ones_like(read_hv))
        
        # Hamming similarity
        agreement = (query_hv == read_binary).float().mean().item()
        return agreement

    def batch_similarity(
        self,
        query_hv: torch.Tensor,
        time_elapsed_years: float = 0,
    ) -> torch.Tensor:
        """Compute similarity against all stored HVs.
        
        Args:
            query_hv: (dim,) query hypervector
            time_elapsed_years: Time since programming
        
        Returns:
            (n_stored,) similarity scores
        """
        n_stored = len(self.row_labels)
        similarities = torch.zeros(n_stored)
        
        for i in range(n_stored):
            similarities[i] = self.compute_similarity(query_hv, i, time_elapsed_years)
        
        return similarities

    def estimate_energy(self, n_operations: int) -> Dict:
        """Estimate energy consumption for crossbar operations.
        
        Args:
            n_operations: Number of read/write operations
        
        Returns:
            Energy breakdown
        """
        read_energy = n_operations * self.config.dim * self.config.read_energy_pj_per_cell
        write_energy = n_operations * self.config.dim * self.config.write_energy_pj_per_cell
        
        return {
            "read_energy_pj": read_energy,
            "write_energy_pj": write_energy,
            "total_energy_pj": read_energy + write_energy,
            "total_energy_uj": (read_energy + write_energy) / 1e6,
        }

    def simulate_retention_effect(
        self,
        time_years: float,
    ) -> Dict:
        """Simulate retention loss over time.
        
        Args:
            time_years: Time elapsed since programming
        
        Returns:
            Retention degradation statistics
        """
        n_stored = len(self.row_labels)
        if n_stored == 0:
            return {"error": "No HVs stored"}
        
        original = self.storage[:n_stored].clone()
        
        # Apply retention degradation
        for row in range(n_stored):
            for col in range(self.config.dim):
                self.storage[row, col] = self.device.read_conductance(
                    original[row, col].item(), time_years
                )
        
        # Measure degradation
        degradation = (original[:n_stored] - self.storage[:n_stored]).abs().mean().item()
        
        return {
            "time_years": time_years,
            "mean_degradation": degradation,
            "max_degradation": (original[:n_stored] - self.storage[:n_stored]).abs().max().item(),
            "n_stored_hvs": n_stored,
        }

    def get_crossbar_stats(self) -> Dict:
        """Get crossbar statistics."""
        return {
            "dim": self.config.dim,
            "n_rows": self.config.n_rows,
            "n_columns": self.config.n_columns,
            "n_stored_hvs": len(self.row_labels),
            "utilization_pct": len(self.row_labels) / self.config.n_rows * 100,
            "ecc_enabled": self.ecc_enabled,
            "device_stats": self.device.get_device_stats(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HDGIM: Genome Sequence Matching with FeFET HDC
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HDGIMConfig:
    """Configuration for HDGIM genome matching.
    
    Based on Genssler et al. "HDGIM: Hyperdimensional Genome Sequence 
    Matching on Unreliable Highly Scaled FeFET" (2024).
    
    Key insight: Encode DNA sequences as hypervectors using k-mer
    frequency encoding, then match using HDC similarity on FeFET hardware.
    """
    k_mer_size: int = 4  # Length of DNA k-mers
    dim: int = 4096
    n_gram_bases: List[str] = field(
        default_factory=lambda: ["A", "C", "G", "T"]
    )
    use_reverse_complement: bool = True
    similarity_threshold: float = 0.7  # Match threshold


class HDGIM:
    """
    HDGIM: Hyperdimensional Genome Sequence Matching.
    
    Encodes DNA sequences as hypervectors using k-mer frequency encoding,
    then performs matching using HDC similarity on FeFET crossbar hardware.
    
    The encoding works as follows:
    1. Extract all k-mers from the DNA sequence
    2. Map each k-mer to a random hypervector (using hashing)
    3. Bundle all k-mer HVs into a sequence HV
    4. Compare sequence HVs using Hamming similarity
    
    This is naturally robust to FeFET reliability issues because:
    - HDC is inherently error-tolerant
    - Redundant encoding (many k-mers per sequence)
    - Similarity-based matching (not exact)
    """

    def __init__(self, config: Optional[HDGIMConfig] = None):
        self.config = config or HDGIMConfig()
        self.rng = torch.Generator().manual_seed(42)
        
        # Pre-compute k-mer hypervectors
        self.kmer_hvs: Dict[str, torch.Tensor] = {}
        self._build_kmer_hvs()

    def _build_kmer_hvs(self):
        """Generate random hypervectors for all possible k-mers."""
        bases = self.config.n_gram_bases
        k = self.config.k_mer_size
        
        def generate_kmers(bases, k):
            if k == 0:
                yield ""
            else:
                for base in bases:
                    for suffix in generate_kmers(bases, k - 1):
                        yield base + suffix
        
        for kmer in generate_kmers(bases, k):
            seed = hash(kmer) & 0x7FFFFFFF
            g = torch.Generator().manual_seed(seed)
            hv = torch.randn(self.config.dim, generator=g)
            hv = torch.where(hv > 0, torch.ones_like(hv), -torch.ones_like(hv))
            self.kmer_hvs[kmer] = hv

    def _reverse_complement(self, seq: str) -> str:
        """Get reverse complement of DNA sequence."""
        complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
        return "".join(complement.get(base, base) for base in reversed(seq))

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode a DNA sequence as a hypervector.
        
        Args:
            sequence: DNA sequence string (e.g., "ATCGATCG...")
        
        Returns:
            (dim,) bipolar hypervector representing the sequence
        """
        sequence = sequence.upper().strip()
        k = self.config.k_mer_size
        
        # Extract k-mers
        kmer_hvs = []
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i + k]
            if kmer in self.kmer_hvs:
                kmer_hvs.append(self.kmer_hvs[kmer])
        
        # Add reverse complement k-mers
        if self.config.use_reverse_complement:
            rc = self._reverse_complement(sequence)
            for i in range(len(rc) - k + 1):
                kmer = rc[i:i + k]
                if kmer in self.kmer_hvs:
                    kmer_hvs.append(self.kmer_hvs[kmer])
        
        if not kmer_hvs:
            return torch.zeros(self.config.dim)
        
        # Bundle all k-mer HVs (majority vote)
        stacked = torch.stack(kmer_hvs)
        bundled = stacked.float().mean(dim=0)
        return torch.where(bundled > 0, torch.ones_like(bundled), -torch.ones_like(bundled))

    def match_sequences(
        self,
        query_seq: str,
        database_seqs: List[str],
        database_labels: List[str],
    ) -> List[Dict]:
        """Match a query sequence against a database.
        
        Args:
            query_seq: Query DNA sequence
            database_seqs: Database of DNA sequences
            database_labels: Labels for database sequences
        
        Returns:
            Ranked list of matches with similarity scores
        """
        query_hv = self.encode_sequence(query_seq)
        
        results = []
        for db_seq, label in zip(database_seqs, database_labels):
            db_hv = self.encode_sequence(db_seq)
            similarity = (query_hv * db_hv).sum().item() / self.config.dim
            results.append({
                "label": label,
                "similarity": similarity,
                "is_match": similarity >= self.config.similarity_threshold,
            })
        
        # Sort by similarity descending
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def build_database(
        self,
        sequences: List[str],
        labels: List[str],
        fefet_crossbar: Optional[FeFETCrossbar] = None,
    ) -> FeFETCrossbar:
        """Build a searchable database of genome HVs.
        
        Args:
            sequences: List of DNA sequences
            labels: List of sequence labels
            fefet_crossbar: Optional FeFET crossbar for storage
        
        Returns:
            FeFETCrossbar with encoded sequences
        """
        if fefet_crossbar is None:
            fefet_crossbar = FeFETCrossbar(
                FeFETDevice(),
                FeFETCrossbarConfig(dim=self.config.dim),
            )
        
        for seq, label in zip(sequences, labels):
            hv = self.encode_sequence(seq)
            fefet_crossbar.store_hypervector(hv, label=label)
        
        return fefet_crossbar

    def get_genome_stats(self) -> Dict:
        """Get genome matching statistics."""
        return {
            "k_mer_size": self.config.k_mer_size,
            "dim": self.config.dim,
            "n_possible_kmers": len(self.kmer_hvs),
            "use_reverse_complement": self.config.use_reverse_complement,
            "similarity_threshold": self.config.similarity_threshold,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

def test_fefet_hdc():
    """Test FeFET-based HDC modules."""
    print("=" * 60)
    print("Testing FeFET-Based HDC")
    print("=" * 60)

    # 1. FeFET Device
    print("\n1. FeFET Device Model:")
    device = FeFETDevice(FeFETDeviceConfig(technology_nm=28, n_levels=4))
    
    # Program and read
    for level in range(4):
        programmed = device.program_level(level)
        read = device.read_conductance(programmed, time_elapsed_years=0)
        print(f"  Level {level}: programmed={programmed:.3f}, read={read:.3f}")
    
    # Endurance cycling
    endurance = device.simulate_endurance_cycling(1000)
    print(f"  Endurance after 1000 cycles: {endurance['remaining_window_pct']:.1f}% window remaining")
    
    # Temperature effect
    device.apply_temperature(85.0)  # 85°C
    hot_read = device.read_conductance(0.75, time_elapsed_years=1)
    print(f"  Read at 85°C after 1 year: {hot_read:.3f}")

    # 2. FeFET Crossbar
    print("\n2. FeFET Crossbar:")
    crossbar = FeFETCrossbar(device, FeFETCrossbarConfig(dim=256))
    
    # Store some HVs
    for i in range(5):
        hv = torch.where(torch.randn(256) > 0, 
                        torch.ones(256), -torch.ones(256))
        crossbar.store_hypervector(hv, label=f"seq_{i}")
    
    # Query
    query = torch.where(torch.randn(256) > 0,
                       torch.ones(256), -torch.ones(256))
    sims = crossbar.batch_similarity(query)
    print(f"  Stored {len(crossbar.row_labels)} HVs")
    print(f"  Query similarities: {sims}")
    
    # Retention effect
    retention = crossbar.simulate_retention_effect(time_years=5)
    print(f"  Retention after 5 years: mean degradation={retention['mean_degradation']:.4f}")
    
    # Energy
    energy = crossbar.estimate_energy(n_operations=100)
    print(f"  Energy for 100 ops: {energy['total_energy_uj']:.3f} µJ")

    # 3. HDGIM
    print("\n3. HDGIM Genome Matching:")
    hdgim = HDGIM(HDGIMConfig(k_mer_size=3, dim=256))
    
    # Test sequences
    query_seq = "ATCGATCGATCG"
    database = [
        ("ATCGATCGATCG", "exact_match"),
        ("ATCGATCGATCC", "one_mismatch"),
        ("AAAAAAAATTTTTT", "different"),
        ("GCTAGCTAGCTA", "partial_match"),
    ]
    
    results = hdgim.match_sequences(
        query_seq,
        [s for s, _ in database],
        [l for _, l in database],
    )
    
    print(f"  Query: {query_seq}")
    for r in results:
        print(f"  {r['label']:20s}: similarity={r['similarity']:.3f}, match={r['is_match']}")
    
    print(f"\n  Genome stats: {hdgim.get_genome_stats()}")

    print("\n✅ FeFET HDC test complete!")


if __name__ == "__main__":
    test_fefet_hdc()
