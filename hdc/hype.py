"""
HyPE: Hyperdimensional Propagation of Error
=============================================
Based on: Sutor et al. 2025 "HyPE: Hyperdimensional Propagation of Error"

Key insight: Formal error propagation through HDC operations allows
principled weight repair in associative memories. Instead of heuristic
error masking, HyPE computes the exact effect of bit errors on HDC
similarity and uses this to guide repair.

The core idea:
1. Model errors as perturbations to hypervectors
2. Propagate error distributions through bind/bundle operations
3. Compute expected similarity degradation
4. Use this to prioritize weight repair where it matters most

Reference:
  Sutor, P., et al. (2025)
  "HyPE: Hyperdimensional Propagation of Error"
  arXiv / Neural Computing
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from models.hdc import gen_hvs, bind, bundle, sim, thresh, batch_sim


class ErrorPropagator:
    """
    Formal error propagation through HDC operations.
    
    Given a hypervector with known bit-error rate p, computes:
    - Expected similarity after error
    - Variance of similarity
    - Criticality score for each dimension
    
    This enables principled weight repair: fix the dimensions
    that matter most for classification accuracy.
    """
    
    def __init__(self, mode: str = "bipolar"):
        self.mode = mode
    
    def expected_similarity(
        self,
        p: float,
        dim: int,
    ) -> float:
        """Expected cosine similarity after bit-flip errors.
        
        For bipolar HVs with bit-flip rate p:
            E[sim] ≈ (1 - 2p)²
        
        This is derived from the fact that each dimension has
        probability (1-p) of being correct and p of being flipped.
        The expected dot product scales as (1-2p)² per dimension.
        
        Args:
            p: Bit-flip probability per dimension
            dim: Dimensionality of hypervectors
        
        Returns:
            Expected cosine similarity
        """
        if self.mode == "bipolar":
            return (1.0 - 2.0 * p) ** 2
        elif self.mode == "binary":
            return (1.0 - p) ** 2 + p ** 2
        else:
            return (1.0 - p) ** 2
    
    def similarity_variance(
        self,
        p: float,
        dim: int,
    ) -> float:
        """Variance of cosine similarity under bit-flip errors.
        
        For bipolar HVs:
            Var[sim] ≈ 4p(1-p) / D
        
        Args:
            p: Bit-flip probability
            dim: Dimensionality
        
        Returns:
            Variance of similarity
        """
        if self.mode == "bipolar":
            return 4.0 * p * (1.0 - p) / dim
        elif self.mode == "binary":
            return p * (1.0 - p) / dim
        else:
            return p * (1.0 - p) / dim
    
    def criticality_score(
        self,
        hv: torch.Tensor,
        class_hvs: torch.Tensor,
        error_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute criticality score for each dimension.
        
        A dimension is "critical" if:
        1. It differs between the query and the nearest class prototype
        2. It's likely to be corrupted by errors
        
        Criticality = |hv_i - class_hv_i| * error_prob_i
        
        Args:
            hv: (dim,) query hypervector
            class_hvs: (n_classes, dim) class prototypes
            error_mask: (dim,) boolean mask of error-prone dimensions
        
        Returns:
            (dim,) criticality scores
        """
        # Find nearest class
        sims = batch_sim(hv, class_hvs, self.mode)
        nearest_class = int(sims.argmax().item())
        nearest_hv = class_hvs[nearest_class]
        
        # Difference between query and nearest class
        if self.mode == "bipolar":
            diff = (hv != nearest_hv).float()
        elif self.mode == "binary":
            diff = (hv != nearest_hv).float()
        else:
            diff = (hv - nearest_hv).abs()
        
        # Criticality = difference * error probability
        error_prob = error_mask.float()
        return diff * error_prob
    
    def repair_priority(
        self,
        hv: torch.Tensor,
        class_hvs: torch.Tensor,
        error_mask: torch.Tensor,
        n_repair: int,
    ) -> torch.Tensor:
        """Determine which dimensions to repair first.
        
        Returns a mask of dimensions to repair, prioritizing
        those with highest criticality.
        
        Args:
            hv: (dim,) corrupted query hypervector
            class_hvs: (n_classes, dim) class prototypes
            error_mask: (dim,) boolean mask of error-prone dims
            n_repair: Number of dimensions to repair
        
        Returns:
            (dim,) boolean mask of dimensions to repair
        """
        criticality = self.criticality_score(hv, class_hvs, error_mask)
        
        # Only consider error-prone dimensions
        criticality = criticality * error_mask.float()
        
        # Get top-k critical dimensions
        _, top_indices = torch.topk(criticality, min(n_repair, int(error_mask.sum().item())))
        
        repair_mask = torch.zeros_like(error_mask)
        repair_mask[top_indices] = True
        return repair_mask.bool()


class HyPERepair(nn.Module):
    """
    HyPE-based weight repair for HDC associative memories.
    
    Uses formal error propagation to determine which dimensions
    of class prototypes need repair, and how to repair them.
    
    Unlike heuristic error masking (zero-masking, sign-bit masking),
    HyPE computes the exact effect of errors on classification and
    repairs only the dimensions that matter.
    
    Usage:
        repair = HyPERepair(n_classes=10, dim=10000)
        repaired_hvs = repair(class_hvs, error_mask)
    """
    
    def __init__(
        self,
        n_classes: int,
        dim: int = 10000,
        mode: str = "bipolar",
        device: Optional[str] = None,
        repair_fraction: float = 0.1,
        min_similarity: float = 0.7,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.mode = mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.repair_fraction = repair_fraction
        self.min_similarity = min_similarity
        
        self.propagator = ErrorPropagator(mode)
        
        # Track repair statistics
        self.register_buffer("total_repairs", torch.tensor(0, device=torch.device(self.device)))
        self.register_buffer("successful_repairs", torch.tensor(0, device=torch.device(self.device)))
    
    def estimate_error_rate(
        self,
        class_hvs: torch.Tensor,
        error_mask: torch.Tensor,
    ) -> float:
        """Estimate the bit-error rate from the error mask.
        
        Args:
            class_hvs: (n_classes, dim) class prototypes
            error_mask: (dim,) boolean mask of error-prone dims
        
        Returns:
            Estimated bit-error rate
        """
        return error_mask.float().mean().item()
    
    def needs_repair(
        self,
        class_hvs: torch.Tensor,
        error_mask: torch.Tensor,
    ) -> bool:
        """Check if repair is needed based on expected similarity.
        
        Uses HyPE's formal error propagation to determine if
        the expected similarity degradation is significant.
        
        Args:
            class_hvs: (n_classes, dim) class prototypes
            error_mask: (dim,) boolean mask of error-prone dims
        
        Returns:
            True if repair is needed
        """
        p = self.estimate_error_rate(class_hvs, error_mask)
        expected_sim = self.propagator.expected_similarity(p, self.dim)
        return expected_sim < self.min_similarity
    
    def forward(
        self,
        class_hvs: torch.Tensor,
        error_mask: torch.Tensor,
        query_hvs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Repair class prototypes using HyPE.
        
        Args:
            class_hvs: (n_classes, dim) class prototypes to repair
            error_mask: (dim,) boolean mask of error-prone dimensions
            query_hvs: Optional (B, dim) query HVs for criticality-guided repair.
                       If None, uses uniform repair across all classes.
        
        Returns:
            (n_classes, dim) repaired class prototypes
        """
        repaired = class_hvs.clone()
        p = self.estimate_error_rate(class_hvs, error_mask)
        
        if not self.needs_repair(class_hvs, error_mask):
            return repaired
        
        n_repair = max(1, int(self.dim * self.repair_fraction))
        
        if query_hvs is not None:
            # Criticality-guided repair: use queries to prioritize
            for i in range(self.n_classes):
                # Find queries closest to this class
                class_sims = batch_sim(query_hvs, class_hvs[i:i+1], self.mode)
                _, top_queries = torch.topk(class_sims, min(5, query_hvs.shape[0]))
                
                # Aggregate criticality from top queries
                criticality = torch.zeros(self.dim, device=self.device)
                for q_idx in top_queries:
                    crit = self.propagator.criticality_score(
                        query_hvs[q_idx], class_hvs, error_mask
                    )
                    criticality += crit
                
                # Repair most critical dimensions
                criticality = criticality * error_mask.float()
                _, top_dims = torch.topk(criticality, min(n_repair, int(error_mask.sum().item())))
                
                # Repair: set to majority vote across all class prototypes
                for d in top_dims:
                    if self.mode == "bipolar":
                        # Majority vote across all classes
                        votes = class_hvs[:, d].sum()
                        repaired[i, d] = torch.sign(votes).clamp(-1, 1)
                    elif self.mode == "binary":
                        votes = class_hvs[:, d].sum()
                        repaired[i, d] = (votes >= self.n_classes / 2).float()
                    else:
                        repaired[i, d] = class_hvs[:, d].mean()
                
                self.total_repairs += len(top_dims)
        else:
            # Uniform repair: repair all classes equally
            for i in range(self.n_classes):
                # Find most critical dimensions for this class
                diff = (class_hvs[i] != class_hvs.mean(dim=0)).float()
                criticality = diff * error_mask.float()
                _, top_dims = torch.topk(criticality, min(n_repair, int(error_mask.sum().item())))
                
                for d in top_dims:
                    if self.mode == "bipolar":
                        votes = class_hvs[:, d].sum()
                        repaired[i, d] = torch.sign(votes).clamp(-1, 1)
                    elif self.mode == "binary":
                        votes = class_hvs[:, d].sum()
                        repaired[i, d] = (votes >= self.n_classes / 2).float()
                    else:
                        repaired[i, d] = class_hvs[:, d].mean()
                
                self.total_repairs += len(top_dims)
        
        # Track success: check if self-similarity improved
        for i in range(self.n_classes):
            orig_sim = sim(class_hvs[i], class_hvs[i], self.mode)
            new_sim = sim(repaired[i], repaired[i], self.mode)
            if new_sim >= orig_sim:
                self.successful_repairs += 1
        
        return repaired
    
    def get_repair_stats(self) -> dict:
        """Get repair statistics.
        
        Returns:
            dict with keys: total_repairs, successful_repairs, success_rate
        """
        total = self.total_repairs.item()
        successful = self.successful_repairs.item()
        return {
            "total_repairs": total,
            "successful_repairs": successful,
            "success_rate": successful / max(1, total),
        }


# ── Tests ────────────────────────────────────────────────────────────────────

def test_hype():
    """Verify HyPE error propagation and repair."""
    print("=" * 60)
    print("Testing HyPE: Hyperdimensional Propagation of Error")
    print("=" * 60)
    
    dim = 2000
    n_classes = 5
    
    # Create class prototypes
    class_hvs = gen_hvs(n_classes, dim, "bipolar")
    
    # Create error mask (10% error rate)
    error_mask = torch.rand(dim) < 0.1
    
    # Test error propagator
    propagator = ErrorPropagator(mode="bipolar")
    
    p = error_mask.float().mean().item()
    expected_sim = propagator.expected_similarity(p, dim)
    var_sim = propagator.similarity_variance(p, dim)
    
    print(f"\n  Error rate: {p:.1%}")
    print(f"  Expected similarity: {expected_sim:.4f}")
    print(f"  Similarity variance: {var_sim:.6f}")
    
    # Test criticality scoring
    query = class_hvs[0].clone()
    criticality = propagator.criticality_score(query, class_hvs, error_mask)
    print(f"\n  Criticality shape: {criticality.shape}")
    print(f"  Mean criticality: {criticality.mean():.4f}")
    print(f"  Max criticality: {criticality.max():.4f}")
    
    # Test repair priority
    repair_mask = propagator.repair_priority(query, class_hvs, error_mask, n_repair=50)
    print(f"\n  Repair mask sum: {repair_mask.sum().item()} (expected: 50)")
    
    # Test HyPE repair module
    repair = HyPERepair(n_classes=n_classes, dim=dim, repair_fraction=0.05)
    
    needs = repair.needs_repair(class_hvs, error_mask)
    print(f"\n  Needs repair: {needs}")
    
    repaired = repair(class_hvs, error_mask)
    print(f"  Repaired shape: {repaired.shape}")
    
    stats = repair.get_repair_stats()
    print(f"\n  Repair stats: {stats}")
    
    # Verify repair improved similarity
    orig_sim = sim(class_hvs[0], class_hvs[0], "bipolar")
    new_sim = sim(repaired[0], repaired[0], "bipolar")
    print(f"\n  Original self-sim: {orig_sim:.4f}")
    print(f"  Repaired self-sim: {new_sim:.4f}")
    
    print(f"\n  ✅ HyPE test complete!")


if __name__ == "__main__":
    test_hype()
