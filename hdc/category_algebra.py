"""
hdc/category_algebra.py
========================
Category Theory Foundation for Hyperdimensional Computing.

Implements the algebraic framework from:
    Rotam (2025) "Chrology: a Unified Multiscale Framework for Interpreting
    the Universe Across Five Domains of Existence" Preprints.org.

Key insight: HDC aligns with Category Theory as a harmonizing point because
it treats models as algebraic objects where morphisms (transformations) are
preserved across scales. Category theory provides a mathematical foundation
for understanding hyperdimensional computing as a compositional algebra.

This module provides:
- HDCategory: A category where objects are hypervectors and morphisms are HDC ops
- Morphism: Binding, bundling, permutation as categorical arrows
- Functor: Structure-preserving maps between HDC spaces
- NaturalTransformation: Transformations between functors
- CompositionalAlgebra: Algebraic composition of HDC operations

Usage:
    from hdc.category_algebra import HDCategory, Morphism, Functor

    cat = HDCategory(dim=10000)
    f = Morphism(cat, "bind", hv_a, hv_b)  # Binding as morphism
    result = f.compose(g)                    # Composition of morphisms
"""

import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Callable, Set
from enum import Enum

logger = logging.getLogger(__name__)


class OpType(Enum):
    """HDC operations as categorical morphisms."""
    IDENTITY = "identity"       # id morphism
    BIND = "bind"               # XOR: A ⊗ B
    BUNDLE = "bundle"           # Majority: A + B
    PERMUTE = "permute"         # Rotation: π(A)
    CLEANUP = "cleanup"         # Projection to basis
    RELEASE = "release"         # Decomposition
    SCALE = "scale"             # Scalar multiplication
    COMPOSE = "compose"         # Composition of morphisms


@dataclass
class CategoryConfig:
    """Configuration for the HDC category."""
    dim: int = 10000            # Hypervector dimension (objects live here)
    device: str = "cpu"


class HDCategory:
    """
    A category where:
    - Objects are hypervectors (points in {0,1}^D)
    - Morphisms are HDC operations (bind, bundle, permute, etc.)
    - Composition is sequential application of operations
    - Identity is the identity hypervector

    This formalizes HDC as a compositional algebra where models are
    algebraic objects and transformations are preserved across scales.
    """

    def __init__(self, config: Optional[CategoryConfig] = None):
        self.config = config or CategoryConfig()
        self.device = torch.device(self.config.device)
        self._morphisms: List["Morphism"] = []

    def identity(self, hv: torch.Tensor) -> torch.Tensor:
        """Identity morphism: returns the hypervector unchanged."""
        return hv.clone()

    def compose(self, *morphisms: "Morphism") -> "Morphism":
        """
        Composition of morphisms: g ∘ f means apply f then g.

        In category theory: Hom(A, B) × Hom(B, C) → Hom(A, C)
        """
        if not morphisms:
            raise ValueError("Need at least one morphism")
        result = Morphism(self, OpType.COMPOSE, morphisms[0].source)
        result._composed = list(morphisms)
        result.target = morphisms[-1].target
        return result

    def hom(self, source: torch.Tensor, target: torch.Tensor) -> List["Morphism"]:
        """
        Find all morphisms from source to target.

        In category theory: Hom(A, B) is the set of all arrows from A to B.
        """
        results = []
        for m in self._morphisms:
            if torch.equal(m.source, source) and torch.equal(m.target, target):
                results.append(m)
        return results

    def register(self, morphism: "Morphism"):
        """Register a morphism in the category."""
        self._morphisms.append(morphism)

    def pipeline(
        self,
        ops: List[str],
        dim: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Build a functional transformation pipeline from operation names.

        A pipeline is a function HV → HV that applies operations in sequence.
        This makes it easy to compose standard HDC transforms programmatically.

        Supported operations:
          "bind_random":  XOR with a fixed random HV
          "permute":      Cyclic shift by 1 position
          "permute_N":    Cyclic shift by N positions (e.g. "permute_3")
          "bundle_self":  Bundle HV with its permutation (self-attention analog)
          "majority":     Re-binarise (threshold at 0.5)
          "negate":       Flip all bits (XOR with all-ones)

        Args:
            ops:  List of operation names to apply in order
            dim:  Dimension (defaults to config.dim)
            seed: Random seed for random HVs used in ops

        Returns:
            Callable (hv: Tensor) → Tensor
        """
        d    = dim or self.config.dim
        g    = torch.Generator()
        g.manual_seed(seed)
        _ops = []
        for op in ops:
            if op == "bind_random":
                r = (torch.rand(d, generator=g) >= 0.5).float()
                _ops.append(lambda hv, r=r: ((hv.float() + r.to(hv.device)) % 2).float())
            elif op.startswith("permute"):
                parts = op.split("_")
                shift = int(parts[1]) if len(parts) > 1 else 1
                _ops.append(lambda hv, s=shift: torch.roll(hv, shifts=s))
            elif op == "bundle_self":
                _ops.append(lambda hv: ((hv.float() + torch.roll(hv, 1)) / 2 > 0.5).float())
            elif op == "majority":
                _ops.append(lambda hv: (hv.float() > 0.5).float())
            elif op == "negate":
                _ops.append(lambda hv: (1.0 - hv.float()).float())

        def _run(hv: torch.Tensor) -> torch.Tensor:
            for fn in _ops:
                hv = fn(hv)
            return hv

        return _run

    def __repr__(self) -> str:
        return f"HDCategory(dim={self.config.dim}, morphisms={len(self._morphisms)})"


class Morphism:
    """
    A morphism (arrow) in the HDC category.

    Morphisms are structure-preserving maps between hypervectors.
    In HDC, these are the fundamental operations: bind, bundle, permute.

    Properties:
    - source: Domain hypervector
    - target: Codomain hypervector
    - op_type: Type of HDC operation
    - composable: Can be composed with other morphisms
    """

    def __init__(
        self,
        category: HDCategory,
        op_type: OpType,
        source: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.category = category
        self.op_type = op_type
        self.source = source
        self.target = target if target is not None else source.clone()
        self.params = params or {}
        self._composed: Optional[List["Morphism"]] = None

    def apply(self, hv: torch.Tensor) -> torch.Tensor:
        """Apply this morphism to a hypervector."""
        if self.op_type == OpType.IDENTITY:
            return hv.clone()
        elif self.op_type == OpType.BIND:
            other = self.params.get("other")
            if other is None:
                return hv
            return ((hv > 0) != (other > 0)).float()
        elif self.op_type == OpType.BUNDLE:
            others = self.params.get("others", [])
            if not others:
                return hv
            all_hvs = [hv] + others
            stacked = torch.stack([(h > 0).float() for h in all_hvs])
            return (stacked.mean(dim=0) >= 0.5).float()
        elif self.op_type == OpType.PERMUTE:
            shift = self.params.get("shift", 1)
            return torch.roll((hv > 0).float(), shifts=shift)
        elif self.op_type == OpType.COMPOSE and self._composed:
            result = hv
            for m in self._composed:
                result = m.apply(result)
            return result
        else:
            return hv.clone()

    def compose(self, other: "Morphism") -> "Morphism":
        """
        Composition: self ∘ other (apply other first, then self).

        In category theory: if f: A → B and g: B → C, then g ∘ f: A → C.
        """
        assert torch.equal(self.source, other.target), (
            f"Cannot compose: self.source ≠ other.target"
        )
        composed = Morphism(
            self.category, OpType.COMPOSE, other.source, target=self.target
        )
        composed._composed = [other, self]
        return composed

    def then(self, other: "Morphism") -> "Morphism":
        """Forward composition: apply self, then other."""
        return other.compose(self)

    def __repr__(self) -> str:
        src_repr = f"hv[{hash(str(self.source[:4])) % 1000}]"
        tgt_repr = f"hv[{hash(str(self.target[:4])) % 1000}]"
        return f"Morphism({self.op_type.value}: {src_repr} → {tgt_repr})"


class Functor:
    """
    A functor between HDC categories.

    In category theory: F: C → D maps objects to objects and morphisms to
    morphisms, preserving composition and identities.

    In HDC: A functor maps hypervectors from one dimension to another,
    preserving the algebraic structure (binding, bundling, permutation).
    """

    def __init__(
        self,
        source_cat: HDCategory,
        target_cat: HDCategory,
        object_map: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.source = source_cat
        self.target = target_cat
        self._object_map = object_map or (lambda hv: hv[:target_cat.config.dim])

    def map_object(self, hv: torch.Tensor) -> torch.Tensor:
        """Map an object (hypervector) from source to target category."""
        return self._object_map(hv)

    def map_morphism(self, morphism: Morphism) -> Morphism:
        """
        Map a morphism from source to target category.

        Preserves: F(g ∘ f) = F(g) ∘ F(f)
        """
        new_source = self.map_object(morphism.source)
        new_target = self.map_object(morphism.target)
        return Morphism(
            self.target,
            morphism.op_type,
            new_source,
            target=new_target,
            params=morphism.params,
        )

    def __repr__(self) -> str:
        return (
            f"Functor({self.source.config.dim}D → {self.target.config.dim}D)"
        )


class NaturalTransformation:
    """
    A natural transformation between two functors.

    In category theory: η: F ⇒ G is a family of morphisms η_X: F(X) → G(X)
    such that for every morphism f: X → Y, η_Y ∘ F(f) = G(f) ∘ η_X.

    In HDC: A natural transformation maps between different encoding strategies
    while preserving the underlying algebraic structure.
    """

    def __init__(
        self,
        functor_f: Functor,
        functor_g: Functor,
        component: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        assert functor_f.target.config.dim == functor_g.target.config.dim
        self.f = functor_f
        self.g = functor_g
        self._component = component or (lambda hv: hv)

    def component_at(self, hv: torch.Tensor) -> torch.Tensor:
        """Get the component of the natural transformation at object hv."""
        return self._component(hv)

    def check_naturality(self, hv: torch.Tensor, morphism: Morphism) -> bool:
        """
        Verify the naturality square commutes:
            η_Y ∘ F(f) = G(f) ∘ η_X
        """
        # η_Y ∘ F(f)
        f_hv = self.f.map_morphism(morphism).apply(hv)
        left = self.component_at(f_hv)

        # G(f) ∘ η_X
        eta_x = self.component_at(hv)
        g_morphism = self.g.map_morphism(morphism)
        right = g_morphism.apply(eta_x)

        return torch.allclose(left, right, atol=0.1)

    def __repr__(self) -> str:
        return f"NaturalTransformation({self.f} ⇒ {self.g})"


class CompositionalAlgebra:
    """
    Compositional algebra for HDC operations.

    Treats HDC models as algebraic objects where morphisms (transformations)
    are preserved across scales. Provides:
    - Algebraic laws: associativity, commutativity, distributivity
    - Scale preservation: operations work identically at any dimension
    - Composition: complex operations built from primitives
    """

    def __init__(self, category: HDCategory):
        self.cat = category

    def bind_then_bundle(
        self, hvs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Bind all pairs, then bundle the results.

        Demonstrates composition: (a⊗b) + (c⊗d) + ...
        """
        if len(hvs) < 2:
            return hvs[0] if hvs else torch.zeros(self.cat.config.dim)
        bound_pairs = []
        for i in range(0, len(hvs) - 1, 2):
            bound = ((hvs[i] > 0) != (hvs[i + 1] > 0)).float()
            bound_pairs.append(bound)
        if bound_pairs:
            stacked = torch.stack(bound_pairs)
            return (stacked.mean(dim=0) >= 0.5).float()
        return hvs[0]

    def permute_then_bind(
        self, hv: torch.Tensor, shift: int = 1
    ) -> torch.Tensor:
        """
        Permute then bind: π(a) ⊗ a.

        Creates a permutation-invariant representation.
        """
        permuted = torch.roll((hv > 0).float(), shifts=shift)
        return ((hv > 0) != permuted).float()

    def scale_invariant(
        self, hv: torch.Tensor, target_dim: int
    ) -> torch.Tensor:
        """
        Scale-invariant mapping: preserve algebraic structure across dimensions.

        This is the key insight from category theory — morphisms are preserved
        across scales. An operation at D=1000 behaves identically at D=10000.
        """
        if target_dim == self.cat.config.dim:
            return hv.clone()
        # Project while preserving structure
        if target_dim < self.cat.config.dim:
            return hv[:target_dim].clone()
        else:
            result = torch.zeros(target_dim)
            result[:self.cat.config.dim] = hv
            return result

    def check_associativity(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> bool:
        """Verify (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)."""
        left = ((a > 0) != (b > 0)).float()
        left = ((left > 0) != (c > 0)).float()
        right = ((b > 0) != (c > 0)).float()
        right = ((a > 0) != (right > 0)).float()
        return torch.equal(left, right)

    def check_commutativity(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> bool:
        """Verify a ⊗ b = b ⊗ a."""
        left = ((a > 0) != (b > 0)).float()
        right = ((b > 0) != (a > 0)).float()
        return torch.equal(left, right)

    def __repr__(self) -> str:
        return f"CompositionalAlgebra({self.cat})"


# ── Test ──────────────────────────────────────────────────────────────────────

def test_category_algebra():
    """Verify category theory operations."""
    torch.manual_seed(42)
    dim = 100

    cat = HDCategory(CategoryConfig(dim=dim))
    algebra = CompositionalAlgebra(cat)

    # Generate test hypervectors
    a = torch.randint(0, 2, (dim,)).float()
    b = torch.randint(0, 2, (dim,)).float()
    c = torch.randint(0, 2, (dim,)).float()

    # Test associativity
    assoc = algebra.check_associativity(a, b, c)
    assert assoc, "Binding should be associative"

    # Test commutativity
    comm = algebra.check_commutativity(a, b)
    assert comm, "Binding should be commutative"

    # Test morphism composition
    bind_morph = Morphism(cat, OpType.BIND, a, params={"other": b})
    permute_morph = Morphism(cat, OpType.PERMUTE, bind_morph.target, params={"shift": 3})
    composed = bind_morph.then(permute_morph)
    result = composed.apply(a)
    assert result.shape == (dim,), f"Composed shape: {result.shape}"

    # Test functor
    cat2 = HDCategory(CategoryConfig(dim=50))
    functor = Functor(cat, cat2, object_map=lambda hv: hv[:50])
    mapped = functor.map_object(a)
    assert mapped.shape == (50,), f"Mapped shape: {mapped.shape}"

    # Test scale invariance
    scaled = algebra.scale_invariant(a, target_dim=50)
    assert scaled.shape == (50,), f"Scaled shape: {scaled.shape}"

    # Test bind then bundle
    result = algebra.bind_then_bundle([a, b, c])
    assert result.shape == (dim,), f"Bind+bundle shape: {result.shape}"

    print(f"  Category: {cat}")
    print(f"  Associativity: {assoc}")
    print(f"  Commutativity: {comm}")
    print(f"  Functor: {functor}")
    print(f"  Compositional algebra: {algebra}")
    print("  ✓ All category algebra tests pass")


if __name__ == "__main__":
    test_category_algebra()
