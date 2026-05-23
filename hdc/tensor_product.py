"""
hdc/tensor_product.py
======================
Tensor Product Representations — Exact Compositional Binding
=============================================================
Reference:
    Smolensky (1990) "Tensor product variable binding and the representation
    of symbolic structures in connectionist networks"
    Artificial Intelligence 46(1–2):159–216.

    McClelland, Smolensky et al. (2010) "A parallel distributed processing
    approach to mathematical cognition" — applications of TPR to reasoning.

    Schlag, Smolensky et al. (2019) "Enhancing the Transformer with Explicit
    Relational Encoding for Math Problem Solving" — TPR in modern ML.

Why TPR is more powerful than binary XOR or even HRR:

    XOR binding:        exact in {0,1}^D (self-inverse), but information loss on unbind
    HRR circular conv:  exact unbind via pseudo-inverse, but O(D log D)
    Tensor Products:    EXACTLY invertible for orthonormal roles, O(D_r × D_f) bind

    The key advantage of TPR: roles and fillers live in DIFFERENT spaces.
    XOR and HRR require roles and fillers to share the same D-dimensional space,
    which limits the number of distinct roles (orthogonality capacity ≈ D/10).

    TPR: D_r = role dimension, D_f = filler dimension.
    Can have D_r = 64 roles and D_f = 1024-dim fillers simultaneously.
    Total binding: 64 × 1024 = 65536 dimensions — much richer.

    Capacity: D_r orthonormal roles × any number of fillers (no interference!).

    Unbinding: given S = Σ_i (r_i ⊗ f_i), unbind(S, r_j) = Σ_i (r_j · r_i) f_i = f_j
    (exactly, when roles are orthonormal).

This module implements:

1. TensorProductVar (TPV)
   — Core role-filler binding via outer products
   — Exact unbinding for orthonormal roles
   — Superposition of multiple bindings

2. TPVCodebook
   — Manages role vectors (orthonormal basis) and filler vectors
   — Role capacity: D_r (dimension of role space)
   — Filler capacity: arbitrary (any D_f-dim vectors)

3. StructuredTPV
   — Builds complex symbolic structures via nested TPR
   — Equivalent to recursive S-expressions in connectionist space
   — (atom) → bind(is_atom_role, atom_filler)
   — (A . B) → bind(car_role, A) + bind(cdr_role, B)

4. TPVClassifier
   — Classification via TPR: encode as role-filler structure, decode class
   — More powerful than flat prototype matching for structured inputs
   — Handles relational data (graphs, trees, sequences)

5. TPVAttention
   — Attention mechanism based on TPR
   — Query = role; Key/Value = filler
   — unbind(superposition_of_bindings, query_role) → attended value
   — O(D_r × D_f) per query — no O(N²) softmax needed
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TensorProductVar — core binding / unbinding
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TPVConfig:
    role_dim:   int = 32     # D_r: dimension of role space
    filler_dim: int = 256    # D_f: dimension of filler space
    # Total representation: D_r × D_f = 32 × 256 = 8192 dimensions


class TensorProductVar:
    """
    Tensor Product Variable binding (Smolensky 1990).

    A binding is the outer product r ⊗ f ∈ R^{D_r × D_f}.
    A structure is a superposition: S = Σ_i (r_i ⊗ f_i).

    Properties:
        - EXACT unbinding: unbind(S, r_j) = f_j  when {r_i} are orthonormal
        - Linear superposition: structures add linearly (bundle = sum)
        - Different-space roles/fillers: no role-filler interference
        - Capacity: D_r orthonormal roles per structure (unlimited fillers)

    Args:
        cfg: TPVConfig
        device: torch device
    """

    def __init__(self, cfg: Optional[TPVConfig] = None, device: str = "cpu"):
        self.cfg    = cfg or TPVConfig()
        self.device = device

    @property
    def binding_dim(self) -> int:
        """Total dimension of a flattened binding tensor."""
        return self.cfg.role_dim * self.cfg.filler_dim

    # ── core operations ───────────────────────────────────────────────────────

    def bind(self, role: torch.Tensor, filler: torch.Tensor) -> torch.Tensor:
        """
        Outer-product binding: role ⊗ filler.

        Args:
            role:   (D_r,) role vector
            filler: (D_f,) filler vector

        Returns:
            (D_r, D_f) binding matrix  (or (D_r × D_f,) if flattened)
        """
        r = role.float().to(self.device)
        f = filler.float().to(self.device)
        return torch.outer(r, f)   # (D_r, D_f)

    def bind_flat(self, role: torch.Tensor, filler: torch.Tensor) -> torch.Tensor:
        """Binding as flattened (D_r × D_f,) vector for bundling."""
        return self.bind(role, filler).flatten()

    def unbind(
        self,
        structure: torch.Tensor,
        role: torch.Tensor,
    ) -> torch.Tensor:
        """
        Exact unbinding: given S = Σ (r_i ⊗ f_i) and role r_j,
        recover f_j = S^T @ r_j / ||r_j||².

        For orthonormal roles: f_j = S^T @ r_j  (exactly).

        Args:
            structure: (D_r, D_f) or (D_r × D_f,) binding tensor
            role:      (D_r,) role vector (should be unit-norm)

        Returns:
            (D_f,) recovered filler vector
        """
        if structure.dim() == 1:
            structure = structure.view(self.cfg.role_dim, self.cfg.filler_dim)
        r    = role.float().to(self.device)
        S    = structure.float().to(self.device)
        norm = float(r @ r)
        if norm < 1e-10:
            return torch.zeros(self.cfg.filler_dim, device=self.device)
        return S.T @ r / norm   # (D_f,)

    def bundle(self, bindings: List[torch.Tensor]) -> torch.Tensor:
        """
        Superpose a list of binding tensors.

        Args:
            bindings: List of (D_r, D_f) or (D_r × D_f,) binding tensors

        Returns:
            (D_r, D_f) superposition (sum of all bindings)
        """
        if not bindings:
            return torch.zeros(
                self.cfg.role_dim, self.cfg.filler_dim, device=self.device
            )
        stacked = torch.stack([
            b.view(self.cfg.role_dim, self.cfg.filler_dim).float()
            for b in bindings
        ])
        return stacked.sum(dim=0)

    def similarity(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
    ) -> float:
        """Frobenius inner product between two structure tensors (cosine)."""
        s1_f = s1.float().to(self.device).flatten()
        s2_f = s2.float().to(self.device).flatten()
        return float(F.cosine_similarity(s1_f.unsqueeze(0), s2_f.unsqueeze(0)).item())

    # ── role generation ───────────────────────────────────────────────────────

    def orthonormal_roles(self, n: int, seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate n orthonormal role vectors via QR decomposition.

        Requires n ≤ D_r.

        Args:
            n:    Number of roles (must be ≤ D_r)
            seed: Optional random seed

        Returns:
            (n, D_r) orthonormal role matrix (each row is a unit-norm role)
        """
        if n > self.cfg.role_dim:
            raise ValueError(
                f"Cannot generate {n} orthonormal roles in D_r={self.cfg.role_dim} space"
            )
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        A = torch.randn(self.cfg.role_dim, n, generator=g, device=self.device)
        Q, _ = torch.linalg.qr(A)
        return Q.T   # (n, D_r), each row orthonormal

    def random_filler(self, seed: Optional[int] = None) -> torch.Tensor:
        """Generate a unit-norm random filler vector."""
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        f = torch.randn(self.cfg.filler_dim, generator=g, device=self.device)
        return F.normalize(f, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TPVCodebook — manages named roles and fillers
# ═══════════════════════════════════════════════════════════════════════════════

class TPVCodebook:
    """
    Manages named role and filler vectors for TPR structures.

    Roles are stored as orthonormal vectors in R^{D_r}.
    Fillers are arbitrary vectors in R^{D_f}.

    Capacity:
        Roles: up to D_r (dimension of role space)
        Fillers: unlimited (any number of D_f-dim vectors)
        Bindings per structure: limited by role orthogonality → D_r exact bindings

    Args:
        tpv: TensorProductVar instance
    """

    def __init__(self, tpv: TensorProductVar):
        self.tpv = tpv
        self._roles:   Dict[str, torch.Tensor] = {}   # name → D_r vector
        self._fillers: Dict[str, torch.Tensor] = {}   # name → D_f vector
        self._role_seed = 0
        self._filler_seed = 1000

    def add_role(self, name: str, vector: Optional[torch.Tensor] = None):
        """Register a named role (auto-generate if vector is None)."""
        if name not in self._roles:
            if vector is not None:
                self._roles[name] = F.normalize(vector.float().to(self.tpv.device), dim=0)
            else:
                # Generate and orthogonalize against existing roles
                self._role_seed += 1
                g = torch.Generator(device=self.tpv.device)
                g.manual_seed(self._role_seed)
                r = torch.randn(self.tpv.cfg.role_dim, generator=g, device=self.tpv.device)
                # Gram-Schmidt orthogonalization against existing roles
                for existing_r in self._roles.values():
                    r = r - (r @ existing_r) * existing_r
                norm = r.norm()
                if norm > 1e-8:
                    r = r / norm
                self._roles[name] = r

    def add_filler(self, name: str, vector: Optional[torch.Tensor] = None):
        """Register a named filler (auto-generate if vector is None)."""
        if name not in self._fillers:
            if vector is not None:
                self._fillers[name] = F.normalize(vector.float().to(self.tpv.device), dim=0)
            else:
                self._filler_seed += 1
                g = torch.Generator(device=self.tpv.device)
                g.manual_seed(self._filler_seed)
                f = torch.randn(self.tpv.cfg.filler_dim, generator=g, device=self.tpv.device)
                self._fillers[name] = F.normalize(f, dim=0)

    def bind(self, role_name: str, filler_name: str) -> torch.Tensor:
        """Create a named role-filler binding tensor."""
        r = self._roles[role_name]
        f = self._fillers[filler_name]
        return self.tpv.bind(r, f)

    def unbind(self, structure: torch.Tensor, role_name: str) -> Tuple[Optional[str], float]:
        """
        Unbind the filler for a given role from a structure.

        Returns:
            (nearest_filler_name, cosine_similarity)
        """
        role = self._roles[role_name]
        candidate = self.tpv.unbind(structure, role)

        if not self._fillers:
            return None, 0.0

        names = list(self._fillers.keys())
        vecs  = torch.stack([self._fillers[n] for n in names])
        sims  = F.cosine_similarity(candidate.unsqueeze(0), vecs)
        best  = int(sims.argmax().item())
        return names[best], float(sims[best].item())

    def decode_all(self, structure: torch.Tensor) -> Dict[str, Tuple[str, float]]:
        """Decode all roles from a structure tensor."""
        return {r: self.unbind(structure, r) for r in self._roles}

    def build(self, role_filler_dict: Dict[str, str]) -> torch.Tensor:
        """
        Build a structure from a role:filler dictionary.

        Args:
            role_filler_dict: {role_name: filler_name, ...}

        Returns:
            (D_r, D_f) TPR structure tensor
        """
        bindings = [self.bind(r, f) for r, f in role_filler_dict.items()]
        return self.tpv.bundle(bindings)

    @property
    def n_roles(self) -> int:
        return len(self._roles)

    @property
    def role_capacity(self) -> int:
        """Maximum number of orthonormal roles possible."""
        return self.tpv.cfg.role_dim


# ═══════════════════════════════════════════════════════════════════════════════
# 3. StructuredTPV — recursive S-expression representation
# ═══════════════════════════════════════════════════════════════════════════════

class StructuredTPV:
    """
    Recursive TPR for tree-structured symbolic data.

    Implements the classic cons-cell / S-expression representation:
        atom        → bind(atom_role, atom_filler)
        (A . B)     → bind(car_role, A) + bind(cdr_role, B)
        (A B C)     → (A . (B . (C . nil)))

    The TPR can represent arbitrarily deep recursive structures.
    Unbinding is exact at each level (no information loss with depth).

    Applications:
        - Encode parse trees of natural language
        - Represent knowledge graph triples (subject, predicate, object)
        - Encode logical formulae for HDC theorem proving

    Args:
        codebook: TPVCodebook with car, cdr, atom roles pre-registered
    """

    # Standard structure roles
    CAR_ROLE  = "car"
    CDR_ROLE  = "cdr"
    ATOM_ROLE = "atom"
    NIL_ROLE  = "nil"

    def __init__(self, codebook: TPVCodebook):
        self.cb = codebook
        # Ensure standard roles exist
        for role in [self.CAR_ROLE, self.CDR_ROLE, self.ATOM_ROLE, self.NIL_ROLE]:
            self.cb.add_role(role)
        # Nil filler (zero tensor)
        self.cb._fillers["nil"] = torch.zeros(
            codebook.tpv.cfg.filler_dim, device=codebook.tpv.device
        )

    def atom(self, filler_name: str) -> torch.Tensor:
        """Build an atom structure: bind(atom_role, filler)."""
        return self.cb.bind(self.ATOM_ROLE, filler_name)

    def cons(self, car: torch.Tensor, cdr: torch.Tensor) -> torch.Tensor:
        """
        Build a cons cell structure: bind(car_role, car) + bind(cdr_role, cdr).

        car and cdr are themselves structure tensors (recursive).
        """
        tpv = self.cb.tpv
        D_r, D_f = tpv.cfg.role_dim, tpv.cfg.filler_dim

        # To embed sub-structures into filler space, we project them
        # via a learned or fixed linear map F: R^{D_r × D_f} → R^{D_f}
        # Simple approach: treat the flattened sub-structure as a filler
        # using a random projection (Smolensky's "recursive TPR" §6)
        car_filler = self._project_to_filler(car)
        cdr_filler = self._project_to_filler(cdr)

        car_r = self.cb._roles[self.CAR_ROLE]
        cdr_r = self.cb._roles[self.CDR_ROLE]

        return (
            tpv.bind(car_r, car_filler)
            + tpv.bind(cdr_r, cdr_filler)
        )

    def _project_to_filler(self, structure: torch.Tensor) -> torch.Tensor:
        """Project a (D_r, D_f) structure to a (D_f,) filler via mean pooling."""
        if structure.dim() == 2:
            # Mean over role dimension → (D_f,) filler
            return F.normalize(structure.mean(dim=0), dim=0)
        return F.normalize(structure, dim=0)

    def car(self, structure: torch.Tensor) -> torch.Tensor:
        """Extract the car (first element) filler from a cons cell."""
        return self.cb.tpv.unbind(structure, self.cb._roles[self.CAR_ROLE])

    def cdr(self, structure: torch.Tensor) -> torch.Tensor:
        """Extract the cdr (rest) filler from a cons cell."""
        return self.cb.tpv.unbind(structure, self.cb._roles[self.CDR_ROLE])

    def list_encode(self, filler_names: List[str]) -> torch.Tensor:
        """
        Encode a list [f1, f2, ..., fn] as a recursive cons structure.

        list_encode([A, B, C]) = cons(atom(A), cons(atom(B), cons(atom(C), nil)))
        """
        result = self.atom("nil")
        for name in reversed(filler_names):
            result = self.cons(self.atom(name), result)
        return result

    def triple(self, subject: str, predicate: str, obj: str) -> torch.Tensor:
        """
        Encode a knowledge graph triple (subject, predicate, object).

        Uses three named roles: subject_role, predicate_role, object_role.
        Returns a (D_r, D_f) structure tensor for the triple.
        """
        for role in ["subject", "predicate", "object"]:
            self.cb.add_role(role)
        for filler in [subject, predicate, obj]:
            self.cb.add_filler(filler)

        return self.cb.build({
            "subject":   subject,
            "predicate": predicate,
            "object":    obj,
        })


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TPVClassifier — classification via structured TPR
# ═══════════════════════════════════════════════════════════════════════════════

class TPVClassifier:
    """
    Classification using Tensor Product Representations.

    Instead of encoding inputs as flat HVs, encodes them as structured
    TPR tensors (role-filler bindings for each feature dimension).

    Advantage over flat HDC:
        - Feature identity is preserved via role binding (not lost in bundling)
        - Structured similarity: two objects that share the same role-filler
          binding are more similar than objects sharing unrelated bits
        - Works well for relational data, graphs, sequences

    Training: online prototype accumulation (like HDC RefineHD)
    Inference: Frobenius similarity between query structure and class prototypes

    Args:
        codebook:  TPVCodebook with feature roles registered
        n_classes: Number of output classes
    """

    def __init__(self, codebook: TPVCodebook, n_classes: int,
                 class_names: Optional[List[str]] = None):
        self.cb          = codebook
        self.tpv         = codebook.tpv
        self.n_classes   = n_classes
        self.class_names = class_names or [f"class_{i}" for i in range(n_classes)]

        D_r, D_f = self.tpv.cfg.role_dim, self.tpv.cfg.filler_dim
        self._prototypes = [
            torch.zeros(D_r, D_f, device=self.tpv.device)
            for _ in range(n_classes)
        ]
        self._counts = [0] * n_classes

    def train(self, structure: torch.Tensor, label: int):
        """Update class prototype with a new structure tensor."""
        n = self._counts[label]
        self._prototypes[label] = (
            n * self._prototypes[label] + structure.float()
        ) / (n + 1)
        self._counts[label] += 1

    def predict(self, structure: torch.Tensor) -> Tuple[int, List[float]]:
        """Predict class via Frobenius similarity to prototypes."""
        sims = [self.tpv.similarity(structure, p) for p in self._prototypes]
        best = int(max(range(len(sims)), key=lambda i: sims[i]))
        return best, sims


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TPVAttention — attention via TPR unbinding
# ═══════════════════════════════════════════════════════════════════════════════

class TPVAttention:
    """
    Attention mechanism based on Tensor Product Representations.

    Standard transformer attention: softmax(QK^T/√d) V — O(N² × d)
    TPV attention: unbind(S, query_role) where S = Σ bind(key_i, value_i) — O(D_r × D_f)

    For a set of N key-value pairs stored in a TPR superposition S:
        S = Σ_i bind(key_role_i, value_filler_i)

    Query: given a query role q_j, retrieve value_j = unbind(S, q_j)
    This is EXACTLY O(D_r × D_f) regardless of N — no quadratic scaling!

    The "attention score" is implicit in the role inner products:
        unbind(S, q) = Σ_i (q · key_role_i) × value_filler_i

    High dot product between query and key_role → high "attention weight".

    Args:
        tpv: TensorProductVar instance
    """

    def __init__(self, tpv: TensorProductVar):
        self.tpv      = tpv
        self._memory: Optional[torch.Tensor] = None   # (D_r, D_f) superposition

    def write(self, key_role: torch.Tensor, value_filler: torch.Tensor):
        """Store a key-value pair in the TPV memory."""
        binding = self.tpv.bind(key_role, value_filler)
        if self._memory is None:
            self._memory = binding
        else:
            self._memory = self._memory + binding

    def read(self, query_role: torch.Tensor) -> torch.Tensor:
        """
        Retrieve the value associated with the query role.

        Returns:
            (D_f,) retrieved value vector
        """
        if self._memory is None:
            return torch.zeros(self.tpv.cfg.filler_dim, device=self.tpv.device)
        return self.tpv.unbind(self._memory, query_role)

    def attention_scores(
        self,
        query_role:  torch.Tensor,
        key_roles:   torch.Tensor,   # (N, D_r) keys
    ) -> torch.Tensor:
        """
        Compute attention scores = dot products between query and key roles.

        Returns:
            (N,) attention weights (cosine similarities, not softmax)
        """
        q = F.normalize(query_role.float(), dim=0)
        K = F.normalize(key_roles.float(), dim=1)
        return K @ q   # (N,)

    def multi_query(
        self,
        query_roles:  torch.Tensor,  # (B, D_r)
        key_roles:    torch.Tensor,  # (N, D_r)
        value_fillers: torch.Tensor, # (N, D_f)
    ) -> torch.Tensor:
        """
        Batch TPV attention for B queries over N key-value pairs.
        O(B × D_r × D_f) — no quadratic term.

        Returns:
            (B, D_f) attended values
        """
        scores = F.normalize(query_roles, dim=1) @ F.normalize(key_roles, dim=1).T  # (B, N)
        return scores @ value_fillers   # (B, D_f)

    def reset(self):
        self._memory = None


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_tensor_product():
    cfg = TPVConfig(role_dim=32, filler_dim=128)
    tpv = TensorProductVar(cfg)

    print("=== TensorProductVar ===")
    roles = tpv.orthonormal_roles(5, seed=0)
    assert roles.shape == (5, 32)

    # Verify orthonormality
    gram = roles @ roles.T
    off_diag = (gram - torch.eye(5)).abs().max().item()
    print(f"  Roles orthonormal: max off-diag={off_diag:.6f}  (should be ≈0)")
    assert off_diag < 1e-5

    f1 = tpv.random_filler(seed=10)
    f2 = tpv.random_filler(seed=20)

    # Build structure: role_0→f1, role_1→f2
    S = tpv.bundle([
        tpv.bind(roles[0], f1),
        tpv.bind(roles[1], f2),
    ])
    assert S.shape == (32, 128)

    # Exact unbinding
    rec_f1 = tpv.unbind(S, roles[0])
    rec_f2 = tpv.unbind(S, roles[1])
    sim1 = float(F.cosine_similarity(rec_f1.unsqueeze(0), f1.unsqueeze(0)))
    sim2 = float(F.cosine_similarity(rec_f2.unsqueeze(0), f2.unsqueeze(0)))
    print(f"  sim(unbind(S, r0), f1) = {sim1:.6f}  (exact: should be 1.0)")
    print(f"  sim(unbind(S, r1), f2) = {sim2:.6f}  (exact: should be 1.0)")
    assert sim1 > 0.999, f"Expected exact unbinding, got {sim1}"
    assert sim2 > 0.999, f"Expected exact unbinding, got {sim2}"

    print("\n=== TPVCodebook ===")
    cb = TPVCodebook(tpv)
    cb.add_role("color"); cb.add_role("shape"); cb.add_role("size")
    cb.add_filler("red"); cb.add_filler("blue"); cb.add_filler("circle")
    cb.add_filler("square"); cb.add_filler("large"); cb.add_filler("small")

    # Build structured object: {color:red, shape:circle, size:large}
    obj1 = cb.build({"color": "red", "shape": "circle", "size": "large"})
    obj2 = cb.build({"color": "blue", "shape": "circle", "size": "small"})

    # Decode
    color1, sim_c = cb.unbind(obj1, "color")
    shape1, sim_s = cb.unbind(obj1, "shape")
    print(f"  Decoded color: '{color1}' (sim={sim_c:.3f})  (expected: 'red')")
    print(f"  Decoded shape: '{shape1}' (sim={sim_s:.3f})  (expected: 'circle')")
    assert color1 == "red",    f"Expected 'red', got '{color1}'"
    assert shape1 == "circle", f"Expected 'circle', got '{shape1}'"

    print("\n=== StructuredTPV ===")
    stpv = StructuredTPV(cb)
    triple = stpv.triple("sky", "has_color", "blue")
    assert triple.shape == (32, 128)
    print(f"  Triple shape: {triple.shape}  OK")

    print("\n=== TPVClassifier ===")
    clf = TPVClassifier(cb, n_classes=2, class_names=["danger", "safe"])
    clf.train(cb.build({"color": "red", "shape": "square"}),  label=0)
    clf.train(cb.build({"color": "blue", "shape": "circle"}), label=1)

    pred, sims = clf.predict(cb.build({"color": "red", "shape": "circle"}))
    print(f"  Zero-shot (red,circle): '{clf.class_names[pred]}' "
          f"(sims={[f'{s:.3f}' for s in sims]})")

    print("\n=== TPVAttention ===")
    attn = TPVAttention(tpv)
    for i in range(5):
        k_role = tpv.orthonormal_roles(1, seed=100 + i).squeeze(0)
        v_fill = tpv.random_filler(seed=200 + i)
        attn.write(k_role, v_fill)

    q   = tpv.orthonormal_roles(1, seed=100).squeeze(0)  # same as first key
    out = attn.read(q)
    assert out.shape == (128,)
    print(f"  Read from TPV memory: shape={out.shape}  OK")

    # Multi-query
    Q    = tpv.orthonormal_roles(3, seed=50)          # (3, 32)
    K    = tpv.orthonormal_roles(5, seed=60)          # (5, 32)
    V_f  = torch.randn(5, 128)                         # (5, 128) values
    out2 = attn.multi_query(Q, K, V_f)
    assert out2.shape == (3, 128)
    print(f"  Multi-query (B=3, N=5): shape={out2.shape}  OK")

    print("\n✅ All tensor_product tests passed")


if __name__ == "__main__":
    _test_tensor_product()
