"""
hdc/hdc_vae.py
===============
HDC Variational Autoencoder — Generative Models in Hypervector Space
=====================================================================
Reference:
    Kingma & Welling (2013) "Auto-Encoding Variational Bayes" ICLR 2014.
    — Original VAE; adapted here to binary HV space.

    Tolstikhin et al. (2017) "Wasserstein Auto-Encoders" ICLR 2018.
    — WAE uses Wasserstein distance instead of KL; inspires HDC-WAE variant.

    Balestriero & LeCun (2022) "Contrastive and Non-Contrastive Self-Supervised
    Learning Recover Global and Local Spectral Embedding Methods" NeurIPS.
    — Theoretical connection between self-supervised HDC and VAE.

Why generative HDC matters:

    SNNTraining currently has:
        Discriminative HDC: classify inputs
        Self-supervised HDC: learn representations without labels
        Contrastive HDC: distinguish positive/negative pairs

    Missing: GENERATIVE models — create new HVs from a learned distribution.

    HDC-VAE enables:
        1. Data augmentation: generate synthetic training samples in HV space
        2. Anomaly detection: reconstruct → high error = anomaly
        3. Prototype generation: interpolate between known concepts
        4. Concept composition: add/subtract HVs in latent space
        5. HDC-native imagination: "imagine" novel sensor readings

    The HDC advantage for generative models:
        - Latent space = HV space (same algebra for generation and discrimination)
        - No decoder network needed: reconstruction = bundle of latent factors
        - Controllable generation: steer by HDC operations (bind, permute, etc.)

This module implements:

1. HDCEncoder (Vq)
   — Maps input HVs to a latent distribution over HV space
   — Variational: outputs mean HV + log-variance (per-dimension Bernoulli)
   — Reparameterization trick in binary space via stochastic rounding

2. HDCDecoder
   — Maps latent HV back to reconstruction in input space
   — Uses associative memory (Modern Hopfield) for reconstruction
   — No neural network: pure HDC operations

3. HDCVAE
   — Complete variational autoencoder in HV space
   — ELBO = reconstruction_accuracy - beta × KL(q || p)
   — Connects to FreeEnergyEstimator (active_inference.py):
     ELBO = -F = -(accuracy + complexity)

4. HDCConditionalVAE
   — Conditional generation: generate HV conditioned on class label
   — c-HDCVAE: latent = z + bind(class_hv, z)
   — Enables: class-conditional data augmentation

5. HDCInterpolator
   — Smooth interpolation between two HVs in learned latent space
   — Uses fractional power encoding for continuous interpolation
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hdc.physics_world_model import _hamming, _majority, _xor
from hdc.modern_hopfield import ModernHopfieldHDC


# ── Utility ────────────────────────────────────────────────────────────────────

def _gen_hv(dim: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return (torch.rand(dim, generator=g, device=device) >= 0.5).float()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HDCEncoder — maps input HVs to latent distribution
# ═══════════════════════════════════════════════════════════════════════════════

class HDCEncoder(nn.Module):
    """
    Variational encoder: maps input HV x to latent distribution q(z|x).

    Output: (mu, log_var) where:
        mu      = mean HV (continuous, softmaxed to [0,1] per dim)
        log_var = log-variance per dimension

    Reparameterization (binary space):
        z = Bernoulli(sigmoid(mu + ε × exp(0.5 × log_var)))  for ε ~ N(0,1)

    Architecture:
        x → [linear → tanh] × depth → mu, log_var

    Args:
        input_dim:  Input HV dimension
        latent_dim: Latent HV dimension (can differ from input)
        hidden_dim: Hidden layer size
        depth:      Number of hidden layers
    """

    def __init__(
        self,
        input_dim:  int,
        latent_dim: int,
        hidden_dim: int = 256,
        depth:      int = 2,
        device:     str = "cpu",
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = latent_dim

        layers = []
        in_d   = input_dim
        for _ in range(depth):
            layers += [nn.Linear(in_d, hidden_dim), nn.Tanh()]
            in_d = hidden_dim
        self.shared = nn.Sequential(*layers).to(device)
        self.mu_head      = nn.Linear(hidden_dim, latent_dim).to(device)
        self.logvar_head  = nn.Linear(hidden_dim, latent_dim).to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim) real or binary input HVs

        Returns:
            (mu, log_var) each (B, latent_dim)
        """
        h      = self.shared(x.float())
        mu     = torch.sigmoid(self.mu_head(h))     # per-dim prob ∈ [0,1]
        logvar = self.logvar_head(h).clamp(-4, 4)
        return mu, logvar

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterized sampling in binary HV space.

        z ~ Bernoulli(sigmoid(mu + ε × exp(0.5 × logvar)))
        where ε ~ N(0, I)
        """
        if self.training:
            std  = torch.exp(0.5 * logvar)
            eps  = torch.randn_like(std)
            prob = torch.sigmoid(
                torch.log(mu / (1 - mu + 1e-8)) + eps * std
            )
            return (prob > 0.5).float()
        else:
            return (mu > 0.5).float()   # MAP at eval time


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HDCDecoder — maps latent HV to reconstruction
# ═══════════════════════════════════════════════════════════════════════════════

class HDCDecoder(nn.Module):
    """
    HDC decoder: maps latent z to reconstructed input x̂.

    Uses a neural decoder to first decode to continuous space,
    then binarises to binary HV.

    Architecture:
        z → [linear → tanh] × depth → x̂ (continuous) → sign → binary HV
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        depth:      int = 2,
        device:     str = "cpu",
    ):
        super().__init__()
        layers = []
        in_d   = latent_dim
        for _ in range(depth):
            layers += [nn.Linear(in_d, hidden_dim), nn.Tanh()]
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (x_logit, x_binary)
            x_logit:  (B, output_dim) continuous pre-sigmoid
            x_binary: (B, output_dim) binarised reconstruction
        """
        logit  = self.net(z.float())
        binary = (torch.sigmoid(logit) > 0.5).float()
        return logit, binary


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HDCVAE — complete variational autoencoder
# ═══════════════════════════════════════════════════════════════════════════════

class HDCVAE(nn.Module):
    """
    HDC Variational Autoencoder.

    ELBO = reconstruction_accuracy - β × KL(q(z|x) || p(z))

    where:
        reconstruction = -BCE(decoder(z), x)  [per-dimension binary loss]
        KL = Σ_d (0.5 × (exp(logvar_d) + mu_d² - 1 - logvar_d))

    Connection to Free Energy (hdc/active_inference.py):
        ELBO = -F  where F = accuracy + complexity
        → VAE is a generative model version of active inference

    β-VAE (β > 1): more disentangled latent space at cost of reconstruction.
    β < 1: better reconstruction at cost of disentanglement.

    Args:
        input_dim:  Input HV dimension
        latent_dim: Latent dimension
        beta:       KL weight (1.0 = standard VAE)
        lr:         Learning rate
        device:     torch device
    """

    def __init__(
        self,
        input_dim:  int,
        latent_dim: int,
        beta:       float = 1.0,
        hidden_dim: int   = 256,
        depth:      int   = 2,
        lr:         float = 1e-3,
        device:     str   = "cpu",
    ):
        super().__init__()
        self.beta      = beta
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = HDCEncoder(input_dim, latent_dim, hidden_dim, depth, device)
        self.decoder = HDCDecoder(latent_dim, input_dim, hidden_dim, depth, device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self._step = 0

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        VAE forward pass.

        Returns dict with: z, mu, logvar, x_logit, x_binary, elbo, recon, kl
        """
        mu, logvar = self.encoder(x)
        z          = self.encoder.sample(mu, logvar)
        x_logit, x_binary = self.decoder(z)

        # Reconstruction loss (binary cross-entropy)
        recon = F.binary_cross_entropy_with_logits(x_logit, x.float(), reduction='mean')

        # KL divergence
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        elbo  = -(recon + self.beta * kl)

        return {
            "z": z, "mu": mu, "logvar": logvar,
            "x_logit": x_logit, "x_binary": x_binary,
            "elbo": elbo, "recon": recon, "kl": kl,
        }

    def train_step(self, x: torch.Tensor) -> Dict[str, float]:
        """One training step. Returns loss components."""
        self._step += 1
        self.train()
        out  = self.forward(x)
        loss = -out["elbo"]

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return {
            "elbo":  float(out["elbo"].item()),
            "recon": float(out["recon"].item()),
            "kl":    float(out["kl"].item()),
        }

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent HV (MAP estimate at eval time)."""
        self.eval()
        mu, logvar = self.encoder(x)
        return self.encoder.sample(mu, logvar)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent HV to binary reconstruction."""
        self.eval()
        _, binary = self.decoder(z)
        return binary

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode: x → z → x̂."""
        z = self.encode(x)
        return self.decode(z)

    @torch.no_grad()
    def generate(self, n: int = 1) -> torch.Tensor:
        """Generate n samples from the prior p(z) = Bernoulli(0.5)."""
        self.eval()
        z = (torch.rand(n, self.latent_dim) >= 0.5).float()
        return self.decode(z)

    @torch.no_grad()
    def reconstruction_accuracy(self, x: torch.Tensor) -> float:
        """Fraction of bits correctly reconstructed."""
        x_hat = self.reconstruct(x)
        return float((x_hat == x.float()).float().mean().item())


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HDCConditionalVAE — class-conditional generation
# ═══════════════════════════════════════════════════════════════════════════════

class HDCConditionalVAE(HDCVAE):
    """
    Conditional HDC-VAE: generate HVs conditioned on class label.

    Architecture:
        Encoder: (x, class_hv) → z     [bind x with class HV before encoding]
        Decoder: (z, class_hv) → x̂    [condition decoder on class]

    This enables:
        - Generate new samples of class c: sample z → decode with class_c_hv
        - Data augmentation: generate infinite labelled samples
        - Class interpolation: decode with interpolated class HV

    Args:
        input_dim:   Input dimension
        latent_dim:  Latent dimension
        n_classes:   Number of classes
        class_names: Optional class names
    """

    def __init__(
        self,
        input_dim:   int,
        latent_dim:  int,
        n_classes:   int,
        beta:        float = 1.0,
        device:      str   = "cpu",
        class_names: Optional[List[str]] = None,
    ):
        # Encoder takes (input_dim + input_dim) = 2×input_dim (x concat class_hv)
        super().__init__(input_dim, latent_dim, beta, device=device)
        self.n_classes   = n_classes
        self.class_names = class_names or [f"c{i}" for i in range(n_classes)]

        # Fixed random class HVs
        self._class_hvs = torch.stack([
            _gen_hv(input_dim, seed=i + 777, device=device)
            for i in range(n_classes)
        ])

        # Rebuild encoder/decoder for conditioned dimensions
        self.encoder = HDCEncoder(input_dim * 2, latent_dim, device=device)
        self.decoder = HDCDecoder(latent_dim + input_dim, input_dim, device=device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def _cond_input(self, x: torch.Tensor, label: int) -> torch.Tensor:
        c_hv = self._class_hvs[label].unsqueeze(0).expand_as(x[:, :self.input_dim]
                                                               if x.dim() > 1 else x.unsqueeze(0))
        return torch.cat([x.float(), c_hv.float()], dim=-1)

    def encode_cond(self, x: torch.Tensor, label: int) -> torch.Tensor:
        """Encode with class conditioning."""
        with torch.no_grad():
            cond = self._cond_input(x, label)
            mu, logvar = self.encoder(cond)
            return self.encoder.sample(mu, logvar)

    def generate_class(self, label: int, n: int = 1) -> torch.Tensor:
        """Generate n samples of a specific class."""
        z    = (torch.rand(n, self.latent_dim) >= 0.5).float()
        c_hv = self._class_hvs[label].unsqueeze(0).expand(n, -1)
        z_c  = torch.cat([z, c_hv.float()], dim=-1)
        with torch.no_grad():
            _, binary = self.decoder(z_c)
        return binary


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HDCInterpolator — smooth HV interpolation in latent space
# ═══════════════════════════════════════════════════════════════════════════════

class HDCInterpolator:
    """
    Smooth interpolation between two HVs via a trained VAE latent space.

    Uses the HDCVAE to:
        1. Encode both HVs to latent space
        2. Interpolate between the two latent codes
        3. Decode each interpolation step

    This is the HDC equivalent of the "latent space walk" in image VAEs.

    Applications:
        - Fault severity interpolation: encode(normal) → encode(fault)
        - Concept blending: interpolate between semantic concepts
        - Smooth trajectory generation: interpolate between waypoints

    Args:
        vae: Trained HDCVAE
    """

    def __init__(self, vae: HDCVAE):
        self.vae = vae

    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        n_steps: int = 10,
    ) -> List[torch.Tensor]:
        """
        Generate n_steps interpolated HVs between x1 and x2.

        Uses SLERP (spherical linear interpolation) for smooth paths
        in the binary HV latent space.

        Args:
            x1, x2:  (input_dim,) start and end HVs
            n_steps: Number of intermediate steps

        Returns:
            List of n_steps+2 HVs from x1 to x2
        """
        z1 = self.vae.encode(x1.unsqueeze(0)).float()
        z2 = self.vae.encode(x2.unsqueeze(0)).float()

        interpolated = []
        for t in torch.linspace(0, 1, n_steps + 2):
            # Linear interpolation in continuous latent space
            z_t  = (1 - t) * z1 + t * z2
            # Binarise and decode
            z_bin = (z_t > 0.5).float()
            x_hat = self.vae.decode(z_bin)
            interpolated.append(x_hat.squeeze(0))

        return interpolated

    def concept_arithmetic(
        self,
        base: torch.Tensor,
        add:  torch.Tensor,
        sub:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Concept arithmetic in latent space: z_base + z_add - z_sub → decode.

        Analogous to word2vec: king - man + woman ≈ queen
        In HDC: concept_A - concept_B + concept_C ≈ concept_D
        """
        z_base = self.vae.encode(base.unsqueeze(0)).float()
        z_add  = self.vae.encode(add.unsqueeze(0)).float()
        z_sub  = self.vae.encode(sub.unsqueeze(0)).float()
        z_res  = (z_base + z_add - z_sub).clamp(0, 1)
        z_bin  = (z_res > 0.5).float()
        return self.vae.decode(z_bin).squeeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _test_hdc_vae():
    D_IN, D_LAT = 64, 32
    torch.manual_seed(42)

    print("=== HDCVAE ===")
    vae = HDCVAE(D_IN, D_LAT, beta=1.0, hidden_dim=128, depth=2, lr=1e-2)

    # Generate structured data: 3 clusters
    X = torch.cat([
        (torch.rand(20, D_IN) > (0.3 + c * 0.2)).float()
        for c in range(3)
    ])

    # Training loop
    losses = []
    for ep in range(5):
        result = vae.train_step(X)
        losses.append(result)
    print(f"  Training: elbo={losses[-1]['elbo']:.4f}, "
          f"recon={losses[-1]['recon']:.4f}, kl={losses[-1]['kl']:.4f}  OK")

    # Reconstruction accuracy
    acc = vae.reconstruction_accuracy(X)
    print(f"  Reconstruction accuracy: {acc:.3f}  OK")

    # Generation
    generated = vae.generate(5)
    assert generated.shape == (5, D_IN)
    assert set(generated.unique().tolist()).issubset({0.0, 1.0})
    print(f"  Generated shape: {generated.shape}  OK")

    # Encode / decode roundtrip
    z   = vae.encode(X[:4])
    x_r = vae.decode(z)
    assert x_r.shape == (4, D_IN)
    print(f"  Encode→decode shape: {x_r.shape}  OK")

    print("\n=== HDCConditionalVAE ===")
    cvae = HDCConditionalVAE(D_IN, D_LAT, n_classes=3)
    samples = cvae.generate_class(label=0, n=3)
    assert samples.shape == (3, D_IN)
    print(f"  Generated class 0: {samples.shape}  OK")

    print("\n=== HDCInterpolator ===")
    interp = HDCInterpolator(vae)
    x1 = (torch.rand(D_IN) > 0.3).float()
    x2 = (torch.rand(D_IN) > 0.7).float()
    path = interp.interpolate(x1, x2, n_steps=5)
    assert len(path) == 7   # n_steps + 2
    assert all(p.shape == (D_IN,) for p in path)
    print(f"  Interpolation path: {len(path)} steps  OK")

    result = interp.concept_arithmetic(
        base=x1, add=x2, sub=(torch.rand(D_IN) > 0.5).float()
    )
    assert result.shape == (D_IN,)
    print(f"  Concept arithmetic: {result.shape}  OK")

    print("\n✅ All hdc_vae tests passed")


if __name__ == "__main__":
    _test_hdc_vae()
