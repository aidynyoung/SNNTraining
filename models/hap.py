"""Hyperdimensional Active Perception (HAP) for Arthedain.
Based on Section V of Amrouch et al. 2022."""
import torch
import torch.nn as nn
from models.hdc import gen_hvs, bind, batch_sim, thresh


class HAPModule(nn.Module):
    def __init__(self, n_actions, n_percepts, dim=10000, mode="bipolar", device=None, seed=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim, self.mode = dim, mode
        self.action_hvs = gen_hvs(n_actions, dim, mode, self.device, seed)
        self.percept_hvs = gen_hvs(n_percepts, dim, mode, self.device, seed)
        self.register_buffer("memory_hv", torch.zeros(dim, device=self.device))

    def train_binding(self, percept_idx, action_idx):
        self.memory_hv = self.memory_hv + bind(self.percept_hvs[percept_idx], self.action_hvs[action_idx], self.mode)

    def predict_action(self, percept_idx):
        probe = bind(self.memory_hv, self.percept_hvs[percept_idx], self.mode)
        return int(batch_sim(probe, self.action_hvs, self.mode).argmax().item())

    def normalize(self):
        if self.mode == "bipolar": self.memory_hv = thresh(self.memory_hv)
        elif self.mode == "binary": self.memory_hv = (self.memory_hv >= 0).float()
        else: self.memory_hv = self.memory_hv / self.memory_hv.norm().clamp(min=1e-12)


class HAPSpikeBridge(nn.Module):
    """Bridges SNN spike output to HAP perception-action binding.

    Encodes the full spike vector into a percept hypervector by
    binding each active spike's position key with its firing rate,
    then uses that percept to index into HAP's percept memory.

    This replaces the previous hardcoded percept_idx=0 with
    actual spike-content-dependent percept indexing.
    """

    def __init__(self, hidden_size, n_actions, dim=10000, mode="bipolar", device=None, seed=None):
        super().__init__()
        self.hap = HAPModule(n_actions, hidden_size, dim, mode, device, seed)
        self.keys = gen_hvs(hidden_size, dim, mode, device, seed)

    def encode_spikes(self, spikes):
        """Encode spike vector into a percept hypervector.

        Each active neuron's position key is bundled into the percept HV,
        weighted by spike magnitude (firing rate / membrane potential).
        This preserves the full spike pattern, not just binary activity.

        Args:
            spikes: (hidden_size,) spike tensor (binary or continuous)

        Returns:
            (dim,) percept hypervector
        """
        hv = torch.zeros(self.hap.dim, device=self.hap.device)
        # Weighted bundling: each active neuron contributes its key,
        # scaled by its firing magnitude
        for i, s in enumerate(spikes):
            if abs(s.item()) > 1e-6:
                hv = hv + s * self.keys[i]
        if self.hap.mode == "bipolar":
            hv = thresh(hv)
        elif self.hap.mode == "binary":
            hv = (hv > 0).float()
        return hv

    def _spikes_to_percept_idx(self, spikes):
        """Map spike pattern to the closest percept hypervector index.

        Uses cosine similarity between the encoded spike HV and all
        percept HVs to find the best-matching percept index.

        Args:
            spikes: (hidden_size,) spike tensor

        Returns:
            percept index (int)
        """
        hv = self.encode_spikes(spikes)
        similarities = batch_sim(hv, self.hap.percept_hvs, self.hap.mode)
        return int(similarities.argmax().item())

    def train(self, spikes, action_idx):
        """Train a (spike_pattern, action) association.

        Encodes the spike pattern into a percept HV, finds the closest
        percept index, and binds it to the action HV in HAP memory.

        Args:
            spikes: (hidden_size,) spike tensor from SNN
            action_idx: Action index to associate with this spike pattern
        """
        percept_idx = self._spikes_to_percept_idx(spikes)
        self.hap.train_binding(percept_idx, action_idx)

    def predict(self, spikes):
        """Predict action from spike pattern.

        Args:
            spikes: (hidden_size,) spike tensor from SNN

        Returns:
            Predicted action index
        """
        percept_idx = self._spikes_to_percept_idx(spikes)
        return self.hap.predict_action(percept_idx)
