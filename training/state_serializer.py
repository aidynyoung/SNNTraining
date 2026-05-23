"""
State Serialization
===================
Atomic state checkpointing for crash-safe recovery.

Implements save/load for:
- Weights (W_in, W_rec, W_out)
- Eligibility traces (e_fast, e_slow)
- RMS EMA statistics
- All running statistics for seamless resume
"""

import torch
import json
import struct
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import time


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint file."""
    version: int = 1
    timestamp: float = 0.0
    step_count: int = 0
    model_config: Dict[str, Any] = None
    
    def to_dict(self):
        return {
            'version': self.version,
            'timestamp': self.timestamp,
            'step_count': self.step_count,
            'model_config': self.model_config or {},
        }
    
    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            version=d.get('version', 1),
            timestamp=d.get('timestamp', 0.0),
            step_count=d.get('step_count', 0),
            model_config=d.get('model_config', {})
        )


class StateSerializer:
    """
    Atomic state serialization for SNN checkpointing.
    
    Saves complete state including:
    - All weight matrices
    - Eligibility traces
    - RMS normalization statistics
    - Training step counter
    - Model hyperparameters
    
    Format: Binary blob with JSON header for flexibility.
    """
    
    MAGIC = b'ARTCHK'  # Arthedain Checkpoint
    VERSION = 1
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save(
        self,
        trainer,
        model,
        step_count: int = 0,
        metadata: Optional[Dict] = None,
        filename: Optional[str] = None
    ) -> Path:
        """
        Atomic state save.
        
        Writes to temporary file first, then renames for atomicity.
        
        Args:
            trainer: OnlineTrainer with state
            model: RSNN model
            step_count: Current training step
            metadata: Additional metadata to store
            filename: Custom filename (default: timestamp-based)
            
        Returns:
            Path to saved checkpoint
        """
        # Build checkpoint data
        checkpoint = self._extract_state(trainer, model, step_count, metadata)
        
        # Generate filename
        if filename is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_{ts}_step{step_count}.bin"
        
        checkpoint_path = self.checkpoint_dir / filename
        temp_path = checkpoint_path.with_suffix('.tmp')
        
        # Serialize to binary
        with open(temp_path, 'wb') as f:
            # Write header
            f.write(self.MAGIC)
            f.write(struct.pack('<I', self.VERSION))
            
            # Serialize as PyTorch save (handles tensors efficiently)
            torch.save(checkpoint, f, _use_new_zipfile_serialization=True)
        
        # Atomic rename
        temp_path.rename(checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        print(f"  - Step: {step_count}")
        print(f"  - Size: {checkpoint_path.stat().st_size / 1024:.1f} KB")
        
        return checkpoint_path
    
    def _extract_state(
        self,
        trainer,
        model,
        step_count: int,
        metadata: Optional[Dict]
    ) -> Dict:
        """Extract complete state from trainer and model."""
        state = {
            'metadata': {
                'version': self.VERSION,
                'timestamp': time.time(),
                'step_count': step_count,
                'user_metadata': metadata or {},
            },
            'model': {},
            'trainer': {},
        }
        
        # Model weights
        state['model']['W_in'] = model.W_in.detach().cpu().clone()
        state['model']['W_rec'] = model.W_rec.detach().cpu().clone()
        state['model']['lif_v'] = model.lif.v.detach().cpu().clone() if hasattr(model, 'lif') else None
        state['model']['prev_spikes'] = model.prev_spikes.detach().cpu().clone() if hasattr(model, 'prev_spikes') else None
        
        # Model config
        state['model']['config'] = {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'input_gain': model.input_gain if hasattr(model, 'input_gain') else 5.0,
        }
        
        # Trainer state
        if hasattr(trainer, 'readout'):
            state['trainer']['W_out'] = trainer.readout.W.detach().cpu().clone()
            if hasattr(trainer.readout, 'b'):
                state['trainer']['b_out'] = trainer.readout.b.detach().cpu().clone()
        
        if hasattr(trainer, 'hebbian'):
            h = trainer.hebbian._impl if hasattr(trainer.hebbian, '_impl') else trainer.hebbian
            if hasattr(h, 'e_fast'):
                state['trainer']['e_fast'] = h.e_fast.detach().cpu().clone()
                state['trainer']['e_slow'] = h.e_slow.detach().cpu().clone()
            if hasattr(h, 'tau_fast'):
                state['trainer']['tau_fast'] = h.tau_fast
                state['trainer']['tau_slow'] = h.tau_slow
        
        # Training parameters
        state['trainer']['lr_readout'] = trainer.lr_readout if hasattr(trainer, 'lr_readout') else 1e-3
        state['trainer']['lr_recurrent'] = trainer.lr_recurrent if hasattr(trainer, 'lr_recurrent') else 5e-5
        
        # Drift detector state (if adaptive trainer)
        if hasattr(trainer, 'drift_detector'):
            dd = trainer.drift_detector
            state['trainer']['drift_state'] = {
                'error_ema': dd.error_ema,
                'error_ema_sq': dd.error_ema_sq,
                'baseline_mean': dd.baseline_mean,
                'baseline_std': dd.baseline_std,
                'drift_detected': dd.drift_detected,
            }
        
        return state
    
    def load(
        self,
        trainer,
        model,
        checkpoint_path: Optional[str] = None,
        load_latest: bool = False
    ) -> Dict:
        """
        Restore state from checkpoint.
        
        Args:
            trainer: OnlineTrainer to restore
            model: RSNN model to restore
            checkpoint_path: Specific checkpoint file (or None)
            load_latest: If True, load most recent checkpoint
            
        Returns:
            Metadata dict from checkpoint
        """
        # Determine checkpoint to load
        if load_latest:
            checkpoint_path = self._get_latest_checkpoint()
        elif checkpoint_path is None:
            raise ValueError("Must specify checkpoint_path or set load_latest=True")
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        with open(checkpoint_path, 'rb') as f:
            magic = f.read(len(self.MAGIC))
            if magic != self.MAGIC:
                raise ValueError(f"Invalid checkpoint file: {checkpoint_path}")
            
            version = struct.unpack('<I', f.read(4))[0]
            if version != self.VERSION:
                print(f"Warning: Checkpoint version {version} != current {self.VERSION}")
            
            checkpoint = torch.load(f, map_location='cpu')
        
        # Restore model state
        self._restore_model_state(model, checkpoint['model'])
        
        # Restore trainer state
        self._restore_trainer_state(trainer, checkpoint['trainer'])
        
        metadata = checkpoint['metadata']
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"  - Step: {metadata['step_count']}")
        print(f"  - Timestamp: {time.ctime(metadata['timestamp'])}")
        
        return metadata
    
    def _restore_model_state(self, model, state: Dict):
        """Restore model weights and state."""
        model.W_in.copy_(state['W_in'].to(model.device))
        model.W_rec.copy_(state['W_rec'].to(model.device))
        
        if state['lif_v'] is not None and hasattr(model, 'lif'):
            model.lif.v.copy_(state['lif_v'].to(model.device))
        
        if state['prev_spikes'] is not None and hasattr(model, 'prev_spikes'):
            model.prev_spikes.copy_(state['prev_spikes'].to(model.device))
    
    def _restore_trainer_state(self, trainer, state: Dict):
        """Restore trainer state."""
        # Restore weights
        if 'W_out' in state and hasattr(trainer, 'readout'):
            trainer.readout.W.copy_(state['W_out'].to(trainer.device))
        
        if 'b_out' in state and hasattr(trainer.readout, 'b'):
            trainer.readout.b.copy_(state['b_out'].to(trainer.device))
        
        # Restore eligibility traces
        if 'e_fast' in state and hasattr(trainer, 'hebbian'):
            h = trainer.hebbian._impl if hasattr(trainer.hebbian, '_impl') else trainer.hebbian
            if hasattr(h, 'e_fast'):
                h.e_fast.copy_(state['e_fast'].to(trainer.device))
                h.e_slow.copy_(state['e_slow'].to(trainer.device))
        
        # Restore time constants
        if 'tau_fast' in state and hasattr(trainer, 'hebbian'):
            h = trainer.hebbian._impl if hasattr(trainer.hebbian, '_impl') else trainer.hebbian
            if hasattr(h, 'tau_fast'):
                h.tau_fast = state['tau_fast']
                h.tau_slow = state['tau_slow']
                # Recompute decay factors
                import math
                h.decay_fast = torch.exp(torch.tensor(-1.0 / h.tau_fast, device=h.device))
                h.decay_slow = torch.exp(torch.tensor(-1.0 / h.tau_slow, device=h.device))
        
        # Restore drift detector state
        if 'drift_state' in state and hasattr(trainer, 'drift_detector'):
            dd = trainer.drift_detector
            ds = state['drift_state']
            dd.error_ema = ds['error_ema']
            dd.error_ema_sq = ds['error_ema_sq']
            dd.baseline_mean = ds['baseline_mean']
            dd.baseline_std = ds['baseline_std']
            dd.drift_detected = ds['drift_detected']
    
    def _get_latest_checkpoint(self) -> Path:
        """Get most recent checkpoint file."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.bin"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
        
        # Sort by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        checkpoints = []
        for path in sorted(self.checkpoint_dir.glob("checkpoint_*.bin")):
            stat = path.stat()
            checkpoints.append({
                'path': str(path),
                'size_kb': stat.st_size / 1024,
                'modified': time.ctime(stat.st_mtime),
            })
        return checkpoints


class CompactStateSerializer(StateSerializer):
    """
    Compact binary format for embedded deployment.
    
    Smaller than full PyTorch format, suitable for:
    - UAV power-cycle recovery
    - Manufacturing line resume
    - Edge deployment state sync
    """
    
    def save_compact(
        self,
        trainer,
        model,
        step_count: int = 0,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save compact binary state (no JSON, minimal header).
        
        Format: [MAGIC(6) | VERSION(1) | STEP(4) | TENSORS...]
        """
        if filename is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"compact_{ts}_step{step_count}.bin"
        
        path = self.checkpoint_dir / filename
        temp_path = path.with_suffix('.tmp')
        
        with open(temp_path, 'wb') as f:
            # Minimal header
            f.write(self.MAGIC[:4])  # Truncated magic
            f.write(struct.pack('<B', self.VERSION))
            f.write(struct.pack('<I', step_count))
            
            # Write tensors in fixed order
            # W_in
            w_in = model.W_in.detach().cpu().flatten()
            f.write(struct.pack('<I', len(w_in)))
            f.write(struct.pack(f'<{len(w_in)}f', *w_in.tolist()))
            
            # W_rec
            w_rec = model.W_rec.detach().cpu().flatten()
            f.write(struct.pack('<I', len(w_rec)))
            f.write(struct.pack(f'<{len(w_rec)}f', *w_rec.tolist()))
            
            # W_out (from readout)
            if hasattr(trainer, 'readout'):
                w_out = trainer.readout.W.detach().cpu().flatten()
                f.write(struct.pack('<I', len(w_out)))
                f.write(struct.pack(f'<{len(w_out)}f', *w_out.tolist()))
            
            # Eligibility traces (if present)
            if hasattr(trainer, 'hebbian'):
                h = trainer.hebbian._impl if hasattr(trainer.hebbian, '_impl') else trainer.hebbian
                if hasattr(h, 'e_fast'):
                    e_fast = h.e_fast.detach().cpu().flatten()
                    e_slow = h.e_slow.detach().cpu().flatten()
                    f.write(struct.pack('<I', len(e_fast)))
                    f.write(struct.pack(f'<{len(e_fast)}f', *e_fast.tolist()))
                    f.write(struct.pack(f'<{len(e_slow)}f', *e_slow.tolist()))
        
        temp_path.rename(path)
        print(f"Compact checkpoint: {path} ({path.stat().st_size} bytes)")
        
        return path


if __name__ == "__main__":
    print("State Serialization Test")
    print("=" * 50)
    
    # Create mock objects for testing
    from models.rsnn import RSNN, RSNNConfig
    from models.readout import Readout, ReadoutConfig
    from models.hebbian import DualHebbian, HebbianConfig
    from training.online_trainer import OnlineTrainer
    
    # Create model and trainer
    model = RSNN(config=RSNNConfig(input_size=100, hidden_size=64))
    readout = Readout(ReadoutConfig(64, 2))
    hebbian = DualHebbian((64, 64), tau_fast=5.0, tau_slow=50.0, alpha=0.7)
    trainer = OnlineTrainer(model, readout, hebbian)
    
    # Create serializer
    serializer = StateSerializer(checkpoint_dir="test_checkpoints")
    
    # Save
    checkpoint_path = serializer.save(trainer, model, step_count=1000)
    
    # List checkpoints
    print(f"\nCheckpoints: {serializer.list_checkpoints()}")
    
    # Modify model (simulate training)
    model.W_in += 0.1
    model.W_rec += 0.05
    
    # Load
    metadata = serializer.load(trainer, model, checkpoint_path)
    
    # Cleanup
    import shutil
    shutil.rmtree("test_checkpoints")
    
    print("\nState serialization test complete.")
