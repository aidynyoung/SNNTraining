"""
Model utilities for saving, loading, and checkpointing.
"""
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from models.rsnn import RSNN
from models.readout import Readout
from models.hebbian import DualHebbian


class ModelCheckpoint:
    """Utility class for saving and loading model checkpoints."""
    
    def __init__(self, save_dir: str = "checkpoints"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("arthedain")
    
    def save_checkpoint(
        self,
        rsnn: RSNN,
        readout: Readout,
        hebbian: DualHebbian,
        config: Dict[str, Any],
        step: int,
        loss: float,
        optimizer_state: Optional[Dict] = None,
        filename: Optional[str] = None
    ) -> str:
        """Save complete model checkpoint.
        
        Args:
            rsnn: Recurrent SNN model
            readout: Readout layer
            hebbian: Hebbian learning module
            config: Training configuration
            step: Current training step
            loss: Current loss value
            optimizer_state: Optional optimizer state dict
            filename: Optional custom filename
            
        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_step_{step}.pt"
        
        checkpoint_path = self.save_dir / filename
        
        checkpoint = {
            'step': step,
            'loss': loss,
            'config': config,
            'model_state': {
                'rsnn': {
                    'W_in': rsnn.W_in,
                    'W_rec': rsnn.W_rec,
                    'prev_spikes': rsnn.prev_spikes
                },
                'readout': {
                    'W': readout.W
                },
                'hebbian': {
                    'e_fast': hebbian.e_fast,
                    'e_slow': hebbian.e_slow,
                    'tau_fast': hebbian.tau_fast,
                    'tau_slow': hebbian.tau_slow,
                    'alpha': hebbian.alpha,
                    'beta': hebbian.beta
                }
            }
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state'] = optimizer_state
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Target device for tensors
            
        Returns:
            Dictionary containing loaded model states and metadata
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
        return checkpoint
    
    def save_weights_only(
        self,
        rsnn: RSNN,
        readout: Readout,
        hebbian: DualHebbian,
        filename: str = "model_weights.pt"
    ) -> str:
        """Save only model weights (minimal file size).
        
        Args:
            rsnn: Recurrent SNN model
            readout: Readout layer
            hebbian: Hebbian learning module
            filename: Output filename
            
        Returns:
            Path to saved weights file
        """
        weights_path = self.save_dir / filename
        
        weights = {
            'rsnn_W_in': rsnn.W_in,
            'rsnn_W_rec': rsnn.W_rec,
            'readout_W': readout.W,
            'hebbian_e_fast': hebbian.e_fast,
            'hebbian_e_slow': hebbian.e_slow
        }
        
        torch.save(weights, weights_path)
        self.logger.info(f"Weights saved to {weights_path}")
        
        return str(weights_path)
    
    def load_weights_only(
        self,
        weights_path: str,
        rsnn: RSNN,
        readout: Readout,
        hebbian: DualHebbian,
        device: Optional[torch.device] = None
    ) -> None:
        """Load only model weights.
        
        Args:
            weights_path: Path to weights file
            rsnn: RSNN model to load weights into
            readout: Readout layer to load weights into
            hebbian: Hebbian module to load weights into
            device: Target device for tensors
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        weights = torch.load(weights_path, map_location=device)
        
        # Load weights into models
        rsnn.W_in = weights['rsnn_W_in'].to(device)
        rsnn.W_rec = weights['rsnn_W_rec'].to(device)
        readout.W = weights['readout_W'].to(device)
        hebbian.e_fast = weights['hebbian_e_fast'].to(device)
        hebbian.e_slow = weights['hebbian_e_slow'].to(device)
        
        self.logger.info(f"Weights loaded from {weights_path}")
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints.
        
        Returns:
            List of checkpoint file paths
        """
        checkpoints = list(self.save_dir.glob("checkpoint_step_*.pt"))
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_last: int = 5) -> None:
        """Remove old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_last: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[keep_last:]:
                checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {checkpoint}")


def save_training_metrics(
    metrics: Dict[str, list],
    filepath: str,
    save_dir: str = "results"
) -> None:
    """Save training metrics to JSON file.
    
    Args:
        metrics: Dictionary of metric lists
        filepath: Output filename
        save_dir: Directory to save to
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    output_path = save_path / filepath
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logging.getLogger("arthedain").info(f"Metrics saved to {output_path}")


def load_training_metrics(filepath: str, save_dir: str = "results") -> Dict[str, list]:
    """Load training metrics from JSON file.
    
    Args:
        filepath: Input filename
        save_dir: Directory to load from
        
    Returns:
        Dictionary of metric lists
    """
    load_path = Path(save_dir) / filepath
    
    if not load_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {load_path}")
    
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    
    logging.getLogger("arthedain").info(f"Metrics loaded from {load_path}")
    return metrics
