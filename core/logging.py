"""
Logging utilities for Arthedain SNN training.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
import torch


def setup_logger(
    name: str = "arthedain",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Set up logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_model_info(model, logger: logging.Logger) -> None:
    """Log model information including parameter counts and device.
    
    Args:
        model: Neural network model
        logger: Logger instance
    """
    # Handle both nn.Module and custom model classes
    if hasattr(model, 'parameters'):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        device = next(model.parameters()).device
    else:
        # Custom model class - count parameters manually
        params = []
        for attr_name in dir(model):
            attr = getattr(model, attr_name)
            if isinstance(attr, torch.Tensor):
                params.append(attr)
        
        total_params = sum(p.numel() for p in params)
        trainable_params = total_params  # Assume all are trainable for custom models
        device = params[0].device if params else torch.device('cpu')
    
    logger.info(f"Model device: {device}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")


def log_training_step(
    step: int,
    loss: float,
    lr: Optional[float] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """Log training step information.
    
    Args:
        step: Training step number
        loss: Current loss value
        lr: Learning rate (optional)
        logger: Logger instance (creates default if None)
    """
    if logger is None:
        logger = logging.getLogger("arthedain")
    
    msg = f"Step {step}: Loss = {loss:.6f}"
    if lr is not None:
        msg += f", LR = {lr:.6f}"
    
    logger.info(msg)


def log_gpu_memory(logger: Optional[logging.Logger] = None) -> None:
    """Log GPU memory usage if CUDA is available.
    
    Args:
        logger: Logger instance (creates default if None)
    """
    if logger is None:
        logger = logging.getLogger("arthedain")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    else:
        logger.info("CUDA not available - using CPU")
