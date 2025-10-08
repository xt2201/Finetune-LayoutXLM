"""
Utility functions for configuration, logging, and helpers
"""

import os
import yaml
import logging
import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging to both file and console
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Logger instance
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    
    return logger


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Returns:
        Dictionary with total, trainable, and frozen parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    best_metric: float,
    save_path: str,
    config: Dict
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch
        step: Current global step
        best_metric: Best validation metric so far
        save_path: Path to save checkpoint
        config: Training configuration
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric,
        'config': config
    }
    
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Any = None
) -> Dict:
    """
    Load model checkpoint
    
    Returns:
        Dictionary with epoch, step, and best_metric
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'best_metric': checkpoint.get('best_metric', float('inf'))
    }


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
