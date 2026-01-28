"""
Seed sabitleme modülü.

Reproducibility için tüm random number generator'ları sabitler.

Kullanım:
    from src.utils.seed import set_seed
    set_seed(42)
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Tüm random number generator'ları sabitler.
    
    Args:
        seed: Seed değeri
        deterministic: CUDNN deterministic mode (yavaşlatabilir)
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS için özel seed yok, manual_seed yeterli
        pass
    
    # CUDNN deterministic (reproducibility vs speed trade-off)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    
    # Environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_generator(seed: int) -> torch.Generator:
    """
    DataLoader için generator oluşturur.
    
    Args:
        seed: Seed değeri
        
    Returns:
        torch.Generator
        
    Kullanım:
        g = get_generator(42)
        DataLoader(..., generator=g)
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker'ları için init function.
    
    Kullanım:
        DataLoader(..., worker_init_fn=worker_init_fn)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
