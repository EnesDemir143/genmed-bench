"""
System monitoring modülü.

GPU/MPS/RAM kullanımını izler.

Kullanım:
    from src.utils.system_monitor import SystemMonitor
    
    monitor = SystemMonitor()
    stats = monitor.get_stats()
"""

import os
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

import torch


@dataclass
class SystemStats:
    """Sistem istatistikleri."""
    timestamp: str
    epoch: int
    
    # CPU
    cpu_percent: Optional[float] = None
    
    # Memory
    ram_used_gb: Optional[float] = None
    ram_total_gb: Optional[float] = None
    ram_percent: Optional[float] = None
    
    # GPU (CUDA)
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_utilization: Optional[float] = None
    
    # MPS (Apple Silicon)
    mps_memory_allocated_gb: Optional[float] = None
    
    # Training stats
    batch_time: Optional[float] = None
    data_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Dict'e çevir."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class SystemMonitor:
    """
    Sistem kaynaklarını izler.
    
    CPU, RAM, GPU/MPS kullanımını takip eder.
    """
    
    def __init__(self):
        self.device = self._detect_device()
        self.history: List[SystemStats] = []
        self._psutil_available = self._check_psutil()
        self._pynvml_available = self._check_pynvml()
    
    def _detect_device(self) -> str:
        """Kullanılan device'ı tespit et."""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def _check_psutil(self) -> bool:
        """psutil mevcut mu?"""
        try:
            import psutil
            return True
        except ImportError:
            return False
    
    def _check_pynvml(self) -> bool:
        """pynvml mevcut mu? (NVIDIA GPU monitoring)"""
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except (ImportError, Exception):
            return False
    
    def get_stats(self, epoch: int = 0) -> SystemStats:
        """
        Güncel sistem istatistiklerini al.
        
        Args:
            epoch: Epoch numarası
            
        Returns:
            SystemStats
        """
        stats = SystemStats(
            timestamp=datetime.now().isoformat(),
            epoch=epoch
        )
        
        # CPU & RAM
        if self._psutil_available:
            import psutil
            
            stats.cpu_percent = psutil.cpu_percent(interval=0.1)
            
            mem = psutil.virtual_memory()
            stats.ram_used_gb = mem.used / (1024**3)
            stats.ram_total_gb = mem.total / (1024**3)
            stats.ram_percent = mem.percent
        
        # GPU (CUDA)
        if self.device == 'cuda':
            stats.gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
            stats.gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            stats.gpu_memory_percent = (stats.gpu_memory_used_gb / stats.gpu_memory_total_gb) * 100
            
            # GPU utilization (pynvml)
            if self._pynvml_available:
                try:
                    import pynvml
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    stats.gpu_utilization = util.gpu
                except Exception:
                    pass
        
        # MPS (Apple Silicon)
        elif self.device == 'mps':
            # MPS memory tracking
            if hasattr(torch.mps, 'current_allocated_memory'):
                stats.mps_memory_allocated_gb = torch.mps.current_allocated_memory() / (1024**3)
            elif hasattr(torch.mps, 'driver_allocated_memory'):
                stats.mps_memory_allocated_gb = torch.mps.driver_allocated_memory() / (1024**3)
        
        # History'e ekle
        self.history.append(stats)
        
        return stats
    
    def get_history_df(self):
        """History'yi pandas DataFrame olarak döner."""
        import pandas as pd
        return pd.DataFrame([s.to_dict() for s in self.history])
    
    def save_history(self, path: str) -> None:
        """History'yi CSV'ye kaydet."""
        df = self.get_history_df()
        df.to_csv(path, index=False)
    
    def get_summary(self) -> Dict[str, float]:
        """Ortalama istatistikleri döner."""
        if not self.history:
            return {}
        
        summary = {}
        
        # Averages
        attrs = ['cpu_percent', 'ram_percent', 'gpu_memory_percent', 'gpu_utilization']
        for attr in attrs:
            values = [getattr(s, attr) for s in self.history if getattr(s, attr) is not None]
            if values:
                summary[f'{attr}_avg'] = sum(values) / len(values)
                summary[f'{attr}_max'] = max(values)
        
        return summary
    
    def reset(self) -> None:
        """History'yi temizle."""
        self.history = []


class Timer:
    """Training zamanlaması için timer."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0
    
    def start(self) -> None:
        """Timer'ı başlat."""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Timer'ı durdur ve elapsed time döner."""
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


class AverageMeter:
    """
    Ortalama hesaplama (loss, accuracy vb. için).
    
    Kullanım:
        meter = AverageMeter()
        for batch in loader:
            loss = ...
            meter.update(loss.item(), batch_size)
        print(meter.avg)
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
