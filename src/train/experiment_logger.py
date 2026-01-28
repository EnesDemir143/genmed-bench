"""
Experiment logging modülü.

File-based logging, metrics CSV, grafikler.

Kullanım:
    from src.train.experiment_logger import ExperimentLogger
    
    logger = ExperimentLogger(
        model_name='resnet50',
        mode='linear_probe',
        augmentation='xdomainmix',
        dataset='nih'
    )
    
    logger.log_train(epoch, {'loss': 0.5, 'lr': 0.01})
    logger.log_val(epoch, metrics_dict, probs, labels)
    logger.save_predictions(probs, labels, is_best=True)
    logger.plot_curves()
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

import numpy as np
import yaml

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ExperimentLogger:
    """
    Experiment logging ve visualization.
    
    Dizin yapısı:
        runs/{model}_{mode}_{aug}_{dataset}_{timestamp}/
        ├── config.yaml
        ├── train_metrics.csv
        ├── val_metrics.csv
        ├── val_predictions_best.npz
        ├── val_predictions_last.npz
        ├── lr_curve.png
        ├── loss_curve.png
        ├── metrics_curve.png
        ├── system_stats.csv
        ├── system_curve.png
        └── checkpoints/
            ├── best.pth
            └── last.pth
    """
    
    def __init__(
        self,
        model_name: str,
        mode: str,
        augmentation: str,
        dataset: str,
        base_dir: str = 'runs',
        config: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ):
        """
        Args:
            model_name: Model adı
            mode: linear_probe veya full_finetune
            augmentation: Augmentation tipi
            dataset: Dataset adı
            base_dir: Base directory
            config: Full config dict
            timestamp: Custom timestamp (test için)
        """
        self.model_name = model_name
        self.mode = mode
        self.augmentation = augmentation
        self.dataset = dataset
        
        # Timestamp
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.timestamp = timestamp
        
        # Run name
        self.run_name = f"{model_name}_{mode}_{augmentation}_{dataset}_{timestamp}"
        
        # Directories
        self.run_dir = Path(base_dir) / self.run_name
        self.checkpoint_dir = self.run_dir / 'checkpoints'
        
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # File paths
        self.config_path = self.run_dir / 'config.yaml'
        self.train_metrics_path = self.run_dir / 'train_metrics.csv'
        self.val_metrics_path = self.run_dir / 'val_metrics.csv'
        self.system_stats_path = self.run_dir / 'system_stats.csv'
        
        # Config kaydet
        if config is not None:
            self.save_config(config)
        
        # CSV headers initialized
        self._train_header_written = False
        self._val_header_written = False
        self._system_header_written = False
        
        # History (plotting için)
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []
        self.system_history: List[Dict] = []
        
        # Best tracking
        self.best_metric = 0.0
        self.best_epoch = 0
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Config'i YAML olarak kaydet."""
        # Numpy/torch types'ı Python'a çevir
        config_clean = self._convert_to_serializable(config)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_clean, f, default_flow_style=False, allow_unicode=True)
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Numpy/torch types'ı JSON serializable'a çevir."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, 'item'):  # torch.Tensor
            return obj.item()
        return obj
    
    # =========================================================================
    # LOGGING
    # =========================================================================
    
    def log_train(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Train metrics'i CSV'ye yaz.
        
        Args:
            epoch: Epoch numarası
            metrics: {'loss': 0.5, 'lr': 0.01, ...}
        """
        metrics['epoch'] = epoch
        self.train_history.append(metrics)
        self._write_csv(self.train_metrics_path, metrics, 'train')
    
    def log_val(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        track_metric: str = 'roc_auc',
    ) -> bool:
        """
        Validation metrics'i CSV'ye yaz.
        
        Args:
            epoch: Epoch numarası
            metrics: compute_all_metrics() çıktısı
            track_metric: Best model için takip edilecek metrik
            
        Returns:
            is_best: Bu epoch en iyi mi?
        """
        # Flatten nested metrics (per_class vb.)
        flat_metrics = self._flatten_metrics(metrics)
        flat_metrics['epoch'] = epoch
        
        self.val_history.append(flat_metrics)
        self._write_csv(self.val_metrics_path, flat_metrics, 'val')
        
        # Best tracking
        current = flat_metrics.get(track_metric, 0) or 0
        is_best = current > self.best_metric
        if is_best:
            self.best_metric = current
            self.best_epoch = epoch
        
        return is_best
    
    def log_system(self, stats: Dict[str, Any]) -> None:
        """System stats'ı CSV'ye yaz."""
        self.system_history.append(stats)
        self._write_csv(self.system_stats_path, stats, 'system')
    
    def _flatten_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Nested metrics'i flatten et."""
        flat = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            flat[f"{key}_{sub_key}_{sub_sub_key}"] = sub_sub_value
                    else:
                        flat[f"{key}_{sub_key}"] = sub_value
            elif isinstance(value, (list, np.ndarray)):
                # Confusion matrix gibi array'leri skip et
                continue
            else:
                flat[key] = value
        return flat
    
    def _write_csv(
        self,
        path: Path,
        data: Dict[str, Any],
        data_type: str
    ) -> None:
        """CSV'ye yaz (append mode)."""
        header_written = getattr(self, f'_{data_type}_header_written', False)
        
        # None değerleri temizle veya N/A yap
        clean_data = {k: (v if v is not None else 'N/A') for k, v in data.items()}
        
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=clean_data.keys())
            
            if not header_written:
                writer.writeheader()
                setattr(self, f'_{data_type}_header_written', True)
            
            writer.writerow(clean_data)
    
    # =========================================================================
    # PREDICTIONS
    # =========================================================================
    
    def save_predictions(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        is_best: bool = False,
        is_last: bool = False,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Predictions'ı kaydet (şişmemesi için sadece best/last).
        
        Args:
            probs: Prediction probabilities [N, C]
            labels: Ground truth labels [N] veya [N, C]
            is_best: Best epoch mi?
            is_last: Son epoch mu?
            epoch: Epoch numarası
        """
        if is_best:
            path = self.run_dir / 'val_predictions_best.npz'
            np.savez_compressed(
                path,
                probs=probs,
                labels=labels,
                epoch=epoch,
                metric=self.best_metric
            )
        
        if is_last:
            path = self.run_dir / 'val_predictions_last.npz'
            np.savez_compressed(
                path,
                probs=probs,
                labels=labels,
                epoch=epoch
            )
    
    def load_predictions(self, which: str = 'best') -> Dict[str, np.ndarray]:
        """Kayıtlı predictions'ı yükle."""
        path = self.run_dir / f'val_predictions_{which}.npz'
        if not path.exists():
            raise FileNotFoundError(f"Predictions not found: {path}")
        return dict(np.load(path))
    
    # =========================================================================
    # PLOTTING
    # =========================================================================
    
    def plot_curves(self) -> None:
        """Tüm grafikleri çiz."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        self._plot_lr_curve()
        self._plot_loss_curve()
        self._plot_metrics_curve()
        self._plot_system_curve()
    
    def _plot_lr_curve(self) -> None:
        """Learning rate curve."""
        if not self.train_history:
            return
        
        epochs = [h['epoch'] for h in self.train_history]
        lrs = [h.get('lr', 0) for h in self.train_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, lrs, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.run_dir / 'lr_curve.png', dpi=150)
        plt.close()
    
    def _plot_loss_curve(self) -> None:
        """Train/val loss curve."""
        if not self.train_history:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Train loss
        epochs = [h['epoch'] for h in self.train_history]
        train_loss = [h.get('loss', 0) for h in self.train_history]
        ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        
        # Val loss
        if self.val_history and 'loss' in self.val_history[0]:
            val_epochs = [h['epoch'] for h in self.val_history]
            val_loss = [h.get('loss', 0) for h in self.val_history]
            ax.plot(val_epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.run_dir / 'loss_curve.png', dpi=150)
        plt.close()
    
    def _plot_metrics_curve(self) -> None:
        """Validation metrics curve."""
        if not self.val_history:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        epochs = [h['epoch'] for h in self.val_history]
        
        # Accuracy
        acc = [h.get('accuracy', 0) for h in self.val_history]
        axes[0].plot(epochs, acc, 'g-', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Validation Accuracy')
        axes[0].grid(True, alpha=0.3)
        
        # AUC
        auc = [h.get('roc_auc', h.get('roc_auc_macro', 0)) or 0 for h in self.val_history]
        axes[1].plot(epochs, auc, 'b-', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('ROC-AUC')
        axes[1].set_title('Validation AUC')
        axes[1].grid(True, alpha=0.3)
        
        # F1
        f1 = [h.get('f1', h.get('f1_macro', 0)) or 0 for h in self.val_history]
        axes[2].plot(epochs, f1, 'r-', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('Validation F1')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'metrics_curve.png', dpi=150)
        plt.close()
    
    def _plot_system_curve(self) -> None:
        """System stats curve."""
        if not self.system_history:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = [h.get('epoch', i) for i, h in enumerate(self.system_history)]
        
        # RAM
        ram = [h.get('ram_percent', 0) or 0 for h in self.system_history]
        if any(ram):
            axes[0].plot(epochs, ram, 'b-', linewidth=2)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('RAM %')
            axes[0].set_title('RAM Usage')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, 100)
        
        # GPU/MPS Memory
        gpu = [h.get('gpu_memory_percent', h.get('mps_memory_allocated_gb', 0) * 10) or 0 
               for h in self.system_history]
        if any(gpu):
            axes[1].plot(epochs, gpu, 'g-', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('GPU Memory %')
            axes[1].set_title('GPU/MPS Memory Usage')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'system_curve.png', dpi=150)
        plt.close()
    
    # =========================================================================
    # CHECKPOINTS
    # =========================================================================
    
    def get_checkpoint_path(self, which: str = 'best') -> Path:
        """Checkpoint path'i döner."""
        return self.checkpoint_dir / f'{which}.pth'
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    def get_summary(self) -> Dict[str, Any]:
        """Run summary'sini döner."""
        return {
            'run_name': self.run_name,
            'run_dir': str(self.run_dir),
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric,
            'total_epochs': len(self.train_history),
        }
    
    def print_summary(self) -> None:
        """Summary'yi yazdır."""
        summary = self.get_summary()
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Run: {summary['run_name']}")
        print(f"Best Epoch: {summary['best_epoch']}")
        print(f"Best Metric: {summary['best_metric']:.4f}")
        print(f"Total Epochs: {summary['total_epochs']}")
        print(f"Run Dir: {summary['run_dir']}")
        print("="*50 + "\n")
