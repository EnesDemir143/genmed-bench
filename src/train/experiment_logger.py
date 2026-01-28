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
        self.figs_dir = self.run_dir / 'figs'
        
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.figs_dir.mkdir(exist_ok=True)
        
        # Per-epoch figure subdirectories
        (self.figs_dir / 'confusion_matrix').mkdir(exist_ok=True)
        (self.figs_dir / 'roc').mkdir(exist_ok=True)
        (self.figs_dir / 'pr_curve').mkdir(exist_ok=True)
        (self.figs_dir / 'class_dist').mkdir(exist_ok=True)
        
        # File paths
        self.config_path = self.run_dir / 'config.yaml'
        self.train_metrics_path = self.run_dir / 'train_metrics.csv'
        self.val_metrics_path = self.run_dir / 'val_metrics.csv'
        self.system_stats_path = self.run_dir / 'system_stats.csv'
        
        # Config kaydet
        if config is not None:
            self.save_config(config)
        
        # History (plotting için)
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []
        self.system_history: List[Dict] = []
        
        # Best tracking
        self.best_metric = 0.0
        self.best_epoch = 0
        
        # Resume: Mevcut CSV dosyalarını kontrol et ve yükle
        self._train_header_written = self.train_metrics_path.exists()
        self._val_header_written = self.val_metrics_path.exists()
        self._system_header_written = self.system_stats_path.exists()
        
        # Mevcut history'yi yükle (resume için)
        if self._train_header_written:
            self._load_existing_history()
    
    def _load_existing_history(self) -> None:
        """Resume durumunda mevcut CSV'lerden history'yi yükle."""
        import pandas as pd
        
        # Train history
        if self.train_metrics_path.exists():
            try:
                df = pd.read_csv(self.train_metrics_path)
                self.train_history = df.to_dict('records')
            except Exception:
                self.train_history = []
        
        # Val history
        if self.val_metrics_path.exists():
            try:
                df = pd.read_csv(self.val_metrics_path)
                self.val_history = df.to_dict('records')
                
                # Best metric'i güncelle
                if 'roc_auc' in df.columns:
                    valid_aucs = df['roc_auc'].dropna()
                    if len(valid_aucs) > 0:
                        self.best_metric = valid_aucs.max()
                        self.best_epoch = int(df.loc[valid_aucs.idxmax(), 'epoch'])
            except Exception:
                self.val_history = []
        
        # System history
        if self.system_stats_path.exists():
            try:
                df = pd.read_csv(self.system_stats_path)
                self.system_history = df.to_dict('records')
            except Exception:
                self.system_history = []
    
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
        """Tüm grafikleri çiz (training sonunda çağrılır)."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        self._plot_lr_curve()
        self._plot_loss_curve()
        self._plot_metrics_curve()
        self._plot_system_curve()
    
    def plot_final_figures(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        class_names: Optional[List[str]] = None,
        multi_label: bool = False,
    ) -> None:
        """
        Training sonunda prediction-based figürleri çiz (son epoch).
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            class_names: Class isimleri
            multi_label: Multi-label classification mi?
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Final figürler (figs/ altına)
        self._plot_confusion_matrix(y_true, y_pred, class_names, epoch=None, multi_label=multi_label)
        self._plot_roc_curve(y_true, y_prob, class_names, epoch=None, multi_label=multi_label)
        self._plot_precision_recall_curve(y_true, y_prob, class_names, epoch=None, multi_label=multi_label)
        self._plot_class_distribution(y_true, y_pred, class_names, epoch=None, multi_label=multi_label)
    
    def plot_epoch_figures(
        self,
        epoch: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        class_names: Optional[List[str]] = None,
        multi_label: bool = False,
    ) -> None:
        """
        Her epoch sonunda per-epoch figürleri çiz.
        
        Args:
            epoch: Current epoch number
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            class_names: Class isimleri
            multi_label: Multi-label classification mi?
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Per-epoch figürler (subfolder'lara)
        self._plot_confusion_matrix(y_true, y_pred, class_names, epoch=epoch, multi_label=multi_label)
        self._plot_roc_curve(y_true, y_prob, class_names, epoch=epoch, multi_label=multi_label)
        self._plot_precision_recall_curve(y_true, y_prob, class_names, epoch=epoch, multi_label=multi_label)
        self._plot_class_distribution(y_true, y_pred, class_names, epoch=epoch, multi_label=multi_label)
        
        # Aggregate curve'ları her epoch güncelle
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
        plt.savefig(self.figs_dir / 'lr_curve.png', dpi=150)
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
        plt.savefig(self.figs_dir / 'loss_curve.png', dpi=150)
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
        plt.savefig(self.figs_dir / 'metrics_curve.png', dpi=150)
        plt.close()
    
    def _plot_system_curve(self) -> None:
        """Comprehensive system stats curves (CPU, RAM, GPU/MPS)."""
        if not self.system_history:
            return
        
        epochs = [h.get('epoch', i) for i, h in enumerate(self.system_history)]
        
        # Collect all available metrics
        cpu_percent = [h.get('cpu_percent', 0) or 0 for h in self.system_history]
        ram_percent = [h.get('ram_percent', 0) or 0 for h in self.system_history]
        ram_used_gb = [h.get('ram_used_gb', 0) or 0 for h in self.system_history]
        
        # GPU (CUDA) or MPS memory
        gpu_memory_percent = [h.get('gpu_memory_percent', 0) or 0 for h in self.system_history]
        gpu_memory_used_gb = [h.get('gpu_memory_used_gb', 0) or 0 for h in self.system_history]
        gpu_utilization = [h.get('gpu_utilization', 0) or 0 for h in self.system_history]
        mps_memory_gb = [h.get('mps_memory_allocated_gb', 0) or 0 for h in self.system_history]
        
        # Determine if CUDA or MPS
        has_cuda = any(gpu_memory_percent) or any(gpu_utilization)
        has_mps = any(mps_memory_gb)
        
        # Create 2x2 figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ===== Plot 1: CPU Usage =====
        ax = axes[0, 0]
        if any(cpu_percent):
            ax.plot(epochs, cpu_percent, 'b-', linewidth=2, marker='o', markersize=3)
            ax.fill_between(epochs, cpu_percent, alpha=0.3)
            ax.set_ylim(0, 100)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('CPU Usage (%)')
        ax.set_title('CPU Utilization', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add stats annotation
        if any(cpu_percent):
            avg_cpu = np.mean(cpu_percent)
            max_cpu = np.max(cpu_percent)
            ax.axhline(y=avg_cpu, color='red', linestyle='--', alpha=0.5, label=f'Avg: {avg_cpu:.1f}%')
            ax.legend(loc='upper right', fontsize=8)
        
        # ===== Plot 2: RAM Usage =====
        ax = axes[0, 1]
        if any(ram_percent):
            ax.plot(epochs, ram_percent, 'g-', linewidth=2, marker='o', markersize=3)
            ax.fill_between(epochs, ram_percent, alpha=0.3, color='green')
            ax.set_ylim(0, 100)
            
            # Add GB values as secondary y-axis
            if any(ram_used_gb):
                ax2 = ax.twinx()
                ax2.plot(epochs, ram_used_gb, 'g--', linewidth=1, alpha=0.7)
                ax2.set_ylabel('RAM (GB)', color='green', alpha=0.7)
                ax2.tick_params(axis='y', labelcolor='green')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RAM Usage (%)')
        ax.set_title('RAM Memory', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if any(ram_percent):
            avg_ram = np.mean(ram_percent)
            ax.axhline(y=avg_ram, color='red', linestyle='--', alpha=0.5, label=f'Avg: {avg_ram:.1f}%')
            ax.legend(loc='upper right', fontsize=8)
        
        # ===== Plot 3: GPU/MPS Memory =====
        ax = axes[1, 0]
        if has_cuda:
            ax.plot(epochs, gpu_memory_percent, 'm-', linewidth=2, marker='o', markersize=3)
            ax.fill_between(epochs, gpu_memory_percent, alpha=0.3, color='magenta')
            ax.set_ylim(0, 100)
            ax.set_ylabel('GPU Memory (%)')
            ax.set_title('GPU Memory Usage (CUDA)', fontweight='bold')
            
            if any(gpu_memory_used_gb):
                ax2 = ax.twinx()
                ax2.plot(epochs, gpu_memory_used_gb, 'm--', linewidth=1, alpha=0.7)
                ax2.set_ylabel('GPU Memory (GB)', color='magenta', alpha=0.7)
                ax2.tick_params(axis='y', labelcolor='magenta')
            
            if any(gpu_memory_percent):
                avg_gpu = np.mean(gpu_memory_percent)
                ax.axhline(y=avg_gpu, color='red', linestyle='--', alpha=0.5, label=f'Avg: {avg_gpu:.1f}%')
                ax.legend(loc='upper right', fontsize=8)
        
        elif has_mps:
            ax.plot(epochs, mps_memory_gb, 'c-', linewidth=2, marker='o', markersize=3)
            ax.fill_between(epochs, mps_memory_gb, alpha=0.3, color='cyan')
            ax.set_ylabel('MPS Memory (GB)')
            ax.set_title('MPS Memory Usage (Apple Silicon)', fontweight='bold')
            
            if any(mps_memory_gb):
                avg_mps = np.mean(mps_memory_gb)
                ax.axhline(y=avg_mps, color='red', linestyle='--', alpha=0.5, label=f'Avg: {avg_mps:.2f} GB')
                ax.legend(loc='upper right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No GPU/MPS Data', ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_title('GPU/MPS Memory', fontweight='bold')
        
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
        
        # ===== Plot 4: GPU Utilization (CUDA only) or Summary =====
        ax = axes[1, 1]
        if has_cuda and any(gpu_utilization):
            ax.plot(epochs, gpu_utilization, 'orange', linewidth=2, marker='o', markersize=3)
            ax.fill_between(epochs, gpu_utilization, alpha=0.3, color='orange')
            ax.set_ylim(0, 100)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('GPU Utilization (%)')
            ax.set_title('GPU Compute Utilization', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            avg_util = np.mean(gpu_utilization)
            ax.axhline(y=avg_util, color='red', linestyle='--', alpha=0.5, label=f'Avg: {avg_util:.1f}%')
            ax.legend(loc='upper right', fontsize=8)
        else:
            # Show summary stats instead
            summary_text = self._generate_system_summary()
            ax.text(0.5, 0.5, summary_text, ha='center', va='center', transform=ax.transAxes, 
                   fontsize=11, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax.set_title('System Summary', fontweight='bold')
            ax.axis('off')
        
        plt.suptitle('System Resource Monitoring', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.figs_dir / 'system_curve.png', dpi=150)
        plt.close()
    
    def _generate_system_summary(self) -> str:
        """Generate summary text for system stats."""
        if not self.system_history:
            return "No data available"
        
        lines = []
        
        # CPU
        cpu = [h.get('cpu_percent', 0) or 0 for h in self.system_history]
        if any(cpu):
            lines.append(f"CPU:  Avg={np.mean(cpu):.1f}%  Max={np.max(cpu):.1f}%")
        
        # RAM
        ram = [h.get('ram_percent', 0) or 0 for h in self.system_history]
        ram_gb = [h.get('ram_used_gb', 0) or 0 for h in self.system_history]
        if any(ram):
            lines.append(f"RAM:  Avg={np.mean(ram):.1f}%  Max={np.max(ram):.1f}%")
            if any(ram_gb):
                lines.append(f"      Avg={np.mean(ram_gb):.1f}GB  Max={np.max(ram_gb):.1f}GB")
        
        # GPU
        gpu = [h.get('gpu_memory_percent', 0) or 0 for h in self.system_history]
        if any(gpu):
            lines.append(f"GPU:  Avg={np.mean(gpu):.1f}%  Max={np.max(gpu):.1f}%")
        
        # MPS
        mps = [h.get('mps_memory_allocated_gb', 0) or 0 for h in self.system_history]
        if any(mps):
            lines.append(f"MPS:  Avg={np.mean(mps):.2f}GB  Max={np.max(mps):.2f}GB")
        
        if not lines:
            return "No system data collected"
        
        return "\n".join(lines)
    
    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        epoch: Optional[int] = None,
        multi_label: bool = False,
    ) -> None:
        """Confusion matrix plot."""
        from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
        
        # Multi-label: Create per-label confusion matrices grid
        if multi_label:
            self._plot_multilabel_confusion_matrix(y_true, y_pred, class_names, epoch)
            return
        
        cm = confusion_matrix(y_true, y_pred)
        num_classes = cm.shape[0]
        
        # Class names
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        
        # Figure size based on num classes
        figsize = max(8, num_classes * 0.8)
        
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        
        # Heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Labels
        title = f'Confusion Matrix (Epoch {epoch})' if epoch else 'Confusion Matrix (Final)'
        ax.set(
            xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel='True Label',
            xlabel='Predicted Label',
            title=title
        )
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Text annotations
        thresh = cm.max() / 2.
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha='center', va='center',
                       color='white' if cm[i, j] > thresh else 'black')
        
        plt.tight_layout()
        
        # Save path
        if epoch is not None:
            save_path = self.figs_dir / 'confusion_matrix' / f'epoch_{epoch:03d}.png'
        else:
            save_path = self.figs_dir / 'confusion_matrix_final.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def _plot_multilabel_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """Multi-label confusion matrix plot (per-label 2x2 matrices grid)."""
        from sklearn.metrics import multilabel_confusion_matrix
        
        # Get per-label confusion matrices: shape [n_labels, 2, 2]
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        num_labels = mcm.shape[0]
        
        if class_names is None:
            class_names = [f'Label {i}' for i in range(num_labels)]
        
        # Create grid layout
        n_cols = min(4, num_labels)
        n_rows = (num_labels + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if num_labels == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx in range(num_labels):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            cm = mcm[idx]  # 2x2 matrix: [[TN, FP], [FN, TP]]
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            
            # Annotations
            thresh = cm.max() / 2.
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha='center', va='center',
                           color='white' if cm[i, j] > thresh else 'black')
            
            ax.set(
                xticks=[0, 1], yticks=[0, 1],
                xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'],
                ylabel='True', xlabel='Pred',
                title=class_names[idx][:20]  # Truncate long names
            )
        
        # Hide empty subplots
        for idx in range(num_labels, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].axis('off')
        
        title = f'Multi-Label Confusion Matrices (Epoch {epoch})' if epoch else 'Multi-Label Confusion Matrices (Final)'
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        # Save path
        if epoch is not None:
            save_path = self.figs_dir / 'confusion_matrix' / f'epoch_{epoch:03d}.png'
        else:
            save_path = self.figs_dir / 'confusion_matrix_final.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def _plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: Optional[List[str]] = None,
        epoch: Optional[int] = None,
        multi_label: bool = False,
    ) -> None:
        """ROC curve plot."""
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=(10, 8))
        
        # Multi-label: Each column is already binary
        if multi_label:
            num_labels = y_prob.shape[1]
            if class_names is None:
                class_names = [f'Label {i}' for i in range(num_labels)]
            
            aucs = []
            for i in range(num_labels):
                # Check if both classes exist for this label
                if len(np.unique(y_true[:, i])) < 2:
                    continue
                fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                plt.plot(fpr, tpr, lw=1.5, alpha=0.7, label=f'{class_names[i][:15]} ({roc_auc:.3f})')
            
            # Macro average
            if aucs:
                macro_auc = np.mean(aucs)
                plt.plot([], [], ' ', label=f'Macro AUC: {macro_auc:.3f}')
        
        # Binary classification
        elif y_prob.shape[1] == 2:
            # Tek sınıf kontrolü
            if len(np.unique(y_true)) < 2:
                plt.close()
                return
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        
        else:
            # Multiclass - one vs rest
            if len(np.unique(y_true)) < 2:
                plt.close()
                return
            if class_names is None:
                class_names = [f'Class {i}' for i in range(y_prob.shape[1])]
            
            for i in range(y_prob.shape[1]):
                y_true_binary = (y_true == i).astype(int)
                if len(np.unique(y_true_binary)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        title = f'ROC Curve (Epoch {epoch})' if epoch else 'ROC Curve (Final)'
        plt.title(title)
        plt.legend(loc='lower right', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save path
        if epoch is not None:
            save_path = self.figs_dir / 'roc' / f'epoch_{epoch:03d}.png'
        else:
            save_path = self.figs_dir / 'roc_curve_final.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def _plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: Optional[List[str]] = None,
        epoch: Optional[int] = None,
        multi_label: bool = False,
    ) -> None:
        """Precision-Recall curve plot."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        plt.figure(figsize=(10, 8))
        
        # Multi-label: Each column is already binary
        if multi_label:
            num_labels = y_prob.shape[1]
            if class_names is None:
                class_names = [f'Label {i}' for i in range(num_labels)]
            
            aps = []
            for i in range(num_labels):
                # Check if both classes exist for this label
                if len(np.unique(y_true[:, i])) < 2:
                    continue
                precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
                ap = average_precision_score(y_true[:, i], y_prob[:, i])
                aps.append(ap)
                plt.plot(recall, precision, lw=1.5, alpha=0.7, label=f'{class_names[i][:15]} ({ap:.3f})')
            
            # Macro average
            if aps:
                macro_ap = np.mean(aps)
                plt.plot([], [], ' ', label=f'Macro AP: {macro_ap:.3f}')
        
        # Binary classification
        elif y_prob.shape[1] == 2:
            # Tek sınıf kontrolü
            if len(np.unique(y_true)) < 2:
                plt.close()
                return
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
            ap = average_precision_score(y_true, y_prob[:, 1])
            plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {ap:.3f})')
        
        else:
            # Multiclass
            if len(np.unique(y_true)) < 2:
                plt.close()
                return
            if class_names is None:
                class_names = [f'Class {i}' for i in range(y_prob.shape[1])]
            
            for i in range(y_prob.shape[1]):
                y_true_binary = (y_true == i).astype(int)
                if len(np.unique(y_true_binary)) < 2:
                    continue
                precision, recall, _ = precision_recall_curve(y_true_binary, y_prob[:, i])
                ap = average_precision_score(y_true_binary, y_prob[:, i])
                plt.plot(recall, precision, lw=2, label=f'{class_names[i]} (AP = {ap:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        title = f'Precision-Recall Curve (Epoch {epoch})' if epoch else 'Precision-Recall Curve (Final)'
        plt.title(title)
        plt.legend(loc='lower left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save path
        if epoch is not None:
            save_path = self.figs_dir / 'pr_curve' / f'epoch_{epoch:03d}.png'
        else:
            save_path = self.figs_dir / 'pr_curve_final.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def _plot_class_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        epoch: Optional[int] = None,
        multi_label: bool = False,
    ) -> None:
        """Class distribution comparison plot."""
        
        # Multi-label: Show per-label positive counts
        if multi_label:
            num_labels = y_true.shape[1]
            if class_names is None:
                class_names = [f'Label {i}' for i in range(num_labels)]
            
            # Count positives per label
            true_counts = y_true.sum(axis=0)
            pred_counts = y_pred.sum(axis=0)
            
            x = np.arange(num_labels)
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(max(10, num_labels * 0.5), 6))
            bars1 = ax.bar(x - width/2, true_counts, width, label='True Positives', color='steelblue', alpha=0.8)
            bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted Positives', color='coral', alpha=0.8)
            
            ax.set_xlabel('Label')
            ax.set_ylabel('Positive Count')
            title = f'Multi-Label Distribution (Epoch {epoch})' if epoch else 'Multi-Label Distribution (Final)'
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels([n[:15] for n in class_names], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
        
        else:
            # Single-label (binary/multiclass)
            num_classes = len(np.unique(y_true))
            
            if class_names is None:
                class_names = [f'Class {i}' for i in range(num_classes)]
            
            # Count distributions
            true_counts = np.bincount(y_true, minlength=num_classes)
            pred_counts = np.bincount(y_pred, minlength=num_classes)
            
            x = np.arange(num_classes)
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars1 = ax.bar(x - width/2, true_counts, width, label='True', color='steelblue', alpha=0.8)
            bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted', color='coral', alpha=0.8)
            
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            title = f'Class Distribution (Epoch {epoch})' if epoch else 'Class Distribution (Final)'
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
        
        # Save path
        if epoch is not None:
            save_path = self.figs_dir / 'class_dist' / f'epoch_{epoch:03d}.png'
        else:
            save_path = self.figs_dir / 'class_distribution_final.png'
        plt.savefig(save_path, dpi=150)
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
