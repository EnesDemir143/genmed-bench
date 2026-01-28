"""
Base trainer modülü.

Tüm trainer'ların base sınıfını içerir.

Kullanım:
    from src.train.trainer_base import BaseTrainer
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.models import load_model_config, get_layer_wise_lr_groups
from src.utils.metrics import compute_all_metrics
from src.utils.system_monitor import SystemMonitor, AverageMeter, Timer
from .experiment_logger import ExperimentLogger


class BaseTrainer(ABC):
    """
    Base trainer sınıfı.
    
    Tüm trainer'ların ortak işlevlerini sağlar:
    - Optimizer/scheduler oluşturma
    - Mixed precision
    - Checkpoint kaydetme/yükleme
    - Logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        logger: ExperimentLogger,
        device: str = 'auto',
    ):
        """
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Training config (merged from models.yaml)
            logger: ExperimentLogger instance
            device: Device ('cuda', 'mps', 'cpu', 'auto')
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        
        # Device setup
        self.device = self._setup_device(device)
        self.model = self.model.to(self.device)
        
        # Mixed precision
        self.use_amp = config.get('mixed_precision', True) and self.device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Optimizer & Scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.best_epoch = 0
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping', 0)
        self.early_stopping_counter = 0
        
        # System monitor
        self.system_monitor = SystemMonitor()
        
        # Multi-label flag
        self.multi_label = config.get('multi_label', False)
    
    def _setup_device(self, device: str) -> str:
        """Device setup."""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            return 'cpu'
        return device
    
    # =========================================================================
    # OPTIMIZER & SCHEDULER
    # =========================================================================
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Optimizer oluştur (config'den)."""
        opt_config = self.config.get('optimizer', {})
        opt_name = opt_config.get('name', 'AdamW')
        lr = self.config.get('lr', opt_config.get('lr', 1e-4))
        weight_decay = self.config.get('weight_decay', opt_config.get('weight_decay', 0.0))
        
        # Layer-wise LR decay (transformerlar için)
        layer_decay = self.config.get('layer_decay', None)
        
        if layer_decay and layer_decay < 1.0:
            param_groups = get_layer_wise_lr_groups(
                self.model,
                base_lr=lr,
                layer_decay=layer_decay,
                weight_decay=weight_decay,
            )
        else:
            param_groups = [{'params': self.model.parameters(), 'lr': lr}]
        
        # Optimizer seçimi
        if opt_name.lower() == 'sgd':
            momentum = opt_config.get('momentum', 0.9)
            nesterov = opt_config.get('nesterov', True)
            return torch.optim.SGD(
                param_groups,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
            )
        
        elif opt_name.lower() == 'adamw':
            betas = opt_config.get('betas', [0.9, 0.999])
            return torch.optim.AdamW(
                param_groups,
                lr=lr,
                betas=tuple(betas),
                weight_decay=weight_decay,
            )
        
        elif opt_name.lower() == 'adam':
            betas = opt_config.get('betas', [0.9, 0.999])
            return torch.optim.Adam(
                param_groups,
                lr=lr,
                betas=tuple(betas),
                weight_decay=weight_decay,
            )
        
        elif opt_name.lower() == 'rmsprop':
            momentum = opt_config.get('momentum', 0.9)
            eps = opt_config.get('eps', 1e-8)
            return torch.optim.RMSprop(
                param_groups,
                lr=lr,
                momentum=momentum,
                eps=eps,
                weight_decay=weight_decay,
            )
        
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Scheduler oluştur (config'den)."""
        sched_config = self.config.get('scheduler', {})
        sched_name = sched_config.get('name', 'cosine')
        epochs = self.config.get('epochs', 100)
        warmup_epochs = self.config.get('warmup_epochs', 0)
        
        # Main scheduler
        if sched_name == 'cosine':
            T_max = sched_config.get('T_max', epochs)
            eta_min = sched_config.get('eta_min', 0)
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=eta_min,
            )
        
        elif sched_name == 'cosine_with_warmup':
            T_max = sched_config.get('T_max', epochs)
            eta_min = sched_config.get('eta_min', 1e-6)
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max - warmup_epochs,
                eta_min=eta_min,
            )
        
        elif sched_name == 'step':
            step_size = sched_config.get('step_size', 30)
            gamma = sched_config.get('gamma', 0.1)
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma,
            )
        
        elif sched_name == 'none' or sched_name is None:
            return None
        
        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")
        
        # Warmup wrapper
        if warmup_epochs > 0:
            warmup_lr_init = sched_config.get('warmup_lr_init', 1e-7)
            # Linear warmup scheduler
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=warmup_lr_init / self.config.get('lr', 1e-4),
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs],
            )
        
        return main_scheduler
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    def fit(
        self,
        epochs: Optional[int] = None,
        save_predictions: str = 'best',  # 'best', 'last', 'both', 'none'
    ) -> Dict[str, Any]:
        """
        Ana training loop.
        
        Args:
            epochs: Epoch sayısı (None ise config'den)
            save_predictions: Prediction kaydetme stratejisi
            
        Returns:
            Training summary
        """
        if epochs is None:
            epochs = self.config.get('epochs', 100)
        
        import logging
        log = logging.getLogger("train")
        
        # Resume kontrolü - checkpoint'ten yüklendiyse devam et
        start_epoch = self.current_epoch + 1 if self.current_epoch > 0 else 1
        
        log.info(f"Training: {self.config.get('model_name', 'model')}")
        log.info(f"Device: {self.device}")
        log.info(f"Epochs: {start_epoch} -> {epochs}")
        if start_epoch > 1:
            log.info(f"Resuming from epoch {start_epoch} (best metric: {self.best_metric:.4f})")
        
        for epoch in range(start_epoch, epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            train_metrics['lr'] = self.optimizer.param_groups[0]['lr']
            self.logger.log_train(epoch, train_metrics)
            
            # Validate
            val_metrics, val_probs, val_labels = self.validate()
            is_best = self.logger.log_val(epoch, val_metrics)
            
            # System stats
            system_stats = self.system_monitor.get_stats(epoch)
            self.logger.log_system(system_stats.to_dict())
            
            # Save predictions
            if save_predictions == 'best' and is_best:
                self.logger.save_predictions(val_probs, val_labels, is_best=True, epoch=epoch)
            elif save_predictions == 'both':
                if is_best:
                    self.logger.save_predictions(val_probs, val_labels, is_best=True, epoch=epoch)
            
            # Save checkpoints
            if is_best:
                self.best_metric = val_metrics.get('roc_auc', val_metrics.get('accuracy', 0))
                self.best_epoch = epoch
                self.save_checkpoint(self.logger.get_checkpoint_path('best'), is_best=True)
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Per-epoch figures + aggregate curves güncelle
            if self.multi_label:
                val_preds = (val_probs >= 0.5).astype(int)
            else:
                val_preds = np.argmax(val_probs, axis=1)
            self.logger.plot_epoch_figures(epoch, val_labels, val_preds, val_probs, multi_label=self.multi_label)
            
            # Print progress
            self._print_progress(epoch, epochs, train_metrics, val_metrics, is_best)
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Early stopping
            if self.early_stopping_patience > 0:
                if self.early_stopping_counter >= self.early_stopping_patience:
                    log.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Son epoch predictions
        if save_predictions in ('last', 'both'):
            self.logger.save_predictions(
                val_probs, val_labels, is_last=True, epoch=self.current_epoch
            )
        
        # Son checkpoint
        self.save_checkpoint(self.logger.get_checkpoint_path('last'), is_best=False)
        
        # Plot training curves
        self.logger.plot_curves()
        
        # Plot final figures (confusion matrix, ROC, PR curves)
        # val_probs ve val_labels son validasyondan geliyor
        if self.multi_label:
            val_preds = (val_probs >= 0.5).astype(int)
        else:
            val_preds = np.argmax(val_probs, axis=1)
        self.logger.plot_final_figures(val_labels, val_preds, val_probs, multi_label=self.multi_label)
        
        # Summary
        self.logger.print_summary()
        
        return self.logger.get_summary()
    
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """
        Bir epoch train et.
        
        Returns:
            {'loss': ..., ...}
        """
        pass
    
    def validate(self) -> tuple:
        """
        Validation yap.
        
        Returns:
            (metrics_dict, probs, labels)
        """
        self.model.eval()
        
        all_probs = []
        all_labels = []
        val_loss = AverageMeter()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating', leave=False):
                images, labels = batch[0], batch[1]
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward
                logits = self.model(images)
                loss = self._compute_loss(logits, labels)
                
                val_loss.update(loss.item(), images.size(0))
                
                # Probabilities
                if self.multi_label:
                    probs = torch.sigmoid(logits)
                else:
                    probs = torch.softmax(logits, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concat
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Predictions
        if self.multi_label:
            all_preds = (all_probs >= 0.5).astype(int)
        else:
            all_preds = np.argmax(all_probs, axis=1)
        
        # Metrics
        metrics = compute_all_metrics(
            all_labels, all_preds, all_probs, multi_label=self.multi_label
        )
        metrics['loss'] = val_loss.avg
        
        return metrics, all_probs, all_labels
    
    @abstractmethod
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Loss hesapla."""
        pass
    
    def _print_progress(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict,
        val_metrics: Dict,
        is_best: bool,
    ) -> None:
        """Epoch progress yazdır."""
        lr = train_metrics.get('lr', 0)
        train_loss = train_metrics.get('loss', 0)
        val_loss = val_metrics.get('loss', 0)
        val_acc = val_metrics.get('accuracy', 0)
        val_auc = val_metrics.get('roc_auc', val_metrics.get('roc_auc_macro', 0)) or 0
        
        best_marker = ' *' if is_best else ''
        
        import logging
        log = logging.getLogger("train")
        log.info(
            f"Epoch [{epoch:3d}/{total_epochs}] "
            f"LR: {lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"AUC: {val_auc:.4f}{best_marker}"
        )
    
    # =========================================================================
    # CHECKPOINTS
    # =========================================================================
    
    def save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Checkpoint kaydet."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'config': self.config,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path, load_optimizer: bool = True) -> None:
        """Checkpoint yükle."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.best_epoch = checkpoint['best_epoch']
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        import logging
        log = logging.getLogger("train")
        log.info(f"Loaded checkpoint from epoch {self.current_epoch}")
