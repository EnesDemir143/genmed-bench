"""
Supervised trainer modülü.

Classification için trainer.

Kullanım:
    from src.train.trainer_sup import SupervisedTrainer
    
    trainer = SupervisedTrainer(model, train_loader, val_loader, config, logger)
    trainer.fit(epochs=100)
"""

from typing import Dict, Any, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

from src.utils.system_monitor import AverageMeter
from .trainer_base import BaseTrainer


class SupervisedTrainer(BaseTrainer):
    """
    Supervised classification trainer.
    
    Binary, multiclass ve multi-label destekler.
    XDomainMix/PipMix augmentation entegrasyonu.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: Dict[str, Any],
        logger,
        device: str = 'auto',
        augmentation: Optional[Callable] = None,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Training config
            logger: ExperimentLogger
            device: Device
            augmentation: Batch-level augmentation (XDomainMixBatch, PipMixBatch)
            label_smoothing: Label smoothing değeri
        """
        super().__init__(model, train_loader, val_loader, config, logger, device)
        
        self.augmentation = augmentation
        self.label_smoothing = label_smoothing
        
        # Loss function
        if self.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Gradient clipping
        self.gradient_clip = config.get('gradient_clip', None)
    
    def train_epoch(self) -> Dict[str, float]:
        """Bir epoch train et."""
        self.model.train()
        
        loss_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}', leave=False)
        
        for batch in pbar:
            images, labels = batch[0], batch[1]
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Domain bilgisi (augmentation için)
            domains = batch[2] if len(batch) > 2 else None
            
            # Batch-level augmentation (XDomainMix, PipMix)
            if self.augmentation is not None and domains is not None:
                images, mixed_labels, lam = self.augmentation(images, labels, domains)
                use_mixup_loss = True
            else:
                mixed_labels = labels
                lam = None
                use_mixup_loss = False
            
            # Zero grad
            self.optimizer.zero_grad()
            
            # Forward
            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss = self._compute_loss(logits, mixed_labels, lam, use_mixup_loss)
                
                # Backward with scaler
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self._compute_loss(logits, mixed_labels, lam, use_mixup_loss)
                
                # Backward
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Update meter
            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
        return {'loss': loss_meter.avg}
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lam: Optional[torch.Tensor] = None,
        use_mixup_loss: bool = False,
    ) -> torch.Tensor:
        """
        Loss hesapla.
        
        Args:
            logits: Model çıktısı [B, C]
            labels: Labels [B] veya mixed labels [B, C]
            lam: Mixup lambda değerleri [B]
            use_mixup_loss: Mixup loss kullan
        """
        if self.multi_label:
            # Multi-label: BCE
            return self.criterion(logits, labels.float())
        
        if use_mixup_loss and lam is not None:
            # Mixup loss: soft labels ile cross entropy
            # labels shape: [B, num_classes] (one-hot veya soft)
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(labels * log_probs).sum(dim=1).mean()
            return loss
        
        # Standard cross entropy
        return self.criterion(logits, labels)


def create_trainer(
    model: nn.Module,
    train_loader,
    val_loader,
    config: Dict[str, Any],
    logger,
    device: str = 'auto',
    augmentation_name: Optional[str] = None,
) -> SupervisedTrainer:
    """
    Trainer factory function.
    
    Args:
        model: Model
        train_loader: Training loader
        val_loader: Validation loader
        config: Config
        logger: Logger
        device: Device
        augmentation_name: Augmentation adı ('xdomainmix', 'pipmix', None)
        
    Returns:
        SupervisedTrainer
    """
    # Augmentation setup
    augmentation = None
    if augmentation_name:
        if augmentation_name.lower() == 'xdomainmix':
            from src.data.augmentation import XDomainMixBatch
            augmentation = XDomainMixBatch(
                alpha=config.get('mixup_alpha', 0.2),
                prob=config.get('mixup_prob', 0.5),
            )
        elif augmentation_name.lower() == 'pipmix':
            from src.data.augmentation import PipMixBatch
            augmentation = PipMixBatch(
                patch_size=config.get('patch_size', 32),
                alpha=config.get('mixup_alpha', 0.4),
            )
    
    return SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
        device=device,
        augmentation=augmentation,
        label_smoothing=config.get('label_smoothing', 0.0),
    )
