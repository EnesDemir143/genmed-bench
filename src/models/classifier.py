"""
Classification head ve model modülü.

Backbone üzerine eklenecek classifier head'leri ve
birleşik model sınıfını içerir.

Kullanım:
    from src.models.classifier import MedicalImageClassifier
    
    # Birleşik model oluştur
    model = MedicalImageClassifier(
        model_name='resnet50',
        num_classes=2,
        pretrained=True,
        mode='linear_probe'
    )
    
    # Forward
    logits = model(images)
    
    # Feature extraction (GradCAM için)
    features = model.get_features(images)
"""

from typing import Optional, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import (
    get_backbone,
    get_feature_dim,
    load_model_config,
    freeze_backbone,
    count_parameters,
)


# =============================================================================
# CLASSIFIER HEAD
# =============================================================================

class ClassifierHead(nn.Module):
    """
    Classification head.
    
    Backbone çıktısına eklenen lineer veya MLP classifier.
    Single-label ve multi-label classification destekler.
    
    Args:
        in_features: Input feature dimension
        num_classes: Output class sayısı
        hidden_dim: MLP hidden layer dimension (None = linear)
        dropout: Dropout oranı
        multi_label: True ise sigmoid, False ise raw logits
        
    Example:
        # Single-label (COVID detection)
        head = ClassifierHead(768, num_classes=2)
        
        # Multi-label (NIH ChestXray - 14 findings)
        head = ClassifierHead(768, num_classes=14, multi_label=True)
        
        # MLP head
        head = ClassifierHead(768, num_classes=2, hidden_dim=256)
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        multi_label: bool = False,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        self.multi_label = multi_label
        
        if hidden_dim is not None:
            # MLP head
            self.classifier = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            # Linear head
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(in_features, num_classes),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Features [B, in_features]
            
        Returns:
            Logits [B, num_classes]
        """
        return self.classifier(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Probability prediction.
        
        Args:
            x: Features [B, in_features]
            
        Returns:
            Probabilities [B, num_classes]
        """
        logits = self.forward(x)
        
        if self.multi_label:
            return torch.sigmoid(logits)
        else:
            return F.softmax(logits, dim=-1)


# =============================================================================
# MEDICAL IMAGE CLASSIFIER
# =============================================================================

class MedicalImageClassifier(nn.Module):
    """
    Backbone + ClassifierHead birleşik model.
    
    configs/models.yaml'dan model konfigürasyonlarını okur.
    Linear probe ve full fine-tuning modlarını destekler.
    
    Args:
        model_name: Backbone model adı (örn: 'resnet50', 'vit_small_patch16')
        num_classes: Output class sayısı
        pretrained: ImageNet pretrained weights kullan
        mode: 'linear_probe' veya 'full_finetune'
        multi_label: Multi-label classification (NIH için)
        head_hidden_dim: MLP head için hidden dim (None = linear)
        head_dropout: Head dropout oranı
        
    Example:
        # Linear probe
        model = MedicalImageClassifier(
            model_name='resnet50',
            num_classes=2,
            pretrained=True,
            mode='linear_probe'
        )
        
        # Full fine-tune
        model = MedicalImageClassifier(
            model_name='vit_small_patch16',
            num_classes=14,
            pretrained=True,
            mode='full_finetune',
            multi_label=True
        )
    """
    
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        mode: str = 'linear_probe',
        multi_label: bool = False,
        head_hidden_dim: Optional[int] = None,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        
        # Config yükle
        self.config = load_model_config(model_name, mode=mode)
        self.model_name = model_name
        self.num_classes = num_classes
        self.mode = mode
        self.multi_label = multi_label
        
        # Backbone oluştur (classifier yok, feature extractor olarak)
        self.backbone = get_backbone(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Feature extractor
            drop_path_rate=self.config.get('drop_path_rate', 0.0),
        )
        
        # Feature dimension
        self.feature_dim = get_feature_dim(self.backbone)
        
        # Classification head
        self.head = ClassifierHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            multi_label=multi_label,
        )
        
        # Linear probe ise backbone'u dondur
        if mode == 'linear_probe':
            freeze_backbone(self.backbone)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Logits [B, num_classes]
        """
        features = self.get_features(x)
        logits = self.head(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Backbone features çıkar (GradCAM için).
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Features [B, feature_dim]
        """
        features = self.backbone(x)
        
        # Bazı modeller spatial output verir [B, C, H, W]
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        # Bazı modeller [B, N, C] verir (ViT)
        elif features.dim() == 3:
            # CLS token veya mean pooling
            features = features[:, 0] if features.size(1) > 1 else features.squeeze(1)
        
        return features
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Probability prediction."""
        features = self.get_features(x)
        return self.head.predict_proba(features)
    
    def get_trainable_params(self) -> int:
        """Eğitilebilir parametre sayısı."""
        return count_parameters(self, trainable_only=True)
    
    def get_total_params(self) -> int:
        """Toplam parametre sayısı."""
        return count_parameters(self, trainable_only=False)
    
    def freeze_backbone(self) -> None:
        """Backbone'u dondur (linear probe)."""
        freeze_backbone(self.backbone)
    
    def unfreeze_backbone(self) -> None:
        """Backbone'u aç (full fine-tune)."""
        from .backbone import unfreeze_backbone
        unfreeze_backbone(self.backbone)
    
    def get_input_size(self) -> int:
        """Model input size."""
        return self.config.get('input_size', 224)
    
    def __repr__(self) -> str:
        return (
            f"MedicalImageClassifier(\n"
            f"  model_name='{self.model_name}',\n"
            f"  num_classes={self.num_classes},\n"
            f"  mode='{self.mode}',\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  trainable_params={self.get_trainable_params():,},\n"
            f"  total_params={self.get_total_params():,}\n"
            f")"
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    mode: str = 'linear_probe',
    multi_label: bool = False,
    **kwargs
) -> MedicalImageClassifier:
    """
    Model factory function.
    
    CLI veya config'den model oluşturmak için convenience function.
    
    Args:
        model_name: Model adı
        num_classes: Class sayısı
        pretrained: Pretrained weights
        mode: Training mode
        multi_label: Multi-label classification
        **kwargs: Ek MedicalImageClassifier argümanları
        
    Returns:
        MedicalImageClassifier instance
    """
    return MedicalImageClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        mode=mode,
        multi_label=multi_label,
        **kwargs
    )
