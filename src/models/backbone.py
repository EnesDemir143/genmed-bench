"""
Backbone model modülü.

timm kütüphanesi ile pretrained backbone modelleri oluşturur.
configs/models.yaml'dan model konfigürasyonlarını okur.

Kullanım:
    from src.models.backbone import get_backbone, load_model_config
    
    # Config yükle
    config = load_model_config('resnet50', mode='linear_probe')
    
    # Backbone oluştur
    model = get_backbone('resnet50', pretrained=True, num_classes=0)
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import copy

import torch
import torch.nn as nn
import timm
import yaml


# =============================================================================
# CONFIG LOADING
# =============================================================================

_CONFIG_CACHE: Dict[str, Any] = {}


def _get_config_path() -> Path:
    """Config dosyasının path'ini döner."""
    # src/models/backbone.py -> configs/models.yaml
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    return project_root / "configs" / "models.yaml"


def _load_yaml_config() -> Dict[str, Any]:
    """YAML config dosyasını yükler ve cache'ler."""
    global _CONFIG_CACHE
    
    if not _CONFIG_CACHE:
        config_path = _get_config_path()
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    
    return _CONFIG_CACHE


def get_available_models() -> List[str]:
    """Kullanılabilir model isimlerini döner."""
    config = _load_yaml_config()
    return list(config.get('models', {}).keys())


def load_model_config(
    model_name: str,
    mode: str = 'linear_probe'
) -> Dict[str, Any]:
    """
    Model konfigürasyonunu yükler.
    
    Model-specific ayarlar, global defaults ile merge edilir.
    
    Args:
        model_name: Model adı (örn: 'resnet50', 'vit_small_patch16')
        mode: Training modu ('linear_probe' veya 'full_finetune')
        
    Returns:
        Merged config dictionary
        
    Raises:
        ValueError: Model veya mode bulunamazsa
    """
    config = _load_yaml_config()
    
    # Model var mı kontrol et
    if model_name not in config.get('models', {}):
        available = get_available_models()
        raise ValueError(
            f"Model '{model_name}' not found. Available: {available}"
        )
    
    # Mode var mı kontrol et
    if mode not in ('linear_probe', 'full_finetune'):
        raise ValueError(
            f"Invalid mode '{mode}'. Use 'linear_probe' or 'full_finetune'"
        )
    
    # Defaults
    defaults = copy.deepcopy(config.get('defaults', {}))
    
    # Training mode defaults
    mode_defaults = copy.deepcopy(
        config.get('training_modes', {}).get(mode, {})
    )
    
    # Model specific config
    model_config = copy.deepcopy(config['models'][model_name])
    
    # Model-specific mode override
    model_mode_config = model_config.pop(mode, {})
    
    # Diğer mode'u da kaldır (temizlik)
    other_mode = 'full_finetune' if mode == 'linear_probe' else 'linear_probe'
    model_config.pop(other_mode, None)
    
    # Merge: defaults <- mode_defaults <- model_config <- model_mode_config
    result = {}
    result.update(defaults)
    
    # Mode defaults'u merge et (nested dict'ler için deep merge)
    _deep_merge(result, mode_defaults)
    
    # Model config'u merge et
    _deep_merge(result, model_config)
    
    # Model-specific mode override'ı merge et
    _deep_merge(result, model_mode_config)
    
    # Mode bilgisini ekle
    result['mode'] = mode
    result['model_name'] = model_name
    
    return result


def _deep_merge(base: Dict, update: Dict) -> None:
    """Nested dictionary'leri deep merge eder (in-place)."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# =============================================================================
# BACKBONE CREATION
# =============================================================================

def get_backbone(
    model_name: str,
    pretrained: bool = True,
    num_classes: int = 0,
    **kwargs
) -> nn.Module:
    """
    Backbone model oluşturur.
    
    Args:
        model_name: Model adı (örn: 'resnet50', 'vit_small_patch16')
        pretrained: ImageNet pretrained weights kullan
        num_classes: Output sınıf sayısı (0 = feature extractor)
        **kwargs: Ek timm.create_model argümanları
        
    Returns:
        PyTorch model
        
    Example:
        # Feature extractor (classifier yok)
        backbone = get_backbone('resnet50', pretrained=True, num_classes=0)
        
        # Classifier ile
        model = get_backbone('resnet50', pretrained=True, num_classes=2)
    """
    # Config'den timm_name al
    config = load_model_config(model_name, mode='linear_probe')
    
    # MedGemma özel handling
    if config.get('type') == 'medgemma':
        raise NotImplementedError(
            "MedGemma şu an desteklenmiyor. HuggingFace entegrasyonu gerekli."
        )
    
    timm_name = config.get('timm_name', model_name)
    input_size = config.get('input_size', 224)
    drop_rate = config.get('drop_rate', 0.0)
    drop_path_rate = kwargs.pop('drop_path_rate', config.get('drop_path_rate', 0.0))
    
    # Model oluştur
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        **kwargs
    )
    
    # Meta bilgileri ekle
    model.config = config
    model.input_size = input_size
    model.model_name = model_name
    
    return model


def get_feature_dim(model: nn.Module) -> int:
    """
    Model'in feature dimension'ını döner.
    
    Args:
        model: timm ile oluşturulmuş model
        
    Returns:
        Feature dimension (int)
    """
    # timm modelleri için num_features attribute'u var
    if hasattr(model, 'num_features'):
        return model.num_features
    
    # Alternatif: forward_features + dummy input ile
    if hasattr(model, 'forward_features'):
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = model.forward_features(dummy)
            if isinstance(features, (tuple, list)):
                features = features[-1]
            # [B, C, H, W] veya [B, N, C] formatı olabilir
            if features.dim() == 4:
                return features.shape[1]
            elif features.dim() == 3:
                return features.shape[-1]
            else:
                return features.shape[-1]
    
    raise ValueError("Cannot determine feature dimension for this model")


# =============================================================================
# BACKBONE FREEZING
# =============================================================================

def freeze_backbone(
    model: nn.Module,
    trainable_patterns: Optional[List[str]] = None
) -> int:
    """
    Backbone parametrelerini dondurur.
    
    Linear probe için: sadece head/fc/classifier eğitilir.
    
    Args:
        model: PyTorch model
        trainable_patterns: Eğitilebilir layer isimleri listesi.
            None ise ['head', 'fc', 'classifier'] kullanılır.
            
    Returns:
        Dondurulan parametre sayısı
        
    Example:
        freeze_backbone(model)  # Sadece head eğitilir
        freeze_backbone(model, ['head', 'fc', 'layer4'])  # layer4 de eğitilir
    """
    if trainable_patterns is None:
        trainable_patterns = ['head', 'fc', 'classifier']
    
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        # Pattern'lerden herhangi biri name içinde mi?
        is_trainable = any(pattern in name for pattern in trainable_patterns)
        
        if is_trainable:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    
    return frozen_count


def unfreeze_backbone(model: nn.Module) -> None:
    """Tüm parametreleri eğitilebilir yapar."""
    for param in model.parameters():
        param.requires_grad = True


# =============================================================================
# LAYER-WISE LEARNING RATE DECAY
# =============================================================================

def get_layer_wise_lr_groups(
    model: nn.Module,
    base_lr: float,
    layer_decay: float = 0.65,
    weight_decay: float = 0.05,
    no_weight_decay_patterns: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Layer-wise learning rate decay için param groups oluşturur.
    
    Derin layer'lar daha düşük lr alır (transfer learning best practice).
    
    Args:
        model: PyTorch model
        base_lr: Head için base learning rate
        layer_decay: Her layer için lr decay factor
        weight_decay: L2 regularization
        no_weight_decay_patterns: Weight decay uygulanmayacak pattern'ler
            (default: ['bias', 'norm', 'bn'])
            
    Returns:
        Optimizer için param groups listesi
    """
    if no_weight_decay_patterns is None:
        no_weight_decay_patterns = ['bias', 'norm', 'bn', 'gamma', 'beta']
    
    # Layer derinliklerini belirle
    param_groups = []
    seen = set()
    
    # Layer ID'leri hesapla (Vision Transformer için)
    num_layers = _get_num_layers(model)
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in seen:
            continue
        seen.add(name)
        
        # Weight decay var mı?
        apply_wd = not any(p in name for p in no_weight_decay_patterns)
        
        # Layer ID ve lr hesapla
        layer_id = _get_layer_id(name, num_layers)
        lr_scale = layer_decay ** (num_layers - layer_id)
        
        param_groups.append({
            'params': [param],
            'lr': base_lr * lr_scale,
            'weight_decay': weight_decay if apply_wd else 0.0,
            'name': name,
            'layer_id': layer_id,
        })
    
    return param_groups


def _get_num_layers(model: nn.Module) -> int:
    """Model'deki layer sayısını tahmin eder."""
    # ViT için: blocks sayısı
    if hasattr(model, 'blocks'):
        return len(model.blocks) + 1
    
    # Swin için: layers sayısı
    if hasattr(model, 'layers'):
        return len(model.layers) + 1
    
    # CNN için: stage/layer sayısı
    for attr in ['layer4', 'stages', 'features']:
        if hasattr(model, attr):
            return 4  # Typical CNN depth
    
    return 12  # Default


def _get_layer_id(name: str, num_layers: int) -> int:
    """Parameter isminden layer ID'sini çıkarır."""
    # Embedding/patch_embed -> layer 0
    if any(x in name for x in ['patch_embed', 'pos_embed', 'cls_token', 'stem']):
        return 0
    
    # Head/classifier -> son layer
    if any(x in name for x in ['head', 'fc', 'classifier']):
        return num_layers
    
    # blocks.X veya layers.X.blocks.Y formatı
    import re
    
    # ViT: blocks.0, blocks.11
    match = re.search(r'blocks\.(\d+)', name)
    if match:
        return int(match.group(1)) + 1
    
    # Swin: layers.0.blocks.0
    match = re.search(r'layers\.(\d+)', name)
    if match:
        return int(match.group(1)) + 1
    
    # CNN: layer1, layer2, layer3, layer4
    match = re.search(r'layer(\d+)', name)
    if match:
        return int(match.group(1))
    
    # Default: orta katman
    return num_layers // 2


# =============================================================================
# UTILITIES
# =============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Model parametrelerini sayar."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_summary(model_name: str) -> Dict[str, Any]:
    """Model özet bilgilerini döner."""
    config = load_model_config(model_name)
    return {
        'name': model_name,
        'timm_name': config.get('timm_name'),
        'type': config.get('type'),
        'params_m': config.get('params_m'),
        'input_size': config.get('input_size', 224),
    }
