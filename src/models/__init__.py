"""
Models modülü.

Backbone modelleri ve classifier'ları içerir.

Kullanım:
    from src.models import MedicalImageClassifier, create_model
    
    # Birleşik model
    model = MedicalImageClassifier('resnet50', num_classes=2, mode='linear_probe')
    
    # Factory function
    model = create_model('vit_small_patch16', num_classes=2, mode='full_finetune')
"""

from .backbone import (
    # Config
    load_model_config,
    get_available_models,
    get_model_summary,
    # Backbone
    get_backbone,
    get_feature_dim,
    # Freezing
    freeze_backbone,
    unfreeze_backbone,
    # Layer-wise LR
    get_layer_wise_lr_groups,
    # Utils
    count_parameters,
)

from .classifier import (
    ClassifierHead,
    MedicalImageClassifier,
    create_model,
)


__all__ = [
    # Config
    'load_model_config',
    'get_available_models',
    'get_model_summary',
    # Backbone
    'get_backbone',
    'get_feature_dim',
    # Freezing
    'freeze_backbone',
    'unfreeze_backbone',
    # Layer-wise LR
    'get_layer_wise_lr_groups',
    # Utils
    'count_parameters',
    # Classifier
    'ClassifierHead',
    'MedicalImageClassifier',
    'create_model',
]
