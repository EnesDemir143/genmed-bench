#!/usr/bin/env python
"""
GenMed-Bench Training Script.

CLI entry point for training models.

Kullanım:
    python train.py --model resnet50 --mode linear_probe --dataset nih
    
    python train.py \
        --model vit_small_patch16 \
        --mode full_finetune \
        --augmentation xdomainmix \
        --dataset covidx \
        --batch_size 32 \
        --epochs 100 \
        --lr 5e-5 \
        --seed 42
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.seed import set_seed, get_generator, worker_init_fn
from src.models import MedicalImageClassifier, load_model_config
from src.train.experiment_logger import ExperimentLogger
from src.train.trainer_sup import SupervisedTrainer, create_trainer


def parse_args():
    """CLI argümanlarını parse et."""
    parser = argparse.ArgumentParser(
        description='GenMed-Bench Training Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--model', type=str, required=True,
        help='Model adı (örn: resnet50, vit_small_patch16)'
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['nih', 'covidx', 'vinbigdata'],
        help='Dataset adı'
    )
    
    # Training mode
    parser.add_argument(
        '--mode', type=str, default='linear_probe',
        choices=['linear_probe', 'full_finetune'],
        help='Training modu'
    )
    
    # Augmentation
    parser.add_argument(
        '--augmentation', type=str, default='none',
        choices=['none', 'xdomainmix', 'pipmix'],
        help='Augmentation tipi'
    )
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=None, help='Epoch sayısı')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=None, help='Warmup epochs')
    
    # Regularization
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing')
    parser.add_argument('--gradient_clip', type=float, default=None, help='Gradient clipping')
    parser.add_argument('--drop_path_rate', type=float, default=None, help='Drop path rate')
    
    # Augmentation params
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha')
    parser.add_argument('--mixup_prob', type=float, default=0.5, help='Mixup probability')
    
    # System
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda, mps, cpu, auto)')
    parser.add_argument('--mixed_precision', action='store_true', help='Mixed precision training')
    parser.add_argument('--no_mixed_precision', action='store_true', help='Disable mixed precision')
    
    # Early stopping & checkpointing
    parser.add_argument('--early_stopping', type=int, default=0, help='Early stopping patience (0=disabled)')
    parser.add_argument(
        '--save_predictions', type=str, default='best',
        choices=['none', 'best', 'last', 'both'],
        help='Prediction kaydetme stratejisi'
    )
    
    # Paths
    parser.add_argument('--run_dir', type=str, default='runs', help='Run directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Multi-label
    parser.add_argument('--multi_label', action='store_true', help='Multi-label classification')
    
    return parser.parse_args()


def get_dataset(
    dataset_name: str,
    split: str,
    transform=None,
    multi_label: bool = False
):
    """Dataset yükle."""
    # Import dataset sınıfları
    from src.data.dataset import (
        NIHChestXrayDataset,
        COVIDxDataset,
        VinBigDataDataset,
    )
    
    # Dataset paths (config'den veya default)
    base_path = Path('data/processed')
    
    if dataset_name == 'nih':
        lmdb_path = base_path / 'nih' / 'nih.lmdb'
        metadata_path = base_path / 'nih' / f'nih_{split}.parquet'
        return NIHChestXrayDataset(
            lmdb_path=str(lmdb_path),
            metadata_path=str(metadata_path),
            transform=transform,
        )
    
    elif dataset_name == 'covidx':
        lmdb_path = base_path / 'covidx' / 'covidx.lmdb'
        metadata_path = base_path / 'covidx' / f'covidx_{split}.parquet'
        return COVIDxDataset(
            lmdb_path=str(lmdb_path),
            metadata_path=str(metadata_path),
            transform=transform,
        )
    
    elif dataset_name == 'vinbigdata':
        lmdb_path = base_path / 'vinbigdata' / 'vinbigdata.lmdb'
        metadata_path = base_path / 'vinbigdata' / f'vinbigdata_{split}.parquet'
        return VinBigDataDataset(
            lmdb_path=str(lmdb_path),
            metadata_path=str(metadata_path),
            transform=transform,
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_transforms(input_size: int, is_train: bool):
    """Transform pipeline oluştur."""
    import torchvision.transforms as T
    
    # Dataset-specific normalization (hesaplandı)
    # Default: ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if is_train:
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


def main():
    """Ana training fonksiyonu."""
    args = parse_args()
    
    # =========================================================================
    # SETUP
    # =========================================================================
    
    # Seed
    set_seed(args.seed)
    print(f"🎲 Seed: {args.seed}")
    
    # Model config yükle
    model_config = load_model_config(args.model, mode=args.mode)
    
    # CLI argümanlarıyla override
    if args.epochs is not None:
        model_config['epochs'] = args.epochs
    if args.batch_size is not None:
        model_config['batch_size'] = args.batch_size
    if args.lr is not None:
        model_config['lr'] = args.lr
    if args.weight_decay is not None:
        model_config['weight_decay'] = args.weight_decay
    if args.warmup_epochs is not None:
        model_config['warmup_epochs'] = args.warmup_epochs
    if args.gradient_clip is not None:
        model_config['gradient_clip'] = args.gradient_clip
    if args.drop_path_rate is not None:
        model_config['drop_path_rate'] = args.drop_path_rate
    if args.early_stopping > 0:
        model_config['early_stopping'] = args.early_stopping
    
    # Mixed precision
    if args.no_mixed_precision:
        model_config['mixed_precision'] = False
    elif args.mixed_precision:
        model_config['mixed_precision'] = True
    
    # Multi-label
    model_config['multi_label'] = args.multi_label
    
    # Augmentation params
    model_config['mixup_alpha'] = args.mixup_alpha
    model_config['mixup_prob'] = args.mixup_prob
    model_config['label_smoothing'] = args.label_smoothing
    
    # Add CLI args to config for logging
    model_config['seed'] = args.seed
    model_config['dataset'] = args.dataset
    model_config['augmentation'] = args.augmentation
    
    # =========================================================================
    # DATA
    # =========================================================================
    
    input_size = model_config.get('input_size', 224)
    batch_size = model_config.get('batch_size', 32)
    
    print(f"📊 Dataset: {args.dataset}")
    print(f"📐 Input size: {input_size}")
    print(f"📦 Batch size: {batch_size}")
    
    # Transforms
    train_transform = get_transforms(input_size, is_train=True)
    val_transform = get_transforms(input_size, is_train=False)
    
    # Datasets
    train_dataset = get_dataset(args.dataset, 'train', train_transform, args.multi_label)
    val_dataset = get_dataset(args.dataset, 'val', val_transform, args.multi_label)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # DataLoaders
    g = get_generator(args.seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=g,
        worker_init_fn=worker_init_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Val'de daha büyük batch
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # =========================================================================
    # MODEL
    # =========================================================================
    
    # Num classes
    if args.dataset == 'nih':
        num_classes = 14  # Multi-label
        model_config['multi_label'] = True
    elif args.dataset == 'covidx':
        num_classes = 2  # Binary
    elif args.dataset == 'vinbigdata':
        num_classes = 2  # Binary (örnek)
    else:
        num_classes = 2
    
    print(f"🧠 Model: {args.model}")
    print(f"   Mode: {args.mode}")
    print(f"   Classes: {num_classes}")
    
    model = MedicalImageClassifier(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=True,
        mode=args.mode,
        multi_label=model_config.get('multi_label', False),
    )
    
    print(f"   Trainable params: {model.get_trainable_params():,}")
    print(f"   Total params: {model.get_total_params():,}")
    
    # =========================================================================
    # LOGGER
    # =========================================================================
    
    augmentation_name = args.augmentation if args.augmentation != 'none' else 'baseline'
    
    logger = ExperimentLogger(
        model_name=args.model,
        mode=args.mode,
        augmentation=augmentation_name,
        dataset=args.dataset,
        base_dir=args.run_dir,
        config=model_config,
    )
    
    print(f"📁 Run dir: {logger.run_dir}")
    
    # =========================================================================
    # TRAINER
    # =========================================================================
    
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=model_config,
        logger=logger,
        device=args.device,
        augmentation_name=args.augmentation if args.augmentation != 'none' else None,
    )
    
    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    summary = trainer.fit(
        epochs=model_config.get('epochs', 100),
        save_predictions=args.save_predictions,
    )
    
    print("\n✅ Training completed!")
    print(f"   Best epoch: {summary['best_epoch']}")
    print(f"   Best metric: {summary['best_metric']:.4f}")
    
    return summary


if __name__ == '__main__':
    main()
