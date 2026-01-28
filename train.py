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
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

# =========================================================================
# WARNING FILTERS - Suppress common noisy warnings
# =========================================================================
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Corrupt EXIF data.*")
warnings.filterwarnings("ignore", message=".*Possibly corrupt EXIF data.*")
warnings.filterwarnings("ignore", message=".*Detected call of `lr_scheduler.step()` before `optimizer.step()`.*")
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*A single label was found.*")
warnings.filterwarnings("ignore", message=".*Only one class is present.*")

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
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
    
    # Resume from run folder (auto-loads config and checkpoint)
    parser.add_argument(
        '--retrain', type=str, default=None,
        help='Resume training from run folder name (e.g., vit_small_patch16_linear_probe_baseline_nih_20260128_123456). '
             'Automatically loads config.yaml and last checkpoint.'
    )
    
    # Multi-label
    parser.add_argument('--multi_label', action='store_true', help='Multi-label classification')
    
    # Val ratio for train/val split
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio (default: 0.2)')
    
    return parser.parse_args()


def get_dataset(
    dataset_name: str,
    split: str,
    transform=None,
    multi_label: bool = False,
    val_ratio: float = 0.2,
):
    """
    Dataset yükle.
    
    Split dosyaları val_ratio'ya göre klasörlerde tutulur:
    data/splits/{dataset}/val_{ratio}/train.parquet
    data/splits/{dataset}/val_{ratio}/val.parquet
    
    Eğer split yoksa otomatik oluşturulur.
    """
    # Import dataset sınıfları
    from src.data.dataset import (
        NIHChestXrayDataset,
        COVIDxDataset,
        VinBigDataDataset,
    )
    
    # Dataset paths: LMDB in processed, splits in splits directory
    processed_path = Path('data/processed')
    splits_base = Path('data/splits')
    
    # Val ratio'ya göre klasör adı: val_0.2, val_0.1, etc.
    ratio_folder = f"val_{val_ratio}"
    splits_path = splits_base / dataset_name / ratio_folder
    
    # Split dosyası var mı kontrol et
    metadata_path = splits_path / f'{split}.parquet'
    if not metadata_path.exists():
        print(f"⚙️  Split bulunamadı, oluşturuluyor: {splits_path}")
        _create_split_for_dataset(dataset_name, processed_path, splits_path, val_ratio)
    
    if dataset_name == 'nih':
        lmdb_path = processed_path / 'nih' / 'nih.lmdb'
        return NIHChestXrayDataset(
            lmdb_path=str(lmdb_path),
            metadata_path=str(metadata_path),
            transform=transform,
        )
    
    elif dataset_name == 'covidx':
        lmdb_path = processed_path / 'covidx' / 'covidx.lmdb'
        return COVIDxDataset(
            lmdb_path=str(lmdb_path),
            metadata_path=str(metadata_path),
            transform=transform,
        )
    
    elif dataset_name == 'vinbigdata':
        lmdb_path = processed_path / 'vinbigdata' / 'vinbigdata.lmdb'
        return VinBigDataDataset(
            lmdb_path=str(lmdb_path),
            metadata_path=str(metadata_path),
            transform=transform,
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _create_split_for_dataset(
    dataset_name: str,
    processed_path: Path,
    output_path: Path,
    val_ratio: float,
    seed: int = 42,
):
    """
    Dataset için train/val split oluştur.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    if dataset_name == 'vinbigdata':
        metadata_path = processed_path / 'vinbigdata' / 'vinbigdata_all.parquet'
        df = pd.read_parquet(metadata_path)
        
        # Filter to train split only (test has no labels)
        if 'split' in df.columns:
            df = df[df['split'] == 'train'].copy()
        
        df = df.sort_values('image_id').reset_index(drop=True)
        df['lmdb_idx'] = df.index
        
    elif dataset_name == 'nih':
        metadata_path = processed_path / 'nih' / 'nih_labels.parquet'
        df = pd.read_parquet(metadata_path)
        df = df.sort_values('image_index').reset_index(drop=True)
        df['lmdb_idx'] = df.index
        
    elif dataset_name == 'covidx':
        # COVIDx has its own splits, just copy train/val
        train_src = processed_path / 'covidx' / 'covidx_train.parquet'
        val_src = processed_path / 'covidx' / 'covidx_val.parquet'
        
        if train_src.exists() and val_src.exists():
            train_df = pd.read_parquet(train_src)
            val_df = pd.read_parquet(val_src)
            train_df.to_parquet(output_path / 'train.parquet', index=False)
            val_df.to_parquet(output_path / 'val.parquet', index=False)
            print(f"   ✅ COVIDx: Train {len(train_df)}, Val {len(val_df)}")
            return
        else:
            raise FileNotFoundError(f"COVIDx split dosyaları bulunamadı: {train_src}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Train/Val split
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
    )
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    
    train_df.to_parquet(output_path / 'train.parquet', index=False)
    val_df.to_parquet(output_path / 'val.parquet', index=False)
    
    print(f"   ✅ {dataset_name}: Train {len(train_df)}, Val {len(val_df)}")


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


def _setup_run_logger(run_dir: Path, mode: str = 'w') -> logging.Logger:
    """
    Setup logger that outputs to both console and run-specific log file.
    
    Args:
        run_dir: Path to run directory
        mode: File open mode ('w' for new run, 'a' for resume)
        
    Returns:
        Configured logger instance
    """
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    
    # Clear existing handlers
    log.handlers.clear()
    
    # Format
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
    
    # File handler - logs to run directory
    log_file = Path(run_dir) / "train.log"
    file_handler = logging.FileHandler(log_file, mode=mode, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    
    return log


def main():
    """Ana training fonksiyonu."""
    args = parse_args()
    
    # =========================================================================
    # RETRAIN MODE - Load config from previous run
    # =========================================================================
    
    retrain_run_dir = None
    if args.retrain:
        # Run klasörü var mı kontrol et
        retrain_run_dir = Path(args.run_dir) / args.retrain
        if not retrain_run_dir.exists():
            print(f"Error: Run folder not found: {retrain_run_dir}")
            sys.exit(1)
        
        # Config dosyasını yükle
        config_path = retrain_run_dir / 'config.yaml'
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        
        import yaml
        with open(config_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        # Args'ı saved config ile override et
        args.model = saved_config.get('model_name', args.model)
        args.mode = saved_config.get('mode', args.mode)
        args.dataset = saved_config.get('dataset', args.dataset)
        args.augmentation = saved_config.get('augmentation', 'none')
        args.multi_label = saved_config.get('multi_label', False)
        args.val_ratio = saved_config.get('val_ratio', 0.2)
        
        # Hyperparameters
        if args.epochs is None:
            args.epochs = saved_config.get('epochs')
        if args.batch_size is None:
            args.batch_size = saved_config.get('batch_size')
        if args.lr is None:
            args.lr = saved_config.get('lr')
        
        # Last checkpoint'i resume olarak ayarla
        last_checkpoint = retrain_run_dir / 'checkpoints' / 'last.pth'
        if last_checkpoint.exists():
            args.resume = str(last_checkpoint)
            print(f"Resuming from: {args.resume}")
        else:
            print(f"Warning: No last checkpoint found in {retrain_run_dir}")
        
        print(f"Loaded config from: {config_path}")
        print(f"  Model: {args.model}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Mode: {args.mode}")
    
    # =========================================================================
    # SETUP
    # =========================================================================
    
    # Seed
    set_seed(args.seed)
    
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
    
    # Transforms
    train_transform = get_transforms(input_size, is_train=True)
    val_transform = get_transforms(input_size, is_train=False)
    
    # Datasets
    train_dataset = get_dataset(args.dataset, 'train', train_transform, args.multi_label, args.val_ratio)
    val_dataset = get_dataset(args.dataset, 'val', val_transform, args.multi_label, args.val_ratio)
    
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
    
    model = MedicalImageClassifier(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=True,
        mode=args.mode,
        multi_label=model_config.get('multi_label', False),
    )
    
    # =========================================================================
    # LOGGER
    # =========================================================================
    
    augmentation_name = args.augmentation if args.augmentation != 'none' else 'baseline'
    
    # Retrain modunda aynı run klasörünü kullan
    if retrain_run_dir:
        # Run name'den timestamp'i çıkar
        run_name = args.retrain
        timestamp = run_name.split('_')[-2] + '_' + run_name.split('_')[-1]
        
        logger = ExperimentLogger(
            model_name=args.model,
            mode=args.mode,
            augmentation=augmentation_name,
            dataset=args.dataset,
            base_dir=args.run_dir,
            config=model_config,
            timestamp=timestamp,  # Aynı timestamp kullanarak aynı klasöre yaz
        )
    else:
        logger = ExperimentLogger(
            model_name=args.model,
            mode=args.mode,
            augmentation=augmentation_name,
            dataset=args.dataset,
            base_dir=args.run_dir,
            config=model_config,
        )
    
    # Setup Python logging to run directory
    log_mode = 'a' if args.retrain else 'w'
    log = _setup_run_logger(logger.run_dir, mode=log_mode)
    
    # Log hyperparameters
    log.info(f"Config: {model_config}")
    log.info(f"Augmentation: {augmentation_name}")
    log.info(f"Dataset: {args.dataset}")
    log.info(f"Run Directory: {logger.run_dir}")
    if args.retrain:
        log.info(f"Retraining from: {args.retrain}")
        log.info(f"Log mode: append")
    log.info(f"Dataset: {args.dataset}")
    log.info(f"Val ratio: {args.val_ratio}")
    log.info(f"Input size: {input_size}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Train samples: {len(train_dataset)}")
    log.info(f"Val samples: {len(val_dataset)}")
    log.info(f"Model: {args.model}")
    log.info(f"Mode: {args.mode}")
    log.info(f"Classes: {num_classes}")
    log.info(f"Trainable params: {model.get_trainable_params():,}")
    log.info(f"Total params: {model.get_total_params():,}")
    log.info(f"Run dir: {logger.run_dir}")
    log.info(f"{'=' * 60}")
    
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
    
    log.info(f"{'=' * 60}")
    log.info(f"Training completed!")
    log.info(f"Best epoch: {summary['best_epoch']}")
    log.info(f"Best metric: {summary['best_metric']:.4f}")
    log.info(f"{'=' * 60}")
    
    return summary


if __name__ == '__main__':
    main()
