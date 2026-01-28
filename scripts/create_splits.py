import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import get_logger

logger = None


def create_nih_splits(
    metadata_path: str,
    output_dir: Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> bool:
    """
    Create stratified splits for NIH ChestX-ray14.
    
    Split by patient_id to prevent data leakage.
    Stratify by most common finding for balanced distribution.
    Adds 'lmdb_idx' column for LMDB key mapping.
    """
    logger.info("Loading NIH metadata...")
    df = pd.read_parquet(metadata_path)
    
    # Sort by image_index to match LMDB key order
    # LMDB keys are sequential integers based on sorted filenames
    df = df.sort_values('image_index').reset_index(drop=True)
    df['lmdb_idx'] = df.index
    
    # Get unique patients
    patients = df['patient_id'].unique()
    logger.info(f"Total patients: {len(patients)}, Total images: {len(df)}")
    
    # Get primary finding per patient (most frequent) for stratification
    patient_findings = df.groupby('patient_id')['finding_labels'].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index()
    patient_findings.columns = ['patient_id', 'primary_finding']
    
    # Bin rare findings into "Other" for stratification
    finding_counts = patient_findings['primary_finding'].value_counts()
    rare_findings = finding_counts[finding_counts < 50].index
    patient_findings['stratify_label'] = patient_findings['primary_finding'].apply(
        lambda x: 'Other' if x in rare_findings else x
    )
    
    # Split patients
    train_patients, temp_patients = train_test_split(
        patient_findings['patient_id'].values,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=patient_findings['stratify_label'].values,
    )
    
    if test_ratio > 0:
        val_patients, test_patients = train_test_split(
            temp_patients,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=seed,
        )
    else:
        val_patients = temp_patients
        test_patients = np.array([])
    
    # Create split dataframes
    train_df = df[df['patient_id'].isin(train_patients)].copy()
    val_df = df[df['patient_id'].isin(val_patients)].copy()
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    
    logger.info(f"Train: {len(train_df)} images ({len(train_patients)} patients)")
    logger.info(f"Val: {len(val_df)} images ({len(val_patients)} patients)")
    
    # Save splits to dataset-specific directory
    dataset_output_dir = output_dir / "nih"
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_parquet(dataset_output_dir / "train.parquet", index=False)
    val_df.to_parquet(dataset_output_dir / "val.parquet", index=False)
    
    if len(test_patients) > 0:
        test_df = df[df['patient_id'].isin(test_patients)].copy()
        test_df['split'] = 'test'
        test_df.to_parquet(dataset_output_dir / "test.parquet", index=False)
        logger.info(f"Test: {len(test_df)} images ({len(test_patients)} patients)")
    
    # Save combined with split column
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    if len(test_patients) > 0:
        all_df = pd.concat([all_df, test_df], ignore_index=True)
    all_df.to_parquet(dataset_output_dir / "all.parquet", index=False)
    
    logger.info(f"Saved splits to {dataset_output_dir}")
    return True


def create_vinbigdata_splits(
    metadata_path: str,
    output_dir: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> bool:
    """
    Create random splits for VinBigData.
    
    VinBigData doesn't have patient_id in metadata, so we split by image_id.
    Adds 'lmdb_idx' column for LMDB key mapping.
    """
    logger.info("Loading VinBigData metadata...")
    df = pd.read_parquet(metadata_path)
    
    # Filter to train split only (test has no labels)
    if 'split' in df.columns:
        df = df[df['split'] == 'train'].copy()
    
    # Sort by image_id to match LMDB key order
    df = df.sort_values('image_id').reset_index(drop=True)
    df['lmdb_idx'] = df.index
    
    logger.info(f"Total images: {len(df)}")
    
    # Random split
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
    )
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    
    logger.info(f"Train: {len(train_df)} images")
    logger.info(f"Val: {len(val_df)} images")
    
    # Save splits to dataset-specific directory
    dataset_output_dir = output_dir / "vinbigdata"
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_parquet(dataset_output_dir / "train.parquet", index=False)
    val_df.to_parquet(dataset_output_dir / "val.parquet", index=False)
    
    # Combined
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    all_df.to_parquet(dataset_output_dir / "all.parquet", index=False)
    
    logger.info(f"Saved splits to {dataset_output_dir}")
    return True


def create_covidx_splits(
    metadata_dir: str,
    output_dir: Path,
) -> bool:
    """
    COVIDx already has train/val/test splits from the source.
    Just copy them to the splits directory with consistent naming.
    Adds 'lmdb_idx' column for LMDB key mapping.
    """
    logger.info("Loading COVIDx metadata...")
    
    metadata_dir = Path(metadata_dir)
    dataset_output_dir = output_dir / "covidx"
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = ['train', 'val', 'test']
    all_dfs = []
    
    for split in splits:
        parquet_file = metadata_dir / f"covidx_{split}.parquet"
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            # Sort by filename to match LMDB key order
            df = df.sort_values('filename').reset_index(drop=True)
            df['lmdb_idx'] = df.index
            df['split'] = split
            all_dfs.append(df)
            
            # Copy to splits directory
            df.to_parquet(dataset_output_dir / f"{split}.parquet", index=False)
            logger.info(f"{split}: {len(df)} images")
    
    if all_dfs:
        all_df = pd.concat(all_dfs, ignore_index=True)
        all_df.to_parquet(dataset_output_dir / "all.parquet", index=False)
        logger.info(f"Saved splits to {dataset_output_dir}")
        return True
    
    return False


def main():
    global logger
    logger = get_logger("create_splits")
    
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (nih, covidx, vinbigdata) or 'all'",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="Test set ratio (default: 0.0, use existing test set)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/splits",
        help="Path to output splits directory",
    )
    
    args = parser.parse_args()
    
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    
    datasets_to_process = (
        ['nih', 'covidx', 'vinbigdata'] if args.dataset == 'all' 
        else [args.dataset]
    )
    
    results = {}
    
    for dataset in datasets_to_process:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Creating splits for: {dataset}")
        logger.info(f"{'=' * 60}")
        
        try:
            if dataset == 'nih':
                results[dataset] = create_nih_splits(
                    metadata_path=str(processed_dir / "nih" / "nih_labels.parquet"),
                    output_dir=output_dir,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    seed=args.seed,
                )
            elif dataset == 'vinbigdata':
                results[dataset] = create_vinbigdata_splits(
                    metadata_path=str(processed_dir / "vinbigdata" / "vinbigdata_all.parquet"),
                    output_dir=output_dir,
                    val_ratio=args.val_ratio,
                    seed=args.seed,
                )
            elif dataset == 'covidx':
                results[dataset] = create_covidx_splits(
                    metadata_dir=str(processed_dir / "covidx"),
                    output_dir=output_dir,
                )
            else:
                logger.error(f"Unknown dataset: {dataset}")
                results[dataset] = False
        except Exception as e:
            logger.error(f"Failed to create splits for {dataset}: {e}")
            results[dataset] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {dataset}: {status}")


if __name__ == "__main__":
    main()
