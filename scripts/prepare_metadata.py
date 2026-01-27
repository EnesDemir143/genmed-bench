import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import get_logger

logger = None


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(path_template: str, config: dict) -> str:
    """Resolve path templates like ${data.raw_root}."""
    if "${data.raw_root}" in path_template:
        return path_template.replace("${data.raw_root}", config["data"]["raw_root"])
    if "${data.processed_root}" in path_template:
        return path_template.replace(
            "${data.processed_root}", config["data"]["processed_root"]
        )
    return path_template


def process_nih(raw_path: str, output_dir: Path) -> bool:
    """
    Process NIH ChestX-ray14 metadata.
    
    Files:
        - Data_Entry_2017.csv: Main labels file
        - BBox_List_2017.csv: Bounding box annotations
    """
    raw_path = Path(raw_path)
    
    # Process main labels
    labels_file = raw_path / "Data_Entry_2017.csv"
    if labels_file.exists():
        logger.info(f"Processing {labels_file}")
        df = pd.read_csv(labels_file)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
        
        # Parse multi-label findings
        df["finding_list"] = df["finding_labels"].str.split("|")
        
        output_file = output_dir / "nih_labels.parquet"
        df.to_parquet(output_file, index=False, compression="snappy")
        logger.info(f"Saved {len(df)} rows to {output_file}")
    else:
        logger.warning(f"Labels file not found: {labels_file}")
        return False
    
    # Process bounding boxes
    bbox_file = raw_path / "BBox_List_2017.csv"
    if bbox_file.exists():
        logger.info(f"Processing {bbox_file}")
        df_bbox = pd.read_csv(bbox_file)
        df_bbox.columns = df_bbox.columns.str.strip().str.replace(" ", "_").str.lower()
        
        output_file = output_dir / "nih_bbox.parquet"
        df_bbox.to_parquet(output_file, index=False, compression="snappy")
        logger.info(f"Saved {len(df_bbox)} rows to {output_file}")
    
    return True


def process_covidx(raw_path: str, output_dir: Path) -> bool:
    """
    Process COVIDx metadata.
    
    Files:
        - train.txt, val.txt, test.txt: Space-separated files
        Format: patient_id filename label source
    """
    raw_path = Path(raw_path)
    
    splits = ["train", "val", "test"]
    all_dfs = []
    
    for split in splits:
        txt_file = raw_path / f"{split}.txt"
        if txt_file.exists():
            logger.info(f"Processing {txt_file}")
            
            # Read space-separated file
            df = pd.read_csv(
                txt_file, 
                sep=" ", 
                header=None,
                names=["patient_id", "filename", "label", "source"]
            )
            df["split"] = split
            all_dfs.append(df)
            
            # Save individual split
            output_file = output_dir / f"covidx_{split}.parquet"
            df.to_parquet(output_file, index=False, compression="snappy")
            logger.info(f"Saved {len(df)} rows to {output_file}")
        else:
            logger.warning(f"File not found: {txt_file}")
    
    # Combine all splits
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        output_file = output_dir / "covidx_all.parquet"
        df_all.to_parquet(output_file, index=False, compression="snappy")
        logger.info(f"Saved combined {len(df_all)} rows to {output_file}")
        return True
    
    return False


def process_vinbigdata(raw_path: str, output_dir: Path) -> bool:
    """
    Process VinBigData metadata.
    
    Files:
        - train_meta.csv, test_meta.csv: Image metadata
    """
    raw_path = Path(raw_path)
    
    splits = ["train", "test"]
    all_dfs = []
    
    for split in splits:
        csv_file = raw_path / f"{split}_meta.csv"
        if csv_file.exists():
            logger.info(f"Processing {csv_file}")
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
            df["split"] = split
            all_dfs.append(df)
            
            # Save individual split
            output_file = output_dir / f"vinbigdata_{split}.parquet"
            df.to_parquet(output_file, index=False, compression="snappy")
            logger.info(f"Saved {len(df)} rows to {output_file}")
        else:
            logger.warning(f"File not found: {csv_file}")
    
    # Combine all splits
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        output_file = output_dir / "vinbigdata_all.parquet"
        df_all.to_parquet(output_file, index=False, compression="snappy")
        logger.info(f"Saved combined {len(df_all)} rows to {output_file}")
        return True
    
    return False


# Dataset processors mapping
PROCESSORS = {
    "nih": process_nih,
    "covidx": process_covidx,
    "vinbigdata": process_vinbigdata,
}


def prepare_metadata(
    dataset_name: str,
    config: dict,
) -> bool:
    """
    Prepare metadata for a single dataset.

    Args:
        dataset_name: Name of the dataset (nih, covidx, vinbigdata).
        config: Configuration dictionary.

    Returns:
        True if successful, False otherwise.
    """
    datasets_config = config["data"]["datasets"]

    if dataset_name not in datasets_config:
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.info(f"Available datasets: {list(datasets_config.keys())}")
        return False

    if dataset_name not in PROCESSORS:
        logger.error(f"No processor defined for: {dataset_name}")
        return False

    dataset_info = datasets_config[dataset_name]
    raw_path = resolve_path(dataset_info["raw"], config)
    
    # Output to processed folder
    output_dir = Path(resolve_path(config["data"]["processed_root"], config)) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if raw data exists
    if not Path(raw_path).exists():
        logger.error(f"Raw data not found: {raw_path}")
        return False

    logger.info(f"{'=' * 60}")
    logger.info(f"Processing metadata: {dataset_name}")
    logger.info(f"  Raw path: {raw_path}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"{'=' * 60}")

    try:
        processor = PROCESSORS[dataset_name]
        success = processor(raw_path, output_dir)
        
        if success:
            logger.info(f"Successfully processed metadata for {dataset_name}")
        return success
    except Exception as e:
        logger.error(f"Failed to process metadata for {dataset_name}: {e}")
        return False


def main():
    global logger
    logger = get_logger("prepare_metadata")
    
    parser = argparse.ArgumentParser(
        description="Convert CSV/TXT metadata files to Parquet format"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (nih, covidx, vinbigdata) or 'all'",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    if args.dataset == "all":
        datasets = list(PROCESSORS.keys())
        logger.info(f"Processing all datasets: {datasets}")

        results = {}
        for dataset in datasets:
            results[dataset] = prepare_metadata(dataset, config)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        for dataset, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"  {dataset}: {status}")
    else:
        success = prepare_metadata(args.dataset, config)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()