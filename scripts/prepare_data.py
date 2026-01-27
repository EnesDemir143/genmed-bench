import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_convert import create_lmdb_fast
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


def prepare_dataset(
    dataset_name: str,
    config: dict,
    image_size: int = 224,
    quality: int = 95,
    n_jobs: int = -1,
) -> bool:
    """
    Prepare a single dataset by converting to LMDB.

    Args:
        dataset_name: Name of the dataset (nih, chexpert, covidx, vinbigdata).
        config: Configuration dictionary.
        image_size: Target image size.
        quality: JPEG compression quality.
        n_jobs: Number of parallel workers.

    Returns:
        True if successful, False otherwise.
    """
    datasets_config = config["data"]["datasets"]

    if dataset_name not in datasets_config:
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.info(f"Available datasets: {list(datasets_config.keys())}")
        return False

    dataset_info = datasets_config[dataset_name]
    raw_path = resolve_path(dataset_info["raw"], config)
    processed_path = resolve_path(dataset_info["processed"], config)

    # Check if raw data exists
    if not Path(raw_path).exists():
        logger.error(f"Raw data not found: {raw_path}")
        return False

    # Create output directory if needed
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'=' * 60}")
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"  Raw path: {raw_path}")
    logger.info(f"  Output path: {processed_path}")
    logger.info(f"  Image size: {image_size}x{image_size}")
    logger.info(f"  Quality: {quality}")
    logger.info(f"{'=' * 60}")

    try:
        create_lmdb_fast(
            data_path=raw_path,
            lmdb_path=processed_path,
            img_size=(image_size, image_size),
            quality=quality,
            n_jobs=n_jobs,
        )
        logger.info(f"Successfully created LMDB for {dataset_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to create LMDB for {dataset_name}: {e}")
        return False


def main():
    
    global logger
    logger = get_logger("prepare_data")
    
    parser = argparse.ArgumentParser(
        description="Convert image datasets to LMDB format"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (nih, chexpert, covidx, vinbigdata) or 'all'",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Target image size (default: 224)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG compression quality (default: 95)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers (-1 for all cores)",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.dataset == "all":
        datasets = list(config["data"]["datasets"].keys())
        logger.info(f"Processing all datasets: {datasets}")

        results = {}
        for dataset in datasets:
            results[dataset] = prepare_dataset(
                dataset,
                config,
                image_size=args.image_size,
                quality=args.quality,
                n_jobs=args.n_jobs,
            )

        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        for dataset, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"  {dataset}: {status}")
    else:
        success = prepare_dataset(
            args.dataset,
            config,
            image_size=args.image_size,
            quality=args.quality,
            n_jobs=args.n_jobs,
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
