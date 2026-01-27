import sys
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.logging import get_logger

# Global logger - will be initialized in main
logger = None


def validate_dataset_integrity(
    root_path: str, 
    extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
) -> List[Tuple[str, str]]:
    """
    Scans a directory for image files and verifies their integrity using PIL.

    This function efficiently retrieves all image files with specified extensions
    and attempts to verify their internal structure without fully loading them
    into memory.

    Args:
        root_path (str): The root directory path to search for images.
        extensions (tuple): A tuple of file extensions to include in the scan.
                            Defaults to ('.png', '.jpg', '.jpeg', '.bmp', '.tiff').

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the path and error message
                               for each corrupted file found. Returns an empty list
                               if all images are valid.
    """
    
    root = Path(root_path)
    corrupted_files = []

    if not root.exists():
        logger.error(f"Path '{root_path}' does not exist.")
        return []
    if not root.is_dir():
        logger.error(f"Path '{root_path}' is not a directory.")
        return []

    logger.info(f"Scanning directory structure: {root_path}...")
    logger.info("This may take a few minutes for large datasets...")

    valid_exts = set(ext.lower() for ext in extensions)
    
    # Ignore macOS metadata files (._*) and other hidden files
    # Use generator with tqdm for progress during scanning
    all_files = []
    for p in tqdm(root.rglob("*"), desc="Scanning files", unit="file"):
        if (p.is_file() 
            and p.suffix.lower() in valid_exts
            and not p.name.startswith("._")
            and not p.name.startswith(".")):
            all_files.append(p)

    total_count = len(all_files)
    if total_count == 0:
        logger.warning(f"No image files found with extensions {extensions}.")
        return []

    logger.info(f"Found {total_count} images. Starting integrity check...")

    for img_path in tqdm(all_files, desc="Verifying images", unit="img"):
        try:
            with Image.open(img_path) as img:
                img.verify() 
        except Exception as e:
            corrupted_files.append((str(img_path), str(e)))

    logger.info("=" * 50)
    if corrupted_files:
        logger.error(f"FAILED. Found {len(corrupted_files)} corrupted files.")
        logger.info("First 5 errors:")
        for path, error in corrupted_files[:5]:
            logger.error(f"  File: {path}")
            logger.error(f"  Error: {error}")
    else:
        logger.info(f"SUCCESS. All {total_count} images are verified and intact.")
    logger.info("=" * 50)

    return corrupted_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check dataset integrity by validating image files."
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the dataset directory to check"
    )
    parser.add_argument(
        "log_name",
        type=str,
        help="Name for the log file (e.g., 'nih_check' creates logs/nih_check.log)"
    )
    
    args = parser.parse_args()
    
    logger = get_logger(args.log_name)
    
    logger.info(f"Starting data check for: {args.data_path}")
    bad_files = validate_dataset_integrity(args.data_path)