"""
Generic Image Processing Converter Entry Point.

This module delegates image dataset conversion to the appropriate backend (e.g., LMDB) based on the type argument.
"""

from typing import Tuple, Optional
from src.utils.logging import get_logger

logger = get_logger("process_image")

def process_image(
    file_path: str,
    img_size: Tuple[int, int],
    quality: int = 95,
    convert_type: str = "lmdb",
    **kwargs
) -> Optional[bytes]:
    """
    Generic image processing function. Delegates to the appropriate converter based on convert_type.
    """
    if convert_type == "lmdb":
        from .lmdb_convert import process_one_image as lmdb_process_one_image
        return lmdb_process_one_image(file_path, img_size, quality=quality)
    else:
        logger.error(f"Unsupported convert_type: {convert_type}")
        return None
