import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(
    name: str,
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """
    Create and configure a logger.

    Args:
        name: Logger name (also used for log filename)
        log_dir: Directory to store log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to output to console
        file: Whether to output to file

    Returns:
        Configured logger instance
    
    Example:
        logger = get_logger("data_check")
        logger.info("Processing started")
        logger.warning("Missing files detected")
        logger.error("Failed to read image")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # Format: [2026-01-27 15:30:45] [INFO] [data_check] Message
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to filename for unique logs
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # log_file = log_path / f"{name}_{timestamp}.log"
        
        # Or simple: just use name
        log_file = log_path / f"{name}.log"
        
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Log the file location on first use
        logger.debug(f"Logging to: {log_file.absolute()}")

    return logger


def get_logger_with_timestamp(
    name: str,
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create logger with timestamped filename.
    Useful for keeping separate logs for each run.
    
    Example:
        logger = get_logger_with_timestamp("training")
        # Creates: logs/training_20260127_153045.log
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger = logging.getLogger(f"{name}_{timestamp}")
    logger.setLevel(level)
    
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File with timestamp
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"{name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
