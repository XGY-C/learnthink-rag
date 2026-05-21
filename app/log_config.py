"""Professional logging configuration for learnthink-rag.

Features:
- Console output with colorized levels (via standard formatting)
- File output with daily rotation (TimedRotatingFileHandler)
- Structured format including filename and line number for debugging
- Configurable log directory and retention period
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


def setup_logger(
    name: str = "learnthink-rag",
    log_dir: Path = Path("logs"),
    level: int = logging.DEBUG,
    console_level: int = logging.INFO,
    backup_count: int = 7,
) -> logging.Logger:
    """Setup a professional logger with console and file handlers.

    Args:
        name: Logger name (usually __name__ or app name)
        log_dir: Directory to store log files
        level: Global logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_level: Minimum level for console output
        backup_count: Number of days to keep log files

    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # --- Formatters ---
    # Detailed format for files (includes source location)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Simpler format for console (easier to read in terminal)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # --- File Handler (with rotation) ---
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "rag_service.log"
        
        file_handler = TimedRotatingFileHandler(
            log_file,
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8",
            delay=True  # 添加这一行，直到真正需要写入时才打开文件
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_format)
        file_handler.suffix = "%Y-%m-%d.log"
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to setup file logging: {e}. Logs will only go to console.")

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    return logger


# Initialize default logger for the application
default_logger = setup_logger()
