"""
Logger configuration using loguru.
"""

import os
import sys
from pathlib import Path

from loguru import logger

# Remove default handler
logger.remove()

# Add console handler with custom format
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
    enqueue=True,  # Thread-safe logging
    backtrace=True,  # Detailed traceback
    diagnose=True,  # Enable exception diagnosis
)

# Add file handler for debugging only if enabled
log_to_file = os.getenv("FORGE_LOG_TO_FILE", "true").lower() == "true"

if log_to_file:
    try:
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)

        logger.add(
            "logs/forge_{time}.log",
            rotation="1 day",  # Create new file daily
            retention="1 week",  # Keep logs for 1 week
            compression="zip",  # Compress rotated logs
            level="DEBUG",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )
    except Exception as e:
        # If file logging fails, continue with console logging only
        logger.warning(f"Failed to initialize file logging: {e}")

# Export logger instance
get_logger = logger.bind
