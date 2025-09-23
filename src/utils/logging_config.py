#!/usr/bin/env python3
"""
Centralized logging configuration for the entire project
"""

import logging
import os
from pathlib import Path

def setup_logging():
    """Configure logging for the entire application"""

    # Get log level from environment
    log_level = os.getenv('LOG_LEVEL', 'INFO')

    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configure handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_dir / 'app.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=handlers
    )

    # Disable noisy loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    return logging.getLogger(__name__)

# Initialize logging when module is imported
logger = setup_logging()
