"""Logging configuration integration with YAML configs.

This module provides integration between the YAML configuration system
and the logging setup.
"""

import logging
from typing import Optional

from ..config.models import Config, LoggingConfig
from .logging import LogLevel, setup_logging


def setup_logging_from_config(config: Config, verbose: bool = False) -> logging.Logger:
    """Setup logging based on configuration object.
    
    Args:
        config: Main configuration object that may contain logging settings
        verbose: If True, override config to use DEBUG level
        
    Returns:
        Configured root logger
    """
    # Default logging configuration
    log_config = LoggingConfig()
    
    # Use config if available
    if config.logging:
        log_config = config.logging
    
    # Override with verbose flag
    level = LogLevel.DEBUG if verbose else log_config.level
    
    # Setup logging with configuration
    logger = setup_logging(
        level=level,
        log_file=log_config.log_file,
        console=log_config.console_output,
        format_string=log_config.format,
        max_file_size=log_config.max_bytes,
        backup_count=log_config.backup_count,
        colored=True,  # Always use colored output when available
        logger_configs={
            # Set specific logger levels
            "rag_client": level,
            "llama_index": "WARNING",
            "httpx": "WARNING",
            "transformers": "WARNING",
        }
    )
    
    return logger


def configure_application_logging(
    config_path: Optional[str] = None,
    verbose: bool = False,
    log_file: Optional[str] = None,
    log_level: Optional[str] = None
) -> logging.Logger:
    """Configure application logging with command-line overrides.
    
    Args:
        config_path: Path to YAML configuration file
        verbose: Enable verbose (DEBUG) logging
        log_file: Override log file from config
        log_level: Override log level from config
        
    Returns:
        Configured root logger
    """
    # Default config
    log_config = LoggingConfig()
    
    # Load from YAML if available
    if config_path:
        try:
            from pathlib import Path

            from ..config.loader import load_config
            config = load_config(Path(config_path))
            if config.logging:
                log_config = config.logging
        except Exception:
            # Fall back to defaults if config loading fails
            pass
    
    # Apply command-line overrides
    if verbose:
        log_config.level = "DEBUG"
    elif log_level:
        log_config.level = log_level.upper()
    
    if log_file:
        log_config.log_file = log_file
    
    # Setup logging
    return setup_logging(
        level=log_config.level,
        log_file=log_config.log_file,
        console=log_config.console_output,
        format_string=log_config.format,
        max_file_size=log_config.max_bytes,
        backup_count=log_config.backup_count,
        colored=True
    )