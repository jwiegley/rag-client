"""Logging configuration module for RAG client.

This module provides centralized logging configuration with support for
console and file handlers, customizable formats, and different log levels.
"""

import logging
import logging.handlers
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class LogLevel(str, Enum):
    """Enumeration of available log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Default logging format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Detailed format for debugging
DETAILED_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "[%(filename)s:%(lineno)d - %(funcName)s()] - %(message)s"
)

# Simple format for production
SIMPLE_FORMAT = "%(levelname)s: %(message)s"

# JSON-compatible format for structured logging
JSON_FORMAT = (
    '{"timestamp":"%(asctime)s","logger":"%(name)s","level":"%(levelname)s",'
    '"file":"%(filename)s","line":%(lineno)d,"function":"%(funcName)s",'
    '"message":"%(message)s"}'
)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors for console output."""
        # Only add colors if outputting to a terminal
        if sys.stderr.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logging(
    level: str | LogLevel = LogLevel.INFO,
    log_file: Optional[str | Path] = None,
    console: bool = True,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    colored: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    logger_configs: Optional[Dict[str, str]] = None,
    propagate: bool = False,
    clear_handlers: bool = True
) -> logging.Logger:
    """Configure application-wide logging.
    
    Args:
        level: Default logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file for file handler
        console: Whether to enable console (stderr) output
        format_string: Custom format string for log messages
        date_format: Custom date format string
        colored: Whether to use colored output for console
        max_file_size: Maximum size for log file before rotation (bytes)
        backup_count: Number of rotated log files to keep
        logger_configs: Dictionary of logger names to their log levels
        propagate: Whether loggers should propagate to parent
        clear_handlers: Whether to clear existing handlers before setup
    
    Returns:
        Configured root logger instance
    
    Examples:
        >>> # Basic setup with INFO level
        >>> logger = setup_logging(level="INFO")
        
        >>> # Setup with file logging and custom format
        >>> logger = setup_logging(
        ...     level="DEBUG",
        ...     log_file="app.log",
        ...     format_string=DETAILED_FORMAT
        ... )
        
        >>> # Setup with specific logger configurations
        >>> logger = setup_logging(
        ...     level="INFO",
        ...     logger_configs={
        ...         "rag_client.api": "DEBUG",
        ...         "llama_index": "WARNING"
        ...     }
        ... )
    """
    # Convert string level to LogLevel enum if needed
    if isinstance(level, str):
        level = LogLevel(level.upper())
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers if requested
    if clear_handlers:
        root_logger.handlers.clear()
    
    # Set root logger level
    root_logger.setLevel(level.value)
    
    # Use provided format or default based on level
    if format_string is None:
        if level == LogLevel.DEBUG:
            format_string = DETAILED_FORMAT
        else:
            format_string = DEFAULT_FORMAT
    
    if date_format is None:
        date_format = DEFAULT_DATE_FORMAT
    
    # Configure console handler
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level.value)
        
        # Use colored formatter for console if requested
        if colored and sys.stderr.isatty():
            console_formatter = ColoredFormatter(format_string, date_format)
        else:
            console_formatter = logging.Formatter(format_string, date_format)
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Configure file handler with rotation
    if log_file:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level.value)
        
        # Always use non-colored formatter for files
        file_formatter = logging.Formatter(format_string, date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    if logger_configs:
        for logger_name, logger_level in logger_configs.items():
            specific_logger = logging.getLogger(logger_name)
            specific_logger.setLevel(logger_level.upper())
            specific_logger.propagate = propagate
    
    # Configure common third-party loggers to reduce noise
    noisy_loggers = [
        "httpx",
        "httpcore",
        "urllib3",
        "asyncio",
        "filelock",
        "huggingface_hub",
        "transformers",
        "sentence_transformers",
        "torch",
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Log initial setup message
    root_logger.info(
        f"Logging configured: level={level.value}, "
        f"console={console}, file={log_file or 'None'}"
    )
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.
    
    Args:
        name: Name of the logger (typically __name__ of the module)
    
    Returns:
        Logger instance for the specified name
    
    Examples:
        >>> # In a module file
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)


def configure_logger(
    name: str,
    level: str | LogLevel | None = None,
    format_string: Optional[str] = None,
    propagate: bool = True
) -> logging.Logger:
    """Configure a specific logger with custom settings.
    
    Args:
        name: Name of the logger to configure
        level: Optional log level for this logger
        format_string: Optional custom format string
        propagate: Whether to propagate to parent logger
    
    Returns:
        Configured logger instance
    
    Examples:
        >>> # Configure a specific module logger
        >>> api_logger = configure_logger(
        ...     "rag_client.api",
        ...     level="DEBUG",
        ...     propagate=False
        ... )
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        if isinstance(level, str):
            level = LogLevel(level.upper())
        logger.setLevel(level.value)
    
    logger.propagate = propagate
    
    # Add custom handler if format is specified
    if format_string and not propagate:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(format_string))
        logger.handlers.clear()
        logger.addHandler(handler)
    
    return logger


def log_exception(
    logger: logging.Logger,
    message: str,
    exc: Exception,
    level: str | LogLevel = LogLevel.ERROR,
    include_traceback: bool = True
) -> None:
    """Log an exception with context.
    
    Args:
        logger: Logger instance to use
        message: Context message for the exception
        exc: Exception instance to log
        level: Log level to use
        include_traceback: Whether to include full traceback
    
    Examples:
        >>> try:
        ...     risky_operation()
        >>> except Exception as e:
        ...     log_exception(logger, "Operation failed", e)
    """
    if isinstance(level, str):
        level = LogLevel(level.upper())
    
    if include_traceback:
        logger.log(
            getattr(logging, level.value),
            f"{message}: {type(exc).__name__}: {str(exc)}",
            exc_info=True
        )
    else:
        logger.log(
            getattr(logging, level.value),
            f"{message}: {type(exc).__name__}: {str(exc)}"
        )


class LogContext:
    """Context manager for temporary log level changes.
    
    Examples:
        >>> logger = get_logger(__name__)
        >>> with LogContext(logger, "DEBUG"):
        ...     logger.debug("This will be shown")
        >>> logger.debug("This might not be shown")
    """
    
    def __init__(self, logger: logging.Logger, level: str | LogLevel):
        """Initialize log context.
        
        Args:
            logger: Logger to modify
            level: Temporary log level
        """
        self.logger = logger
        self.original_level = logger.level
        if isinstance(level, str):
            level = LogLevel(level.upper())
        self.temp_level = getattr(logging, level.value)
    
    def __enter__(self):
        """Enter context and set temporary level."""
        self.logger.setLevel(self.temp_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original level."""
        self.logger.setLevel(self.original_level)
        return False


# Module-level logger for this module
logger = get_logger(__name__)

# Re-export commonly used items
__all__ = [
    'setup_logging',
    'get_logger',
    'configure_logger',
    'log_exception',
    'LogContext',
    'LogLevel',
    'DEFAULT_FORMAT',
    'DETAILED_FORMAT',
    'SIMPLE_FORMAT',
    'JSON_FORMAT',
    'ColoredFormatter',
]