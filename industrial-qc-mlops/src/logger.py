"""Structured logging with JSON formatting."""

import logging
import sys
from typing import Any, Dict, Optional
from datetime import datetime

import structlog
from structlog.types import Processor


def setup_logging(
    log_level: str = "INFO",
    environment: str = "development",
    service_name: str = "industrial-qc",
) -> None:
    """
    Configure structured logging with JSON output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        environment: Environment name (development, staging, production).
        service_name: Service identifier for log correlation.
    """
    
    # Shared processors for all loggers
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(
            fmt="iso",
            utc=True,
        ),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add environment-specific processors
    if environment == "production":
        # JSON format for production (better for log aggregation)
        shared_processors.extend([
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ])
        formatter_class = structlog.stdlib.ProcessorFormatter
    else:
        # Console format for development (human-readable)
        shared_processors.extend([
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.rich_traceback,
            ),
        ])
        formatter_class = None
    
    # Configure structlog
    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level_int)
    
    if formatter_class:
        # Production: JSON formatter
        processor_formatter = formatter_class(
            processors=shared_processors,
            foreign_pre_chain=[
                structlog.stdlib.ExtraAdder(),
                structlog.stdlib.filter_by_level,
            ],
        )
        handler.setFormatter(processor_formatter)
    
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level_int)
    
    # Reduce noise from third-party libraries
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    

def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__).
        
    Returns:
        Configured structlog BoundLogger instance.
    """
    logger = structlog.get_logger(name)
    return logger  # type: ignore


# Convenience function for logging with context
def log_with_context(
    message: str,
    level: str = "info",
    **kwargs: Any,
) -> None:
    """
    Log a message with additional context.
    
    Args:
        message: Log message.
        level: Log level (debug, info, warning, error, critical).
        **kwargs: Additional context to include in the log.
    """
    logger = get_logger()
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    setup_logging(log_level="DEBUG", environment="development")
    
    logger = get_logger(__name__)
    
    # Basic logging
    logger.info("Application started")
    
    # Logging with context
    logger.debug(
        "Processing image",
        image_id="img_001",
        width=640,
        height=480,
        channels=3,
    )
    
    # Logging errors with exceptions
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.error(
            "An error occurred",
            error_type=type(e).__name__,
            exc_info=True,
        )
    
    # Logging with nested context
    with structlog.contextvars.bound_contextvars(
        request_id="req-123",
        user_id="user-456",
    ):
        logger.info("Processing request")
        logger.debug("Loading model", model_name="yolov8n")
        logger.info("Request completed", duration_ms=45.2)
