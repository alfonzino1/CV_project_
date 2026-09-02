"""
Structured logging configuration using structlog.

Provides JSON-formatted logs with context for production monitoring.
"""

import logging
import sys
from typing import Any, Dict, Optional

import structlog
from structlog.types import Processor


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    service_name: str = "industrial-qc-mlops",
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format (json, text)
        service_name: Service identifier for log correlation
    """
    
    # Map string level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure processors based on format
    if log_format == "json":
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)
    
    # Add service name to all logs
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(service=service_name)


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class LogContext:
    """
    Context manager for adding temporary context to logs.
    
    Usage:
        with LogContext(request_id="123", user="admin"):
            logger.info("Processing request")
    """
    
    def __init__(self, **kwargs: Any):
        self.context = kwargs
    
    def __enter__(self) -> None:
        structlog.contextvars.bind_contextvars(**self.context)
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Remove context keys that were added
        for key in self.context.keys():
            structlog.contextvars.unbind_contextvars(key)


def log_execution_time(func_name: str):
    """
    Decorator to log function execution time.
    
    Usage:
        @log_execution_time("data_loading")
        def load_data():
            ...
    """
    import time
    from functools import wraps
    
    @wraps(func_name)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger(func_name)
        start = time.perf_counter()
        try:
            result = await func_name(*args, **kwargs)
            duration = time.perf_counter() - start
            logger.info(f"{func_name} completed", duration_ms=duration * 1000)
            return result
        except Exception as e:
            duration = time.perf_counter() - start
            logger.error(f"{func_name} failed", error=str(e), duration_ms=duration * 1000)
            raise
    
    @wraps(func_name)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger(func_name)
        start = time.perf_counter()
        try:
            result = func_name(*args, **kwargs)
            duration = time.perf_counter() - start
            logger.info(f"{func_name} completed", duration_ms=duration * 1000)
            return result
        except Exception as e:
            duration = time.perf_counter() - start
            logger.error(f"{func_name} failed", error=str(e), duration_ms=duration * 1000)
            raise
    
    # Detect if function is async
    import inspect
    if inspect.iscoroutinefunction(func_name):
        return async_wrapper
    return sync_wrapper
