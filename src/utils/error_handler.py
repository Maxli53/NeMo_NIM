#!/usr/bin/env python3
"""
Centralized error handling for the project
Following CLAUDE.md best practices
"""

import logging
import traceback
from typing import Optional, Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)


class MoEException(Exception):
    """Base exception for all MoE-related errors"""
    pass


class ConfigurationError(MoEException):
    """Configuration-related errors"""
    pass


class ModelLoadError(MoEException):
    """Model loading errors"""
    pass


class OptimizationError(MoEException):
    """Optimization-related errors"""
    pass


class PerformanceError(MoEException):
    """Performance degradation errors"""
    pass


def safe_execute(
    func: Callable,
    fallback_value: Any = None,
    raise_on_error: bool = False,
    error_message: Optional[str] = None
):
    """
    Safely execute a function with proper error handling

    Args:
        func: Function to execute
        fallback_value: Value to return on error
        raise_on_error: Whether to re-raise the exception
        error_message: Custom error message

    Returns:
        Function result or fallback_value on error
    """
    try:
        return func()
    except Exception as e:
        # Log the error with context
        logger.error(
            error_message or f"Error in {func.__name__}",
            extra={
                "function": getattr(func, "__name__", str(func)),
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

        if raise_on_error:
            raise

        return fallback_value


def error_handler(
    fallback_value: Any = None,
    raise_on_error: bool = True,
    log_level: str = "ERROR"
):
    """
    Decorator for standardized error handling

    Args:
        fallback_value: Value to return on error
        raise_on_error: Whether to re-raise the exception
        log_level: Logging level for errors

    Usage:
        @error_handler(fallback_value=None, raise_on_error=False)
        def my_function():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error with full context
                log_func = getattr(logger, log_level.lower(), logger.error)
                log_func(
                    f"Error in {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "module": func.__module__,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "args": str(args)[:100],  # Truncate for safety
                        "kwargs": str(kwargs)[:100],
                        "traceback": traceback.format_exc()
                    }
                )

                # Handle specific error types
                if isinstance(e, PerformanceError):
                    logger.critical("Performance degradation detected! Consider rollback.")

                if isinstance(e, ConfigurationError):
                    logger.error("Configuration error - check configs/ directory")

                if raise_on_error:
                    raise

                return fallback_value

        return wrapper
    return decorator


def validate_input(
    param_name: str,
    param_value: Any,
    expected_type: type,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allowed_values: Optional[list] = None
):
    """
    Validate input parameters

    Args:
        param_name: Name of the parameter
        param_value: Value to validate
        expected_type: Expected type
        min_value: Minimum value (for numeric types)
        max_value: Maximum value (for numeric types)
        allowed_values: List of allowed values

    Raises:
        ValueError: If validation fails
    """
    # Type check
    if not isinstance(param_value, expected_type):
        raise ValueError(
            f"{param_name} must be {expected_type.__name__}, "
            f"got {type(param_value).__name__}"
        )

    # Range check for numeric types
    if isinstance(param_value, (int, float)):
        if min_value is not None and param_value < min_value:
            raise ValueError(f"{param_name} must be >= {min_value}, got {param_value}")
        if max_value is not None and param_value > max_value:
            raise ValueError(f"{param_name} must be <= {max_value}, got {param_value}")

    # Allowed values check
    if allowed_values is not None and param_value not in allowed_values:
        raise ValueError(
            f"{param_name} must be one of {allowed_values}, got {param_value}"
        )


class ErrorContext:
    """
    Context manager for error handling with cleanup

    Usage:
        with ErrorContext("Loading model", cleanup_func=cleanup):
            # Code that might fail
    """

    def __init__(
        self,
        operation: str,
        cleanup_func: Optional[Callable] = None,
        raise_on_error: bool = True
    ):
        self.operation = operation
        self.cleanup_func = cleanup_func
        self.raise_on_error = raise_on_error

    def __enter__(self):
        logger.debug(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(
                f"Error during {self.operation}",
                extra={
                    "error": str(exc_val),
                    "type": exc_type.__name__,
                    "traceback": traceback.format_exc()
                }
            )

            # Run cleanup if provided
            if self.cleanup_func:
                try:
                    logger.info(f"Running cleanup for {self.operation}")
                    self.cleanup_func()
                except Exception as cleanup_error:
                    logger.error(
                        f"Cleanup failed for {self.operation}: {cleanup_error}"
                    )

            # Don't suppress the exception if raise_on_error is True
            return not self.raise_on_error
        else:
            logger.debug(f"Completed: {self.operation}")
            return True


# Example usage patterns
"""
# 1. Using decorator
@error_handler(fallback_value=None, raise_on_error=False)
def load_model(path):
    # Model loading code
    return model

# 2. Using context manager
with ErrorContext("Model inference", cleanup_func=torch.cuda.empty_cache):
    output = model.generate(input_ids)

# 3. Using safe_execute
result = safe_execute(
    lambda: risky_operation(),
    fallback_value=default_value,
    error_message="Failed to perform risky operation"
)

# 4. Input validation
validate_input("temperature", temp, float, min_value=0.0, max_value=2.0)
"""