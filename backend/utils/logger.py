import logging
import sys
import time
from functools import wraps

def get_logger(name: str = "freshlens") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_logger("freshlens.app")

def timed_execution(action_name: str):
    """Decorator to log execution time of ML operations and endpoints."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000.0
                logger.info(f"{action_name} completed in {duration_ms:.2f} ms")
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000.0
                logger.error(f"{action_name} failed after {duration_ms:.2f} ms with error: {e}")
                raise
        return wrapper
    return decorator
