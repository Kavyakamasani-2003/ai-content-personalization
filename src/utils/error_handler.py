import traceback
from functools import wraps
from .logger import logger

def error_handler(func):
    """
    Decorator for handling and logging errors in functions
    
    Args:
        func (callable): Function to be decorated
    
    Returns:
        callable: Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Optional: Send error to monitoring service
            # send_error_to_monitoring_service(func.__name__, str(e))
            
            raise
    return wrapper