import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir='logs', log_level=logging.INFO):
    """
    Configure comprehensive logging for the recommendation system
    
    Args:
        log_dir (str): Directory to store log files
        log_level (int): Logging level
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    
    # File Handler (Rotating)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'recommendation_system.log'),
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger