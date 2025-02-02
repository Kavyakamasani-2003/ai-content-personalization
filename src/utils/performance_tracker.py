# src/utils/performance_tracker.py
import time
import logging
from contextlib import contextmanager
from typing import Dict, List

class PerformanceTracker:
    def __init__(self, log_level=logging.INFO):
        """
        Performance tracking for recommendation system
        
        Args:
            log_level: Logging level for tracking
        """
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.metrics = {
            'total_recommendations': 0,
            'processing_times': []
        }
    
    @contextmanager
    def track_recommendation(self):
        """
        Context manager to track recommendation performance
        """
        start_time = time.time()
        try:
            yield
        finally:
            processing_time = time.time() - start_time
            self.metrics['processing_times'].append(processing_time)
            self.metrics['total_recommendations'] += 1
            
            self.logger.info(f"Recommendation processed in {processing_time:.4f} seconds")
    
    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get performance summary of recommendations
        
        Returns:
            Performance metrics dictionary
        """
        if not self.metrics['processing_times']:
            return {
                'total_recommendations': 0,
                'average_processing_time': 0,
                'max_processing_time': 0,
                'min_processing_time': 0
            }
        
        metrics = {
            'total_recommendations': self.metrics['total_recommendations'],
            'average_processing_time': sum(self.metrics['processing_times']) / len(self.metrics['processing_times']),
            'max_processing_time': max(self.metrics['processing_times']),
            'min_processing_time': min(self.metrics['processing_times'])
        }
        
        # Log performance metrics
        self.logger.info("Performance Summary:")
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value}")
        
        return metrics