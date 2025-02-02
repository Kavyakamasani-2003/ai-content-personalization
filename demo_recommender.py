#!/usr/bin/env python3
"""
AI Content Personalization Demo Script

This script demonstrates the capabilities of the AI Content Recommender system,
showcasing feature extraction, recommendation generation, and performance tracking.
"""

import logging
from typing import List, Dict
from src.ml_predictors.advanced_recommender import AIContentRecommender
from src.utils.performance_tracker import PerformanceTracker

def setup_logging() -> logging.Logger:
    """
    Configure logging for the demo script.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
    )
    return logging.getLogger(__name__)

def prepare_content_library() -> List[str]:
    """
    Prepare a diverse content library for demonstration.
    
    Returns:
        List[str]: A list of content items
    """
    return [
        "Machine learning is transforming industries with advanced algorithms",
        "Artificial Intelligence enables predictive analytics and smart decision making",
        "Data science combines statistics, computer science, and domain expertise",
        "Deep learning neural networks are revolutionizing image and speech recognition",
        "Natural language processing helps machines understand human communication",
        "Recommender systems personalize content based on user preferences and behavior",
        "Big data analytics provides insights from large and complex datasets",
        "Cloud computing offers scalable and flexible computational resources",
        "Cybersecurity protects digital systems from potential threats and vulnerabilities",
        "Quantum computing promises exponential computational power for complex problems"
    ]

def demonstrate_recommendation_system(logger: logging.Logger) -> None:
    """
    Demonstrate the AI Content Recommender system's capabilities.
    
    Args:
        logger (logging.Logger): Logging instance for output
    """
    # Initialize Performance Tracker
    performance_tracker = PerformanceTracker()
    
    # Create AI Content Recommender
    recommender = AIContentRecommender()
    
    # Prepare content library
    content_library = prepare_content_library()
    
    # Add content to recommender
    logger.info("🚀 Adding content to recommender...")
    recommender.add_content(content_library)
    
    # Demonstration queries
    demo_queries = [
        "advanced technology trends",
        "machine learning applications",
        "artificial intelligence insights",
        "data science innovations"
    ]
    
    # Generate recommendations with performance tracking
    for query in demo_queries:
        logger.info(f"\n🔍 Generating recommendations for query: '{query}'")
        
        with performance_tracker.track_recommendation():
            recommendations = recommender.recommend(
                query, 
                top_k=3, 
                personalization_weight=0.7
            )
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"Recommendation {i}:")
            logger.info(f"  Content: {rec['content']}")
            logger.info(f"  Similarity Score: {rec['similarity_score']:.4f}")
    
    # Evaluate overall performance
    performance_summary = recommender.evaluate_recommendations(demo_queries)
    
    logger.info("\n📊 Performance Summary:")
    logger.info(f"Total Recommendations: {performance_summary['total_recommendations']}")
    logger.info(f"Average Processing Time: {performance_summary['average_processing_time']:.4f} seconds")

def main():
    """
    Main entry point for the demo script.
    """
    logger = setup_logging()
    demonstrate_recommendation_system(logger)

if __name__ == "__main__":
    main()
