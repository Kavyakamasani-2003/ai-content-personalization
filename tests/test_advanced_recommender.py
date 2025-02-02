# tests/test_advanced_recommender.py
import pytest
import numpy as np
from src.ml_predictors.advanced_recommender import AIContentRecommender

class TestAIContentRecommender:
    def setup_method(self):
        """
        Setup method for each test
        """
        self.recommender = AIContentRecommender()
    
    def test_add_content(self):
        """
        Test adding content to the recommender
        """
        contents = [
            "Machine learning is fascinating",
            "AI transforms industries",
            "Data science requires advanced algorithms"
        ]
        
        self.recommender.add_content(contents)
        
        assert len(self.recommender.content_repository) == 3
        assert self.recommender.content_features is not None
        assert self.recommender.content_features.shape[0] == 3
    
    def test_recommend(self):
        """
        Test recommendation generation
        """
        contents = [
            "Machine learning revolutionizes data science",
            "AI algorithms improve predictive analytics",
            "Deep learning transforms image recognition"
        ]
        self.recommender.add_content(contents)
        
        query = "advanced data technology"
        recommendations = self.recommender.recommend(query, top_k=2)
        
        assert len(recommendations) == 2
        assert all('content' in rec for rec in recommendations)
        assert all('similarity_score' in rec for rec in recommendations)
    
    def test_performance_tracking(self):
        """
        Test performance tracking during recommendations
        """
        contents = [
            "Machine learning is powerful",
            "AI enhances decision making",
            "Data analysis requires sophisticated tools"
        ]
        self.recommender.add_content(contents)
        
        # Generate multiple recommendations
        for _ in range(3):
            self.recommender.recommend("advanced technology")
        
        performance = self.recommender.evaluate_recommendations([])
        
        assert performance['total_recommendations'] == 3
        assert performance['average_processing_time'] > 0
    
    def test_empty_repository(self):
        """
        Test recommendation with empty content repository
        """
        query = "test query"
        
        with pytest.raises(Exception):  # Expect an exception due to no content
            self.recommender.recommend(query)

    def test_personalization_weight(self):
        """
        Test recommendation with different personalization weights
        """
        contents = [
            "Machine learning is transformative",
            "AI drives innovation",
            "Data science evolves rapidly"
        ]
        self.recommender.add_content(contents)
        
        # Test different personalization weights
        recommendations1 = self.recommender.recommend(
            "advanced technology", 
            personalization_weight=0.5
        )
        recommendations2 = self.recommender.recommend(
            "advanced technology", 
            personalization_weight=0.9
        )
        
        assert len(recommendations1) > 0
        assert len(recommendations2) > 0