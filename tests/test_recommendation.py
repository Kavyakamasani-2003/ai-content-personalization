# tests/test_recommendation.py
import sys
import pytest
import numpy as np

from src.recommendation.content_recommender import ContentRecommender
from src.ml_predictors.advanced_recommender import AdvancedRecommender
from src.utils.performance_tracker import PerformanceTracker
from src.utils.advanced_feature_extraction import MultiModalFeatureExtractor

def test_basic_recommendation_system():
    """
    Test basic recommendation functionality
    """
    # Sample document corpus
    documents = [
        "Machine learning is transforming artificial intelligence",
        "Deep learning requires large neural network models",
        "AI is revolutionizing various industries with advanced algorithms",
        "Neural networks simulate human brain processing",
        "Data science uses statistical methods for insights"
    ]
    
    try:
        # Initialize recommender
        recommender = ContentRecommender(
            n_features=10
        )
        recommender.add_documents(documents)
        
        # Test with different queries
        queries = [
            "AI and machine learning technologies",
            "neural network models",
            "data science insights"
        ]
        
        for query in queries:
            recommendations = recommender.recommend_documents(query, top_n=2)
            
            # Assertions
            assert len(recommendations) > 0, f"No recommendations for query: {query}"
            assert len(recommendations) <= 2, f"Too many recommendations for query: {query}"
            assert all(isinstance(doc, str) for doc in recommendations), "Recommendations must be strings"
            
            print(f"\nQuery: {query}")
            print("Recommended Documents:")
            for doc in recommendations:
                print(f"- {doc}")
        
        print("\nTest Passed: Basic Recommendations generated successfully!")
        return True
    
    except Exception as e:
        print(f"Test Failed: {e}")
        return False

def test_recommendation_system_with_personalization():
    """
    Test recommendation system with personalization and dynamic feature weighting
    """
    # Sample document corpus
    documents = [
        "Machine learning is transforming artificial intelligence",
        "Deep learning requires large neural network models",
        "AI is revolutionizing various industries with advanced algorithms",
        "Neural networks simulate human brain processing",
        "Data science uses statistical methods for insights"
    ]
    
    try:
        # Initialize recommender with dynamic feature weighting
        recommender = ContentRecommender(
            n_features=10, 
            enable_dynamic_weighting=True
        )
        recommender.add_documents(documents)
        
        # Simulate user interactions
        user_id = "test_user_001"
        
        # Multiple queries to build interaction history
        queries = [
            "AI technologies",
            "machine learning algorithms",
            "neural network models"
        ]
        
        for query in queries:
            recommendations = recommender.recommend_documents(
                query, 
                top_n=2, 
                user_id=user_id
            )
            
            # Simulate user interaction (e.g., clicking on a recommendation)
            recommender.interaction_tracker.log_recommendation(
                user_id, 
                query, 
                recommendations, 
                interaction_type='click'
            )
        
        # Update feature weights based on interaction history
        recommender.update_feature_weights(user_id)
        
        # Verify feature weights have been updated
        assert recommender.feature_weighter is not None
        assert len(recommender.feature_weighter.feature_weights) > 0
        
        # Print feature weights for debugging
        print("\nFeature Weights:")
        for name, weight in recommender.feature_weighter.feature_weights.items():
            print(f"{name}: {weight}")
        
        print("\nTest Passed: Personalized Recommendations with Dynamic Weighting!")
        return True
    
    except Exception as e:
        print(f"Test Failed: {e}")
        return False

def test_performance_tracking():
    """
    Test performance tracking and logging
    """
    # Sample document corpus
    documents = [
        "Machine learning is transforming artificial intelligence",
        "Deep learning requires large neural network models",
        "AI is revolutionizing various industries with advanced algorithms",
        "Neural networks simulate human brain processing",
        "Data science uses statistical methods for insights"
    ]
    
    try:
        # Initialize recommender with performance tracking
        recommender = ContentRecommender(
            n_features=10, 
            enable_performance_tracking=True
        )
        recommender.add_documents(documents)
        
        # Simulate multiple recommendations
        queries = [
            "AI technologies",
            "machine learning algorithms",
            "neural network models",
            "advanced data science"
        ]
        
        for query in queries:
            recommender.recommend_documents(query, top_n=2)
        
        # Get performance summary
        performance_summary = recommender.get_performance_summary()
        
        # Verify performance tracking
        assert performance_summary is not None, "Performance tracking failed"
        assert 'recommend_documents' in performance_summary, "Recommendation method not tracked"
        
        # Print performance summary
        print("\nPerformance Summary:")
        for method, metrics in performance_summary.items():
            print(f"{method}:")
            print(f"  Total Calls: {metrics['total_calls']}")
            print(f"  Total Time: {metrics['total_time']:.4f} seconds")
            print(f"  Average Time: {metrics['avg_time']:.4f} seconds")
        
        print("\nTest Passed: Performance Tracking Successful!")
        return True
    
    except Exception as e:
        print(f"Test Failed: {e}")
        return False

def test_ml_relevance_prediction():
    """
    Test machine learning-based relevance prediction
    """
    documents = [
        "Machine learning is transforming artificial intelligence",
        "Deep learning requires large neural network models",
        "AI is revolutionizing various industries with advanced algorithms",
        "Neural networks simulate human brain processing",
        "Data science uses statistical methods for insights"
    ]
    
    try:
        recommender = ContentRecommender(
            n_features=10, 
            enable_ml_relevance_prediction=True
        )
        recommender.add_documents(documents)
        
        # Simulate interaction history
        interaction_history = [
            {'query': 'AI tech', 'relevance_score': 0.8},
            {'query': 'machine learning', 'relevance_score': 0.7},
            {'query': 'neural networks', 'relevance_score': 0.6}
        ]
        
        # Train relevance predictor
        recommender.train_relevance_predictor(interaction_history)
        
        # Test recommendations with trained predictor
        queries = [
            "AI technologies",
            "machine learning algorithms",
            "neural network models"
        ]
        
        for query in queries:
            recommendations = recommender.recommend_documents(
                query, 
                top_n=2, 
                use_ml_relevance=True
            )
            
            assert len(recommendations) > 0, f"No recommendations for query: {query}"
            assert len(recommendations) <= 2, f"Too many recommendations for query: {query}"
        
        print("\nTest Passed: ML Relevance Prediction Successful!")
        return True
    
    except Exception as e:
        print(f"Test Failed: {e}")
        return False

def test_multi_modal_feature_extraction():
    """
    Comprehensive test for multi-modal feature extraction
    """
    from src.utils.advanced_feature_extraction import MultiModalFeatureExtractor
    import numpy as np
    import pytest
    
    def prepare_test_data():
        """
        Prepare sample data for multi-modal feature extraction
        """
        # Sample text documents
        text_documents = [
            "Machine learning is transforming artificial intelligence",
            "Deep learning requires large neural network models",
            "AI is revolutionizing various industries with advanced algorithms",
            "Neural networks simulate human brain processing",
            "Data science uses statistical methods for insights"
        ]
        
        # Simulate image features (random vectors)
        image_features = np.random.rand(len(text_documents), 128)
        
        # Simulate metadata features
        metadata_features = {
            'domain': ['tech', 'research', 'industry', 'science', 'analytics'],
            'complexity': [0.7, 0.8, 0.6, 0.9, 0.5],
            'relevance_score': [0.9, 0.7, 0.8, 0.6, 0.7]
        }
        
        return text_documents, image_features, metadata_features
    
    def test_basic_feature_extraction():
        """
        Test basic multi-modal feature extraction functionality
        """
        # Prepare test data
        text_documents, image_features, metadata_features = prepare_test_data()
        
        # Initialize multi-modal feature extractor
        extractor = MultiModalFeatureExtractor(
            text_dim=100,  # Number of text features
            image_dim=64,  # Number of image features
            metadata_dim=30,  # Number of metadata features
            fusion_method='concatenate'
        )
        
        # Extract multi-modal features
        multi_modal_features = extractor.extract_features(
            text_documents=text_documents,
            image_features=image_features,
            metadata=metadata_features
        )
        
        # Assertions
        assert multi_modal_features is not None, "Multi-modal feature extraction failed"
        assert isinstance(multi_modal_features, np.ndarray), "Features must be a numpy array"
        assert multi_modal_features.shape[0] == len(text_documents), "Incorrect number of feature vectors"
        
        # Check individual feature components
        assert extractor.text_features is not None, "Text features not extracted"
        assert extractor.image_features is not None, "Image features not extracted"
        assert extractor.metadata_features is not None, "Metadata features not extracted"
        
        return multi_modal_features, extractor
    
    def test_feature_properties(multi_modal_features):
        """
        Analyze feature properties and diversity
        """
        # Check feature dimensionality
        assert multi_modal_features.ndim == 2, "Features should be 2-dimensional"
        
        # Compute feature statistics
        feature_mean = np.mean(multi_modal_features, axis=0)
        feature_std = np.std(multi_modal_features, axis=0)
        
        # Check feature normalization
        assert np.all(np.abs(feature_mean) < 1e-5), "Features not properly normalized"
        assert np.all(feature_std > 0), "Features lack variance"
        
        # Check for non-trivial correlations
        correlation_matrix = np.corrcoef(multi_modal_features.T)
        non_trivial_correlation = np.abs(correlation_matrix - np.eye(correlation_matrix.shape[0])).mean()
        
        assert non_trivial_correlation > 0.1, "Features appear to be trivially correlated"
    
    def test_fusion_methods():
        """
        Test different feature fusion approaches
        """
        text_documents, image_features, metadata_features = prepare_test_data()
        
        # Test concatenation fusion
        extractor_concat = MultiModalFeatureExtractor(fusion_method='concatenate')
        features_concat = extractor_concat.extract_features(
            text_documents=text_documents,
            image_features=image_features,
            metadata=metadata_features
        )
        
        # Test weighted sum fusion
        extractor_weighted = MultiModalFeatureExtractor(fusion_method='weighted_sum')
        features_weighted = extractor_weighted.extract_features(
            text_documents=text_documents,
            image_features=image_features,
            metadata=metadata_features
        )
        
        # Assertions
        assert features_concat is not None, "Concatenation fusion failed"
        assert features_weighted is not None, "Weighted sum fusion failed"
        
        # Ensure different fusion methods produce different feature representations
        assert not np.array_equal(features_concat, features_weighted), "Fusion methods produce identical features"
    
    def test_edge_cases():
        """
        Test edge cases and error handling
        """
        extractor = MultiModalFeatureExtractor()
        
        # Test with empty documents
        with pytest.raises(ValueError, match="No text documents provided"):
            extractor.extract_features(
                text_documents=[],
                image_features=None,
                metadata=None
            )
        
        # Test with mismatched feature dimensions
        text_documents = ["Sample document"]
        image_features = np.random.rand(2, 128)  # Mismatched dimensions
        
        with pytest.raises(ValueError, match="Mismatched feature dimensions"):
            extractor.extract_features(
                text_documents=text_documents,
                image_features=image_features,
                metadata=None
            )
    
    # Main test execution
    try:
        # Run test scenarios
        multi_modal_features, extractor = test_basic_feature_extraction()
        test_feature_properties(multi_modal_features)
        test_fusion_methods()
        test_edge_cases()
        
        print("Multi-Modal Feature Extraction Tests Passed Successfully!")
        return True
    
    except AssertionError as e:
        print(f"Multi-Modal Feature Extraction Test Failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error in Multi-Modal Feature Extraction Test: {e}")
        return False

def run_all_tests():
    """
    Run all recommendation system tests
    """
    basic_test_result = test_basic_recommendation_system()
    personalization_test_result = test_recommendation_system_with_personalization()
    performance_test_result = test_performance_tracking()
    ml_relevance_test_result = test_ml_relevance_prediction()
    multi_modal_test_result = test_multi_modal_feature_extraction()
    
    return (
        basic_test_result and 
        personalization_test_result and 
        performance_test_result and
        ml_relevance_test_result and
        multi_modal_test_result
    )

if __name__ == "__main__":
    result = run_all_tests()
    sys.exit(0 if result else 1)