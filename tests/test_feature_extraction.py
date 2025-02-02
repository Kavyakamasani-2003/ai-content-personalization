# tests/test_feature_extraction.py
import pytest
import numpy as np
from src.utils.feature_engineering import AdvancedFeatureExtractor

class TestAdvancedFeatureExtractor:
    def setup_method(self):
        """
        Setup method to initialize test documents before each test
        """
        self.documents = [
            "Machine learning is transforming artificial intelligence",
            "Deep learning requires large neural network models",
            "AI is revolutionizing various industries with advanced algorithms",
            "Neural networks simulate human brain processing",
            "Data science uses statistical methods for insights"
        ]
    
    def test_basic_feature_extraction(self):
        """
        Test basic feature extraction functionality
        """
        # Test with TF-IDF and SVD
        extractor = AdvancedFeatureExtractor(
            use_tfidf=True,
            use_svd=True,
            n_components=10
        )
        
        # Extract features
        features = extractor.fit_transform(self.documents)
        
        # Assertions
        assert isinstance(features, np.ndarray), "Features must be a numpy array"
        assert features.shape[0] == len(self.documents), "Incorrect number of document features"
        assert features.shape[1] == 10, "Incorrect number of feature components"
    
    def test_feature_extraction_without_svd(self):
        """
        Test feature extraction without SVD
        """
        extractor = AdvancedFeatureExtractor(
            use_tfidf=True,
            use_svd=False,
            n_components=20
        )
        
        features = extractor.fit_transform(self.documents)
        
        assert isinstance(features, np.ndarray), "Features must be a numpy array"
        assert features.shape[0] == len(self.documents), "Incorrect number of document features"
    
    def test_transform_method(self):
        """
        Test transform method with new documents
        """
        # First fit on initial documents
        extractor = AdvancedFeatureExtractor(
            use_tfidf=True,
            use_svd=True,
            n_components=10
        )
        extractor.fit_transform(self.documents)
        
        # New documents for transformation
        new_documents = [
            "Advanced machine learning techniques",
            "Innovative AI solutions"
        ]
        
        # Transform new documents
        new_features = extractor.transform(new_documents)
        
        assert isinstance(new_features, np.ndarray), "Transformed features must be a numpy array"
        assert new_features.shape[1] == 10, "Incorrect number of feature components"
    
    def test_feature_names(self):
        """
        Test retrieval of feature names
        """
        extractor = AdvancedFeatureExtractor(
            use_tfidf=True,
            use_svd=True,
            n_components=10
        )
        extractor.fit_transform(self.documents)
        
        feature_names = extractor.get_feature_names()
        
        assert isinstance(feature_names, list), "Feature names must be a list"
        assert len(feature_names) > 0, "No feature names retrieved"
    
    def test_feature_importance(self):
        """
        Test feature importance computation
        """
        extractor = AdvancedFeatureExtractor(
            use_tfidf=True,
            use_svd=True,
            n_components=10
        )
        extractor.fit_transform(self.documents)
        
        # Compute feature importance
        feature_importance = extractor.compute_feature_importance()
        
        assert isinstance(feature_importance, dict), "Feature importance must be a dictionary"
        assert len(feature_importance) > 0, "No feature importance computed"
        
        # Check if features are sorted by importance
        importance_values = list(feature_importance.values())
        assert importance_values == sorted(importance_values, reverse=True), "Features not sorted by importance"
    
    def test_preprocessing(self):
        """
        Test text preprocessing
        """
        extractor = AdvancedFeatureExtractor()
        preprocessed_docs = extractor.preprocess_text(self.documents)
        
        assert len(preprocessed_docs) == len(self.documents), "Preprocessing changed document count"
        assert all(len(doc.split()) > 0 for doc in preprocessed_docs), "Some documents became empty after preprocessing"
    
    def test_edge_cases(self):
        """
        Test edge cases and error handling
        """
        # Empty document list
        with pytest.raises(ValueError, match="No documents provided"):
            extractor = AdvancedFeatureExtractor()
            extractor.fit_transform([])
        
        # Single very short document
        single_doc = ["a"]
        extractor = AdvancedFeatureExtractor(n_components=5)
        features = extractor.fit_transform(single_doc)
        
        assert features.shape[0] == 1, "Failed to handle single document"
# Optional: Main execution for standalone testing
if __name__ == "__main__":
    pytest.main([__file__])