import numpy as np
from typing import List, Dict, Any
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize

from src.utils.feature_engineering import AdvancedFeatureExtractor
from src.utils.feature_weighting import DynamicFeatureWeighter
from src.utils.user_interaction_tracker import UserInteractionTracker
from src.utils.performance_tracker import PerformanceTracker
from src.utils.logging_config import setup_logging
from src.ml_predictors.relevance_scorer import RelevancePredictor
from src.utils.visualization import RecommendationVisualizer
from src.utils.advanced_feature_extraction import MultiModalFeatureExtractor
from src.utils.similarity_metrics import AdvancedSimilarityMetrics

import logging

class AdvancedMLCachePredictor:
    def __init__(self, cache_size=50):
        """
        Machine Learning-based Cache Predictor
        
        Args:
            cache_size (int): Maximum number of cache entries
        """
        self.cache = {}
        self.cache_hit_history = []
        self.cache_size = cache_size

    def predict_cache_hit(self, query):
        """
        Predict likelihood of cache hit for a query
        
        Args:
            query (str): Input query
        
        Returns:
            bool: Predicted cache hit
        """
        # Placeholder for ML-based cache prediction
        # In a real implementation, this would use ML models
        return len(self.cache) > 0 and hash(query) in self.cache

    def update_cache_history(self, cache_hit):
        """
        Update cache hit history
        
        Args:
            cache_hit (bool): Whether a cache hit occurred
        """
        self.cache_hit_history.append(cache_hit)
        
        # Maintain fixed-size history
        if len(self.cache_hit_history) > self.cache_size:
            self.cache_hit_history.pop(0)

class ContentRecommender:
    def __init__(self, 
                 feature_method='tfidf', 
                 n_features=100, 
                 similarity_metric='cosine', 
                 cache_size=100, 
                 cache_strategy='lru', 
                 ml_cache_prediction=False,
                 use_svd=True,
                 enable_dynamic_weighting=True,
                 enable_performance_tracking=True,
                 enable_ml_relevance_prediction=True,
                 use_multi_modal_features=True):
        """
        Enhanced Content-Based Recommender with advanced features
        
        Args:
            feature_method (str): Feature extraction method
            n_features (int): Number of features to extract
            similarity_metric (str): Metric to calculate document similarity
            cache_size (int): Size of recommendation cache
            cache_strategy (str): Cache management strategy
            ml_cache_prediction (bool): Enable machine learning cache prediction
            use_svd (bool): Use Singular Value Decomposition for dimensionality reduction
            enable_dynamic_weighting (bool): Enable dynamic feature weighting
            enable_performance_tracking (bool): Enable performance monitoring
            enable_ml_relevance_prediction (bool): Enable ML-based relevance prediction
            use_multi_modal_features (bool): Use multi-modal feature extraction
        """
        # Logging setup
        self.logger = setup_logging()
        
        # Feature extraction
        self.feature_extractor = AdvancedFeatureExtractor(
            use_tfidf=True, 
            use_svd=use_svd, 
            n_components=n_features,
            ngram_range=(1, 2)
        )
        
        # Multi-modal feature extraction
        self.multi_modal_extractor = (
            MultiModalFeatureExtractor() 
            if use_multi_modal_features 
            else None
        )
        
        # Document storage
        self.documents = []
        self.feature_matrix = None
        
        # Recommendation cache
        self.recommendation_cache = {}
        self.cache_strategy = cache_strategy
        
        # Performance tracking
        self.performance_tracker = (
            PerformanceTracker() if enable_performance_tracking else None
        )
        
        # Cache prediction
        self.ml_cache_predictor = (
            AdvancedMLCachePredictor(cache_size) if ml_cache_prediction else None
        )
        
        # Dynamic Feature Weighting
        self.feature_weighter = (
            DynamicFeatureWeighter() if enable_dynamic_weighting else None
        )
        
        # User Interaction Tracking
        self.interaction_tracker = UserInteractionTracker()
        
        # ML Relevance Predictor
        self.relevance_predictor = (
            RelevancePredictor(n_features) 
            if enable_ml_relevance_prediction 
            else None
        )
        
        # Similarity metric
        self.similarity_metric = similarity_metric
        self.cache_size = cache_size

    def _generate_cache_key(self, query_document):
        """
        Generate a unique cache key for a query document
        
        Args:
            query_document (str): Query document
        
        Returns:
            str: Unique cache key
        """
        return hash(query_document)

    def _manage_cache(self, cache_key, recommendations):
        """
        Manage recommendation cache based on strategy
        
        Args:
            cache_key (str): Cache key for the query
            recommendations (List[str]): Recommended documents
        """
        if len(self.recommendation_cache) >= self.cache_size:
            if self.cache_strategy == 'lru':
                # Remove least recently used
                self.recommendation_cache.pop(
                    next(iter(self.recommendation_cache))
                )
            elif self.cache_strategy == 'fifo':
                # Remove first added
                self.recommendation_cache.pop(
                    list(self.recommendation_cache.keys())[0]
                )
        
        self.recommendation_cache[cache_key] = recommendations

    def recommend_documents(self, 
                             query: str, 
                             top_n: int = 5, 
                             min_similarity: float = 0.01,
                             similarity_method: str = 'cosine',
                             user_id: str = None) -> List[str]:
        """
        Generate document recommendations based on query similarity
        
        Args:
            query (str): Input query text
            top_n (int): Number of recommendations to return
            min_similarity (float): Minimum similarity threshold
            similarity_method (str): Method to calculate similarity
            user_id (str): Optional user identifier for personalization
        
        Returns:
            List[str]: Top recommended documents
        """
        # Performance tracking decorator
        track_performance = (
            self.performance_tracker.track_performance 
            if self.performance_tracker 
            else lambda x: x
        )
        
        # Generate cache key
        cache_key = self._generate_cache_key(query)
        
        # Check cache first
        if cache_key in self.recommendation_cache:
            return self.recommendation_cache[cache_key]
        
        # Extract features for query
        query_features = self.feature_extractor.transform([query])
        
        # Dynamic feature weighting
        if self.feature_weighter:
            feature_names = self.get_feature_names()
            query_features = self.feature_weighter.apply_weights(
                query_features, 
                feature_names
            )
        
        # Ensure query_features is 1-D for similarity calculation
        query_features = query_features.flatten()
        
        # Calculate similarities
        similarities = []
        for i, doc_features in enumerate(self.feature_matrix):
            # Ensure doc_features is 1-D
            doc_features = doc_features.flatten()
            
            # Choose similarity method
            if similarity_method == 'cosine':
                similarity = AdvancedSimilarityMetrics.cosine_similarity(
                    query_features, doc_features
                )
            elif similarity_method == 'jaccard':
                similarity = AdvancedSimilarityMetrics.jaccard_similarity(
                    query_features, doc_features
                )
            elif similarity_method == 'weighted':
                # Use feature weights if available
                weights = (
                    self.feature_weighter.feature_weights 
                    if self.feature_weighter 
                    else None
                )
                similarity = AdvancedSimilarityMetrics.weighted_similarity(
                    query_features, doc_features, weights
                )
            else:
                # Default to cosine similarity
                similarity = 1 - cosine(query_features, doc_features)
            
            # ML-based relevance scoring if enabled
            if self.relevance_predictor and self.relevance_predictor.is_trained:
                relevance_score = self.relevance_predictor.predict_relevance(
                    doc_features.reshape(1, -1)
                )[0]
                similarity *= relevance_score
            
            # Filter by minimum similarity
            if similarity >= min_similarity:
                similarities.append((similarity, i))
        
        # Sort and get top recommendations
        similarities.sort(reverse=True)
        recommended_indices = [idx for _, idx in similarities[:top_n]]
        recommendations = [self.documents[idx] for idx in recommended_indices]
        
        # Update cache
        self._manage_cache(cache_key, recommendations)
        
        # Track user interaction if user_id provided
        if user_id:
            self.interaction_tracker.log_recommendation(
                user_id, 
                query, 
                recommendations
            )
        
        return recommendations

    def add_documents(self, documents: List[str]):
        """
        Add documents to the recommendation system
        
        Args:
            documents (List[str]): List of documents to add
        """
        # Add new documents
        self.documents.extend(documents)
        
        # Multi-modal feature extraction
        if self.multi_modal_extractor:
            # Train Word2Vec if enabled
            self.multi_modal_extractor.train_word2vec(documents)
            
            # Extract multi-modal features
            multi_modal_features = self.multi_modal_extractor.extract_features(documents)
            
            # Combine or choose feature representation
            self.feature_matrix = multi_modal_features.get('tfidf', 
                multi_modal_features.get('word2vec')
            )
        else:
            # Fallback to existing feature extraction
            self.feature_matrix = self.feature_extractor.transform(self.documents)
        
        # Log document addition
        self.logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")

    def get_feature_names(self) -> List[str]:
        """
        Get names of extracted features
        
        Returns:
            List[str]: Names of features extracted by the vectorizer
        """
        return self.feature_extractor.get_feature_names()

    def update_feature_weights(self, 
                                user_id: str, 
                                learning_rate: float = 0.1):
        """
        Update feature weights based on user interaction history
        
        Args:
            user_id (str): User identifier
            learning_rate (float): Rate of weight adjustment
        """
        if not self.feature_weighter:
            return
        
        interaction_history = self.interaction_tracker.get_user_interactions(user_id)
        feature_names = self.get_feature_names()
        
        # Compute interaction scores (example: based on click-through rate)
        interaction_scores = [
            interaction.get('relevance_score', 0.5) 
            for interaction in interaction_history
        ]
        
        self.feature_weighter.update_weights(
            feature_names, 
            interaction_scores, 
            learning_rate
        )

    def train_relevance_predictor(self, interaction_history):
        """
        Train ML-based relevance predictor
        
        Args:
            interaction_history (List[Dict]): User interaction data
        """
        if not self.relevance_predictor:
            return
        
        # Prepare training data
        features = self.feature_matrix
        interaction_scores = [
            interaction.get('relevance_score', 0.5) 
            for interaction in interaction_history
        ]
        
        self.relevance_predictor.prepare_training_data(
            features, interaction_scores
        )

    def get_performance_summary(self):
        """
        Retrieve performance metrics
        
        Returns:
            dict: Performance summary or None
        """
        return (
            self.performance_tracker.get_performance_summary() 
            if self.performance_tracker 
            else None
        )

    def visualize_feature_weights(self):
        """
        Visualize current feature weights
        """
        if self.feature_weighter:
            RecommendationVisualizer.plot_feature_weights(
                self.feature_weighter.feature_weights
            )
    
    def visualize_performance(self):
        """
        Visualize performance metrics
        """
        performance_summary = self.get_performance_summary()
        if performance_summary:
            RecommendationVisualizer.plot_performance_metrics(
                performance_summary
            )