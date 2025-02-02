import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import MinMaxScaler

class DynamicFeatureWeighter:
    def __init__(self, initial_weights: Dict[str, float] = None):
        """
        Dynamic feature weighting mechanism
        
        Args:
            initial_weights (Dict[str, float]): Initial feature weights
        """
        self.feature_weights = initial_weights or {}
        self.scaler = MinMaxScaler()
    
    def update_weights(self, 
                       feature_names: List[str], 
                       interaction_scores: List[float], 
                       learning_rate: float = 0.1):
        """
        Update feature weights based on user interactions
        
        Args:
            feature_names (List[str]): Names of features
            interaction_scores (List[float]): Interaction importance for each feature
            learning_rate (float): Rate of weight adjustment
        """
        # Normalize interaction scores
        normalized_scores = self.scaler.fit_transform(
            np.array(interaction_scores).reshape(-1, 1)
        ).flatten()
        
        for name, score in zip(feature_names, normalized_scores):
            if name not in self.feature_weights:
                self.feature_weights[name] = 1.0
            
            # Adjust weight based on interaction score
            self.feature_weights[name] += learning_rate * (score - self.feature_weights[name])
    
    def apply_weights(self, feature_matrix: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Apply dynamic weights to feature matrix
        
        Args:
            feature_matrix (np.ndarray): Original feature matrix
            feature_names (List[str]): Names of features
        
        Returns:
            np.ndarray: Weighted feature matrix
        """
        # Ensure feature_matrix is 2D
        if feature_matrix.ndim == 1:
            feature_matrix = feature_matrix.reshape(1, -1)
        
        # Create weight vector
        weight_vector = np.array([
            self.feature_weights.get(name, 1.0) 
            for name in feature_names[:feature_matrix.shape[1]]
        ])
        
        # Ensure weight vector matches feature matrix width
        if len(weight_vector) < feature_matrix.shape[1]:
            # Pad with 1s if needed
            padding = np.ones(feature_matrix.shape[1] - len(weight_vector))
            weight_vector = np.concatenate([weight_vector, padding])
        
        # Apply weights
        return feature_matrix * weight_vector