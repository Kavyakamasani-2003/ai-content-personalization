import numpy as np
from scipy.spatial.distance import cosine, jaccard
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances

class AdvancedSimilarityMetrics:
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors
        
        Args:
            vec1 (np.ndarray): First vector
            vec2 (np.ndarray): Second vector
        
        Returns:
            float: Cosine similarity score
        """
        return 1 - cosine(vec1, vec2)
    
    @staticmethod
    def jaccard_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute Jaccard similarity between two binary vectors
        
        Args:
            vec1 (np.ndarray): First binary vector
            vec2 (np.ndarray): Second binary vector
        
        Returns:
            float: Jaccard similarity score
        """
        return 1 - jaccard(vec1, vec2)
    
    @staticmethod
    def weighted_similarity(
        vec1: np.ndarray, 
        vec2: np.ndarray, 
        weights: np.ndarray = None
    ) -> float:
        """
        Compute weighted cosine similarity
        
        Args:
            vec1 (np.ndarray): First vector
            vec2 (np.ndarray): Second vector
            weights (np.ndarray): Optional feature weights
        
        Returns:
            float: Weighted similarity score
        """
        # Apply weights if provided
        if weights is not None:
            weighted_vec1 = vec1 * weights
            weighted_vec2 = vec2 * weights
        else:
            weighted_vec1, weighted_vec2 = vec1, vec2
        
        return 1 - cosine(weighted_vec1, weighted_vec2)