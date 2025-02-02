# tests/test_content_similarity.py
import unittest
import numpy as np
from src.similarity.content_similarity import ContentSimilarityAnalyzer

class TestContentSimilarityAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = ContentSimilarityAnalyzer()
        self.documents = [
            "The quick brown fox jumps over the lazy dog",
            "A quick brown fox is very energetic",
            "Lazy dogs sleep all day long",
            "Foxes are intelligent animals"
        ]
    
    def test_compute_similarity(self):
        similarity_matrix = self.analyzer.compute_similarity(self.documents)
        
        # Check matrix shape
        self.assertEqual(
            similarity_matrix.shape, 
            (len(self.documents), len(self.documents))
        )
        
        # Diagonal should be close to 1 (self-similarity)
        for i in range(len(self.documents)):
            self.assertAlmostEqual(similarity_matrix[i][i], 1.0, places=5)
    
    def test_find_most_similar(self):
        # Find most similar to first document
        similar_indices = self.analyzer.find_most_similar(
            self.documents, 
            target_doc_index=0, 
            top_n=2
        )
        
        # Check returned indices
        self.assertEqual(len(similar_indices), 2)
        self.assertTrue(all(0 <= idx < len(self.documents) for idx in similar_indices))

if __name__ == '__main__':
    unittest.main()