# src/similarity/content_similarity.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentSimilarityAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    def compute_similarity(self, documents):
        """
        Compute similarity matrix for a list of documents
        
        Args:
            documents (list): List of text documents
        
        Returns:
            numpy.ndarray: Similarity matrix
        """
        # Convert documents to TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        return similarity_matrix
    
    def find_most_similar(self, documents, target_doc_index, top_n=3):
        """
        Find most similar documents to a target document
        
        Args:
            documents (list): List of text documents
            target_doc_index (int): Index of target document
            top_n (int): Number of top similar documents to return
        
        Returns:
            list: Indices of most similar documents
        """
        similarity_matrix = self.compute_similarity(documents)
        
        # Get similarities for target document
        target_similarities = similarity_matrix[target_doc_index]
        
        # Sort similarities (excluding self-similarity)
        similar_indices = np.argsort(target_similarities)[::-1]
        
        # Remove self and return top N
        similar_indices = [
            idx for idx in similar_indices 
            if idx != target_doc_index
        ][:top_n]
        
        return similar_indices