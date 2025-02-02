# src/feature_extraction/feature_extractor.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

class FeatureExtractor:
    def __init__(self, method='tfidf'):
        """
        Initialize feature extraction method
        
        Args:
            method (str): Feature extraction method 
                          Options: 'tfidf', 'count', 'lda', 'lsi'
        """
        self.method = method
        self._vectorizer = None
        self._transformer = None
        self._feature_names = []  # Initialize as empty list
    
    def extract_features(self, documents, n_features=100):
        """
        Extract features from documents
        
        Args:
            documents (list): List of text documents
            n_features (int): Number of features to extract
        
        Returns:
            numpy.ndarray: Extracted features
        """
        if self.method == 'tfidf':
            self._vectorizer = TfidfVectorizer(max_features=n_features)
            features = self._vectorizer.fit_transform(documents)
            self._feature_names = list(self._vectorizer.get_feature_names_out())
            return features.toarray()
        
        elif self.method == 'count':
            self._vectorizer = CountVectorizer(max_features=n_features)
            features = self._vectorizer.fit_transform(documents)
            self._feature_names = list(self._vectorizer.get_feature_names_out())
            return features.toarray()
        
        elif self.method == 'lda':
            # Latent Dirichlet Allocation for topic modeling
            vectorizer = CountVectorizer(max_features=n_features)
            doc_term_matrix = vectorizer.fit_transform(documents)
            
            # Store vectorizer for feature names
            self._vectorizer = vectorizer
            self._feature_names = list(vectorizer.get_feature_names_out())
            
            self._transformer = LatentDirichletAllocation(
                n_components=n_features, 
                random_state=42
            )
            features = self._transformer.fit_transform(doc_term_matrix)
            return features
        
        elif self.method == 'lsi':
            # Latent Semantic Indexing
            vectorizer = TfidfVectorizer(max_features=n_features)
            doc_term_matrix = vectorizer.fit_transform(documents)
            
            # Store vectorizer for feature names
            self._vectorizer = vectorizer
            self._feature_names = list(vectorizer.get_feature_names_out())
            
            self._transformer = TruncatedSVD(
                n_components=n_features, 
                random_state=42
            )
            features = self._transformer.fit_transform(doc_term_matrix)
            return features
        
        else:
            raise ValueError(f"Unsupported feature extraction method: {self.method}")
    
    def get_top_features(self, n_top=10):
        """
        Get top features based on the extraction method
        
        Args:
            n_top (int): Number of top features to return
        
        Returns:
            list: Top feature names or topics
        """
        # Check if feature names list is empty
        if len(self._feature_names) == 0:
            raise ValueError("Features not extracted. Call extract_features() first.")
        
        if self.method == 'tfidf' or self.method == 'count':
            return self._feature_names[:n_top]
        
        elif self.method == 'lda':
            # Get top words for each topic
            topics = []
            for topic_idx, topic in enumerate(self._transformer.components_):
                top_features_ind = topic.argsort()[:-n_top - 1:-1]
                top_features = [self._feature_names[i] for i in top_features_ind]
                topics.append(top_features)
            return topics
        
        elif self.method == 'lsi':
            # Get top components
            return self._transformer.components_[:n_top].tolist()