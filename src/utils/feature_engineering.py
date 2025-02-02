# src/utils/feature_engineering.py
import os
import re
import numpy as np
import nltk
from typing import List, Dict, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK Resource Management
nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt', download_dir=nltk_data_dir)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK 'stopwords'...")
    nltk.download('stopwords', download_dir=nltk_data_dir)



# ... (previous NLTK setup remains the same)

class AdvancedFeatureExtractor:
    def __init__(self, 
                use_tfidf=True, 
                use_svd=True, 
                n_components=50, 
                ngram_range=(1, 2)):
        """
        Advanced feature extraction with multiple techniques
        
        Args:
            use_tfidf (bool): Use TF-IDF vectorization
            use_svd (bool): Apply Truncated SVD for dimensionality reduction
            n_components (int): Number of components for SVD
            ngram_range (tuple): Range of n-grams to consider
        """
        self.use_tfidf = use_tfidf
        self.use_svd = use_svd
        self.n_components = n_components
        self.ngram_range = ngram_range
        
        # Initialize TF-IDF Vectorizer with more flexible settings
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=self.ngram_range,
            min_df=1,  # Include even single document features
            token_pattern=r'\b\w+\b'  # More flexible token pattern
        )
        
        # Initialize feature scaling
        self.scaler = StandardScaler()
        
        # Initialize SVD
        self.svd = TruncatedSVD(
            n_components=self.n_components, 
            random_state=42
        ) if use_svd else None
        
        # Store the last fitted feature matrix
        self.feature_matrix = None
        
        # Custom stopwords
        self.custom_stopwords = set(stopwords.words('english'))

    def preprocess_text(self, documents: List[str]) -> List[str]:
        """
        Preprocess input documents
        
        Args:
            documents (List[str]): Input documents
        
        Returns:
            List[str]: Preprocessed documents
        """
        # Check for empty document list
        if not documents:
            raise ValueError("No documents provided")
        
        preprocessed_docs = []
        for doc in documents:
            # Handle very short or empty documents
            if not doc or len(doc.strip()) < 3:
                preprocessed_docs.append("default document")
                continue
            
            # Convert to lowercase
            doc = doc.lower()
            
            # Remove special characters and digits
            doc = re.sub(r'[^a-zA-Z\s]', '', doc)
            
            # Tokenize
            tokens = word_tokenize(doc)
            
            # Remove stopwords and short words
            tokens = [
                token for token in tokens 
                if token not in self.custom_stopwords and len(token) > 2
            ]
            
            # If no tokens remain, use a default
            preprocessed_docs.append(' '.join(tokens) if tokens else "default document")
        
        return preprocessed_docs

    def get_feature_names(self) -> List[str]:
        """
        Get feature names from the vectorizer
        
        Returns:
            List[str]: Feature names
        """
        try:
            # Explicitly convert to list and handle potential numpy array
            feature_names = self.vectorizer.get_feature_names_out()
            return feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names)
        except Exception:
            return []

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fit and transform documents into feature matrix
        
        Args:
            documents (List[str]): Input documents
        
        Returns:
            np.ndarray: Feature matrix
        """
        # Check for empty document list
        if not documents:
            raise ValueError("No documents provided")
        
        # Preprocess documents
        preprocessed_docs = self.preprocess_text(documents)
        
        # TF-IDF Vectorization
        tfidf_matrix = self.vectorizer.fit_transform(preprocessed_docs)
        
        # Optional SVD Dimensionality Reduction
        if self.use_svd:
            # Ensure we don't request more components than available
            max_components = min(self.n_components, tfidf_matrix.shape[1])
            
            # Reinitialize SVD with correct number of components
            self.svd = TruncatedSVD(n_components=max_components, random_state=42)
            
            # Perform SVD
            svd_matrix = self.svd.fit_transform(tfidf_matrix)
            
            # Scale the features
            scaled_matrix = self.scaler.fit_transform(svd_matrix)
            
            # Adjust matrix to match n_components
            scaled_matrix = self._adjust_matrix_size(scaled_matrix)
            
            self.feature_matrix = scaled_matrix
            return scaled_matrix
        
        # Convert sparse matrix to dense array
        dense_matrix = tfidf_matrix.toarray()
        
        # Scale the features
        scaled_matrix = self.scaler.fit_transform(dense_matrix)
        
        # Adjust matrix to match n_components
        scaled_matrix = self._adjust_matrix_size(scaled_matrix)
        
        self.feature_matrix = scaled_matrix
        return scaled_matrix

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform new documents using the fitted vectorizer
        
        Args:
            documents (List[str]): Input documents
        
        Returns:
            np.ndarray: Transformed feature matrix
        """
        # Preprocess documents
        preprocessed_docs = self.preprocess_text(documents)
        
        # TF-IDF Vectorization
        tfidf_matrix = self.vectorizer.transform(preprocessed_docs)
        
        # Optional SVD Dimensionality Reduction
        if self.use_svd:
            # Transform using fitted SVD
            svd_matrix = self.svd.transform(tfidf_matrix)
            
            # Scale the features
            scaled_matrix = self.scaler.transform(svd_matrix)
            
            # Adjust matrix to match n_components
            scaled_matrix = self._adjust_matrix_size(scaled_matrix)
            
            return scaled_matrix
        
        # Convert and scale
        dense_matrix = tfidf_matrix.toarray()
        scaled_matrix = self.scaler.transform(dense_matrix)
        
        # Adjust matrix to match n_components
        scaled_matrix = self._adjust_matrix_size(scaled_matrix)
        
        return scaled_matrix

    def _adjust_matrix_size(self, matrix: np.ndarray) -> np.ndarray:
        """
        Adjust matrix size to match n_components
        
        Args:
            matrix (np.ndarray): Input matrix
        
        Returns:
            np.ndarray: Adjusted matrix
        """
        # Truncate or pad to match n_components
        if matrix.shape[1] > self.n_components:
            matrix = matrix[:, :self.n_components]
        elif matrix.shape[1] < self.n_components:
            padding = np.zeros((matrix.shape[0], self.n_components - matrix.shape[1]))
            matrix = np.hstack([matrix, padding])
        
        return matrix

    def compute_feature_importance(self) -> Dict[str, float]:
        """
        Compute feature importance
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if not self.use_svd:
            raise ValueError("SVD must be enabled to compute feature importance")
        
        feature_names = self.get_feature_names()
        feature_importance = {}
        
        # Use the first SVD component for feature importance
        first_component = self.svd.components_[0]
        
        for feature, importance in zip(feature_names, first_component):
            feature_importance[feature] = abs(importance)
        
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))