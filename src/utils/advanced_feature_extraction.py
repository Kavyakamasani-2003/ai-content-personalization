# src/utils/advanced_feature_extraction.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class MultiModalFeatureExtractor:
    def __init__(self, 
                 text_dim=100, 
                 image_dim=128, 
                 metadata_dim=50,
                 fusion_method='concatenate'):
        """
        Multi-modal feature extraction and fusion
        
        Args:
            text_dim (int): Dimensionality of text features
            image_dim (int): Dimensionality of image features
            metadata_dim (int): Dimensionality of metadata features
            fusion_method (str): Method of feature fusion
        """
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.metadata_dim = metadata_dim
        self.fusion_method = fusion_method
        
        # Feature storage
        self.text_features = None
        self.image_features = None
        self.metadata_features = None
    
    def _preprocess_text_features(self, text_documents):
        """
        Preprocess text features using TF-IDF or embedding
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            max_features=self.text_dim,
            stop_words='english'
        )
        return vectorizer.fit_transform(text_documents).toarray()
    
    def _preprocess_image_features(self, image_features):
        """
        Preprocess image features
        """
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(image_features)
        
        # Dimensionality reduction if needed
        if scaled_features.shape[1] > self.image_dim:
            pca = PCA(n_components=self.image_dim)
            scaled_features = pca.fit_transform(scaled_features)
        
        return scaled_features
    
    def _preprocess_metadata_features(self, metadata):
        """
        Convert metadata to numerical features
        """
        # Example metadata processing
        processed_metadata = []
        for key in metadata:
            if isinstance(metadata[key], list):
                # Convert categorical to numerical if needed
                processed_metadata.append(
                    StandardScaler().fit_transform(
                        np.array(metadata[key]).reshape(-1, 1)
                    )
                )
        
        return np.hstack(processed_metadata) if processed_metadata else None
    
    def extract_features(self, 
                         text_documents, 
                         image_features=None, 
                         metadata=None):
        """
        Extract and fuse multi-modal features
        
        Args:
            text_documents (List[str]): Input text documents
            image_features (np.ndarray, optional): Image features
            metadata (Dict, optional): Additional metadata
        
        Returns:
            np.ndarray: Fused multi-modal features
        """
        # Text feature extraction
        self.text_features = self._preprocess_text_features(text_documents)
        
        # Image feature extraction
        self.image_features = (
            self._preprocess_image_features(image_features) 
            if image_features is not None 
            else np.zeros((len(text_documents), self.image_dim))
        )
        
        # Metadata feature extraction
        self.metadata_features = (
            self._preprocess_metadata_features(metadata) 
            if metadata is not None 
            else np.zeros((len(text_documents), self.metadata_dim))
        )
        
        # Feature fusion
        if self.fusion_method == 'concatenate':
            multi_modal_features = np.hstack([
                self.text_features, 
                self.image_features, 
                self.metadata_features
            ])
        elif self.fusion_method == 'weighted_sum':
            # Example weighted fusion (adjust weights as needed)
            multi_modal_features = (
                0.5 * self.text_features + 
                0.3 * self.image_features + 
                0.2 * self.metadata_features
            )
        
        return multi_modal_features