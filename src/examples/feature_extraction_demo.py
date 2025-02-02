# src/examples/feature_extraction_demo.py
from feature_extraction.feature_extractor import FeatureExtractor

def main():
    # Sample documents
    documents = [
        "Machine learning is fascinating",
        "AI is transforming various industries",
        "Deep learning requires large datasets",
        "Neural networks simulate human brain"
    ]
    
    # Demonstrate different feature extraction methods
    methods = ['tfidf', 'count', 'lda', 'lsi']
    
    for method in methods:
        print(f"\n--- {method.upper()} Feature Extraction ---")
        extractor = FeatureExtractor(method=method)
        
        try:
            # Extract features
            features = extractor.extract_features(documents, n_features=10)
            print(f"Feature Matrix Shape: {features.shape}")
            
            # Get top features
            top_features = extractor.get_top_features(n_top=5)
            print("Top Features:")
            print(top_features)
        except Exception as e:
            print(f"Error with {method} method: {e}")

if __name__ == "__main__":
    main()