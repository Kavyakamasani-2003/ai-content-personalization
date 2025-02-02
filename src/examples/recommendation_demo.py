import sys
import os


# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.recommendation.content_recommender import ContentRecommender
from src.utils.feature_engineering import AdvancedFeatureExtractor

def advanced_recommendation_demo():
    """
    Demonstrate advanced content recommendation with multiple feature extraction techniques
    """
    # Sample document corpus
    tech_documents = [
        "Machine learning transforms artificial intelligence",
        "Deep learning neural networks improve computer vision",
        "Natural language processing enables advanced text understanding",
        "Quantum computing revolutionizes computational capabilities",
        "Blockchain technology enhances secure digital transactions",
        "Artificial intelligence drives innovative software solutions",
        "Computer vision algorithms detect complex visual patterns",
        "Neural networks simulate human brain learning processes"
    ]

    # Demonstration of different feature extraction configurations
    feature_configs = [
        {
            'name': 'Basic TF-IDF',
            'use_svd': False,
            'n_features': 50
        },
        {
            'name': 'TF-IDF with SVD',
            'use_svd': True,
            'n_features': 20
        }
    ]

    # Similarity metrics to compare
    similarity_metrics = ['cosine', 'euclidean']

    # Comparative analysis
    print("Advanced Content Recommendation Demonstration\n")
    
    for config in feature_configs:
        for metric in similarity_metrics:
            print(f"\n=== {config['name']} | Similarity: {metric} ===")
            
            # Initialize recommender with specific configuration
            recommender = ContentRecommender(
                n_features=config['n_features'],
                similarity_metric=metric,
                use_svd=config['use_svd']
            )
            
            # Add documents to the recommender
            recommender.add_documents(tech_documents)
            
            # Example query documents
            query_docs = [
                "AI and machine learning innovations",
                "Advanced computational technologies"
            ]
            
            for query in query_docs:
                print(f"\nQuery: '{query}'")
                recommendations = recommender.recommend_documents(
                    query, 
                    top_n=3, 
                    min_similarity=0.1
                )
                
                print("Recommended Documents:")
                for idx, doc in enumerate(recommendations, 1):
                    print(f"{idx}. {doc}")

def performance_comparison():
    """
    Compare performance of different feature extraction techniques
    """
    from timeit import default_timer as timer
    
    tech_documents = [
        "Machine learning transforms artificial intelligence",
        "Deep learning neural networks improve computer vision",
        "Natural language processing enables advanced text understanding",
        "Quantum computing revolutionizes computational capabilities",
        "Blockchain technology enhances secure digital transactions"
    ]
    
    feature_extractors = [
        AdvancedFeatureExtractor(use_tfidf=True, use_svd=False),
        AdvancedFeatureExtractor(use_tfidf=True, use_svd=True)
    ]
    
    print("\nFeature Extraction Performance Comparison:")
    for extractor in feature_extractors:
        start_time = timer()
        features = extractor.extract_features(tech_documents)
        end_time = timer()
        
        print(f"\nExtractor Configuration:")
        print(f"Use SVD: {extractor.use_svd}")
        print(f"Feature Matrix Shape: {features.shape}")
        print(f"Extraction Time: {(end_time - start_time)*1000:.4f} ms")

if __name__ == "__main__":
    advanced_recommendation_demo()
    performance_comparison()