# src/examples/advanced_recommendation_demo.py
import logging
from recommendation.content_recommender import ContentRecommender

def advanced_recommendation_demo():
    # Comprehensive document corpus
    documents = [
        "Machine learning is transforming artificial intelligence in various industries",
        "Deep learning requires large neural network models with complex architectures",
        "AI is revolutionizing research and development with advanced algorithms",
        "Neural networks simulate human brain processing and cognitive functions",
        "Data science uses statistical methods and machine learning for insights",
        "Big data analytics helps in decision making and predictive modeling",
        "Artificial intelligence improves automation in manufacturing and healthcare",
        "Natural language processing enables advanced text understanding",
        "Computer vision applications are expanding in robotics and autonomous systems",
        "Reinforcement learning creates intelligent agents for complex problem solving"
    ]
    
    # Demonstrate different similarity metrics
    similarity_metrics = ['cosine', 'euclidean']
    
    for metric in similarity_metrics:
        print(f"\n=== Recommendation Demonstration (Metric: {metric}) ===")
        
        # Create recommender with specific similarity metric
        recommender = ContentRecommender(
            n_features=15, 
            similarity_metric=metric
        )
        recommender.add_documents(documents)
        
        # Query documents with varying complexity
        query_documents = [
            "AI technologies in modern industries",
            "Machine learning and neural network applications",
            "Advanced data analysis techniques"
        ]
        
        for query in query_documents:
            print(f"\n--- Query: {query} ---")
            
            # Recommendations with different parameters
            recommendation_strategies = [
                {"top_n": 3, "min_similarity": 0.1},
                {"top_n": 2, "min_similarity": 0.2}
            ]
            
            for strategy in recommendation_strategies:
                print(f"\nRecommendation Strategy: top_n={strategy['top_n']}, min_similarity={strategy['min_similarity']}")
                recommendations = recommender.recommend_documents(
                    query, 
                    top_n=strategy['top_n'], 
                    min_similarity=strategy['min_similarity']
                )
                
                for idx, doc in enumerate(recommendations, 1):
                    print(f"{idx}. {doc}")
            
            # Print extracted features
            print("\nExtracted Features:")
            print(recommender.vectorizer.get_feature_names_out())

if __name__ == "__main__":
    advanced_recommendation_demo()