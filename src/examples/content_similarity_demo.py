# src/examples/content_similarity_demo.py
from similarity.content_similarity import ContentSimilarityAnalyzer

def main():
    # Sample documents
    documents = [
        "Machine learning is fascinating",
        "AI is transforming various industries",
        "Deep learning requires large datasets",
        "Neural networks simulate human brain"
    ]
    
    # Initialize analyzer
    analyzer = ContentSimilarityAnalyzer()
    
    # Compute similarity matrix
    similarity_matrix = analyzer.compute_similarity(documents)
    print("Similarity Matrix:")
    print(similarity_matrix)
    
    # Find most similar documents to first document
    similar_docs = analyzer.find_most_similar(documents, target_doc_index=0, top_n=2)
    
    print("\nMost similar documents to first document:")
    for idx in similar_docs:
        print(f"- {documents[idx]}")

if __name__ == "__main__":
    main()