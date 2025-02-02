# src/examples/recommendation_performance.py
import time
import random
import string
import numpy as np
import statistics
import seaborn as sns

import matplotlib.pyplot as plt
from recommendation.content_recommender import ContentRecommender

def generate_sample_documents(num_docs=1000, max_words=50, seed=42):
    """
    Generate synthetic documents for performance testing with consistent randomness
    
    Args:
        num_docs (int): Number of documents to generate
        max_words (int): Maximum number of words in a document
        seed (int): Random seed for reproducibility
    
    Returns:
        list: Generated documents
    """
    random.seed(seed)
    
    def random_word():
        return ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
    
    documents = [
        ' '.join(random_word() for _ in range(random.randint(5, max_words)))
        for _ in range(num_docs)
    ]
    
    return documents

def performance_benchmark():
    """
    Comprehensive performance benchmark for recommendation system
    """
    # Performance metrics configuration
    metrics = ['cosine', 'euclidean']
    corpus_sizes = [100, 500, 1000, 5000]
    cache_strategies = ['lru', 'lfu', 'fifo']
    
    # Detailed performance results storage
    performance_results = {}
    
    print("Recommendation System Performance Benchmark")
    print("-" * 50)
    
    for strategy in cache_strategies:
        performance_results[strategy] = {}
        
        for metric in metrics:
            performance_results[strategy][metric] = {}
            
            for corpus_size in corpus_sizes:
                documents = generate_sample_documents(num_docs=corpus_size)
                recommender = ContentRecommender(
                    n_features=100, 
                    similarity_metric=metric,
                    cache_size=50,
                    cache_strategy=strategy,
                    ml_cache_prediction=True
                )
                recommender.add_documents(documents)
                
                # Performance measurement variables
                addition_times = []
                recommendation_times = []
                recommendation_counts = []
                cache_hit_rates = []
                
                # Multiple runs for more reliable statistics
                num_runs = 5
                for _ in range(num_runs):
                    # Initialize recommender
                    recommender = ContentRecommender(
                        n_features=100, 
                        similarity_metric=metric,
                        cache_size=50,
                        cache_strategy=strategy,
                        ml_cache_prediction=True
                    )
                    recommender.add_documents(documents)
                    
                    # Measure document addition time
                    start_time = time.time()
                    recommender.add_documents(documents)
                    addition_time = time.time() - start_time
                    addition_times.append(addition_time)
                    
                    # Simulate repeated queries for cache hit rate
                    cache_hits = 0
                    total_queries = 10
                    
                    for _ in range(total_queries):
                        query = random.choice(documents)
                        
                        # First query
                        first_recommendations = recommender.recommend_documents(query)
                        
                        # Repeated query
                        repeated_recommendations = recommender.recommend_documents(query)
                        
                        # Check cache hit
                        if first_recommendations == repeated_recommendations:
                            cache_hits += 1
                    
                    cache_hit_rate = cache_hits / total_queries
                    cache_hit_rates.append(cache_hit_rate)
                    
                    # Measure recommendation time
                    start_time = time.time()
                    recommendations = recommender.recommend_documents(
                        random.choice(documents), 
                        top_n=5, 
                        min_similarity=0.01
                    )
                    recommendation_time = time.time() - start_time
                    recommendation_times.append(recommendation_time)
                    recommendation_counts.append(len(recommendations))
                
                # Store results
                performance_results[strategy][metric][corpus_size] = {
                    'addition_time': {
                        'mean': statistics.mean(addition_times),
                        'std_dev': statistics.stdev(addition_times)
                    },
                    'recommendation_time': {
                        'mean': statistics.mean(recommendation_times),
                        'std_dev': statistics.stdev(recommendation_times)
                    },
                    'recommendation_count': {
                        'mean': statistics.mean(recommendation_counts),
                        'std_dev': statistics.stdev(recommendation_counts)
                    },
                    'cache_hit_rate': {
                        'mean': statistics.mean(cache_hit_rates),
                        'std_dev': statistics.stdev(cache_hit_rates)
                    }
                }
    
    # Print detailed results
    print("\n=== Performance Results ===")
    for strategy, strategy_results in performance_results.items():
        print(f"\nCache Strategy: {strategy}")
        for metric, metric_results in strategy_results.items():
            print(f"\nSimilarity Metric: {metric}")
            for corpus_size, results in metric_results.items():
                print(f"\nCorpus Size: {corpus_size}")
                print(f"Document Addition Time: {results['addition_time']['mean']:.4f} ± {results['addition_time']['std_dev']:.4f} seconds")
                print(f"Recommendation Time: {results['recommendation_time']['mean']:.4f} ± {results['recommendation_time']['std_dev']:.4f} seconds")
                print(f"Number of Recommendations: {results['recommendation_count']['mean']:.1f} ± {results['recommendation_count']['std_dev']:.1f}")
                print(f"Cache Hit Rate: {results['cache_hit_rate']['mean'] * 100:.2f}% ± {results['cache_hit_rate']['std_dev'] * 100:.2f}%")
    
    return performance_results

def complexity_analysis():
    """
    Analyze computational complexity of recommendation system
    """
    print("\n=== Computational Complexity Analysis ===")
    
    # Test increasing corpus sizes to analyze time complexity
    corpus_sizes = [100, 500, 1000, 2000, 5000, 10000]
    metrics = ['cosine', 'euclidean']
    
    for metric in metrics:
        print(f"\nSimilarity Metric: {metric}")
        print("Corpus Size | Addition Time | Recommendation Time")
        print("-" * 50)
        
        for size in corpus_sizes:
            documents = generate_sample_documents(num_docs=size)
            recommender = ContentRecommender(
                n_features=100, 
                similarity_metric=metric
            )
            
            # Measure document addition time
            start_time = time.time()
            recommender.add_documents(documents)
            addition_time = time.time() - start_time
            
            # Measure recommendation time
            query = random.choice(documents)
            start_time = time.time()
            recommender.recommend_documents(query, top_n=5, min_similarity=0.01)
            recommendation_time = time.time() - start_time
            
            print(f"{size:10} | {addition_time:12.4f} | {recommendation_time:16.4f}")

def plot_performance_metrics(metrics_data):
    """
    Advanced visualization of recommendation system performance metrics
    
    Args:
        metrics_data (dict): Performance metrics from benchmark
    """
    plt.figure(figsize=(20, 15))
    plt.suptitle('Recommendation System Performance Analysis', fontsize=16, fontweight='bold')
    
    # Subplot configurations with enhanced details
    subplot_configs = [
        ('Document Addition Time', 'addition_time', 'Time (seconds)', True),
        ('Recommendation Time', 'recommendation_time', 'Time (seconds)', True),
        ('Recommendations Count', 'recommendation_count', 'Number of Recommendations', False),
        ('Cache Hit Rate', 'cache_hit_rate', 'Hit Rate (%)', False)
    ]
    
    for i, (title, metric_key, ylabel, use_log_scale) in enumerate(subplot_configs, 1):
        plt.subplot(2, 2, i)
        
        for strategy, strategy_data in metrics_data.items():
            for similarity_metric, metric_results in strategy_data.items():
                corpus_sizes = list(metric_results.keys())
                
                # Adjust metric values based on the specific metric
                if metric_key == 'cache_hit_rate':
                    metric_values = [
                        results[metric_key]['mean'] * 100
                        for results in metric_results.values()
                    ]
                else:
                    metric_values = [
                        results[metric_key]['mean']
                        for results in metric_results.values()
                    ]
                
                # Error bars representing standard deviation
                std_devs = [
                    results[metric_key]['std_dev'] * (100 if metric_key == 'cache_hit_rate' else 1)
                    for results in metric_results.values()
                ]
                
                # Plot with error bars
                plt.errorbar(
                    corpus_sizes, 
                    metric_values, 
                    yerr=std_devs,
                    marker='o', 
                    capsize=5,
                    label=f'{strategy} - {similarity_metric}',
                    linestyle='-',
                    markersize=8,
                    alpha=0.7
                )
        
        plt.title(title, fontweight='bold')
        plt.xlabel('Corpus Size', fontweight='bold')
        plt.ylabel(ylabel, fontweight='bold')
        
        # Logarithmic scale for time-based metrics
        if use_log_scale:
            plt.yscale('log')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('recommendation_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create a separate heatmap for performance comparison
    plt.figure(figsize=(15, 10))
    plt.title('Performance Metrics Heatmap', fontsize=16, fontweight='bold')
    
    # Prepare data for heatmap
    strategies = list(metrics_data.keys())
    metrics_list = ['addition_time', 'recommendation_time', 'recommendation_count', 'cache_hit_rate']
    
    heatmap_data = np.zeros((len(strategies), len(metrics_list)))
    
    for i, strategy in enumerate(strategies):
        for j, metric in enumerate(metrics_list):
            # Average performance across corpus sizes and similarity metrics
            values = [
                results[metric]['mean']
                for similarity_metric_results in metrics_data[strategy].values()
                for results in similarity_metric_results.values()
            ]
            heatmap_data[i, j] = np.mean(values)
    
    # Safe normalization with handling for constant or zero values
    def safe_normalize(column):
        col_min, col_max = column.min(), column.max()
        if col_min == col_max:
            return np.zeros_like(column)
        return (column - col_min) / (col_max - col_min)
    
    # Normalize each column separately
    heatmap_data_normalized = np.apply_along_axis(safe_normalize, 0, heatmap_data)
    
    sns.heatmap(
        heatmap_data_normalized, 
        annot=np.round(heatmap_data, 4), 
        cmap='YlGnBu', 
        xticklabels=metrics_list, 
        yticklabels=strategies,
        fmt='.4f'
    )
    
    plt.tight_layout()
    plt.savefig('recommendation_performance_heatmap.png', dpi=300)
    plt.close()

    print("Performance visualization completed. Check 'recommendation_performance_analysis.png' and 'recommendation_performance_heatmap.png'.")

# Update main block
if __name__ == "__main__":
    performance_results = performance_benchmark()
    complexity_analysis()
    plot_performance_metrics(performance_results)