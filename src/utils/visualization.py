import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class RecommendationVisualizer:
    @staticmethod
    def plot_feature_weights(feature_weights: dict, title='Feature Weights'):
        """
        Create a bar plot of feature weights
        
        Args:
            feature_weights (dict): Dictionary of feature names and their weights
            title (str): Plot title
        """
        plt.figure(figsize=(12, 6))
        features = list(feature_weights.keys())
        weights = list(feature_weights.values())
        
        sns.barplot(x=features, y=weights)
        plt.title(title)
        plt.xlabel('Features')
        plt.ylabel('Weight')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_performance_metrics(performance_summary: dict):
        """
        Visualize performance metrics
        
        Args:
            performance_summary (dict): Performance metrics from recommender
        """
        methods = list(performance_summary.keys())
        avg_times = [metrics['avg_time'] for metrics in performance_summary.values()]
        total_calls = [metrics['total_calls'] for metrics in performance_summary.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Average Time Plot
        sns.barplot(x=methods, y=avg_times, ax=ax1)
        ax1.set_title('Average Execution Time')
        ax1.set_xlabel('Methods')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Total Calls Plot
        sns.barplot(x=methods, y=total_calls, ax=ax2)
        ax2.set_title('Total Method Calls')
        ax2.set_xlabel('Methods')
        ax2.set_ylabel('Number of Calls')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()