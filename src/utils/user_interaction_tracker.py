from typing import List, Dict, Any
from collections import defaultdict
import uuid
from datetime import datetime

class UserInteractionTracker:
    def __init__(self, max_history_size: int = 100):
        """
        Track user interactions with recommendations
        
        Args:
            max_history_size (int): Maximum number of interactions to store per user
        """
        self.user_interactions = defaultdict(list)
        self.max_history_size = max_history_size
    
    def log_recommendation(self, 
                            user_id: str, 
                            query: str, 
                            recommendations: List[str],
                            interaction_type: str = 'view'):
        """
        Log a recommendation interaction
        
        Args:
            user_id (str): User identifier
            query (str): Original query
            recommendations (List[str]): Recommended documents
            interaction_type (str): Type of interaction
        """
        interaction = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now(),
            'query': query,
            'recommendations': recommendations,
            'type': interaction_type,
            'relevance_score': 0.5  # Default score, can be updated later
        }
        
        # Maintain max history size
        if len(self.user_interactions[user_id]) >= self.max_history_size:
            self.user_interactions[user_id].pop(0)
        
        self.user_interactions[user_id].append(interaction)
    
    def get_user_interactions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve user interaction history
        
        Args:
            user_id (str): User identifier
        
        Returns:
            List of interaction records
        """
        return self.user_interactions.get(user_id, [])