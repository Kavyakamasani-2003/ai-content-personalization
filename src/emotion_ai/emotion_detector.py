import nltk
from textblob import TextBlob
import numpy as np
from typing import Dict, Any, Optional

class EmotionDetector:
    def __init__(self):
        # Define emotion categories with their characteristics
        self.emotion_categories = {
            'joy': {
                'keywords': ['happy', 'excited', 'wonderful', 'great'],
                'intensity_range': (0.6, 1.0)
            },
            'sadness': {
                'keywords': ['sad', 'depressed', 'unhappy', 'lonely'],
                'intensity_range': (0.1, 0.4)
            },
            'anger': {
                'keywords': ['angry', 'frustrated', 'irritated', 'annoyed'],
                'intensity_range': (0.4, 0.7)
            },
            'fear': {
                'keywords': ['scared', 'worried', 'anxious', 'nervous'],
                'intensity_range': (0.2, 0.5)
            },
            'neutral': {
                'keywords': ['okay', 'fine', 'normal', 'average'],
                'intensity_range': (0.4, 0.6)
            }
        }

    def detect_emotion(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive emotion detection using multiple signals
        
        Args:
            user_data (Dict): User interaction and activity data
        
        Returns:
            Dict: Detected emotion with metadata
        """
        try:
            # Extract text interactions
            text_samples = user_data.get('text_interactions', [])
            
            # Sentiment analysis
            sentiment_score = self._analyze_sentiment(text_samples)
            
            # Emotion mapping
            primary_emotion = self._map_sentiment_to_emotion(sentiment_score)
            
            # Interaction-based emotion detection
            interaction_emotion = self._detect_emotion_from_interactions(user_data)
            
            # Combine emotion signals
            final_emotion = self._combine_emotion_signals(primary_emotion, interaction_emotion)
            
            # Emotion metadata
            emotion_result = {
                'emotion': final_emotion,
                'sentiment_score': sentiment_score,
                'detection_methods': {
                    'text_sentiment': primary_emotion,
                    'interaction_pattern': interaction_emotion
                },
                'confidence': self._calculate_emotion_confidence(sentiment_score)
            }
            
            return emotion_result
        
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return {'emotion': 'neutral', 'confidence': 0.5}

    def _analyze_sentiment(self, text_samples: list) -> float:
        """
        Advanced sentiment analysis using TextBlob
        
        Args:
            text_samples (list): List of text interactions
        
        Returns:
            float: Sentiment polarity score
        """
        if not text_samples:
            return 0.0
        
        sentiments = [TextBlob(str(text)).sentiment.polarity for text in text_samples]
        return float(np.mean(sentiments))

    def _map_sentiment_to_emotion(self, sentiment_score: float) -> str:
        """
        Map sentiment score to emotion categories
        
        Args:
            sentiment_score (float): Sentiment polarity score
        
        Returns:
            str: Emotion category
        """
        if sentiment_score > 0.5:
            return 'joy'
        elif sentiment_score < -0.5:
            return 'sadness'
        elif sentiment_score < -0.2:
            return 'anger'
        elif 0.2 > sentiment_score > -0.2:
            return 'neutral'
        else:
            return 'fear'

    def _detect_emotion_from_interactions(self, user_data: Dict[str, Any]) -> Optional[str]:
        """
        Detect emotion based on interaction patterns
        
        Args:
            user_data (Dict): User activity data
        
        Returns:
            Optional[str]: Emotion inferred from interactions
        """
        app_usage = user_data.get('app_usage', {})
        feature_usage = app_usage.get('feature_usage', {})
        social_activity = user_data.get('social_activity', {})
        
        # Emotion detection heuristics
        if feature_usage.get('search', 0) > 80:
            return 'fear'
        
        if social_activity.get('likes', 0) > 100:
            return 'joy'
        
        if social_activity.get('shares', 0) > 50:
            return 'excitement'
        
        return None

    def _combine_emotion_signals(self, sentiment_emotion: str, interaction_emotion: Optional[str]) -> str:
        """
        Combine emotion signals from different sources
        
        Args:
            sentiment_emotion (str): Emotion from sentiment analysis
            interaction_emotion (Optional[str]): Emotion from interaction patterns
        
        Returns:
            str: Final emotion category
        """
        emotion_priority = {
            'joy': 5,
            'anger': 4,
            'sadness': 3,
            'fear': 2,
            'neutral': 1
        }
        
        if interaction_emotion and emotion_priority.get(interaction_emotion, 0) > emotion_priority.get(sentiment_emotion, 0):
            return interaction_emotion
        
        return sentiment_emotion

    def _calculate_emotion_confidence(self, sentiment_score: float) -> float:
        """
        Calculate confidence of emotion detection
        
        Args:
            sentiment_score (float): Sentiment polarity score
        
        Returns:
            float: Emotion detection confidence
        """
        return min(abs(sentiment_score) * 2, 1.0)