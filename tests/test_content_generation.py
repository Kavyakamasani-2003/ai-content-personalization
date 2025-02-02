import unittest
from src.content_generation.multimodal_content_creator import MultimodalContentCreator
from src.personalization_engine.recommendation_optimizer import RecommendationOptimizer
from src.emotion_ai.emotion_detector import EmotionDetector

class TestContentGeneration(unittest.TestCase):
    def setUp(self):
        self.content_creator = MultimodalContentCreator()
        self.recommendation_optimizer = RecommendationOptimizer()
        self.emotion_detector = EmotionDetector()

    def test_content_generation(self):
        # Simulate user data
        user_data = {
            'web_interactions': {
                'content_categories': ['technology', 'science']
            }
        }

        # Generate recommendations
        recommendations = self.recommendation_optimizer.optimize(user_data)
        
        # Detect emotion
        emotion_context = self.emotion_detector.detect_emotion(user_data)
        
        # Generate content
        generated_content = self.content_creator.generate_content(
            recommendations, 
            emotion_context
        )

        # Assertions
        self.assertIn('text_content', generated_content)
        self.assertIn('visual_content', generated_content)
        self.assertIn('metadata', generated_content)

if __name__ == '__main__':
    unittest.main()