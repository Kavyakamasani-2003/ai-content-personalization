import numpy as np
import json
from typing import Dict, Any, Optional
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import random

class MultimodalContentCreator:
    def __init__(self):
        # Initialize content generation modes
        self.generation_modes = [
            'text_generation',
            'image_synthesis',
            'video_creation',
            'audio_generation'
        ]
        
        # Load pre-trained language model
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            self.language_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        except Exception as e:
            print(f"Language model loading error: {e}")
            self.tokenizer = None
            self.language_model = None
        
        # Content templates for different emotions
        self.emotion_content_templates = {
            'joy': [
                "Exciting news about {topic}!",
                "Celebrating the amazing world of {topic}",
                "Incredible breakthrough in {topic}"
            ],
            'sadness': [
                "Reflecting on challenges in {topic}",
                "Understanding the complexities of {topic}",
                "Finding hope amidst {topic}"
            ],
            'anger': [
                "Critical insights about {topic}",
                "Addressing issues in {topic}",
                "Challenging perspectives on {topic}"
            ],
            'fear': [
                "Navigating uncertainties in {topic}",
                "Understanding risks in {topic}",
                "Strategies for managing {topic}"
            ],
            'neutral': [
                "Exploring {topic} objectively",
                "Balanced perspectives on {topic}",
                "Comprehensive overview of {topic}"
            ]
        }

    def generate_content(self, recommendation_data: Dict[str, Any], emotion_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate multimodal content based on recommendations and emotional context
        
        Args:
            recommendation_data (Dict): Recommended content data
            emotion_context (Optional[Dict]): Emotional context of the user
        
        Returns:
            Dict: Generated multimodal content
        """
        try:
            # Extract content categories from recommendations
            content_categories = recommendation_data.get('recommended_content', [])
            
            # Determine emotion for content generation
            emotion = emotion_context.get('emotion', 'neutral') if emotion_context else 'neutral'
            
            # Generate different content types
            generated_content = {
                'text_content': self._generate_text_content(content_categories, emotion),
                'visual_content': self._generate_visual_content(content_categories),
                'metadata': self._add_content_metadata(recommendation_data, emotion)
            }
            
            return generated_content
        
        except Exception as e:
            print(f"Content generation error: {e}")
            return self._fallback_content()

    def _generate_text_content(self, categories: list, emotion: str) -> str:
        """
        Generate text content using GPT-2 or fallback method
        
        Args:
            categories (list): Content categories
            emotion (str): Emotional context
        
        Returns:
            str: Generated text content
        """
        if not categories:
            categories = ['technology']
        
        topic = random.choice(categories)
        template = random.choice(self.emotion_content_templates.get(emotion, self.emotion_content_templates['neutral']))
        
        # Use GPT-2 if available, otherwise use template
        if self.language_model and self.tokenizer:
            try:
                input_text = template.format(topic=topic)
                input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
                
                output = self.language_model.generate(
                    input_ids, 
                    max_length=100, 
                    num_return_sequences=1,
                    temperature=0.7
                )
                
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                return generated_text
            except Exception as e:
                print(f"GPT-2 generation error: {e}")
        
        # Fallback to template
        return template.format(topic=topic)

    def _generate_visual_content(self, categories: list) -> Dict[str, str]:
        """
        Simulate visual content generation
        
        Args:
            categories (list): Content categories
        
        Returns:
            Dict: Visual content URLs/paths
        """
        return {
            'image_url': f"generated_image_{random.choice(categories)}_{np.random.randint(1, 1000)}.jpg",
            'video_url': f"generated_video_{random.choice(categories)}_{np.random.randint(1, 500)}.mp4"
        }

    def _add_content_metadata(self, recommendation_data: Dict[str, Any], emotion: str) -> Dict[str, Any]:
        """
        Add metadata to generated content
        
        Args:
            recommendation_data (Dict): Recommendation data
            emotion (str): Emotional context
        
        Returns:
            Dict: Content metadata
        """
        return {
            'generation_timestamp': np.random.randint(1000000000, 9999999999),
            'emotion_context': emotion,
            'confidence_scores': recommendation_data.get('confidence_scores', {}),
            'content_tags': recommendation_data.get('recommended_content', []),
            'content_complexity': np.random.uniform(0.3, 0.9)
        }

    def _fallback_content(self) -> Dict[str, Any]:
        """
        Provide fallback content in case of generation failure
        
        Returns:
            Dict: Fallback multimodal content
        """
        return {
            'text_content': "Personalized content could not be generated.",
            'visual_content': {
                'image_url': 'fallback_image.jpg',
                'video_url': 'fallback_video.mp4'
            },
            'metadata': {
                'generation_status': 'fallback',
                'timestamp': np.random.randint(1000000000, 9999999999)
            }
        }