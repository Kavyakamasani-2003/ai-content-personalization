# tests/test_text_preprocessor.py
import unittest
from src.preprocessing.text_preprocessor import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor()
    
    def test_tokenization(self):
        text = "Hello, world! How are you?"
        tokens = self.preprocessor.tokenize(text)
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)
    
    def test_lemmatization(self):
        tokens = ["running", "dogs", "better"]
        lemmatized = self.preprocessor.lemmatize(tokens)
        self.assertIsInstance(lemmatized, list)
        self.assertTrue("run" in lemmatized)
        self.assertTrue("dog" in lemmatized)
    
    def test_sentiment_analysis(self):
        positive_text = "This is amazing and wonderful!"
        negative_text = "I hate this terrible experience."
        neutral_text = "The weather is."
        
        positive_sentiment = self.preprocessor.analyze_sentiment(positive_text)
        negative_sentiment = self.preprocessor.analyze_sentiment(negative_text)
        neutral_sentiment = self.preprocessor.analyze_sentiment(neutral_text)
        
        self.assertIn('compound', positive_sentiment)
        self.assertTrue(positive_sentiment['compound'] > 0)
        self.assertTrue(negative_sentiment['compound'] < 0)
    
    def test_full_preprocessing(self):
        text = "The quick brown foxes are jumping over lazy dogs."
        result = self.preprocessor.preprocess(text)
        
        self.assertIn('tokens', result)
        self.assertIn('lemmatized_tokens', result)
        self.assertIn('sentiment', result)
        
        self.assertTrue(len(result['tokens']) > 0)
        self.assertTrue(len(result['lemmatized_tokens']) > 0)

if __name__ == '__main__':
    unittest.main()