# src/preprocessing/text_preprocessor.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

class TextPreprocessor:
    def __init__(self):
        # Initialize lemmatizer and sentiment analyzer
        self.lemmatizer = WordNetLemmatizer()
        self.sia = SentimentIntensityAnalyzer()
    
    def tokenize(self, text):
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def get_wordnet_pos(self, treebank_tag):
        """Convert treebank POS tag to WordNet POS tag"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def lemmatize(self, tokens):
        """Lemmatize tokens with their appropriate POS tags"""
        # Get POS tags
        pos_tags = pos_tag(tokens)
        
        # Lemmatize with appropriate POS
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos)) 
            for word, pos in pos_tags
        ]
        
        return lemmatized_tokens
    
    def analyze_sentiment(self, text):
        """Perform sentiment analysis on text"""
        return self.sia.polarity_scores(text)
    
    def preprocess(self, text):
        """Comprehensive text preprocessing pipeline"""
        # Tokenize
        tokens = self.tokenize(text)
        
        # Lemmatize
        lemmatized_tokens = self.lemmatize(tokens)
        
        # Sentiment analysis
        sentiment = self.analyze_sentiment(text)
        
        return {
            'tokens': tokens,
            'lemmatized_tokens': lemmatized_tokens,
            'sentiment': sentiment
        }