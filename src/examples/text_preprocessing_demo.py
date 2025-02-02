# src/examples/text_preprocessing_demo.py
from preprocessing.text_preprocessor import TextPreprocessor

def main():
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Sample text
    sample_text = "The quick brown foxes are jumping over lazy dogs. This is an amazing and exciting example!"
    
    # Preprocess text
    result = preprocessor.preprocess(sample_text)
    
    # Print results
    print("Original Text:", sample_text)
    print("\nTokens:", result['tokens'])
    print("\nLemmatized Tokens:", result['lemmatized_tokens'])
    print("\nSentiment Analysis:", result['sentiment'])

if __name__ == "__main__":
    main()