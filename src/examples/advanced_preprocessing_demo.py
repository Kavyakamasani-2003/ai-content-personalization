# src/examples/advanced_preprocessing_demo.py
from preprocessing.text_preprocessor import TextPreprocessor

def analyze_multiple_texts():
    preprocessor = TextPreprocessor()
    
    texts = [
        "I absolutely love this amazing product!",
        "This is the worst experience ever.",
        "The weather is quite neutral today."
    ]
    
    for text in texts:
        print("\n--- Text Analysis ---")
        print("Original Text:", text)
        
        result = preprocessor.preprocess(text)
        
        print("\nTokens:", result['tokens'])
        print("Lemmatized Tokens:", result['lemmatized_tokens'])
        print("Sentiment Analysis:")
        for key, value in result['sentiment'].items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    analyze_multiple_texts()