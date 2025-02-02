# debug_import.py
import sys
import os

def print_python_path():
    print("Python Path:")
    for path in sys.path:
        print(f"  - {path}")

def check_module_import():
    try:
        # Try different import methods
        print("Attempting imports:")
        
        print("\n1. Direct import:")
        from src.preprocessing.text_preprocessor import TextPreprocessor
        print("Direct import successful!")
        
        print("\n2. Relative import:")
        from preprocessing.text_preprocessor import TextPreprocessor
        print("Relative import successful!")
    
    except ImportError as e:
        print(f"Import Error: {e}")
        print("\nCurrent Working Directory:", os.getcwd())
        print_python_path()

if __name__ == "__main__":
    check_module_import()