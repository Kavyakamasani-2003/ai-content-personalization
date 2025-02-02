import nltk
import os
import sys

def detailed_nltk_verification():
    """Perform detailed verification of NLTK resources"""
    print("Detailed NLTK Resource Verification")
    print("-" * 40)

    # Resources to verify
    resources = [
        ('punkt', 'tokenizers/punkt'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
        ('wordnet', 'corpora/wordnet'),
        ('vader_lexicon', 'sentiment/vader_lexicon')
    ]

    # Print NLTK data paths
    print("\nNLTK Data Paths:")
    for path in nltk.data.path:
        print(f"  - {path}")
        if not os.path.exists(path):
            print(f"    ⚠️ Path does not exist!")

    print("\nResource Verification:")
    for resource_name, resource_path in resources:
        try:
            # Attempt to find the resource
            found_path = nltk.data.find(resource_path)
            
            if found_path:
                print(f"{resource_name}:")
                print(f"  - Found at: {found_path}")
                
                # Detailed path checks
                if os.path.exists(found_path):
                    print(f"  - Path Exists: Yes")
                    print(f"  - Is Directory: {os.path.isdir(found_path)}")
                    print(f"  - Is File: {os.path.isfile(found_path)}")
                    
                    # List contents if it's a directory
                    if os.path.isdir(found_path):
                        try:
                            contents = os.listdir(found_path)
                            print(f"  - Directory Contents: {contents}")
                        except Exception as list_err:
                            print(f"  - Unable to list directory: {list_err}")
                else:
                    print(f"  - ⚠️ Path does not actually exist!")
            else:
                print(f"{resource_name}: ✗ Not found")
        
        except Exception as e:
            print(f"{resource_name}: ✗ Verification Error - {e}")

def main():
    detailed_nltk_verification()

if __name__ == "__main__":
    main()