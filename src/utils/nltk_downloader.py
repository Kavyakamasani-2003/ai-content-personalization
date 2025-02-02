import nltk
import os
import ssl
import sys
import shutil
import urllib.request
import traceback

def setup_nltk_data_directory():
    """Create a centralized NLTK data directory for the project"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    nltk_data_path = os.path.join(project_root, 'nltk_data')
    
    # Create directory if it doesn't exist
    os.makedirs(nltk_data_path, exist_ok=True)
    
    return nltk_data_path

def configure_nltk_download():
    """Configure NLTK download settings"""
    # Fix SSL certificate issue for downloading
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

def manual_nltk_download(resource, download_dir):
    """
    Manually download NLTK resources using multiple methods
    
    Args:
        resource (str): Name of the NLTK resource to download
        download_dir (str): Directory to download the resource
    
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        # Ensure download directory exists
        os.makedirs(download_dir, exist_ok=True)
        
        # Resource download URLs (alternative sources)
        resource_urls = {
            'punkt': [
                'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip',
                'https://github.com/nltk/nltk_data/raw/gh-pages/packages/tokenizers/punkt.zip'
            ],
            'averaged_perceptron_tagger': [
                'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip',
                'https://github.com/nltk/nltk_data/raw/gh-pages/packages/taggers/averaged_perceptron_tagger.zip'
            ],
            'wordnet': [
                'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip',
                'https://github.com/nltk/nltk_data/raw/gh-pages/packages/corpora/wordnet.zip'
            ],
            'vader_lexicon': [
                'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip',
                'https://github.com/nltk/nltk_data/raw/gh-pages/packages/sentiment/vader_lexicon.zip'
            ]
        }

        if resource not in resource_urls:
            print(f"No manual download URL for {resource}")
            return False

        # Try multiple download methods
        for url in resource_urls.get(resource, []):
            try:
                # Create resource-specific directory
                resource_dir = os.path.join(download_dir, resource)
                os.makedirs(resource_dir, exist_ok=True)
                
                # Download zip file
                zip_path = os.path.join(resource_dir, f'{resource}.zip')
                
                print(f"Attempting to download {resource} from {url}")
                urllib.request.urlretrieve(url, zip_path)
                
                # Extract zip file
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(resource_dir)
                
                # Remove zip file
                os.remove(zip_path)
                
                print(f"{resource}: ✓ Manually downloaded")
                return True
            
            except Exception as url_error:
                print(f"{resource}: ✗ Download from {url} failed - {url_error}")
        
        return False
    
    except Exception as e:
        print(f"{resource}: ✗ Manual download failed - {e}")
        return False

def force_nltk_download(resource, download_dir):
    """
    Force download of NLTK resources with comprehensive error handling
    
    Args:
        resource (str): Name of the NLTK resource to download
        download_dir (str): Directory to download the resource
    
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        # Ensure download directory exists
        os.makedirs(download_dir, exist_ok=True)
        
        # Set NLTK data path
        if download_dir not in nltk.data.path:
            nltk.data.path.append(download_dir)
        
        # Verbose download with error tracking
        print(f"Attempting to download {resource}...")
        
        # Multiple download attempts
        download_methods = [
            lambda: nltk.download(resource, download_dir=download_dir),
            lambda: nltk.download(resource),
            lambda: manual_nltk_download(resource, download_dir)
        ]
        
        for method in download_methods:
            try:
                method()
                print(f"{resource}: ✓ Download initiated")
                
                # Verify download
                try:
                    resource_path = nltk.data.find(resource)
                    if resource_path:
                        print(f"{resource}: ✓ Successfully verified at {resource_path}")
                        return True
                except Exception:
                    pass
            
            except Exception as download_error:
                print(f"{resource}: ✗ Download method failed - {download_error}")
        
        # Final fallback
        return manual_nltk_download(resource, download_dir)
    
    except Exception as e:
        print(f"{resource}: ✗ Unexpected error - {e}")
        traceback.print_exc()
        return False

def download_nltk_resources(nltk_data_path):
    """Download NLTK resources with comprehensive error handling"""
    # Resources to download
    resources = [
        'punkt', 
        'averaged_perceptron_tagger', 
        'wordnet', 
        'vader_lexicon'
    ]

    print("Downloading NLTK Resources:")
    for resource in resources:
        success = force_nltk_download(resource, nltk_data_path)
        
        if not success:
            print(f"{resource}: ❌ Download failed completely")

def verify_nltk_resources(nltk_data_path):
    """Comprehensive verification of NLTK resources"""
    print("\nNLTK Resource Verification:")
    
    # Resources to verify
    resources = [
        ('punkt', 'tokenizers/punkt'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
        ('wordnet', 'corpora/wordnet'),
        ('vader_lexicon', 'sentiment/vader_lexicon')
    ]

    # Add project-specific NLTK data path
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)

    for resource_name, resource_path in resources:
        try:
            # Attempt to find the resource
            found_path = None
            
            try:
                found_path = nltk.data.find(resource_path)
            except LookupError:
                # Try alternative paths
                alternative_paths = [
                    os.path.join(nltk_data_path, resource_name),
                    os.path.join(nltk_data_path, resource_path)
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        found_path = alt_path
                        break
            
            if found_path:
                print(f"{resource_name}:")
                print(f"  - Found at: {found_path}")
                
                # Detailed path checks
                if os.path.exists(found_path):
                    print(f"  - Path Exists: Yes")
                    print(f"  - Is Directory: {os.path.isdir(found_path)}")
                    print(f"  - Is File: {os.path.isfile(found_path)}")
                else:
                    print(f"  - ⚠️ Path does not actually exist!")
                    force_nltk_download(resource_name, nltk_data_path)
            else:
                print(f"{resource_name}: ✗ Not found - Attempting download")
                force_nltk_download(resource_name, nltk_data_path)
        
        except Exception as e:
            print(f"{resource_name}: ✗ Verification Error - {e}")
            traceback.print_exc()

def main():
    try:
        # Configure download settings
        configure_nltk_download()
        
        # Setup NLTK data directory
        nltk_data_path = setup_nltk_data_directory()
        
        # Print current NLTK data paths
        print("Current NLTK Data Paths:")
        for path in nltk.data.path:
            print(f"  - {path}")
        
        # Download resources
        download_nltk_resources(nltk_data_path)
        
        # Verify resources
        verify_nltk_resources(nltk_data_path)
        
        # Additional system information
        print("\nSystem Information:")
        print(f"Python Version: {sys.version}")
        print(f"NLTK Version: {nltk.__version__}")
        print(f"NLTK Data Directory: {nltk_data_path}")
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        print("Please check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()