# conftest.py
import os
import sys

# Get the absolute path of the project root
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')

# Add project root and src directory to Python path
if SRC_PATH not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, SRC_PATH)