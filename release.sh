#!/bin/bash
# Release script for AI Content Personalization

# Ensure you're on the main branch
git checkout main
git pull origin main

# Run tests
python -m pytest tests/

# Build documentation
sphinx-build docs docs/_build

# Build package
python -m build

# Show distribution files
ls dist/

echo "Ready for release. Next steps:"
echo "1. Commit changes"
echo "2. Create GitHub release"
echo "3. Upload to PyPI"
