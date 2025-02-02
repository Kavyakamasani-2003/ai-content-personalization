#!/bin/bash
# Release Preparation and Publication Script

# Ensure we're on the main branch
git checkout main
git pull origin main

# Run tests
echo "Running tests..."
python -m pytest tests/

# Build documentation
echo "Building documentation..."
sphinx-build docs docs/_build

# Build package
echo "Building package..."
python -m build

# Show distribution files
echo "Distribution files:"
ls dist/

# Suggest release tag
echo ""
echo "Suggested git tag command:"
echo "git tag -a v0.1.0 -m 'Release version 0.1.0'"
echo "git push origin v0.1.0"
