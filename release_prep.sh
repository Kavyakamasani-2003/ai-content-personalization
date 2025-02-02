#!/bin/bash
# Release Preparation Script

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

# Suggest next steps
echo ""
echo "Release Preparation Checklist:"
echo "1. Review changes in CHANGELOG.md"
echo "2. Verify package contents in dist/"
echo "3. Create a new git tag"
echo "4. Push tag to trigger release workflow"
echo ""
echo "Suggested git tag command:"
echo "git tag -a v0.1.0 -m 'Release version 0.1.0'"
