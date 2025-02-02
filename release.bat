@echo off
REM Release script for AI Content Personalization

REM Ensure you're on the main branch
git checkout main
git pull origin main

REM Run tests
python -m pytest tests/

REM Build documentation
sphinx-build docs docs/_build

REM Build package
python -m build

REM Show distribution files
dir dist\

echo Ready for release. Next steps:
echo 1. Commit changes
echo 2. Create GitHub release
echo 3. Upload to PyPI
