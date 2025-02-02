@echo off
REM Release Preparation and Publication Script

REM Ensure we're on the main branch
git checkout main
git pull origin main

REM Run tests
echo Running tests...
python -m pytest tests/

REM Build documentation
echo Building documentation...
sphinx-build docs docs/_build

REM Build package
echo Building package...
python -m build

REM Show distribution files
echo Distribution files:
dir dist\

REM Suggest release tag
echo.
echo Suggested git tag command:
echo git tag -a v0.1.0 -m "Release version 0.1.0"
echo git push origin v0.1.0
