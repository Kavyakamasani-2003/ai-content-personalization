# Release Preparation Guide for AI Content Personalization

## Pre-Release Checklist

### Code Quality
- [ ] All tests pass
- [ ] Documentation is up to date
- [ ] Code is linted and type-checked

### Version Update
- [ ] Update version in:
  - pyproject.toml
  - src\__init__.py
  - CHANGELOG.md

### Release Steps
1. Run tests
   \\\powershell
   python -m pytest tests/
   \\\

2. Lint code
   \\\powershell
   flake8 src\ tests\
   \\\

3. Type checking
   \\\powershell
   mypy src\
   \\\

4. Build documentation
   \\\powershell
   sphinx-build docs docs\_build
   \\\

5. Build package
   \\\powershell
   python -m build
   \\\

6. Create git tag
   \\\powershell
   git tag -a v0.1.0 -m 'Release version 0.1.0'
   git push origin v0.1.0
   \\\

## Detailed Release Process

### Preparation
1. **Verify Current Branch**
   \\\powershell
   git checkout main
   git pull origin main
   \\\

2. **Run Comprehensive Tests**
   \\\powershell
   # Unit tests
   python -m pytest tests/

   # Coverage report
   pytest --cov=src --cov-report=xml
   \\\

3. **Code Quality Checks**
   \\\powershell
   # Linting
   flake8 src\ tests\

   # Type checking
   mypy src\
   \\\

### Version Update
1. **Update Version Files**
   - Update pyproject.toml:
     \\\	oml
     version = '0.1.0'
     \\\
   
   - Update src\__init__.py:
     \\\python
     __version__ = '0.1.0'
     \\\

   - Update CHANGELOG.md with release notes

2. **Commit Version Changes**
   \\\powershell
   git add pyproject.toml src\__init__.py CHANGELOG.md
   git commit -m 'Prepare for v0.1.0 release'
   \\\

### Package Build and Verification
1. **Build Distribution**
   \\\powershell
   # Build wheel and source distribution
   python -m build
   \\\

2. **Verify Package**
   \\\powershell
   # Check distribution files
   dir dist\

   # Verify package metadata
   twine check dist\*
   \\\

### Release Creation
1. **Create Git Tag**
   \\\powershell
   git tag -a v0.1.0 -m 'Release version 0.1.0'
   git push origin v0.1.0
   \\\

2. **GitHub Release**
   - Triggered automatically by tag
   - Verify release artifacts
   - Check GitHub Actions workflow

### Post-Release Tasks
1. **PyPI Verification**
   - Check PyPI page
   - Verify package is published correctly

2. **Documentation**
   - Update project documentation
   - Verify Read the Docs build

3. **Announcement**
   - Update project website
   - Announce in relevant community channels

## Troubleshooting

### If Release Fails
1. **Revert Tag**
   \\\powershell
   git tag -d v0.1.0
   git push --delete origin v0.1.0
   \\\

2. **Fix Issues**
   - Address test failures
   - Correct linting problems
   - Resolve type checking errors

3. **Retry Release**

## Best Practices
- Always release from main branch
- Ensure all tests pass
- Use semantic versioning
- Maintain clear changelog
- Document all changes

## Need Help?
Open an issue on the GitHub repository for support.
