# PyPI Publication Checklist

## Pre-Publication Checks
- [ ] All tests pass
- [ ] Documentation is up to date
- [ ] Code is linted and type-checked
- [ ] Version number updated in:
  - pyproject.toml
  - src/__init__.py
  - CHANGELOG.md

## Publication Steps
1. Build distribution
\\\ash
python -m build
\\\

2. Verify package
\\\ash
twine check dist/*
\\\

3. Create GitHub Release
- Tag version (e.g., v0.1.0)
- Add release notes

## Post-Publication
- [ ] Verify PyPI listing
- [ ] Check package installation
- [ ] Announce release

## Troubleshooting
- Ensure all dependencies are listed
- Check README formatting
- Verify package structure
