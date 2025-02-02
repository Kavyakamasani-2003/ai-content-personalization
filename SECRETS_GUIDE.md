# Secrets Management Guide for AI Content Personalization

## GitHub Secrets Setup

### Required Secrets
1. PYPI_USERNAME
   - Your PyPI username
   - Used for publishing packages

2. PYPI_PASSWORD
   - PyPI access token
   - NEVER share this publicly

### Optional Secrets
1. CODECOV_TOKEN
   - For code coverage reporting
   - Obtain from Codecov.io

## Security Best Practices
- Use access tokens instead of passwords
- Limit token scope
- Rotate tokens periodically
- Never commit secrets to repository

## How to Add Secrets in GitHub
1. Go to repository Settings
2. Click 'Secrets and variables'
3. Select 'Actions'
4. Click 'New repository secret'
5. Add each secret individually

## Recommended Token Permissions
- PyPI Token: 
  - Scope: Entire account
  - Limit: Package upload only

## Troubleshooting
- If token is compromised, immediately:
  1. Revoke the token
  2. Generate a new one
  3. Update GitHub secrets
