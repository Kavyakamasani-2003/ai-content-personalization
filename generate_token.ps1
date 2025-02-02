# Secure Token Generation Script

function Generate-SecureToken {
    param(
        [string]
    )

    Write-Host "Generating secure token for "
    Write-Host "IMPORTANT: Keep this token confidential!"
    
     = [System.Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes((New-Guid)))
    
    Write-Host "
Generated Token: "
    Write-Host "
Instructions:"
    Write-Host "1. Copy this token"
    Write-Host "2. Go to GitHub repository settings"
    Write-Host "3. Add as a new secret"
}

# Example usage
Generate-SecureToken -Service "PyPI"
