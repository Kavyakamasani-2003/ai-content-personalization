# Secure PyPI Token Management

function Protect-PyPIToken {
    param(
        [string]
    )

    # Basic token validation
    if (-not ) {
        Write-Error "No token provided"
        return 
    }

    # Encrypt token (Windows-specific)
     = ConvertTo-SecureString  -AsPlainText -Force
     =  | ConvertFrom-SecureString

    # Save encrypted token securely
     | Out-File "C:\Users\kavya\.pypi_token" -Encoding UTF8

    Write-Host "Token encrypted and stored securely"
}

function Get-PyPIToken {
     = "C:\Users\kavya\.pypi_token"
    
    if (Test-Path ) {
         = Get-Content 
         =  | ConvertTo-SecureString
         = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR()
         = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto()
        return 
    }
    else {
        Write-Error "No token found. Please save a token first."
        return 
    }
}

# Usage example
# Protect-PyPIToken -Token "your-pypi-token-here"
#  = Get-PyPIToken
