# KineticsForge PowerShell Launcher
Write-Host ""
Write-Host "  ====================================" -ForegroundColor Red
Write-Host "   KINETICSFORGE PLATFORM LAUNCHER" -ForegroundColor White
Write-Host "  ====================================" -ForegroundColor Red
Write-Host ""

Set-Location $PSScriptRoot

# Check Python
try {
    $pyVer = python --version 2>&1
    Write-Host "[OK] $pyVer" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found. Install Python 3.11+" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install dependencies
Write-Host "[1/3] Checking dependencies..." -ForegroundColor Yellow
$hasFastapi = pip show fastapi 2>$null
if (-not $hasFastapi) {
    Write-Host "       Installing fastapi, uvicorn, numpy, pydantic..." -ForegroundColor Gray
    pip install fastapi "uvicorn[standard]" numpy pydantic --quiet 2>$null
}

# Extract checkpoints
Write-Host "[2/3] Extracting trained model checkpoints..." -ForegroundColor Yellow
python scripts/extract_checkpoints.py 2>$null

# Count checkpoints
$ckptDir = Join-Path $PSScriptRoot "checkpoints/trained"
if (Test-Path $ckptDir) {
    $ckptCount = (Get-ChildItem $ckptDir -Filter "*.pt").Count
    Write-Host "       Found $ckptCount trained checkpoints" -ForegroundColor Green
}

# Launch
Write-Host "[3/3] Starting KineticsForge server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  ============================================" -ForegroundColor Red
Write-Host "   Server: http://localhost:8000" -ForegroundColor White
Write-Host "   API:    http://localhost:8000/health" -ForegroundColor Gray
Write-Host "   Docs:   http://localhost:8000/docs" -ForegroundColor Gray
Write-Host "   Press Ctrl+C to stop" -ForegroundColor DarkGray
Write-Host "  ============================================" -ForegroundColor Red
Write-Host ""

# Open browser
Start-Process "http://localhost:8000"

# Start server
python serve_lite.py
