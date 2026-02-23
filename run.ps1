# Script de lancement - Graph Fake News Detection
# Usage: .\run.ps1 [diagnose|run-all]

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=== Graph Fake News Detection ===" -ForegroundColor Cyan
Write-Host "Repertoire: $scriptDir" -ForegroundColor Gray

# Activer l'environnement virtuel s'il existe
$venvActivate = Join-Path $scriptDir ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "Activation de l'environnement .venv..." -ForegroundColor Yellow
    & $venvActivate
} else {
    Write-Host "Pas de .venv - utilisation de Python systeme" -ForegroundColor Yellow
}

# Installer les dependances si besoin
Write-Host "Verification des dependances..." -ForegroundColor Yellow
pip install -q -r requirements.txt 2>$null

# Creer sample_data si absent
if (-not (Test-Path "sample_data\twitter15\tree")) {
    Write-Host "Creation des donnees de test..." -ForegroundColor Yellow
    python create_sample_data.py
}

$config = "config.yaml"
$datasetPath = "C:\Users\SAMI YOUSSEF\OneDrive\Desktop\Graph Analytics & Applications\Twitter15_16_dataset"
if (-not (Test-Path $datasetPath)) {
    $config = "config_sample.yaml"
    Write-Host "Dataset principal absent - utilisation de $config" -ForegroundColor Yellow
}

$mode = if ($args[0]) { $args[0] } else { "run-all" }
Write-Host "`nLancement: python -m src.gfn.run --config $config --$mode" -ForegroundColor Green
Write-Host ""

python -u -m src.gfn.run --config $config --$mode

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERREUR: Le script a echoue (code $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`nResultats dans: outputs\tables\ et outputs\figures\" -ForegroundColor Green
exit 0
