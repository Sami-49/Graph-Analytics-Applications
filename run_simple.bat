@echo off
cd /d "%~dp0"

echo === Graph Fake News Detection (Python systeme) ===
echo.

if not exist "sample_data\twitter15\tree" (
    echo Creation des donnees de test...
    python create_sample_data.py
)

set CONFIG=config_sample.yaml
echo Lancement: python -m src.gfn.run --config %CONFIG% --run-all
echo.

python -u -m src.gfn.run --config %CONFIG% --run-all

echo.
if %ERRORLEVEL% equ 0 (
    echo OK - Resultats: outputs\tables\ et outputs\figures\
) else (
    echo ERREUR code %ERRORLEVEL%
)
pause
