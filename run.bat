@echo off
cd /d "%~dp0"

echo === Graph Fake News Detection ===
echo.

if exist ".venv\Scripts\activate.bat" (
    echo Activation de l'environnement .venv...
    call .venv\Scripts\activate.bat
) else (
    echo Creation de l'environnement .venv...
    python -m venv .venv
    call .venv\Scripts\activate.bat
)

echo Installation des dependances...
pip install -r requirements.txt -q

set CONFIG=config.yaml
if not exist "sample_data\twitter15\tree" (
    echo Creation des donnees de test (sample_data)...
    python create_sample_data.py
)
if not exist "C:\Users\SAMI YOUSSEF\OneDrive\Desktop\Graph Analytics & Applications\Twitter15_16_dataset" (
    set CONFIG=config_sample.yaml
    echo Utilisation des donnees de test: config_sample.yaml
)

echo.
echo Lancement du pipeline...
python -u -m src.gfn.run --config %CONFIG% --run-all

echo.
if %ERRORLEVEL% equ 0 (
    echo Resultats: outputs\tables\ et outputs\figures\
) else (
    echo ERREUR code %ERRORLEVEL%
)
pause
