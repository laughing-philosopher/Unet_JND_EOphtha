@echo off
title Aakhi - First Time Setup
echo ============================================
echo   Aakhi - First Time Setup
echo   This will take 5-10 minutes.
echo   Please do NOT close this window.
echo ============================================
echo.

:: Directory where this .bat file lives (root of Aakhi folder)
set BASE_DIR=%~dp0
set PYTHON=%BASE_DIR%python\python.exe
set PIP=%BASE_DIR%python\Scripts\pip.exe
set GET_PIP=%BASE_DIR%python\get-pip.py

:: Check portable Python exists
if not exist "%PYTHON%" (
    echo ERROR: python\python.exe not found.
    echo Make sure you extracted the embeddable Python into the python\ folder.
    pause
    exit /b 1
)

:: Check get-pip.py exists
if not exist "%GET_PIP%" (
    echo ERROR: python\get-pip.py not found.
    echo Download it from https://bootstrap.pypa.io/get-pip.py
    echo and place it in the python\ folder.
    pause
    exit /b 1
)

:: Step 1 - Enable site-packages in embeddable Python
echo [1/4] Configuring Python...
set PTH_FILE=%BASE_DIR%python\python310._pth
if exist "%PTH_FILE%" (
    powershell -Command "(Get-Content '%PTH_FILE%') -replace '#import site', 'import site' | Set-Content '%PTH_FILE%'"
    echo     python310._pth configured.
) else (
    echo     WARNING: python310._pth not found, skipping.
)

:: Step 2 - Install pip
echo [2/4] Installing pip...
if not exist "%PIP%" (
    "%PYTHON%" "%GET_PIP%" --no-warn-script-location
) else (
    echo     pip already installed, skipping.
)

:: Step 3 - Install all packages
echo [3/4] Installing packages (this may take several minutes)...
"%PIP%" install --no-warn-script-location streamlit opencv-python-headless numpy pillow reportlab scipy tensorflow-cpu timm segmentation-models-pytorch

:: PyTorch CPU-only (much smaller than default CUDA build)
"%PIP%" install --no-warn-script-location torch torchvision --index-url https://download.pytorch.org/whl/cpu

:: Step 4 - Verify
echo.
echo [4/4] Verifying installation...
"%PYTHON%" -c "import streamlit, cv2, numpy, PIL, reportlab, tensorflow, torch, timm; print('All packages OK')"
if errorlevel 1 (
    echo WARNING: Some packages may not have installed correctly.
    echo Try running this file again.
) else (
    echo All packages verified successfully.
)

echo.
echo ============================================
echo   Setup complete! You can now run:
echo   run_aakhi.bat
echo ============================================
pause
