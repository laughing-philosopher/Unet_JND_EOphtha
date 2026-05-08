@echo off
title Aakhi - Retinal Image Analysis

:: Root of Aakhi folder (where this .bat lives)
:: Your flat structure: all .py files are directly here, NOT in an app\ subfolder
set BASE_DIR=%~dp0
set PYTHON=%BASE_DIR%python\python.exe

:: Check Python exists
if not exist "%PYTHON%" (
    echo ERROR: Python not found.
    echo Please run install.bat first.
    pause
    exit /b 1
)

:: Check Streamlit is installed
"%PYTHON%" -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ERROR: Packages not installed.
    echo Please run install.bat first.
    pause
    exit /b 1
)

echo ============================================
echo   Starting Aakhi...
echo   A browser window will open shortly.
echo   To stop the app, close this window.
echo ============================================
echo.

:: Change to the project root (where app.py lives)
cd /d "%BASE_DIR%"

:: Open browser after short delay
start "" cmd /c "timeout /t 4 >nul && start http://localhost:8501"

:: Launch Streamlit using portable Python
"%PYTHON%" -m streamlit run app.py ^
    --server.port 8501 ^
    --server.headless true ^
    --server.fileWatcherType none ^
    --browser.gatherUsageStats false
