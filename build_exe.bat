@echo off
REM ============================================================
REM  build_exe.bat — Build Aakhi portable exe
REM  Run from the project root:  build_exe.bat
REM  Output: dist\Aakhi\  (zip this folder to share)
REM ============================================================

set "PROJECT=%~dp0"
cd /d "%PROJECT%"

echo.
echo ============================================================
echo  Aakhi ^| Building Portable EXE
echo  Project: %PROJECT%
echo ============================================================
echo.

REM ── Auto-discover pyinstaller ────────────────────────────────
REM Search common venv folder names relative to the project root.
set "PYINSTALLER="

for %%V in (
    .venv\Scripts\pyinstaller.exe
    venv\Scripts\pyinstaller.exe
    env\Scripts\pyinstaller.exe
    model_check\.venv\Scripts\pyinstaller.exe
) do (
    if exist "%PROJECT%%%V" (
        set "PYINSTALLER=%PROJECT%%%V"
        goto :found
    )
)

REM Fall back to system pyinstaller if installed globally
where pyinstaller >nul 2>&1
if %errorlevel% == 0 (
    set "PYINSTALLER=pyinstaller"
    goto :found
)

echo ERROR: Cannot find pyinstaller.
echo.
echo Install it inside your project venv, e.g.:
echo   .venv\Scripts\pip install pyinstaller
echo.
pause
exit /b 1

:found
echo Using pyinstaller: %PYINSTALLER%
echo.

REM ── Clean previous build ─────────────────────────────────────
if exist build  rd /s /q build
if exist dist   rd /s /q dist

REM ── Run build ────────────────────────────────────────────────
"%PYINSTALLER%" aakhi.spec --noconfirm --clean

if errorlevel 1 (
    echo.
    echo *** BUILD FAILED — check errors above ***
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  BUILD COMPLETE
echo  Executable folder:  dist\Aakhi\
echo  To share: zip dist\Aakhi\ and copy to USB
echo  To run:   double-click dist\Aakhi\Aakhi.exe
echo ============================================================
pause
