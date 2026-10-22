@echo off
setlocal
cd /d "%~dp0"
title Scheduler Lab - dashboard

echo ============================================================
echo  SCHEDULER LAB  -  AI Task Scheduler dashboard
echo ============================================================

where python >nul 2>nul
if errorlevel 1 (
    echo [X] Python was not found. Install Python 3.10+ from https://www.python.org/downloads/
    echo     and tick "Add python.exe to PATH" during setup, then run this file again.
    goto :end
)

echo.
echo [1/3] Installing / checking required packages...
python -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo [X] Package installation failed. Check your internet connection and try again.
    goto :end
)

echo.
echo [2/3] Checking experiment results for the Results tab...
if not exist "experiments\results\e5_prediction_noise.csv" (
    echo     Not found - running the experiments once, this takes about a minute...
    python experiments\run_experiments.py >nul
    if errorlevel 1 goto :fail
) else (
    echo     Found.
)

echo.
echo [3/3] Starting the dashboard. A browser tab opens at http://127.0.0.1:5050
echo       Keep this window open while you use it. Press Ctrl+C here to stop.
echo.
python dashboard\app.py %*
goto :end

:fail
echo.
echo [X] A step failed - read the message above.

:end
if /i not "%~1"=="--no-browser" pause
endlocal
