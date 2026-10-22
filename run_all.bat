@echo off
setlocal
cd /d "%~dp0"
title AI Task Scheduler - run everything

rem Usage:  run_all.bat            -> a fresh random batch every time
rem         run_all.bat fixed      -> the fixed seed-7 batch used in the report and slides
set MODE=random
set SEED=%RANDOM%
if /i "%~1"=="fixed" set MODE=fixed
if /i "%~2"=="fixed" set MODE=fixed

echo ============================================================
echo  AI TASK SCHEDULER  -  one-click setup, run and dashboard
echo ============================================================

where python >nul 2>nul
if errorlevel 1 (
    echo [X] Python was not found. Install Python 3.10+ from https://www.python.org/downloads/
    echo     and tick "Add python.exe to PATH" during setup, then run this file again.
    goto :end
)

echo.
echo [1/7] Installing / checking required packages...
python -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo [X] Package installation failed. Check your internet connection and try again.
    goto :end
)

echo.
echo [2/7] Training the Random Forest models...
python main.py train
if errorlevel 1 goto :fail

echo.
echo [3/7] Comparing candidate ML models...
python main.py compare
if errorlevel 1 goto :fail

echo.
if /i "%MODE%"=="fixed" (
    echo [4/7] Running the fixed 8-process demo batch ^(seed 7^)...
    python main.py demo
) else (
    echo [4/7] Running a fresh random batch of 10 processes ^(seed %SEED%^)...
    python main.py run --n 10 --seed %SEED% --quantum 4
)
if errorlevel 1 goto :fail

echo.
echo [5/7] Opening the dashboard with the run you just made...
if /i "%~1"=="nopause" (
    echo       Skipped in test mode.
) else (
    start "Scheduler Lab - dashboard (keep this window open)" cmd /k python dashboard\app.py
    echo       The dashboard opens in your browser at http://127.0.0.1:5050
    echo       It runs in its own window. Close that window to stop it.
)

echo.
echo [6/7] Running the unit tests...
python -m unittest discover -s tests
if errorlevel 1 goto :fail

echo.
echo [7/7] Running the full experiments (200 batches, takes about a minute)...
python experiments\run_experiments.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo  DONE.  Charts are saved in "outputs\output_N",
echo         experiment tables are in "experiments\results".
echo         The dashboard is running in its own window.
echo ============================================================
goto :end

:fail
echo.
echo [X] A step failed - read the message above.

:end
if /i not "%~1"=="nopause" pause
endlocal
