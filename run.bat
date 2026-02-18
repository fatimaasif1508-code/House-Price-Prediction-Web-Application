@echo off
echo ==========================================
echo   House Price Predictor - Windows Runner
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Check if model files exist
if not exist "model.pkl" (
    echo [INFO] Model files not found. Running setup first...
    python setup.py
    if errorlevel 1 (
        echo [ERROR] Setup failed!
        pause
        exit /b 1
    )
)

if not exist "features.pkl" (
    echo [INFO] Feature file not found. Running setup first...
    python setup.py
    if errorlevel 1 (
        echo [ERROR] Setup failed!
        pause
        exit /b 1
    )
)

echo [OK] Model files found
echo.

REM Install dependencies if needed
echo [INFO] Checking dependencies...
pip install -q flask scikit-learn numpy pandas 2>nul
if errorlevel 1 (
    echo [INFO] Installing dependencies (first time only)...
    pip install flask scikit-learn numpy pandas
)
echo [OK] Dependencies ready
echo.

echo ==========================================
echo   Starting Flask Server...
echo   Open http://localhost:5000 in browser
echo   Press CTRL+C to stop
echo ==========================================
echo.

python app.py

pause
