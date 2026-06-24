@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found. Run setup first:
    echo   python -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install -r requirements.txt maturin ^&^& maturin develop --release
    pause
    exit /b 1
)
if not exist "python\ogame_optimizer\_ogame_combat.cp310-win_amd64.pyd" (
    echo ERROR: Rust extension not built. Run: maturin develop --release
    pause
    exit /b 1
)
call .venv\Scripts\activate
echo Starting OGame Fleet Optimizer at http://127.0.0.1:8000
echo Press Ctrl+C to stop.
echo.
uvicorn ogame_optimizer.api.app:app --host 127.0.0.1 --port 8000 --reload %*
endlocal
