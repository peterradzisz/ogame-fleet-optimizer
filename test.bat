@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found.
    pause
    exit /b 1
)
call .venv\Scripts\activate
echo Running all tests...
pytest python_tests/ -v %*
endlocal
