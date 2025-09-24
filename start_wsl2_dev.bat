@echo off
title NeMo + NIM WSL2 Development Environment
color 0A

echo ========================================
echo     NeMo + NIM WSL2 Environment
echo ========================================
echo.

REM Check if WSL2 is installed
wsl --list >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: WSL2 not installed!
    echo Please run setup_wsl2_windows.ps1 first
    pause
    exit /b 1
)

echo Starting WSL2 Ubuntu with GPU support...
echo.

REM Navigate to project and run setup
wsl -d Ubuntu -e bash -c "cd /mnt/c/Users/maxli/PycharmProjects/PythonProject/AI_agents && chmod +x setup_wsl2.sh && ./setup_wsl2.sh"

REM After setup, enter interactive shell
echo.
echo Entering WSL2 development environment...
wsl -d Ubuntu -e bash -c "cd /mnt/c/Users/maxli/PycharmProjects/PythonProject/AI_agents && echo 'Ready! Type nemo-enter to access the Docker container' && exec bash"