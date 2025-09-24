@echo off
REM One-Click Docker Launch for Windows
REM NeMo + NIM Complete Environment

setlocal enabledelayedexpansion

echo ========================================
echo     NeMo + NIM Complete Environment
echo         Windows Docker Launcher
echo ========================================
echo.

REM Check Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker not found!
    echo Please install Docker Desktop from:
    echo https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)
echo [OK] Docker installed

REM Check GPU
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: No NVIDIA GPU detected
    echo The container will run in CPU mode (very slow)
    set /p continue="Continue anyway? (y/n): "
    if /i not "!continue!"=="y" exit /b 1
) else (
    echo [OK] NVIDIA GPU detected
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
)

REM Check .env file
if not exist .env (
    echo ERROR: .env file not found!
    echo Please ensure .env file exists with your API key
    pause
    exit /b 1
)
echo [OK] Environment file found

REM Main menu
:menu
echo.
echo What would you like to do?
echo 1) Build and start containers
echo 2) Enter development container
echo 3) View container logs
echo 4) Stop all containers
echo 5) Rebuild containers (fresh)
echo 6) Quick start (build + enter)
echo 7) Exit
echo.
set /p choice="Select option (1-7): "

if "%choice%"=="1" goto build_start
if "%choice%"=="2" goto enter_container
if "%choice%"=="3" goto view_logs
if "%choice%"=="4" goto stop_containers
if "%choice%"=="5" goto rebuild
if "%choice%"=="6" goto quick_start
if "%choice%"=="7" exit /b 0

echo Invalid option!
goto menu

:build_start
echo.
echo Building Docker image...
echo This may take 10-30 minutes on first build
docker build -f Dockerfile.all -t nemo-nim-complete:latest .
if %errorlevel% neq 0 (
    echo Build failed! Trying with docker-compose...
    docker-compose build nemo-nim-dev
)

echo Starting containers...
docker-compose up -d nemo-nim-dev

echo.
echo ========================================
echo     Environment Ready!
echo ========================================
echo.
echo Quick Commands:
echo.
echo Enter container:
echo   docker exec -it nemo-nim-dev bash
echo.
echo Train a model:
echo   docker exec -it nemo-nim-dev python train.py --data data/sample.jsonl
echo.
echo Start Jupyter:
echo   docker exec -it nemo-nim-dev jupyter lab --ip=0.0.0.0 --allow-root
echo   Then visit: http://localhost:8888
echo.
echo View logs:
echo   docker-compose logs -f nemo-nim-dev
echo.
echo Stop containers:
echo   docker-compose down
echo.
pause
goto menu

:enter_container
echo Entering container...
docker exec -it nemo-nim-dev bash
goto menu

:view_logs
docker-compose logs -f nemo-nim-dev
goto menu

:stop_containers
docker-compose down
echo Containers stopped
pause
goto menu

:rebuild
docker-compose down
docker-compose build --no-cache nemo-nim-dev
docker-compose up -d nemo-nim-dev
echo Containers rebuilt
pause
goto menu

:quick_start
call :build_start
docker exec -it nemo-nim-dev bash
goto menu