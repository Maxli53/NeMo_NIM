# PowerShell script to prepare Windows for WSL2 GPU development
# Run as Administrator

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  WSL2 GPU Setup for NeMo + NIM" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "This script needs Administrator privileges!" -ForegroundColor Red
    Write-Host "Right-click and select 'Run as Administrator'" -ForegroundColor Yellow
    pause
    exit 1
}

# 1. Check Windows version
Write-Host "Step 1: Checking Windows version..." -ForegroundColor Yellow
$winVer = [System.Environment]::OSVersion.Version
$build = (Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion").CurrentBuild

if ($winVer.Major -ge 10 -and $build -ge 19041) {
    Write-Host "✓ Windows version compatible (Build: $build)" -ForegroundColor Green
} else {
    Write-Host "✗ Windows version too old. Need Windows 10 build 19041+" -ForegroundColor Red
    Write-Host "Please update Windows first" -ForegroundColor Yellow
    pause
    exit 1
}

# 2. Enable WSL2
Write-Host "`nStep 2: Enabling WSL2..." -ForegroundColor Yellow

# Enable WSL feature
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# Enable Virtual Machine Platform
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

Write-Host "✓ WSL2 features enabled" -ForegroundColor Green

# 3. Set WSL2 as default
Write-Host "`nStep 3: Setting WSL2 as default..." -ForegroundColor Yellow
wsl --set-default-version 2
Write-Host "✓ WSL2 set as default" -ForegroundColor Green

# 4. Check for existing WSL distros
Write-Host "`nStep 4: Checking WSL distributions..." -ForegroundColor Yellow
$distros = wsl -l -v

if ($distros -match "Ubuntu") {
    Write-Host "✓ Ubuntu already installed" -ForegroundColor Green

    # Check if it's WSL2
    if ($distros -match "Ubuntu.*2") {
        Write-Host "✓ Ubuntu running on WSL2" -ForegroundColor Green
    } else {
        Write-Host "Converting Ubuntu to WSL2..." -ForegroundColor Yellow
        wsl --set-version Ubuntu 2
    }
} else {
    Write-Host "Installing Ubuntu..." -ForegroundColor Yellow
    wsl --install -d Ubuntu
    Write-Host "✓ Ubuntu installed" -ForegroundColor Green
}

# 5. Check NVIDIA GPU
Write-Host "`nStep 5: Checking NVIDIA GPU..." -ForegroundColor Yellow
$gpu = Get-WmiObject Win32_VideoController | Where-Object {$_.Name -match "NVIDIA"}

if ($gpu) {
    Write-Host "✓ NVIDIA GPU detected: $($gpu.Name)" -ForegroundColor Green

    # Check driver version
    $nvidiaDriver = Get-WmiObject Win32_PnPSignedDriver | Where-Object {$_.DeviceName -match "NVIDIA"} | Select-Object -First 1
    if ($nvidiaDriver) {
        Write-Host "  Driver version: $($nvidiaDriver.DriverVersion)" -ForegroundColor Cyan
    }

    # Check for WSL GPU support (driver must be 510.06+)
    $driverPath = "${env:ProgramFiles}\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    if (Test-Path $driverPath) {
        Write-Host "✓ NVIDIA drivers installed" -ForegroundColor Green
        & $driverPath
    } else {
        Write-Host "⚠ NVIDIA drivers may need update for WSL2 GPU" -ForegroundColor Yellow
        Write-Host "Download from: https://developer.nvidia.com/cuda/wsl" -ForegroundColor Cyan
    }
} else {
    Write-Host "⚠ No NVIDIA GPU detected" -ForegroundColor Yellow
    Write-Host "CPU-only mode will be used (slower)" -ForegroundColor Yellow
}

# 6. Configure WSL2 memory
Write-Host "`nStep 6: Configuring WSL2 memory limits..." -ForegroundColor Yellow

$wslConfig = @"
[wsl2]
memory=24GB
processors=8
swap=8GB
localhostForwarding=true

[experimental]
sparseVhd=true
autoMemoryReclaim=gradual
"@

$configPath = "$env:USERPROFILE\.wslconfig"
if (Test-Path $configPath) {
    Write-Host "Backing up existing .wslconfig..." -ForegroundColor Yellow
    Copy-Item $configPath "$configPath.backup"
}

$wslConfig | Out-File -FilePath $configPath -Encoding ASCII
Write-Host "✓ WSL2 configured for 24GB memory" -ForegroundColor Green

# 7. Create batch file for easy WSL2 launch
Write-Host "`nStep 7: Creating launch script..." -ForegroundColor Yellow

$launchScript = @"
@echo off
title NeMo + NIM WSL2 Environment
echo Starting WSL2 with GPU support...
wsl -d Ubuntu bash -c "cd /mnt/c/Users/$env:USERNAME/PycharmProjects/PythonProject/AI_agents && ./setup_wsl2.sh"
"@

$launchScript | Out-File -FilePath "$PSScriptRoot\start_wsl2_dev.bat" -Encoding ASCII
Write-Host "✓ Launch script created: start_wsl2_dev.bat" -ForegroundColor Green

# 8. Final instructions
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "      Windows Setup Complete! 🎉" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "`n📋 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Restart your computer (required for WSL2)" -ForegroundColor Yellow
Write-Host "2. After restart, run: " -NoNewline
Write-Host "start_wsl2_dev.bat" -ForegroundColor Green
Write-Host "3. This will set up Docker and everything in WSL2" -ForegroundColor Yellow

Write-Host "`n💡 Quick Commands After Setup:" -ForegroundColor Cyan
Write-Host "  Enter WSL2:     " -NoNewline -ForegroundColor White
Write-Host "wsl" -ForegroundColor Green
Write-Host "  Open project:   " -NoNewline -ForegroundColor White
Write-Host "wsl -d Ubuntu -e bash -c 'cd /mnt/c/Users/$env:USERNAME/PycharmProjects/PythonProject/AI_agents && bash'" -ForegroundColor Green

Write-Host "`n🔧 Troubleshooting:" -ForegroundColor Cyan
Write-Host "  If GPU not working: Update NVIDIA drivers from https://developer.nvidia.com/cuda/wsl" -ForegroundColor Yellow
Write-Host "  If WSL2 issues: Run 'wsl --update' in PowerShell" -ForegroundColor Yellow

$restart = Read-Host "`nRestart now? (y/n)"
if ($restart -eq 'y') {
    Write-Host "Restarting in 5 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    Restart-Computer -Force
} else {
    Write-Host "Please restart manually before continuing" -ForegroundColor Yellow
}