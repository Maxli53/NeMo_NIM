#!/bin/bash
# WSL2 Complete Setup Script for NeMo + NIM Development
# Optimized for WSL2 with GPU passthrough

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║    WSL2 NeMo + NIM Environment Setup       ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════╝${NC}"
echo ""

# 1. Check if we're in WSL2
echo -e "${YELLOW}Step 1: Checking WSL2 environment...${NC}"
if grep -q microsoft /proc/version; then
    echo -e "${GREEN}✓ Running in WSL2${NC}"

    # Check WSL version
    if [[ $(wsl.exe -l -v 2>/dev/null | grep -c "VERSION 2") -gt 0 ]] || [[ -f /proc/sys/fs/binfmt_misc/WSLInterop ]]; then
        echo -e "${GREEN}✓ WSL2 confirmed${NC}"
    else
        echo -e "${RED}✗ This appears to be WSL1. Please upgrade to WSL2${NC}"
        echo "Run in PowerShell: wsl --set-version Ubuntu 2"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ Not running in WSL. This script is optimized for WSL2${NC}"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 2. Update system
echo -e "\n${YELLOW}Step 2: Updating system packages...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# 3. Install Docker in WSL2
echo -e "\n${YELLOW}Step 3: Setting up Docker for WSL2...${NC}"

# Check if Docker is already installed
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker already installed${NC}"
    docker --version
else
    echo -e "${YELLOW}Installing Docker in WSL2...${NC}"

    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh

    # Add user to docker group
    sudo usermod -aG docker $USER

    # Start Docker service
    sudo service docker start

    echo -e "${GREEN}✓ Docker installed${NC}"
    echo -e "${YELLOW}NOTE: You may need to log out and back in for group changes${NC}"
fi

# 4. Install NVIDIA Container Toolkit for WSL2
echo -e "\n${YELLOW}Step 4: Installing NVIDIA Container Toolkit for WSL2...${NC}"

# Check if GPU is available in WSL2
if nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected in WSL2${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

    # Install NVIDIA Container Toolkit
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit

    # Configure Docker for GPU
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker || sudo service docker restart

    # Test GPU access in Docker
    echo -e "${YELLOW}Testing GPU access in Docker...${NC}"
    if docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ Docker can access GPU!${NC}"
    else
        echo -e "${RED}✗ Docker GPU access failed${NC}"
        echo "Troubleshooting GPU access..."

        # Try to fix common issues
        sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
        sudo service docker restart
        sleep 2

        # Retry
        if docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi &> /dev/null; then
            echo -e "${GREEN}✓ Fixed! Docker can now access GPU${NC}"
        else
            echo -e "${RED}GPU passthrough not working. Continuing without GPU...${NC}"
        fi
    fi
else
    echo -e "${RED}✗ No NVIDIA GPU detected in WSL2${NC}"
    echo -e "${YELLOW}Make sure:${NC}"
    echo "1. You have Windows 11 or Windows 10 build 21H2+"
    echo "2. NVIDIA GPU drivers are installed in Windows"
    echo "3. WSL2 GPU support is enabled"
    echo ""
    echo "Install NVIDIA drivers for WSL from:"
    echo "https://developer.nvidia.com/cuda/wsl"

    read -p "Continue without GPU? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 5. Install Docker Compose
echo -e "\n${YELLOW}Step 5: Installing Docker Compose...${NC}"
if command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✓ Docker Compose already installed${NC}"
    docker-compose --version
else
    sudo apt-get install -y docker-compose
    echo -e "${GREEN}✓ Docker Compose installed${NC}"
fi

# 6. Configure WSL2 memory limits
echo -e "\n${YELLOW}Step 6: Optimizing WSL2 configuration...${NC}"

# Create .wslconfig in Windows user directory
WIN_USER=$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r\n')
WIN_HOME="/mnt/c/Users/${WIN_USER}"

if [ -d "$WIN_HOME" ]; then
    cat > /tmp/.wslconfig << 'EOF'
[wsl2]
memory=24GB
processors=8
swap=8GB
localhostForwarding=true

[experimental]
sparseVhd=true
EOF

    if [ -f "$WIN_HOME/.wslconfig" ]; then
        echo -e "${YELLOW}WSL2 config already exists. Backing up...${NC}"
        cp "$WIN_HOME/.wslconfig" "$WIN_HOME/.wslconfig.backup"
    fi

    cp /tmp/.wslconfig "$WIN_HOME/.wslconfig"
    echo -e "${GREEN}✓ WSL2 config optimized for 24GB RAM${NC}"
    echo -e "${YELLOW}NOTE: Restart WSL2 for changes to take effect${NC}"
else
    echo -e "${YELLOW}Could not find Windows home directory${NC}"
fi

# 7. Install development tools
echo -e "\n${YELLOW}Step 7: Installing development tools...${NC}"
sudo apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    tmux \
    htop \
    python3-pip \
    python3-venv

echo -e "${GREEN}✓ Development tools installed${NC}"

# 8. Set up project
echo -e "\n${YELLOW}Step 8: Setting up NeMo + NIM project...${NC}"

# Make sure we're in project directory
cd /mnt/c/Users/maxli/PycharmProjects/PythonProject/AI_agents

# Build Docker image
echo -e "${YELLOW}Building Docker image with all frameworks...${NC}"
echo "This will take 10-30 minutes on first build..."

if docker build -f Dockerfile.all -t nemo-nim-complete:latest .; then
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "${YELLOW}Trying with docker-compose...${NC}"
    docker-compose build nemo-nim-dev
fi

# 9. Start containers
echo -e "\n${YELLOW}Step 9: Starting Docker containers...${NC}"
docker-compose up -d nemo-nim-dev

# Wait for container to be ready
echo -e "${YELLOW}Waiting for container to initialize...${NC}"
for i in {1..30}; do
    if docker exec nemo-nim-dev python -c "import torch; print('Ready!')" 2>/dev/null; then
        echo -e "\n${GREEN}✓ Container is ready!${NC}"
        break
    fi
    sleep 2
    echo -n "."
done

# 10. Create helpful aliases
echo -e "\n${YELLOW}Step 10: Creating helpful aliases...${NC}"

cat >> ~/.bashrc << 'EOF'

# NeMo + NIM Docker aliases
alias nemo-enter='docker exec -it nemo-nim-dev bash'
alias nemo-train='docker exec -it nemo-nim-dev python train.py'
alias nemo-deploy='docker exec -it nemo-nim-dev python deploy.py'
alias nemo-jupyter='docker exec -it nemo-nim-dev jupyter lab --ip=0.0.0.0 --allow-root --no-browser'
alias nemo-logs='docker-compose logs -f nemo-nim-dev'
alias nemo-stop='docker-compose down'
alias nemo-restart='docker-compose restart nemo-nim-dev'
alias nemo-gpu='docker exec -it nemo-nim-dev nvidia-smi'

# Quick navigation
alias ai='cd /mnt/c/Users/maxli/PycharmProjects/PythonProject/AI_agents'
EOF

source ~/.bashrc
echo -e "${GREEN}✓ Aliases created${NC}"

# Final summary
echo -e "\n${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}       WSL2 Setup Complete! 🎉${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"

echo -e "\n${CYAN}📋 Quick Commands (from anywhere):${NC}"
echo -e "${YELLOW}nemo-enter${NC}     - Enter the container"
echo -e "${YELLOW}nemo-train${NC}     - Run training"
echo -e "${YELLOW}nemo-deploy${NC}    - Deploy model"
echo -e "${YELLOW}nemo-jupyter${NC}   - Start Jupyter Lab"
echo -e "${YELLOW}nemo-logs${NC}      - View logs"
echo -e "${YELLOW}nemo-gpu${NC}       - Check GPU status"
echo -e "${YELLOW}ai${NC}             - Go to project directory"

echo -e "\n${CYAN}🌐 Access from Windows:${NC}"
echo "Jupyter Lab:  http://localhost:8888"
echo "FastAPI:      http://localhost:8000"
echo "TensorBoard:  http://localhost:6006"

echo -e "\n${GREEN}💡 Next Steps:${NC}"
echo "1. Run: ${YELLOW}nemo-enter${NC} to enter the container"
echo "2. Inside container: ${YELLOW}python test_installation.py${NC}"
echo "3. Train: ${YELLOW}python train.py --data data/sample.jsonl${NC}"

echo -e "\n${YELLOW}NOTE:${NC} If this is first time setup:"
echo "  1. Exit WSL: ${YELLOW}exit${NC}"
echo "  2. In PowerShell: ${YELLOW}wsl --shutdown${NC}"
echo "  3. Restart WSL: ${YELLOW}wsl${NC}"
echo "  This ensures all configurations are loaded"

# Test GPU one more time
echo -e "\n${CYAN}Final GPU Check:${NC}"
if docker exec nemo-nim-dev nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ GPU working in container!${NC}"
    docker exec nemo-nim-dev nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader
else
    echo -e "${YELLOW}⚠ GPU not accessible in container${NC}"
    echo "Training will work but will be slower"
fi