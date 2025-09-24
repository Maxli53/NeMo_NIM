#!/bin/bash
# One-Click Docker Launch Script for NeMo+NIM Development
# Handles everything: build, launch, and provide instructions

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ASCII Art Banner
echo -e "${CYAN}"
cat << "EOF"
TPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPW
Q     NeMo + NIM Complete Environment        Q
Q         Docker Launch Script                Q
ZPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP]
EOF
echo -e "${NC}"

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}= Checking prerequisites...${NC}"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}L Docker not found!${NC}"
        echo "Please install Docker from: https://docs.docker.com/get-docker/"
        exit 1
    fi
    echo -e "${GREEN} Docker installed${NC}"

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo -e "${RED}L Docker Compose not found!${NC}"
        echo "Please install Docker Compose"
        exit 1
    fi
    echo -e "${GREEN} Docker Compose installed${NC}"

    # Check GPU
    if ! nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}   No NVIDIA GPU detected${NC}"
        echo "The container will run in CPU mode (very slow)"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN} NVIDIA GPU detected${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    fi

    # Check NVIDIA Container Toolkit
    if ! docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi &> /dev/null 2>&1; then
        echo -e "${YELLOW}   NVIDIA Container Toolkit not configured${NC}"
        echo "Installing NVIDIA Container Toolkit..."

        # Try to install based on OS
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
            curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
            curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
                sudo tee /etc/apt/sources.list.d/nvidia-docker.list
            sudo apt-get update
            sudo apt-get install -y nvidia-container-toolkit
            sudo systemctl restart docker
        else
            echo -e "${RED}Please install NVIDIA Container Toolkit manually${NC}"
            echo "Visit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            exit 1
        fi
    else
        echo -e "${GREEN} NVIDIA Container Toolkit configured${NC}"
    fi
}

# Function to check .env file
check_env_file() {
    echo -e "\n${YELLOW}= Checking environment configuration...${NC}"

    if [ ! -f .env ]; then
        echo -e "${YELLOW}Creating .env file...${NC}"
        echo "# Copy your NVIDIA API key here" > .env
        echo "NVIDIA_API_KEY=your_api_key_here" >> .env
        echo "NVIDIA_NGC_API_KEY=your_api_key_here" >> .env
        echo "ENVIRONMENT=development" >> .env
        echo -e "${RED}L No .env file found!${NC}"
        echo "Please add your NVIDIA API key to the .env file"
        exit 1
    fi

    # Check if API key is configured
    if grep -q "your_api_key_here" .env; then
        echo -e "${YELLOW}   API key not configured in .env${NC}"
        echo "Edit .env and add your NVIDIA API key"
        echo "Get your key from: https://build.nvidia.com"
        read -p "Continue without API key? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN} Environment configured${NC}"
    fi
}

# Function to build Docker image
build_docker_image() {
    echo -e "\n${YELLOW}=( Building Docker image...${NC}"
    echo "This may take 10-30 minutes on first build"

    # Build the complete image
    docker build -f Dockerfile.all -t nemo-nim-complete:latest . || {
        echo -e "${RED}L Build failed!${NC}"
        echo "Trying with docker-compose..."
        docker-compose build nemo-nim-dev || exit 1
    }

    echo -e "${GREEN} Docker image built successfully${NC}"
}

# Function to start containers
start_containers() {
    echo -e "\n${YELLOW}=€ Starting containers...${NC}"

    # Stop any existing containers
    docker-compose down 2>/dev/null || true

    # Start the development container
    docker-compose up -d nemo-nim-dev

    # Wait for container to be healthy
    echo -e "${YELLOW}ó Waiting for container to be ready...${NC}"
    for i in {1..30}; do
        if docker exec nemo-nim-dev python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            echo -e "${GREEN} Container is ready!${NC}"
            break
        fi
        sleep 2
        echo -n "."
    done
}

# Function to show instructions
show_instructions() {
    echo -e "\n${GREEN}PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP${NC}"
    echo -e "${GREEN}     Environment Ready! <‰${NC}"
    echo -e "${GREEN}PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP${NC}"

    echo -e "\n${CYAN}=Ë Quick Commands:${NC}"
    echo -e "${YELLOW}Enter container:${NC}"
    echo "  docker exec -it nemo-nim-dev bash"

    echo -e "\n${YELLOW}Train a model:${NC}"
    echo "  docker exec -it nemo-nim-dev python train.py --data data/sample.jsonl"

    echo -e "\n${YELLOW}Deploy with NIM:${NC}"
    echo "  docker exec -it nemo-nim-dev python deploy.py --model models/checkpoint"

    echo -e "\n${YELLOW}Start Jupyter Lab:${NC}"
    echo "  docker exec -it nemo-nim-dev jupyter lab --ip=0.0.0.0 --allow-root --no-browser"
    echo "  Then visit: http://localhost:8888"

    echo -e "\n${YELLOW}View logs:${NC}"
    echo "  docker-compose logs -f nemo-nim-dev"

    echo -e "\n${YELLOW}Stop containers:${NC}"
    echo "  docker-compose down"

    echo -e "\n${CYAN}< Web Interfaces:${NC}"
    echo "  FastAPI:     http://localhost:8000/docs"
    echo "  Jupyter:     http://localhost:8888"
    echo "  TensorBoard: http://localhost:6006"

    echo -e "\n${PURPLE}=Ú Documentation:${NC}"
    echo "  README.md for quick start"
    echo "  docs/ for detailed guides"
    echo "  CLAUDE.md for development guidelines"

    echo -e "\n${GREEN}=¡ Tips:${NC}"
    echo "  " All dependencies are pre-installed in the container"
    echo "  " Your files are mounted at /workspace"
    echo "  " GPU is automatically available"
    echo "  " Use 'docker exec' to run commands inside"
}

# Function to enter container
enter_container() {
    echo -e "\n${YELLOW}= Entering container...${NC}"
    docker exec -it nemo-nim-dev bash
}

# Main menu
main_menu() {
    echo -e "\n${CYAN}What would you like to do?${NC}"
    echo "1) Build and start containers"
    echo "2) Enter development container"
    echo "3) View container logs"
    echo "4) Stop all containers"
    echo "5) Rebuild containers (fresh build)"
    echo "6) Run tests"
    echo "7) Start Jupyter Lab"
    echo "8) Exit"

    read -p "Select option (1-8): " choice

    case $choice in
        1)
            check_prerequisites
            check_env_file
            build_docker_image
            start_containers
            show_instructions
            ;;
        2)
            enter_container
            ;;
        3)
            docker-compose logs -f nemo-nim-dev
            ;;
        4)
            docker-compose down
            echo -e "${GREEN} Containers stopped${NC}"
            ;;
        5)
            docker-compose down
            docker-compose build --no-cache nemo-nim-dev
            docker-compose up -d nemo-nim-dev
            echo -e "${GREEN} Containers rebuilt${NC}"
            ;;
        6)
            docker exec -it nemo-nim-dev pytest tests/
            ;;
        7)
            docker exec -d nemo-nim-dev jupyter lab --ip=0.0.0.0 --allow-root --no-browser
            echo -e "${GREEN}Jupyter Lab starting...${NC}"
            echo "Visit: http://localhost:8888"
            ;;
        8)
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
}

# Quick start option for first-time users
quick_start() {
    echo -e "${CYAN}=€ Quick Start Mode${NC}"
    check_prerequisites
    check_env_file
    build_docker_image
    start_containers
    show_instructions

    read -p "Enter container now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        enter_container
    fi
}

# Parse command line arguments
if [ "$1" == "--quick" ] || [ "$1" == "-q" ]; then
    quick_start
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --quick, -q    Quick start (build and enter)"
    echo "  --menu, -m     Show interactive menu"
    echo "  --help, -h     Show this help"
else
    main_menu
fi