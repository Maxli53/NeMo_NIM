#!/bin/bash
#################################################
# Production Stop Script for AI Agents + MoE
# Version: 1.0.0
#################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Stopping Production Services...${NC}"

# Function to stop service
stop_service() {
    local SERVICE_NAME=$1
    local PID_FILE="logs/${SERVICE_NAME}.pid"

    if [ -f "$PID_FILE" ]; then
        PID=$(cat $PID_FILE)
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "Stopping $SERVICE_NAME (PID: $PID)..."
            kill $PID
            sleep 2

            # Force kill if still running
            if ps -p $PID > /dev/null 2>&1; then
                echo -e "${YELLOW}Force stopping $SERVICE_NAME...${NC}"
                kill -9 $PID
            fi

            rm -f $PID_FILE
            echo -e "${GREEN}[✓]${NC} $SERVICE_NAME stopped"
        else
            echo -e "${YELLOW}[!]${NC} $SERVICE_NAME not running"
            rm -f $PID_FILE
        fi
    else
        echo -e "${YELLOW}[!]${NC} No PID file for $SERVICE_NAME"
    fi
}

# Clear GPU cache first
echo -e "\n${BLUE}Clearing GPU cache...${NC}"
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Stop all services
stop_service "moe-backend"
stop_service "monitoring"
stop_service "streamlit"

# Kill any remaining python processes on our ports
echo -e "\n${BLUE}Checking for remaining processes...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8501 | xargs kill -9 2>/dev/null || true
lsof -ti:9090 | xargs kill -9 2>/dev/null || true

echo -e "\n${GREEN}All production services stopped${NC}"

# Show GPU status
echo -e "\n${BLUE}GPU Status:${NC}"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

echo -e "\n${GREEN}Shutdown complete${NC}"