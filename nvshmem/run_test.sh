#!/bin/bash
# NVSHMEM Bandwidth Test Runner Script

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== NVSHMEM GPU Direct RDMA Bandwidth Test Runner ===${NC}\n"

# Check if compiled
if [ ! -f "./nvshmem_bw" ]; then
    echo -e "${YELLOW}Binary not found. Compiling...${NC}"
    make
    if [ $? -ne 0 ]; then
        echo -e "${RED}Compilation failed!${NC}"
        exit 1
    fi
    echo -e "${GREEN}Compilation successful!${NC}\n"
fi

# Check NVSHMEM installation
if [ -z "$NVSHMEM_HOME" ]; then
    echo -e "${YELLOW}Warning: NVSHMEM_HOME not set${NC}"
    echo "Please set NVSHMEM_HOME to your NVSHMEM installation directory"
    echo "Example: export NVSHMEM_HOME=/opt/nvshmem"
    echo ""
fi

# Detect launcher
LAUNCHER=""
NUM_NODES=${NUM_NODES:-2}

if command -v nvshmrun &> /dev/null; then
    LAUNCHER="nvshmrun"
    echo -e "${GREEN}Using nvshmrun launcher${NC}"
elif command -v mpirun &> /dev/null; then
    LAUNCHER="mpirun"
    echo -e "${GREEN}Using mpirun launcher${NC}"
elif command -v srun &> /dev/null; then
    LAUNCHER="srun"
    echo -e "${GREEN}Using srun launcher (SLURM)${NC}"
else
    echo -e "${RED}Error: No suitable launcher found (nvshmrun/mpirun/srun)${NC}"
    exit 1
fi

# Run based on detected launcher
echo -e "Number of processes: ${NUM_NODES}\n"
echo -e "${YELLOW}Starting test...${NC}\n"

case $LAUNCHER in
    "nvshmrun")
        nvshmrun -np $NUM_NODES ./nvshmem_bw
        ;;
    "mpirun")
        # For single node test
        if [ $NUM_NODES -eq 2 ]; then
            mpirun -np 2 ./nvshmem_bw
        else
            # For multi-node, you may need to customize this
            echo -e "${YELLOW}For multi-node test, please customize the hostfile${NC}"
            mpirun -np $NUM_NODES --map-by ppr:1:node ./nvshmem_bw
        fi
        ;;
    "srun")
        # SLURM configuration
        srun -N $NUM_NODES -n $NUM_NODES --ntasks-per-node=1 --gres=gpu:1 ./nvshmem_bw
        ;;
esac

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}Test completed successfully!${NC}"
else
    echo -e "\n${RED}Test failed with exit code $EXIT_CODE${NC}"
fi

exit $EXIT_CODE

