#!/bin/bash
# NVSHMEM Test Runner with PMI2 Bootstrap (for SLURM, no MPI needed)

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== NVSHMEM PMI2 Bootstrap Test Runner ===${NC}\n"

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

# Configuration
NUM_PES=${NUM_PES:-2}

echo -e "${GREEN}Configuration:${NC}"
echo "  Number of PEs: $NUM_PES"
echo "  Bootstrap method: PMI2 (via SLURM)"
echo "  Launcher: srun"
echo ""

# Check if we're in a SLURM allocation
if [ -z "$SLURM_JOB_ID" ]; then
    echo -e "${YELLOW}Not in a SLURM allocation. Using srun directly...${NC}"
    echo -e "${YELLOW}Note: This may request an interactive allocation.${NC}\n"
    
    # Use srun directly (will allocate resources)
    echo -e "${GREEN}Running: srun -n $NUM_PES --gpus-per-task=1 ./nvshmem_bw${NC}\n"
    srun -n $NUM_PES --gpus-per-task=1 ./nvshmem_bw
    EXIT_CODE=$?
else
    echo -e "${GREEN}In SLURM allocation: $SLURM_JOB_ID${NC}"
    echo -e "${GREEN}Running: srun -n $NUM_PES ./nvshmem_bw${NC}\n"
    srun -n $NUM_PES ./nvshmem_bw
    EXIT_CODE=$?
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}Test completed successfully!${NC}"
else
    echo -e "\n${RED}Test failed with exit code $EXIT_CODE${NC}"
fi

exit $EXIT_CODE



