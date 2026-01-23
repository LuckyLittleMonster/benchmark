#!/bin/bash
# NVSHMEM Test Runner with File-based Bootstrap (No MPI required)

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== NVSHMEM File-based Bootstrap Test Runner ===${NC}\n"

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

# Clean up old bootstrap files
echo -e "${YELLOW}Cleaning old bootstrap files...${NC}"
rm -f nvshmem_bootstrap*

# Configuration
NUM_PES=${NUM_PES:-2}
BOOTSTRAP_FILE="$(pwd)/nvshmem_bootstrap"

# Export NVSHMEM environment variables
export NVSHMEM_BOOTSTRAP="PLUGIN"
export NVSHMEM_BOOTSTRAP_PLUGIN="file"
export NVSHMEM_BOOTSTRAP_FILE="$BOOTSTRAP_FILE"
export NVSHMEM_BOOTSTRAP_TWO_STAGE=1

echo -e "${GREEN}Configuration:${NC}"
echo "  Number of PEs: $NUM_PES"
echo "  Bootstrap file: $BOOTSTRAP_FILE"
echo "  Bootstrap method: File-based (no MPI)"
echo ""

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    pkill -P $$ nvshmem_bw 2>/dev/null
    rm -f nvshmem_bootstrap*
    exit
}

trap cleanup SIGINT SIGTERM

# Launch processes
echo -e "${YELLOW}Starting NVSHMEM processes...${NC}\n"

PIDS=()

for ((i=0; i<NUM_PES; i++)); do
    # Assign different GPUs to different PEs
    GPU_ID=$((i % $(nvidia-smi -L | wc -l)))
    
    echo -e "${GREEN}Starting PE $i on GPU $GPU_ID...${NC}"
    CUDA_VISIBLE_DEVICES=$GPU_ID ./nvshmem_bw &
    PID=$!
    PIDS+=($PID)
    echo "  PID: $PID"
    
    # Small delay between launches
    sleep 0.5
done

echo -e "\n${YELLOW}All processes started. Waiting for completion...${NC}\n"

# Wait for all processes to complete
FAILED=0
for ((i=0; i<NUM_PES; i++)); do
    wait ${PIDS[$i]}
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "${RED}PE $i (PID ${PIDS[$i]}) failed with exit code $EXIT_CODE${NC}"
        FAILED=1
    fi
done

# Cleanup
rm -f nvshmem_bootstrap*

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All tests completed successfully!${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed!${NC}"
    exit 1
fi

