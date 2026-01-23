#!/bin/bash

echo "════════════════════════════════════════════════════════"
echo "  UCCL OFI Backend Performance Test Suite"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Strategy: UCCL with libfabric/CXI backend (Perlmutter Slingshot)"
echo "  - Python benchmark using uccl.p2p module"
echo "  - libfabric CXI provider for GPU Direct RDMA"
echo "  - Busy polling for low latency"
echo ""
echo "Test Suite:"
echo "  1. Unidirectional Bandwidth (Rank 0 -> Rank 1)"
echo "  2. Bidirectional Bandwidth (--dual option)"
echo "  3. IPC Benchmark (--ipc option, same node)"
echo ""

# Load required modules
module load PrgEnv-gnu
module load cudatoolkit/12.4
module load craype-accel-nvidia80

# Set paths
UCCL_ROOT="/global/homes/h/hqin/ldata/benchmark/uccl"
OFI_DIR="${UCCL_ROOT}/collective/ofi"
BENCHMARK_SCRIPT="${UCCL_ROOT}/p2p/benchmarks/benchmark_uccl.py"

# Use NCCL's bundled libfabric version (same as bwl.sh)
NCCL_OFI_LIB=/global/common/software/nersc9/nccl/2.24.3/plugin/deps/lib
SYSTEM_OFI_INCLUDE=/opt/cray/libfabric/1.22.0/include
SYSTEM_OFI_LIB=/opt/cray/libfabric/1.22.0/lib64

# Conda environment
CONDA_PREFIX="/global/homes/h/hqin/miniconda3/envs/bm"

echo "════════════════════════════════════════════════════════"
echo "  Compiling UCCL OFI Backend"
echo "════════════════════════════════════════════════════════"

# Set up build environment
export CUDA_HOME=${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4}
export OFI_HOME=/opt/cray/libfabric/1.22.0
export CONDA_PREFIX="${CONDA_PREFIX}"

# Define build_p2p function locally for OFI support
build_p2p_local() {
    local TARGET="cuda"
    local ARCH="$(uname -m)"
    local IS_EFA=""

    echo "[local] build_p2p Target: $TARGET"

    # Check if OFI backend should be enabled
    if [[ -n "${UCCL_USE_LIBFABRIC:-}" && "${UCCL_USE_LIBFABRIC}" == "1" ]]; then
        echo "[local] Building p2p with OFI backend support (libfabric/CXI)"
        export UCCL_USE_LIBFABRIC=1
        export UCCL_DISABLE_IB=1
    else
        echo "[local] Building p2p without OFI backend (standard IB Verbs)"
        unset UCCL_USE_LIBFABRIC
        unset UCCL_DISABLE_IB
    fi

    cd "${UCCL_ROOT}/p2p"
    make clean > /dev/null 2>&1
    make -j$(nproc)
    cd - > /dev/null
}


# Build OFI backend shared library (if needed)
if [ -f "${OFI_DIR}/libnccl-net-ofi.so" ]; then
    echo "✓ Shared library already exists: ${OFI_DIR}/libnccl-net-ofi.so"
    ls -lh "${OFI_DIR}/libnccl-net-ofi.so"
else
    echo "Compiling libnccl-net-ofi.so..."
    cd "${OFI_DIR}"
    
    # Compile using make (OFI backend doesn't have a build.sh wrapper)
    make libnccl-net-ofi.so
    
    if [ $? -ne 0 ]; then
        echo "✗ Compilation failed!"
        exit 1
    fi
    
    echo "✓ Compilation successful"
    ls -lh "${OFI_DIR}/libnccl-net-ofi.so"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Environment Configuration"
echo "════════════════════════════════════════════════════════"

# Set runtime environment
# Priority: NCCL bundled libfabric > System libfabric > Conda libs
export LD_LIBRARY_PATH=${NCCL_OFI_LIB}:${SYSTEM_OFI_LIB}:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# Libfabric configuration (same as bwl.sh)
export FI_PROVIDER=cxi
export FI_LOG_LEVEL=error
export FI_HMEM_CUDA_ENABLE=1
export FI_CXI_DISABLE_HMEM_DEV_REGISTER=0

# GPU support
export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=nvidia80

# CXI specific settings
export FI_CXI_RX_MATCH_MODE=hardware
export FI_MR_CACHE_MONITOR=memhooks

# UCCL specific settings
export UCCL_USE_LIBFABRIC=1
export UCCL_DISABLE_IB=1  # Disable IB Verbs, use OFI backend

# Optional: Set OFI device names explicitly if auto-discovery fails
# export UCCL_OFI_DEVICES=cxi0,cxi1
# export UCCL_OFI_PROVIDER=cxi

# Debugging: Enable verbose logging for UCCL (optional)
# Set to 0 for minimal logs, 1 for INFO, 2 for DEBUG
export GLOG_minloglevel=0  # Show INFO and above
export GLOG_logtostderr=1   # Log to stderr for visibility

# Ensure Python can see all environment variables
# These will be passed via --export=ALL in srun

# Python environment
export CONDA_PREFIX="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

echo "Provider:         ${FI_PROVIDER:-cxi}"
echo "Libfabric:        NCCL bundled (v1.23.1) + System (v1.22.0)"
echo "GPU Direct RDMA:  ${FI_HMEM_CUDA_ENABLE:-Enabled}"
echo "OFI Backend:      ${UCCL_USE_LIBFABRIC:-Enabled}"
echo "IB Verbs:         ${UCCL_DISABLE_IB:-Disabled}"
echo ""
echo "Key Environment Variables:"
echo "  UCCL_USE_LIBFABRIC=${UCCL_USE_LIBFABRIC}"
echo "  UCCL_DISABLE_IB=${UCCL_DISABLE_IB}"
echo "  FI_PROVIDER=${FI_PROVIDER}"
echo "  FI_HMEM_CUDA_ENABLE=${FI_HMEM_CUDA_ENABLE}"
echo "  FI_CXI_RX_MATCH_MODE=${FI_CXI_RX_MATCH_MODE}"
echo "  GLOG_minloglevel=${GLOG_minloglevel:-0}"
echo ""

# Rebuild Python p2p module with OFI support if UCCL_USE_LIBFABRIC=1
# This must be done AFTER environment variables are set
# Use local compilation method that works on Perlmutter
if [ -n "$UCCL_USE_LIBFABRIC" ] && [ "$UCCL_USE_LIBFABRIC" = "1" ]; then
    echo "════════════════════════════════════════════════════════"
    echo "  Compiling Python P2P Module with OFI Support"
    echo "════════════════════════════════════════════════════════"

    echo "UCCL_USE_LIBFABRIC=1 detected, rebuilding p2p module with OFI support..."

    # Use local compilation method (build_p2p_local function defined above)
    # This compiles directly without container builds
    build_p2p_local

    if [ $? -ne 0 ]; then
        echo "✗ P2P module compilation failed!"
        exit 1
    fi

    echo "✓ P2P module compiled successfully with OFI support"
    ls -lh "${UCCL_ROOT}/p2p/p2p"*.so 2>/dev/null | head -1
    echo ""

    # Install the compiled UCCL package to Python environment
    echo "Installing compiled UCCL package to Python environment..."

    # First uninstall any existing uccl package
    pip uninstall uccl -y || true

    # Install using pip install -e (works with the compiled .so files)
    pip install -e "${UCCL_ROOT}" --no-deps

    if [ $? -ne 0 ]; then
        echo "✗ UCCL package installation failed!"
        exit 1
    fi

    echo "✓ UCCL package installed successfully"
fi

echo "════════════════════════════════════════════════════════"
echo "  Test Configuration"
echo "════════════════════════════════════════════════════════"
echo "Nodes:            2"
echo "Tasks:            2 (1 per node)"
echo "GPUs per task:    1"
echo ""

# Parse command line arguments
BENCHMARK_MODE="standard"
SIZES="256,1024,4096,16384,65536,262144,1048576,10485760,16777216,104857600"
ITERS=1000
NUM_KVBLOCKS=1
ASYNC_API=""
DUAL=""
IPC=""
LOCAL_GPU_IDX=0
NUM_CPUS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            BENCHMARK_MODE="$2"
            shift 2
            ;;
        --sizes)
            SIZES="$2"
            shift 2
            ;;
        --iters)
            ITERS="$2"
            shift 2
            ;;
        --num-kvblocks)
            NUM_KVBLOCKS="$2"
            shift 2
            ;;
        --async)
            ASYNC_API="--async-api"
            shift
            ;;
        --dual)
            DUAL="--dual"
            shift
            ;;
        --ipc)
            IPC="--ipc"
            shift
            ;;
        --local-gpu-idx)
            LOCAL_GPU_IDX="$2"
            shift 2
            ;;
        --num-cpus)
            NUM_CPUS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --mode MODE          Benchmark mode: standard, dual, ipc (default: standard)"
            echo "  --sizes SIZES        Comma-separated message sizes (default: 256,1024,...)"
            echo "  --iters ITERS        Iterations per message size (default: 1000)"
            echo "  --num-kvblocks N     Number of key-value blocks (default: 1)"
            echo "  --async              Use asynchronous API"
            echo "  --dual               Run bidirectional benchmark"
            echo "  --ipc                Run IPC benchmark (same node)"
            echo "  --local-gpu-idx IDX  Local GPU index (default: 0)"
            echo "  --num-cpus N         Number of CPU threads (default: 4)"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Standard benchmark"
            echo "  $0 --dual                             # Bidirectional benchmark"
            echo "  $0 --ipc                              # IPC benchmark (single node)"
            echo "  $0 --sizes 1048576,16777216 --iters 100  # Custom sizes and iterations"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set mode flags based on mode argument
if [ "$BENCHMARK_MODE" = "dual" ]; then
    DUAL="--dual"
elif [ "$BENCHMARK_MODE" = "ipc" ]; then
    IPC="--ipc"
fi

if [ -n "$IPC" ]; then
    echo "Benchmark mode:   IPC (same node, Unix Domain Sockets)"
    echo "  - Iterations:   ${ITERS}"
    echo "  - Message size: ${SIZES}"
elif [ -n "$DUAL" ]; then
    echo "Benchmark mode:   Bidirectional"
    echo "  - Iterations:   ${ITERS}"
    echo "  - Message size: ${SIZES}"
    echo "  - KV blocks:    ${NUM_KVBLOCKS}"
else
    echo "Benchmark mode:   Standard (unidirectional)"
    echo "  - Iterations:   ${ITERS}"
    echo "  - Message size: ${SIZES}"
    echo "  - KV blocks:    ${NUM_KVBLOCKS}"
fi

if [ -n "$ASYNC_API" ]; then
    echo "  - API:          Asynchronous"
else
    echo "  - API:          Synchronous"
fi

echo ""

# Verify benchmark script exists
if [ ! -f "${BENCHMARK_SCRIPT}" ]; then
    echo "✗ Benchmark script not found: ${BENCHMARK_SCRIPT}"
    exit 1
fi

# Verify Python can import uccl.p2p
echo "════════════════════════════════════════════════════════"
echo "  Verifying Python Environment"
echo "════════════════════════════════════════════════════════"

# Python will use the installed UCCL package from pip install -e

python3 << 'PYTHON_EOF'
import sys
import os

# Set up LD_LIBRARY_PATH for glog and libfabric
if "CONDA_PREFIX" in os.environ:
    conda_lib = os.path.join(os.environ["CONDA_PREFIX"], "lib")
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if conda_lib not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{current_ld_path}" if current_ld_path else conda_lib

# Ensure UCCL environment variables are set
# These should be inherited from parent shell, but verify
required_vars = {
    "UCCL_USE_LIBFABRIC": "1",
    "UCCL_DISABLE_IB": "1",
    "FI_PROVIDER": "cxi",
    "FI_HMEM_CUDA_ENABLE": "1",
}

for var, default_value in required_vars.items():
    if var not in os.environ:
        os.environ[var] = default_value
        print(f"⚠ Set {var}={default_value} (was not set)")

# Verify critical environment variables
print("Environment check:")
print(f"  UCCL_USE_LIBFABRIC={os.environ.get('UCCL_USE_LIBFABRIC', 'NOT SET')}")
print(f"  UCCL_DISABLE_IB={os.environ.get('UCCL_DISABLE_IB', 'NOT SET')}")
print(f"  FI_PROVIDER={os.environ.get('FI_PROVIDER', 'NOT SET')}")
print(f"  FI_HMEM_CUDA_ENABLE={os.environ.get('FI_HMEM_CUDA_ENABLE', 'NOT SET')}")
print(f"  LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', 'NOT SET')[:100]}...")  # Truncate for display
print(f"  Python sys.path (first 3): {sys.path[:3]}")

# Try to import uccl.p2p from the installed package
try:
    from uccl import p2p
    print(f"✓ Successfully imported uccl.p2p from: {p2p.__file__}")
except ImportError as e:
    print(f"✗ Failed to import uccl.p2p: {e}")
    print(f"  Python path: {sys.path}")
    sys.exit(1)
PYTHON_EOF

if [ $? -ne 0 ]; then
    echo "✗ Python environment check failed"
    exit 1
fi

echo ""

echo "════════════════════════════════════════════════════════"
echo "  Starting Test"
echo "════════════════════════════════════════════════════════"
echo ""

# Explicitly re-export critical variables before srun (ensures they're available in compute nodes)
# This is a backup in case --export=ALL doesn't work as expected
export UCCL_USE_LIBFABRIC=1
export UCCL_DISABLE_IB=1
export FI_PROVIDER=cxi
export FI_LOG_LEVEL=error
export FI_HMEM_CUDA_ENABLE=1
export FI_CXI_DISABLE_HMEM_DEV_REGISTER=0
export FI_CXI_RX_MATCH_MODE=hardware
export FI_MR_CACHE_MONITOR=memhooks
export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=nvidia80
export GLOG_minloglevel=0
export GLOG_logtostderr=1

# Python will use the installed UCCL package from pip install -e
export UCCL_ROOT="${UCCL_ROOT}"

# Determine srun parameters based on mode
# --export=ALL ensures all environment variables are passed to compute nodes
if [ -n "$IPC" ]; then
    # IPC benchmark: single node, 2 tasks
    SRUN_CMD="srun --nodes=1 --ntasks=2 --ntasks-per-node=2 --time=00:15:00 -C gpu --gpus-per-task=1 -A dasrepo -q debug -u --export=ALL"
else
    # Standard/Dual benchmark: 2 nodes, 1 task per node
    SRUN_CMD="srun --nodes=2 --ntasks-per-node=1 --ntasks=2 --time=00:15:00 -C gpu --gpus-per-task=1 -A dasrepo -q debug -u --export=ALL"
fi

# Build Python command with environment variable check
PYTHON_CMD="python3 ${BENCHMARK_SCRIPT} \
    --local-gpu-idx ${LOCAL_GPU_IDX} \
    --num-cpus ${NUM_CPUS} \
    --device gpu \
    --sizes ${SIZES} \
    --iters ${ITERS} \
    --num-kvblocks ${NUM_KVBLOCKS} \
    ${ASYNC_API} \
    ${DUAL} \
    ${IPC}"

echo "Running: ${SRUN_CMD} ${PYTHON_CMD}"
echo ""

# Create wrapper script in a shared filesystem location (scratch or home)
# Use scratch if available, otherwise use home directory
if [ -d "/pscratch/sd/h/hqin" ]; then
    WRAPPER_DIR="/pscratch/sd/h/hqin/ldata/benchmark"
elif [ -d "${HOME}/ldata/benchmark" ]; then
    WRAPPER_DIR="${HOME}/ldata/benchmark"
else
    WRAPPER_DIR="${HOME}"
fi
WRAPPER_SCRIPT="${WRAPPER_DIR}/.uccl_wrapper_$$.sh"

cat > "${WRAPPER_SCRIPT}" << 'WRAPPER_EOF'
#!/bin/bash
echo "════════════════════════════════════════════════════════"
echo "  Environment Variables on Compute Node"
echo "════════════════════════════════════════════════════════"
echo "Hostname: $(hostname)"
echo "PID: $$"
echo ""
echo "Critical UCCL/OFI Environment Variables:"
echo "  UCCL_USE_LIBFABRIC=${UCCL_USE_LIBFABRIC:-NOT SET}"
echo "  UCCL_DISABLE_IB=${UCCL_DISABLE_IB:-NOT SET}"
echo "  FI_PROVIDER=${FI_PROVIDER:-NOT SET}"
echo "  FI_HMEM_CUDA_ENABLE=${FI_HMEM_CUDA_ENABLE:-NOT SET}"
echo "  FI_CXI_RX_MATCH_MODE=${FI_CXI_RX_MATCH_MODE:-NOT SET}"
echo "  GLOG_minloglevel=${GLOG_minloglevel:-NOT SET}"
echo "  GLOG_logtostderr=${GLOG_logtostderr:-NOT SET}"
echo ""
echo "LD_LIBRARY_PATH (first 200 chars):"
echo "  ${LD_LIBRARY_PATH:0:200}..."
echo ""
echo "════════════════════════════════════════════════════════"
echo ""

# Execute the actual Python command with all arguments
exec "$@"
WRAPPER_EOF
chmod +x "${WRAPPER_SCRIPT}"

# Run with wrapper script - pass Python command and all arguments
${SRUN_CMD} "${WRAPPER_SCRIPT}" python3 ${BENCHMARK_SCRIPT} \
    --local-gpu-idx ${LOCAL_GPU_IDX} \
    --num-cpus ${NUM_CPUS} \
    --device gpu \
    --sizes ${SIZES} \
    --iters ${ITERS} \
    --num-kvblocks ${NUM_KVBLOCKS} \
    ${ASYNC_API} \
    ${DUAL} \
    ${IPC}

EXIT_CODE=$?

# Cleanup wrapper script (if it exists)
rm -f "${WRAPPER_SCRIPT}" 2>/dev/null || true

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Test Complete"
echo "════════════════════════════════════════════════════════"
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Test completed successfully"
    echo ""
    echo "Key Findings:"
    echo "  - UCCL OFI backend compiled and functional"
    echo "  - GPU Direct RDMA enabled via libfabric CXI"
    echo "  - Busy polling for low latency"
    echo ""
    echo "Performance Notes:"
    echo "  - Compare bandwidth with NCCL and MPI"
    echo "  - Latency should be competitive with NCCL (~2-3 µs)"
    echo "  - Peak bandwidth typically at large message sizes (>1MB)"
else
    echo "✗ Test failed (exit code: $EXIT_CODE)"
    echo ""
    echo "Possible causes:"
    echo "  - Resource allocation failure"
    echo "  - Network configuration issue"
    echo "  - GPU access problem"
    echo "  - CXI provider unavailable"
    echo "  - Python module import error"
    echo "  - UCCL backend initialization failure"
fi

echo ""

