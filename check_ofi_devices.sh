#!/bin/bash
# Quick diagnostic script to check OFI device discovery

echo "════════════════════════════════════════════════════════"
echo "  OFI Device Discovery Diagnostic"
echo "════════════════════════════════════════════════════════"
echo ""

# Load modules
module load PrgEnv-gnu
module load cudatoolkit/12.4
module load craype-accel-nvidia80

# Set environment
export FI_PROVIDER=cxi
export FI_LOG_LEVEL=info
export FI_HMEM_CUDA_ENABLE=1

echo "Environment:"
echo "  FI_PROVIDER=$FI_PROVIDER"
echo "  FI_LOG_LEVEL=$FI_LOG_LEVEL"
echo "  FI_HMEM_CUDA_ENABLE=$FI_HMEM_CUDA_ENABLE"
echo ""

# Check if libfabric is available
echo "Checking libfabric installation:"
if [ -d "/opt/cray/libfabric/1.22.0" ]; then
    echo "  ✓ Found system libfabric: /opt/cray/libfabric/1.22.0"
    if [ -f "/opt/cray/libfabric/1.22.0/lib64/libfabric.so" ]; then
        echo "    ✓ libfabric.so found"
    else
        echo "    ✗ libfabric.so not found"
    fi
else
    echo "  ✗ System libfabric not found"
fi

# Check NCCL bundled libfabric
NCCL_HOME="/pscratch/sd/h/hqin/ldata/benchmark/uccl/thirdparty/nccl-sg"
if [ -d "${NCCL_HOME}/build/lib" ]; then
    echo "  ✓ Found NCCL bundled libfabric: ${NCCL_HOME}/build/lib"
    if [ -f "${NCCL_HOME}/build/lib/libfabric.so" ]; then
        echo "    ✓ libfabric.so found"
    fi
fi
echo ""

# Try to query libfabric directly
echo "Querying libfabric for CXI provider:"
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:${LD_LIBRARY_PATH}

# Use fi_info if available
if command -v fi_info &> /dev/null; then
    echo "Running: fi_info -p cxi"
    fi_info -p cxi 2>&1 | head -20
else
    echo "  fi_info command not found, trying Python test..."
    
    # Simple Python test using ctypes
    python3 << 'PYEOF'
import ctypes
import os

# Try to load libfabric
libfabric_paths = [
    "/opt/cray/libfabric/1.22.0/lib64/libfabric.so",
    "/usr/lib64/libfabric.so",
    "libfabric.so"
]

libfabric = None
for path in libfabric_paths:
    try:
        libfabric = ctypes.CDLL(path)
        print(f"  ✓ Loaded libfabric from: {path}")
        break
    except OSError:
        continue

if libfabric is None:
    print("  ✗ Failed to load libfabric")
    exit(1)

# Try to call fi_getinfo
try:
    # Define FI_VERSION macro
    FI_VERSION = lambda major, minor: ((major << 16) | minor)
    
    # Define structures (simplified)
    class fi_info(ctypes.Structure):
        pass
    
    fi_info._fields_ = [
        ("next", ctypes.POINTER(fi_info)),
        ("fabric_attr", ctypes.c_void_p),
        ("domain_attr", ctypes.c_void_p),
        ("ep_attr", ctypes.c_void_p),
        ("tx_attr", ctypes.c_void_p),
        ("rx_attr", ctypes.c_void_p),
    ]
    
    # Get fi_getinfo function
    libfabric.fi_getinfo.argtypes = [
        ctypes.c_uint64,  # version
        ctypes.c_char_p,  # node
        ctypes.c_char_p,  # service
        ctypes.c_uint64,  # flags
        ctypes.c_void_p,  # hints
        ctypes.POINTER(ctypes.POINTER(fi_info))  # info
    ]
    libfabric.fi_getinfo.restype = ctypes.c_int
    
    # Try to call fi_getinfo with CXI provider
    info_ptr = ctypes.POINTER(fi_info)()
    ret = libfabric.fi_getinfo(FI_VERSION(1, 6), None, None, 0, None, ctypes.byref(info_ptr))
    
    if ret == 0:
        print(f"  ✓ fi_getinfo succeeded (ret={ret})")
        if info_ptr:
            print("  ✓ Found at least one provider")
        else:
            print("  ⚠ fi_getinfo returned NULL info")
    else:
        print(f"  ✗ fi_getinfo failed (ret={ret})")
        # Try to get error string
        try:
            libfabric.fi_strerror.argtypes = [ctypes.c_int]
            libfabric.fi_strerror.restype = ctypes.c_char_p
            err_str = libfabric.fi_strerror(-ret)
            print(f"    Error: {err_str.decode() if err_str else 'unknown'}")
        except:
            pass
except Exception as e:
    print(f"  ✗ Python test failed: {e}")

PYEOF
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Checking UCCL OFI Device Discovery"
echo "════════════════════════════════════════════════════════"

# Compile and run a simple test
cd /global/homes/h/hqin/ldata/benchmark/uccl/collective/ofi

if [ -f "util_ofi_device_name_list_test" ]; then
    echo "Running UCCL OFI device discovery test:"
    export UCCL_USE_LIBFABRIC=1
    export UCCL_DISABLE_IB=1
    ./util_ofi_device_name_list_test 2>&1 | head -30
else
    echo "Test binary not found, compiling..."
    make util_ofi_device_name_list_test 2>&1 | tail -10
    if [ -f "util_ofi_device_name_list_test" ]; then
        echo "Running test:"
        export UCCL_USE_LIBFABRIC=1
        export UCCL_DISABLE_IB=1
        ./util_ofi_device_name_list_test 2>&1 | head -30
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Testing fi_getinfo() directly"
echo "════════════════════════════════════════════════════════"

# Compile and run simple test
cd /global/homes/h/hqin/ldata/benchmark
if [ -f "test_fi_getinfo" ]; then
    echo "Running test_fi_getinfo:"
    ./test_fi_getinfo 2>&1
else
    echo "Compiling test_fi_getinfo..."
    gcc -o test_fi_getinfo test_fi_getinfo.c -I/opt/cray/libfabric/1.22.0/include -L/opt/cray/libfabric/1.22.0/lib64 -lfabric 2>&1 | tail -5
    if [ -f "test_fi_getinfo" ]; then
        echo "Running test:"
        ./test_fi_getinfo 2>&1
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Summary"
echo "════════════════════════════════════════════════════════"
echo ""
echo "⚠️  IMPORTANT: This script was run on a LOGIN NODE"
echo ""
echo "CXI devices are ONLY available on COMPUTE NODES."
echo "To properly test OFI device discovery:"
echo ""
echo "  1. Submit a job to a compute node:"
echo "     srun --nodes=1 --ntasks=1 --time=00:10:00 -C gpu --gpus-per-task=1 -A dasrepo -q debug --pty bash"
echo ""
echo "  2. Then run this script again, or run:"
echo "     ./test_fi_getinfo"
echo ""
echo "  3. Or run the full benchmark:"
echo "     ./run_uccl.sh"
echo ""
echo "Expected behavior:"
echo "  - Login node: No CXI devices found (expected)"
echo "  - Compute node: Should find CXI devices"
echo ""

