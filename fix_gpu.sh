#!/bin/bash
# Script to diagnose and help fix GPU access issues

set -e

echo "======================================================================"
echo "GPU Fix Diagnostic Script"
echo "======================================================================"
echo ""

# Check if GPU hardware is present
echo "1. Checking GPU hardware..."
if lspci | grep -qi nvidia; then
    echo "   ✓ GPU hardware detected:"
    lspci | grep -i nvidia
else
    echo "   ❌ No NVIDIA GPU hardware found"
    exit 1
fi
echo ""

# Check if NVIDIA driver kernel module is loaded
echo "2. Checking NVIDIA driver kernel module..."
if [ -f /proc/driver/nvidia/version ]; then
    echo "   ✓ NVIDIA driver kernel module is loaded:"
    cat /proc/driver/nvidia/version
    DRIVER_LOADED=true
else
    echo "   ❌ NVIDIA driver kernel module NOT loaded"
    echo "   This is the main issue - the driver needs to be loaded"
    DRIVER_LOADED=false
fi
echo ""

# Check for nvidia-smi
echo "3. Checking nvidia-smi availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✓ nvidia-smi is available"
    echo "   Running nvidia-smi..."
    nvidia-smi
elif [ -f /usr/bin/nvidia-smi ] || [ -f /usr/local/bin/nvidia-smi ]; then
    echo "   ⚠ nvidia-smi exists but not in PATH"
    echo "   Try: /usr/bin/nvidia-smi or /usr/local/bin/nvidia-smi"
else
    echo "   ❌ nvidia-smi not found"
    echo "   You may need to install nvidia-utils"
fi
echo ""

# Check for module system (common on clusters)
echo "4. Checking for module system (common on HPC clusters)..."
if command -v module &> /dev/null; then
    echo "   ✓ Module system detected"
    echo ""
    echo "   Available CUDA/NVIDIA modules:"
    module avail 2>&1 | grep -i -E "cuda|nvidia|gpu" | head -10 || echo "     (none found)"
    echo ""
    echo "   📋 SOLUTION: Try loading CUDA module:"
    echo "      module load cuda"
    echo "      module load cuda/11.8"
    echo "      # Or check: module avail cuda"
else
    echo "   ⚠ No module system (this might be a standalone node)"
fi
echo ""

# Check for system-wide CUDA
echo "5. Checking for system CUDA installation..."
if [ -d /usr/local/cuda ]; then
    echo "   ✓ CUDA found at /usr/local/cuda"
    echo "   Version: $(cat /usr/local/cuda/version.txt 2>/dev/null || echo 'unknown')"
elif [ -d /opt/cuda ]; then
    echo "   ✓ CUDA found at /opt/cuda"
elif [ -n "$CUDA_HOME" ]; then
    echo "   ✓ CUDA_HOME is set: $CUDA_HOME"
else
    echo "   ⚠ No system CUDA installation found"
fi
echo ""

# Check conda environment CUDA
echo "6. Checking conda environment CUDA setup..."
if [ -f ~/miniconda3/envs/ml-benchmark/bin/python ]; then
    ~/miniconda3/envs/ml-benchmark/bin/python -c "import torch; print(f'   PyTorch: {torch.__version__}'); print(f'   CUDA compiled: {torch.version.cuda}'); print(f'   CUDA available: {torch.cuda.is_available()}')"
else
    echo "   ⚠ Conda environment not found"
fi
echo ""

# Provide recommendations
echo "======================================================================"
echo "RECOMMENDATIONS"
echo "======================================================================"
echo ""

if [ "$DRIVER_LOADED" = false ]; then
    echo "❌ MAIN ISSUE: NVIDIA driver kernel module is not loaded"
    echo ""
    echo "SOLUTION OPTIONS (choose based on your environment):"
    echo ""
    echo "OPTION A - HPC/Cluster with Module System:"
    echo "  1. Load CUDA module:"
    echo "     module load cuda"
    echo "     # or: module load cuda/11.8"
    echo ""
    echo "  2. Verify GPU is accessible:"
    echo "     nvidia-smi"
    echo ""
    echo "  3. Re-run benchmark:"
    echo "     python run_benchmark.py"
    echo ""
    echo "OPTION B - Standalone Node (requires sudo/admin):"
    echo "  1. Install NVIDIA driver (if not installed):"
    echo "     sudo apt update"
    echo "     sudo apt install nvidia-driver-XXX  # Replace XXX with version"
    echo "     sudo reboot  # May need reboot"
    echo ""
    echo "  2. Install nvidia-utils:"
    echo "     sudo apt install nvidia-utils-535  # or appropriate version"
    echo ""
    echo "  3. Verify:"
    echo "     nvidia-smi"
    echo ""
    echo "OPTION C - Docker/Container:"
    echo "  1. Run container with GPU support:"
    echo "     docker run --gpus all ..."
    echo "     # or: docker run --device=/dev/nvidia0 ..."
    echo ""
    echo "OPTION D - Check Permissions:"
    echo "  1. Check if user has access:"
    echo "     ls -l /dev/nvidia*"
    echo "  2. If permission denied, may need to add user to group"
    echo ""
else
    echo "✓ Driver is loaded, but PyTorch still can't see GPU"
    echo ""
    echo "Try these steps:"
    echo "  1. Set LD_LIBRARY_PATH (if needed):"
    echo "     export LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    echo ""
    echo "  2. Verify nvidia-smi works"
    echo ""
    echo "  3. Test PyTorch:"
    echo "     python check_gpu.py"
fi

echo ""
echo "======================================================================"
echo "After fixing, verify with:"
echo "  python check_gpu.py"
echo "======================================================================"



