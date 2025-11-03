#!/bin/bash
# Startup script to initialize environment for ML Compiler Benchmark Framework

echo "======================================================================"
echo "Initializing ML Compiler Benchmark Environment"
echo "======================================================================"
echo ""

# Source bashrc to get conda initialized
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
    echo "✓ Sourced ~/.bashrc"
else
    echo "⚠ ~/.bashrc not found"
fi

# Ensure conda is in PATH
export PATH="$HOME/miniconda3/bin:$PATH"

# Source conda.sh if it exists
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    echo "✓ Initialized conda"
fi

# Activate the ml-benchmark environment
if conda env list | grep -q "ml-benchmark"; then
    conda activate ml-benchmark
    echo "✓ Activated conda environment: ml-benchmark"
else
    echo "⚠ ml-benchmark environment not found"
    echo "  Run ./setup.sh to create it"
    exit 1
fi

# Use the environment's Python explicitly
ENV_PYTHON="$HOME/miniconda3/envs/ml-benchmark/bin/python"

# Check Python and PyTorch
echo ""
echo "Environment Check:"
echo "  Python: $($ENV_PYTHON --version 2>&1)"
$ENV_PYTHON -c "import torch; print('  PyTorch:', torch.__version__); print('  CUDA available:', torch.cuda.is_available())" 2>/dev/null || echo "  ⚠ PyTorch not available"

# Check GPU status
echo ""
echo "GPU Status:"
$ENV_PYTHON check_gpu.py 2>&1 | grep -E "(CUDA available|Using GPU|GPU not detected)" | head -3 || echo "  (Run python check_gpu.py for full status)"

# Change to project directory
cd "$(dirname "${BASH_SOURCE[0]}")"
echo ""
echo "======================================================================"
echo "Environment ready!"
echo "======================================================================"
echo ""
echo "Current directory: $(pwd)"
echo "Conda environment: ml-benchmark"
echo ""
echo "You can now run:"
echo "  python run_benchmark.py"
echo "  python check_gpu.py"
echo "  python analyze_results.py"
echo "======================================================================"

