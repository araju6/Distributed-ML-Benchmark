#!/bin/bash
# Setup script for ML Compiler Benchmark Framework
# This script installs conda (if needed) and sets up the conda environment

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
CONDA_INSTALL_DIR="$HOME/miniconda3"
CONDA_BIN="$CONDA_INSTALL_DIR/bin/conda"

echo "======================================================================"
echo "ML Compiler Benchmark Framework - Setup Script"
echo "======================================================================"
echo ""

# Check if conda is already installed
if [ ! -f "$CONDA_BIN" ]; then
    echo "Conda not found. Installing Miniconda..."
    
    # Check for installer script
    INSTALLER="$PROJECT_DIR/Miniconda3-latest-Linux-x86_64.sh"
    if [ ! -f "$INSTALLER" ]; then
        echo "Downloading Miniconda installer..."
        cd "$PROJECT_DIR"
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        INSTALLER="$PROJECT_DIR/Miniconda3-latest-Linux-x86_64.sh"
        chmod +x "$INSTALLER"
    fi
    
    # Install Miniconda
    echo "Installing Miniconda to $CONDA_INSTALL_DIR..."
    bash "$INSTALLER" -b -p "$CONDA_INSTALL_DIR"
    
    # Initialize conda
    echo "Initializing conda..."
    "$CONDA_BIN" init bash
    
    echo "Miniconda installed successfully!"
else
    echo "Conda already installed at $CONDA_INSTALL_DIR"
fi

# Add conda to PATH for this script
export PATH="$CONDA_INSTALL_DIR/bin:$PATH"

# Source conda.sh if it exists
if [ -f "$CONDA_INSTALL_DIR/etc/profile.d/conda.sh" ]; then
    source "$CONDA_INSTALL_DIR/etc/profile.d/conda.sh"
fi

# Accept conda Terms of Service if needed
echo ""
echo "Accepting conda Terms of Service..."
"$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
"$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Navigate to project directory
cd "$PROJECT_DIR"

# Check if environment.yml exists
if [ ! -f "environment.yml" ]; then
    echo "Error: environment.yml not found in $PROJECT_DIR"
    exit 1
fi

# Remove existing environment if it exists (for clean reinstall)
ENV_NAME=$(grep "^name:" environment.yml | cut -d' ' -f2)
if conda env list | grep -q "^$ENV_NAME "; then
    echo ""
    echo "Removing existing environment '$ENV_NAME' for clean reinstall..."
    conda env remove -n "$ENV_NAME" -y || true
fi

# Create conda environment from environment.yml
echo ""
echo "Creating conda environment from environment.yml..."
echo "This may take several minutes as it downloads large packages (PyTorch, CUDA, etc.)..."
"$CONDA_BIN" env create -f environment.yml

# Verify environment was created
if conda env list | grep -q "^$ENV_NAME "; then
    # Check for GPU availability
    echo ""
    echo "Checking GPU availability..."
    "$CONDA_INSTALL_DIR/envs/$ENV_NAME/bin/python" "$PROJECT_DIR/check_gpu.py" 2>/dev/null || echo "  (Skipping GPU check - run check_gpu.py manually)"
    
    echo ""
    echo "======================================================================"
    echo "✓ Setup complete!"
    echo "======================================================================"
    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate $ENV_NAME"
    echo ""
    echo "Or to run the benchmark directly:"
    echo "  $CONDA_INSTALL_DIR/envs/$ENV_NAME/bin/python $PROJECT_DIR/run_benchmark.py"
    echo ""
    echo "To check GPU status:"
    echo "  $CONDA_INSTALL_DIR/envs/$ENV_NAME/bin/python $PROJECT_DIR/check_gpu.py"
    echo ""
    echo "To use conda in new terminal sessions, reload your shell or run:"
    echo "  source ~/.bashrc"
    echo ""
else
    echo "Error: Environment '$ENV_NAME' was not created successfully"
    exit 1
fi

