#!/bin/bash
# Setup script for Multi-Agent Digital Twin on Apple M4 Max
# This script sets up the complete environment for running experiments

set -e  # Exit on any error

echo "🍎 Setting up Multi-Agent Digital Twin for Apple M4 Max"
echo "=================================================="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS (Apple M4 Max)"
    echo "For other systems, please use the standard setup instructions"
    exit 1
fi

# Check if we're on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "❌ This script is optimized for Apple Silicon (M1/M2/M3/M4)"
    echo "You may need to modify the setup for Intel Macs"
    exit 1
fi

echo "✅ Detected Apple Silicon Mac"

# Check for Xcode Command Line Tools
echo "🔧 Checking for Xcode Command Line Tools..."
if ! xcode-select -p &> /dev/null; then
    echo "📦 Installing Xcode Command Line Tools..."
    xcode-select --install
    echo "⏳ Please complete the Xcode Command Line Tools installation and re-run this script"
    exit 1
else
    echo "✅ Xcode Command Line Tools found"
fi

# Check for Homebrew
echo "🍺 Checking for Homebrew..."
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for Apple Silicon
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
else
    echo "✅ Homebrew found"
fi

# Install system dependencies
echo "📦 Installing system dependencies..."
brew update
brew install python@3.11 cmake pkg-config openblas lapack gfortran

# Check for conda/miniconda
echo "🐍 Checking for conda..."
if ! command -v conda &> /dev/null; then
    echo "📦 Installing Miniconda for Apple Silicon..."
    
    # Download and install Miniconda for Apple Silicon
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
    bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda3
    rm Miniconda3-latest-MacOSX-arm64.sh
    
    # Initialize conda
    $HOME/miniconda3/bin/conda init zsh
    $HOME/miniconda3/bin/conda init bash
    
    # Source conda
    source $HOME/miniconda3/etc/profile.d/conda.sh
    
    echo "✅ Miniconda installed"
else
    echo "✅ Conda found"
    source $(conda info --base)/etc/profile.d/conda.sh
fi

# Create conda environment
echo "🌍 Creating conda environment..."
if conda env list | grep -q "agenttwin"; then
    echo "⚠️  Environment 'agenttwin' already exists. Removing..."
    conda env remove -n agenttwin -y
fi

conda env create -f environment.yml
echo "✅ Conda environment created"

# Activate environment
echo "🔄 Activating environment..."
conda activate agenttwin

# Install additional pip packages that might not be in conda
echo "📦 Installing additional packages..."
pip install --upgrade pip

# Install PyTorch with Metal Performance Shaders (MPS) support
echo "🔥 Installing PyTorch with MPS support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Verify PyTorch MPS availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"

# Install the project in development mode
echo "🔧 Installing project in development mode..."
pip install -e .

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p experiments
mkdir -p results
mkdir -p logs
mkdir -p checkpoints

# Set up environment variables for Apple Silicon optimization
echo "⚙️  Setting up environment variables..."
cat >> ~/.zshrc << 'EOF'

# Multi-Agent Digital Twin Environment Variables (Apple M4 Max)
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=8  # Optimize for M4 Max performance cores
export MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# CVXPY solver preferences for Apple Silicon
export CVXPY_DEFAULT_SOLVER=OSQP

# Matplotlib backend for Apple Silicon
export MPLBACKEND=MacOSX

EOF

# Source the updated shell configuration
source ~/.zshrc 2>/dev/null || true

# Run quick verification test
echo "🧪 Running verification test..."
python -c "
import numpy as np
import torch
import stable_baselines3
import gymnasium
import cvxpy
import matplotlib.pyplot as plt

print('✅ All core packages imported successfully')
print(f'NumPy version: {np.__version__}')
print(f'PyTorch version: {torch.__version__}')
print(f'Stable-Baselines3 version: {stable_baselines3.__version__}')
print(f'Gymnasium version: {gymnasium.__version__}')
print(f'CVXPY version: {cvxpy.__version__}')

# Test MPS availability
if torch.backends.mps.is_available():
    print('✅ PyTorch MPS (Metal Performance Shaders) is available')
    device = torch.device('mps')
    x = torch.randn(100, 100).to(device)
    y = torch.mm(x, x.t())
    print('✅ MPS computation test passed')
else:
    print('⚠️  MPS not available, using CPU')

print('✅ Verification test completed successfully')
"

# Test the project
echo "🚀 Testing project functionality..."
python -c "
from envs.tep_env import TEPEnvironment
from agents.multi_agent_system import MultiAgentRLSystem
print('✅ Project modules imported successfully')

# Quick environment test
env = TEPEnvironment(scenario='S1')
obs, info = env.reset()
print(f'✅ TEP Environment test passed - observation shape: {obs.shape}')
"

echo ""
echo "🎉 Setup completed successfully!"
echo "=================================================="
echo ""
echo "📋 Next steps:"
echo "1. Activate the environment: conda activate agenttwin"
echo "2. Run quick test: python test_quick_training.py"
echo "3. Run full experiments: make reproduce"
echo "4. Generate results: python -m eval.generate_results"
echo ""
echo "🔧 Apple M4 Max optimizations enabled:"
echo "- PyTorch with Metal Performance Shaders (MPS)"
echo "- NumPy with Accelerate framework"
echo "- Optimized thread counts for M4 Max"
echo "- Apple Silicon native packages"
echo ""
echo "📚 For more information, see README.md"
echo "🐛 If you encounter issues, check the troubleshooting section"
echo ""
echo "Happy experimenting! 🚀"

