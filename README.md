# Multi-Agent Digital Twin

This guide provides instructions for setting up and running the Multi-Agent Digital Twin.

## 🚀 Quick Start

### One-Command Setup
```bash
# Clone the repository
git clone https://github.com/alqithami/agenttwin.git
cd agenttwin

# Run Apple M4 Max optimized setup
make apple-setup
```

### Manual Setup
If you prefer manual setup:

```bash
# 1. Install Xcode Command Line Tools (if not already installed)
xcode-select --install

# 2. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 3. Install system dependencies
brew install python@3.11 cmake pkg-config openblas lapack gfortran

# 4. Create conda environment
conda env create -f environment.yml
conda activate agenttwin

# 5. Install the package
pip install -e .
```

## 🧪 Verification

Test your installation:

```bash
# Activate environment
conda activate agenttwin

# Run verification
make verify

# Run performance benchmark
make benchmark

# Run quick functionality test
make test
```

Expected output for successful setup:
```
✅ NumPy 1.24.3
✅ PyTorch 2.0.1
✅ MPS (Metal Performance Shaders) available
✅ Stable-Baselines3 2.0.0
✅ Gymnasium 0.29.1
✅ CVXPY 1.3.2
✅ TEP Environment module
✅ Multi-Agent RL System module
```

## 🏃‍♂️ Running Experiments

### Quick Test 
```bash
make reproduce-quick
```

### Full Reproduction 
```bash
make reproduce
```

### Individual Components
```bash
# Training only
make train

# Evaluation only
make eval

# Results generation only
make results
```


## 🔧 Specific Settings

The setup automatically configures:

```bash
# Environment variables for optimal performance
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=8  # Optimized for M4 Max performance cores
export MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export CVXPY_DEFAULT_SOLVER=OSQP
export MPLBACKEND=MacOSX
```

## 🐛 Troubleshooting

### Common Issues

#### MPS Not Available
```bash
# Check MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# If False, reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### CVXPY Solver Issues
```bash
# Install additional solvers
pip install cvxopt scs ecos
```

#### Memory Issues
```bash
# Reduce batch size in training config
# Edit agents/multi_agent_system.py
# Change batch_size from 256 to 128
```

#### Conda Environment Issues
```bash
# Remove and recreate environment
conda env remove -n agenttwin
conda env create -f environment.yml
```

### Performance Issues

If performance is slower than expected:

1. **Check MPS Usage**:
   ```bash
   # Monitor GPU usage
   sudo powermetrics --samplers gpu_power -n 1 -i 1000
   ```

2. **Verify Thread Settings**:
   ```bash
   python -c "import torch; print(f'Threads: {torch.get_num_threads()}')"
   ```

3. **Check Memory Pressure**:
   ```bash
   # Monitor memory usage
   memory_pressure
   ```


## 🔄 Updates and Maintenance

Keep your environment updated:

```bash
# Update conda packages
conda update --all

# Update pip packages
pip install --upgrade -r requirements.txt

# Update the project
git pull origin main
pip install -e .
```

---

**Happy experimenting! 🚀**

