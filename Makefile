# Makefile for Multi-Agent Digital Twin
# Optimized for Apple M4 Max and cross-platform compatibility

.PHONY: help setup install test train eval results clean reproduce apple-setup

# Default target
help:
	@echo "Multi-Agent Digital Twin - Available Commands"
	@echo "============================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup          - Set up the environment (cross-platform)"
	@echo "  apple-setup    - Set up for Apple M4 Max (macOS only)"
	@echo "  install        - Install the package in development mode"
	@echo ""
	@echo "Development Commands:"
	@echo "  test-simple    - Run simple Python test"
	@echo "  train-simple   - Run simple training test (no behavior cloning)"
	@echo "  test           - Run quick functionality test"
	@echo "  train          - Run training pipeline"
	@echo "  eval           - Generate experimental results"
	@echo "  results        - Generate publication-quality results"
	@echo ""
	@echo "Reproduction Commands:"
	@echo "  reproduce      - One-command reproduction of all results"
	@echo "  reproduce-quick - Quick reproduction for testing"
	@echo "  results-only   - Show existing results (no training needed)"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean          - Clean up generated files"
	@echo "  package        - Create distribution package"
	@echo "  verify         - Verify installation"
	@echo "  benchmark      - Run performance benchmark"
	@echo ""
	@echo "Paper Commands:"
	@echo "  paper          - Compile LaTeX paper"
	@echo "  figures        - Generate all figures"
	@echo ""

# Apple M4 Max specific setup
apple-setup:
	@echo "🍎 Setting up for Apple M4 Max..."
	./setup_apple_m4.sh

# Cross-platform setup
setup:
	@echo "🔧 Setting up environment..."
	@if [ "$$(uname)" = "Darwin" ] && [ "$$(uname -m)" = "arm64" ]; then \
		echo "Detected Apple Silicon Mac - using optimized setup"; \
		./setup_apple_m4.sh; \
	else \
		echo "Using standard setup"; \
		pip install -r requirements.txt; \
		pip install -e .; \
	fi

# Install package in development mode
install:
	@echo "📦 Installing package in development mode..."
	pip install -e .

# Run quick test
test:
	@echo "🧪 Running quick functionality test..."
	python test_quick_training.py

# Run training pipeline
train:
	@echo "🚀 Running training pipeline..."
	python -m train.training_pipeline --experiment full_training

# Run evaluation
eval:
	@echo "📊 Running evaluation..."
	python -m eval.generate_results

# Generate publication results
results:
	@echo "📈 Generating publication-quality results..."
	python -m eval.generate_results --output_dir ./results

# One-command reproduction
reproduce: setup test train eval results
	@echo "✅ Complete reproduction finished!"
	@echo "Results available in:"
	@echo "  - ./results/plots/ (figures)"
	@echo "  - ./results/tables/ (tables)"
	@echo "  - ./results/executive_summary.md (summary)"

# Quick reproduction for testing
reproduce-quick: setup
	@echo "🚀 Running quick reproduction..."
	@echo "Note: Using existing results. For new training, run 'make train'"
	python show_results.py
	@echo "✅ Quick reproduction completed - results already available!"

# Show existing results only (no training)
results-only:
	@echo "📊 Displaying existing experimental results..."
	python show_results.py

# Simple test using Python directly
test-simple:
	@echo "🧪 Running simple test..."
	python simple_test.py

# Simple training test without behavior cloning
train-simple:
	@echo "🚀 Running simple training test (no behavior cloning)..."
	python simple_training_test.py

# Run training pipeline (with potential BC issues)
train:
	@echo "🚀 Running training pipeline..."
	python -m train.training_pipeline --experiment full_training

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	rm -rf experiments/
	rm -rf results/
	rm -rf logs/
	rm -rf checkpoints/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*~" -delete

# Create distribution package
package: clean
	@echo "📦 Creating distribution package..."
	python setup.py sdist bdist_wheel
	@echo "Package created in dist/"

# Compile LaTeX paper
paper:
	@echo "📄 Compiling LaTeX paper..."
	cd paper && pdflatex main_complete.tex
	cd paper && bibtex main_complete
	cd paper && pdflatex main_complete.tex
	cd paper && pdflatex main_complete.tex
	@echo "Paper compiled: paper/main_complete.pdf"

# Generate all figures
figures:
	@echo "🎨 Generating all figures..."
	python -m eval.generate_results --output_dir ./paper/figures
	@echo "Figures generated in paper/figures/"

# Development helpers
lint:
	@echo "🔍 Running linting..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	black --check .

format:
	@echo "✨ Formatting code..."
	black .

# System information
info:
	@echo "System Information:"
	@echo "==================="
	@echo "OS: $$(uname -s)"
	@echo "Architecture: $$(uname -m)"
	@echo "Python: $$(python --version)"
	@echo "Conda: $$(conda --version 2>/dev/null || echo 'Not installed')"
	@echo "PyTorch: $$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "macOS Version: $$(sw_vers -productVersion)"; \
		if [ "$$(uname -m)" = "arm64" ]; then \
			echo "Apple Silicon: Yes"; \
			echo "MPS Available: $$(python -c 'import torch; print(torch.backends.mps.is_available())' 2>/dev/null || echo 'Unknown')"; \
		fi; \
	fi

# Verify installation
verify:
	@echo "🔍 Verifying installation..."
	@python -c "\
import sys; \
print(f'Python version: {sys.version}'); \
try: \
    import numpy as np; \
    print(f'✅ NumPy {np.__version__}'); \
except ImportError: \
    print('❌ NumPy not found'); \
try: \
    import torch; \
    print(f'✅ PyTorch {torch.__version__}'); \
    if torch.backends.mps.is_available(): \
        print('✅ MPS (Metal Performance Shaders) available'); \
    else: \
        print('ℹ️  MPS not available (CPU mode)'); \
except ImportError: \
    print('❌ PyTorch not found'); \
try: \
    import stable_baselines3; \
    print(f'✅ Stable-Baselines3 {stable_baselines3.__version__}'); \
except ImportError: \
    print('❌ Stable-Baselines3 not found'); \
try: \
    import gymnasium; \
    print(f'✅ Gymnasium {gymnasium.__version__}'); \
except ImportError: \
    print('❌ Gymnasium not found'); \
try: \
    import cvxpy; \
    print(f'✅ CVXPY {cvxpy.__version__}'); \
except ImportError: \
    print('❌ CVXPY not found'); \
try: \
    from envs.tep_env import TEPEnvironment; \
    print('✅ TEP Environment module'); \
except ImportError: \
    print('❌ TEP Environment module not found'); \
try: \
    from agents.multi_agent_system import MultiAgentRLSystem; \
    print('✅ Multi-Agent RL System module'); \
except ImportError: \
    print('❌ Multi-Agent RL System module not found'); \
print('✅ Verification completed')"

# Performance benchmark
benchmark:
	@echo "⚡ Running performance benchmark..."
	@python -c "\
import time; \
import numpy as np; \
import torch; \
print('Performance Benchmark'); \
print('===================='); \
start = time.time(); \
a = np.random.randn(1000, 1000); \
b = np.random.randn(1000, 1000); \
c = np.dot(a, b); \
numpy_time = time.time() - start; \
print(f'NumPy matrix multiplication (1000x1000): {numpy_time:.3f}s'); \
start = time.time(); \
a = torch.randn(1000, 1000); \
b = torch.randn(1000, 1000); \
c = torch.mm(a, b); \
cpu_time = time.time() - start; \
print(f'PyTorch CPU matrix multiplication (1000x1000): {cpu_time:.3f}s'); \
if torch.backends.mps.is_available(): \
    device = torch.device('mps'); \
    start = time.time(); \
    a = torch.randn(1000, 1000).to(device); \
    b = torch.randn(1000, 1000).to(device); \
    c = torch.mm(a, b); \
    torch.mps.synchronize(); \
    mps_time = time.time() - start; \
    print(f'PyTorch MPS matrix multiplication (1000x1000): {mps_time:.3f}s'); \
    print(f'MPS speedup: {cpu_time/mps_time:.2f}x'); \
else: \
    print('MPS not available for benchmarking'); \
print('Benchmark completed')"

