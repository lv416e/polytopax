# Installation

This guide covers different ways to install PolytopAX and its dependencies.

## Quick Install

For most users, the simplest installation method is using pip:

```bash
pip install polytopax
```

## Requirements

PolytopAX requires:

- **Python 3.8 or later**
- **JAX 0.4.0 or later**
- **NumPy**

Optional dependencies for examples and tutorials:
- **Matplotlib** (for visualization)
- **Jupyter** (for notebooks)
- **SciPy** (for comparisons and benchmarks)

## Installation Methods

### Option 1: PyPI (Recommended)

Install the latest stable release:

```bash
pip install polytopax
```

Install with optional dependencies:

```bash
# For visualization and examples
pip install polytopax[examples]

# For development and testing
pip install polytopax[dev]

# Install everything
pip install polytopax[all]
```

### Option 2: From Source

For the latest development version or to contribute:

```bash
git clone https://github.com/your-org/polytopax.git
cd polytopax
pip install -e .
```

For development installation:

```bash
git clone https://github.com/your-org/polytopax.git
cd polytopax
pip install -e .[dev]
```

### Option 3: Conda (Coming Soon)

Conda packages will be available in the future:

```bash
# Not yet available
conda install -c conda-forge polytopax
```

## JAX Installation

PolytopAX depends on JAX. The basic JAX installation works for CPU-only usage:

```bash
pip install jax
```

For GPU support, install the appropriate JAX version:

### NVIDIA GPU (CUDA)

```bash
# For CUDA 12
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CUDA 11
pip install -U "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Google TPU

```bash
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Apple Silicon (M1/M2) Mac

JAX has experimental support for Apple Silicon:

```bash
pip install -U jax jaxlib
```

## Verify Installation

After installation, verify that PolytopAX is working correctly:

```python
import polytopax as ptx
import jax.numpy as jnp

# Check version
print(f"PolytopAX version: {ptx.__version__}")

# Basic functionality test
points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
hull = ptx.ConvexHull.from_points(points)
print(f"Hull area: {hull.volume():.3f}")

# JAX device info
import jax
print(f"JAX devices: {jax.devices()}")
```

Expected output:
```
PolytopAX version: 0.1.0
Hull area: 0.500
JAX devices: [CpuDevice(id=0)]
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'polytopax'

Make sure PolytopAX is installed in the correct environment:

```bash
pip list | grep polytopax
```

If not found, reinstall:

```bash
pip install polytopax
```

#### JAX-related errors

If you encounter JAX-related issues:

1. **Update JAX**: `pip install -U jax jaxlib`
2. **Check compatibility**: Ensure JAX version ≥ 0.4.0
3. **Platform-specific**: Install correct JAX variant for your platform

#### Version conflicts

Create a fresh environment:

```bash
# Using conda
conda create -n polytopax python=3.10
conda activate polytopax
pip install polytopax

# Using venv
python -m venv polytopax_env
source polytopax_env/bin/activate  # Linux/Mac
# or: polytopax_env\Scripts\activate  # Windows
pip install polytopax
```

### Platform-Specific Notes

#### Linux

Standard installation should work. For GPU support, ensure CUDA drivers are installed.

#### macOS

- **Intel Macs**: Standard installation
- **Apple Silicon (M1/M2)**: JAX support is experimental but functional

#### Windows

Standard installation should work. For best performance, consider using WSL2.

### Performance Verification

Test performance with a simple benchmark:

```python
import jax
import jax.numpy as jnp
import polytopax as ptx
import time

# Create test data
key = jax.random.PRNGKey(0)
points = jax.random.normal(key, (100, 3))

# Benchmark
def compute_hull():
    return ptx.ConvexHull.from_points(points, n_directions=20)

# Time regular execution
start = time.time()
hull = compute_hull()
regular_time = time.time() - start

# Time JIT execution
jit_compute = jax.jit(compute_hull)
_ = jit_compute()  # Warm-up

start = time.time()
hull_jit = jit_compute()
jit_time = time.time() - start

print(f"Regular: {regular_time:.4f}s")
print(f"JIT:     {jit_time:.4f}s")
print(f"Speedup: {regular_time/jit_time:.1f}x")
```

## Next Steps

After successful installation:

1. **[Getting Started](getting_started.md)** - Learn the basics
2. **[Examples](../examples/)** - See practical applications
3. **[Tutorials](tutorials/index.md)** - Follow step-by-step guides
4. **[User Guide](user_guide/index.md)** - Comprehensive documentation

## Getting Help

If you encounter installation issues:

1. **Check the troubleshooting section above**
2. **Search existing issues** on GitHub
3. **Create a new issue** with:
   - Your operating system
   - Python version (`python --version`)
   - JAX version (`pip show jax`)
   - Full error message
   - Installation method used

We're here to help! 🚀