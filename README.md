# 🔷 PolytopAX

**GPU-accelerated differentiable convex hull computation powered by JAX**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/polytopax.svg?cache=no)](https://pypi.org/project/polytopax/)
[![Tests](https://github.com/lv416e/polytopax/actions/workflows/tests.yml/badge.svg)](https://github.com/lv416e/polytopax/actions/workflows/tests.yml)
[![Lint](https://github.com/lv416e/polytopax/actions/workflows/lint.yml/badge.svg)](https://github.com/lv416e/polytopax/actions/workflows/lint.yml)
[![Docs](https://github.com/lv416e/polytopax/actions/workflows/docs.yml/badge.svg)](https://github.com/lv416e/polytopax/actions/workflows/docs.yml)
[![Release](https://github.com/lv416e/polytopax/actions/workflows/release.yml/badge.svg)](https://github.com/lv416e/polytopax/actions/workflows/release.yml)

PolytopAX brings modern computational geometry to the JAX ecosystem, enabling **differentiable convex hull computation** with GPU acceleration and automatic differentiation for machine learning applications.

## ✨ Key Features

- 🚀 **GPU-Accelerated**: Leverage JAX/XLA for high-performance computation
- 🔄 **Differentiable**: Full compatibility with JAX transformations (`jit`, `grad`, `vmap`)
- 🎯 **ML-Ready**: Seamless integration into machine learning pipelines
- 📦 **Easy to Use**: Both functional and object-oriented APIs
- 🔬 **Research-Grade**: Built for computational geometry research and applications

## 🚀 Quick Start

### Installation

```bash
pip install polytopax
```

### Basic Usage

```python
import jax.numpy as jnp
import polytopax as ptx

# Generate random points
points = jax.random.normal(jax.random.PRNGKey(0), (100, 3))

# Compute convex hull
hull_vertices = ptx.convex_hull(points)
print(f"Hull has {len(hull_vertices)} vertices")

# Object-oriented API
hull = ptx.ConvexHull.from_points(points)
print(f"Volume: {hull.volume():.4f}")
print(f"Surface area: {hull.surface_area():.4f}")

# Check point containment
test_point = jnp.array([0.0, 0.0, 0.0])
is_inside = hull.contains(test_point)
```

### Differentiable Optimization

```python
import jax

# Differentiable volume computation
def volume_loss(points):
    hull = ptx.ConvexHull.from_points(points)
    return -hull.volume()  # Maximize volume

# Compute gradients
grad_fn = jax.grad(volume_loss)
gradients = grad_fn(points)

# Batch processing with vmap
batch_points = jnp.stack([points, points + 0.1])
batch_volumes = jax.vmap(lambda p: ptx.ConvexHull.from_points(p).volume())(batch_points)
```

## 🔧 API Overview

### Core Functions
- `convex_hull(points)` - Compute convex hull vertices
- `point_in_convex_hull(point, vertices)` - Point containment test
- `convex_hull_volume(vertices)` - Volume computation
- `convex_hull_surface_area(vertices)` - Surface area computation

### ConvexHull Class
- `ConvexHull.from_points(points)` - Create from point cloud
- `.volume()` - Compute volume
- `.surface_area()` - Compute surface area
- `.contains(point)` - Test point containment
- `.centroid()` - Compute centroid

## 🎯 Use Cases

- **Robotics**: Path planning and obstacle avoidance
- **Machine Learning**: Constraint optimization and geometric deep learning
- **Computer Graphics**: Collision detection and rendering
- **Computational Physics**: Molecular dynamics and simulations
- **Finance**: Risk management and portfolio optimization

## 🏗️ Why PolytopAX?

| Feature | PolytopAX | SciPy | Qhull |
|---------|-----------|-------|-------|
| GPU Acceleration | ✅ | ❌ | ❌ |
| Differentiable | ✅ | ❌ | ❌ |
| JAX Integration | ✅ | ❌ | ❌ |
| Batch Processing | ✅ | ❌ | ❌ |
| ML Pipeline Ready | ✅ | ❌ | ❌ |

## 📖 Documentation

- [Getting Started](docs/getting_started.md)
- [API Reference](docs/api/index.rst)
- [Examples](examples/)
- [Design Documents](design/)

## 🔬 Examples

Explore comprehensive examples in the [`examples/`](examples/) directory:

- [Basic Usage](examples/basic/basic_convex_hull.py)
- [JAX Integration](examples/basic/jax_integration.py)
- [Differentiable Optimization](examples/advanced/differentiable_optimization.py)
- [Interactive Notebooks](examples/notebooks/)

## 🛣️ Roadmap

- **v0.1.0** ✅ Core differentiable convex hull algorithms
- **v0.5.0** 🔄 Exact algorithms (Quickhull, incremental)
- **v0.8.0** 📋 Advanced operations (intersections, Minkowski sums)
- **v1.0.0** 🎯 Riemannian manifold integration (→ GeomAX)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CODE_OF_CONDUCT.md) and [Development Setup](docs/development.md).

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## 🔗 Links

- **Documentation**: [polytopax.readthedocs.io](https://polytopax.readthedocs.io) (coming soon)
- **PyPI**: [pypi.org/project/polytopax](https://pypi.org/project/polytopax) (coming soon)
- **Issues**: [GitHub Issues](https://github.com/lv416e/polytopax/issues)

---

**Built with ❤️ for the JAX and computational geometry communities**
