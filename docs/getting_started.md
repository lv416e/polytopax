# Getting Started with PolytopAX

Welcome to PolytopAX! This guide will help you get up and running with differentiable convex hull computation using JAX.

## What is PolytopAX?

PolytopAX is a computational geometry library designed for machine learning and scientific computing applications that require:

- **Differentiable operations**: All computations maintain gradients for optimization
- **High performance**: GPU/TPU acceleration through JAX and XLA
- **Batch processing**: Native support for processing multiple point sets
- **Easy integration**: Compatible with existing JAX/ML workflows

## Core Concepts

### Convex Hull

A convex hull is the smallest convex set that contains all given points. PolytopAX computes approximate convex hulls using differentiable algorithms.

### Direction Vector Sampling

PolytopAX uses direction vector sampling to compute convex hulls:
1. Generate direction vectors on the unit sphere
2. Find extreme points in each direction using soft selection
3. Combine results to form the approximate hull

### Differentiability

Traditional convex hull algorithms are not differentiable due to discrete operations. PolytopAX uses:
- **Soft selection** with temperature-controlled softmax
- **Continuous approximations** instead of discrete choices
- **Smooth geometric predicates** for robust computation

## Installation

### From PyPI (recommended)

```bash
pip install polytopax
```

### From Source

```bash
git clone https://github.com/your-org/polytopax.git
cd polytopax
pip install -e .
```

### Dependencies

- Python 3.8+
- JAX 0.4.0+
- NumPy
- SciPy (optional)

## Verify Installation

```python
import polytopax as ptx
print(ptx.__version__)
print(ptx.get_info())
```

## Your First Convex Hull

Let's compute a simple 2D convex hull:

```python
import jax.numpy as jnp
import polytopax as ptx

# Create a square of points
points = jnp.array([
    [0.0, 0.0],  # bottom-left
    [1.0, 0.0],  # bottom-right
    [1.0, 1.0],  # top-right
    [0.0, 1.0],  # top-left
    [0.5, 0.5],  # center (inside)
])

# Compute convex hull
hull = ptx.ConvexHull.from_points(points, n_directions=20)

print(f"Original points: {points.shape[0]}")
print(f"Hull vertices: {hull.n_vertices}")
print(f"Hull area: {hull.volume():.3f}")
print(f"Hull perimeter: {hull.surface_area():.3f}")
```

## JAX Integration

PolytopAX is designed to work seamlessly with JAX transformations:

### JIT Compilation

```python
import jax

# JIT compile for performance
@jax.jit
def compute_hull_volume(points):
    hull = ptx.ConvexHull.from_points(points)
    return hull.volume()

# Use with any point set
volume = compute_hull_volume(points)
```

### Automatic Differentiation

```python
# Compute gradients with respect to input points
def hull_volume_loss(points):
    hull = ptx.ConvexHull.from_points(points)
    return hull.volume()

grad_fn = jax.grad(hull_volume_loss)
gradients = grad_fn(points)
```

### Vectorization

```python
# Process multiple point sets in parallel
batch_points = jnp.stack([points, points * 2.0, points * 0.5])

# Vectorize over batch dimension
batch_volumes = jax.vmap(compute_hull_volume)(batch_points)
```

## Next Steps

- **[Basic Usage](user_guide/basic_usage.md)**: Learn the fundamental operations
- **[Tutorials](tutorials/index.md)**: Follow step-by-step examples
- **[API Reference](api/index.md)**: Explore all available functions and classes
- **[Examples](examples/index.md)**: See practical applications

## Performance Tips

1. **Use JIT compilation** for repeated computations
2. **Batch operations** when processing multiple point sets
3. **Adjust n_directions** based on accuracy vs speed requirements
4. **Use GPU/TPU** for large-scale computations

## Getting Help

- **Documentation**: This documentation site
- **Examples**: Check the examples directory
- **Issues**: Report bugs or request features on GitHub
- **Discussions**: Join community discussions on GitHub

Welcome to the PolytopAX community! 🎉