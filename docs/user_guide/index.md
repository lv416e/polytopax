# User Guide

This user guide provides comprehensive documentation for using PolytopAX effectively in your projects.

## Overview

PolytopAX is designed to make differentiable convex hull computation accessible and efficient for machine learning and scientific computing applications. This guide covers:

- **Basic concepts** and terminology
- **Common usage patterns** and best practices  
- **Advanced features** and customization options
- **Performance optimization** techniques
- **Integration** with existing workflows

## Table of Contents

```{toctree}
:maxdepth: 2

basic_usage
advanced_features
performance_tips
```

## Quick Navigation

### For Beginners
- [Basic Usage](basic_usage.md) - Start here for fundamental operations
- [Getting Started](../getting_started.md) - Installation and first steps
- [Examples](../examples/index.md) - Practical code examples

### For Advanced Users
- [Advanced Features](advanced_features.md) - Complex operations and customization
- [Performance Tips](performance_tips.md) - Optimization and scaling
- [API Reference](../api/index.rst) - Complete function documentation

### For Specific Use Cases
- **Machine Learning**: Differentiable optimization with convex hull constraints
- **Scientific Computing**: High-performance geometric computations
- **Research**: Novel algorithms and experimental features

## Key Concepts

### Differentiable Convex Hulls

Traditional convex hull algorithms use discrete operations that break differentiability. PolytopAX uses:

- **Direction vector sampling** on the unit sphere
- **Soft selection** with temperature-controlled operations
- **Continuous approximations** instead of discrete choices

This enables gradient-based optimization while maintaining computational efficiency.

### JAX Integration

PolytopAX is built on JAX, providing:

- **JIT compilation** for performance
- **Automatic differentiation** for gradients
- **Vectorization** for batch operations
- **Device acceleration** on GPU/TPU

All PolytopAX operations are compatible with JAX transformations.

### Approximation vs Exactness

PolytopAX computes *approximate* convex hulls that:

- **Maintain differentiability** throughout computation
- **Provide controllable accuracy** via algorithm parameters
- **Scale efficiently** to large point sets
- **Support batch operations** natively

The approximation quality can be tuned based on application requirements.

## Common Workflows

### 1. Basic Convex Hull Computation

```python
import polytopax as ptx
import jax.numpy as jnp

# Compute hull
points = jnp.array([[0, 0], [1, 0], [0, 1]])
hull = ptx.ConvexHull.from_points(points)

# Access properties
area = hull.volume()
perimeter = hull.surface_area()
```

### 2. Gradient-Based Optimization

```python
import jax

def objective(points):
    hull = ptx.ConvexHull.from_points(points)
    return hull.volume()  # Maximize area

# Compute gradients
grad_fn = jax.grad(objective)
gradients = grad_fn(points)
```

### 3. Batch Processing

```python
# Process multiple point sets
batch_points = jnp.stack([points1, points2, points3])
batch_hulls = jax.vmap(ptx.ConvexHull.from_points)(batch_points)
```

### 4. Performance Optimization

```python
# JIT compile for speed
@jax.jit
def fast_hull_computation(points):
    return ptx.ConvexHull.from_points(points, n_directions=20)
```

## Best Practices

1. **Choose appropriate parameters** for your accuracy/speed tradeoff
2. **Use JIT compilation** for repeated computations
3. **Batch operations** when processing multiple point sets
4. **Profile your code** to identify bottlenecks
5. **Validate results** for critical applications

## Getting Help

- **Documentation**: Browse this documentation site
- **Examples**: Check the examples directory for practical code
- **API Reference**: See detailed function documentation
- **Community**: Join discussions on GitHub
- **Issues**: Report bugs or request features

Let's get started with [Basic Usage](basic_usage.md)!