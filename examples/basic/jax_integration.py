#!/usr/bin/env python3
"""
JAX Integration Example.

This example demonstrates how to use PolytopAX with JAX transformations
including JIT compilation, automatic differentiation, and vectorization.
"""

import os
import sys

# Add the parent directory to the path to import polytopax
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp

import polytopax as ptx


def example_jit_compilation():
    """Demonstrate JIT compilation for performance."""
    print("=== JIT Compilation Example ===")

    # Define a function that computes convex hull volume
    def compute_hull_volume(points):
        hull = ptx.ConvexHull.from_points(points, n_directions=20)
        return hull.volume()

    # Create test points
    points = jnp.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]
    ])

    # Regular function call
    print("Regular function call:")
    volume_regular = compute_hull_volume(points)
    print(f"Volume: {volume_regular:.4f}")

    # JIT compiled version
    print("\nJIT compiled function:")
    jit_compute_volume = jax.jit(compute_hull_volume)

    # First call compiles the function
    volume_jit = jit_compute_volume(points)
    print(f"Volume (JIT): {volume_jit:.4f}")

    # Subsequent calls are faster
    volume_jit2 = jit_compute_volume(points)
    print(f"Volume (JIT, 2nd call): {volume_jit2:.4f}")

    print("✅ JIT compilation preserves results while improving performance")


def example_automatic_differentiation():
    """Demonstrate automatic differentiation."""
    print("\n=== Automatic Differentiation Example ===")

    # Define a loss function based on convex hull volume
    def volume_loss(points):
        hull = ptx.ConvexHull.from_points(points, n_directions=15)
        # We want to maximize volume, so minimize negative volume
        return -hull.volume()

    # Create initial points (a triangle)
    points = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.5],  # This point can be optimized
    ])

    print(f"Initial points:\n{points}")
    initial_volume = -volume_loss(points)
    print(f"Initial volume: {initial_volume:.4f}")

    # Compute gradients
    grad_fn = jax.grad(volume_loss)
    gradients = grad_fn(points)

    print(f"\nGradients:\n{gradients}")
    print("Gradients show how to move each point to increase volume")

    # Perform a simple gradient step
    learning_rate = 0.1
    updated_points = points - learning_rate * gradients
    updated_volume = -volume_loss(updated_points)

    print("\nAfter gradient step:")
    print(f"Updated points:\n{updated_points}")
    print(f"Updated volume: {updated_volume:.4f}")
    print(f"Volume change: {updated_volume - initial_volume:.4f}")


def example_vectorization():
    """Demonstrate vectorization with vmap."""
    print("\n=== Vectorization Example ===")

    # Define function that works on a single point set
    def single_hull_volume(points):
        hull = ptx.ConvexHull.from_points(points, n_directions=10)
        return hull.volume()

    # Create multiple point sets (batch dimension first)
    batch_size = 3
    base_points = jnp.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]
    ])

    # Create batch by scaling base points
    scales = jnp.array([1.0, 1.5, 2.0])
    batch_points = scales[:, None, None] * base_points[None, :, :]

    print(f"Batch shape: {batch_points.shape}")
    print(f"Processing {batch_size} point sets simultaneously")

    # Vectorize the function to handle batches
    batch_volume_fn = jax.vmap(single_hull_volume)
    batch_volumes = batch_volume_fn(batch_points)

    print("\nVolumes for each scale:")
    for _i, (scale, volume) in enumerate(zip(scales, batch_volumes, strict=False)):
        print(f"  Scale {scale}: Volume = {volume:.4f}")

    print("\n✅ Vectorization processes multiple inputs efficiently")


def example_complex_transformation():
    """Demonstrate combining JIT, grad, and vmap."""
    print("\n=== Complex Transformation Example ===")

    def hull_area_objective(points):
        """Objective function: maximize hull area minus distance from origin."""
        hull = ptx.ConvexHull.from_points(points, n_directions=12)
        area = hull.volume()

        # Penalty for being far from origin
        centroid = hull.centroid()
        distance_penalty = 0.1 * jnp.sum(centroid**2)

        return -(area - distance_penalty)  # Minimize negative of objective

    # Create a batch of initial point configurations
    key = jax.random.PRNGKey(42)
    batch_size = 4
    n_points = 5

    # Generate random initial configurations
    batch_points = jax.random.uniform(
        key, (batch_size, n_points, 2), minval=-1.0, maxval=1.0
    )

    print(f"Optimizing {batch_size} different point configurations")
    print(f"Each configuration has {n_points} points in 2D")

    # Create combined transformation: vmap over batch, then grad, then jit
    objective_grad = jax.grad(hull_area_objective)
    batch_grad = jax.vmap(objective_grad)
    jit_batch_grad = jax.jit(batch_grad)

    # Compute gradients for all configurations
    print("\nComputing gradients for batch...")
    batch_gradients = jit_batch_grad(batch_points)

    print(f"Gradient shape: {batch_gradients.shape}")

    # Perform optimization step
    learning_rate = 0.05
    updated_batch = batch_points - learning_rate * batch_gradients

    # Evaluate objectives before and after
    batch_objective = jax.vmap(hull_area_objective)
    initial_objectives = batch_objective(batch_points)
    updated_objectives = batch_objective(updated_batch)

    print("\nOptimization results:")
    for i in range(batch_size):
        improvement = updated_objectives[i] - initial_objectives[i]
        print(f"  Config {i}: {initial_objectives[i]:.4f} → "
              f"{updated_objectives[i]:.4f} (Δ={improvement:.4f})")

    print("\n✅ Complex transformations enable efficient batch optimization")


def example_performance_comparison():
    """Compare performance with and without JAX transformations."""
    print("\n=== Performance Comparison Example ===")

    import time

    def compute_volume(points):
        hull = ptx.ConvexHull.from_points(points, n_directions=15)
        return hull.volume()

    # Create test data
    points = jnp.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        [0.3, 0.3], [0.7, 0.7], [0.2, 0.8]
    ])

    n_iterations = 100

    # Time regular function calls
    print(f"Timing {n_iterations} regular function calls...")
    start_time = time.time()
    for _ in range(n_iterations):
        compute_volume(points)
    regular_time = time.time() - start_time

    # Time JIT compiled version
    print(f"Timing {n_iterations} JIT compiled calls...")
    jit_compute_volume = jax.jit(compute_volume)

    # Warm-up call (compilation happens here)
    _ = jit_compute_volume(points)

    start_time = time.time()
    for _ in range(n_iterations):
        jit_compute_volume(points)
    jit_time = time.time() - start_time

    print("\nPerformance Results:")
    print(f"  Regular: {regular_time:.4f} seconds")
    print(f"  JIT:     {jit_time:.4f} seconds")
    print(f"  Speedup: {regular_time / jit_time:.2f}x")

    if jit_time < regular_time:
        print("✅ JIT compilation provides significant speedup!")
    else:
        print("ℹ️ JIT overhead may dominate for simple cases")


if __name__ == "__main__":
    print("PolytopAX JAX Integration Examples")
    print("=================================")

    # Run all examples
    example_jit_compilation()
    example_automatic_differentiation()
    example_vectorization()
    example_complex_transformation()
    example_performance_comparison()

    print("\n🎉 All JAX integration examples completed!")
    print("\nKey takeaways:")
    print("- Use jax.jit for performance on repeated computations")
    print("- Use jax.grad for optimization and sensitivity analysis")
    print("- Use jax.vmap for efficient batch processing")
    print("- Combine transformations for powerful workflows")
    print("- PolytopAX is fully compatible with all JAX transformations")
