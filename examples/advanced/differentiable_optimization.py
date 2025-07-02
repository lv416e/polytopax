#!/usr/bin/env python3
"""
Differentiable Optimization Example.

This example demonstrates how to use PolytopAX for optimization problems
where convex hull properties are part of the objective function.
"""

import os
import sys

# Add the parent directory to the path to import polytopax
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp

import polytopax as ptx


def example_shape_optimization():
    """Optimize point positions to achieve target convex hull properties."""
    print("=== Shape Optimization Example ===")

    def objective(points, target_area=1.0, target_centroid=None):
        """Objective function: minimize difference from target properties."""
        if target_centroid is None:
            target_centroid = jnp.array([0.5, 0.5])

        hull = ptx.ConvexHull.from_points(points, n_directions=20)

        # Current properties
        area = hull.volume()
        centroid = hull.centroid()

        # Losses
        area_loss = (area - target_area)**2
        centroid_loss = jnp.sum((centroid - target_centroid)**2)

        # Regularization: prefer points not too far from each other
        mean_point = jnp.mean(points, axis=0)
        spread_loss = 0.1 * jnp.sum((points - mean_point)**2)

        return area_loss + centroid_loss + spread_loss

    # Initialize points randomly
    key = jax.random.PRNGKey(123)
    n_points = 6
    points = jax.random.uniform(key, (n_points, 2), minval=0.2, maxval=0.8)

    print(f"Initial points shape: {points.shape}")

    # Target properties
    target_area = 0.8
    target_centroid = jnp.array([0.4, 0.6])

    print(f"Target area: {target_area}")
    print(f"Target centroid: {target_centroid}")

    # Initial properties
    initial_hull = ptx.ConvexHull.from_points(points, n_directions=20)
    initial_area = initial_hull.volume()
    initial_centroid = initial_hull.centroid()
    initial_loss = objective(points, target_area, target_centroid)

    print("\nInitial properties:")
    print(f"  Area: {initial_area:.4f}")
    print(f"  Centroid: [{initial_centroid[0]:.4f}, {initial_centroid[1]:.4f}]")
    print(f"  Loss: {initial_loss:.4f}")

    # Optimization with simple gradient descent
    grad_fn = jax.grad(lambda p: objective(p, target_area, target_centroid))

    learning_rate = 0.01
    n_steps = 200

    print(f"\nOptimizing for {n_steps} steps...")

    current_points = points
    for step in range(n_steps):
        gradients = grad_fn(current_points)
        current_points = current_points - learning_rate * gradients

        # Clip to reasonable bounds
        current_points = jnp.clip(current_points, 0.0, 1.0)

        if step % 50 == 0:
            current_loss = objective(current_points, target_area, target_centroid)
            current_hull = ptx.ConvexHull.from_points(current_points, n_directions=20)
            current_area = current_hull.volume()
            current_centroid = current_hull.centroid()

            print(f"  Step {step}: loss={current_loss:.4f}, "
                  f"area={current_area:.4f}, "
                  f"centroid=[{current_centroid[0]:.4f}, {current_centroid[1]:.4f}]")

    # Final results
    final_hull = ptx.ConvexHull.from_points(current_points, n_directions=20)
    final_area = final_hull.volume()
    final_centroid = final_hull.centroid()
    final_loss = objective(current_points, target_area, target_centroid)

    print("\nFinal properties:")
    print(f"  Area: {final_area:.4f} (target: {target_area:.4f})")
    print(f"  Centroid: [{final_centroid[0]:.4f}, {final_centroid[1]:.4f}] "
          f"(target: [{target_centroid[0]:.4f}, {target_centroid[1]:.4f}])")
    print(f"  Loss: {final_loss:.4f} (improvement: {initial_loss - final_loss:.4f})")


def example_constrained_optimization():
    """Optimize with constraints using penalty methods."""
    print("\n=== Constrained Optimization Example ===")

    def constrained_objective(points):
        """Maximize area while keeping points inside a circle."""
        hull = ptx.ConvexHull.from_points(points, n_directions=15)
        area = hull.volume()

        # Constraint: all points should be within unit circle centered at origin
        distances_from_origin = jnp.linalg.norm(points, axis=1)
        max_radius = 1.0

        # Penalty for violating constraints
        constraint_violations = jnp.maximum(0, distances_from_origin - max_radius)
        penalty = 100.0 * jnp.sum(constraint_violations**2)

        # We want to maximize area, so minimize negative area plus penalty
        return -area + penalty

    # Initialize points within the constraint
    jax.random.PRNGKey(456)
    n_points = 5
    angles = jnp.linspace(0, 2*jnp.pi, n_points, endpoint=False)
    radius = 0.8  # Start inside the constraint
    points = radius * jnp.column_stack([jnp.cos(angles), jnp.sin(angles)])

    print(f"Initial points (arranged in circle, radius={radius}):")
    print(points)

    initial_hull = ptx.ConvexHull.from_points(points, n_directions=15)
    initial_area = initial_hull.volume()
    initial_objective = constrained_objective(points)

    print(f"\nInitial area: {initial_area:.4f}")
    print(f"Initial objective: {initial_objective:.4f}")

    # Check initial constraint satisfaction
    initial_distances = jnp.linalg.norm(points, axis=1)
    print(f"Initial max distance from origin: {jnp.max(initial_distances):.4f}")

    # Optimization
    grad_fn = jax.grad(constrained_objective)

    learning_rate = 0.02
    n_steps = 300

    print(f"\nOptimizing for {n_steps} steps...")

    current_points = points
    for step in range(n_steps):
        gradients = grad_fn(current_points)
        current_points = current_points - learning_rate * gradients

        if step % 75 == 0:
            current_objective = constrained_objective(current_points)
            current_hull = ptx.ConvexHull.from_points(current_points, n_directions=15)
            current_area = current_hull.volume()
            max_distance = jnp.max(jnp.linalg.norm(current_points, axis=1))

            print(f"  Step {step}: objective={current_objective:.4f}, "
                  f"area={current_area:.4f}, max_dist={max_distance:.4f}")

    # Final results
    final_hull = ptx.ConvexHull.from_points(current_points, n_directions=15)
    final_area = final_hull.volume()
    final_objective = constrained_objective(current_points)
    final_max_distance = jnp.max(jnp.linalg.norm(current_points, axis=1))

    print("\nFinal results:")
    print(f"  Area: {final_area:.4f} (improvement: {final_area - initial_area:.4f})")
    print(f"  Objective: {final_objective:.4f}")
    print(f"  Max distance from origin: {final_max_distance:.4f} (constraint: ≤ 1.0)")
    print(f"  Constraint satisfied: {final_max_distance <= 1.0}")


def example_multi_objective_optimization():
    """Optimize multiple competing objectives."""
    print("\n=== Multi-Objective Optimization Example ===")

    def multi_objective(points, weight_area=1.0, weight_compactness=1.0):
        """Balance between area and compactness."""
        hull = ptx.ConvexHull.from_points(points, n_directions=15)

        # Objective 1: Maximize area
        area = hull.volume()

        # Objective 2: Maximize compactness (minimize perimeter for given area)
        perimeter = hull.surface_area()
        # Compactness: 4π * area / perimeter^2 (circle = 1, line = 0)
        compactness = 4 * jnp.pi * area / (perimeter**2 + 1e-8)

        # Combined objective (we minimize, so negate what we want to maximize)
        return -(weight_area * area + weight_compactness * compactness)

    # Initialize with random points
    key = jax.random.PRNGKey(789)
    n_points = 8
    points = jax.random.normal(key, (n_points, 2)) * 0.5

    print(f"Optimizing {n_points} points for area vs compactness tradeoff")

    # Test different weight combinations
    weight_combinations = [
        (1.0, 0.0),   # Only area
        (0.5, 0.5),   # Balanced
        (0.0, 1.0),   # Only compactness
    ]

    for i, (w_area, w_comp) in enumerate(weight_combinations):
        print(f"\n--- Configuration {i+1}: area_weight={w_area}, compactness_weight={w_comp} ---")

        # Reset to initial points
        current_points = points.copy()

        # Define objective with current weights
        def objective(p, area_weight=w_area, comp_weight=w_comp):
            return multi_objective(p, area_weight, comp_weight)
        grad_fn = jax.grad(objective)

        # Initial metrics
        initial_hull = ptx.ConvexHull.from_points(current_points, n_directions=15)
        initial_area = initial_hull.volume()
        initial_perimeter = initial_hull.surface_area()
        initial_compactness = 4 * jnp.pi * initial_area / (initial_perimeter**2 + 1e-8)

        print(f"  Initial: area={initial_area:.4f}, "
              f"perimeter={initial_perimeter:.4f}, compactness={initial_compactness:.4f}")

        # Optimize
        learning_rate = 0.01
        n_steps = 150

        for _step in range(n_steps):
            gradients = grad_fn(current_points)
            current_points = current_points - learning_rate * gradients

        # Final metrics
        final_hull = ptx.ConvexHull.from_points(current_points, n_directions=15)
        final_area = final_hull.volume()
        final_perimeter = final_hull.surface_area()
        final_compactness = 4 * jnp.pi * final_area / (final_perimeter**2 + 1e-8)

        print(f"  Final:   area={final_area:.4f}, "
              f"perimeter={final_perimeter:.4f}, compactness={final_compactness:.4f}")
        print(f"  Changes: area={final_area - initial_area:+.4f}, "
              f"compactness={final_compactness - initial_compactness:+.4f}")


def example_batch_optimization():
    """Optimize multiple point sets simultaneously."""
    print("\n=== Batch Optimization Example ===")

    def batch_objective(batch_points):
        """Objective for a batch of point sets."""
        # Target: each hull should have area ≈ 1.0
        target_area = 1.0

        def single_objective(points):
            hull = ptx.ConvexHull.from_points(points, n_directions=12)
            area = hull.volume()
            return (area - target_area)**2

        # Apply to each point set in the batch
        return jax.vmap(single_objective)(batch_points)

    # Create batch of point sets
    key = jax.random.PRNGKey(999)
    batch_size = 4
    n_points = 6

    batch_points = jax.random.uniform(
        key, (batch_size, n_points, 2), minval=0.1, maxval=0.9
    )

    print(f"Optimizing {batch_size} point sets simultaneously")
    print(f"Each set has {n_points} points")

    # Initial areas
    initial_objectives = batch_objective(batch_points)
    print("\nInitial objectives (squared error from target area 1.0):")
    for i, obj in enumerate(initial_objectives):
        print(f"  Set {i}: {obj:.4f}")

    # Batch optimization
    grad_fn = jax.grad(lambda bp: jnp.sum(batch_objective(bp)))

    learning_rate = 0.01
    n_steps = 200

    print(f"\nOptimizing for {n_steps} steps...")

    current_batch = batch_points
    for step in range(n_steps):
        gradients = grad_fn(current_batch)
        current_batch = current_batch - learning_rate * gradients

        # Clip to bounds
        current_batch = jnp.clip(current_batch, 0.0, 1.0)

        if step % 50 == 0:
            current_objectives = batch_objective(current_batch)
            total_objective = jnp.sum(current_objectives)
            print(f"  Step {step}: total_objective={total_objective:.4f}")

    # Final results
    final_objectives = batch_objective(current_batch)

    print("\nFinal objectives:")
    for i, (initial, final) in enumerate(zip(initial_objectives, final_objectives, strict=False)):
        improvement = initial - final
        print(f"  Set {i}: {final:.4f} (improvement: {improvement:.4f})")

    # Compute actual areas
    print("\nActual final areas:")
    for i in range(batch_size):
        hull = ptx.ConvexHull.from_points(current_batch[i], n_directions=12)
        area = hull.volume()
        print(f"  Set {i}: {area:.4f}")


if __name__ == "__main__":
    print("PolytopAX Differentiable Optimization Examples")
    print("=============================================")

    # Run all examples
    example_shape_optimization()
    example_constrained_optimization()
    example_multi_objective_optimization()
    example_batch_optimization()

    print("\n🎯 All optimization examples completed!")
    print("\nKey insights:")
    print("- Convex hull properties are differentiable and can be optimized")
    print("- Constraints can be handled with penalty methods")
    print("- Multiple objectives can be balanced with weights")
    print("- Batch optimization enables parallel parameter tuning")
    print("- JAX transformations make complex optimizations efficient")
