#!/usr/bin/env python3
"""
Basic Convex Hull Example.

This example demonstrates the fundamental usage of PolytopAX for computing
convex hulls in 2D and 3D.
"""

import os
import sys

# Add the parent directory to the path to import polytopax
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax.numpy as jnp

import polytopax as ptx


def example_2d_convex_hull():
    """Compute a simple 2D convex hull."""
    print("=== 2D Convex Hull Example ===")

    # Create a set of 2D points (including some interior points)
    points = jnp.array([
        [0.0, 0.0],   # corner
        [1.0, 0.0],   # corner
        [1.0, 1.0],   # corner
        [0.0, 1.0],   # corner
        [0.5, 0.5],   # interior point
        [0.3, 0.7],   # interior point
        [0.8, 0.2],   # interior point
    ])

    print(f"Input points shape: {points.shape}")
    print(f"Input points:\n{points}")

    # Compute convex hull using the ConvexHull class
    hull = ptx.ConvexHull.from_points(points, n_directions=20)

    print(f"\nHull vertices shape: {hull.vertices.shape}")
    print(f"Number of hull vertices: {hull.n_vertices}")
    print(f"Hull dimension: {hull.dimension}")

    # Compute geometric properties
    area = hull.volume()  # In 2D, volume() returns area
    perimeter = hull.surface_area()  # In 2D, surface_area() returns perimeter
    centroid = hull.centroid()

    print("\nGeometric Properties:")
    print(f"  Area: {area:.4f}")
    print(f"  Perimeter: {perimeter:.4f}")
    print(f"  Centroid: [{centroid[0]:.4f}, {centroid[1]:.4f}]")

    # Test point inclusion
    test_points = jnp.array([
        [0.5, 0.5],   # should be inside
        [0.0, 0.0],   # on boundary
        [1.5, 1.5],   # outside
    ])

    print("\nPoint Inclusion Tests:")
    for _i, point in enumerate(test_points):
        is_inside = hull.contains(point)
        distance = hull.distance_to(point)
        print(f"  Point {point}: inside={is_inside}, distance={distance:.4f}")


def example_3d_convex_hull():
    """Compute a simple 3D convex hull."""
    print("\n=== 3D Convex Hull Example ===")

    # Create a set of 3D points (vertices of a cube plus some interior points)
    points = jnp.array([
        # Cube vertices
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        # Interior points
        [0.5, 0.5, 0.5],
        [0.3, 0.3, 0.3],
        [0.7, 0.7, 0.7],
    ])

    print(f"Input points shape: {points.shape}")

    # Compute convex hull
    hull = ptx.ConvexHull.from_points(points, n_directions=30)

    print(f"Hull vertices shape: {hull.vertices.shape}")
    print(f"Number of hull vertices: {hull.n_vertices}")
    print(f"Hull dimension: {hull.dimension}")

    # Compute geometric properties
    volume = hull.volume()
    surface_area = hull.surface_area()
    centroid = hull.centroid()
    diameter = hull.diameter()

    print("\nGeometric Properties:")
    print(f"  Volume: {volume:.4f}")
    print(f"  Surface Area: {surface_area:.4f}")
    print(f"  Centroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")
    print(f"  Diameter: {diameter:.4f}")

    # Bounding box
    bbox_min, bbox_max = hull.bounding_box()
    print(f"  Bounding box: [{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}] to "
          f"[{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}]")


def example_functional_api():
    """Demonstrate the lower-level functional API."""
    print("\n=== Functional API Example ===")

    # Import jax for random number generation
    import jax

    # Create some random points
    key = jax.random.PRNGKey(42)
    points = jax.random.uniform(key, (10, 2), minval=0.0, maxval=1.0)

    print(f"Random points shape: {points.shape}")

    # Use the functional API directly
    hull_vertices = ptx.convex_hull(points, algorithm="approximate", n_directions=15)

    print(f"Hull vertices shape: {hull_vertices.shape}")

    # Use individual predicates
    volume = ptx.convex_hull_volume(hull_vertices)
    surface_area = ptx.convex_hull_surface_area(hull_vertices)

    print(f"Volume: {volume:.4f}")
    print(f"Surface area: {surface_area:.4f}")

    # Test point inclusion
    test_point = jnp.array([0.5, 0.5])
    is_inside = ptx.point_in_convex_hull(test_point, hull_vertices)
    distance = ptx.distance_to_convex_hull(test_point, hull_vertices)

    print(f"Test point {test_point}: inside={is_inside}, distance={distance:.4f}")


def example_different_algorithms():
    """Demonstrate different algorithm options."""
    print("\n=== Algorithm Options Example ===")

    points = jnp.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]
    ])

    # Different numbers of directions
    for n_dirs in [5, 10, 20, 50]:
        hull = ptx.ConvexHull.from_points(points, n_directions=n_dirs)
        print(f"n_directions={n_dirs}: {hull.n_vertices} vertices, "
              f"area={hull.volume():.4f}")

    # Different sampling methods
    print("\nDifferent sampling methods:")

    # Uniform sampling (default)
    hull_uniform = ptx.ConvexHull.from_points(points, method="uniform")
    print(f"Uniform: {hull_uniform.n_vertices} vertices, area={hull_uniform.volume():.4f}")

    # For 3D points, we can use icosphere sampling
    points_3d = jnp.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    ])
    hull_icosphere = ptx.ConvexHull.from_points(points_3d, method="icosphere")
    print(f"Icosphere (3D): {hull_icosphere.n_vertices} vertices, "
          f"volume={hull_icosphere.volume():.4f}")


if __name__ == "__main__":
    print("PolytopAX Basic Convex Hull Examples")
    print("===================================")

    # Import jax for random number generation

    # Run all examples
    example_2d_convex_hull()
    example_3d_convex_hull()
    example_functional_api()
    example_different_algorithms()

    print("\n✅ All examples completed successfully!")
    print("\nNext steps:")
    print("- Try modifying the point sets")
    print("- Experiment with different parameters")
    print("- Check out the advanced examples")
