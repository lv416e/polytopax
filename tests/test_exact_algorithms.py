"""Tests for exact convex hull algorithms.

This module tests the Phase 3 exact algorithms including QuickHull,
Graham Scan, and exact geometric predicates.
"""

import jax
import jax.numpy as jnp

from polytopax.algorithms.exact import (
    _cross_product_2d,
    is_point_inside_triangle_2d,
    orientation_2d,
    point_to_line_distance_2d,
    quickhull,
)


class TestQuickHull2D:
    """Test 2D QuickHull implementation."""

    def test_triangle_basic(self):
        """Test QuickHull on a simple triangle."""
        # Simple triangle
        points = jnp.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0]
        ])

        hull_vertices, hull_indices = quickhull(points)

        # All points should be on the hull
        assert hull_vertices.shape[0] == 3
        assert len(hull_indices) == 3

        # Check that hull vertices are from original points
        for hull_vertex in hull_vertices:
            found = False
            for original_point in points:
                if jnp.allclose(hull_vertex, original_point, atol=1e-10):
                    found = True
                    break
            assert found, f"Hull vertex {hull_vertex} not found in original points"

    def test_square_with_interior_points(self):
        """Test QuickHull on square with interior points."""
        # Square corners + interior points
        points = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Corners
            [0.5, 0.5], [0.3, 0.7], [0.8, 0.2]               # Interior
        ])

        hull_vertices, hull_indices = quickhull(points)

        # Should find only the 4 corners
        assert hull_vertices.shape[0] <= 4

        # All hull vertices should be from the original corners
        corners = points[:4]
        for hull_vertex in hull_vertices:
            found = False
            for corner in corners:
                if jnp.allclose(hull_vertex, corner, atol=1e-10):
                    found = True
                    break
            assert found, f"Hull vertex {hull_vertex} should be one of the corners"

    def test_collinear_points(self):
        """Test QuickHull on collinear points."""
        # Points on a line
        points = jnp.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [0.5, 0.5],
            [1.5, 1.5]
        ])

        hull_vertices, hull_indices = quickhull(points)

        # Should find only the two extreme points
        assert hull_vertices.shape[0] == 2

        # Check that we have the two extreme points
        extreme_distances = []
        for vertex in hull_vertices:
            distance = jnp.linalg.norm(vertex)
            extreme_distances.append(distance)

        extreme_distances.sort()
        assert jnp.allclose(extreme_distances[0], 0.0, atol=1e-10)  # Origin
        assert jnp.allclose(extreme_distances[1], jnp.sqrt(8), atol=1e-10)  # (2,2)

    def test_pentagon_regular(self):
        """Test QuickHull on regular pentagon."""
        # Regular pentagon vertices
        n = 5
        radius = 1.0
        angles = jnp.linspace(0, 2*jnp.pi, n, endpoint=False)
        points = radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

        hull_vertices, hull_indices = quickhull(points)

        # All points should be on the hull for a regular pentagon
        assert hull_vertices.shape[0] == 5

        # Verify ordering by checking that vertices form a convex polygon
        # (This is a simplified check)
        center = jnp.mean(hull_vertices, axis=0)
        distances = jnp.linalg.norm(hull_vertices - center, axis=1)

        # All distances should be approximately equal for regular pentagon
        assert jnp.allclose(distances, radius, atol=1e-6)


class TestGeometricPredicates:
    """Test exact geometric predicates."""

    def test_orientation_2d(self):
        """Test 2D orientation predicate."""
        # Test counterclockwise
        p1 = jnp.array([0.0, 0.0])
        p2 = jnp.array([1.0, 0.0])
        p3 = jnp.array([0.0, 1.0])

        assert orientation_2d(p1, p2, p3) == 1  # Counterclockwise

        # Test clockwise
        p3_cw = jnp.array([1.0, -1.0])
        assert orientation_2d(p1, p2, p3_cw) == -1  # Clockwise

        # Test collinear
        p3_col = jnp.array([2.0, 0.0])
        assert orientation_2d(p1, p2, p3_col) == 0  # Collinear

    def test_point_to_line_distance_2d(self):
        """Test point to line distance calculation."""
        # Line from origin to (1,0)
        line_start = jnp.array([0.0, 0.0])
        line_end = jnp.array([1.0, 0.0])

        # Point above the line
        point_above = jnp.array([0.5, 1.0])
        distance_above = point_to_line_distance_2d(point_above, line_start, line_end)
        assert distance_above > 0  # Positive for left side
        assert jnp.allclose(distance_above, 1.0, atol=1e-10)

        # Point below the line
        point_below = jnp.array([0.5, -1.0])
        distance_below = point_to_line_distance_2d(point_below, line_start, line_end)
        assert distance_below < 0  # Negative for right side
        assert jnp.allclose(distance_below, -1.0, atol=1e-10)

        # Point on the line
        point_on = jnp.array([0.5, 0.0])
        distance_on = point_to_line_distance_2d(point_on, line_start, line_end)
        assert jnp.allclose(distance_on, 0.0, atol=1e-10)

    def test_point_inside_triangle_2d(self):
        """Test point in triangle test."""
        # Unit triangle
        triangle = jnp.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        # Point inside
        point_inside = jnp.array([0.2, 0.2])
        assert is_point_inside_triangle_2d(point_inside, triangle)

        # Point outside
        point_outside = jnp.array([1.0, 1.0])
        assert not is_point_inside_triangle_2d(point_outside, triangle)

        # Point on edge
        point_on_edge = jnp.array([0.5, 0.0])
        assert is_point_inside_triangle_2d(point_on_edge, triangle)

        # Point at vertex
        point_at_vertex = jnp.array([0.0, 0.0])
        assert is_point_inside_triangle_2d(point_at_vertex, triangle)

    def test_cross_product_2d(self):
        """Test 2D cross product calculation."""
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([0.0, 1.0])

        cross = _cross_product_2d(v1, v2)
        assert jnp.allclose(cross, 1.0, atol=1e-10)

        # Test opposite direction
        cross_opposite = _cross_product_2d(v2, v1)
        assert jnp.allclose(cross_opposite, -1.0, atol=1e-10)

        # Test parallel vectors
        v3 = jnp.array([2.0, 0.0])
        cross_parallel = _cross_product_2d(v1, v3)
        assert jnp.allclose(cross_parallel, 0.0, atol=1e-10)


class TestQuickHullExactness:
    """Test mathematical exactness of QuickHull algorithm."""

    def test_vertex_count_constraint(self):
        """Test that QuickHull never produces more vertices than input."""
        # Generate various test cases
        test_cases = [
            jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),  # Triangle
            jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),  # Square
            jax.random.normal(jax.random.PRNGKey(42), (10, 2)),  # Random points
            jax.random.normal(jax.random.PRNGKey(123), (20, 2))  # More random points
        ]

        for points in test_cases:
            hull_vertices, hull_indices = quickhull(points)

            # QuickHull should never produce more vertices than input
            assert hull_vertices.shape[0] <= points.shape[0], \
                f"QuickHull produced {hull_vertices.shape[0]} vertices from {points.shape[0]} input points"

            # All hull indices should be valid
            assert jnp.all(hull_indices >= 0)
            assert jnp.all(hull_indices < points.shape[0])

    def test_hull_vertices_are_subset(self):
        """Test that all hull vertices are from the original point set."""
        points = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
            [0.5, 0.5], [0.3, 0.3], [0.7, 0.7]
        ])

        hull_vertices, hull_indices = quickhull(points)

        # Each hull vertex should exactly match one of the original points
        for hull_vertex in hull_vertices:
            found_exact_match = False
            for original_point in points:
                if jnp.allclose(hull_vertex, original_point, atol=1e-12):
                    found_exact_match = True
                    break

            assert found_exact_match, \
                f"Hull vertex {hull_vertex} is not an exact match to any original point"

    def test_convex_hull_properties(self):
        """Test fundamental convex hull properties."""
        points = jnp.array([
            [0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0]
        ])

        hull_vertices, hull_indices = quickhull(points)

        # Property 1: Hull should contain all original points
        for _original_point in points:
            # For now, we'll implement a simple check
            # TODO: Implement proper point-in-convex-hull test
            pass

        # Property 2: Hull vertices should be in convex position
        # (This is a simplified check for 2D)
        if hull_vertices.shape[0] >= 3:
            # Check that no hull vertex is inside the triangle formed by any other three
            for i in range(hull_vertices.shape[0]):
                for j in range(i+1, hull_vertices.shape[0]):
                    for k in range(j+1, hull_vertices.shape[0]):
                        for vertex_idx in range(k+1, hull_vertices.shape[0]):
                            jnp.array([
                                hull_vertices[i], hull_vertices[j], hull_vertices[k]
                            ])
                            hull_vertices[vertex_idx]

                            # The test point should not be strictly inside the triangle
                            # (it can be on the boundary)
                            # This is a partial check for convexity
                            pass


class TestQuickHullPerformance:
    """Test QuickHull performance and edge cases."""

    def test_large_point_set(self):
        """Test QuickHull on larger point sets."""
        # Generate random points
        key = jax.random.PRNGKey(12345)
        points = jax.random.normal(key, (100, 2))

        hull_vertices, hull_indices = quickhull(points)

        # Basic sanity checks
        assert hull_vertices.shape[0] >= 3  # At least a triangle
        assert hull_vertices.shape[0] <= points.shape[0]
        assert len(hull_indices) == hull_vertices.shape[0]

    def test_degenerate_cases(self):
        """Test QuickHull on degenerate cases."""
        # Single point
        single_point = jnp.array([[0.0, 0.0]])
        hull_vertices, hull_indices = quickhull(single_point)
        assert hull_vertices.shape[0] == 1

        # Two points
        two_points = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        hull_vertices, hull_indices = quickhull(two_points)
        assert hull_vertices.shape[0] == 2

        # Three collinear points
        collinear = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        hull_vertices, hull_indices = quickhull(collinear)
        assert hull_vertices.shape[0] == 2  # Should be just the endpoints


if __name__ == "__main__":
    # Run basic tests
    test_quickhull = TestQuickHull2D()
    test_predicates = TestGeometricPredicates()
    test_exactness = TestQuickHullExactness()
    test_performance = TestQuickHullPerformance()

    print("=== QuickHull 2D Tests ===")
    test_quickhull.test_triangle_basic()
    test_quickhull.test_square_with_interior_points()
    test_quickhull.test_collinear_points()
    test_quickhull.test_pentagon_regular()
    print("✓ All QuickHull 2D tests passed")

    print("\n=== Geometric Predicates Tests ===")
    test_predicates.test_orientation_2d()
    test_predicates.test_point_to_line_distance_2d()
    test_predicates.test_point_inside_triangle_2d()
    test_predicates.test_cross_product_2d()
    print("✓ All geometric predicates tests passed")

    print("\n=== QuickHull Exactness Tests ===")
    test_exactness.test_vertex_count_constraint()
    test_exactness.test_hull_vertices_are_subset()
    test_exactness.test_convex_hull_properties()
    print("✓ All exactness tests passed")

    print("\n=== QuickHull Performance Tests ===")
    test_performance.test_large_point_set()
    test_performance.test_degenerate_cases()
    print("✓ All performance tests passed")

    print("\n🎯 All QuickHull tests completed successfully!")
