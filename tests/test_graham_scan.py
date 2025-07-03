"""Tests for Graham Scan 2D convex hull algorithm.

This module tests the Graham Scan implementation and compares it with QuickHull.
"""

import jax
import jax.numpy as jnp
import pytest

from polytopax.algorithms.graham_scan import (
    _ccw,
    _find_starting_point,
    _sort_points_by_angle,
    compare_graham_quickhull,
    graham_scan,
    graham_scan_monotone,
)


class TestGrahamScan:
    """Test Graham Scan implementation."""

    def test_triangle_basic(self):
        """Test Graham Scan on a simple triangle."""
        # Simple triangle
        points = jnp.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0]
        ])

        hull_vertices, hull_indices = graham_scan(points)

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
        """Test Graham Scan on square with interior points."""
        # Square corners + interior points
        points = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Corners
            [0.5, 0.5], [0.3, 0.7], [0.8, 0.2]               # Interior
        ])

        hull_vertices, hull_indices = graham_scan(points)

        # Should find only the 4 corners
        assert hull_vertices.shape[0] == 4

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
        """Test Graham Scan on collinear points."""
        # Points on a line
        points = jnp.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [0.5, 0.5],
            [1.5, 1.5]
        ])

        hull_vertices, hull_indices = graham_scan(points)

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
        """Test Graham Scan on regular pentagon."""
        # Regular pentagon vertices
        n = 5
        radius = 1.0
        angles = jnp.linspace(0, 2*jnp.pi, n, endpoint=False)
        points = radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

        hull_vertices, hull_indices = graham_scan(points)

        # All points should be on the hull for a regular pentagon
        assert hull_vertices.shape[0] == 5

        # Verify that all points are at the expected radius
        center = jnp.array([0.0, 0.0])
        distances = jnp.linalg.norm(hull_vertices - center, axis=1)

        # All distances should be approximately equal for regular pentagon
        assert jnp.allclose(distances, radius, atol=1e-6)


class TestGrahamScanMonotone:
    """Test monotone chain variant of Graham Scan."""

    def test_monotone_triangle(self):
        """Test monotone Graham Scan on triangle."""
        points = jnp.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0]
        ])

        hull_vertices, hull_indices = graham_scan_monotone(points)

        assert hull_vertices.shape[0] == 3

        # All vertices should be from original points
        for hull_vertex in hull_vertices:
            found = False
            for original_point in points:
                if jnp.allclose(hull_vertex, original_point, atol=1e-10):
                    found = True
                    break
            assert found

    def test_monotone_square(self):
        """Test monotone Graham Scan on square."""
        points = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]
        ])

        hull_vertices, hull_indices = graham_scan_monotone(points)

        assert hull_vertices.shape[0] == 4

        # Should find the 4 corners
        corners = points[:4]
        for hull_vertex in hull_vertices:
            found = False
            for corner in corners:
                if jnp.allclose(hull_vertex, corner, atol=1e-10):
                    found = True
                    break
            assert found


class TestGrahamScanHelpers:
    """Test helper functions for Graham Scan."""

    def test_find_starting_point(self):
        """Test finding the starting point (bottommost, then leftmost)."""
        # Points with clear bottommost point
        points = jnp.array([
            [1.0, 1.0],
            [0.0, 0.0],  # This should be the starting point
            [2.0, 2.0],
            [0.5, 0.5]
        ])

        start_index = _find_starting_point(points)
        assert start_index == 1

        # Points with tied y-coordinates (test leftmost breaking)
        points_tied = jnp.array([
            [1.0, 0.0],
            [0.0, 0.0],  # This should be chosen (leftmost)
            [2.0, 0.0],
            [0.5, 1.0]
        ])

        start_index_tied = _find_starting_point(points_tied)
        assert start_index_tied == 1

    def test_sort_points_by_angle(self):
        """Test sorting points by polar angle."""
        points = jnp.array([
            [0.0, 0.0],  # Starting point
            [1.0, 0.0],  # 0 degrees
            [1.0, 1.0],  # 45 degrees
            [0.0, 1.0],  # 90 degrees
            [-1.0, 0.0]  # 180 degrees
        ])

        start_index = 0
        sorted_indices = _sort_points_by_angle(points, start_index, 1e-12)

        # First point should be the starting point
        assert sorted_indices[0] == start_index

        # Check that angles are in increasing order
        start_point = points[start_index]
        prev_angle = -jnp.pi  # Start with smallest possible angle

        for i in range(1, len(sorted_indices)):
            point_index = sorted_indices[i]
            vector = points[point_index] - start_point
            angle = jnp.arctan2(vector[1], vector[0])
            assert angle >= prev_angle - 1e-10  # Allow for small numerical errors
            prev_angle = angle

    def test_ccw_predicate(self):
        """Test counterclockwise predicate."""
        # Test counterclockwise
        p1 = jnp.array([0.0, 0.0])
        p2 = jnp.array([1.0, 0.0])
        p3 = jnp.array([0.0, 1.0])

        assert _ccw(p1, p2, p3) == 1  # Counterclockwise

        # Test clockwise
        p3_cw = jnp.array([1.0, -1.0])
        assert _ccw(p1, p2, p3_cw) == -1  # Clockwise

        # Test collinear
        p3_col = jnp.array([2.0, 0.0])
        assert _ccw(p1, p2, p3_col) == 0  # Collinear


class TestGrahamScanExactness:
    """Test mathematical exactness of Graham Scan algorithm."""

    def test_vertex_count_constraint(self):
        """Test that Graham Scan never produces more vertices than input."""
        # Generate various test cases
        test_cases = [
            jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),  # Triangle
            jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),  # Square
            jax.random.normal(jax.random.PRNGKey(42), (10, 2)),  # Random points
            jax.random.normal(jax.random.PRNGKey(123), (20, 2))  # More random points
        ]

        for points in test_cases:
            hull_vertices, hull_indices = graham_scan(points)

            # Graham Scan should never produce more vertices than input
            assert hull_vertices.shape[0] <= points.shape[0], \
                f"Graham Scan produced {hull_vertices.shape[0]} vertices from {points.shape[0]} input points"

            # All hull indices should be valid
            assert jnp.all(hull_indices >= 0)
            assert jnp.all(hull_indices < points.shape[0])

    def test_hull_vertices_are_subset(self):
        """Test that all hull vertices are from the original point set."""
        points = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
            [0.5, 0.5], [0.3, 0.3], [0.7, 0.7]
        ])

        hull_vertices, hull_indices = graham_scan(points)

        # Each hull vertex should exactly match one of the original points
        for hull_vertex in hull_vertices:
            found_exact_match = False
            for original_point in points:
                if jnp.allclose(hull_vertex, original_point, atol=1e-12):
                    found_exact_match = True
                    break

            assert found_exact_match, \
                f"Hull vertex {hull_vertex} is not an exact match to any original point"


class TestGrahamScanComparison:
    """Test Graham Scan comparison with QuickHull."""

    def test_comparison_triangle(self):
        """Test Graham Scan vs QuickHull comparison on triangle."""
        points = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, 1.0]
        ])

        comparison = compare_graham_quickhull(points)

        # Both algorithms should find the same vertices
        assert comparison["vertices_match"]
        assert comparison["graham_vertex_count"] == comparison["quickhull_vertex_count"]
        assert comparison["graham_vertex_count"] == 3

    def test_comparison_square_with_interior(self):
        """Test Graham Scan vs QuickHull comparison on square with interior points."""
        points = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
            [0.5, 0.5], [0.3, 0.7], [0.8, 0.2]
        ])

        comparison = compare_graham_quickhull(points)

        # Both algorithms should find the same 4 corner vertices
        assert comparison["vertices_match"]
        assert comparison["graham_vertex_count"] == comparison["quickhull_vertex_count"]
        assert comparison["graham_vertex_count"] == 4

    def test_comparison_random_points(self):
        """Test Graham Scan vs QuickHull comparison on random points."""
        key = jax.random.PRNGKey(12345)
        points = jax.random.normal(key, (15, 2))

        comparison = compare_graham_quickhull(points)

        # Both algorithms should find the same vertices
        assert comparison["vertices_match"], \
            f"Algorithms disagree: symmetric difference = {comparison['symmetric_difference']}"
        assert comparison["graham_vertex_count"] == comparison["quickhull_vertex_count"]


class TestGrahamScanPerformance:
    """Test Graham Scan performance and edge cases."""

    def test_degenerate_cases(self):
        """Test Graham Scan on degenerate cases."""
        # Single point
        single_point = jnp.array([[0.0, 0.0]])
        hull_vertices, hull_indices = graham_scan(single_point)
        assert hull_vertices.shape[0] == 1

        # Two points
        two_points = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        hull_vertices, hull_indices = graham_scan(two_points)
        assert hull_vertices.shape[0] == 2

        # Three collinear points
        collinear = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        hull_vertices, hull_indices = graham_scan(collinear)
        assert hull_vertices.shape[0] == 2  # Should be just the endpoints

    def test_3d_points_error(self):
        """Test that Graham Scan raises error for 3D points."""
        points_3d = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])

        with pytest.raises(ValueError, match="Graham Scan only works with 2D points"):
            graham_scan(points_3d)

    def test_large_point_set(self):
        """Test Graham Scan on larger point sets."""
        # Generate random points
        key = jax.random.PRNGKey(12345)
        points = jax.random.normal(key, (100, 2))

        hull_vertices, hull_indices = graham_scan(points)

        # Basic sanity checks
        assert hull_vertices.shape[0] >= 3  # At least a triangle
        assert hull_vertices.shape[0] <= points.shape[0]
        assert len(hull_indices) == hull_vertices.shape[0]


if __name__ == "__main__":
    # Run basic tests
    test_graham = TestGrahamScan()
    test_monotone = TestGrahamScanMonotone()
    test_helpers = TestGrahamScanHelpers()
    test_exactness = TestGrahamScanExactness()
    test_comparison = TestGrahamScanComparison()
    test_performance = TestGrahamScanPerformance()

    print("=== Graham Scan Tests ===")
    test_graham.test_triangle_basic()
    test_graham.test_square_with_interior_points()
    test_graham.test_collinear_points()
    test_graham.test_pentagon_regular()
    print("✓ All Graham Scan tests passed")

    print("\n=== Graham Scan Monotone Tests ===")
    test_monotone.test_monotone_triangle()
    test_monotone.test_monotone_square()
    print("✓ All Graham Scan monotone tests passed")

    print("\n=== Graham Scan Helpers Tests ===")
    test_helpers.test_find_starting_point()
    test_helpers.test_sort_points_by_angle()
    test_helpers.test_ccw_predicate()
    print("✓ All Graham Scan helper tests passed")

    print("\n=== Graham Scan Exactness Tests ===")
    test_exactness.test_vertex_count_constraint()
    test_exactness.test_hull_vertices_are_subset()
    print("✓ All Graham Scan exactness tests passed")

    print("\n=== Graham Scan Comparison Tests ===")
    test_comparison.test_comparison_triangle()
    test_comparison.test_comparison_square_with_interior()
    test_comparison.test_comparison_random_points()
    print("✓ All Graham Scan comparison tests passed")

    print("\n=== Graham Scan Performance Tests ===")
    test_performance.test_degenerate_cases()
    test_performance.test_3d_points_error()
    test_performance.test_large_point_set()
    print("✓ All Graham Scan performance tests passed")

    print("\n🎯 All Graham Scan tests completed successfully!")
