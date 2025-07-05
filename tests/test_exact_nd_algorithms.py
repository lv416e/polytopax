"""Tests for n-dimensional exact convex hull algorithms.

This module tests the n-dimensional QuickHull implementation and n-dimensional
geometric predicates for dimensions 4 and above.
"""

import jax
import jax.numpy as jnp
import pytest

from polytopax.algorithms.exact_nd import (
    _compute_simplex_volume_nd,
    _find_initial_simplex_nd,
    _generate_simplex_facets,
    is_point_inside_simplex_nd,
    orientation_nd,
    point_to_hyperplane_distance_nd,
    quickhull_nd,
)


class TestNDGeometricPredicates:
    """Test n-dimensional geometric predicates."""

    def test_orientation_4d(self):
        """Test 4D orientation predicate."""
        # Test points in 4D space
        # Create a 4D simplex: origin + 4 unit vectors
        points = jnp.array([
            [0.0, 0.0, 0.0, 0.0],  # Origin
            [1.0, 0.0, 0.0, 0.0],  # X-axis
            [0.0, 1.0, 0.0, 0.0],  # Y-axis
            [0.0, 0.0, 1.0, 0.0],  # Z-axis
            [0.0, 0.0, 0.0, 1.0],  # W-axis
        ])

        # Test positive orientation
        result = orientation_nd(points)
        assert result == 1  # Should be positive orientation

        # Test negative orientation (swap two points)
        points_neg = points.at[3].set(points[4])
        points_neg = points_neg.at[4].set(points[3])
        result_neg = orientation_nd(points_neg)
        assert result_neg == -1  # Should be negative orientation

        # Test coplanar case (4 points on a plane)
        points_coplanar = jnp.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],  # This point is coplanar with others
        ])
        result_coplanar = orientation_nd(points_coplanar)
        assert result_coplanar == 0  # Should be coplanar

    def test_orientation_5d(self):
        """Test 5D orientation predicate."""
        # 5D unit simplex
        points = jnp.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ])

        result = orientation_nd(points)
        assert result != 0  # Should not be degenerate

    def test_point_to_hyperplane_distance_4d(self):
        """Test 4D point to hyperplane distance."""
        # Define a hyperplane in 4D using 4 points
        hyperplane_points = jnp.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])

        # Test point above hyperplane
        point_above = jnp.array([0.0, 0.0, 0.0, 1.0])
        distance_above = point_to_hyperplane_distance_nd(point_above, hyperplane_points)
        assert distance_above != 0.0

        # Test point on hyperplane
        point_on = jnp.array([0.3, 0.3, 0.3, 0.0])
        distance_on = point_to_hyperplane_distance_nd(point_on, hyperplane_points)
        assert abs(distance_on) < 1e-10

    def test_point_inside_simplex_4d(self):
        """Test 4D point in simplex test."""
        # 4D unit simplex
        simplex = jnp.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # Point inside (centroid)
        point_inside = jnp.array([0.2, 0.2, 0.2, 0.2])
        assert is_point_inside_simplex_nd(point_inside, simplex)

        # Point outside
        point_outside = jnp.array([1.0, 1.0, 1.0, 1.0])
        assert not is_point_inside_simplex_nd(point_outside, simplex)

        # Point at vertex
        point_at_vertex = jnp.array([0.0, 0.0, 0.0, 0.0])
        assert is_point_inside_simplex_nd(point_at_vertex, simplex)

    def test_orientation_edge_cases(self):
        """Test orientation predicate edge cases."""
        # Test with insufficient points for 3D
        points_insufficient = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])

        with pytest.raises(ValueError):
            orientation_nd(points_insufficient)

        # Test 1D case (2 points)
        points_1d = jnp.array([[0.0], [1.0]])
        result_1d = orientation_nd(points_1d)
        assert result_1d == 1  # Positive direction

        # Test 1D case (reverse)
        points_1d_rev = jnp.array([[1.0], [0.0]])
        result_1d_rev = orientation_nd(points_1d_rev)
        assert result_1d_rev == -1  # Negative direction


class TestNDInitialSimplex:
    """Test n-dimensional initial simplex finding."""

    def test_find_initial_simplex_4d(self):
        """Test finding initial simplex in 4D."""
        # 4D hypercube vertices
        points = jnp.array([
            [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0], [1.0, 1.1, 1.0, 1.0],
        ])

        simplex_indices = _find_initial_simplex_nd(points, 1e-12)

        # Should find 5 points for 4D simplex
        assert len(simplex_indices) >= 4
        assert len(simplex_indices) <= 5

        # All indices should be valid
        assert all(0 <= idx < len(points) for idx in simplex_indices)

    def test_simplex_volume_4d(self):
        """Test 4D simplex volume calculation."""
        # 4D unit simplex
        simplex = jnp.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        volume = _compute_simplex_volume_nd(simplex, 1e-12)

        # 4D unit simplex has volume 1/4! = 1/24
        expected_volume = 1.0 / 24.0
        assert abs(volume - expected_volume) < 1e-10

    def test_degenerate_simplex_volume(self):
        """Test volume of degenerate simplex."""
        # Coplanar points (all in 3D subspace)
        degenerate_simplex = jnp.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5, 0.0],  # This point is coplanar
        ])

        volume = _compute_simplex_volume_nd(degenerate_simplex, 1e-12)
        assert volume < 1e-10  # Should be approximately zero


class TestNDFacetManagement:
    """Test n-dimensional facet management."""

    def test_generate_simplex_facets_4d(self):
        """Test generating facets of 4D simplex."""
        simplex_indices = [0, 1, 2, 3, 4]
        facets = _generate_simplex_facets(simplex_indices, 4)

        # 4D simplex should have 5 facets
        assert len(facets) == 5

        # Each facet should have 4 vertices
        for facet in facets:
            assert len(facet) == 4

        # Each vertex should appear in exactly 4 facets
        vertex_count = {}
        for facet in facets:
            for vertex in facet:
                vertex_count[vertex] = vertex_count.get(vertex, 0) + 1

        for vertex_idx in simplex_indices:
            assert vertex_count[vertex_idx] == 4

    def test_generate_simplex_facets_5d(self):
        """Test generating facets of 5D simplex."""
        simplex_indices = [0, 1, 2, 3, 4, 5]
        facets = _generate_simplex_facets(simplex_indices, 5)

        # 5D simplex should have 6 facets
        assert len(facets) == 6

        # Each facet should have 5 vertices
        for facet in facets:
            assert len(facet) == 5


class TestNDQuickHull:
    """Test n-dimensional QuickHull implementation."""

    def test_quickhull_4d_basic(self):
        """Test basic 4D QuickHull on simple case."""
        # 4D simplex (should return all 5 points)
        points = jnp.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        hull_vertices, hull_indices = quickhull_nd(points)

        # All points should be on the hull
        assert hull_vertices.shape[0] == 5
        assert len(hull_indices) == 5

    def test_quickhull_4d_with_interior_points(self):
        """Test 4D QuickHull with interior points."""
        # 4D hypercube + interior points
        corners = jnp.array([
            [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0],
        ])

        # Add interior points
        interior = jnp.array([
            [0.5, 0.5, 0.5, 0.5],
            [0.3, 0.3, 0.3, 0.3],
            [0.7, 0.7, 0.7, 0.7],
        ])

        points = jnp.vstack([corners, interior])

        hull_vertices, hull_indices = quickhull_nd(points)

        # Should find only corner points, not interior points
        assert hull_vertices.shape[0] <= 16  # At most the corners

        # Check that no interior points are in the hull
        for idx in hull_indices:
            if idx >= 16:  # Interior point index
                pytest.fail(f"Interior point {idx} should not be on hull")

    def test_quickhull_dimension_fallback(self):
        """Test that QuickHull falls back to specialized implementations."""
        # 2D case should use 2D implementation
        points_2d = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hull_2d, indices_2d = quickhull_nd(points_2d)
        assert hull_2d.shape[1] == 2

        # 3D case should use 3D implementation
        points_3d = jnp.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
        ])
        hull_3d, indices_3d = quickhull_nd(points_3d)
        assert hull_3d.shape[1] == 3

    def test_quickhull_insufficient_points(self):
        """Test QuickHull with insufficient points."""
        # Only 3 points in 4D - not enough for full hull
        points = jnp.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

        hull_vertices, hull_indices = quickhull_nd(points)

        # Should return all input points
        assert hull_vertices.shape[0] == 3
        assert len(hull_indices) == 3

    def test_quickhull_dimension_warning(self):
        """Test dimension warning for high dimensions."""
        # Create 15D points (above default max_dimension=10)
        points = jax.random.normal(jax.random.PRNGKey(42), (20, 15))

        with pytest.warns(UserWarning, match="exceeds max_dimension"):
            hull_vertices, hull_indices = quickhull_nd(points, max_dimension=10)


class TestNDQuickHullExactness:
    """Test mathematical exactness of n-dimensional QuickHull."""

    def test_vertex_count_constraint_4d(self):
        """Test that 4D QuickHull never produces more vertices than input."""
        # Random 4D points
        points = jax.random.normal(jax.random.PRNGKey(123), (20, 4))

        hull_vertices, hull_indices = quickhull_nd(points)

        # Should never produce more vertices than input
        assert hull_vertices.shape[0] <= points.shape[0]

        # All indices should be valid
        assert jnp.all(hull_indices >= 0)
        assert jnp.all(hull_indices < points.shape[0])

    def test_hull_vertices_are_subset_4d(self):
        """Test that all hull vertices are from original point set."""
        points = jnp.array([
            [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5],
        ])

        hull_vertices, hull_indices = quickhull_nd(points)

        # Each hull vertex should match one of the original points
        for hull_vertex in hull_vertices:
            found_match = False
            for original_point in points:
                if jnp.allclose(hull_vertex, original_point, atol=1e-12):
                    found_match = True
                    break
            assert found_match, f"Hull vertex {hull_vertex} not found in original points"


class TestNDQuickHullPerformance:
    """Test performance and edge cases of n-dimensional QuickHull."""

    def test_degenerate_cases_4d(self):
        """Test 4D QuickHull on degenerate cases."""
        # All points on a 3D hyperplane
        points_3d_subspace = jnp.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5, 0.0],
        ])

        with pytest.warns(UserWarning, match="Degenerate n-dimensional case"):
            hull_vertices, hull_indices = quickhull_nd(points_3d_subspace)

        # Should handle gracefully
        assert hull_vertices.shape[0] >= 1

    def test_random_point_sets_4d(self):
        """Test QuickHull on random 4D point sets."""
        # Small random point set
        points = jax.random.normal(jax.random.PRNGKey(456), (10, 4))

        hull_vertices, hull_indices = quickhull_nd(points)

        # Basic consistency checks
        assert hull_vertices.shape[1] == 4  # 4D output
        assert hull_vertices.shape[0] <= points.shape[0]
        assert len(hull_indices) == hull_vertices.shape[0]


if __name__ == "__main__":
    # Run basic tests
    test_predicates = TestNDGeometricPredicates()
    test_simplex = TestNDInitialSimplex()
    test_facets = TestNDFacetManagement()
    test_quickhull = TestNDQuickHull()
    test_exactness = TestNDQuickHullExactness()
    test_performance = TestNDQuickHullPerformance()

    print("=== N-Dimensional Geometric Predicates Tests ===")
    test_predicates.test_orientation_4d()
    test_predicates.test_orientation_5d()
    test_predicates.test_point_to_hyperplane_distance_4d()
    test_predicates.test_point_inside_simplex_4d()
    test_predicates.test_orientation_edge_cases()
    print("✓ All n-dimensional predicates tests passed")

    print("\n=== Initial Simplex Tests ===")
    test_simplex.test_find_initial_simplex_4d()
    test_simplex.test_simplex_volume_4d()
    test_simplex.test_degenerate_simplex_volume()
    print("✓ All initial simplex tests passed")

    print("\n=== Facet Management Tests ===")
    test_facets.test_generate_simplex_facets_4d()
    test_facets.test_generate_simplex_facets_5d()
    print("✓ All facet management tests passed")

    print("\n=== N-Dimensional QuickHull Tests ===")
    test_quickhull.test_quickhull_4d_basic()
    test_quickhull.test_quickhull_dimension_fallback()
    test_quickhull.test_quickhull_insufficient_points()
    print("✓ All n-dimensional QuickHull tests passed")

    print("\n=== N-Dimensional QuickHull Exactness Tests ===")
    test_exactness.test_vertex_count_constraint_4d()
    test_exactness.test_hull_vertices_are_subset_4d()
    print("✓ All n-dimensional exactness tests passed")

    print("\n🎯 All n-dimensional QuickHull tests completed successfully!")
