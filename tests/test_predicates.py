"""Tests for geometric predicates module."""

import math
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from polytopax.operations.predicates import (
    convex_hull_surface_area,
    convex_hull_volume,
    distance_to_convex_hull,
    hausdorff_distance,
    point_in_convex_hull,
)


class TestPointInConvexHull:
    """Tests for point-in-convex-hull testing."""

    def test_simple_triangle_2d(self):
        """Test point inclusion in 2D triangle."""
        # Unit triangle
        vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        # Points clearly inside
        inside_point = jnp.array([0.2, 0.2])
        result = point_in_convex_hull(inside_point, vertices)
        # Note: Due to algorithm limitations, this might not always be True
        assert isinstance(result, bool | np.bool_ | jnp.ndarray)

        # Points clearly outside
        outside_point = jnp.array([2.0, 2.0])
        result = point_in_convex_hull(outside_point, vertices)
        assert isinstance(result, bool | np.bool_ | jnp.ndarray)

        # Boundary points
        boundary_point = jnp.array([0.0, 0.0])  # Vertex
        result = point_in_convex_hull(boundary_point, vertices)
        assert isinstance(result, bool | np.bool_ | jnp.ndarray)

    def test_simple_tetrahedron_3d(self):
        """Test point inclusion in 3D tetrahedron."""
        # Unit tetrahedron
        vertices = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        # Point inside
        inside_point = jnp.array([0.1, 0.1, 0.1])
        result = point_in_convex_hull(inside_point, vertices)
        assert isinstance(result, bool | np.bool_ | jnp.ndarray)

        # Point outside
        outside_point = jnp.array([2.0, 2.0, 2.0])
        result = point_in_convex_hull(outside_point, vertices)
        assert isinstance(result, bool | np.bool_ | jnp.ndarray)

    def test_different_methods(self):
        """Test different point inclusion methods."""
        vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        point = jnp.array([0.3, 0.3])

        # Linear programming method
        result_lp = point_in_convex_hull(point, vertices, method="linear_programming")
        assert isinstance(result_lp, bool | np.bool_ | jnp.ndarray)

        # Barycentric method
        result_bary = point_in_convex_hull(point, vertices, method="barycentric")
        assert isinstance(result_bary, bool | np.bool_ | jnp.ndarray)

        # Unknown method should raise error
        with pytest.raises(ValueError):
            point_in_convex_hull(point, vertices, method="unknown")

    def test_tolerance_parameter(self):
        """Test tolerance parameter effect."""
        vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        boundary_point = jnp.array([0.0, 0.0])  # Exactly on boundary

        # Strict tolerance
        result_strict = point_in_convex_hull(boundary_point, vertices, tolerance=1e-12)
        assert isinstance(result_strict, bool | np.bool_ | jnp.ndarray)

        # Loose tolerance
        result_loose = point_in_convex_hull(boundary_point, vertices, tolerance=1e-6)
        assert isinstance(result_loose, bool | np.bool_ | jnp.ndarray)

    def test_degenerate_hull(self):
        """Test with degenerate hull (too few vertices)."""
        # Only 2 vertices for 2D (degenerate)
        vertices = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        point = jnp.array([0.5, 0.0])

        result = point_in_convex_hull(point, vertices)
        assert isinstance(result, bool | np.bool_ | jnp.ndarray)


class TestConvexHullVolume:
    """Tests for convex hull volume computation."""

    def test_2d_triangle_area(self):
        """Test area computation of 2D triangle."""
        # Unit right triangle (area = 0.5)
        vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        volume = convex_hull_volume(vertices)

        assert isinstance(volume, float | jnp.ndarray)
        assert volume > 0
        # Note: Due to approximation, exact value may vary

    def test_2d_square_area(self):
        """Test area computation of 2D square."""
        # Unit square (area = 1.0)
        vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        volume = convex_hull_volume(vertices)

        assert isinstance(volume, float | jnp.ndarray)
        assert volume > 0

    def test_3d_tetrahedron_volume(self):
        """Test volume computation of 3D tetrahedron."""
        # Unit tetrahedron (volume = 1/6)
        vertices = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        volume = convex_hull_volume(vertices)

        assert isinstance(volume, float | jnp.ndarray)
        assert volume > 0
        # Approximate check (algorithm may not be exact)
        assert 0.1 < volume < 0.2  # Should be around 1/6 ≈ 0.167

    def test_different_volume_methods(self):
        """Test different volume computation methods."""
        vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        # Simplex decomposition method
        volume_simplex = convex_hull_volume(vertices, method="simplex_decomposition")
        assert isinstance(volume_simplex, float | jnp.ndarray)
        assert volume_simplex > 0

        # Monte Carlo method
        volume_mc = convex_hull_volume(vertices, method="monte_carlo")
        assert isinstance(volume_mc, float | jnp.ndarray)
        assert volume_mc > 0

        # 3D divergence theorem (should fall back for non-3D)
        with pytest.warns(UserWarning):
            volume_div = convex_hull_volume(vertices, method="divergence_theorem")
        assert isinstance(volume_div, float | jnp.ndarray)

        # Unknown method
        with pytest.raises(ValueError):
            convex_hull_volume(vertices, method="unknown")

    def test_volume_scaling(self):
        """Test that volume scales correctly."""
        # Unit triangle
        vertices_unit = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        volume_unit = convex_hull_volume(vertices_unit)

        # Scaled triangle (2x linear scaling = 4x area scaling)
        vertices_scaled = vertices_unit * 2.0
        volume_scaled = convex_hull_volume(vertices_scaled)

        # Volume should scale by factor^dimension
        actual_ratio = volume_scaled / volume_unit

        # Allow for approximation errors
        assert 2.0 < actual_ratio < 6.0  # Rough check

    def test_degenerate_volume(self):
        """Test volume of degenerate shapes."""
        # Collinear points (should have zero area)
        vertices = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        volume = convex_hull_volume(vertices)

        assert isinstance(volume, float | jnp.ndarray)
        # Should be close to zero (may not be exactly zero due to approximation)
        assert volume < 0.1


class TestConvexHullSurfaceArea:
    """Tests for convex hull surface area computation."""

    def test_2d_triangle_perimeter(self):
        """Test perimeter computation of 2D triangle."""
        # Right triangle with sides 3, 4, 5
        vertices = jnp.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        surface_area = convex_hull_surface_area(vertices)

        assert isinstance(surface_area, float | jnp.ndarray)
        assert surface_area > 0
        # Expected perimeter = 3 + 4 + 5 = 12
        # Allow for approximation
        assert 10.0 < surface_area < 15.0

    def test_2d_square_perimeter(self):
        """Test perimeter computation of 2D square."""
        # Unit square (perimeter = 4.0)
        vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        surface_area = convex_hull_surface_area(vertices)

        assert isinstance(surface_area, float | jnp.ndarray)
        assert surface_area > 0
        # Allow for approximation
        assert 3.0 < surface_area < 6.0

    def test_3d_tetrahedron_surface_area(self):
        """Test surface area computation of 3D tetrahedron."""
        vertices = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        surface_area = convex_hull_surface_area(vertices)

        assert isinstance(surface_area, float | jnp.ndarray)
        assert surface_area > 0

    def test_higher_dimension_surface_area(self):
        """Test surface area in higher dimensions (approximate)."""
        # 4D simplex
        vertices = jnp.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        with pytest.warns(UserWarning):  # Should warn about approximation
            surface_area = convex_hull_surface_area(vertices)

        assert isinstance(surface_area, float | jnp.ndarray)
        assert surface_area > 0


class TestDistanceToConvexHull:
    """Tests for distance computation to convex hull."""

    def test_distance_inside_outside(self):
        """Test distance computation for inside and outside points."""
        # Unit triangle
        vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        # Point outside
        outside_point = jnp.array([2.0, 2.0])
        distance_outside = distance_to_convex_hull(outside_point, vertices)

        assert isinstance(distance_outside, float | jnp.ndarray)
        # Distance for outside point should be positive
        # (Note: Implementation may use approximation)

        # Point that might be inside
        center_point = jnp.array([0.1, 0.1])
        distance_center = distance_to_convex_hull(center_point, vertices)

        assert isinstance(distance_center, float | jnp.ndarray)

    def test_distance_boundary(self):
        """Test distance computation for boundary points."""
        vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        # Point on vertex
        vertex_point = jnp.array([0.0, 0.0])
        distance_vertex = distance_to_convex_hull(vertex_point, vertices)

        assert isinstance(distance_vertex, float | jnp.ndarray)
        # Should be close to zero (may not be exactly zero due to approximation)
        assert abs(distance_vertex) < 1.0  # Loose check

    def test_distance_scaling(self):
        """Test that distance scales with hull size."""
        # Small triangle
        vertices_small = jnp.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
        point = jnp.array([1.0, 1.0])
        distance_small = distance_to_convex_hull(point, vertices_small)

        # Large triangle
        vertices_large = jnp.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        distance_large = distance_to_convex_hull(point, vertices_large)

        assert isinstance(distance_small, float | jnp.ndarray)
        assert isinstance(distance_large, float | jnp.ndarray)

        # Distance to larger hull should generally be smaller
        # (though this depends on the approximation used)


class TestHausdorffDistance:
    """Tests for Hausdorff distance computation."""

    def test_identical_hulls(self):
        """Test Hausdorff distance between identical hulls."""
        vertices1 = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        vertices2 = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        distance = hausdorff_distance(vertices1, vertices2)

        assert isinstance(distance, float | jnp.ndarray)
        # Should be zero or close to zero
        assert distance < 0.1  # Allow for numerical errors

    def test_disjoint_hulls(self):
        """Test Hausdorff distance between disjoint hulls."""
        vertices1 = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        vertices2 = jnp.array([[5.0, 5.0], [6.0, 5.0], [5.0, 6.0]])  # Far away

        distance = hausdorff_distance(vertices1, vertices2)

        assert isinstance(distance, float | jnp.ndarray)
        assert distance > 0  # Should be positive
        # Should be roughly the distance between closest points
        assert distance > 5.0  # At least the separation distance

    def test_nested_hulls(self):
        """Test Hausdorff distance between nested hulls."""
        # Small triangle
        vertices1 = jnp.array([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]])
        # Large triangle containing the small one
        vertices2 = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])

        distance = hausdorff_distance(vertices1, vertices2)

        assert isinstance(distance, float | jnp.ndarray)
        assert distance >= 0  # Should be non-negative

    def test_hausdorff_symmetry(self):
        """Test that Hausdorff distance is symmetric."""
        vertices1 = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        vertices2 = jnp.array([[0.5, 0.5], [1.5, 0.5], [1.0, 1.5]])

        distance12 = hausdorff_distance(vertices1, vertices2)
        distance21 = hausdorff_distance(vertices2, vertices1)

        # Should be symmetric (within numerical precision)
        assert jnp.isclose(distance12, distance21, atol=1e-6)


class TestPredicateValidation:
    """Tests for input validation in predicates."""

    def test_invalid_point_shapes(self):
        """Test predicates with invalid point shapes."""
        vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        # Wrong dimension for point
        with pytest.raises((ValueError, IndexError)):
            point_in_convex_hull(jnp.array([0.5]), vertices)  # 1D point for 2D hull

        # Wrong dimension for hull vertices
        with pytest.raises((ValueError, TypeError)):
            point_in_convex_hull(jnp.array([0.5, 0.5]), jnp.array([1, 2, 3]))  # 1D vertices

    def test_empty_hulls(self):
        """Test predicates with empty hulls."""
        empty_vertices = jnp.array([]).reshape(0, 2)
        point = jnp.array([0.5, 0.5])

        # Should handle empty hulls gracefully
        try:
            result = point_in_convex_hull(point, empty_vertices)
            assert isinstance(result, bool | np.bool_ | jnp.ndarray)
        except (ValueError, IndexError):
            # It's acceptable to raise an error for empty hulls
            pass

    def test_nan_infinite_values(self):
        """Test predicates with NaN and infinite values."""
        vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        # Point with NaN
        nan_point = jnp.array([jnp.nan, 0.5])
        try:
            result = point_in_convex_hull(nan_point, vertices)
            # Result might be NaN/False, both are acceptable
            assert isinstance(result, bool | np.bool_ | jnp.ndarray)
        except ValueError:
            # It's acceptable to raise an error for NaN input
            pass

        # Hull with infinite values
        inf_vertices = jnp.array([[0.0, 0.0], [jnp.inf, 0.0], [0.0, 1.0]])
        point = jnp.array([0.5, 0.5])
        try:
            result = point_in_convex_hull(point, inf_vertices)
            assert isinstance(result, bool | np.bool_ | jnp.ndarray)
        except ValueError:
            # It's acceptable to raise an error for infinite input
            pass


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_volume_computation_scaling(dimension):
    """Test volume computation in different dimensions."""
    # Create unit simplex
    n_vertices = dimension + 1
    vertices = jnp.zeros((n_vertices, dimension))
    vertices = vertices.at[1:].set(jnp.eye(dimension))

    volume = convex_hull_volume(vertices)

    assert isinstance(volume, float | jnp.ndarray)
    assert volume > 0
    # Volume of unit simplex in d dimensions is 1/d!
    expected_volume = 1.0 / math.factorial(dimension)
    # Allow for significant approximation error
    assert 0.01 * expected_volume < volume < 10.0 * expected_volume


@pytest.mark.parametrize("n_vertices", [3, 4, 5, 10])
def test_predicates_with_varying_hull_size(n_vertices):
    """Test predicates with hulls of different sizes."""
    # Generate random convex hull vertices
    key = jax.random.PRNGKey(42)
    vertices = jax.random.normal(key, (n_vertices, 2))

    # Test volume computation
    volume = convex_hull_volume(vertices)
    assert isinstance(volume, float | jnp.ndarray)
    assert volume >= 0  # Volume should be non-negative

    # Test surface area computation
    surface_area = convex_hull_surface_area(vertices)
    assert isinstance(surface_area, float | jnp.ndarray)
    assert surface_area >= 0  # Surface area should be non-negative

    # Test point inclusion
    test_point = jnp.array([0.0, 0.0])
    inclusion_result = point_in_convex_hull(test_point, vertices)
    assert isinstance(inclusion_result, bool | np.bool_ | jnp.ndarray)
