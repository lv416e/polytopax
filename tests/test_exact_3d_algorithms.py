"""Tests for 3D exact convex hull algorithms.

This module tests the 3D QuickHull implementation and 3D geometric predicates.
"""

import jax
import jax.numpy as jnp
import pytest

from polytopax.algorithms.exact_3d import (
    quickhull_3d,
    orientation_3d,
    point_to_plane_distance_3d,
    is_point_inside_tetrahedron_3d,
    _find_initial_tetrahedron_3d,
    _handle_degenerate_3d_case,
)


class TestQuickHull3D:
    """Test 3D QuickHull implementation."""
    
    def test_tetrahedron_basic(self):
        """Test QuickHull on a simple tetrahedron."""
        # Regular tetrahedron
        points = jnp.array([
            [0.0, 0.0, 0.0],         # Origin
            [1.0, 0.0, 0.0],         # X-axis
            [0.5, jnp.sqrt(3)/2, 0.0],  # Equilateral triangle base
            [0.5, jnp.sqrt(3)/6, jnp.sqrt(2/3)]  # Apex
        ])
        
        hull_vertices, hull_indices = quickhull_3d(points)
        
        # All points should be on the hull for a tetrahedron
        assert hull_vertices.shape[0] == 4
        assert len(hull_indices) == 4
        
        # Check that hull vertices are from original points
        for hull_vertex in hull_vertices:
            found = False
            for original_point in points:
                if jnp.allclose(hull_vertex, original_point, atol=1e-10):
                    found = True
                    break
            assert found, f"Hull vertex {hull_vertex} not found in original points"
    
    def test_cube_with_interior_points(self):
        """Test QuickHull on cube with interior points."""
        # Unit cube corners + interior points
        points = jnp.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],  # Bottom face
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],  # Top face
            [0.5, 0.5, 0.5], [0.3, 0.3, 0.3], [0.7, 0.7, 0.7]                    # Interior
        ])
        
        hull_vertices, hull_indices = quickhull_3d(points)
        
        # Should find only the 8 corners
        assert hull_vertices.shape[0] <= 8
        
        # All hull vertices should be from the original corners
        corners = points[:8]
        for hull_vertex in hull_vertices:
            found = False
            for corner in corners:
                if jnp.allclose(hull_vertex, corner, atol=1e-10):
                    found = True
                    break
            assert found, f"Hull vertex {hull_vertex} should be one of the corners"
    
    def test_coplanar_points(self):
        """Test QuickHull on coplanar points (should reduce to 2D problem)."""
        # Points on the XY plane
        points = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0]
        ])
        
        hull_vertices, hull_indices = quickhull_3d(points)
        
        # Should find the convex hull in the plane
        assert hull_vertices.shape[0] <= 4  # At most the 4 corners
        
        # All Z coordinates should be zero
        assert jnp.allclose(hull_vertices[:, 2], 0.0, atol=1e-10)
    
    def test_collinear_points_3d(self):
        """Test QuickHull on collinear points in 3D."""
        # Points on a line in 3D
        t_values = jnp.array([0.0, 1.0, 2.0, 0.5, 1.5])
        direction = jnp.array([1.0, 1.0, 1.0])
        points = t_values[:, None] * direction[None, :]
        
        hull_vertices, hull_indices = quickhull_3d(points)
        
        # Should find only the two extreme points
        assert hull_vertices.shape[0] == 2
        
        # Check that we have the two extreme points
        distances = jnp.linalg.norm(hull_vertices, axis=1)
        distances_sorted = jnp.sort(distances)
        
        expected_min = 0.0
        expected_max = 2.0 * jnp.sqrt(3)
        
        assert jnp.allclose(distances_sorted[0], expected_min, atol=1e-10)
        assert jnp.allclose(distances_sorted[1], expected_max, atol=1e-10)


class Test3DGeometricPredicates:
    """Test 3D geometric predicates."""
    
    def test_orientation_3d(self):
        """Test 3D orientation predicate."""
        # Test points above and below a plane
        p1 = jnp.array([0.0, 0.0, 0.0])
        p2 = jnp.array([1.0, 0.0, 0.0])
        p3 = jnp.array([0.0, 1.0, 0.0])
        
        # Point above the XY plane
        p4_above = jnp.array([0.0, 0.0, 1.0])
        result_above = orientation_3d(p1, p2, p3, p4_above)
        assert result_above != 0  # Should not be coplanar
        
        # Point below the XY plane
        p4_below = jnp.array([0.0, 0.0, -1.0])
        result_below = orientation_3d(p1, p2, p3, p4_below)
        assert result_below != 0  # Should not be coplanar
        
        # The two results should have opposite signs
        assert result_above * result_below < 0
        
        # Point on the plane
        p4_on = jnp.array([0.5, 0.5, 0.0])
        assert orientation_3d(p1, p2, p3, p4_on) == 0  # Coplanar
    
    def test_point_to_plane_distance_3d(self):
        """Test point to plane distance calculation."""
        # XY plane at Z=0
        plane_point = jnp.array([0.0, 0.0, 0.0])
        plane_normal = jnp.array([0.0, 0.0, 1.0])  # Z direction
        
        # Point above the plane
        point_above = jnp.array([1.0, 1.0, 2.0])
        distance_above = point_to_plane_distance_3d(point_above, plane_point, plane_normal)
        assert jnp.allclose(distance_above, 2.0, atol=1e-10)
        
        # Point below the plane
        point_below = jnp.array([1.0, 1.0, -1.5])
        distance_below = point_to_plane_distance_3d(point_below, plane_point, plane_normal)
        assert jnp.allclose(distance_below, -1.5, atol=1e-10)
        
        # Point on the plane
        point_on = jnp.array([1.0, 1.0, 0.0])
        distance_on = point_to_plane_distance_3d(point_on, plane_point, plane_normal)
        assert jnp.allclose(distance_on, 0.0, atol=1e-10)
    
    def test_point_inside_tetrahedron_3d(self):
        """Test point in tetrahedron test."""
        # Unit tetrahedron
        tetrahedron = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Point inside
        point_inside = jnp.array([0.2, 0.2, 0.2])
        assert is_point_inside_tetrahedron_3d(point_inside, tetrahedron)
        
        # Point outside
        point_outside = jnp.array([1.0, 1.0, 1.0])
        assert not is_point_inside_tetrahedron_3d(point_outside, tetrahedron)
        
        # Point at vertex
        point_at_vertex = jnp.array([0.0, 0.0, 0.0])
        assert is_point_inside_tetrahedron_3d(point_at_vertex, tetrahedron)
        
        # Point on face
        point_on_face = jnp.array([0.3, 0.3, 0.0])
        assert is_point_inside_tetrahedron_3d(point_on_face, tetrahedron)


class TestQuickHull3DExactness:
    """Test mathematical exactness of 3D QuickHull algorithm."""
    
    def test_vertex_count_constraint_3d(self):
        """Test that 3D QuickHull never produces more vertices than input."""
        # Generate test cases
        test_cases = [
            jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),  # Tetrahedron
            jax.random.normal(jax.random.PRNGKey(42), (8, 3)),  # Random points
            jax.random.normal(jax.random.PRNGKey(123), (15, 3))  # More random points
        ]
        
        for points in test_cases:
            hull_vertices, hull_indices = quickhull_3d(points)
            
            # QuickHull should never produce more vertices than input
            assert hull_vertices.shape[0] <= points.shape[0], \
                f"3D QuickHull produced {hull_vertices.shape[0]} vertices from {points.shape[0]} input points"
            
            # All hull indices should be valid
            assert jnp.all(hull_indices >= 0)
            assert jnp.all(hull_indices < points.shape[0])
    
    def test_3d_hull_vertices_are_subset(self):
        """Test that all 3D hull vertices are from the original point set."""
        points = jnp.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
            [0.5, 0.5, 0.5]
        ])
        
        hull_vertices, hull_indices = quickhull_3d(points)
        
        # Each hull vertex should exactly match one of the original points
        for hull_vertex in hull_vertices:
            found_exact_match = False
            for original_point in points:
                if jnp.allclose(hull_vertex, original_point, atol=1e-12):
                    found_exact_match = True
                    break
            
            assert found_exact_match, \
                f"3D hull vertex {hull_vertex} is not an exact match to any original point"


class TestQuickHull3DPerformance:
    """Test 3D QuickHull performance and edge cases."""
    
    def test_degenerate_cases_3d(self):
        """Test 3D QuickHull on degenerate cases."""
        # Single point
        single_point = jnp.array([[0.0, 0.0, 0.0]])
        hull_vertices, hull_indices = quickhull_3d(single_point)
        assert hull_vertices.shape[0] == 1
        
        # Two points
        two_points = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        hull_vertices, hull_indices = quickhull_3d(two_points)
        assert hull_vertices.shape[0] == 2
        
        # Three points (should form a triangle)
        three_points = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        hull_vertices, hull_indices = quickhull_3d(three_points)
        assert hull_vertices.shape[0] == 3
    
    def test_initial_tetrahedron_finding(self):
        """Test the initial tetrahedron finding algorithm."""
        # Test with a clear tetrahedron
        points = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5]  # Interior point
        ])
        
        tetrahedron_indices = _find_initial_tetrahedron_3d(points, 1e-12)
        
        # Should find 4 points for the tetrahedron
        assert len(tetrahedron_indices) == 4
        
        # The tetrahedron should be the first 4 points (corners)
        for i in range(4):
            assert i in tetrahedron_indices
    
    def test_degenerate_case_handling(self):
        """Test handling of degenerate cases."""
        # Collinear points
        collinear_points = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0]
        ])
        
        hull_vertices, hull_indices = _handle_degenerate_3d_case(
            collinear_points, [0, 2], 1e-12
        )
        
        # Should return the two extreme points
        assert hull_vertices.shape[0] == 2
        
        # Coplanar points
        coplanar_points = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0]
        ])
        
        hull_vertices, hull_indices = _handle_degenerate_3d_case(
            coplanar_points, [0, 1, 2], 1e-12
        )
        
        # Should return a 2D convex hull (triangle in this case)
        assert hull_vertices.shape[0] >= 3
        assert jnp.allclose(hull_vertices[:, 2], 0.0, atol=1e-10)  # All Z=0


if __name__ == "__main__":
    # Run basic tests
    test_quickhull_3d = TestQuickHull3D()
    test_predicates_3d = Test3DGeometricPredicates()
    test_exactness_3d = TestQuickHull3DExactness()
    test_performance_3d = TestQuickHull3DPerformance()
    
    print("=== QuickHull 3D Tests ===")
    test_quickhull_3d.test_tetrahedron_basic()
    test_quickhull_3d.test_cube_with_interior_points()
    test_quickhull_3d.test_coplanar_points()
    test_quickhull_3d.test_collinear_points_3d()
    print("✓ All QuickHull 3D tests passed")
    
    print("\n=== 3D Geometric Predicates Tests ===")
    test_predicates_3d.test_orientation_3d()
    test_predicates_3d.test_point_to_plane_distance_3d()
    test_predicates_3d.test_point_inside_tetrahedron_3d()
    print("✓ All 3D geometric predicates tests passed")
    
    print("\n=== QuickHull 3D Exactness Tests ===")
    test_exactness_3d.test_vertex_count_constraint_3d()
    test_exactness_3d.test_3d_hull_vertices_are_subset()
    print("✓ All 3D exactness tests passed")
    
    print("\n=== QuickHull 3D Performance Tests ===")
    test_performance_3d.test_degenerate_cases_3d()
    test_performance_3d.test_initial_tetrahedron_finding()
    test_performance_3d.test_degenerate_case_handling()
    print("✓ All 3D performance tests passed")
    
    print("\n🎯 All 3D QuickHull tests completed successfully!")