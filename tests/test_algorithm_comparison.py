"""Comparison tests between original and improved algorithms.

This module compares the performance and mathematical correctness
of the original and improved convex hull algorithms.
"""

import jax
import jax.numpy as jnp
import pytest

from polytopax.algorithms.approximation import (
    approximate_convex_hull,
    compute_hull_quality_metrics,
    improved_approximate_convex_hull,
)


class TestAlgorithmComparison:
    """Compare original and improved algorithms."""

    def test_vertex_count_improvement_triangle(self):
        """Test vertex count improvement with triangle."""
        points = jnp.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0]
        ])

        # Original algorithm
        old_hull, _ = approximate_convex_hull(
            points, n_directions=20, random_key=jax.random.PRNGKey(42)
        )

        # Improved algorithm
        new_hull, _ = improved_approximate_convex_hull(
            points, n_directions=20, random_key=jax.random.PRNGKey(42)
        )

        print(f"Original algorithm: {old_hull.shape[0]} vertices")
        print(f"Improved algorithm: {new_hull.shape[0]} vertices")

        # Improved algorithm should have ≤ input vertices
        assert new_hull.shape[0] <= points.shape[0]
        assert new_hull.shape[0] <= 3

        # Original algorithm likely violates this
        assert old_hull.shape[0] > points.shape[0]  # Documents the problem

    def test_vertex_count_improvement_square(self):
        """Test vertex count improvement with square."""
        points = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]
        ])

        # Original algorithm
        old_hull, _ = approximate_convex_hull(
            points, n_directions=16, random_key=jax.random.PRNGKey(123)
        )

        # Improved algorithm
        new_hull, _ = improved_approximate_convex_hull(
            points, n_directions=16, random_key=jax.random.PRNGKey(123)
        )

        print(f"Square - Original: {old_hull.shape[0]}, Improved: {new_hull.shape[0]}")

        # Improved algorithm respects constraint
        assert new_hull.shape[0] <= 4
        # Original algorithm violates constraint
        assert old_hull.shape[0] > 4

    def test_quality_metrics_comparison(self):
        """Compare quality metrics between algorithms."""
        points = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
            [0.5, 0.5]  # Interior point
        ])

        # Original algorithm
        old_hull, _ = approximate_convex_hull(points)
        old_metrics = compute_hull_quality_metrics(old_hull, points)

        # Improved algorithm
        new_hull, _ = improved_approximate_convex_hull(points)
        new_metrics = compute_hull_quality_metrics(new_hull, points)

        print("Original algorithm metrics:")
        for key, value in old_metrics.items():
            print(f"  {key}: {value}")

        print("Improved algorithm metrics:")
        for key, value in new_metrics.items():
            print(f"  {key}: {value}")

        # Improved algorithm should satisfy constraint
        assert new_metrics["constraint_satisfied"] is True
        # Original algorithm likely violates constraint
        assert old_metrics["constraint_satisfied"] is False

        # Improved algorithm should have better boundary efficiency
        assert new_metrics["boundary_efficiency"] >= old_metrics["boundary_efficiency"]

    def test_interior_point_handling(self):
        """Test how algorithms handle interior points."""
        # Square with many interior points
        corners = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        interior = jnp.array([[0.3, 0.3], [0.7, 0.3], [0.5, 0.7], [0.6, 0.6]])
        points = jnp.concatenate([corners, interior], axis=0)

        # Improved algorithm should still find only boundary points
        new_hull, _ = improved_approximate_convex_hull(points, n_directions=24)

        # Should have ≤ 4 vertices (the corners)
        assert new_hull.shape[0] <= 4

        # Check that the hull vertices are close to the original corners
        metrics = compute_hull_quality_metrics(new_hull, points)
        print(f"Interior points test - Hull vertices: {new_hull.shape[0]}")
        print(f"Boundary efficiency: {metrics['boundary_efficiency']:.3f}")

        # Should have high boundary efficiency (hull vertices match input points)
        assert metrics["boundary_efficiency"] >= 0.75

    def test_differentiability_preservation(self):
        """Test that improved algorithm maintains differentiability."""
        points = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, 1.0]
        ])

        def hull_area_objective(pts):
            hull_vertices, _ = improved_approximate_convex_hull(pts)
            # Simple area computation (for triangle)
            v0, v1, v2 = hull_vertices[0], hull_vertices[1], hull_vertices[2]
            return 0.5 * jnp.abs(jnp.cross(v1 - v0, v2 - v0))

        # Should be differentiable
        try:
            grad_fn = jax.grad(hull_area_objective)
            gradients = grad_fn(points)
            assert gradients.shape == points.shape
            assert not jnp.any(jnp.isnan(gradients))
            print("Differentiability test: PASSED")
        except Exception as e:
            pytest.fail(f"Differentiability test failed: {e}")


class TestMathematicalCorrectness:
    """Test mathematical correctness of improved algorithm."""

    def test_convex_hull_properties(self):
        """Test fundamental convex hull properties."""
        points = jnp.array([
            [0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0]
        ])

        hull_vertices, hull_indices = improved_approximate_convex_hull(points)

        # Property 1: Hull vertices should be a subset of input points
        for hull_vertex in hull_vertices:
            found_match = False
            for input_point in points:
                if jnp.allclose(hull_vertex, input_point, atol=1e-6):
                    found_match = True
                    break
            assert found_match, f"Hull vertex {hull_vertex} not found in input points"

        # Property 2: Number of hull vertices ≤ number of input points
        assert hull_vertices.shape[0] <= points.shape[0]

        # Property 3: All hull indices should be valid
        assert jnp.all(hull_indices >= 0)
        assert jnp.all(hull_indices < points.shape[0])

    def test_exact_hull_cases(self):
        """Test cases where we know the exact convex hull."""
        # Unit square - hull should be the 4 corners
        square = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]
        ])

        hull_vertices, _ = improved_approximate_convex_hull(square, n_directions=20)

        # Should have exactly 4 vertices for a square
        assert hull_vertices.shape[0] == 4

        # All hull vertices should match the input corners
        metrics = compute_hull_quality_metrics(hull_vertices, square)
        assert metrics["coverage"] == 1.0  # All input points are hull vertices
        assert metrics["boundary_efficiency"] == 1.0  # All hull vertices are input points


if __name__ == "__main__":
    # Run comparison tests
    test_comparison = TestAlgorithmComparison()
    test_math = TestMathematicalCorrectness()

    print("=== Algorithm Comparison Tests ===")
    test_comparison.test_vertex_count_improvement_triangle()
    test_comparison.test_vertex_count_improvement_square()
    test_comparison.test_quality_metrics_comparison()
    test_comparison.test_interior_point_handling()
    test_comparison.test_differentiability_preservation()

    print("\n=== Mathematical Correctness Tests ===")
    test_math.test_convex_hull_properties()
    test_math.test_exact_hull_cases()
