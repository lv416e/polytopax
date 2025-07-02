"""Tests for volume computation accuracy improvements.

This module tests and improves the accuracy of convex hull volume calculations
by implementing and comparing multiple computational methods.
"""

import jax
import jax.numpy as jnp
import pytest
import math

from polytopax.operations.predicates import convex_hull_volume, compute_volume_accuracy_metrics
from polytopax.algorithms.approximation import improved_approximate_convex_hull


class TestCurrentVolumeAccuracy:
    """Analyze current volume computation accuracy issues."""

    def test_triangle_area_accuracy(self):
        """Test 2D triangle area computation accuracy."""
        # Right triangle with known area
        triangle_vertices = jnp.array([
            [0.0, 0.0],  # Origin
            [3.0, 0.0],  # Base = 3
            [0.0, 4.0]   # Height = 4
        ])
        exact_area = 0.5 * 3.0 * 4.0  # = 6.0

        # Test different volume computation methods
        volume_simplex = convex_hull_volume(triangle_vertices, method="simplex_decomposition")

        print(f"Triangle area - Exact: {exact_area}")
        print(f"Triangle area - Simplex method: {volume_simplex}")

        relative_error = abs(volume_simplex - exact_area) / exact_area
        print(f"Relative error: {relative_error:.6f}")

        # Current method should be reasonably accurate for simple shapes
        assert relative_error < 0.1, f"Triangle area error too large: {relative_error}"

    def test_unit_square_area_accuracy(self):
        """Test unit square area computation."""
        square_vertices = jnp.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        exact_area = 1.0

        computed_volume = convex_hull_volume(square_vertices, method="simplex_decomposition")

        print(f"Square area - Exact: {exact_area}")
        print(f"Square area - Computed: {computed_volume}")

        relative_error = abs(computed_volume - exact_area) / exact_area
        print(f"Square relative error: {relative_error:.6f}")

        # Document current accuracy level
        assert relative_error < 0.2  # Allow 20% error for now

    def test_tetrahedron_volume_accuracy(self):
        """Test 3D tetrahedron volume computation."""
        # Regular tetrahedron with edge length a = 2
        a = 2.0
        h = a * jnp.sqrt(2/3)  # Height of regular tetrahedron

        tetrahedron_vertices = jnp.array([
            [0.0, 0.0, 0.0],           # Origin
            [a, 0.0, 0.0],             # X-axis
            [a/2, a*jnp.sqrt(3)/2, 0.0],  # Equilateral triangle base
            [a/2, a*jnp.sqrt(3)/6, h]     # Apex
        ])

        # Exact volume = a³/(6√2)
        exact_volume = (a**3) / (6 * jnp.sqrt(2))

        computed_volume = convex_hull_volume(tetrahedron_vertices, method="simplex_decomposition")

        print(f"Tetrahedron volume - Exact: {exact_volume:.6f}")
        print(f"Tetrahedron volume - Computed: {computed_volume:.6f}")

        relative_error = abs(computed_volume - exact_volume) / exact_volume
        print(f"Tetrahedron relative error: {relative_error:.6f}")

        # Document current 3D accuracy
        assert relative_error < 0.3  # Allow 30% error for 3D

    def test_volume_method_consistency(self):
        """Test consistency between different volume computation methods."""
        # Simple triangle for testing
        vertices = jnp.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        try:
            volume_simplex = convex_hull_volume(vertices, method="simplex_decomposition")
            volume_shoelace = convex_hull_volume(vertices, method="shoelace")

            print(f"Simplex method: {volume_simplex}")
            print(f"Shoelace method: {volume_shoelace}")

            # Check consistency between methods
            relative_diff = abs(volume_simplex - volume_shoelace) / max(volume_simplex, volume_shoelace)
            print(f"Method consistency error: {relative_diff:.6f}")

            assert relative_diff < 0.05  # Methods should agree within 5%

        except (NotImplementedError, ValueError) as e:
            print(f"Some methods not implemented: {e}")
            pytest.skip("Multiple methods not available yet")


class TestImprovedVolumeComputation:
    """Test improved volume computation methods."""

    def test_shoelace_formula_accuracy(self):
        """Test shoelace formula for 2D area computation."""
        # Right triangle
        triangle_vertices = jnp.array([
            [0.0, 0.0],  # Origin
            [3.0, 0.0],  # Base = 3
            [0.0, 4.0]   # Height = 4
        ])
        exact_area = 0.5 * 3.0 * 4.0  # = 6.0

        # Test shoelace method
        shoelace_area = convex_hull_volume(triangle_vertices, method="shoelace")

        print(f"Shoelace triangle area - Exact: {exact_area}")
        print(f"Shoelace triangle area - Computed: {shoelace_area}")

        relative_error = abs(shoelace_area - exact_area) / exact_area
        print(f"Shoelace relative error: {relative_error:.6f}")

        # Shoelace should be very accurate for 2D
        assert relative_error < 0.01  # 1% error threshold

    def test_multi_method_consensus(self):
        """Test multi-method consensus volume computation."""
        # Unit square
        square_vertices = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]
        ])
        exact_area = 1.0

        # Test multi-method consensus
        consensus_volume = convex_hull_volume(square_vertices, method="multi_method")

        print(f"Multi-method square area - Exact: {exact_area}")
        print(f"Multi-method square area - Computed: {consensus_volume}")

        relative_error = abs(consensus_volume - exact_area) / exact_area
        print(f"Multi-method relative error: {relative_error:.6f}")

        # Multi-method should be very accurate
        assert relative_error < 0.05  # 5% error threshold

    def test_volume_method_comparison(self):
        """Compare accuracy across different volume methods."""
        # Regular pentagon (known area formula)
        n = 5  # pentagon
        radius = 1.0
        angles = jnp.linspace(0, 2*jnp.pi, n, endpoint=False)
        pentagon_vertices = radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

        # Exact area = (1/2) * n * r² * sin(2π/n)
        exact_area = 0.5 * n * radius**2 * jnp.sin(2*jnp.pi/n)

        # Test different methods
        methods = ["simplex_decomposition", "shoelace", "multi_method"]
        results = {}

        for method in methods:
            try:
                volume = convex_hull_volume(pentagon_vertices, method=method)
                relative_error = abs(volume - exact_area) / exact_area
                results[method] = {
                    "volume": float(volume),
                    "relative_error": float(relative_error),
                    "accurate": relative_error < 0.05
                }
                print(f"{method}: volume={volume:.6f}, error={relative_error:.6f}")
            except Exception as e:
                print(f"{method} failed: {e}")
                results[method] = None

        # At least one method should be accurate
        accurate_methods = [k for k, v in results.items() if v and v["accurate"]]
        assert len(accurate_methods) > 0, "No method achieved 5% accuracy"

    def test_volume_accuracy_metrics(self):
        """Test volume accuracy metrics computation."""
        # Simple triangle with known area
        triangle_vertices = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]
        ])
        exact_area = 0.5

        # Get accuracy metrics
        metrics = compute_volume_accuracy_metrics(triangle_vertices, exact_area)

        print("Volume accuracy metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Check that metrics make sense
        assert "volumes" in metrics
        assert "method_consistency" in metrics
        assert "exact_volume" in metrics
        assert metrics["exact_volume"] == exact_area

        # At least one method should be accurate
        accurate_methods = [k for k, v in metrics.items() if k.endswith("_accurate") and v]
        assert len(accurate_methods) > 0, "No method achieved accuracy threshold"


class TestVolumeComputationIntegration:
    """Test volume computation with improved hull algorithm."""

    def test_improved_hull_volume_accuracy(self):
        """Test volume accuracy with improved hull algorithm."""
        # Unit square points
        points = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]
        ])

        # Get hull from improved algorithm
        hull_vertices, _ = improved_approximate_convex_hull(points)

        # Compute volume
        computed_volume = convex_hull_volume(hull_vertices)
        expected_volume = 1.0

        print(f"Improved hull volume - Expected: {expected_volume}")
        print(f"Improved hull volume - Computed: {computed_volume}")

        relative_error = abs(computed_volume - expected_volume) / expected_volume
        print(f"Improved hull volume error: {relative_error:.6f}")

        # Should be more accurate than before
        assert relative_error < 0.1

    def test_volume_computation_scaling(self):
        """Test volume computation scaling properties."""
        # Test that volume scales correctly with point scaling

        # Original triangle
        original_points = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]
        ])

        # Scaled triangle (2x)
        scale_factor = 2.0
        scaled_points = original_points * scale_factor

        # Get hulls
        original_hull, _ = improved_approximate_convex_hull(original_points)
        scaled_hull, _ = improved_approximate_convex_hull(scaled_points)

        # Compute volumes
        original_volume = convex_hull_volume(original_hull)
        scaled_volume = convex_hull_volume(scaled_hull)

        # Volume should scale by scale_factor^dimension
        expected_scaled_volume = original_volume * (scale_factor ** 2)  # 2D

        print(f"Original volume: {original_volume}")
        print(f"Scaled volume: {scaled_volume}")
        print(f"Expected scaled volume: {expected_scaled_volume}")

        scaling_error = abs(scaled_volume - expected_scaled_volume) / expected_scaled_volume
        print(f"Volume scaling error: {scaling_error:.6f}")

        # Volume scaling should be accurate
        assert scaling_error < 0.1


if __name__ == "__main__":
    # Run volume accuracy analysis
    current_tests = TestCurrentVolumeAccuracy()
    improved_tests = TestImprovedVolumeComputation()
    integration_tests = TestVolumeComputationIntegration()

    print("=== Current Volume Accuracy Analysis ===")
    current_tests.test_triangle_area_accuracy()
    print()
    current_tests.test_unit_square_area_accuracy()
    print()
    current_tests.test_tetrahedron_volume_accuracy()
    print()
    current_tests.test_volume_method_consistency()

    print("\n=== Improved Volume Computation Tests ===")
    improved_tests.test_shoelace_formula_accuracy()
    print()
    improved_tests.test_multi_method_consensus()
    print()
    improved_tests.test_volume_method_comparison()
    print()
    improved_tests.test_volume_accuracy_metrics()

    print("\n=== Volume Computation Integration Tests ===")
    integration_tests.test_improved_hull_volume_accuracy()
    print()
    integration_tests.test_volume_computation_scaling()
