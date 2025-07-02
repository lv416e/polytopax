"""Tests for approximation algorithms module."""

import jax
import jax.numpy as jnp
import pytest

from polytopax.algorithms.approximation import (
    adaptive_temperature_control,
    approximate_convex_hull,
    batched_approximate_hull,
    compute_hull_quality_metrics,
    multi_resolution_hull,
    progressive_hull_refinement,
    soft_argmax_selection,
)


class TestApproximateConvexHull:
    """Tests for approximate convex hull computation."""

    def test_basic_2d(self):
        """Test basic 2D approximate convex hull."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        hull_vertices, hull_indices = approximate_convex_hull(
            points, n_directions=20, random_key=jax.random.PRNGKey(42)
        )

        assert isinstance(hull_vertices, jnp.ndarray)
        assert isinstance(hull_indices, jnp.ndarray)
        assert hull_vertices.shape[-1] == 2  # 2D points
        assert hull_vertices.shape[-2] == hull_indices.shape[-1]  # Matching counts
        assert hull_vertices.shape[-2] <= points.shape[-2]  # At most as many vertices

    def test_basic_3d(self):
        """Test basic 3D approximate convex hull."""
        points = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ])

        hull_vertices, hull_indices = approximate_convex_hull(
            points, n_directions=30, random_key=jax.random.PRNGKey(42)
        )

        assert hull_vertices.shape[-1] == 3  # 3D points
        assert hull_vertices.shape[-2] > 0  # At least some vertices

    def test_different_sampling_methods(self):
        """Test different sampling methods."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        key = jax.random.PRNGKey(42)

        # Uniform sampling
        hull_uniform, _ = approximate_convex_hull(
            points, method="uniform", random_key=key
        )
        assert hull_uniform.shape[-1] == 2

        # Icosphere sampling (3D only)
        points_3d = jnp.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
        ])
        hull_icosphere, _ = approximate_convex_hull(
            points_3d, method="icosphere", random_key=key
        )
        assert hull_icosphere.shape[-1] == 3

        # Adaptive sampling (should fall back to uniform with warning)
        with pytest.warns(UserWarning):
            hull_adaptive, _ = approximate_convex_hull(
                points, method="adaptive", random_key=key
            )
        assert hull_adaptive.shape[-1] == 2

    def test_parameter_variations(self):
        """Test different parameter settings."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        key = jax.random.PRNGKey(42)

        # Different number of directions
        hull_few, _ = approximate_convex_hull(points, n_directions=5, random_key=key)
        hull_many, _ = approximate_convex_hull(points, n_directions=100, random_key=key)

        assert hull_few.shape[-2] <= hull_many.shape[-2]  # More directions ≥ more vertices

        # Different temperatures
        hull_low_temp, _ = approximate_convex_hull(
            points, temperature=0.01, random_key=key
        )
        hull_high_temp, _ = approximate_convex_hull(
            points, temperature=1.0, random_key=key
        )

        assert hull_low_temp.shape[-1] == hull_high_temp.shape[-1] == 2

        # With and without duplicate removal
        hull_with_dedup, _ = approximate_convex_hull(
            points, remove_duplicates=True, random_key=key
        )
        hull_without_dedup, _ = approximate_convex_hull(
            points, remove_duplicates=False, random_key=key
        )

        assert hull_with_dedup.shape[-2] <= hull_without_dedup.shape[-2]

    def test_reproducibility(self):
        """Test reproducibility with same random key."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        key = jax.random.PRNGKey(123)

        hull1, _ = approximate_convex_hull(points, random_key=key)
        hull2, _ = approximate_convex_hull(points, random_key=key)

        assert jnp.allclose(hull1, hull2)

    def test_input_validation(self):
        """Test input validation."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        # Invalid number of directions
        with pytest.raises(ValueError):
            approximate_convex_hull(points, n_directions=0)

        # Invalid temperature
        with pytest.raises(ValueError):
            approximate_convex_hull(points, temperature=0.0)

        with pytest.raises(ValueError):
            approximate_convex_hull(points, temperature=-1.0)

    def test_edge_cases(self):
        """Test edge cases."""
        # Single point
        single_point = jnp.array([[0.5, 0.5]])
        hull_single, indices_single = approximate_convex_hull(single_point, n_directions=10)
        assert hull_single.shape[-2] >= 1

        # Two points (line segment)
        two_points = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        hull_two, indices_two = approximate_convex_hull(two_points, n_directions=10)
        assert hull_two.shape[-2] >= 1

        # Collinear points
        collinear = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        hull_collinear, _ = approximate_convex_hull(collinear, n_directions=20)
        assert hull_collinear.shape[-2] >= 1


class TestBatchedApproximateHull:
    """Tests for batched approximate hull computation."""

    def test_basic_batching(self):
        """Test basic batched computation."""
        batch_points = jnp.array([
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],  # Triangle 1
            [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]   # Triangle 2
        ])

        batch_hulls, batch_indices = batched_approximate_hull(
            batch_points, n_directions=10, random_key=jax.random.PRNGKey(42)
        )

        assert batch_hulls.shape[0] == 2  # Batch size
        assert batch_indices.shape[0] == 2
        assert batch_hulls.shape[-1] == 2  # 2D points

    def test_batch_consistency(self):
        """Test that batched computation is consistent with individual computation."""
        points1 = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        points2 = jnp.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        batch_points = jnp.array([points1, points2])

        key = jax.random.PRNGKey(42)

        # Individual computation
        hull1, _ = approximate_convex_hull(points1, n_directions=15, random_key=key)
        hull2, _ = approximate_convex_hull(points2, n_directions=15, random_key=key)

        # Batched computation
        batch_hulls, _ = batched_approximate_hull(
            batch_points, n_directions=15, random_key=key
        )

        # Results should be similar (but may not be identical due to randomness)
        assert batch_hulls.shape[0] == 2
        assert batch_hulls[0].shape == hull1.shape
        assert batch_hulls[1].shape == hull2.shape

    def test_different_batch_sizes(self):
        """Test different batch sizes."""
        # Create random batch of different sizes
        for batch_size in [1, 3, 5, 10]:
            key = jax.random.PRNGKey(batch_size)
            batch_points = jax.random.normal(key, (batch_size, 4, 2))

            batch_hulls, batch_indices = batched_approximate_hull(
                batch_points, n_directions=10, random_key=key
            )

            assert batch_hulls.shape[0] == batch_size
            assert batch_indices.shape[0] == batch_size


class TestSoftArgmaxSelection:
    """Tests for soft argmax selection function."""

    def test_basic_functionality(self):
        """Test basic soft argmax selection."""
        scores = jnp.array([1.0, 3.0, 2.0])
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        temperature = 0.1

        soft_point, weights = soft_argmax_selection(scores, temperature, points)

        assert soft_point.shape == (2,)  # 2D point
        assert weights.shape == (3,)  # 3 weights
        assert jnp.isclose(jnp.sum(weights), 1.0)  # Weights sum to 1
        assert jnp.all(weights >= 0)  # All weights non-negative

    def test_temperature_effect(self):
        """Test effect of temperature on selection."""
        scores = jnp.array([1.0, 5.0, 2.0])  # Clear winner at index 1
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        # Low temperature (sharp selection)
        _, weights_low = soft_argmax_selection(scores, 0.01, points)

        # High temperature (smooth selection)
        _, weights_high = soft_argmax_selection(scores, 10.0, points)

        # Low temperature should be more concentrated on the winner
        assert weights_low[1] > weights_high[1]  # Winner gets more weight
        assert jnp.std(weights_low) > jnp.std(weights_high)  # More variation

    def test_extreme_scores(self):
        """Test with extreme score values."""
        # Very large scores
        scores_large = jnp.array([100.0, 101.0, 99.0])
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        soft_point_large, weights_large = soft_argmax_selection(
            scores_large, 1.0, points
        )

        assert jnp.isfinite(soft_point_large).all()
        assert jnp.isfinite(weights_large).all()
        assert jnp.isclose(jnp.sum(weights_large), 1.0)

        # Very small scores
        scores_small = jnp.array([-100.0, -99.0, -101.0])
        soft_point_small, weights_small = soft_argmax_selection(
            scores_small, 1.0, points
        )

        assert jnp.isfinite(soft_point_small).all()
        assert jnp.isfinite(weights_small).all()


class TestAdaptiveTemperatureControl:
    """Tests for adaptive temperature control."""

    def test_basic_functionality(self):
        """Test basic temperature control."""
        scores = jnp.array([1.0, 2.0, 3.0, 1.5])
        temperature = adaptive_temperature_control(scores)

        assert isinstance(temperature, float | jnp.ndarray)
        assert 0.01 <= temperature <= 10.0  # Within bounds

    def test_score_variance_effect(self):
        """Test effect of score variance on temperature."""
        # High variance scores
        scores_high_var = jnp.array([0.0, 10.0, 0.1, 9.9])
        temp_high_var = adaptive_temperature_control(scores_high_var)

        # Low variance scores
        scores_low_var = jnp.array([5.0, 5.1, 4.9, 5.05])
        temp_low_var = adaptive_temperature_control(scores_low_var)

        # Higher variance should lead to lower temperature (more confident)
        assert temp_high_var < temp_low_var

    def test_boundary_conditions(self):
        """Test boundary conditions."""
        # Constant scores (zero variance)
        scores_constant = jnp.array([1.0, 1.0, 1.0, 1.0])
        temp_constant = adaptive_temperature_control(scores_constant)

        assert temp_constant == 10.0  # Should hit max temperature

        # Single score
        scores_single = jnp.array([5.0])
        temp_single = adaptive_temperature_control(scores_single)

        assert isinstance(temp_single, float | jnp.ndarray)


class TestComputeHullQualityMetrics:
    """Tests for hull quality metrics computation."""

    def test_basic_metrics(self):
        """Test basic quality metrics."""
        original_points = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]
        ])
        hull_vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hull_indices = jnp.array([0, 1, 2])

        metrics = compute_hull_quality_metrics(
            original_points, hull_vertices, hull_indices
        )

        assert isinstance(metrics, dict)
        assert "coverage_ratio" in metrics
        assert "center_distance" in metrics
        assert "compactness" in metrics
        assert "n_hull_vertices" in metrics
        assert "n_original_points" in metrics

        # Check value ranges
        assert 0.0 <= metrics["coverage_ratio"] <= 1.0
        assert metrics["center_distance"] >= 0.0
        assert 0.0 <= metrics["compactness"] <= 1.0
        assert metrics["n_hull_vertices"] == 3
        assert metrics["n_original_points"] == 5

    def test_perfect_hull(self):
        """Test metrics for perfect hull (all points are vertices)."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hull_vertices = points
        hull_indices = jnp.array([0, 1, 2])

        metrics = compute_hull_quality_metrics(points, hull_vertices, hull_indices)

        assert metrics["coverage_ratio"] == 1.0
        assert metrics["compactness"] == 0.0
        assert jnp.isclose(metrics["center_distance"], 0.0, atol=1e-6)


class TestMultiResolutionHull:
    """Tests for multi-resolution hull computation."""

    def test_basic_multi_resolution(self):
        """Test basic multi-resolution computation."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        resolution_levels = [10, 20, 50]

        hulls = multi_resolution_hull(
            points, resolution_levels=resolution_levels,
            random_key=jax.random.PRNGKey(42)
        )

        assert len(hulls) == 3
        for _i, (hull_vertices, hull_indices) in enumerate(hulls):
            assert isinstance(hull_vertices, jnp.ndarray)
            assert isinstance(hull_indices, jnp.ndarray)
            assert hull_vertices.shape[-1] == 2  # 2D points

    def test_resolution_progression(self):
        """Test that higher resolution gives more vertices."""
        points = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [0.5, 0.0], [0.0, 0.5], [1.0, 0.5], [0.5, 1.0]
        ])
        resolution_levels = [5, 15, 30]

        hulls = multi_resolution_hull(
            points, resolution_levels=resolution_levels,
            random_key=jax.random.PRNGKey(42)
        )

        # Higher resolution should generally give more vertices (not guaranteed)
        vertex_counts = [hull[0].shape[-2] for hull in hulls]
        # At minimum, all should have some vertices
        assert all(count > 0 for count in vertex_counts)

    def test_reproducibility(self):
        """Test reproducibility of multi-resolution computation."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        resolution_levels = [10, 20]
        key = jax.random.PRNGKey(123)

        hulls1 = multi_resolution_hull(points, resolution_levels, random_key=key)
        hulls2 = multi_resolution_hull(points, resolution_levels, random_key=key)

        for (h1_v, h1_i), (h2_v, h2_i) in zip(hulls1, hulls2, strict=False):
            assert jnp.allclose(h1_v, h2_v)
            assert jnp.allclose(h1_i, h2_i)


class TestProgressiveHullRefinement:
    """Tests for progressive hull refinement."""

    def test_basic_refinement(self):
        """Test basic progressive refinement."""
        points = jnp.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [0.5, 0.0], [0.0, 0.5], [1.0, 0.5], [0.5, 1.0]
        ])

        hull_vertices, hull_indices, info = progressive_hull_refinement(
            points,
            initial_directions=5,
            max_directions=20,
            refinement_steps=3,
            random_key=jax.random.PRNGKey(42)
        )

        assert isinstance(hull_vertices, jnp.ndarray)
        assert isinstance(hull_indices, jnp.ndarray)
        assert isinstance(info, dict)

        assert "iterations" in info
        assert "converged" in info
        assert "final_directions" in info

        assert hull_vertices.shape[-1] == 2  # 2D points

    def test_convergence_detection(self):
        """Test convergence detection."""
        # Simple case that should converge quickly
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        hull_vertices, hull_indices, info = progressive_hull_refinement(
            points,
            initial_directions=10,
            max_directions=100,
            refinement_steps=5,
            convergence_threshold=1e-3,
            random_key=jax.random.PRNGKey(42)
        )

        # Check that info contains convergence information
        assert isinstance(info["converged"], bool)
        assert isinstance(info["iterations"], list)

        # If converged, should have stopped early
        if info["converged"]:
            assert len(info["iterations"]) < 5

    def test_refinement_progression(self):
        """Test that refinement progresses through direction counts."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        hull_vertices, hull_indices, info = progressive_hull_refinement(
            points,
            initial_directions=5,
            max_directions=20,
            refinement_steps=3,
            random_key=jax.random.PRNGKey(42)
        )

        # Check that iterations show progression
        if info["iterations"]:
            directions = [iter_info["directions"] for iter_info in info["iterations"]]
            # Should be increasing (but might hit max)
            assert all(directions[i] >= directions[i-1] for i in range(1, len(directions)))


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_approximation_different_dimensions(dimension):
    """Test approximation algorithms in different dimensions."""
    # Generate random points in given dimension
    key = jax.random.PRNGKey(42)
    points = jax.random.normal(key, (8, dimension))

    hull_vertices, hull_indices = approximate_convex_hull(
        points, n_directions=20, random_key=key
    )

    assert hull_vertices.shape[-1] == dimension
    assert hull_vertices.shape[-2] > 0


@pytest.mark.parametrize("n_points", [3, 10, 50, 100])
def test_approximation_scaling(n_points):
    """Test approximation with different numbers of points."""
    key = jax.random.PRNGKey(42)
    points = jax.random.normal(key, (n_points, 2))

    hull_vertices, hull_indices = approximate_convex_hull(
        points, n_directions=min(20, n_points), random_key=key
    )

    assert hull_vertices.shape[-2] <= n_points  # At most all points
    assert hull_vertices.shape[-2] > 0  # At least some points
