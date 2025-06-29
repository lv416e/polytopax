"""Basic tests for PolytopAX package."""

import pytest
import jax.numpy as jnp
import polytopax
from polytopax.core.hull import convex_hull, approximate_convex_hull


def test_package_import():
    """Test that the package can be imported."""
    assert hasattr(polytopax, "__version__")
    assert isinstance(polytopax.__version__, str)


def test_convex_hull_placeholder():
    """Test placeholder convex hull function."""
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    # Test that the function runs without error
    result = convex_hull(points, algorithm="approximate")
    assert result.shape == points.shape

    # Test that unimplemented algorithms raise NotImplementedError
    with pytest.raises(NotImplementedError):
        convex_hull(points, algorithm="quickhull")


def test_approximate_convex_hull_placeholder():
    """Test approximate convex hull placeholder function."""
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    hull_points, hull_indices = approximate_convex_hull(points)

    # Check that function returns expected types and shapes
    assert hull_points.shape == points.shape
    assert hull_indices.shape == (points.shape[0],)
    assert jnp.allclose(hull_points, points)


def test_jax_compatibility():
    """Test that JAX operations work correctly."""
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    # Test that we can use JAX operations
    result = convex_hull(points)
    assert isinstance(result, jnp.ndarray)

    # Test that the function works with JAX transformations
    from jax import jit

    jit_hull = jit(convex_hull)
    jit_result = jit_hull(points)
    assert jnp.allclose(result, jit_result)
