"""Basic tests for PolytopAX package."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import polytopax
from polytopax.core.hull import convex_hull, approximate_convex_hull


def test_package_import():
    """Test that the package can be imported."""
    assert hasattr(polytopax, "__version__")
    assert isinstance(polytopax.__version__, str)
    assert polytopax.__version__ == "0.1.0"


def test_package_info():
    """Test package information functions."""
    info = polytopax.get_info()
    assert info["version"] == "0.1.0"
    assert info["core_available"] is True
    assert info["available_functions"] > 0


def test_convex_hull_basic():
    """Test basic convex hull computation."""
    # Simple 2D square
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = convex_hull(points, algorithm="approximate")
    
    # Result should be an array
    assert isinstance(result, jnp.ndarray)
    assert result.shape[-1] == 2  # 2D points
    assert result.shape[-2] > 0   # At least some vertices


def test_convex_hull_algorithms():
    """Test different algorithm options."""
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    
    # Approximate algorithm should work
    result = convex_hull(points, algorithm="approximate")
    assert isinstance(result, jnp.ndarray)
    
    # Unimplemented algorithms should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        convex_hull(points, algorithm="quickhull")
    
    with pytest.raises(NotImplementedError):
        convex_hull(points, algorithm="graham_scan")
    
    # Unknown algorithm should raise ValueError
    with pytest.raises(ValueError):
        convex_hull(points, algorithm="unknown")


def test_approximate_convex_hull_backward_compatibility():
    """Test backward compatibility of approximate_convex_hull function."""
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    
    hull_points, hull_indices = approximate_convex_hull(points, n_directions=20)
    
    # Check return types and shapes
    assert isinstance(hull_points, jnp.ndarray)
    assert isinstance(hull_indices, jnp.ndarray)
    assert hull_points.shape[-1] == 2  # 2D points
    assert hull_indices.shape[-1] == hull_points.shape[-2]  # Matching indices


def test_convex_hull_parameters():
    """Test convex hull with different parameters."""
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    
    # Different number of directions
    result1 = convex_hull(points, n_directions=10)
    result2 = convex_hull(points, n_directions=50)
    
    assert isinstance(result1, jnp.ndarray)
    assert isinstance(result2, jnp.ndarray)
    
    # Different methods
    result_uniform = convex_hull(points, method="uniform")
    assert isinstance(result_uniform, jnp.ndarray)
    
    # Different temperature
    result_low_temp = convex_hull(points, temperature=0.01)
    result_high_temp = convex_hull(points, temperature=1.0)
    
    assert isinstance(result_low_temp, jnp.ndarray)
    assert isinstance(result_high_temp, jnp.ndarray)


def test_jax_transformations():
    """Test JAX transformation compatibility."""
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    
    # Test JIT compilation
    jit_hull = jax.jit(convex_hull, static_argnames=['algorithm'])
    result = jit_hull(points, algorithm="approximate")
    assert isinstance(result, jnp.ndarray)
    
    # Test vmap (vectorization)
    batch_points = jnp.array([
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    ])
    
    vmap_hull = jax.vmap(convex_hull, in_axes=(0,))
    batch_result = vmap_hull(batch_points)
    assert isinstance(batch_result, jnp.ndarray)
    assert batch_result.shape[0] == 2  # Batch size


def test_input_validation():
    """Test input validation."""
    # Test invalid input types
    with pytest.raises(TypeError):
        convex_hull([[0, 0], [1, 0]])  # List instead of array
    
    # Test invalid shapes
    with pytest.raises(ValueError):
        convex_hull(jnp.array([1, 2, 3]))  # 1D array
    
    with pytest.raises(ValueError):
        convex_hull(jnp.array([]))  # Empty array
    
    # Test invalid numerical values
    with pytest.raises(ValueError):
        convex_hull(jnp.array([[jnp.nan, 0], [1, 0]]))  # NaN values
    
    with pytest.raises(ValueError):
        convex_hull(jnp.array([[jnp.inf, 0], [1, 0]]))  # Infinite values


def test_different_dimensions():
    """Test convex hull computation in different dimensions."""
    # 2D triangle
    points_2d = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    result_2d = convex_hull(points_2d)
    assert result_2d.shape[-1] == 2
    
    # 3D tetrahedron
    points_3d = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0], 
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    result_3d = convex_hull(points_3d)
    assert result_3d.shape[-1] == 3
    
    # Higher dimension
    points_4d = jnp.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    result_4d = convex_hull(points_4d)
    assert result_4d.shape[-1] == 4


def test_reproducibility():
    """Test that results are reproducible with same random seed."""
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    
    # Same seed should give same results
    result1 = convex_hull(points, random_key=jax.random.PRNGKey(42))
    result2 = convex_hull(points, random_key=jax.random.PRNGKey(42))
    
    assert jnp.allclose(result1, result2)
    
    # Different seeds may give different results
    result3 = convex_hull(points, random_key=jax.random.PRNGKey(123))
    # Note: Due to the nature of the algorithm, results might still be similar
