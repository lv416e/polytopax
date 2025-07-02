"""Convex hull computation functions - unified interface."""


import jax
from jax import Array

from ..algorithms.approximation import approximate_convex_hull as _approximate_convex_hull
from ..operations.predicates import convex_hull_surface_area as hull_surface_area
from ..operations.predicates import convex_hull_volume as hull_volume
from ..operations.predicates import distance_to_convex_hull as distance_to_hull
from ..operations.predicates import point_in_convex_hull as point_in_hull
from .utils import HullVertices, PointCloud, validate_point_cloud


def convex_hull(
    points: PointCloud,
    algorithm: str = "approximate",
    **kwargs
) -> HullVertices:
    """Compute convex hull of a set of points.

    This is the main unified interface for convex hull computation.
    It supports multiple algorithms and provides a consistent API.

    Args:
        points: Input points array with shape (..., n_points, dimension)
        algorithm: Algorithm to use:
            - "approximate": Differentiable approximate hull (default)
            - "quickhull": Exact Quickhull algorithm (Phase 2)
            - "graham_scan": 2D Graham scan algorithm (Phase 2)
        **kwargs: Algorithm-specific parameters passed to the underlying function

    Returns:
        Array of convex hull vertices

    Example:
        >>> import jax.numpy as jnp
        >>> points = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        >>> hull_vertices = convex_hull(points, algorithm="approximate")
        >>> print(hull_vertices.shape)  # (n_hull_vertices, 2)

    Algorithm-specific parameters:
        For algorithm="approximate":
            - n_directions (int): Number of sampling directions (default: 100)
            - method (str): Sampling method ("uniform", "icosphere", "adaptive")
            - temperature (float): Softmax temperature for differentiability
            - random_key (Array): JAX random key
    """
    points = validate_point_cloud(points)

    if algorithm == "approximate":
        hull_vertices, _ = _approximate_convex_hull(points, **kwargs)
        return hull_vertices
    elif algorithm == "quickhull":
        raise NotImplementedError(
            "Quickhull algorithm will be implemented in Phase 2. "
            "Use algorithm='approximate' for now."
        )
    elif algorithm == "graham_scan":
        raise NotImplementedError(
            "Graham scan algorithm will be implemented in Phase 2. "
            "Use algorithm='approximate' for now."
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def approximate_convex_hull(
    points: Array,
    n_directions: int = 100,
    method: str = "uniform",
    random_seed: int = 0
) -> tuple[Array, Array]:
    """Differentiable approximate convex hull computation.

    This function maintains backward compatibility with the original API
    while forwarding to the new implementation.

    Args:
        points: Point cloud with shape [..., n_points, dim]
        n_directions: Number of sampling directions
        method: Sampling strategy ('uniform', 'adaptive', 'icosphere')
        random_seed: Random seed

    Returns:
        Tuple of (hull_points, hull_indices)

    Note:
        This function is maintained for backward compatibility.
        For new code, consider using the unified convex_hull() function
        or the algorithms.approximation module directly.
    """
    # Convert to new API parameters
    random_key = jax.random.PRNGKey(random_seed) if random_seed else None

    return _approximate_convex_hull(
        points,
        n_directions=n_directions,
        method=method,
        random_key=random_key
    )


# Re-export key functions for convenience
__all__ = [
    "approximate_convex_hull",
    "convex_hull",
    "distance_to_hull",
    "hull_surface_area",
    "hull_volume",
    "point_in_hull"
]


# JIT-compiled versions for performance
convex_hull_jit = jax.jit(convex_hull, static_argnames=['algorithm'])
approximate_convex_hull_jit = jax.jit(approximate_convex_hull, static_argnames=['method'])
