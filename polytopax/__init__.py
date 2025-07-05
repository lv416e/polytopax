"""PolytopAX: JAX-based computational geometry library.

A high-performance convex hull computation and polytope manipulation library
built on the JAX ecosystem with support for automatic differentiation and GPU acceleration.

Features:
- Differentiable approximate convex hull computation
- JAX-native implementation for GPU/TPU acceleration
- Compatible with jit, grad, vmap, and other JAX transformations
- Object-oriented and functional APIs
- Geometric predicates and polytope operations

Examples:
    Basic usage:
        >>> import polytopax as ptx
        >>> import jax.numpy as jnp
        >>> points = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        >>> hull_vertices = ptx.convex_hull(points)

    Object-oriented API:
        >>> hull = ptx.ConvexHull.from_points(points)
        >>> print(f"Volume: {hull.volume()}")

    Machine learning integration:
        >>> import jax
        >>> grad_fn = jax.grad(lambda pts: ptx.ConvexHull.from_points(pts).volume())
"""

__version__ = "0.1.0"
__author__ = "PolytopAX Development Team"

# Core imports - using lazy loading to avoid circular imports
_CORE_AVAILABLE = True
_import_error = None


def _lazy_import():
    """Lazy import to avoid circular dependencies."""
    global _CORE_AVAILABLE, _import_error
    if _import_error:
        return False

    try:
        # Test import to check for circular dependencies
        from .core.polytope import ConvexHull  # noqa: F401
        from .core.utils import (  # noqa: F401
            generate_direction_vectors,
            remove_duplicate_points,
            scale_to_unit_ball,
            validate_point_cloud,
        )
        from .operations.predicates import (  # noqa: F401
            convex_hull_surface_area,
            convex_hull_volume,
            distance_to_convex_hull,
            hausdorff_distance,
            point_in_convex_hull,
        )

        return True
    except ImportError as e:
        _CORE_AVAILABLE = False
        _import_error = e
        print(f"Warning: Core PolytopAX functionality not available: {e}")
        return False


def __getattr__(name):
    """Lazy attribute access for imported modules."""
    if not _lazy_import():
        raise AttributeError(f"module '{__name__}' has no attribute '{name}' (core functionality unavailable)")

    # Handle hull functions
    if name in ("convex_hull", "approximate_convex_hull"):
        from .core.hull import approximate_convex_hull, convex_hull

        if name == "convex_hull":
            return convex_hull
        elif name == "approximate_convex_hull":
            return approximate_convex_hull

    # Handle approximation algorithms
    elif name in (
        "approximate_hull_advanced",
        "batched_approximate_hull",
        "multi_resolution_hull",
        "progressive_hull_refinement",
    ):
        from .algorithms.approximation import (
            approximate_convex_hull as approximate_hull_advanced,
        )
        from .algorithms.approximation import (
            batched_approximate_hull,
            multi_resolution_hull,
            progressive_hull_refinement,
        )

        return {
            "approximate_hull_advanced": approximate_hull_advanced,
            "batched_approximate_hull": batched_approximate_hull,
            "multi_resolution_hull": multi_resolution_hull,
            "progressive_hull_refinement": progressive_hull_refinement,
        }[name]

    # Handle core classes
    elif name == "ConvexHull":
        from .core.polytope import ConvexHull

        return ConvexHull

    # Handle utilities
    elif name in (
        "generate_direction_vectors",
        "remove_duplicate_points",
        "scale_to_unit_ball",
        "validate_point_cloud",
    ):
        from .core.utils import (
            generate_direction_vectors,
            remove_duplicate_points,
            scale_to_unit_ball,
            validate_point_cloud,
        )

        return {
            "generate_direction_vectors": generate_direction_vectors,
            "remove_duplicate_points": remove_duplicate_points,
            "scale_to_unit_ball": scale_to_unit_ball,
            "validate_point_cloud": validate_point_cloud,
        }[name]

    # Handle predicates
    elif name in (
        "convex_hull_surface_area",
        "convex_hull_volume",
        "distance_to_convex_hull",
        "hausdorff_distance",
        "point_in_convex_hull",
    ):
        from .operations.predicates import (
            convex_hull_surface_area,
            convex_hull_volume,
            distance_to_convex_hull,
            hausdorff_distance,
            point_in_convex_hull,
        )

        return {
            "convex_hull_surface_area": convex_hull_surface_area,
            "convex_hull_volume": convex_hull_volume,
            "distance_to_convex_hull": distance_to_convex_hull,
            "hausdorff_distance": hausdorff_distance,
            "point_in_convex_hull": point_in_convex_hull,
        }[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Version and metadata
__all__ = [
    "__author__",
    "__version__",
]

# Core functions will be available through __getattr__
__all__.extend(
    [
        "ConvexHull",
        "approximate_convex_hull",
        # Advanced algorithms
        "approximate_hull_advanced",
        "batched_approximate_hull",
        # Main interface
        "convex_hull",
        "convex_hull_surface_area",
        "convex_hull_volume",
        "distance_to_convex_hull",
        "generate_direction_vectors",
        "hausdorff_distance",
        "multi_resolution_hull",
        "point_in_convex_hull",
        "progressive_hull_refinement",
        "remove_duplicate_points",
        "scale_to_unit_ball",
        "validate_point_cloud",
    ]
)


# Expose version at package level
def get_version():
    """Get PolytopAX version string."""
    return __version__


def get_info():
    """Get PolytopAX package information."""
    # Test if core functionality is available
    core_available = _lazy_import()

    info = {
        "version": __version__,
        "author": __author__,
        "core_available": core_available,
        "description": "JAX-based computational geometry library",
    }

    if core_available:
        info["available_functions"] = len(__all__) - 2  # Exclude version and author

    return info
