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

# Core imports
try:
    # Main convex hull functions
    from .core.hull import (
        convex_hull,
        approximate_convex_hull,
        point_in_hull,
        hull_volume,
        hull_surface_area,
        distance_to_hull
    )
    
    # ConvexHull class
    from .core.polytope import ConvexHull
    
    # Geometric predicates
    from .operations.predicates import (
        point_in_convex_hull,
        convex_hull_volume,
        convex_hull_surface_area,
        distance_to_convex_hull,
        hausdorff_distance
    )
    
    # Approximation algorithms
    from .algorithms.approximation import (
        approximate_convex_hull as approximate_hull_advanced,
        batched_approximate_hull,
        multi_resolution_hull,
        progressive_hull_refinement
    )
    
    # Utility functions
    from .core.utils import (
        validate_point_cloud,
        generate_direction_vectors,
        remove_duplicate_points,
        scale_to_unit_ball
    )

    _CORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Core PolytopAX functionality not available: {e}")
    _CORE_AVAILABLE = False

# Version and metadata
__all__ = [
    "__version__",
    "__author__",
]

# Add core functions to __all__ if available
if _CORE_AVAILABLE:
    __all__.extend([
        # Main interface
        "convex_hull",
        "approximate_convex_hull", 
        "ConvexHull",
        
        # Geometric predicates
        "point_in_hull",
        "hull_volume", 
        "hull_surface_area",
        "distance_to_hull",
        "point_in_convex_hull",
        "convex_hull_volume",
        "convex_hull_surface_area", 
        "distance_to_convex_hull",
        "hausdorff_distance",
        
        # Advanced algorithms
        "approximate_hull_advanced",
        "batched_approximate_hull",
        "multi_resolution_hull",
        "progressive_hull_refinement",
        
        # Utilities
        "validate_point_cloud",
        "generate_direction_vectors", 
        "remove_duplicate_points",
        "scale_to_unit_ball"
    ])

# Expose version at package level  
def get_version():
    """Get PolytopAX version string."""
    return __version__

def get_info():
    """Get PolytopAX package information."""
    info = {
        "version": __version__,
        "author": __author__,
        "core_available": _CORE_AVAILABLE,
        "description": "JAX-based computational geometry library"
    }
    
    if _CORE_AVAILABLE:
        info["available_functions"] = len(__all__) - 2  # Exclude version and author
    
    return info
