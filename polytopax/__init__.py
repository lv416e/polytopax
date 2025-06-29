"""PolytopAX: JAX-based computational geometry library.

A high-performance convex hull computation and polytope manipulation library
built on the JAX ecosystem.
"""

__version__ = "0.0.1"
__author__ = "PolytopAX Development Team"

# Core imports
try:
    from .core.hull import convex_hull, approximate_convex_hull

    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False

__all__ = [
    "__version__",
]

# Add core functions to __all__ if available
if _CORE_AVAILABLE:
    __all__.extend(
        [
            "convex_hull",
            "approximate_convex_hull",
        ]
    )
