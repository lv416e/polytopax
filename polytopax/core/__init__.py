"""Core PolytopAX functionality."""

from .polytope import ConvexHull
from .utils import (
    generate_direction_vectors,
    remove_duplicate_points,
    scale_to_unit_ball,
    validate_point_cloud,
)


# Lazy import to avoid circular dependencies
def _get_hull_functions():
    from .hull import approximate_convex_hull, convex_hull

    return approximate_convex_hull, convex_hull


# Expose hull functions through module-level getattr
def __getattr__(name):
    if name in ("approximate_convex_hull", "convex_hull"):
        approximate_convex_hull, convex_hull = _get_hull_functions()
        if name == "approximate_convex_hull":
            return approximate_convex_hull
        elif name == "convex_hull":
            return convex_hull
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "ConvexHull",
    "approximate_convex_hull",
    "convex_hull",
    "generate_direction_vectors",
    "remove_duplicate_points",
    "scale_to_unit_ball",
    "validate_point_cloud",
]
