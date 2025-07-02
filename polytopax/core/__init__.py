"""Core PolytopAX functionality."""

from .hull import convex_hull, approximate_convex_hull
from .polytope import ConvexHull
from .utils import (
    validate_point_cloud,
    generate_direction_vectors,
    remove_duplicate_points,
    scale_to_unit_ball
)

__all__ = [
    "convex_hull",
    "approximate_convex_hull", 
    "ConvexHull",
    "validate_point_cloud",
    "generate_direction_vectors",
    "remove_duplicate_points", 
    "scale_to_unit_ball"
]
