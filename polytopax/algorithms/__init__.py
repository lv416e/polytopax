"""PolytopAX algorithms module."""

# Lazy import to avoid circular dependencies
def _get_approximation_functions():
    from .approximation import (
        approximate_convex_hull,
        batched_approximate_hull,
        multi_resolution_hull,
        progressive_hull_refinement,
    )
    return approximate_convex_hull, batched_approximate_hull, multi_resolution_hull, progressive_hull_refinement

# Expose functions through module-level getattr
def __getattr__(name):
    if name in ("approximate_convex_hull", "batched_approximate_hull", "multi_resolution_hull", "progressive_hull_refinement"):
        approximate_convex_hull, batched_approximate_hull, multi_resolution_hull, progressive_hull_refinement = _get_approximation_functions()
        return {
            "approximate_convex_hull": approximate_convex_hull,
            "batched_approximate_hull": batched_approximate_hull,
            "multi_resolution_hull": multi_resolution_hull,
            "progressive_hull_refinement": progressive_hull_refinement,
        }[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "approximate_convex_hull",
    "batched_approximate_hull",
    "multi_resolution_hull",
    "progressive_hull_refinement"
]
