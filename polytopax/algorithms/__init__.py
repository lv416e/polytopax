"""PolytopAX algorithms module."""

# Lazy import to avoid circular dependencies
def _get_approximation_functions():
    from .approximation import (
        approximate_convex_hull,
        batched_approximate_hull,
        multi_resolution_hull,
        progressive_hull_refinement,
        improved_approximate_convex_hull,
    )
    return approximate_convex_hull, batched_approximate_hull, multi_resolution_hull, progressive_hull_refinement, improved_approximate_convex_hull

def _get_exact_functions():
    from .exact import (
        quickhull,
        orientation_2d,
        point_to_line_distance_2d,
        is_point_inside_triangle_2d,
    )
    from .exact_3d import (
        quickhull_3d,
        orientation_3d,
        point_to_plane_distance_3d,
        is_point_inside_tetrahedron_3d,
    )
    from .graham_scan import (
        graham_scan,
        graham_scan_monotone,
        compare_graham_quickhull,
    )
    return (quickhull, orientation_2d, point_to_line_distance_2d, is_point_inside_triangle_2d,
            quickhull_3d, orientation_3d, point_to_plane_distance_3d, is_point_inside_tetrahedron_3d,
            graham_scan, graham_scan_monotone, compare_graham_quickhull)

# Expose functions through module-level getattr
def __getattr__(name):
    approximation_functions = ("approximate_convex_hull", "batched_approximate_hull", "multi_resolution_hull", "progressive_hull_refinement", "improved_approximate_convex_hull")
    exact_functions = ("quickhull", "orientation_2d", "point_to_line_distance_2d", "is_point_inside_triangle_2d",
                       "quickhull_3d", "orientation_3d", "point_to_plane_distance_3d", "is_point_inside_tetrahedron_3d",
                       "graham_scan", "graham_scan_monotone", "compare_graham_quickhull")
    
    if name in approximation_functions:
        approximate_convex_hull, batched_approximate_hull, multi_resolution_hull, progressive_hull_refinement, improved_approximate_convex_hull = _get_approximation_functions()
        return {
            "approximate_convex_hull": approximate_convex_hull,
            "batched_approximate_hull": batched_approximate_hull,
            "multi_resolution_hull": multi_resolution_hull,
            "progressive_hull_refinement": progressive_hull_refinement,
            "improved_approximate_convex_hull": improved_approximate_convex_hull,
        }[name]
    elif name in exact_functions:
        (quickhull, orientation_2d, point_to_line_distance_2d, is_point_inside_triangle_2d,
         quickhull_3d, orientation_3d, point_to_plane_distance_3d, is_point_inside_tetrahedron_3d,
         graham_scan, graham_scan_monotone, compare_graham_quickhull) = _get_exact_functions()
        return {
            "quickhull": quickhull,
            "orientation_2d": orientation_2d,
            "point_to_line_distance_2d": point_to_line_distance_2d,
            "is_point_inside_triangle_2d": is_point_inside_triangle_2d,
            "quickhull_3d": quickhull_3d,
            "orientation_3d": orientation_3d,
            "point_to_plane_distance_3d": point_to_plane_distance_3d,
            "is_point_inside_tetrahedron_3d": is_point_inside_tetrahedron_3d,
            "graham_scan": graham_scan,
            "graham_scan_monotone": graham_scan_monotone,
            "compare_graham_quickhull": compare_graham_quickhull,
        }[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Approximation algorithms (Phase 1 & 2)
    "approximate_convex_hull",
    "batched_approximate_hull", 
    "multi_resolution_hull",
    "progressive_hull_refinement",
    "improved_approximate_convex_hull",
    # Exact algorithms (Phase 3)
    "quickhull",
    "orientation_2d",
    "point_to_line_distance_2d",
    "is_point_inside_triangle_2d",
    "quickhull_3d",
    "orientation_3d",
    "point_to_plane_distance_3d",
    "is_point_inside_tetrahedron_3d",
    "graham_scan",
    "graham_scan_monotone",
    "compare_graham_quickhull",
]
