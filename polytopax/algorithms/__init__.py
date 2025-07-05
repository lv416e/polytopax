"""PolytopAX algorithms module."""


# Lazy import to avoid circular dependencies
def _get_approximation_functions():
    from .approximation import (
        approximate_convex_hull,
        batched_approximate_hull,
        improved_approximate_convex_hull,
        multi_resolution_hull,
        progressive_hull_refinement,
    )

    return (
        approximate_convex_hull,
        batched_approximate_hull,
        multi_resolution_hull,
        progressive_hull_refinement,
        improved_approximate_convex_hull,
    )


def _get_exact_functions():
    from .exact import (
        is_point_inside_triangle_2d,
        orientation_2d,
        point_to_line_distance_2d,
        quickhull,
    )
    from .exact_3d import (
        is_point_inside_tetrahedron_3d,
        orientation_3d,
        point_to_plane_distance_3d,
        quickhull_3d,
    )
    from .exact_nd import (
        is_point_inside_simplex_nd,
        orientation_nd,
        point_to_hyperplane_distance_nd,
        quickhull_nd,
    )
    from .graham_scan import (
        compare_graham_quickhull,
        graham_scan,
        graham_scan_monotone,
    )

    return (
        quickhull,
        orientation_2d,
        point_to_line_distance_2d,
        is_point_inside_triangle_2d,
        quickhull_3d,
        orientation_3d,
        point_to_plane_distance_3d,
        is_point_inside_tetrahedron_3d,
        quickhull_nd,
        orientation_nd,
        point_to_hyperplane_distance_nd,
        is_point_inside_simplex_nd,
        graham_scan,
        graham_scan_monotone,
        compare_graham_quickhull,
    )


# Expose functions through module-level getattr
def __getattr__(name):
    approximation_functions = (
        "approximate_convex_hull",
        "batched_approximate_hull",
        "multi_resolution_hull",
        "progressive_hull_refinement",
        "improved_approximate_convex_hull",
    )
    exact_functions = (
        "quickhull",
        "orientation_2d",
        "point_to_line_distance_2d",
        "is_point_inside_triangle_2d",
        "quickhull_3d",
        "orientation_3d",
        "point_to_plane_distance_3d",
        "is_point_inside_tetrahedron_3d",
        "quickhull_nd",
        "orientation_nd",
        "point_to_hyperplane_distance_nd",
        "is_point_inside_simplex_nd",
        "graham_scan",
        "graham_scan_monotone",
        "compare_graham_quickhull",
    )

    if name in approximation_functions:
        (
            approximate_convex_hull,
            batched_approximate_hull,
            multi_resolution_hull,
            progressive_hull_refinement,
            improved_approximate_convex_hull,
        ) = _get_approximation_functions()
        return {
            "approximate_convex_hull": approximate_convex_hull,
            "batched_approximate_hull": batched_approximate_hull,
            "multi_resolution_hull": multi_resolution_hull,
            "progressive_hull_refinement": progressive_hull_refinement,
            "improved_approximate_convex_hull": improved_approximate_convex_hull,
        }[name]
    elif name in exact_functions:
        (
            quickhull,
            orientation_2d,
            point_to_line_distance_2d,
            is_point_inside_triangle_2d,
            quickhull_3d,
            orientation_3d,
            point_to_plane_distance_3d,
            is_point_inside_tetrahedron_3d,
            quickhull_nd,
            orientation_nd,
            point_to_hyperplane_distance_nd,
            is_point_inside_simplex_nd,
            graham_scan,
            graham_scan_monotone,
            compare_graham_quickhull,
        ) = _get_exact_functions()
        return {
            "quickhull": quickhull,
            "orientation_2d": orientation_2d,
            "point_to_line_distance_2d": point_to_line_distance_2d,
            "is_point_inside_triangle_2d": is_point_inside_triangle_2d,
            "quickhull_3d": quickhull_3d,
            "orientation_3d": orientation_3d,
            "point_to_plane_distance_3d": point_to_plane_distance_3d,
            "is_point_inside_tetrahedron_3d": is_point_inside_tetrahedron_3d,
            "quickhull_nd": quickhull_nd,
            "orientation_nd": orientation_nd,
            "point_to_hyperplane_distance_nd": point_to_hyperplane_distance_nd,
            "is_point_inside_simplex_nd": is_point_inside_simplex_nd,
            "graham_scan": graham_scan,
            "graham_scan_monotone": graham_scan_monotone,
            "compare_graham_quickhull": compare_graham_quickhull,
        }[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Approximation algorithms (Phase 1 & 2)
    "approximate_convex_hull",
    "batched_approximate_hull",
    "compare_graham_quickhull",
    "graham_scan",
    "graham_scan_monotone",
    "improved_approximate_convex_hull",
    "is_point_inside_simplex_nd",
    "is_point_inside_tetrahedron_3d",
    "is_point_inside_triangle_2d",
    "multi_resolution_hull",
    "orientation_2d",
    "orientation_3d",
    "orientation_nd",
    "point_to_hyperplane_distance_nd",
    "point_to_line_distance_2d",
    "point_to_plane_distance_3d",
    "progressive_hull_refinement",
    # Exact algorithms (Phase 3)
    "quickhull",
    "quickhull_3d",
    # N-dimensional algorithms (Phase 4)
    "quickhull_nd",
]
