"""Graham Scan algorithm for 2D convex hull computation.

This module implements the Graham Scan algorithm, which is often more efficient
than QuickHull for 2D cases, especially when the number of hull vertices is small
relative to the total number of points.
"""


import jax
import jax.numpy as jnp
from jax import Array

from ..core.utils import (
    HullVertices,
    PointCloud,
    validate_point_cloud,
)


def graham_scan(
    points: PointCloud,
    tolerance: float = 1e-12
) -> tuple[HullVertices, Array]:
    """Graham Scan algorithm for 2D convex hull computation.

    The Graham Scan algorithm works by:
    1. Finding the bottommost point (or leftmost in case of tie)
    2. Sorting all other points by polar angle with respect to this point
    3. Building the hull by iteratively adding points and removing concave turns

    Args:
        points: Input point cloud with shape (..., n_points, 2)
        tolerance: Numerical tolerance for geometric predicates

    Returns:
        Tuple of (hull_vertices, hull_indices)

    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(n)

    Note:
        This is optimized for 2D only. For higher dimensions, use QuickHull.
    """
    points = validate_point_cloud(points)
    n_points, dim = points.shape[-2], points.shape[-1]

    if dim != 2:
        raise ValueError(f"Graham Scan only works with 2D points, got {dim}D")

    if n_points < 3:
        # Not enough points for a hull
        return points, jnp.arange(n_points)

    # Step 1: Find the starting point (bottommost, then leftmost)
    start_index = _find_starting_point(points)
    points[start_index]

    # Step 2: Sort points by polar angle
    sorted_indices = _sort_points_by_angle(points, start_index, tolerance)

    # Step 3: Build the convex hull
    hull_indices = _build_hull_graham(points, sorted_indices, tolerance)

    hull_vertices = points[hull_indices]

    return hull_vertices, hull_indices


def _find_starting_point(points: Array) -> int:
    """Find the starting point for Graham Scan (bottommost, then leftmost)."""
    # Find the point with minimum y-coordinate, breaking ties by x-coordinate
    min_y = jnp.min(points[:, 1])
    candidates = jnp.where(jnp.abs(points[:, 1] - min_y) < 1e-12)[0]

    if len(candidates) == 1:
        return int(candidates[0])

    # Break ties by choosing leftmost (minimum x)
    candidate_points = points[candidates]
    min_x_among_candidates = jnp.min(candidate_points[:, 0])
    leftmost_candidates = jnp.where(
        jnp.abs(candidate_points[:, 0] - min_x_among_candidates) < 1e-12
    )[0]

    return int(candidates[leftmost_candidates[0]])


def _sort_points_by_angle(
    points: Array,
    start_index: int,
    tolerance: float
) -> Array:
    """Sort points by polar angle with respect to the starting point."""
    start_point = points[start_index]
    n_points = points.shape[0]

    # Compute angles for all points except the starting point
    angles_list = []
    distances_list = []
    indices_list = []

    for i in range(n_points):
        if i == start_index:
            continue

        vector = points[i] - start_point
        angle = jnp.arctan2(vector[1], vector[0])
        distance = jnp.linalg.norm(vector)

        angles_list.append(angle)
        distances_list.append(distance)
        indices_list.append(i)

    angles = jnp.array(angles_list)
    distances = jnp.array(distances_list)
    indices = jnp.array(indices_list)

    # Sort by angle, then by distance (closer points first for same angle)
    # Use lexsort: primary key = angles, secondary key = distances
    sort_indices = jnp.lexsort((distances, angles))

    # Return indices in sorted order, with starting point first
    sorted_indices = jnp.concatenate([
        jnp.array([start_index]),
        indices[sort_indices]
    ])

    return sorted_indices


def _build_hull_graham(
    points: Array,
    sorted_indices: Array,
    tolerance: float
) -> Array:
    """Build the convex hull using the Graham Scan algorithm."""
    n_points = len(sorted_indices)

    if n_points < 3:
        return sorted_indices

    # Initialize hull with first two points
    hull = [sorted_indices[0], sorted_indices[1]]

    # Process remaining points
    for i in range(2, n_points):
        current_point_index = sorted_indices[i]
        current_point = points[current_point_index]

        # Remove points that create right turns (non-convex angles)
        while len(hull) >= 2:
            # Check if the last three points make a left turn
            p1 = points[hull[-2]]  # Second-to-last point
            p2 = points[hull[-1]]  # Last point
            p3 = current_point     # Current point

            cross_product = _cross_product_2d(p2 - p1, p3 - p1)

            if cross_product > tolerance:
                # Left turn (counterclockwise) - keep the point
                break
            else:
                # Right turn or collinear - remove the last point
                hull.pop()

        # Add the current point
        hull.append(current_point_index)

    return jnp.array(hull)


def _cross_product_2d(v1: Array, v2: Array) -> float:
    """Compute 2D cross product (determinant)."""
    return v1[0] * v2[1] - v1[1] * v2[0]


def _ccw(p1: Array, p2: Array, p3: Array, tolerance: float = 1e-12) -> int:
    """Test if three points make a counterclockwise turn.

    Returns:
        1 if counterclockwise
        -1 if clockwise
        0 if collinear
    """
    cross = _cross_product_2d(p2 - p1, p3 - p1)

    if abs(cross) < tolerance:
        return 0  # Collinear
    elif cross > 0:
        return 1  # Counterclockwise
    else:
        return -1  # Clockwise


# =============================================================================
# OPTIMIZED VARIANTS
# =============================================================================

def graham_scan_monotone(
    points: PointCloud,
    tolerance: float = 1e-12
) -> tuple[HullVertices, Array]:
    """Monotone chain variant of Graham Scan (Andrew's algorithm).

    This variant builds the upper and lower hulls separately, which can be
    more numerically stable and easier to implement correctly.

    Args:
        points: Input point cloud with shape (..., n_points, 2)
        tolerance: Numerical tolerance for geometric predicates

    Returns:
        Tuple of (hull_vertices, hull_indices)
    """
    points = validate_point_cloud(points)
    n_points, dim = points.shape[-2], points.shape[-1]

    if dim != 2:
        raise ValueError(f"Monotone Graham Scan only works with 2D points, got {dim}D")

    if n_points < 3:
        return points, jnp.arange(n_points)

    # Sort points lexicographically (first by x, then by y)
    sorted_indices = jnp.lexsort((points[:, 1], points[:, 0]))
    sorted_points = points[sorted_indices]

    # Build lower hull
    lower_hull: list[int] = []
    for i in range(n_points):
        while (len(lower_hull) >= 2 and
               _ccw(sorted_points[lower_hull[-2]],
                   sorted_points[lower_hull[-1]],
                   sorted_points[i], tolerance) <= 0):
            lower_hull.pop()
        lower_hull.append(i)

    # Build upper hull
    upper_hull: list[int] = []
    for i in range(n_points - 1, -1, -1):
        while (len(upper_hull) >= 2 and
               _ccw(sorted_points[upper_hull[-2]],
                   sorted_points[upper_hull[-1]],
                   sorted_points[i], tolerance) <= 0):
            upper_hull.pop()
        upper_hull.append(i)

    # Remove the last point of each half because it's repeated
    lower_hull.pop()
    upper_hull.pop()

    # Combine hulls
    hull_sorted_indices = lower_hull + upper_hull
    hull_indices = sorted_indices[jnp.array(hull_sorted_indices)]
    hull_vertices = points[hull_indices]

    return hull_vertices, hull_indices


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_graham_quickhull(
    points: PointCloud,
    tolerance: float = 1e-12
) -> dict:
    """Compare Graham Scan and QuickHull results for verification."""
    from .exact import quickhull

    # Run both algorithms
    graham_vertices, graham_indices = graham_scan(points, tolerance)
    quickhull_vertices, quickhull_indices = quickhull(points, tolerance)

    # Compare results by converting JAX arrays to numpy for hashing
    graham_set = {tuple(float(x) for x in v) for v in graham_vertices}
    quickhull_set = {tuple(float(x) for x in v) for v in quickhull_vertices}

    return {
        "graham_vertex_count": len(graham_vertices),
        "quickhull_vertex_count": len(quickhull_vertices),
        "vertices_match": graham_set == quickhull_set,
        "symmetric_difference": graham_set.symmetric_difference(quickhull_set),
        "graham_indices": graham_indices,
        "quickhull_indices": quickhull_indices
    }


# =============================================================================
# JIT-COMPILED VERSIONS
# =============================================================================

graham_scan_jit = jax.jit(graham_scan)
graham_scan_monotone_jit = jax.jit(graham_scan_monotone)
_ccw_jit = jax.jit(_ccw)
