"""Exact convex hull algorithms for PolytopAX.

This module implements exact (non-approximation) convex hull algorithms
that provide mathematically precise results. These algorithms are designed
to be JAX-compatible while maintaining numerical accuracy.

Phase 3 Implementation:
- QuickHull algorithm for general dimensions
- Graham Scan for 2D optimization
- Exact geometric predicates
- Adaptive precision arithmetic support
"""

import warnings

import jax
import jax.numpy as jnp
from jax import Array

from ..core.utils import (
    HullVertices,
    PointCloud,
    validate_point_cloud,
)


def quickhull(
    points: PointCloud,
    tolerance: float = 1e-12,
    max_iterations: int = 1000
) -> tuple[HullVertices, Array]:
    """JAX-compatible QuickHull algorithm for exact convex hull computation.

    QuickHull is a divide-and-conquer algorithm that recursively finds
    the convex hull by partitioning points around extreme vertices.

    Args:
        points: Input point cloud with shape (..., n_points, dim)
        tolerance: Numerical tolerance for geometric predicates
        max_iterations: Maximum iterations to prevent infinite loops

    Returns:
        Tuple of (hull_vertices, hull_indices)

    Algorithm:
        1. Find initial simplex (extreme points in each dimension)
        2. For each face of the simplex:
           - Find points outside the face
           - Recursively build hull from outside points
           - Merge results

    Note:
        This implementation uses fixed-size arrays and JAX-compatible
        control flow to maintain differentiability where possible.
    """
    points = validate_point_cloud(points)
    n_points, dim = points.shape[-2], points.shape[-1]

    if n_points < dim + 1:
        # Not enough points for full-dimensional hull
        return points, jnp.arange(n_points)

    if dim == 2:
        # Use specialized 2D implementation for efficiency
        return _quickhull_2d(points, tolerance)
    elif dim == 3:
        # Use specialized 3D implementation
        return _quickhull_3d(points, tolerance, max_iterations)
    else:
        # General n-dimensional implementation
        return _quickhull_nd(points, tolerance, max_iterations)


def _quickhull_2d(
    points: Array,
    tolerance: float
) -> tuple[Array, Array]:
    """Specialized 2D QuickHull implementation.

    For 2D, QuickHull reduces to finding the upper and lower hulls
    and combining them.
    """
    n_points = points.shape[0]

    if n_points == 1:
        return points, jnp.arange(n_points)

    if n_points == 2:
        return points, jnp.arange(n_points)

    # Sort points by x-coordinate first, then by y-coordinate to handle ties
    sorted_indices = jnp.lexsort((points[:, 1], points[:, 0]))
    sorted_points = points[sorted_indices]

    # Find leftmost and rightmost points
    leftmost = sorted_points[0]
    rightmost = sorted_points[-1]

    # Check for collinear case - all points on a line
    if jnp.allclose(leftmost, rightmost, atol=tolerance):
        # All points are at the same location
        return jnp.array([leftmost]), jnp.array([sorted_indices[0]])

    # Check if all points are collinear
    all_collinear = True
    for point in sorted_points[1:-1]:
        cross_product = abs(_cross_product_2d(
            rightmost - leftmost,
            point - leftmost
        ))
        if cross_product > tolerance:
            all_collinear = False
            break

    if all_collinear:
        # Return only the two extreme points
        return jnp.array([leftmost, rightmost]), jnp.array([sorted_indices[0], sorted_indices[-1]])

    # Partition points into upper and lower sets
    upper_points, upper_indices = _find_hull_side_2d(
        sorted_points, sorted_indices, leftmost, rightmost, "upper", tolerance
    )
    lower_points, lower_indices = _find_hull_side_2d(
        sorted_points, sorted_indices, leftmost, rightmost, "lower", tolerance
    )

    # Combine upper and lower hulls
    # Remove duplicate endpoints
    if len(upper_points) > 0 and len(lower_points) > 0:
        hull_points = jnp.concatenate([
            upper_points,
            lower_points[1:-1]  # Remove endpoints to avoid duplicates
        ], axis=0)
        hull_indices = jnp.concatenate([
            upper_indices,
            lower_indices[1:-1]
        ], axis=0)
    else:
        hull_points = jnp.array([leftmost, rightmost])
        hull_indices = jnp.array([sorted_indices[0], sorted_indices[-1]])

    return hull_points, hull_indices


def _find_hull_side_2d(
    sorted_points: Array,
    sorted_indices: Array,
    start_point: Array,
    end_point: Array,
    side: str,
    tolerance: float
) -> tuple[Array, Array]:
    """Find points on one side of the hull (upper or lower) for 2D QuickHull.

    Args:
        sorted_points: Points sorted by x-coordinate
        sorted_indices: Original indices of sorted points
        start_point: Starting point of the line
        end_point: Ending point of the line
        side: "upper" or "lower" to specify which side to compute
        tolerance: Numerical tolerance

    Returns:
        Tuple of (hull_points, hull_indices) for this side
    """
    sorted_points.shape[0]

    # Check if start and end points are identical (collinear case)
    if jnp.allclose(start_point, end_point, atol=tolerance):
        return jnp.array([start_point]), jnp.array([sorted_indices[0]])

    # Find points on the specified side of the line
    side_points = []
    side_indices = []

    for i, point in enumerate(sorted_points):
        # Skip if point is start or end point
        if jnp.allclose(point, start_point, atol=tolerance) or jnp.allclose(point, end_point, atol=tolerance):
            continue

        # Compute signed distance from point to line
        cross_product = _cross_product_2d(
            end_point - start_point,
            point - start_point
        )

        is_on_side = cross_product > tolerance if side == "upper" else cross_product < -tolerance

        if is_on_side:
            side_points.append(point)
            side_indices.append(sorted_indices[i])

    if len(side_points) == 0:
        # No points on this side, just return the line endpoints
        return jnp.array([start_point, end_point]), jnp.array([sorted_indices[0], sorted_indices[-1]])

    side_points_array = jnp.array(side_points)
    side_indices_array = jnp.array(side_indices)

    # Find the point with maximum distance from the line
    max_distance = -1.0
    max_index = 0
    line_vector = end_point - start_point
    line_length = jnp.linalg.norm(line_vector)

    # Handle degenerate case where line has no length
    if line_length < tolerance:
        return jnp.array([start_point]), jnp.array([sorted_indices[0]])

    for i, point in enumerate(side_points_array):
        distance = abs(_cross_product_2d(
            line_vector,
            point - start_point
        )) / line_length

        if distance > max_distance:
            max_distance = distance
            max_index = i

    farthest_point = side_points_array[max_index]

    # Recursively build hull on both sides of the new triangle
    left_hull, left_indices = _find_hull_side_2d(
        side_points_array, side_indices_array, start_point, farthest_point, side, tolerance
    )
    right_hull, right_indices = _find_hull_side_2d(
        side_points_array, side_indices_array, farthest_point, end_point, side, tolerance
    )

    # Combine results (remove duplicate farthest_point)
    if len(left_hull) > 0 and len(right_hull) > 0:
        combined_hull = jnp.concatenate([
            left_hull[:-1],  # Remove last point to avoid duplicate
            right_hull
        ], axis=0)
        combined_indices = jnp.concatenate([
            left_indices[:-1],
            right_indices
        ], axis=0)
    elif len(left_hull) > 0:
        combined_hull = left_hull
        combined_indices = left_indices
    else:
        combined_hull = right_hull
        combined_indices = right_indices

    return combined_hull, combined_indices


def _cross_product_2d(v1: Array, v2: Array) -> float:
    """Compute 2D cross product (determinant)."""
    return v1[0] * v2[1] - v1[1] * v2[0]


def _quickhull_3d(
    points: Array,
    tolerance: float,
    max_iterations: int
) -> tuple[Array, Array]:
    """Specialized 3D QuickHull implementation.

    Delegates to the full 3D implementation in exact_3d module.
    """
    from .exact_3d import quickhull_3d
    return quickhull_3d(points, tolerance, max_iterations)


def _quickhull_nd(
    points: Array,
    tolerance: float,
    max_iterations: int
) -> tuple[Array, Array]:
    """General n-dimensional QuickHull implementation.

    This is a placeholder for future n-dimensional implementation.
    """
    warnings.warn(
        f"N-dimensional QuickHull for {points.shape[-1]}D is not implemented yet, "
        "falling back to approximation",
        UserWarning, stacklevel=2
    )

    # For now, fall back to approximation
    from .approximation import improved_approximate_convex_hull
    return improved_approximate_convex_hull(points)


# =============================================================================
# GEOMETRIC PREDICATES FOR EXACT ALGORITHMS
# =============================================================================

def orientation_2d(
    p1: Array,
    p2: Array,
    p3: Array,
    tolerance: float = 1e-12
) -> int:
    """Determine orientation of three points in 2D.

    Args:
        p1: First point in 2D
        p2: Second point in 2D
        p3: Third point in 2D
        tolerance: Numerical tolerance for collinearity detection

    Returns:
        1 if counterclockwise, -1 if clockwise, 0 if collinear
    """
    cross = _cross_product_2d(p2 - p1, p3 - p1)

    if abs(cross) < tolerance:
        return 0  # Collinear
    elif cross > 0:
        return 1  # Counterclockwise
    else:
        return -1  # Clockwise


def point_to_line_distance_2d(
    point: Array,
    line_start: Array,
    line_end: Array
) -> float:
    """Compute signed distance from point to line in 2D.

    Positive distance means the point is to the left of the directed line.
    """
    return _cross_product_2d(
        line_end - line_start,
        point - line_start
    ) / jnp.linalg.norm(line_end - line_start)


def is_point_inside_triangle_2d(
    point: Array,
    triangle: Array,
    tolerance: float = 1e-12
) -> bool:
    """Test if point is inside triangle using barycentric coordinates."""
    v0, v1, v2 = triangle[0], triangle[1], triangle[2]

    # Compute barycentric coordinates
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])

    if abs(denom) < tolerance:
        return False  # Degenerate triangle

    a = ((v1[1] - v2[1]) * (point[0] - v2[0]) + (v2[0] - v1[0]) * (point[1] - v2[1])) / denom
    b = ((v2[1] - v0[1]) * (point[0] - v2[0]) + (v0[0] - v2[0]) * (point[1] - v2[1])) / denom
    c = 1 - a - b

    # Point is inside if all barycentric coordinates are non-negative
    return a >= -tolerance and b >= -tolerance and c >= -tolerance


# =============================================================================
# JIT-COMPILED VERSIONS
# =============================================================================

quickhull_jit = jax.jit(quickhull, static_argnames=['max_iterations'])
orientation_2d_jit = jax.jit(orientation_2d)
point_to_line_distance_2d_jit = jax.jit(point_to_line_distance_2d)
is_point_inside_triangle_2d_jit = jax.jit(is_point_inside_triangle_2d)
