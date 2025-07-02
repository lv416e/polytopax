"""3D exact convex hull algorithms for PolytopAX.

This module implements 3D-specific exact convex hull algorithms optimized
for JAX compatibility and numerical accuracy.
"""

import warnings
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array

from ..core.utils import (
    HullVertices,
    PointCloud,
    validate_point_cloud,
)


def quickhull_3d(
    points: PointCloud,
    tolerance: float = 1e-12,
    max_iterations: int = 1000
) -> Tuple[HullVertices, Array]:
    """3D QuickHull algorithm for exact convex hull computation.
    
    This implements the 3D QuickHull algorithm which builds the convex hull
    by iteratively finding extreme points and constructing the hull faces.
    
    Args:
        points: Input point cloud with shape (..., n_points, 3)
        tolerance: Numerical tolerance for geometric predicates
        max_iterations: Maximum iterations to prevent infinite loops
        
    Returns:
        Tuple of (hull_vertices, hull_indices)
        
    Algorithm:
        1. Find initial tetrahedron from extreme points
        2. For each face of the tetrahedron:
           - Find points outside the face
           - Find the point with maximum distance from the face
           - Create new faces connecting this point to visible face edges
           - Recursively process new faces
    """
    points = validate_point_cloud(points)
    n_points, dim = points.shape[-2], points.shape[-1]
    
    if dim != 3:
        raise ValueError(f"quickhull_3d only works with 3D points, got {dim}D")
    
    if n_points < 4:
        # Not enough points for a 3D hull
        return points, jnp.arange(n_points)
    
    # Find initial tetrahedron
    tetrahedron_indices = _find_initial_tetrahedron_3d(points, tolerance)
    
    if len(tetrahedron_indices) < 4:
        # Points are coplanar or collinear
        return _handle_degenerate_3d_case(points, tetrahedron_indices, tolerance)
    
    # Initialize hull faces from the tetrahedron
    initial_faces = [
        [tetrahedron_indices[0], tetrahedron_indices[1], tetrahedron_indices[2]],
        [tetrahedron_indices[0], tetrahedron_indices[1], tetrahedron_indices[3]],
        [tetrahedron_indices[0], tetrahedron_indices[2], tetrahedron_indices[3]],
        [tetrahedron_indices[1], tetrahedron_indices[2], tetrahedron_indices[3]]
    ]
    
    # Build complete hull by processing remaining points
    hull_faces = _build_3d_hull_recursive(
        points, initial_faces, tetrahedron_indices, tolerance, max_iterations
    )
    
    # Extract unique vertices from faces
    hull_vertices, hull_indices = _extract_vertices_from_faces(points, hull_faces)
    
    return hull_vertices, hull_indices


def _find_initial_tetrahedron_3d(
    points: Array,
    tolerance: float
) -> List[int]:
    """Find four points that form a non-degenerate tetrahedron."""
    n_points = points.shape[0]
    
    # Find the first two points with maximum distance
    max_distance = 0.0
    best_pair = [0, 1]
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            distance = jnp.linalg.norm(points[i] - points[j])
            if distance > max_distance:
                max_distance = distance
                best_pair = [i, j]
    
    if max_distance < tolerance:
        # All points are essentially at the same location
        return [0]
    
    # Find the third point with maximum distance from the line formed by the first two
    point1, point2 = points[best_pair[0]], points[best_pair[1]]
    line_vector = point2 - point1
    max_distance_from_line = 0.0
    third_point = 0
    
    for i in range(n_points):
        if i in best_pair:
            continue
        
        # Distance from point to line
        point_vector = points[i] - point1
        cross_product = jnp.cross(line_vector, point_vector)
        distance = jnp.linalg.norm(cross_product) / jnp.linalg.norm(line_vector)
        
        if distance > max_distance_from_line:
            max_distance_from_line = distance
            third_point = i
    
    if max_distance_from_line < tolerance:
        # All points are collinear
        return best_pair
    
    # Find the fourth point with maximum distance from the plane formed by the first three
    p1, p2, p3 = points[best_pair[0]], points[best_pair[1]], points[third_point]
    plane_normal = jnp.cross(p2 - p1, p3 - p1)
    plane_normal_length = jnp.linalg.norm(plane_normal)
    
    if plane_normal_length < tolerance:
        # First three points are collinear (shouldn't happen if we got here)
        return best_pair + [third_point]
    
    plane_normal = plane_normal / plane_normal_length
    max_distance_from_plane = 0.0
    fourth_point = 0
    
    for i in range(n_points):
        if i in best_pair or i == third_point:
            continue
        
        # Distance from point to plane
        point_vector = points[i] - p1
        distance = abs(jnp.dot(point_vector, plane_normal))
        
        if distance > max_distance_from_plane:
            max_distance_from_plane = distance
            fourth_point = i
    
    if max_distance_from_plane < tolerance:
        # All points are coplanar
        return best_pair + [third_point]
    
    return best_pair + [third_point, fourth_point]


def _handle_degenerate_3d_case(
    points: Array,
    extreme_indices: List[int],
    tolerance: float
) -> Tuple[Array, Array]:
    """Handle degenerate cases where points are coplanar or collinear."""
    if len(extreme_indices) <= 2:
        # Collinear case - fall back to 2D QuickHull projected onto appropriate plane
        from .exact import _quickhull_2d
        
        # Project points onto the line and treat as 1D problem
        if len(extreme_indices) == 2:
            return points[jnp.array(extreme_indices)], jnp.array(extreme_indices)
        else:
            return points[jnp.array([extreme_indices[0]])], jnp.array([extreme_indices[0]])
    
    elif len(extreme_indices) == 3:
        # Coplanar case - project to 2D and solve
        p1, p2, p3 = points[extreme_indices[0]], points[extreme_indices[1]], points[extreme_indices[2]]
        
        # Create 2D coordinate system in the plane
        u = p2 - p1
        u = u / jnp.linalg.norm(u)
        
        temp_v = p3 - p1
        v = temp_v - jnp.dot(temp_v, u) * u
        v_norm = jnp.linalg.norm(v)
        
        if v_norm < tolerance:
            # Points are actually collinear
            return points[jnp.array(extreme_indices[:2])], jnp.array(extreme_indices[:2])
        
        v = v / v_norm
        
        # Project all points to 2D
        points_2d = []
        for point in points:
            rel_point = point - p1
            x = jnp.dot(rel_point, u)
            y = jnp.dot(rel_point, v)
            points_2d.append([x, y])
        
        points_2d = jnp.array(points_2d)
        
        # Solve 2D convex hull
        from .exact import _quickhull_2d
        hull_2d, hull_indices_2d = _quickhull_2d(points_2d, tolerance)
        
        # Map back to 3D
        hull_vertices_3d = points[hull_indices_2d]
        
        return hull_vertices_3d, hull_indices_2d
    
    else:
        # Should not reach here
        return points[jnp.array(extreme_indices)], jnp.array(extreme_indices)


def _build_3d_hull_recursive(
    points: Array,
    initial_faces: List[List[int]],
    processed_points: List[int],
    tolerance: float,
    max_iterations: int
) -> List[List[int]]:
    """Build the complete 3D hull by recursively processing faces."""
    # This is a simplified implementation
    # A full 3D QuickHull would require more complex face management
    
    warnings.warn(
        "Full 3D QuickHull implementation is still in development. "
        "Using simplified approach.",
        UserWarning, stacklevel=2
    )
    
    # For now, return the initial tetrahedron faces
    # TODO: Implement full recursive face processing
    return initial_faces


def _extract_vertices_from_faces(
    points: Array,
    faces: List[List[int]]
) -> Tuple[Array, Array]:
    """Extract unique vertices from a list of faces."""
    unique_indices = set()
    for face in faces:
        unique_indices.update(face)
    
    unique_indices = sorted(list(unique_indices))
    hull_indices = jnp.array(unique_indices)
    hull_vertices = points[hull_indices]
    
    return hull_vertices, hull_indices


# =============================================================================
# 3D GEOMETRIC PREDICATES
# =============================================================================

def orientation_3d(
    p1: Array,
    p2: Array,
    p3: Array,
    p4: Array,
    tolerance: float = 1e-12
) -> int:
    """Determine orientation of four points in 3D.
    
    Args:
        p1, p2, p3, p4: Points in 3D
        tolerance: Numerical tolerance for coplanarity detection
        
    Returns:
        1 if p4 is above the plane defined by p1, p2, p3 (counterclockwise orientation)
        -1 if p4 is below the plane (clockwise orientation)
        0 if coplanar
        
    Note:
        This uses the signed volume of the tetrahedron formed by the four points.
        The sign convention follows the right-hand rule: if p1, p2, p3 form a
        counterclockwise triangle when viewed from p4, the result is positive.
    """
    # Use triple scalar product: (p2-p1) · ((p3-p1) × (p4-p1))
    # This is equivalent to the determinant but more numerically stable
    v1 = p2 - p1
    v2 = p3 - p1
    v3 = p4 - p1
    
    # Compute cross product v2 × v3
    cross = jnp.cross(v2, v3)
    
    # Dot product with v1
    det = jnp.dot(v1, cross)
    
    if abs(det) < tolerance:
        return 0  # Coplanar
    elif det > 0:
        return 1  # Positive orientation
    else:
        return -1  # Negative orientation


def point_to_plane_distance_3d(
    point: Array,
    plane_point: Array,
    plane_normal: Array
) -> float:
    """Compute signed distance from point to plane in 3D."""
    return jnp.dot(point - plane_point, plane_normal) / jnp.linalg.norm(plane_normal)


def is_point_inside_tetrahedron_3d(
    point: Array,
    tetrahedron: Array,
    tolerance: float = 1e-12
) -> bool:
    """Test if point is inside tetrahedron using orientation tests.
    
    A point is inside a tetrahedron if it lies on the same side of all four
    faces as the interior of the tetrahedron.
    """
    if tetrahedron.shape[0] != 4:
        raise ValueError("Tetrahedron must have exactly 4 vertices")
    
    v0, v1, v2, v3 = tetrahedron[0], tetrahedron[1], tetrahedron[2], tetrahedron[3]
    
    # For each face, check if the point is on the same side as the opposite vertex
    # Face 0: (v1, v2, v3), opposite vertex is v0
    face0_orientation = orientation_3d(v1, v2, v3, point, tolerance)
    face0_reference = orientation_3d(v1, v2, v3, v0, tolerance)
    
    # Face 1: (v0, v2, v3), opposite vertex is v1  
    face1_orientation = orientation_3d(v0, v2, v3, point, tolerance)
    face1_reference = orientation_3d(v0, v2, v3, v1, tolerance)
    
    # Face 2: (v0, v1, v3), opposite vertex is v2
    face2_orientation = orientation_3d(v0, v1, v3, point, tolerance)
    face2_reference = orientation_3d(v0, v1, v3, v2, tolerance)
    
    # Face 3: (v0, v1, v2), opposite vertex is v3
    face3_orientation = orientation_3d(v0, v1, v2, point, tolerance)
    face3_reference = orientation_3d(v0, v1, v2, v3, tolerance)
    
    # Point is inside if it has the same orientation as the reference for all faces
    # Allow for points on the boundary (orientation == 0)
    conditions = [
        face0_orientation * face0_reference >= 0,
        face1_orientation * face1_reference >= 0,
        face2_orientation * face2_reference >= 0,
        face3_orientation * face3_reference >= 0
    ]
    
    return all(conditions)


# =============================================================================
# JIT-COMPILED VERSIONS
# =============================================================================

quickhull_3d_jit = jax.jit(quickhull_3d, static_argnames=['max_iterations'])
orientation_3d_jit = jax.jit(orientation_3d)
point_to_plane_distance_3d_jit = jax.jit(point_to_plane_distance_3d)
is_point_inside_tetrahedron_3d_jit = jax.jit(is_point_inside_tetrahedron_3d)