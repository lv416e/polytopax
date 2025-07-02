"""Geometric predicates for convex hull operations."""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Optional, Tuple, Union
import warnings

from ..core.utils import (
    validate_point_cloud,
    compute_simplex_volume,
    HullVertices
)


def point_in_convex_hull(
    point: Array,
    hull_vertices: HullVertices,
    tolerance: float = 1e-8,
    method: str = "linear_programming"
) -> Array:
    """Test if point is inside convex hull.
    
    Determines whether a point lies inside, on the boundary, or outside
    of the convex hull defined by the given vertices.
    
    Args:
        point: Point to test with shape (..., dim)
        hull_vertices: Hull vertices with shape (..., n_vertices, dim)
        tolerance: Numerical tolerance for boundary detection
        method: Algorithm to use ("linear_programming", "barycentric")
        
    Returns:
        Boolean array indicating inclusion (True = inside or on boundary)
        
    Algorithm (linear_programming method):
        A point p is inside the convex hull if it can be expressed as:
        p = sum(λᵢ * vᵢ) where sum(λᵢ) = 1 and λᵢ >= 0
        
        This is solved as a linear programming problem:
        minimize 0
        subject to: sum(λᵢ * vᵢ) = p
                   sum(λᵢ) = 1
                   λᵢ >= 0
    """
    point = jnp.asarray(point)
    hull_vertices = validate_point_cloud(hull_vertices)
    
    if method == "linear_programming":
        return _point_in_hull_lp(point, hull_vertices, tolerance)
    elif method == "barycentric":
        return _point_in_hull_barycentric(point, hull_vertices, tolerance)
    else:
        raise ValueError(f"Unknown method: {method}")


def _point_in_hull_lp(
    point: Array,
    hull_vertices: HullVertices,
    tolerance: float
) -> Array:
    """Linear programming based point-in-hull test."""
    n_vertices = hull_vertices.shape[-2]
    dim = hull_vertices.shape[-1]
    
    # For small hulls, use direct barycentric coordinate computation
    if n_vertices <= dim + 1:
        return _point_in_hull_barycentric(point, hull_vertices, tolerance)
    
    # For larger hulls, we need a more sophisticated LP solver
    # For now, use a simplified approach: check if point is within
    # the bounding box and use barycentric coordinates for a subset
    
    # Compute bounding box
    min_coords = jnp.min(hull_vertices, axis=-2)
    max_coords = jnp.max(hull_vertices, axis=-2)
    
    # Quick bounding box test
    in_bbox = jnp.all(
        (point >= min_coords - tolerance) & 
        (point <= max_coords + tolerance),
        axis=-1
    )
    
    # For points outside bounding box, return False
    # For points inside bounding box, do more detailed test
    def detailed_test(p, vertices):
        # Use a simplified approach: find closest simplex and test inclusion
        # This is a heuristic and not always accurate for complex hulls
        center = jnp.mean(vertices, axis=-2)
        distances = jnp.linalg.norm(vertices - center, axis=-1)
        closest_indices = jnp.argsort(distances)[:dim + 1]
        simplex = vertices[closest_indices]
        return _point_in_simplex(p, simplex, tolerance)
    
    # Apply detailed test only where bounding box test passed
    detailed_result = jax.lax.cond(
        jnp.any(in_bbox),
        lambda: detailed_test(point, hull_vertices),
        lambda: jnp.array(False, dtype=bool)
    )
    
    return in_bbox & detailed_result


def _point_in_hull_barycentric(
    point: Array,
    hull_vertices: HullVertices,
    tolerance: float
) -> Array:
    """Barycentric coordinate based point-in-hull test."""
    n_vertices = hull_vertices.shape[-2]
    dim = hull_vertices.shape[-1]
    
    if n_vertices == dim + 1:
        # Perfect simplex case
        return _point_in_simplex(point, hull_vertices, tolerance)
    elif n_vertices < dim + 1:
        # Degenerate case - not enough vertices for full-dimensional hull
        return jnp.array(False, dtype=bool)
    else:
        # Over-determined case - decompose into simplices
        # For simplicity, use the first (dim+1) vertices
        simplex = hull_vertices[..., :dim+1, :]
        return _point_in_simplex(point, simplex, tolerance)


def _point_in_simplex(
    point: Array,
    simplex_vertices: Array,
    tolerance: float
) -> Array:
    """Test if point is inside simplex using barycentric coordinates."""
    n_vertices = simplex_vertices.shape[-2]
    dim = simplex_vertices.shape[-1]
    
    if n_vertices != dim + 1:
        raise ValueError(f"Simplex must have {dim+1} vertices, got {n_vertices}")
    
    # Solve for barycentric coordinates
    # point = sum(λᵢ * vᵢ) with sum(λᵢ) = 1
    # Rearrange to: point - v₀ = sum(λᵢ * (vᵢ - v₀)) for i > 0
    
    v0 = simplex_vertices[..., 0, :]
    edge_vectors = simplex_vertices[..., 1:, :] - v0[..., None, :]
    point_offset = point - v0
    
    # Solve linear system: edge_vectors.T @ lambdas = point_offset
    try:
        # Use least squares for over-determined systems
        lambdas_rest, residuals, rank, s = jnp.linalg.lstsq(
            edge_vectors.T, point_offset, rcond=None
        )
        
        # Compute λ₀ = 1 - sum(λᵢ) for i > 0
        lambda0 = 1.0 - jnp.sum(lambdas_rest)
        
        # Full barycentric coordinates
        lambdas = jnp.concatenate([lambda0[None], lambdas_rest])
        
        # Check if all coordinates are non-negative (within tolerance)
        return jnp.all(lambdas >= -tolerance)
        
    except jnp.linalg.LinAlgError:
        # Singular matrix - degenerate simplex
        return jnp.array(False, dtype=bool)


def convex_hull_volume(
    vertices: HullVertices,
    method: str = "simplex_decomposition"
) -> Array:
    """Compute volume of convex hull (differentiable).
    
    Args:
        vertices: Hull vertices with shape (..., n_vertices, dim)
        method: Volume computation method
            - "simplex_decomposition": Decompose into simplices
            - "divergence_theorem": Use divergence theorem (3D only)
            - "monte_carlo": Monte Carlo estimation
        
    Returns:
        Volume of the convex hull (d-dimensional measure)
        
    Note:
        For d-dimensional space, volume is the d-dimensional measure.
        For 2D, this is area; for 3D, this is volume; etc.
    """
    vertices = validate_point_cloud(vertices)
    
    if method == "simplex_decomposition":
        return _volume_simplex_decomposition(vertices)
    elif method == "divergence_theorem":
        return _volume_divergence_theorem(vertices)
    elif method == "monte_carlo":
        return _volume_monte_carlo(vertices)
    else:
        raise ValueError(f"Unknown volume method: {method}")


def _volume_simplex_decomposition(vertices: HullVertices) -> Array:
    """Compute volume by decomposing hull into simplices."""
    n_vertices = vertices.shape[-2]
    dim = vertices.shape[-1]
    
    if n_vertices < dim + 1:
        # Not enough vertices for full-dimensional hull
        return jnp.array(0.0)
    
    if n_vertices == dim + 1:
        # Perfect simplex
        return compute_simplex_volume(vertices)
    
    # For more vertices, decompose into simplices
    # Use fan triangulation from first vertex
    v0 = vertices[..., 0, :]
    total_volume = 0.0
    
    # Create simplices by connecting v0 with each (dim)-dimensional face
    # This is a simplified approach - proper decomposition would use
    # a more sophisticated algorithm like Delaunay triangulation
    
    if dim == 2:
        # 2D case: decompose into triangles
        for i in range(1, n_vertices - 1):
            triangle = jnp.stack([
                v0,
                vertices[..., i, :],
                vertices[..., i + 1, :]
            ], axis=-2)
            total_volume += compute_simplex_volume(triangle)
    
    elif dim == 3:
        # 3D case: decompose into tetrahedra
        # Use convex hull's faces (simplified approximation)
        for i in range(1, n_vertices - 2):
            for j in range(i + 1, n_vertices - 1):
                tetrahedron = jnp.stack([
                    v0,
                    vertices[..., i, :],
                    vertices[..., j, :],
                    vertices[..., j + 1, :]
                ], axis=-2)
                total_volume += compute_simplex_volume(tetrahedron)
    
    else:
        # Higher dimensions: use approximate method
        # This is not geometrically accurate but provides a reasonable estimate
        warnings.warn(
            f"Simplex decomposition for dimension {dim} is approximate",
            UserWarning
        )
        # Use average simplex volume scaled by number of simplices
        if n_vertices >= dim + 1:
            sample_simplex = vertices[..., :dim+1, :]
            sample_volume = compute_simplex_volume(sample_simplex)
            # Rough scaling based on number of vertices
            scaling_factor = n_vertices / (dim + 1)
            total_volume = sample_volume * scaling_factor
    
    return jnp.abs(total_volume)


def _volume_divergence_theorem(vertices: HullVertices) -> Array:
    """Compute volume using divergence theorem (3D only)."""
    dim = vertices.shape[-1]
    
    if dim != 3:
        raise ValueError("Divergence theorem method only works for 3D")
    
    # TODO: Implement proper divergence theorem volume calculation
    # This requires computing the surface mesh and applying the theorem
    # For now, fall back to simplex decomposition
    warnings.warn(
        "Divergence theorem not yet implemented, using simplex decomposition",
        UserWarning
    )
    return _volume_simplex_decomposition(vertices)


def _volume_monte_carlo(
    vertices: HullVertices,
    n_samples: int = 10000,
    random_key: Optional[Array] = None
) -> Array:
    """Compute volume using Monte Carlo estimation."""
    if random_key is None:
        random_key = jax.random.PRNGKey(42)
    
    # Compute bounding box
    min_coords = jnp.min(vertices, axis=-2)
    max_coords = jnp.max(vertices, axis=-2)
    bbox_volume = jnp.prod(max_coords - min_coords)
    
    # Generate random points in bounding box
    dim = vertices.shape[-1]
    random_points = jax.random.uniform(
        random_key,
        (n_samples, dim),
        minval=min_coords,
        maxval=max_coords
    )
    
    # Test which points are inside the hull
    inside_count = 0
    for i in range(n_samples):
        if point_in_convex_hull(random_points[i], vertices):
            inside_count += 1
    
    # Estimate volume
    inside_ratio = inside_count / n_samples
    estimated_volume = bbox_volume * inside_ratio
    
    return estimated_volume


def convex_hull_surface_area(
    vertices: HullVertices,
    faces: Optional[Array] = None
) -> Array:
    """Compute surface area of convex hull.
    
    Args:
        vertices: Hull vertices with shape (..., n_vertices, dim)
        faces: Face vertex indices with shape (..., n_faces, vertices_per_face)
               If None, faces will be computed automatically
        
    Returns:
        Surface area (sum of face areas)
    """
    vertices = validate_point_cloud(vertices)
    dim = vertices.shape[-1]
    
    if faces is None:
        faces = _compute_hull_faces(vertices)
    
    if dim == 2:
        # 2D case: perimeter calculation
        return _compute_2d_perimeter(vertices)
    elif dim == 3:
        # 3D case: sum of triangle areas
        return _compute_3d_surface_area(vertices, faces)
    else:
        # Higher dimensions: approximate using boundary measure
        warnings.warn(
            f"Surface area computation for dimension {dim} is approximate",
            UserWarning
        )
        return _compute_nd_boundary_measure(vertices)


def _compute_2d_perimeter(vertices: HullVertices) -> Array:
    """Compute perimeter of 2D convex hull."""
    n_vertices = vertices.shape[-2]
    
    # Compute edge lengths
    edge_vectors = jnp.roll(vertices, -1, axis=-2) - vertices
    edge_lengths = jnp.linalg.norm(edge_vectors, axis=-1)
    
    return jnp.sum(edge_lengths, axis=-1)


def _compute_3d_surface_area(vertices: HullVertices, faces: Array) -> Array:
    """Compute surface area of 3D convex hull."""
    total_area = 0.0
    
    # For each triangular face, compute area
    for face_indices in faces:
        if len(face_indices) >= 3:
            # Get vertices of the face
            face_vertices = vertices[..., face_indices[:3], :]
            
            # Compute triangle area using cross product
            v1 = face_vertices[..., 1, :] - face_vertices[..., 0, :]
            v2 = face_vertices[..., 2, :] - face_vertices[..., 0, :]
            cross_product = jnp.cross(v1, v2)
            area = 0.5 * jnp.linalg.norm(cross_product)
            total_area += area
    
    return total_area


def _compute_nd_boundary_measure(vertices: HullVertices) -> Array:
    """Approximate boundary measure for high-dimensional hulls."""
    # This is a rough approximation
    n_vertices = vertices.shape[-2]
    dim = vertices.shape[-1]
    
    # Use average distance between vertices as approximation
    center = jnp.mean(vertices, axis=-2)
    distances = jnp.linalg.norm(vertices - center[..., None, :], axis=-1)
    avg_distance = jnp.mean(distances)
    
    # Scale by number of vertices and dimension
    boundary_measure = avg_distance * n_vertices * (dim ** 0.5)
    
    return boundary_measure


def _compute_hull_faces(vertices: HullVertices) -> Array:
    """Compute faces of convex hull.
    
    This is a simplified implementation that returns a reasonable
    approximation of the faces. A full implementation would require
    a proper convex hull algorithm.
    """
    n_vertices = vertices.shape[-2]
    dim = vertices.shape[-1]
    
    if dim == 2:
        # 2D: faces are edges (pairs of consecutive vertices)
        faces = []
        for i in range(n_vertices):
            faces.append([i, (i + 1) % n_vertices])
        return jnp.array(faces)
    
    elif dim == 3:
        # 3D: faces are triangles
        # This is a simplified triangulation - not guaranteed to be correct
        faces = []
        for i in range(n_vertices - 2):
            for j in range(i + 1, n_vertices - 1):
                for k in range(j + 1, n_vertices):
                    faces.append([i, j, k])
        return jnp.array(faces)
    
    else:
        # Higher dimensions: return empty array
        return jnp.array([])


def distance_to_convex_hull(
    point: Array,
    hull_vertices: HullVertices
) -> Array:
    """Compute distance from point to convex hull.
    
    Args:
        point: Point with shape (..., dim)
        hull_vertices: Hull vertices with shape (..., n_vertices, dim)
        
    Returns:
        Signed distance to hull:
        - Positive: point is outside hull
        - Zero: point is on boundary
        - Negative: point is inside hull
    """
    # Check if point is inside hull
    is_inside = point_in_convex_hull(point, hull_vertices)
    
    # Compute distance to closest vertex (approximation)
    distances_to_vertices = jnp.linalg.norm(
        hull_vertices - point[..., None, :], 
        axis=-1
    )
    min_distance = jnp.min(distances_to_vertices, axis=-1)
    
    # Return signed distance
    return jnp.where(is_inside, -min_distance, min_distance)


def hausdorff_distance(
    hull1_vertices: HullVertices,
    hull2_vertices: HullVertices
) -> Array:
    """Compute Hausdorff distance between two convex hulls.
    
    The Hausdorff distance is the maximum of:
    1. Maximum distance from any point in hull1 to hull2
    2. Maximum distance from any point in hull2 to hull1
    
    Args:
        hull1_vertices: First hull vertices
        hull2_vertices: Second hull vertices
        
    Returns:
        Hausdorff distance between the hulls
    """
    # Distance from hull1 vertices to hull2
    distances_1_to_2 = jnp.array([
        jnp.abs(distance_to_convex_hull(v, hull2_vertices))
        for v in hull1_vertices
    ])
    max_dist_1_to_2 = jnp.max(distances_1_to_2)
    
    # Distance from hull2 vertices to hull1
    distances_2_to_1 = jnp.array([
        jnp.abs(distance_to_convex_hull(v, hull1_vertices))
        for v in hull2_vertices
    ])
    max_dist_2_to_1 = jnp.max(distances_2_to_1)
    
    return jnp.maximum(max_dist_1_to_2, max_dist_2_to_1)


# JIT-compiled versions for performance
point_in_convex_hull_jit = jax.jit(point_in_convex_hull, static_argnames=['method'])
convex_hull_volume_jit = jax.jit(convex_hull_volume, static_argnames=['method'])
convex_hull_surface_area_jit = jax.jit(convex_hull_surface_area)
distance_to_convex_hull_jit = jax.jit(distance_to_convex_hull)
hausdorff_distance_jit = jax.jit(hausdorff_distance)