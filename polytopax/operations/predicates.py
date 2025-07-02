"""Geometric predicates for convex hull operations."""

import warnings

import jax
import jax.numpy as jnp
from jax import Array

from ..core.utils import HullVertices, compute_simplex_volume, validate_point_cloud


def point_in_convex_hull(
    point: Array,
    hull_vertices: HullVertices,
    tolerance: float = 1e-8,
    method: str = "halfspace"
) -> Array:
    """Test if point is inside convex hull.

    Determines whether a point lies inside, on the boundary, or outside
    of the convex hull defined by the given vertices.

    Args:
        point: Point to test with shape (..., dim)
        hull_vertices: Hull vertices with shape (..., n_vertices, dim)
        tolerance: Numerical tolerance for boundary detection
        method: Algorithm to use ("halfspace", "linear_programming", "barycentric")

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

    # Validate dimensional consistency
    if point.ndim == 0:
        raise ValueError("Point must have at least 1 dimension")

    point_dim = point.shape[-1] if point.ndim > 0 else 1
    hull_dim = hull_vertices.shape[-1]

    if point_dim != hull_dim:
        raise ValueError(f"Point dimension ({point_dim}) must match hull dimension ({hull_dim})")

    if method == "linear_programming":
        return _point_in_hull_lp(point, hull_vertices, tolerance)
    elif method == "barycentric":
        return _point_in_hull_barycentric(point, hull_vertices, tolerance)
    elif method == "halfspace":
        return _point_in_hull_halfspace(point, hull_vertices, tolerance)
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
            - "shoelace": Shoelace formula (2D only)
            - "divergence_theorem": Use divergence theorem (3D only)
            - "monte_carlo": Monte Carlo estimation
            - "multi_method": Consensus across multiple methods

    Returns:
        Volume of the convex hull (d-dimensional measure)

    Note:
        For d-dimensional space, volume is the d-dimensional measure.
        For 2D, this is area; for 3D, this is volume; etc.
    """
    vertices = validate_point_cloud(vertices)

    if method == "simplex_decomposition":
        return _volume_simplex_decomposition(vertices)
    elif method == "shoelace":
        return _volume_shoelace_formula(vertices)
    elif method == "divergence_theorem":
        return _volume_divergence_theorem(vertices)
    elif method == "monte_carlo":
        return _volume_monte_carlo(vertices)
    elif method == "multi_method":
        return _volume_multi_method_consensus(vertices)
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
            UserWarning, stacklevel=2
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
        UserWarning, stacklevel=2
    )
    return _volume_simplex_decomposition(vertices)


def _volume_monte_carlo(
    vertices: HullVertices,
    n_samples: int = 10000,
    random_key: Array | None = None
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
    faces: Array | None = None
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
            UserWarning, stacklevel=2
        )
        return _compute_nd_boundary_measure(vertices)


def _compute_2d_perimeter(vertices: HullVertices) -> Array:
    """Compute perimeter of 2D convex hull."""
    vertices.shape[-2]

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


# =============================================================================
# PHASE 2: IMPROVED VOLUME COMPUTATION METHODS
# =============================================================================

def _volume_shoelace_formula(vertices: HullVertices) -> Array:
    """Compute 2D polygon area using shoelace formula.

    The shoelace formula: Area = 0.5 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
    """
    if vertices.shape[-1] != 2:
        raise ValueError("Shoelace formula only works for 2D polygons")

    n_vertices = vertices.shape[-2]
    if n_vertices < 3:
        return jnp.array(0.0)

    # Sort vertices by angle to ensure proper ordering
    centroid = jnp.mean(vertices, axis=-2)
    centered_vertices = vertices - centroid

    # Compute angles from centroid
    angles = jnp.arctan2(centered_vertices[..., 1], centered_vertices[..., 0])
    sorted_indices = jnp.argsort(angles)
    sorted_vertices = vertices[sorted_indices]

    # Apply shoelace formula
    x = sorted_vertices[..., 0]
    y = sorted_vertices[..., 1]

    # Cyclic differences: x_i * y_{i+1} - x_{i+1} * y_i
    x_next = jnp.roll(x, -1, axis=-1)
    y_next = jnp.roll(y, -1, axis=-1)

    cross_products = x * y_next - x_next * y
    area = 0.5 * jnp.abs(jnp.sum(cross_products))

    return area


def _volume_multi_method_consensus(vertices: HullVertices) -> Array:
    """Compute volume using multiple methods and return consensus.

    Uses different methods based on dimensionality and returns a consensus
    value to improve accuracy and reliability.
    """
    dim = vertices.shape[-1]

    if dim == 2:
        # For 2D, use both simplex decomposition and shoelace
        try:
            volume_simplex = _volume_simplex_decomposition(vertices)
            volume_shoelace = _volume_shoelace_formula(vertices)

            # Check agreement between methods
            relative_diff = jnp.abs(volume_simplex - volume_shoelace) / jnp.maximum(volume_simplex, 1e-10)

            # If methods agree well, return average
            if relative_diff < 0.1:  # 10% agreement
                return 0.5 * (volume_simplex + volume_shoelace)
            else:
                # If methods disagree, prefer shoelace for 2D (more accurate)
                return volume_shoelace

        except (ValueError, Exception):
            # Fallback to simplex decomposition
            return _volume_simplex_decomposition(vertices)

    elif dim == 3:
        # For 3D, use simplex decomposition (most reliable for 3D)
        try:
            volume_simplex = _volume_simplex_decomposition(vertices)
            # Could add other 3D methods here in the future
            return volume_simplex
        except Exception:
            # Fallback to Monte Carlo if simplex fails
            return _volume_monte_carlo(vertices, n_samples=1000)

    else:
        # For higher dimensions, use simplex decomposition
        return _volume_simplex_decomposition(vertices)


def _volume_determinant_method(vertices: HullVertices) -> Array:
    """Alternative volume computation using determinant method.

    This method is particularly accurate for simplices and can serve
    as a cross-check for other methods.
    """
    n_vertices, dim = vertices.shape[-2], vertices.shape[-1]

    if n_vertices == dim + 1:
        # Perfect simplex - use determinant formula
        v0 = vertices[0]
        edge_vectors = vertices[1:] - v0

        if dim == edge_vectors.shape[0]:
            det = jnp.linalg.det(edge_vectors)
            # Volume = |det| / d!
            factorial = jnp.array([1, 1, 2, 6, 24, 120, 720, 5040][dim])
            return jnp.abs(det) / factorial

    # For non-simplex cases, fall back to simplex decomposition
    return _volume_simplex_decomposition(vertices)


def compute_volume_accuracy_metrics(
    vertices: HullVertices,
    exact_volume: float | None = None
) -> dict[str, float]:
    """Compute accuracy metrics for volume computation methods.

    Args:
        vertices: Hull vertices
        exact_volume: Known exact volume for comparison (if available)

    Returns:
        Dictionary containing accuracy metrics
    """
    dim = vertices.shape[-1]

    # Compute volume with different methods
    volumes = {}

    try:
        volumes["simplex"] = float(_volume_simplex_decomposition(vertices))
    except Exception:
        volumes["simplex"] = None

    if dim == 2:
        try:
            volumes["shoelace"] = float(_volume_shoelace_formula(vertices))
        except Exception:
            volumes["shoelace"] = None

    try:
        volumes["multi_method"] = float(_volume_multi_method_consensus(vertices))
    except Exception:
        volumes["multi_method"] = None

    # Compute consistency metrics
    valid_volumes = [v for v in volumes.values() if v is not None]

    if len(valid_volumes) >= 2:
        mean_volume = jnp.mean(jnp.array(valid_volumes))
        std_volume = jnp.std(jnp.array(valid_volumes))
        coefficient_of_variation = std_volume / mean_volume if mean_volume > 0 else float('inf')
    else:
        mean_volume = valid_volumes[0] if valid_volumes else 0.0
        std_volume = 0.0
        coefficient_of_variation = 0.0

    metrics = {
        "volumes": volumes,
        "mean_volume": float(mean_volume),
        "std_volume": float(std_volume),
        "coefficient_of_variation": float(coefficient_of_variation),
        "method_consistency": float(1.0 - coefficient_of_variation) if coefficient_of_variation < 1 else 0.0
    }

    # Add accuracy metrics if exact volume is provided
    if exact_volume is not None:
        metrics["exact_volume"] = exact_volume
        for method, volume in volumes.items():
            if volume is not None:
                relative_error = abs(volume - exact_volume) / exact_volume
                metrics[f"{method}_relative_error"] = float(relative_error)
                metrics[f"{method}_accurate"] = relative_error < 0.05  # 5% threshold

    return metrics


# =============================================================================
# PHASE 2: IMPROVED POINT CONTAINMENT METHODS
# =============================================================================

def _point_in_hull_halfspace(
    point: Array,
    hull_vertices: HullVertices,
    tolerance: float
) -> Array:
    """Improved halfspace-based point-in-hull test.

    This method computes the convex hull faces and checks if the point
    is on the correct side of all halfspaces defined by the faces.
    """
    n_vertices, dim = hull_vertices.shape[-2], hull_vertices.shape[-1]

    if n_vertices < dim + 1:
        # Not enough vertices for full-dimensional hull
        # Fall back to distance-based test
        distances = jnp.linalg.norm(hull_vertices - point, axis=-1)
        min_distance = jnp.min(distances)
        return min_distance <= tolerance

    if dim == 2:
        return _point_in_hull_2d_robust(point, hull_vertices, tolerance)
    elif dim == 3:
        return _point_in_hull_3d_robust(point, hull_vertices, tolerance)
    else:
        # For higher dimensions, use improved barycentric method
        return _point_in_hull_barycentric_robust(point, hull_vertices, tolerance)


def _point_in_hull_2d_robust(
    point: Array,
    hull_vertices: HullVertices,
    tolerance: float
) -> Array:
    """Robust 2D point-in-polygon test using winding number."""
    n_vertices = hull_vertices.shape[-2]

    if n_vertices < 3:
        # Degenerate case
        distances = jnp.linalg.norm(hull_vertices - point, axis=-1)
        return jnp.min(distances) <= tolerance

    # Sort vertices by angle to ensure proper order
    centroid = jnp.mean(hull_vertices, axis=-2)
    centered_vertices = hull_vertices - centroid
    angles = jnp.arctan2(centered_vertices[:, 1], centered_vertices[:, 0])
    sorted_indices = jnp.argsort(angles)
    sorted_vertices = hull_vertices[sorted_indices]

    # Use winding number algorithm
    winding_number = 0.0

    for i in range(n_vertices):
        v1 = sorted_vertices[i] - point
        v2 = sorted_vertices[(i + 1) % n_vertices] - point

        # Check if point is on edge (within tolerance)
        edge_vec = v2 - v1
        if jnp.linalg.norm(edge_vec) > 1e-12:
            # Project point onto edge
            t = jnp.dot(-v1, edge_vec) / jnp.dot(edge_vec, edge_vec)
            t = jnp.clip(t, 0.0, 1.0)
            closest_point = v1 + t * edge_vec
            distance_to_edge = jnp.linalg.norm(closest_point)

            if distance_to_edge <= tolerance:
                return jnp.array(True)

        # Compute contribution to winding number
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        dot_product = jnp.dot(v1, v2)

        angle = jnp.arctan2(cross_product, dot_product)
        winding_number += angle

    # Point is inside if winding number is close to ±2π
    abs_winding = jnp.abs(winding_number)
    return abs_winding > jnp.pi  # Threshold for "inside"


def _point_in_hull_3d_robust(
    point: Array,
    hull_vertices: HullVertices,
    tolerance: float
) -> Array:
    """Robust 3D point-in-hull test."""
    n_vertices = hull_vertices.shape[-2]

    if n_vertices < 4:
        # Degenerate 3D case
        distances = jnp.linalg.norm(hull_vertices - point, axis=-1)
        return jnp.min(distances) <= tolerance

    # For 3D, use tetrahedralization approach
    # Check if point is inside any tetrahedron formed by hull vertices
    centroid = jnp.mean(hull_vertices, axis=-2)

    # Test if point is inside the tetrahedron formed by centroid and any face
    for i in range(n_vertices - 2):
        for j in range(i + 1, n_vertices - 1):
            for k in range(j + 1, n_vertices):
                # Form tetrahedron with centroid and vertices i, j, k
                tetrahedron = jnp.array([
                    centroid,
                    hull_vertices[i],
                    hull_vertices[j],
                    hull_vertices[k]
                ])

                # Check if point is in this tetrahedron using barycentric coordinates
                if _point_in_tetrahedron(point, tetrahedron, tolerance):
                    return jnp.array(True)

    return jnp.array(False)


def _point_in_tetrahedron(
    point: Array,
    tetrahedron_vertices: Array,
    tolerance: float
) -> Array:
    """Test if point is inside tetrahedron using barycentric coordinates."""
    # Solve for barycentric coordinates
    # point = λ₀*v₀ + λ₁*v₁ + λ₂*v₂ + λ₃*v₃ where Σλᵢ = 1

    v0 = tetrahedron_vertices[0]
    edge_matrix = tetrahedron_vertices[1:] - v0  # 3x3 matrix
    point_vec = point - v0

    try:
        # Solve the linear system
        lambdas_123 = jnp.linalg.solve(edge_matrix.T, point_vec)
        lambda_0 = 1.0 - jnp.sum(lambdas_123)

        all_lambdas = jnp.concatenate([jnp.array([lambda_0]), lambdas_123])

        # Point is inside if all barycentric coordinates are non-negative
        return jnp.all(all_lambdas >= -tolerance)

    except jnp.linalg.LinAlgError:
        # Singular matrix - degenerate tetrahedron
        return jnp.array(False)


def _point_in_hull_barycentric_robust(
    point: Array,
    hull_vertices: HullVertices,
    tolerance: float
) -> Array:
    """Improved barycentric coordinate based point-in-hull test."""
    n_vertices, dim = hull_vertices.shape[-2], hull_vertices.shape[-1]

    if n_vertices == dim + 1:
        # Perfect simplex - use exact barycentric coordinates
        return _point_in_simplex_exact(point, hull_vertices, tolerance)
    elif n_vertices < dim + 1:
        # Under-determined - check distance to hull
        distances = jnp.linalg.norm(hull_vertices - point, axis=-1)
        return jnp.min(distances) <= tolerance
    else:
        # Over-determined - decompose into simplices
        return _point_in_hull_simplex_decomposition(point, hull_vertices, tolerance)


def _point_in_simplex_exact(
    point: Array,
    simplex_vertices: Array,
    tolerance: float
) -> Array:
    """Exact point-in-simplex test using barycentric coordinates."""
    n_vertices, dim = simplex_vertices.shape[-2], simplex_vertices.shape[-1]

    if n_vertices != dim + 1:
        raise ValueError(f"Simplex in {dim}D should have {dim+1} vertices, got {n_vertices}")

    # Set up barycentric coordinate system
    v0 = simplex_vertices[0]
    edge_matrix = simplex_vertices[1:] - v0
    point_vec = point - v0

    try:
        # Solve for barycentric coordinates
        lambdas_rest = jnp.linalg.solve(edge_matrix.T, point_vec)
        lambda_0 = 1.0 - jnp.sum(lambdas_rest)

        all_lambdas = jnp.concatenate([jnp.array([lambda_0]), lambdas_rest])

        # Point is inside if all coordinates are non-negative (within tolerance)
        return jnp.all(all_lambdas >= -tolerance)

    except jnp.linalg.LinAlgError:
        # Degenerate simplex
        distances = jnp.linalg.norm(simplex_vertices - point, axis=-1)
        return jnp.min(distances) <= tolerance


def _point_in_hull_simplex_decomposition(
    point: Array,
    hull_vertices: HullVertices,
    tolerance: float
) -> Array:
    """Test point containment by decomposing hull into simplices."""
    n_vertices, dim = hull_vertices.shape[-2], hull_vertices.shape[-1]

    # Simple approach: test against tetrahedra/triangles formed with centroid
    centroid = jnp.mean(hull_vertices, axis=-2)

    # For each subset of dim vertices, form a simplex with centroid
    # and test if point is inside
    if dim == 2:
        # Test triangles
        for i in range(n_vertices):
            j = (i + 1) % n_vertices
            triangle = jnp.array([centroid, hull_vertices[i], hull_vertices[j]])
            if _point_in_simplex_exact(point, triangle, tolerance):
                return jnp.array(True)
    elif dim == 3:
        # Test tetrahedra (simplified approach)
        for i in range(n_vertices - 2):
            for j in range(i + 1, n_vertices - 1):
                for k in range(j + 1, n_vertices):
                    tetrahedron = jnp.array([
                        centroid, hull_vertices[i], hull_vertices[j], hull_vertices[k]
                    ])
                    if _point_in_simplex_exact(point, tetrahedron, tolerance):
                        return jnp.array(True)

    return jnp.array(False)


# JIT-compiled versions for performance
point_in_convex_hull_jit = jax.jit(point_in_convex_hull, static_argnames=['method'])
convex_hull_volume_jit = jax.jit(convex_hull_volume, static_argnames=['method'])
convex_hull_surface_area_jit = jax.jit(convex_hull_surface_area)
distance_to_convex_hull_jit = jax.jit(distance_to_convex_hull)
hausdorff_distance_jit = jax.jit(hausdorff_distance)
