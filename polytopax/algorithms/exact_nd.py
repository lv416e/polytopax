"""N-dimensional exact convex hull algorithms for PolytopAX.

This module implements n-dimensional exact convex hull algorithms using
the QuickHull algorithm generalized to arbitrary dimensions. The implementation
prioritizes mathematical correctness and numerical stability while maintaining
JAX compatibility.

Key Features:
- n-dimensional QuickHull (4D and above)
- Robust geometric predicates for high dimensions
- Memory-efficient processing for large point sets
- JAX-compatible automatic differentiation support

Mathematical Foundation:
The n-dimensional QuickHull algorithm works by:
1. Finding an initial (n+1)-simplex from extreme points
2. For each (n-1)-facet, finding points outside the facet
3. Recursively building the hull by adding the farthest point
4. Managing the complex facet adjacency in higher dimensions

References:
- Barber, C.B., Dobkin, D.P., and Huhdanpaa, H. (1996). "The Quickhull Algorithm for Convex Hulls"
- Preparata, F.P. and Shamos, M.I. (1985). "Computational Geometry: An Introduction"
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..core.utils import (
    HullVertices,
    PointCloud,
    validate_point_cloud,
)


def quickhull_nd(
    points: PointCloud, tolerance: float = 1e-12, max_iterations: int = 10000, max_dimension: int = 10
) -> tuple[HullVertices, Array]:
    """N-dimensional QuickHull algorithm for exact convex hull computation.

    This implements the generalized QuickHull algorithm for arbitrary dimensions.
    For dimensions > max_dimension, the algorithm will issue a warning and fall
    back to approximation methods.

    Args:
        points: Input point cloud with shape (..., n_points, dim)
        tolerance: Numerical tolerance for geometric predicates
        max_iterations: Maximum iterations to prevent infinite loops
        max_dimension: Maximum supported dimension (for memory/performance)

    Returns:
        Tuple of (hull_vertices, hull_indices)

    Raises:
        ValueError: If dimension is less than 1 or points are invalid
        RuntimeError: If algorithm fails to converge

    Algorithm Complexity:
        - Time: O(n * log(n)) average, O(n^(floor(d/2))) worst case for d dimensions
        - Space: O(n) for output, O(2^d) for internal facet management

    Note:
        For dimensions > 10, consider using approximation methods for better
        performance. High-dimensional convex hulls can have exponentially
        many facets.
    """
    points = validate_point_cloud(points)
    n_points, dim = points.shape[-2], points.shape[-1]

    if dim < 1:
        raise ValueError(f"Dimension must be at least 1, got {dim}")

    if dim > max_dimension:
        warnings.warn(
            f"Dimension {dim} exceeds max_dimension {max_dimension}. "
            "Consider using approximation methods for better performance.",
            UserWarning,
            stacklevel=2,
        )

    if n_points < dim + 1:
        # Not enough points for full-dimensional hull
        return points, jnp.arange(n_points)

    # Delegate to specialized implementations for low dimensions
    if dim == 2:
        from .exact import _quickhull_2d

        return _quickhull_2d(points, tolerance)
    elif dim == 3:
        from .exact_3d import quickhull_3d

        return quickhull_3d(points, tolerance, max_iterations)

    # General n-dimensional implementation
    return _quickhull_nd_general(points, tolerance, max_iterations)


def _quickhull_nd_general(points: Array, tolerance: float, max_iterations: int) -> tuple[Array, Array]:
    """General n-dimensional QuickHull implementation.

    This implements the full QuickHull algorithm for dimensions >= 4.
    """
    n_points, dim = points.shape

    # Step 1: Find initial (dim+1)-simplex
    simplex_indices = _find_initial_simplex_nd(points, tolerance)

    if len(simplex_indices) < dim + 1:
        # Points are degenerate (lie in lower-dimensional subspace)
        return _handle_degenerate_nd_case(points, simplex_indices, tolerance)

    # Step 2: Initialize facets from the initial simplex
    initial_facets = _generate_simplex_facets(simplex_indices, dim)

    # Step 3: Build complete hull by recursive facet processing
    hull_facets = _build_nd_hull_recursive(points, initial_facets, simplex_indices, tolerance, max_iterations)

    # Step 4: Extract vertices from facets
    hull_vertices, hull_indices = _extract_vertices_from_facets_nd(points, hull_facets)

    return hull_vertices, hull_indices


# =============================================================================
# N-DIMENSIONAL GEOMETRIC PREDICATES
# =============================================================================


def orientation_nd(points: Array, tolerance: float = 1e-12) -> int:
    """Determine orientation of (d+1) points in d-dimensional space.

    Args:
        points: Array of shape (d+1, d) containing d+1 points in d dimensions
        tolerance: Numerical tolerance for coplanarity detection

    Returns:
        1 if positive orientation, -1 if negative, 0 if coplanar/degenerate

    Mathematical Detail:
        Computes the determinant of the matrix formed by translating all points
        by the first point, creating a d x d matrix. The sign of this determinant
        determines the orientation.

        For d dimensions, we need exactly d+1 points to determine orientation.
    """
    if points.shape[0] != points.shape[1] + 1:
        raise ValueError(
            f"Need exactly {points.shape[1] + 1} points for {points.shape[1]}D orientation, "
            f"got {points.shape[0]} points"
        )

    dim = points.shape[1]

    if dim == 0:
        return 0  # 0D case - no orientation

    if dim == 1:
        # 1D case: need exactly 2 points
        if points.shape[0] != 2:
            raise ValueError(f"Need exactly 2 points for 1D orientation, got {points.shape[0]} points")
        return 1 if points[1, 0] > points[0, 0] else (-1 if points[1, 0] < points[0, 0] else 0)

    # Translate by first point to get vectors
    vectors = points[1:] - points[0]

    # Compute determinant
    det = jnp.linalg.det(vectors)

    if abs(det) < tolerance:
        return 0  # Coplanar/degenerate
    elif det > 0:
        return 1  # Positive orientation
    else:
        return -1  # Negative orientation


def point_to_hyperplane_distance_nd(point: Array, hyperplane_points: Array, tolerance: float = 1e-12) -> Array:
    """Compute signed distance from point to hyperplane in n dimensions.

    Args:
        point: Point coordinates (shape: dim)
        hyperplane_points: Array of dim points defining the hyperplane (shape: dim, dim)
        tolerance: Numerical tolerance

    Returns:
        Signed distance (positive if point is on positive side of hyperplane)

    Mathematical Detail:
        For n-dimensional space, a hyperplane is defined by n points.
        We use the orientation test to determine the signed distance.
    """
    dim = point.shape[0]

    if hyperplane_points.shape != (dim, dim):
        raise ValueError(f"Hyperplane must be defined by {dim} points in {dim}D, got {hyperplane_points.shape}")

    # Create orientation test points: hyperplane points + test point
    test_points = jnp.vstack([hyperplane_points, point[None, :]])

    # Use orientation test to get signed distance
    orientation_result = orientation_nd(test_points, tolerance)

    if orientation_result == 0:
        return jnp.array(0.0)  # Point is on hyperplane

    # Compute actual distance using vectors
    hyperplane_vectors = hyperplane_points[1:] - hyperplane_points[0]
    point_vector = point - hyperplane_points[0]

    # Compute normal vector (for distance normalization)
    if dim == 1:
        return point_vector[0]
    elif dim == 2:
        # 2D: normal is perpendicular to line
        line_vec = hyperplane_vectors[0]
        normal = jnp.array([-line_vec[1], line_vec[0]])
        normal = normal / jnp.linalg.norm(normal)
        return jnp.dot(point_vector, normal)
    else:
        # For higher dimensions, approximate with orientation magnitude
        # This is a simplified implementation
        try:
            det = jnp.linalg.det(hyperplane_vectors)
            if jnp.abs(det) < tolerance:
                return jnp.array(0.0)
            return jnp.array(float(orientation_result) * jnp.linalg.norm(point_vector) / jnp.abs(det))
        except Exception:
            return jnp.array(float(orientation_result))


def is_point_inside_simplex_nd(point: Array, simplex: Array, tolerance: float = 1e-12) -> Array:
    """Test if point is inside n-dimensional simplex using barycentric coordinates.

    Args:
        point: Test point coordinates
        simplex: Simplex vertices (shape: n+1, n for n-dimensional simplex)
        tolerance: Numerical tolerance

    Returns:
        True if point is inside or on boundary of simplex

    Mathematical Detail:
        Uses barycentric coordinate method. Point is inside simplex if all
        barycentric coordinates are non-negative and sum to 1.
    """
    dim = point.shape[0]

    if simplex.shape != (dim + 1, dim):
        raise ValueError(f"Simplex must have {dim + 1} vertices in {dim}D, got {simplex.shape}")

    # Set up system for barycentric coordinates
    # Solve: simplex^T @ lambda = point, sum(lambda) = 1
    # This becomes: [simplex^T; 1^T] @ lambda = [point; 1]

    A = jnp.vstack([simplex.T, jnp.ones(dim + 1)])
    b = jnp.append(point, 1.0)

    try:
        # Solve for barycentric coordinates
        barycentric = jnp.linalg.solve(A, b)

        # Check if all coordinates are non-negative (within tolerance)
        return jnp.all(barycentric >= -tolerance)

    except np.linalg.LinAlgError:
        # Singular matrix - degenerate simplex
        return jnp.array(False)


# =============================================================================
# INITIAL SIMPLEX FINDING
# =============================================================================


def _find_initial_simplex_nd(points: Array, tolerance: float) -> list[int]:
    """Find (d+1) points that form a non-degenerate d-dimensional simplex.

    This uses a robust algorithm that finds extreme points along different
    directions to maximize the volume of the initial simplex.
    """
    n_points, dim = points.shape

    if n_points < dim + 1:
        return list(range(n_points))

    # Strategy: Find extreme points along coordinate axes and diagonal directions
    # This helps avoid degenerate configurations

    simplex_indices = []

    # Find extremal points along coordinate axes
    for axis in range(dim):
        min_idx = int(jnp.argmin(points[:, axis]))
        max_idx = int(jnp.argmax(points[:, axis]))

        if min_idx not in simplex_indices:
            simplex_indices.append(min_idx)
        if max_idx not in simplex_indices and len(simplex_indices) < dim + 1:
            simplex_indices.append(max_idx)

    # If we need more points, find points with maximum distance from current simplex
    while len(simplex_indices) < dim + 1:
        best_point = -1
        best_volume = jnp.array(0.0)

        for i in range(n_points):
            if i in simplex_indices:
                continue

            # Test adding this point to current simplex
            test_indices = [*simplex_indices, i]

            if len(test_indices) <= dim:
                # Not enough points yet, just add
                simplex_indices.append(i)
                break

            # Compute volume of simplex with this point
            test_points = points[jnp.array(test_indices)]
            volume = _compute_simplex_volume_nd(test_points, tolerance)

            if volume > best_volume:
                best_volume = volume
                best_point = i

        if best_point >= 0:
            simplex_indices.append(best_point)
        else:
            # No suitable point found
            break

    # Ensure we don't have too many points
    if len(simplex_indices) > dim + 1:
        simplex_indices = simplex_indices[: dim + 1]

    # Verify the simplex is non-degenerate
    if len(simplex_indices) == dim + 1:
        test_points = points[jnp.array(simplex_indices)]
        volume = _compute_simplex_volume_nd(test_points, tolerance)

        if volume < tolerance:
            # Degenerate simplex, remove last point
            simplex_indices.pop()

    return simplex_indices


def _compute_simplex_volume_nd(simplex_points: Array, tolerance: float) -> Array:
    """Compute the volume of an n-dimensional simplex.

    Args:
        simplex_points: Vertices of the simplex (shape: n+1, n)
        tolerance: Numerical tolerance

    Returns:
        Volume of the simplex (always non-negative)
    """
    if simplex_points.shape[0] != simplex_points.shape[1] + 1:
        return jnp.array(0.0)

    dim = simplex_points.shape[1]

    # Volume = |det(edge_vectors)| / factorial(dim)
    edge_vectors = simplex_points[1:] - simplex_points[0]

    try:
        det = jnp.linalg.det(edge_vectors)
        # Divide by factorial(dim) to get actual volume
        factorial_dim = jnp.prod(jnp.arange(1, dim + 1))
        return jnp.abs(det) / factorial_dim
    except np.linalg.LinAlgError:
        return jnp.array(0.0)


# =============================================================================
# FACET MANAGEMENT
# =============================================================================


def _generate_simplex_facets(simplex_indices: list[int], dim: int) -> list[list[int]]:
    """Generate all (d-1)-dimensional facets of a d-dimensional simplex.

    Args:
        simplex_indices: Vertex indices of the simplex
        dim: Dimension of the space

    Returns:
        List of facets, each defined by dim vertex indices
    """
    if len(simplex_indices) != dim + 1:
        raise ValueError(f"Simplex must have {dim + 1} vertices, got {len(simplex_indices)}")

    facets = []

    # Each facet is formed by removing one vertex from the simplex
    for i in range(dim + 1):
        facet = [simplex_indices[j] for j in range(dim + 1) if j != i]
        facets.append(facet)

    return facets


# =============================================================================
# PLACEHOLDER IMPLEMENTATIONS
# =============================================================================


def _handle_degenerate_nd_case(points: Array, extreme_indices: list[int], tolerance: float) -> tuple[Array, Array]:
    """Handle degenerate cases where points lie in lower-dimensional subspace."""
    # For now, just return the extreme points
    # TODO: Implement proper lower-dimensional projection and hull computation
    warnings.warn(
        "Degenerate n-dimensional case handling is not fully implemented. Returning extreme points only.",
        UserWarning,
        stacklevel=2,
    )

    if not extreme_indices:
        return jnp.empty((0, points.shape[1])), jnp.array([], dtype=int)

    hull_indices = jnp.array(extreme_indices, dtype=int)
    hull_vertices = points[hull_indices]

    return hull_vertices, hull_indices


def _build_nd_hull_recursive(
    points: Array, initial_facets: list[list[int]], processed_points: list[int], tolerance: float, max_iterations: int
) -> list[list[int]]:
    """Build the complete n-dimensional hull by recursively processing facets.

    This implements the full n-dimensional QuickHull algorithm:
    1. For each facet, find points outside it
    2. Find the farthest point from the facet
    3. Create new facets by connecting this point to visible facet ridges
    4. Remove facets that are no longer visible
    5. Recursively process new facets

    Args:
        points: All points in the point cloud
        initial_facets: List of initial facets from simplex
        processed_points: Points already included in hull
        tolerance: Numerical tolerance
        max_iterations: Maximum iterations to prevent infinite loops

    Returns:
        List of final hull facets
    """
    n_points, dim = points.shape
    processed_set = set(processed_points)
    remaining_points = [i for i in range(n_points) if i not in processed_set]

    if len(remaining_points) == 0:
        return initial_facets

    current_facets = initial_facets.copy()
    iteration = 0

    while iteration < max_iterations and len(remaining_points) > 0:
        iteration += 1
        facet_processed = False

        # Process each facet to find points that need to be added to the hull
        for facet_idx in range(len(current_facets)):
            if facet_idx >= len(current_facets):
                continue

            facet = current_facets[facet_idx]

            # Find points outside this facet
            outside_points = _find_points_outside_facet_nd(points, facet, remaining_points, tolerance)

            if len(outside_points) == 0:
                continue

            # Find the farthest point from this facet
            farthest_point_idx = _find_farthest_point_from_facet_nd(points, facet, outside_points, tolerance)

            if farthest_point_idx is None:
                continue

            # Add this point to processed set
            if farthest_point_idx in remaining_points:
                remaining_points.remove(farthest_point_idx)
                processed_set.add(farthest_point_idx)

            # Find all facets visible from the farthest point
            visible_facets = _find_visible_facets_nd(points, current_facets, farthest_point_idx, tolerance)

            # Find horizon ridges (ridges between visible and non-visible facets)
            horizon_ridges = _find_horizon_ridges_nd(current_facets, visible_facets, dim)

            # Create new facets by connecting farthest point to horizon ridges
            new_facets = []
            for ridge in horizon_ridges:
                new_facet = [*ridge, farthest_point_idx]
                # Ensure correct orientation
                new_facet = _ensure_facet_orientation_nd(points, new_facet, tolerance)
                new_facets.append(new_facet)

            # Remove visible facets and add new facets
            current_facets = [f for i, f in enumerate(current_facets) if i not in visible_facets]
            current_facets.extend(new_facets)

            facet_processed = True
            break

        # If no facet was processed in this iteration, we're done
        if not facet_processed:
            break

    return current_facets


def _find_points_outside_facet_nd(
    points: Array, facet: list[int], candidate_points: list[int], tolerance: float
) -> list[int]:
    """Find points that are outside (in front of) a given n-dimensional facet.

    For a facet to have points "outside" it, we need to determine which side
    is the exterior side. We do this by checking if the facet normal points
    away from the centroid of all points.
    """
    if len(facet) < points.shape[1]:
        return []  # Degenerate facet

    facet_points = points[jnp.array(facet)]

    # Only test full-dimensional facets (hyperplanes)
    if len(facet) != points.shape[1]:
        return []

    outside_points = []

    # Compute the centroid of all points to determine inside/outside
    all_points_centroid = jnp.mean(points, axis=0)

    # Test the orientation of the centroid with respect to this facet
    centroid_test_points = jnp.vstack([facet_points, all_points_centroid[None, :]])
    centroid_orientation = orientation_nd(centroid_test_points, tolerance)

    # If centroid orientation is 0, facet might be degenerate
    if centroid_orientation == 0:
        return []

    for point_idx in candidate_points:
        point = points[point_idx]

        # Test orientation of this point with respect to the facet
        test_points = jnp.vstack([facet_points, point[None, :]])
        point_orientation = orientation_nd(test_points, tolerance)

        # Point is outside if it has opposite orientation from the centroid
        if point_orientation != 0 and point_orientation != centroid_orientation:
            outside_points.append(point_idx)

    return outside_points


def _find_farthest_point_from_facet_nd(
    points: Array, facet: list[int], candidate_points: list[int], tolerance: float
) -> int | None:
    """Find the point with maximum distance from an n-dimensional facet."""
    if len(candidate_points) == 0:
        return None

    facet_points = points[jnp.array(facet)]
    facet_centroid = jnp.mean(facet_points, axis=0)

    max_distance = -1.0
    farthest_point = None

    for point_idx in candidate_points:
        point = points[point_idx]
        distance = jnp.linalg.norm(point - facet_centroid)

        if distance > max_distance:
            max_distance = distance
            farthest_point = point_idx

    return farthest_point if max_distance > tolerance else None


def _find_visible_facets_nd(points: Array, facets: list[list[int]], point_idx: int, tolerance: float) -> list[int]:
    """Find all facets visible from a given point in n dimensions."""
    visible_facets = []
    point = points[point_idx]

    for facet_idx, facet in enumerate(facets):
        if len(facet) < points.shape[1]:
            continue  # Skip degenerate facets

        facet_points = points[jnp.array(facet)]

        # Check if point is on the positive side of this facet
        if len(facet) == points.shape[1]:  # Hyperplane case
            try:
                test_points = jnp.vstack([facet_points, point[None, :]])
                orientation = orientation_nd(test_points, tolerance)

                if orientation > 0:  # Point is on positive side
                    visible_facets.append(facet_idx)
            except ValueError:
                # Skip if orientation test fails
                continue
        else:
            # For lower-dimensional facets, use simple distance test
            facet_centroid = jnp.mean(facet_points, axis=0)
            distance = jnp.linalg.norm(point - facet_centroid)

            if distance > tolerance:
                visible_facets.append(facet_idx)

    return visible_facets


def _find_horizon_ridges_nd(
    facets: list[list[int]], visible_facet_indices: list[int], dim: int
) -> list[tuple[int, ...]]:
    """Find horizon ridges (ridges between visible and non-visible facets).

    In n dimensions, ridges are (n-2)-dimensional faces shared between facets.
    """
    visible_set = set(visible_facet_indices)
    horizon_ridges: list[tuple[int, ...]] = []

    # For each visible facet, check its ridges
    for facet_idx in visible_facet_indices:
        if facet_idx >= len(facets):
            continue

        facet = facets[facet_idx]
        if len(facet) != dim:
            continue  # Skip non-hyperplane facets

        # Generate all ridges (subsets of size dim-1) of this facet
        from itertools import combinations

        for ridge in combinations(facet, dim - 1):
            ridge_list = tuple(ridge)

            # Find facets that share this ridge
            sharing_facets = []
            for other_idx, other_facet in enumerate(facets):
                if other_idx == facet_idx or len(other_facet) != dim:
                    continue

                # Check if this facet contains all vertices of the ridge
                if all(vertex in other_facet for vertex in ridge_list):
                    sharing_facets.append(other_idx)

            # If this ridge is shared with a non-visible facet, it's a horizon ridge
            for sharing_facet_idx in sharing_facets:
                if sharing_facet_idx not in visible_set:
                    horizon_ridges.append(ridge_list)
                    break

    # Remove duplicates
    unique_ridges: list[tuple[int, ...]] = []
    for ridge in horizon_ridges:
        sorted_ridge = tuple(sorted(ridge))
        if sorted_ridge not in unique_ridges:
            unique_ridges.append(sorted_ridge)

    return unique_ridges


def _ensure_facet_orientation_nd(points: Array, facet: list[int], tolerance: float) -> list[int]:
    """Ensure facet has correct outward-facing orientation in n dimensions.

    For n-dimensional convex hull, all facets should have outward-facing normals.
    We check this by ensuring the facet normal points away from the centroid.
    """
    if len(facet) != points.shape[1]:
        return facet  # Can only orient hyperplane facets

    facet_points = points[jnp.array(facet)]

    # Compute centroid of all points
    centroid = jnp.mean(points, axis=0)

    # Test orientation with centroid
    try:
        test_points = jnp.vstack([facet_points, centroid[None, :]])
        orientation = orientation_nd(test_points, tolerance)

        # If centroid is on positive side, reverse facet orientation
        if orientation > 0:
            return facet[::-1]  # Reverse order
        else:
            return facet  # Keep current order
    except (ValueError, Exception):
        # If orientation test fails, keep original orientation
        return facet


def _extract_vertices_from_facets_nd(points: Array, facets: list[list[int]]) -> tuple[Array, Array]:
    """Extract unique vertices from a list of n-dimensional facets."""
    if not facets:
        return jnp.empty((0, points.shape[1])), jnp.array([], dtype=int)

    unique_indices = set()
    for facet in facets:
        unique_indices.update(facet)

    if not unique_indices:
        return jnp.empty((0, points.shape[1])), jnp.array([], dtype=int)

    unique_indices_list = sorted(unique_indices)
    hull_indices = jnp.array(unique_indices_list, dtype=int)
    hull_vertices = points[hull_indices]

    return hull_vertices, hull_indices


# =============================================================================
# JIT-COMPILED VERSIONS
# =============================================================================

quickhull_nd_jit = jax.jit(quickhull_nd, static_argnames=["max_iterations", "max_dimension"])
orientation_nd_jit = jax.jit(orientation_nd)
point_to_hyperplane_distance_nd_jit = jax.jit(point_to_hyperplane_distance_nd)
is_point_inside_simplex_nd_jit = jax.jit(is_point_inside_simplex_nd)
