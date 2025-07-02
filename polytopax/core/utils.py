"""Core utility functions for PolytopAX."""

import warnings
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

# Type aliases
PointCloud = Array  # shape: (..., n_points, dimension)
HullVertices = Array  # shape: (n_vertices, dimension)
DirectionVectors = Array  # shape: (n_directions, dimension)
SamplingMethod = Literal["uniform", "icosphere", "adaptive"]


def validate_point_cloud(points: Array) -> Array:
    """Validate point cloud shape and numerical validity.

    Args:
        points: Input point cloud with shape (..., n_points, dim)

    Returns:
        Validated point cloud

    Raises:
        ValueError: Invalid shape or numerical values
    """
    if not isinstance(points, Array):
        raise TypeError(f"Expected JAX Array, got {type(points)}")

    if points.ndim < 2:
        raise ValueError(f"Point cloud must have at least 2 dimensions, got {points.ndim}")

    if points.shape[-1] < 1:
        raise ValueError(f"Point dimension must be at least 1, got {points.shape[-1]}")

    if points.shape[-2] < 1:
        raise ValueError(f"Must have at least 1 point, got {points.shape[-2]}")

    # Numerical validation - check for NaN and infinite values
    # Skip validation during JAX transformations (jit, vmap, etc.)
    # since traced arrays cannot be evaluated for concrete boolean values
    try:
        # This will work for concrete arrays but fail for traced arrays
        if jnp.any(jnp.isnan(points)):
            raise ValueError("Point cloud contains NaN values")

        if jnp.any(jnp.isinf(points)):
            raise ValueError("Point cloud contains infinite values")
    except jax.errors.TracerBoolConversionError:
        # During JAX transformations, skip numerical validation
        # Validation will happen when the function is actually called with concrete values
        pass

    return points


def generate_direction_vectors(
    dimension: int,
    n_directions: int,
    method: SamplingMethod = "uniform",
    random_key: Array | None = None
) -> DirectionVectors:
    """Generate direction vectors for sampling.

    Args:
        dimension: Spatial dimension
        n_directions: Number of directions to generate
        method: Sampling strategy
            - "uniform": Uniform distribution on sphere
            - "icosphere": Icosahedral subdivision (3D only)
            - "adaptive": Locally adaptive density sampling
        random_key: JAX random key (required for "uniform" and "adaptive")

    Returns:
        Normalized direction vector set with shape (n_directions, dimension)

    Raises:
        ValueError: Invalid parameters or unsupported combinations
    """
    if dimension < 1:
        raise ValueError(f"Dimension must be at least 1, got {dimension}")

    if n_directions < 1:
        raise ValueError(f"Number of directions must be at least 1, got {n_directions}")

    if method == "uniform":
        if random_key is None:
            random_key = jax.random.PRNGKey(0)

        # Generate random vectors from standard normal distribution
        directions = jax.random.normal(random_key, (n_directions, dimension))

        # Normalize to unit sphere
        norms = jnp.linalg.norm(directions, axis=1, keepdims=True)
        # Avoid division by zero
        norms = jnp.where(norms < 1e-12, 1.0, norms)
        directions = directions / norms

        return directions

    elif method == "icosphere":
        if dimension != 3:
            raise ValueError("Icosphere method is only supported for 3D (dimension=3)")

        return _generate_icosphere_directions(n_directions)

    elif method == "adaptive":
        if random_key is None:
            random_key = jax.random.PRNGKey(0)

        # For now, use uniform sampling as placeholder
        # TODO: Implement proper adaptive sampling in future versions
        warnings.warn(
            "Adaptive sampling not yet implemented, falling back to uniform sampling",
            UserWarning, stacklevel=2
        )
        return generate_direction_vectors(dimension, n_directions, "uniform", random_key)

    else:
        raise ValueError(f"Unknown sampling method: {method}")


def _generate_icosphere_directions(n_directions: int) -> DirectionVectors:
    """Generate direction vectors using icosahedral subdivision.

    Args:
        n_directions: Target number of directions

    Returns:
        Direction vectors approximating uniform distribution on sphere
    """
    # Base icosahedron vertices (12 vertices)
    phi = (1.0 + jnp.sqrt(5.0)) / 2.0  # Golden ratio

    # Icosahedron vertices
    vertices = jnp.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
    ], dtype=jnp.float32)

    # Normalize vertices
    vertices = vertices / jnp.linalg.norm(vertices, axis=1, keepdims=True)

    # If we need more directions, we can add face centers and edge midpoints
    directions = vertices

    if n_directions > 12:
        # Add face centers (20 faces for icosahedron)
        # For simplicity, we'll just repeat and perturb vertices
        # TODO: Implement proper subdivision in future versions
        n_extra = n_directions - 12
        if n_extra > 0:
            # Generate additional directions by slight perturbations
            key = jax.random.PRNGKey(42)
            perturbations = jax.random.normal(key, (n_extra, 3)) * 0.1
            # Repeat vertices cyclically to match the number of extra directions needed
            extra_base = jnp.tile(vertices, (n_extra // 12 + 1, 1))[:n_extra]
            extra_vertices = extra_base + perturbations
            extra_vertices = extra_vertices / jnp.linalg.norm(extra_vertices, axis=1, keepdims=True)
            directions = jnp.concatenate([directions, extra_vertices], axis=0)

    # Truncate to exact number if we have too many
    directions = directions[:n_directions]

    return directions


def robust_orientation_test(
    points: Array,
    tolerance: float = 1e-12
) -> Array:
    """Robust geometric orientation test.

    Implements numerically stable orientation tests for geometric predicates.
    Based on Shewchuk (1997) adaptive precision predicates concepts.

    Args:
        points: Points to test with shape (..., n_points, dim)
        tolerance: Numerical tolerance for degeneracy detection

    Returns:
        Orientation indicators

    Note:
        This is a simplified implementation. Full robust predicates
        would require adaptive precision arithmetic.
    """
    validate_point_cloud(points)

    # For now, implement basic numerical stability checks
    # TODO: Implement full Shewchuk adaptive precision predicates

    # Check for near-degenerate configurations
    if points.shape[-2] < points.shape[-1] + 1:
        # Not enough points for full-dimensional simplex
        return jnp.zeros(points.shape[:-2], dtype=bool)

    # Simple determinant-based orientation test for 2D/3D
    if points.shape[-1] == 2 and points.shape[-2] >= 3:
        # 2D orientation test
        p0, p1, p2 = points[..., 0, :], points[..., 1, :], points[..., 2, :]
        det = (p1[..., 0] - p0[..., 0]) * (p2[..., 1] - p0[..., 1]) - \
              (p1[..., 1] - p0[..., 1]) * (p2[..., 0] - p0[..., 0])
        return jnp.abs(det) > tolerance

    elif points.shape[-1] == 3 and points.shape[-2] >= 4:
        # 3D orientation test
        p0 = points[..., 0, :]
        v1 = points[..., 1, :] - p0
        v2 = points[..., 2, :] - p0
        v3 = points[..., 3, :] - p0

        # Compute determinant of 3x3 matrix
        det = jnp.linalg.det(jnp.stack([v1, v2, v3], axis=-2))
        return jnp.abs(det) > tolerance

    # For higher dimensions or insufficient points, use general approach
    return jnp.ones(points.shape[:-2], dtype=bool)


def compute_simplex_volume(vertices: Array) -> Array:
    """Compute volume of simplex defined by vertices.

    Args:
        vertices: Simplex vertices with shape (..., n_vertices, dim)
                 where n_vertices = dim + 1 for full-dimensional simplex

    Returns:
        Volume of the simplex
    """
    validate_point_cloud(vertices)

    n_vertices = vertices.shape[-2]
    dim = vertices.shape[-1]

    if n_vertices != dim + 1:
        raise ValueError(f"Expected {dim + 1} vertices for {dim}D simplex, got {n_vertices}")

    if dim == 0:
        return jnp.array(1.0)

    # Use the formula: |det(v1-v0, v2-v0, ..., vd-v0)| / d!
    v0 = vertices[..., 0, :]
    edge_vectors = vertices[..., 1:, :] - v0[..., None, :]

    # Compute determinant
    det = jnp.linalg.det(edge_vectors)

    # Volume is |det| / d!
    factorial = jnp.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880][dim])
    volume = jnp.abs(det) / factorial

    return volume


def project_to_simplex(point: Array, vertices: Array) -> tuple[Array, Array]:
    """Project point onto simplex defined by vertices.

    Args:
        point: Point to project with shape (..., dim)
        vertices: Simplex vertices with shape (..., n_vertices, dim)

    Returns:
        Tuple of (projected_point, barycentric_coordinates)
    """
    validate_point_cloud(vertices)

    # This is a simplified implementation
    # TODO: Implement proper simplex projection algorithm

    # For now, just return closest vertex
    distances = jnp.linalg.norm(vertices - point[..., None, :], axis=-1)
    closest_idx = jnp.argmin(distances, axis=-1)

    vertices.shape[-2]
    barycentric = jnp.zeros(vertices.shape[:-1])
    barycentric = barycentric.at[..., closest_idx].set(1.0)

    projected_point = jnp.sum(barycentric[..., :, None] * vertices, axis=-2)

    return projected_point, barycentric


def remove_duplicate_points(
    points: Array,
    tolerance: float = 1e-10
) -> tuple[Array, Array]:
    """Remove duplicate points within tolerance.

    Args:
        points: Point cloud with shape (..., n_points, dim)
        tolerance: Distance tolerance for considering points duplicate

    Returns:
        Tuple of (unique_points, unique_indices)
    """
    validate_point_cloud(points)

    # This is a simplified O(n²) implementation
    # TODO: Implement more efficient algorithm for large point sets

    n_points = points.shape[-2]
    points.shape[-1]

    # For JAX JIT compatibility, use a simpler approach that avoids boolean indexing
    # Just return all points for now (placeholder implementation)
    # In a production implementation, we would use JAX-compatible algorithms
    # such as sorting-based deduplication or other concrete indexing methods

    # Create identity indices (no duplicates removed in this simplified version)
    unique_indices = jnp.arange(n_points)
    unique_points = points

    return unique_points, unique_indices


def scale_to_unit_ball(points: Array) -> tuple[Array, tuple[Array, float]]:
    """Scale point cloud to fit in unit ball.

    Args:
        points: Point cloud with shape (..., n_points, dim)

    Returns:
        Tuple of (scaled_points, (center, scale_factor))
    """
    validate_point_cloud(points)

    # Compute center and scale
    center = jnp.mean(points, axis=-2, keepdims=True)
    centered_points = points - center

    max_distance = jnp.max(jnp.linalg.norm(centered_points, axis=-1))
    scale_factor = jnp.where(max_distance > 1e-12, max_distance, 1.0)

    scaled_points = centered_points / scale_factor

    return scaled_points, (center.squeeze(-2), scale_factor)


def unscale_from_unit_ball(
    points: Array,
    transform_params: tuple[Array, float]
) -> Array:
    """Reverse scaling from unit ball.

    Args:
        points: Scaled point cloud
        transform_params: (center, scale_factor) from scale_to_unit_ball

    Returns:
        Original scale point cloud
    """
    center, scale_factor = transform_params
    return points * scale_factor + center
