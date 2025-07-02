"""Differentiable approximate convex hull algorithms."""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Tuple, Optional
import warnings

from ..core.utils import (
    validate_point_cloud,
    generate_direction_vectors,
    remove_duplicate_points,
    PointCloud,
    HullVertices,
    SamplingMethod
)


def approximate_convex_hull(
    points: PointCloud,
    n_directions: int = 100,
    method: SamplingMethod = "uniform",
    temperature: float = 0.1,
    random_key: Optional[Array] = None,
    remove_duplicates: bool = True,
    tolerance: float = 1e-10
) -> Tuple[HullVertices, Array]:
    """Differentiable approximate convex hull computation.
    
    This function computes an approximate convex hull using direction vector
    sampling and differentiable soft selection. The approximation is suitable
    for machine learning applications where gradient computation is required.
    
    Args:
        points: Point cloud with shape (..., n_points, dim)
        n_directions: Number of sampling directions
        method: Sampling strategy ("uniform", "icosphere", "adaptive")
        temperature: Softmax temperature for differentiability control
                    Lower values → more sparse selection (closer to hard argmax)
                    Higher values → more uniform weighting
        random_key: JAX random key (auto-generated if None)
        remove_duplicates: Whether to remove duplicate vertices
        tolerance: Tolerance for duplicate removal
        
    Returns:
        Tuple of (hull_vertices, hull_indices):
            - hull_vertices: Approximate convex hull vertices
            - hull_indices: Indices of selected points in original array
            
    Algorithm:
        1. Generate direction vectors using specified sampling method
        2. For each direction, find the "farthest" point using soft selection:
           - Compute dot products (projection scores)
           - Apply softmax to make selection differentiable
           - Compute weighted combination of points
        3. Remove duplicate vertices if requested
        
    Example:
        >>> import jax.numpy as jnp
        >>> points = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        >>> hull_vertices, indices = approximate_convex_hull(points, n_directions=20)
        >>> print(hull_vertices.shape)  # (n_hull_vertices, 2)
    """
    # Validate inputs
    points = validate_point_cloud(points)
    
    if n_directions < 1:
        raise ValueError(f"n_directions must be positive, got {n_directions}")
    
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    
    # Get dimensions
    batch_dims = points.shape[:-2]
    n_points = points.shape[-2]
    dim = points.shape[-1]
    
    # Generate random key if not provided
    if random_key is None:
        random_key = jax.random.PRNGKey(0)
    
    # Generate direction vectors
    directions = generate_direction_vectors(
        dimension=dim,
        n_directions=n_directions,
        method=method,
        random_key=random_key
    )
    
    # Compute projection scores for all directions
    # Shape: (..., n_points, n_directions)
    scores = jnp.dot(points, directions.T)
    
    # Apply soft selection for differentiability
    # Shape: (..., n_points, n_directions)
    weights = jax.nn.softmax(scores / temperature, axis=-2)
    
    # Compute soft hull points as weighted combinations
    # Shape: (..., n_directions, dim)
    soft_hull_points = jnp.sum(
        weights[..., :, :, None] * points[..., :, None, :], 
        axis=-3
    )
    
    # For indices, use hard selection (non-differentiable but needed for indexing)
    hard_indices = jnp.argmax(scores, axis=-2)  # Shape: (..., n_directions)
    
    # Reshape to standard hull format
    hull_vertices = soft_hull_points
    hull_indices = hard_indices
    
    # Remove duplicates if requested
    if remove_duplicates:
        hull_vertices, unique_indices = remove_duplicate_points(
            hull_vertices, tolerance=tolerance
        )
        # Update indices to reflect unique selection
        hull_indices = hull_indices[..., unique_indices]
    
    return hull_vertices, hull_indices


def batched_approximate_hull(
    batch_points: Array,
    **kwargs
) -> Tuple[Array, Array]:
    """Batch processing version of approximate_convex_hull.
    
    Args:
        batch_points: Batched point clouds with shape (batch_size, n_points, dim)
        **kwargs: Arguments passed to approximate_convex_hull
        
    Returns:
        Tuple of batched (hull_vertices, hull_indices)
        
    Example:
        >>> batch_points = jnp.array([
        ...     [[0, 0], [1, 0], [0, 1]],  # Triangle 1
        ...     [[0, 0], [2, 0], [0, 2]]   # Triangle 2
        ... ])
        >>> hulls, indices = batched_approximate_hull(batch_points)
    """
    return jax.vmap(
        approximate_convex_hull, 
        in_axes=(0,),
        out_axes=(0, 0)
    )(batch_points, **kwargs)


def soft_argmax_selection(
    scores: Array,
    temperature: float,
    points: Array
) -> Tuple[Array, Array]:
    """Differentiable soft selection of extreme points.
    
    This function replaces the non-differentiable argmax operation with
    a differentiable soft selection using the softmax function.
    
    Args:
        scores: Selection scores with shape (..., n_points)
        temperature: Softmax temperature parameter
        points: Points corresponding to scores with shape (..., n_points, dim)
        
    Returns:
        Tuple of (soft_selected_point, selection_weights)
        
    Mathematical formulation:
        Traditional (non-differentiable):
            idx = argmax(scores)
            selected_point = points[idx]
            
        Soft (differentiable):
            weights = softmax(scores / temperature)
            selected_point = sum(weights * points)
    """
    # Compute soft selection weights
    weights = jax.nn.softmax(scores / temperature, axis=-1)
    
    # Compute weighted combination
    soft_point = jnp.sum(weights[..., :, None] * points, axis=-2)
    
    return soft_point, weights


def adaptive_temperature_control(
    scores: Array,
    target_sparsity: float = 0.1,
    min_temperature: float = 0.01,
    max_temperature: float = 10.0
) -> float:
    """Adaptive temperature control for soft selection.
    
    Automatically adjusts the softmax temperature to achieve a target
    sparsity level in the selection weights.
    
    Args:
        scores: Selection scores
        target_sparsity: Target sparsity (fraction of "active" selections)
        min_temperature: Minimum allowed temperature
        max_temperature: Maximum allowed temperature
        
    Returns:
        Optimal temperature value
        
    Note:
        This is a simplified implementation. A full implementation would
        use iterative optimization to find the optimal temperature.
    """
    # Simple heuristic: use score variance to estimate appropriate temperature
    score_std = jnp.std(scores)
    
    # Higher variance → lower temperature (more confident selection)
    # Lower variance → higher temperature (more uniform selection)
    temperature = jnp.clip(
        1.0 / (score_std + 1e-6),
        min_temperature,
        max_temperature
    )
    
    return temperature


def compute_hull_quality_metrics(
    original_points: Array,
    hull_vertices: Array,
    hull_indices: Array
) -> dict:
    """Compute quality metrics for approximate hull.
    
    Args:
        original_points: Original point cloud
        hull_vertices: Computed hull vertices
        hull_indices: Indices of hull vertices in original points
        
    Returns:
        Dictionary of quality metrics
    """
    n_original = original_points.shape[-2]
    n_hull = hull_vertices.shape[-2]
    
    # Coverage ratio
    coverage_ratio = n_hull / n_original
    
    # Approximation error (if we have ground truth hull)
    # For now, compute average distance from original points to hull
    # This is a simplified metric
    hull_center = jnp.mean(hull_vertices, axis=-2)
    original_center = jnp.mean(original_points, axis=-2)
    center_distance = jnp.linalg.norm(hull_center - original_center)
    
    # Compactness (ratio of hull points to total points)
    compactness = 1.0 - coverage_ratio
    
    return {
        "coverage_ratio": coverage_ratio,
        "center_distance": center_distance,
        "compactness": compactness,
        "n_hull_vertices": n_hull,
        "n_original_points": n_original
    }


def multi_resolution_hull(
    points: PointCloud,
    resolution_levels: list = [50, 100, 200],
    method: SamplingMethod = "uniform",
    random_key: Optional[Array] = None
) -> list:
    """Compute multi-resolution approximate hulls.
    
    This function computes multiple approximations with different numbers
    of directions, useful for hierarchical or adaptive algorithms.
    
    Args:
        points: Input point cloud
        resolution_levels: List of n_directions values
        method: Sampling method for all levels
        random_key: Random key for reproducibility
        
    Returns:
        List of (hull_vertices, hull_indices) tuples for each resolution
        
    Example:
        >>> points = jnp.random.normal(jax.random.PRNGKey(0), (100, 3))
        >>> hulls = multi_resolution_hull(points, [20, 50, 100])
        >>> len(hulls)  # 3 different resolutions
        3
    """
    if random_key is None:
        random_key = jax.random.PRNGKey(42)
    
    hulls = []
    
    for n_directions in resolution_levels:
        # Use different subkeys for each resolution
        subkey = jax.random.fold_in(random_key, n_directions)
        
        hull_vertices, hull_indices = approximate_convex_hull(
            points,
            n_directions=n_directions,
            method=method,
            random_key=subkey
        )
        
        hulls.append((hull_vertices, hull_indices))
    
    return hulls


def progressive_hull_refinement(
    points: PointCloud,
    initial_directions: int = 20,
    max_directions: int = 200,
    refinement_steps: int = 3,
    convergence_threshold: float = 1e-4,
    random_key: Optional[Array] = None
) -> Tuple[HullVertices, Array, dict]:
    """Progressive refinement of approximate hull.
    
    Starts with a coarse approximation and progressively refines it
    until convergence or maximum resolution is reached.
    
    Args:
        points: Input point cloud
        initial_directions: Starting number of directions
        max_directions: Maximum number of directions
        refinement_steps: Number of refinement iterations
        convergence_threshold: Threshold for convergence detection
        random_key: Random key for reproducibility
        
    Returns:
        Tuple of (final_hull_vertices, final_hull_indices, refinement_info)
    """
    if random_key is None:
        random_key = jax.random.PRNGKey(123)
    
    current_directions = initial_directions
    previous_hull = None
    refinement_info = {
        "iterations": [],
        "converged": False,
        "final_directions": current_directions
    }
    
    for step in range(refinement_steps):
        subkey = jax.random.fold_in(random_key, step)
        
        hull_vertices, hull_indices = approximate_convex_hull(
            points,
            n_directions=current_directions,
            random_key=subkey
        )
        
        # Check convergence if we have a previous hull
        if previous_hull is not None:
            # Simple convergence check: compare hull centers
            current_center = jnp.mean(hull_vertices, axis=-2)
            previous_center = jnp.mean(previous_hull, axis=-2)
            center_change = jnp.linalg.norm(current_center - previous_center)
            
            refinement_info["iterations"].append({
                "step": step,
                "directions": current_directions,
                "center_change": center_change,
                "n_vertices": hull_vertices.shape[-2]
            })
            
            if center_change < convergence_threshold:
                refinement_info["converged"] = True
                break
        
        previous_hull = hull_vertices
        
        # Increase resolution for next iteration
        current_directions = min(current_directions * 2, max_directions)
        if current_directions >= max_directions:
            break
    
    refinement_info["final_directions"] = current_directions
    
    return hull_vertices, hull_indices, refinement_info


# JIT-compiled versions for performance
approximate_convex_hull_jit = jax.jit(approximate_convex_hull, static_argnames=['method', 'remove_duplicates'])
batched_approximate_hull_jit = jax.jit(batched_approximate_hull, static_argnames=['method', 'remove_duplicates'])