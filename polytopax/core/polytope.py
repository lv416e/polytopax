"""ConvexHull class for object-oriented API."""

import contextlib
import warnings
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

# Removed direct imports to avoid circular dependency
# from ..algorithms.approximation import approximate_convex_hull
# from ..operations.predicates import (...) - Using lazy imports instead
from .utils import HullVertices, PointCloud, validate_point_cloud


@dataclass
class ConvexHull:
    """JAX-compatible ConvexHull class with object-oriented API.

    This class provides a high-level interface for convex hull operations,
    including geometric queries, property computation, and transformations.
    It is designed to be compatible with JAX transformations (jit, grad, vmap).

    Attributes:
        vertices: Hull vertex coordinates with shape (n_vertices, dim)
        faces: Face composition with shape (n_faces, vertices_per_face) [optional]
        algorithm_info: Metadata about the computation algorithm
        _volume_cache: Cached volume value for performance
        _surface_area_cache: Cached surface area value for performance
        _centroid_cache: Cached centroid value for performance

    Example:
        >>> import jax.numpy as jnp
        >>> from polytopax import ConvexHull
        >>> points = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        >>> hull = ConvexHull.from_points(points)
        >>> print(f"Volume: {hull.volume():.4f}")
        >>> print(f"Centroid: {hull.centroid()}")
    """
    vertices: HullVertices
    faces: Array | None = None
    algorithm_info: dict[str, Any] = field(default_factory=dict)
    _volume_cache: float | None = field(default=None, init=False)
    _surface_area_cache: float | None = field(default=None, init=False)
    _centroid_cache: Array | None = field(default=None, init=False)

    def __post_init__(self):
        """Post-initialization processing."""
        # Validate vertices
        self.vertices = validate_point_cloud(self.vertices)

        # Ensure algorithm_info is a proper dict
        if self.algorithm_info is None:
            self.algorithm_info = {}

        # Register as JAX pytree for compatibility with transformations
        self._register_pytree()

    def _register_pytree(self):
        """Register ConvexHull as JAX pytree for transformation compatibility."""
        def _tree_flatten(self):
            # Children are the arrays that should be transformed
            children = (self.vertices, self.faces)
            # Auxiliary data includes everything else
            aux_data = {
                'algorithm_info': self.algorithm_info,
                '_volume_cache': self._volume_cache,
                '_surface_area_cache': self._surface_area_cache,
                '_centroid_cache': self._centroid_cache
            }
            return children, aux_data

        def _tree_unflatten(aux_data, children):
            vertices, faces = children
            # Create ConvexHull with basic parameters
            hull = ConvexHull(
                vertices=vertices,
                faces=faces,
                algorithm_info=aux_data['algorithm_info']
            )
            # Restore cache fields
            for k, v in aux_data.items():
                if k.startswith('_'):
                    setattr(hull, k, v)
            return hull

        # Register the pytree only if not already registered
        try:
            jax.tree_util.register_pytree_node(
                ConvexHull,
                _tree_flatten,
                _tree_unflatten
            )
        except ValueError as e:
            if "Duplicate custom PyTreeDef type registration" in str(e):
                # Already registered, ignore
                pass
            else:
                # Re-raise other ValueError
                raise

    @classmethod
    def from_points(
        cls,
        points: PointCloud,
        algorithm: str = "approximate",
        **kwargs
    ) -> "ConvexHull":
        """Create ConvexHull from point cloud.

        Args:
            points: Input point cloud with shape (n_points, dim)
            algorithm: Hull computation algorithm ("approximate")
            **kwargs: Algorithm-specific parameters

        Returns:
            ConvexHull instance

        Example:
            >>> points = jnp.random.normal(jax.random.PRNGKey(0), (50, 3))
            >>> hull = ConvexHull.from_points(points, n_directions=100)
        """
        points = validate_point_cloud(points)

        if algorithm == "approximate":
            # Lazy import to avoid circular dependency
            from ..algorithms.approximation import approximate_convex_hull
            hull_vertices, hull_indices = approximate_convex_hull(points, **kwargs)
            algorithm_info = {
                "algorithm": algorithm,
                "n_original_points": points.shape[-2],
                "n_hull_vertices": hull_vertices.shape[-2],
                **kwargs
            }
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        return cls(
            vertices=hull_vertices,
            algorithm_info=algorithm_info
        )

    @property
    def n_vertices(self) -> int:
        """Number of hull vertices."""
        return self.vertices.shape[-2]

    @property
    def dimension(self) -> int:
        """Spatial dimension."""
        return self.vertices.shape[-1]

    def volume(self, method: str = "simplex_decomposition") -> float:
        """Compute hull volume (with caching).

        Args:
            method: Volume computation method

        Returns:
            Volume of the convex hull
        """
        cache_key = f"volume_{method}"
        if cache_key not in self.__dict__ or self.__dict__[cache_key] is None:
            # Lazy import to avoid circular dependency
            from ..operations.predicates import convex_hull_volume
            volume_value = convex_hull_volume(self.vertices, method=method)
            with contextlib.suppress(TypeError, ValueError):
                # Convert to Python scalar only if not in JAX transformation
                volume_value = float(volume_value)
            setattr(self, cache_key, volume_value)

        return getattr(self, cache_key)

    def surface_area(self) -> float:
        """Compute hull surface area (with caching).

        Returns:
            Surface area of the convex hull
        """
        if self._surface_area_cache is None:
            # Lazy import to avoid circular dependency
            from ..operations.predicates import convex_hull_surface_area
            surface_area_value = convex_hull_surface_area(self.vertices, self.faces)
            with contextlib.suppress(TypeError, ValueError):
                # Convert to Python scalar only if not in JAX transformation
                surface_area_value = float(surface_area_value)
            self._surface_area_cache = surface_area_value
        return self._surface_area_cache

    def contains(self, point: Array, tolerance: float = 1e-8) -> bool:
        """Test if point is inside hull.

        Args:
            point: Point to test with shape (dim,)
            tolerance: Numerical tolerance

        Returns:
            True if point is inside or on boundary
        """
        # Lazy import to avoid circular dependency
        from ..operations.predicates import point_in_convex_hull
        return bool(point_in_convex_hull(point, self.vertices, tolerance))

    def distance_to(self, point: Array) -> float:
        """Compute distance from point to hull.

        Args:
            point: Point with shape (dim,)

        Returns:
            Signed distance (positive=outside, negative=inside)
        """
        # Lazy import to avoid circular dependency
        from ..operations.predicates import distance_to_convex_hull
        return float(distance_to_convex_hull(point, self.vertices))

    def centroid(self) -> Array:
        """Compute hull centroid (with caching).

        Returns:
            Centroid coordinates
        """
        if self._centroid_cache is None:
            self._centroid_cache = jnp.mean(self.vertices, axis=-2)
        return self._centroid_cache

    def bounding_box(self) -> tuple[Array, Array]:
        """Compute axis-aligned bounding box.

        Returns:
            Tuple of (min_coords, max_coords)
        """
        min_coords = jnp.min(self.vertices, axis=-2)
        max_coords = jnp.max(self.vertices, axis=-2)
        return min_coords, max_coords

    def diameter(self) -> float:
        """Compute hull diameter (maximum distance between vertices).

        Returns:
            Maximum distance between any two vertices
        """
        # Compute pairwise distances
        diff = self.vertices[:, None, :] - self.vertices[None, :, :]
        distances = jnp.linalg.norm(diff, axis=-1)
        return float(jnp.max(distances))

    def is_degenerate(self, tolerance: float = 1e-10) -> bool:
        """Check if hull is degenerate (lower-dimensional).

        Args:
            tolerance: Tolerance for degeneracy detection

        Returns:
            True if hull is degenerate
        """
        if self.n_vertices < self.dimension + 1:
            return True

        # Check if vertices are linearly independent
        centered_vertices = self.vertices - self.centroid()

        try:
            # Use SVD to check rank
            _, s, _ = jnp.linalg.svd(centered_vertices.T, full_matrices=False)
            rank = jnp.sum(s > tolerance)
            return int(rank) < self.dimension
        except jnp.linalg.LinAlgError:
            return True

    def summary(self) -> dict[str, Any]:
        """Get summary statistics of the hull.

        Returns:
            Dictionary containing various hull properties
        """
        min_coords, max_coords = self.bounding_box()

        summary_dict = {
            "n_vertices": self.n_vertices,
            "dimension": self.dimension,
            "volume": self.volume(),
            "surface_area": self.surface_area(),
            "centroid": self.centroid().tolist(),
            "diameter": self.diameter(),
            "bounding_box_min": min_coords.tolist(),
            "bounding_box_max": max_coords.tolist(),
            "is_degenerate": self.is_degenerate(),
            "algorithm_info": self.algorithm_info
        }

        return summary_dict

    def vertices_array(self) -> Array:
        """Get vertices as JAX array.

        Returns:
            Vertex coordinates array
        """
        return self.vertices

    def to_dict(self) -> dict[str, Any]:
        """Convert hull to dictionary representation.

        Returns:
            Dictionary representation suitable for serialization
        """
        result = {
            "vertices": self.vertices.tolist(),
            "algorithm_info": self.algorithm_info
        }

        if self.faces is not None:
            result["faces"] = self.faces.tolist()

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConvexHull":
        """Create ConvexHull from dictionary representation.

        Args:
            data: Dictionary containing hull data

        Returns:
            ConvexHull instance
        """
        vertices = jnp.array(data["vertices"])
        faces = None
        if "faces" in data:
            faces = jnp.array(data["faces"])

        algorithm_info = data.get("algorithm_info", {})

        return cls(
            vertices=vertices,
            faces=faces,
            algorithm_info=algorithm_info
        )

    def __repr__(self) -> str:
        """String representation of ConvexHull."""
        return (
            f"ConvexHull(n_vertices={self.n_vertices}, "
            f"dimension={self.dimension}, "
            f"volume={self.volume():.6f})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        summary = self.summary()
        lines = [
            "ConvexHull Summary:",
            f"  Vertices: {summary['n_vertices']}",
            f"  Dimension: {summary['dimension']}",
            f"  Volume: {summary['volume']:.6f}",
            f"  Surface Area: {summary['surface_area']:.6f}",
            f"  Centroid: {summary['centroid']}",
            f"  Diameter: {summary['diameter']:.6f}",
            f"  Algorithm: {summary['algorithm_info'].get('algorithm', 'unknown')}"
        ]
        return "\n".join(lines)

    # Future methods for Phase 2 (transformations)
    # These are placeholder methods for future implementation

    def scale(self, factor: float | Array) -> "ConvexHull":
        """Scale the convex hull (Phase 2 implementation).

        Args:
            factor: Scaling factor (scalar or per-dimension)

        Returns:
            New scaled ConvexHull instance
        """
        warnings.warn(
            "Scale transformation not implemented in Phase 1",
            UserWarning, stacklevel=2
        )
        # Placeholder implementation
        scaled_vertices = self.vertices * factor if jnp.isscalar(factor) else self.vertices * factor[None, :]

        return ConvexHull(
            vertices=scaled_vertices,
            algorithm_info={**self.algorithm_info, "transformed": "scaled"}
        )

    def translate(self, vector: Array) -> "ConvexHull":
        """Translate the convex hull (Phase 2 implementation).

        Args:
            vector: Translation vector

        Returns:
            New translated ConvexHull instance
        """
        warnings.warn(
            "Translate transformation not implemented in Phase 1",
            UserWarning, stacklevel=2
        )
        # Placeholder implementation
        translated_vertices = self.vertices + vector[None, :]

        return ConvexHull(
            vertices=translated_vertices,
            algorithm_info={**self.algorithm_info, "transformed": "translated"}
        )

    def rotate(self, angle: float, axis: Array | None = None) -> "ConvexHull":
        """Rotate the convex hull (Phase 2 implementation).

        Args:
            angle: Rotation angle (radians)
            axis: Rotation axis (3D only)

        Returns:
            New rotated ConvexHull instance
        """
        warnings.warn(
            "Rotate transformation not implemented in Phase 1",
            UserWarning, stacklevel=2
        )
        # Placeholder - just return copy for now
        return ConvexHull(
            vertices=self.vertices,
            faces=self.faces,
            algorithm_info={**self.algorithm_info, "transformed": "rotated"}
        )


# Utility functions for working with ConvexHull objects

def hull_intersection(hull1: ConvexHull, hull2: ConvexHull) -> ConvexHull:
    """Compute intersection of two convex hulls (Phase 2 implementation).

    Args:
        hull1: First convex hull
        hull2: Second convex hull

    Returns:
        ConvexHull representing the intersection
    """
    warnings.warn(
        "Hull intersection not implemented in Phase 1",
        UserWarning, stacklevel=2
    )
    # Placeholder - return first hull for now
    return hull1


def hull_union(hull1: ConvexHull, hull2: ConvexHull) -> ConvexHull:
    """Compute union (convex hull) of two convex hulls.

    Args:
        hull1: First convex hull
        hull2: Second convex hull

    Returns:
        ConvexHull representing the union
    """
    # Combine vertices and compute new hull
    combined_vertices = jnp.concatenate([hull1.vertices, hull2.vertices], axis=0)

    return ConvexHull.from_points(
        combined_vertices,
        algorithm="approximate",
        n_directions=max(100, combined_vertices.shape[0] // 2)
    )


def minkowski_sum(hull1: ConvexHull, hull2: ConvexHull) -> ConvexHull:
    """Compute Minkowski sum of two convex hulls (Phase 2 implementation).

    Args:
        hull1: First convex hull
        hull2: Second convex hull

    Returns:
        ConvexHull representing the Minkowski sum
    """
    warnings.warn(
        "Minkowski sum not implemented in Phase 1",
        UserWarning, stacklevel=2
    )
    # Placeholder - return union for now
    return hull_union(hull1, hull2)


# JAX-compatible functions for batch operations

def batch_hull_volumes(hulls: list) -> Array:
    """Compute volumes for a batch of hulls.

    Args:
        hulls: List of ConvexHull objects

    Returns:
        Array of volumes
    """
    return jnp.array([hull.volume() for hull in hulls])


def batch_hull_contains(hulls: list, points: Array) -> Array:
    """Test point containment for a batch of hulls.

    Args:
        hulls: List of ConvexHull objects
        points: Points to test with shape (n_points, dim)

    Returns:
        Boolean array with shape (n_hulls, n_points)
    """
    results = []
    for hull in hulls:
        hull_results = jnp.array([hull.contains(point) for point in points])
        results.append(hull_results)

    return jnp.array(results)
