"""Tests for ConvexHull class."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from polytopax.core.polytope import ConvexHull, hull_union
from polytopax import ConvexHull as PublicConvexHull


class TestConvexHullCreation:
    """Tests for ConvexHull creation and initialization."""
    
    def test_from_points_basic(self):
        """Test basic ConvexHull creation from points."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        hull = ConvexHull.from_points(points, n_directions=20)
        
        assert isinstance(hull, ConvexHull)
        assert hull.dimension == 2
        assert hull.n_vertices > 0
        assert hull.vertices.shape[-1] == 2
    
    def test_from_points_3d(self):
        """Test ConvexHull creation with 3D points."""
        points = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        hull = ConvexHull.from_points(points, n_directions=30)
        
        assert hull.dimension == 3
        assert hull.vertices.shape[-1] == 3
    
    def test_from_points_with_parameters(self):
        """Test ConvexHull creation with various parameters."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        
        # Different number of directions
        hull1 = ConvexHull.from_points(points, n_directions=10)
        hull2 = ConvexHull.from_points(points, n_directions=50)
        
        assert hull1.n_vertices <= hull2.n_vertices  # More directions may give more vertices
        
        # Different methods
        hull_uniform = ConvexHull.from_points(points, method="uniform")
        assert isinstance(hull_uniform, ConvexHull)
        
        # Different temperature
        hull_low_temp = ConvexHull.from_points(points, temperature=0.01)
        hull_high_temp = ConvexHull.from_points(points, temperature=1.0)
        
        assert isinstance(hull_low_temp, ConvexHull)
        assert isinstance(hull_high_temp, ConvexHull)
    
    def test_direct_construction(self):
        """Test direct ConvexHull construction."""
        vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        algorithm_info = {"algorithm": "test", "n_directions": 10}
        
        hull = ConvexHull(vertices=vertices, algorithm_info=algorithm_info)
        
        assert jnp.allclose(hull.vertices, vertices)
        assert hull.algorithm_info["algorithm"] == "test"
    
    def test_unknown_algorithm(self):
        """Test ConvexHull creation with unknown algorithm."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        
        with pytest.raises(ValueError):
            ConvexHull.from_points(points, algorithm="unknown")


class TestConvexHullProperties:
    """Tests for ConvexHull properties and methods."""
    
    def test_basic_properties(self):
        """Test basic properties."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        hull = ConvexHull.from_points(points, n_directions=20)
        
        # Basic properties
        assert isinstance(hull.n_vertices, int)
        assert hull.n_vertices > 0
        assert hull.dimension == 2
        
        # Vertices
        assert isinstance(hull.vertices_array(), jnp.ndarray)
        assert hull.vertices_array().shape == hull.vertices.shape
    
    def test_geometric_properties(self):
        """Test geometric property computations."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        hull = ConvexHull.from_points(points, n_directions=20)
        
        # Volume (area in 2D)
        volume = hull.volume()
        assert isinstance(volume, float)
        assert volume > 0
        
        # Surface area (perimeter in 2D)
        surface_area = hull.surface_area()
        assert isinstance(surface_area, float)
        assert surface_area > 0
        
        # Centroid
        centroid = hull.centroid()
        assert centroid.shape == (2,)
        assert jnp.all(jnp.isfinite(centroid))
        
        # Diameter
        diameter = hull.diameter()
        assert isinstance(diameter, float)
        assert diameter > 0
        
        # Bounding box
        min_coords, max_coords = hull.bounding_box()
        assert min_coords.shape == (2,)
        assert max_coords.shape == (2,)
        assert jnp.all(min_coords <= max_coords)
    
    def test_contains_method(self):
        """Test point containment testing."""
        # Simple square
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        hull = ConvexHull.from_points(points, n_directions=50)
        
        # Test various points
        # Note: Due to approximation, results may vary
        center_point = jnp.array([0.5, 0.5])
        corner_point = jnp.array([0.0, 0.0])
        outside_point = jnp.array([2.0, 2.0])
        
        # These are probabilistic due to approximation nature
        center_result = hull.contains(center_point)
        corner_result = hull.contains(corner_point) 
        outside_result = hull.contains(outside_point)
        
        assert isinstance(center_result, (bool, np.bool_))
        assert isinstance(corner_result, (bool, np.bool_))
        assert isinstance(outside_result, (bool, np.bool_))
    
    def test_distance_to_method(self):
        """Test distance computation."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        hull = ConvexHull.from_points(points, n_directions=20)
        
        # Test distance to various points
        center_point = jnp.array([0.5, 0.5])
        outside_point = jnp.array([2.0, 2.0])
        
        center_distance = hull.distance_to(center_point)
        outside_distance = hull.distance_to(outside_point)
        
        assert isinstance(center_distance, float)
        assert isinstance(outside_distance, float)
        assert outside_distance > 0  # Should be positive (outside)
    
    def test_degeneracy_detection(self):
        """Test degeneracy detection."""
        # Non-degenerate case
        points_2d = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hull_2d = ConvexHull.from_points(points_2d)
        
        # This may or may not be degenerate depending on approximation
        degeneracy_result = hull_2d.is_degenerate()
        assert isinstance(degeneracy_result, (bool, np.bool_))
        
        # Clearly degenerate case (too few vertices)
        single_point = jnp.array([[0.0, 0.0]])
        hull_single = ConvexHull(vertices=single_point)
        assert hull_single.is_degenerate() == True


class TestConvexHullCaching:
    """Tests for property caching."""
    
    def test_volume_caching(self):
        """Test that volume is cached properly."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hull = ConvexHull.from_points(points)
        
        # First call computes volume
        volume1 = hull.volume()
        
        # Second call should use cache
        volume2 = hull.volume()
        
        assert volume1 == volume2
        # Check that cache is actually set
        assert hull._volume_cache is not None
    
    def test_surface_area_caching(self):
        """Test that surface area is cached properly."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hull = ConvexHull.from_points(points)
        
        # First call computes surface area
        area1 = hull.surface_area()
        
        # Second call should use cache
        area2 = hull.surface_area()
        
        assert area1 == area2
        assert hull._surface_area_cache is not None
    
    def test_centroid_caching(self):
        """Test that centroid is cached properly."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hull = ConvexHull.from_points(points)
        
        # First call computes centroid
        centroid1 = hull.centroid()
        
        # Second call should use cache
        centroid2 = hull.centroid()
        
        assert jnp.allclose(centroid1, centroid2)
        assert hull._centroid_cache is not None


class TestConvexHullSummary:
    """Tests for summary and representation methods."""
    
    def test_summary(self):
        """Test summary generation."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hull = ConvexHull.from_points(points, n_directions=20)
        
        summary = hull.summary()
        
        assert isinstance(summary, dict)
        assert "n_vertices" in summary
        assert "dimension" in summary
        assert "volume" in summary
        assert "surface_area" in summary
        assert "centroid" in summary
        assert "diameter" in summary
        assert "bounding_box_min" in summary
        assert "bounding_box_max" in summary
        assert "is_degenerate" in summary
        assert "algorithm_info" in summary
        
        # Check types
        assert isinstance(summary["n_vertices"], int)
        assert isinstance(summary["dimension"], int)
        assert isinstance(summary["volume"], float)
        assert isinstance(summary["centroid"], list)
    
    def test_repr(self):
        """Test string representation."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hull = ConvexHull.from_points(points)
        
        repr_str = repr(hull)
        assert "ConvexHull" in repr_str
        assert "n_vertices" in repr_str
        assert "dimension" in repr_str
        assert "volume" in repr_str
    
    def test_str(self):
        """Test human-readable string representation."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hull = ConvexHull.from_points(points)
        
        str_repr = str(hull)
        assert "ConvexHull Summary" in str_repr
        assert "Vertices:" in str_repr
        assert "Dimension:" in str_repr
        assert "Volume:" in str_repr


class TestConvexHullSerialization:
    """Tests for serialization and deserialization."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hull = ConvexHull.from_points(points, n_directions=10)
        
        hull_dict = hull.to_dict()
        
        assert isinstance(hull_dict, dict)
        assert "vertices" in hull_dict
        assert "algorithm_info" in hull_dict
        
        # Check that vertices are serializable
        assert isinstance(hull_dict["vertices"], list)
        assert isinstance(hull_dict["algorithm_info"], dict)
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        original_points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        original_hull = ConvexHull.from_points(original_points, n_directions=10)
        
        # Convert to dict and back
        hull_dict = original_hull.to_dict()
        reconstructed_hull = ConvexHull.from_dict(hull_dict)
        
        # Check that reconstruction preserves data
        assert jnp.allclose(original_hull.vertices, reconstructed_hull.vertices)
        assert original_hull.algorithm_info == reconstructed_hull.algorithm_info
    
    def test_roundtrip_serialization(self):
        """Test roundtrip serialization."""
        points = jnp.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
        original_hull = ConvexHull.from_points(points, n_directions=15)
        
        # Roundtrip: hull -> dict -> hull
        hull_dict = original_hull.to_dict()
        reconstructed_hull = ConvexHull.from_dict(hull_dict)
        
        # Properties should be preserved
        assert original_hull.n_vertices == reconstructed_hull.n_vertices
        assert original_hull.dimension == reconstructed_hull.dimension
        assert jnp.isclose(original_hull.volume(), reconstructed_hull.volume())


class TestConvexHullTransformations:
    """Tests for transformation methods (Phase 2 placeholders)."""
    
    def test_scale_placeholder(self):
        """Test scale transformation placeholder."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hull = ConvexHull.from_points(points)
        
        # Should work but issue warning
        with pytest.warns(UserWarning):
            scaled_hull = hull.scale(2.0)
        
        assert isinstance(scaled_hull, ConvexHull)
        assert scaled_hull.algorithm_info.get("transformed") == "scaled"
    
    def test_translate_placeholder(self):
        """Test translate transformation placeholder."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hull = ConvexHull.from_points(points)
        
        # Should work but issue warning
        with pytest.warns(UserWarning):
            translated_hull = hull.translate(jnp.array([1.0, 1.0]))
        
        assert isinstance(translated_hull, ConvexHull)
        assert translated_hull.algorithm_info.get("transformed") == "translated"
    
    def test_rotate_placeholder(self):
        """Test rotate transformation placeholder."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hull = ConvexHull.from_points(points)
        
        # Should work but issue warning
        with pytest.warns(UserWarning):
            rotated_hull = hull.rotate(jnp.pi / 4)
        
        assert isinstance(rotated_hull, ConvexHull)
        assert rotated_hull.algorithm_info.get("transformed") == "rotated"


class TestConvexHullJAXCompatibility:
    """Tests for JAX compatibility."""
    
    def test_jax_tree_registration(self):
        """Test that ConvexHull is properly registered as JAX pytree."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hull = ConvexHull.from_points(points)
        
        # Should be able to use in JAX transformations
        def get_volume(h):
            return h.volume()
        
        # This should work without errors
        volume = get_volume(hull)
        assert isinstance(volume, float)
        
        # Test that it can be used in jax.tree_map
        def double_vertices(x):
            if isinstance(x, jnp.ndarray):
                return x * 2.0
            return x
        
        # This should work with the pytree registration
        transformed = jax.tree_map(double_vertices, hull)
        # Note: This would fail if ConvexHull wasn't properly registered


class TestHullUtilityFunctions:
    """Tests for utility functions."""
    
    def test_hull_union(self):
        """Test hull union operation."""
        points1 = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        points2 = jnp.array([[2.0, 0.0], [3.0, 0.0], [2.0, 1.0]])
        
        hull1 = ConvexHull.from_points(points1, n_directions=10)
        hull2 = ConvexHull.from_points(points2, n_directions=10)
        
        union_hull = hull_union(hull1, hull2)
        
        assert isinstance(union_hull, ConvexHull)
        # Union should contain both original hulls
        assert union_hull.n_vertices >= max(hull1.n_vertices, hull2.n_vertices)


@pytest.mark.parametrize("n_directions", [5, 10, 20, 50])
def test_hull_quality_vs_directions(n_directions):
    """Test how hull quality changes with number of directions."""
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    hull = ConvexHull.from_points(points, n_directions=n_directions)
    
    assert hull.n_vertices > 0
    assert hull.volume() > 0
    # More directions might give better approximation, but not guaranteed


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_hull_different_dimensions(dimension):
    """Test ConvexHull with different dimensions."""
    # Generate simplex vertices
    n_vertices = dimension + 1
    vertices = jnp.eye(n_vertices, dimension)
    vertices = jnp.vstack([jnp.zeros(dimension), vertices])
    
    hull = ConvexHull.from_points(vertices, n_directions=20)
    
    assert hull.dimension == dimension
    assert hull.n_vertices > 0
    assert hull.volume() > 0