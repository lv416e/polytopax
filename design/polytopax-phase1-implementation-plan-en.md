# PolytopAX Phase 1 Implementation Plan (v0.1.0)

## Project Overview

**PolytopAX v0.1.0** serves as the foundational MVP for a differentiable convex hull computation library built on the JAX ecosystem. This version provides approximate convex hull algorithms and basic geometric operations that seamlessly integrate with machine learning pipelines.

### Development Goals

- **Differentiability**: Full compatibility with JAX's automatic differentiation
- **GPU Optimization**: High-speed computation through XLA compilation
- **Usability**: Intuitive functional and object-oriented APIs
- **Extensibility**: Architecture designed for Phase 2+ feature expansion

## Technical Architecture

### Module Structure

```
polytopax/
├── core/
│   ├── __init__.py              # Core module exports
│   ├── hull.py                  # Basic convex hull computation (existing, extended)
│   ├── utils.py                 # Common utilities【NEW】
│   └── polytope.py              # ConvexHull class【NEW】
├── algorithms/
│   ├── __init__.py
│   └── approximation.py         # Approximation algorithms【NEW】
├── operations/
│   ├── __init__.py
│   └── predicates.py            # Geometric predicates【NEW】
└── __init__.py                  # Package entry point
```

### Implementation Priority

| Priority | Module | Content | Dependencies |
|----------|---------|---------|--------------|
| **1** | `core/utils.py` | Type definitions, validation, direction vector generation | Independent |
| **2** | `algorithms/approximation.py` | Differentiable approximate convex hull | utils.py |
| **3** | `operations/predicates.py` | Geometric predicates (point inclusion, volume, surface area) | utils.py |
| **4** | `core/polytope.py` | ConvexHull class | All above |
| **5** | `core/hull.py` | Extension and refactoring of existing functions | All above |

## Detailed Technical Specifications

### 1. core/utils.py - Foundation Utilities

#### Type Definitions

```python
from typing import Union, Tuple, Optional, Literal
from jax import Array
import jax.numpy as jnp

# Type aliases
PointCloud = Array  # shape: (..., n_points, dimension)
HullVertices = Array  # shape: (n_vertices, dimension)
DirectionVectors = Array  # shape: (n_directions, dimension)
SamplingMethod = Literal["uniform", "icosphere", "adaptive"]
```

#### Core Functions

```python
def validate_point_cloud(points: Array) -> Array:
    """Validate point cloud shape and numerical validity
    
    Args:
        points: Input point cloud with shape (..., n_points, dim)
        
    Returns:
        Validated point cloud
        
    Raises:
        ValueError: Invalid shape or numerical values
    """

def generate_direction_vectors(
    dimension: int,
    n_directions: int,
    method: SamplingMethod = "uniform",
    random_key: Optional[Array] = None
) -> DirectionVectors:
    """Generate direction vectors for sampling
    
    Args:
        dimension: Spatial dimension
        n_directions: Number of directions to generate
        method: Sampling strategy
            - "uniform": Uniform distribution on sphere
            - "icosphere": Icosahedral subdivision (3D only)
            - "adaptive": Locally adaptive density sampling
        random_key: JAX random key
        
    Returns:
        Normalized direction vector set
    """

def robust_orientation_test(
    points: Array,
    tolerance: float = 1e-12
) -> Array:
    """Robust geometric orientation test
    
    Implements numerically stable orientation tests
    Based on Shewchuk (1997) adaptive precision predicates
    """
```

### 2. algorithms/approximation.py - Differentiable Approximate Convex Hull

#### Core Algorithm

**Direction Vector Sampling Method**:
1. Generate specified direction vector set
2. Compute farthest point in each direction (differentiable)
3. Remove duplicates and cleanup

```python
def approximate_convex_hull(
    points: PointCloud,
    n_directions: int = 100,
    method: SamplingMethod = "uniform",
    temperature: float = 0.1,
    random_key: Optional[Array] = None
) -> Tuple[HullVertices, Array]:
    """Differentiable approximate convex hull computation
    
    Args:
        points: Point cloud with shape (..., n_points, dim)
        n_directions: Number of sampling directions
        method: Sampling strategy
        temperature: Softmax temperature (differentiability control)
        random_key: Random key
        
    Returns:
        (hull_vertices, hull_indices): Hull vertices and original indices
        
    Algorithm:
        1. Direction vector generation: generate_direction_vectors()
        2. Direction-wise farthest point search:
           scores = jnp.dot(points, directions.T)  # shape: (n_points, n_directions)
           weights = jax.nn.softmax(scores / temperature, axis=0)
           soft_points = jnp.sum(weights[..., None] * points, axis=0)
        3. Duplicate removal: unique_vertices_removal()
    """

def batched_approximate_hull(
    batch_points: Array,
    **kwargs
) -> Tuple[Array, Array]:
    """Batch processing version
    
    Args:
        batch_points: shape (batch_size, n_points, dim)
        
    Returns:
        Batched convex hull results
        
    Implementation:
        return jax.vmap(approximate_convex_hull, in_axes=(0,))(batch_points, **kwargs)
    """
```

#### Numerical Stability

```python
def soft_argmax_selection(
    scores: Array,
    temperature: float,
    points: Array
) -> Array:
    """Differentiable farthest point selection
    
    Traditional argmax is non-differentiable:
        idx = jnp.argmax(scores)  # Non-differentiable
        
    Soft selection solution:
        weights = softmax(scores / temperature)
        soft_point = sum(weights * points)  # Differentiable
    """
    
def adaptive_temperature_control(
    scores: Array,
    target_sparsity: float = 0.1
) -> float:
    """Adaptive temperature control
    
    Dynamically adjust temperature to maintain appropriate sparsity
    """
```

### 3. operations/predicates.py - Geometric Predicates

#### Basic Predicates

```python
def point_in_convex_hull(
    point: Array,
    hull_vertices: HullVertices,
    tolerance: float = 1e-8
) -> bool:
    """Point-in-convex-hull inclusion test
    
    Algorithm: Linear programming implementation
        Determine if point can be expressed as convex combination of hull vertices
        sum(λᵢ * vᵢ) = point, sum(λᵢ) = 1, λᵢ >= 0
    """

def convex_hull_volume(
    vertices: HullVertices,
    method: str = "simplex_decomposition"
) -> float:
    """Convex hull volume computation (differentiable)
    
    Methods:
        - "simplex_decomposition": Integration via simplex decomposition
        - "monte_carlo": Monte Carlo estimation (for large-scale data)
        
    Returns:
        Volume value (d-dimensional measure)
    """

def convex_hull_surface_area(
    vertices: HullVertices,
    faces: Optional[Array] = None
) -> float:
    """Convex hull surface area computation
    
    Args:
        vertices: Hull vertices
        faces: Face vertex indices (auto-computed if None)
        
    Algorithm:
        1. Face extraction (convex_hull_faces())
        2. Area computation for each face
        3. Sum total
    """
```

#### Distance Computations

```python
def distance_to_convex_hull(
    point: Array,
    hull_vertices: HullVertices
) -> float:
    """Shortest distance from point to convex hull
    
    Returns:
        distance: Positive (exterior), 0 (boundary), negative (interior)
    """

def hausdorff_distance(
    hull1: HullVertices,
    hull2: HullVertices
) -> float:
    """Hausdorff distance between two convex hulls"""
```

### 4. core/polytope.py - ConvexHull Class

#### Class Design

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
import jax

@dataclass
class ConvexHull:
    """JAX-compatible ConvexHull class
    
    Attributes:
        vertices: Hull vertex coordinates
        faces: Face composition (optional)
        algorithm_info: Computation metadata
        _volume_cache: Volume computation cache
    """
    vertices: HullVertices
    faces: Optional[Array] = None
    algorithm_info: Dict[str, Any] = None
    _volume_cache: Optional[float] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.algorithm_info is None:
            self.algorithm_info = {}
        # JAX tree registration
        jax.tree_util.register_pytree_node(
            ConvexHull,
            self._tree_flatten,
            self._tree_unflatten
        )
    
    # JAX tree compatibility
    def _tree_flatten(self):
        children = (self.vertices, self.faces)
        aux_data = (self.algorithm_info, self._volume_cache)
        return children, aux_data
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        vertices, faces = children
        algorithm_info, volume_cache = aux_data
        return cls(vertices, faces, algorithm_info, volume_cache)
```

#### Method Implementation

```python
class ConvexHull:
    # ... (above __init__ section)
    
    def volume(self) -> float:
        """Volume computation (with caching)"""
        if self._volume_cache is None:
            self._volume_cache = convex_hull_volume(self.vertices)
        return self._volume_cache
    
    def surface_area(self) -> float:
        """Surface area computation"""
        return convex_hull_surface_area(self.vertices, self.faces)
    
    def contains(self, point: Array) -> bool:
        """Point inclusion test"""
        return point_in_convex_hull(point, self.vertices)
    
    def distance_to(self, point: Array) -> float:
        """Distance to point"""
        return distance_to_convex_hull(point, self.vertices)
    
    def centroid(self) -> Array:
        """Centroid computation"""
        return jnp.mean(self.vertices, axis=0)
    
    def bounding_box(self) -> Tuple[Array, Array]:
        """Bounding box"""
        min_coords = jnp.min(self.vertices, axis=0)
        max_coords = jnp.max(self.vertices, axis=0)
        return min_coords, max_coords
    
    # Future method chaining (Phase 2 implementation)
    # def scale(self, factor: Union[float, Array]) -> 'ConvexHull': ...
    # def translate(self, vector: Array) -> 'ConvexHull': ...
    # def rotate(self, angle: float, axis: Array = None) -> 'ConvexHull': ...
```

### 5. core/hull.py - Extension of Existing Code

#### Unified API

```python
def convex_hull(
    points: PointCloud,
    algorithm: str = "approximate",
    **kwargs
) -> HullVertices:
    """Unified convex hull computation interface
    
    Args:
        points: Input point cloud
        algorithm: Algorithm selection
            - "approximate": Differentiable approximation (default)
            - "quickhull": Exact Quickhull (Phase 2 implementation)
            - "graham_scan": 2D Graham scan (Phase 2 implementation)
        **kwargs: Algorithm-specific parameters
        
    Returns:
        Hull vertex array
    """
    if algorithm == "approximate":
        hull_vertices, _ = approximate_convex_hull(points, **kwargs)
        return hull_vertices
    else:
        raise NotImplementedError(f"Algorithm '{algorithm}' not implemented in v0.1.0")

# Maintain backward compatibility of existing functions
def approximate_convex_hull(
    points: Array,
    n_directions: int = 100,
    method: str = "uniform",
    random_seed: int = 0
) -> Tuple[Array, Array]:
    """Maintain existing signature"""
    # Forward to new implementation
    from ..algorithms.approximation import approximate_convex_hull as new_impl
    key = jax.random.PRNGKey(random_seed) if random_seed else None
    return new_impl(points, n_directions, method, random_key=key)
```

## API Design and Use Cases

### Functional API (Low-level)

```python
import polytopax as ptx
import jax.numpy as jnp

# Basic convex hull computation
points = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]])
hull_vertices = ptx.convex_hull(points, algorithm='approximate')

# Parameter adjustment
hull_vertices = ptx.approximate_convex_hull(
    points,
    n_directions=200,
    method='icosphere',
    temperature=0.05
)

# Batch processing
batch_points = jnp.array([...])  # shape: (batch_size, n_points, dim)
batch_hulls = jax.vmap(ptx.convex_hull)(batch_points)

# Basic geometric tests
is_inside = ptx.point_in_hull(test_point, hull_vertices)
volume = ptx.hull_volume(hull_vertices)
surface_area = ptx.hull_surface_area(hull_vertices)
```

### Object-Oriented API (High-level)

```python
from polytopax import ConvexHull
import jax.numpy as jnp

# ConvexHull object creation
points = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]])
hull = ConvexHull.from_points(points, algorithm='approximate', n_directions=100)

# Geometric property access
print(f"Volume: {hull.volume():.6f}")
print(f"Surface area: {hull.surface_area():.6f}")
print(f"Centroid: {hull.centroid()}")
print(f"Number of vertices: {len(hull.vertices)}")

# Geometric tests
test_point = jnp.array([0.5, 0.5])
contains_point = hull.contains(test_point)
distance = hull.distance_to(test_point)

# Bounding box
min_coords, max_coords = hull.bounding_box()
```

### Machine Learning Integration Example

```python
import jax
import jax.numpy as jnp
import polytopax as ptx

def neural_polytope_layer(points, weights):
    """Convex hull computation as neural network layer"""
    # Affine transformation
    transformed_points = jnp.dot(points, weights)
    
    # Differentiable convex hull computation
    hull_vertices, _ = ptx.approximate_convex_hull(
        transformed_points,
        n_directions=50,
        temperature=0.1
    )
    
    # Return volume as feature
    return ptx.hull_volume(hull_vertices)

# Gradient computation
volume_grad = jax.grad(neural_polytope_layer, argnums=1)

# JIT compilation
jit_layer = jax.jit(neural_polytope_layer)
```

## JAX Integration and Performance Optimization

### JAX Transform Support

```python
# JIT compilation
@jax.jit
def fast_hull_computation(points):
    return ptx.convex_hull(points, algorithm='approximate')

# Automatic differentiation
@jax.grad
def hull_volume_gradient(points):
    hull_vertices, _ = ptx.approximate_convex_hull(points)
    return ptx.hull_volume(hull_vertices)

# Vectorization
batched_computation = jax.vmap(fast_hull_computation)

# Parallelization (multi-device)
@jax.pmap
def parallel_hull_computation(points_shards):
    return jax.vmap(ptx.convex_hull)(points_shards)
```

### Optimization Strategies

1. **XLA Optimization**
   - Compilation optimization through pure JAX implementation
   - Efficient memory access via fused kernels

2. **Numerical Stability**
   - Implementation of robust geometric predicates
   - Degenerate case handling through adaptive precision

3. **Memory Efficiency**
   - Utilization of in-place operations
   - Reduction of unnecessary intermediate arrays

## Testing Strategy

### Test Categories

```python
# 1. Unit tests
class TestUtils:
    def test_validate_point_cloud(self): ...
    def test_generate_direction_vectors(self): ...
    def test_robust_orientation_test(self): ...

class TestApproximation:
    def test_approximate_convex_hull_2d(self): ...
    def test_approximate_convex_hull_3d(self): ...
    def test_batched_computation(self): ...

class TestPredicates:
    def test_point_in_convex_hull(self): ...
    def test_convex_hull_volume(self): ...
    def test_distance_computation(self): ...

class TestConvexHull:
    def test_from_points_creation(self): ...
    def test_geometric_properties(self): ...
    def test_jax_tree_compatibility(self): ...

# 2. JAX transformation tests
class TestJAXIntegration:
    def test_jit_compilation(self): ...
    def test_gradient_computation(self): ...
    def test_vmap_batching(self): ...
    def test_pmap_parallelization(self): ...

# 3. Numerical accuracy tests
class TestNumericalAccuracy:
    def test_known_geometries(self):
        """Accuracy verification with known geometries"""
        # Square, cube, sphere, etc.
        
    def test_degenerate_cases(self):
        """Degenerate case handling"""
        # Collinear points, coplanar points, duplicate points, etc.

# 4. Performance tests
class TestPerformance:
    def test_scalability(self): ...
    def test_memory_usage(self): ...
    def test_gpu_acceleration(self): ...
```

### Numerical Accuracy Verification

```python
def test_numerical_accuracy():
    """Accuracy verification through comparison with known solutions"""
    
    # 2D square
    square_points = jnp.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    hull = ConvexHull.from_points(square_points)
    assert abs(hull.volume() - 1.0) < 1e-6
    assert abs(hull.surface_area() - 4.0) < 1e-6
    
    # 3D cube
    cube_vertices = jnp.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])
    hull = ConvexHull.from_points(cube_vertices)
    assert abs(hull.volume() - 1.0) < 1e-6
    assert abs(hull.surface_area() - 6.0) < 1e-6
```

## Performance Goals and Evaluation Metrics

### Quantitative Targets

| Metric | Target Value | Evaluation Method |
|--------|--------------|-------------------|
| **Computation Speed** | 2-5x faster than SciPy | Processing time comparison with 1K-100K points |
| **Memory Efficiency** | Linear scaling | Memory usage vs. point count increase |
| **Numerical Accuracy** | Relative error < 1e-6 | Comparison with analytical solutions |
| **GPU Acceleration** | 10-50x faster than CPU | Large dataset comparison |
| **Compilation Time** | < 5 seconds | Initial JIT compilation time |

### Benchmark Environment

```python
import time
import psutil
import jax.numpy as jnp
from scipy.spatial import ConvexHull as ScipyHull

def benchmark_comparison():
    """SciPy vs PolytopAX performance comparison"""
    
    sizes = [100, 1000, 10000, 100000]
    dimensions = [2, 3, 4, 5]
    
    results = {}
    
    for n_points in sizes:
        for dim in dimensions:
            # Test data generation
            points = jax.random.normal(
                jax.random.PRNGKey(42),
                (n_points, dim)
            )
            
            # SciPy measurement
            start_time = time.time()
            scipy_hull = ScipyHull(points)
            scipy_time = time.time() - start_time
            
            # PolytopAX measurement
            start_time = time.time()
            ptx_hull = ptx.convex_hull(points, algorithm='approximate')
            ptx_time = time.time() - start_time
            
            results[(n_points, dim)] = {
                'scipy_time': scipy_time,
                'polytopax_time': ptx_time,
                'speedup': scipy_time / ptx_time
            }
    
    return results
```

## Development Schedule and Milestones

### Week 1-2: Foundation Implementation
- [ ] Implement `core/utils.py`
- [ ] Create basic test suite
- [ ] Set up CI/CD pipeline

### Week 3-4: Algorithm Implementation
- [ ] Implement `algorithms/approximation.py`
- [ ] Verify differentiability
- [ ] Add batch processing support

### Week 5-6: Geometric Predicates Implementation
- [ ] Implement `operations/predicates.py`
- [ ] Ensure numerical stability
- [ ] Add distance computation features

### Week 7-8: ConvexHull Class
- [ ] Implement `core/polytope.py`
- [ ] JAX tree compatibility
- [ ] Object-oriented API

### Week 9-10: Integration and Testing
- [ ] Extend `core/hull.py`
- [ ] Comprehensive test suite
- [ ] Performance benchmarks

### Week 11-12: Documentation and Quality Assurance
- [ ] Create API documentation
- [ ] Usage examples and tutorials
- [ ] Release preparation

## Risk Management and Mitigation

### Technical Risks

1. **Numerical Stability Challenges**
   - **Risk**: Unexpected results due to floating-point errors
   - **Mitigation**: Implementation of robust geometric predicates, adaptive precision

2. **JAX Constraint Compliance**
   - **Risk**: Implementation difficulty due to JAX's functional constraints
   - **Mitigation**: Incremental implementation, extensive test cases

3. **Performance Goal Shortfall**
   - **Risk**: Insufficient performance advantage over SciPy
   - **Mitigation**: Profiling-driven optimization, XLA optimization utilization

### Project Risks

1. **Implementation Scope Expansion**
   - **Risk**: Development timeline extension due to feature additions
   - **Mitigation**: Focus on MVP, defer features to Phase 2

2. **Quality Assurance Deficiency**
   - **Risk**: Latent bugs due to insufficient testing
   - **Mitigation**: TDD approach, continuous testing

## Success Criteria

### Release Standards

- [ ] All unit tests pass (coverage > 90%)
- [ ] JAX transforms (jit, grad, vmap) function properly
- [ ] Benchmark goals achieved (2x+ faster than SciPy)
- [ ] Numerical accuracy standards met (relative error < 1e-6)
- [ ] Complete documentation

### Technical KPIs

- **Code Quality**: Zero mypy, ruff warnings
- **Test Coverage**: 90%+ coverage
- **Performance Improvement**: 2-5x faster than SciPy
- **Memory Efficiency**: Maintain linear scaling
- **GPU Utilization**: 10x+ acceleration on large data

This implementation plan provides a systematic roadmap for Phase 1 development, aiming for the v0.1.0 release.