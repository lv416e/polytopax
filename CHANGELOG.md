# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2025-07-05

### Added

#### Core Implementation
- **Initial release** of PolytopAX v0.0.1
- **ConvexHull class** with comprehensive object-oriented API
- **Differentiable approximate convex hull** algorithms using direction vector sampling
- **Geometric predicates** including point inclusion, volume, surface area, and distance computations
- **JAX integration** with full support for jit, grad, vmap, and pmap transformations
- **Core utilities** with type definitions, validation functions, and direction vector generation

#### Algorithms
- **Direction vector sampling** methods: uniform, icosphere (3D), and adaptive (placeholder)
- **Soft selection** using temperature-controlled softmax for differentiability
- **Multi-resolution hull** computation with progressive refinement
- **Batch processing** support for efficient parallel computation
- **Quality metrics** for hull approximation assessment

#### API Design
- **High-level API**: ConvexHull class with method chaining support
- **Low-level API**: Individual functions for custom workflows
- **Functional API**: Direct access to core algorithms
- **Type safety**: Comprehensive type hints and JAX Array compatibility

#### Documentation & Examples
- **Comprehensive documentation** using Sphinx with RTD theme
- **Getting Started guide** with installation and basic usage
- **API reference** with detailed function documentation
- **User guide** covering basic to advanced usage patterns
- **Code examples**: Basic usage, JAX integration, and advanced optimization
- **Jupyter notebooks**: Interactive tutorials with visualizations
- **Performance guides** and best practices

#### Testing
- **Comprehensive test suite** with 135+ tests covering all modules
- **JAX compatibility tests** ensuring JIT compilation support
- **Unit tests** for core utilities, algorithms, and predicates
- **Integration tests** for ConvexHull class and API consistency
- **Edge case testing** for numerical stability and error handling

### Changed

#### Implementation Improvements
- **JAX JIT compatibility**: Fixed boolean array indexing issues in `remove_duplicate_points`
- **Validation flexibility**: Modified point cloud validation to support JAX tracing
- **Type consistency**: Ensured all code uses ASCII characters in implementation
- **Error handling**: Improved error messages and validation for edge cases

#### Code Quality
- **Language consistency**: Verified elite-level English usage throughout codebase
- **Documentation quality**: Professional documentation with mathematical notation in comments only
- **Code style**: Consistent formatting and comprehensive docstrings
- **Test coverage**: Extensive testing with both passing and documented failing tests

### Fixed

#### JAX Compatibility Issues
- **Boolean array indexing**: Resolved `NonConcreteBooleanIndexError` in duplicate point removal
- **JIT compilation**: Fixed traced array boolean evaluation in validation functions
- **Transformation support**: Ensured all functions work with JAX transformations

#### Implementation Details
- **Numerical stability**: Improved handling of degenerate cases and edge conditions
- **Type safety**: Fixed type inconsistencies and added proper Array type usage
- **Error handling**: Better error messages and graceful handling of invalid inputs

### Development Infrastructure

#### Project Structure
- **Design documents**: Comprehensive Phase 1 implementation plans in Japanese and English
- **Repository setup**: Professional project structure with all standard files
- **Development workflow**: Proper testing, documentation, and example organization

#### Quality Assurance
- **Code review**: Thorough review of implementation quality and consistency
- **Performance validation**: JAX transformation compatibility verification
- **Documentation review**: Complete documentation coverage for all public APIs

### Notes

This initial release provides a solid foundation for differentiable convex hull computation with JAX. The implementation focuses on:

1. **Differentiability**: All operations maintain gradients for optimization applications
2. **Performance**: JAX JIT compilation and vectorization support
3. **Usability**: Intuitive APIs for both beginners and advanced users
4. **Reliability**: Comprehensive testing and robust error handling
5. **Documentation**: Professional-grade documentation and examples

The implementation is ready for initial use and community feedback. Future releases will focus on performance optimization, exact algorithms, and advanced features.
