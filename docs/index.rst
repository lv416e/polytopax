PolytopAX Documentation
=======================

**PolytopAX** is a JAX-based computational geometry library for differentiable convex hull computation and polytope operations. It provides efficient, GPU-accelerated algorithms for convex hull approximation with automatic differentiation support.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   user_guide/basic_usage
   user_guide/advanced_features
   user_guide/performance_tips

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index
   tutorials/basic_convex_hull
   tutorials/differentiable_optimization
   tutorials/batch_processing

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/core
   api/algorithms
   api/operations

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   roadmap

Key Features
------------

* **Differentiable Convex Hull**: Approximate convex hull computation that maintains differentiability
* **JAX Integration**: Full compatibility with JAX transformations (jit, grad, vmap, pmap)
* **GPU Acceleration**: Efficient computation on GPU and TPU through XLA compilation
* **Object-Oriented API**: Intuitive ConvexHull class with comprehensive methods
* **Functional API**: Low-level functions for custom workflows
* **Batch Processing**: Native support for batch operations
* **Geometric Predicates**: Point inclusion, volume, surface area, and distance computations

Quick Example
-------------

.. code-block:: python

   import jax.numpy as jnp
   import polytopax as ptx

   # Create some 2D points
   points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

   # Compute convex hull
   hull = ptx.ConvexHull.from_points(points)

   # Access properties
   print(f"Volume: {hull.volume()}")
   print(f"Surface area: {hull.surface_area()}")
   print(f"Centroid: {hull.centroid()}")

   # Test point inclusion
   test_point = jnp.array([0.5, 0.5])
   is_inside = hull.contains(test_point)

Installation
------------

Install PolytopAX using pip:

.. code-block:: bash

   pip install polytopax

Or install from source:

.. code-block:: bash

   git clone https://github.com/your-org/polytopax.git
   cd polytopax
   pip install -e .

Requirements
------------

* Python 3.8+
* JAX 0.4.0+
* NumPy
* SciPy (optional, for comparisons)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`