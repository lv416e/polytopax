API Reference
=============

This section contains the complete API reference for PolytopAX.

Overview
--------

PolytopAX provides both high-level object-oriented APIs and low-level functional APIs:

* **High-level API**: :class:`polytopax.ConvexHull` class for easy-to-use convex hull operations
* **Low-level API**: Individual functions for custom workflows and advanced usage
* **Utilities**: Helper functions for validation, preprocessing, and geometric computations

Quick Reference
---------------

.. currentmodule:: polytopax

**Main Classes:**

.. autosummary::
   :toctree: generated/

   ConvexHull

**Core Functions:**

.. autosummary::
   :toctree: generated/

   convex_hull
   approximate_convex_hull

**Geometric Predicates:**

.. autosummary::
   :toctree: generated/

   point_in_convex_hull
   convex_hull_volume
   convex_hull_surface_area
   distance_to_convex_hull
   hausdorff_distance

**Utility Functions:**

.. autosummary::
   :toctree: generated/

   get_info
   validate_point_cloud
   generate_direction_vectors

Detailed Documentation
----------------------

.. toctree::
   :maxdepth: 2

   core
   algorithms
   operations

Module Structure
----------------

The PolytopAX library is organized into several modules:

* :mod:`polytopax.core` - Core data structures and main APIs
* :mod:`polytopax.algorithms` - Convex hull algorithms and approximation methods
* :mod:`polytopax.operations` - Geometric predicates and operations

Each module contains related functionality and can be imported independently for advanced usage.