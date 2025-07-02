Core Module
===========

.. currentmodule:: polytopax.core

The core module contains the main data structures and high-level APIs for PolytopAX.

ConvexHull Class
----------------

.. autoclass:: polytopax.ConvexHull
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/

      ~ConvexHull.from_points
      ~ConvexHull.from_dict
      ~ConvexHull.to_dict
      ~ConvexHull.volume
      ~ConvexHull.surface_area
      ~ConvexHull.centroid
      ~ConvexHull.diameter
      ~ConvexHull.bounding_box
      ~ConvexHull.contains
      ~ConvexHull.distance_to
      ~ConvexHull.is_degenerate
      ~ConvexHull.vertices_array
      ~ConvexHull.summary
      ~ConvexHull.scale
      ~ConvexHull.translate
      ~ConvexHull.rotate

Main Functions
--------------

.. automodule:: polytopax.core.hull
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: polytopax.core.utils
   :members:
   :undoc-members:
   :show-inheritance:

Type Definitions
----------------

.. currentmodule:: polytopax.core.utils

The core module defines several type aliases for better code documentation:

.. autodata:: PointCloud
   :annotation: = Array

   Point cloud array with shape (..., n_points, dimension)

.. autodata:: HullVertices
   :annotation: = Array

   Hull vertices array with shape (n_vertices, dimension)

.. autodata:: DirectionVectors
   :annotation: = Array

   Direction vectors array with shape (n_directions, dimension)

.. autodata:: SamplingMethod
   :annotation: = Literal["uniform", "icosphere", "adaptive"]

   Direction vector sampling method specification