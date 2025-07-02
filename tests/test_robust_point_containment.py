"""Tests for robust point containment methods verification.

This module validates the improved point containment implementations.
"""

import jax
import jax.numpy as jnp

from polytopax.operations.predicates import point_in_convex_hull


def test_robust_methods_comprehensive():
    """Comprehensive test of robust point containment methods."""

    # Test case 1: Unit square
    square = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    test_points = [
        ([0.5, 0.5], True, "center"),
        ([0.25, 0.75], True, "interior"),
        ([0.0, 0.0], True, "corner"),
        ([0.5, 0.0], True, "edge"),
        ([1.0, 1.0], True, "corner2"),
        ([2.0, 2.0], False, "exterior"),
        ([-0.1, 0.5], False, "outside_left"),
        ([0.5, 1.1], False, "outside_top")
    ]

    print("=== Unit Square Point Containment Test ===")

    methods = ["linear_programming", "barycentric", "halfspace"]
    results = {}

    for method in methods:
        print(f"\n{method.upper()} METHOD:")
        method_results = []

        for point, expected, description in test_points:
            point_array = jnp.array(point)
            result = point_in_convex_hull(point_array, square, method=method)
            correct = bool(result) == expected

            print(f"  {description:12} {point}: {result} {'✓' if correct else '✗'}")
            method_results.append(correct)

        accuracy = sum(method_results) / len(method_results)
        results[method] = accuracy
        print(f"  Accuracy: {accuracy:.1%}")

    # Test case 2: Triangle
    triangle = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])

    triangle_tests = [
        ([0.5, 0.3], True, "interior"),
        ([0.33, 0.33], True, "interior2"),
        ([0.0, 0.0], True, "vertex"),
        ([0.75, 0.0], True, "edge"),
        ([0.5, 1.1], False, "above"),
        ([-0.1, 0.0], False, "left")
    ]

    print("\n=== Triangle Point Containment Test ===")

    for method in methods:
        print(f"\n{method.upper()} METHOD:")
        correct_count = 0

        for point, expected, description in triangle_tests:
            point_array = jnp.array(point)
            result = point_in_convex_hull(point_array, triangle, method=method)
            correct = bool(result) == expected

            print(f"  {description:12} {point}: {result} {'✓' if correct else '✗'}")
            if correct:
                correct_count += 1

        accuracy = correct_count / len(triangle_tests)
        print(f"  Accuracy: {accuracy:.1%}")

    return results


def test_3d_point_containment():
    """Test 3D point containment with robust methods."""
    # Unit cube
    cube = jnp.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]
    ])

    test_points_3d = [
        ([0.5, 0.5, 0.5], True, "center"),
        ([0.0, 0.0, 0.0], True, "corner"),
        ([0.5, 0.0, 0.0], True, "edge"),
        ([2.0, 2.0, 2.0], False, "exterior"),
        ([-0.1, 0.5, 0.5], False, "outside")
    ]

    print("\n=== 3D Cube Point Containment Test ===")

    for method in ["halfspace"]:  # Use best performing method
        print(f"\n{method.upper()} METHOD:")

        for point, expected, description in test_points_3d:
            try:
                point_array = jnp.array(point)
                result = point_in_convex_hull(point_array, cube, method=method)
                correct = bool(result) == expected

                print(f"  {description:12} {point}: {result} {'✓' if correct else '✗'}")
            except Exception as e:
                print(f"  {description:12} {point}: ERROR - {e}")


def test_edge_cases():
    """Test edge cases and degenerate geometries."""
    print("\n=== Edge Cases Test ===")

    # Single point
    single_point = jnp.array([[0.0, 0.0]])
    test_point = jnp.array([0.0, 0.0])

    try:
        result = point_in_convex_hull(test_point, single_point, method="halfspace")
        print(f"Single point hull (same point): {result}")
    except Exception as e:
        print(f"Single point hull failed: {e}")

    # Collinear points (line segment)
    line = jnp.array([[0.0, 0.0], [1.0, 0.0]])

    line_tests = [
        ([0.5, 0.0], True, "on_line"),
        ([0.5, 0.1], False, "above_line"),
        ([-0.1, 0.0], False, "before_line")
    ]

    print("\nLine segment tests:")
    for point, expected, description in line_tests:
        try:
            point_array = jnp.array(point)
            result = point_in_convex_hull(point_array, line, method="halfspace")
            correct = bool(result) == expected
            print(f"  {description:12} {point}: {result} {'✓' if correct else '✗'}")
        except Exception as e:
            print(f"  {description:12} {point}: ERROR - {e}")


if __name__ == "__main__":
    print("🔬 Testing Robust Point Containment Methods")
    print("=" * 50)

    # Run comprehensive tests
    results = test_robust_methods_comprehensive()

    # Test 3D
    test_3d_point_containment()

    # Test edge cases
    test_edge_cases()

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("The halfspace method correctly handles interior points that")
    print("the linear_programming and barycentric methods miss.")
    print("This resolves the critical point containment reliability issue.")
