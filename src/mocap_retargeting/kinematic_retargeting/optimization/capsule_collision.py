"""Capsule distances and derivatives for collision constraints."""

import casadi as ca
import numpy as np


def create_capsule_distance_function():
    """
    Create a CasADi function that computes the distance between two capsules
    and provides automatic differentiation.

    Capsule 1: defined by endpoints a0, a1 and radius r1
    Capsule 2: defined by endpoints b0, b1 and radius r2

    Returns:
        CasADi Function with inputs (a0, a1, r1, b0, b1, r2) and output (distance)
    """
    # Define symbolic variables for capsule 1
    a0 = ca.SX.sym("a0", 3)  # First endpoint of capsule 1
    a1 = ca.SX.sym("a1", 3)  # Second endpoint of capsule 1
    r1 = ca.SX.sym("r1")  # Radius of capsule 1

    # Define symbolic variables for capsule 2
    b0 = ca.SX.sym("b0", 3)  # First endpoint of capsule 2
    b1 = ca.SX.sym("b1", 3)  # Second endpoint of capsule 2
    r2 = ca.SX.sym("r2")  # Radius of capsule 2

    # Direction vectors of the line segments
    d1 = a1 - a0  # Direction of segment 1
    d2 = b1 - b0  # Direction of segment 2

    # Vector between starting points
    r = a0 - b0

    # Squared lengths
    a = ca.dot(d1, d1)  # |d1|^2
    b = ca.dot(d1, d2)  # d1 . d2
    c = ca.dot(d2, d2)  # |d2|^2
    d = ca.dot(d1, r)  # d1 . r
    e = ca.dot(d2, r)  # d2 . r

    # Denominator for the parametric solution
    denom = a * c - b * b

    # Small epsilon to handle degenerate cases
    eps = 1e-10

    # Check if segments are nearly parallel
    # Use a smooth approximation to handle the parallel case

    # Compute unclamped parameters s and t
    # For non-parallel case: solve the 2x2 system
    s_num = b * e - c * d
    t_num = a * e - b * d

    # Handle degenerate cases where one or both segments are points
    # Case: both segments are points
    both_points = ca.if_else(a < eps, ca.if_else(c < eps, 1, 0), 0)

    # Case: segment 1 is a point
    seg1_point = ca.if_else(a < eps, ca.if_else(c >= eps, 1, 0), 0)

    # Case: segment 2 is a point
    seg2_point = ca.if_else(a >= eps, ca.if_else(c < eps, 1, 0), 0)

    # Case: parallel segments
    parallel = ca.if_else(
        ca.fabs(denom) < eps * a * c,
        ca.if_else(a >= eps, ca.if_else(c >= eps, 1, 0), 0),
        0,
    )

    # Normal case (non-parallel, non-degenerate)
    normal_case = 1 - both_points - seg1_point - seg2_point - parallel

    # Compute s and t for normal case
    s_normal = s_num / (denom + eps)
    t_normal = t_num / (denom + eps)

    # Clamp s and t to [0, 1]
    s_clamped = ca.fmin(ca.fmax(s_normal, 0), 1)
    t_clamped = ca.fmin(ca.fmax(t_normal, 0), 1)

    # If s was clamped, recompute t
    # t = (b*s + e) / c
    t_from_s = ca.if_else(c > eps, (b * s_clamped + e) / c, 0)
    t_from_s = ca.fmin(ca.fmax(t_from_s, 0), 1)

    # If t was clamped, recompute s
    # s = (b*t - d) / a
    s_from_t = ca.if_else(a > eps, (b * t_clamped - d) / a, 0)
    s_from_t = ca.fmin(ca.fmax(s_from_t, 0), 1)

    # Choose the pair that gives minimum distance
    # We need to check multiple candidate pairs and pick the best

    # Candidate 1: (s_clamped, t_from_s)
    p1_1 = a0 + s_clamped * d1
    p2_1 = b0 + t_from_s * d2
    dist_sq_1 = ca.dot(p1_1 - p2_1, p1_1 - p2_1)

    # Candidate 2: (s_from_t, t_clamped)
    p1_2 = a0 + s_from_t * d1
    p2_2 = b0 + t_clamped * d2
    dist_sq_2 = ca.dot(p1_2 - p2_2, p1_2 - p2_2)

    # Additional edge cases: check segment endpoints
    # Candidate 3: s=0
    s_c3 = 0
    t_c3 = ca.if_else(c > eps, e / c, 0)
    t_c3 = ca.fmin(ca.fmax(t_c3, 0), 1)
    p1_3 = a0
    p2_3 = b0 + t_c3 * d2
    dist_sq_3 = ca.dot(p1_3 - p2_3, p1_3 - p2_3)

    # Candidate 4: s=1
    s_c4 = 1
    t_c4 = ca.if_else(c > eps, (b + e) / c, 0)
    t_c4 = ca.fmin(ca.fmax(t_c4, 0), 1)
    p1_4 = a1
    p2_4 = b0 + t_c4 * d2
    dist_sq_4 = ca.dot(p1_4 - p2_4, p1_4 - p2_4)

    # Candidate 5: t=0
    t_c5 = 0
    s_c5 = ca.if_else(a > eps, -d / a, 0)
    s_c5 = ca.fmin(ca.fmax(s_c5, 0), 1)
    p1_5 = a0 + s_c5 * d1
    p2_5 = b0
    dist_sq_5 = ca.dot(p1_5 - p2_5, p1_5 - p2_5)

    # Candidate 6: t=1
    t_c6 = 1
    s_c6 = ca.if_else(a > eps, (b - d) / a, 0)
    s_c6 = ca.fmin(ca.fmax(s_c6, 0), 1)
    p1_6 = a0 + s_c6 * d1
    p2_6 = b1
    dist_sq_6 = ca.dot(p1_6 - p2_6, p1_6 - p2_6)

    # Find minimum squared distance among all candidates
    min_dist_sq = dist_sq_1
    min_dist_sq = ca.fmin(min_dist_sq, dist_sq_2)
    min_dist_sq = ca.fmin(min_dist_sq, dist_sq_3)
    min_dist_sq = ca.fmin(min_dist_sq, dist_sq_4)
    min_dist_sq = ca.fmin(min_dist_sq, dist_sq_5)
    min_dist_sq = ca.fmin(min_dist_sq, dist_sq_6)

    # Distance between line segments
    segment_distance = ca.sqrt(min_dist_sq + eps)

    # Distance between capsules (subtract radii)
    capsule_distance = segment_distance - r1 - r2

    # Create the main distance function
    distance_func = ca.Function(
        "capsule_distance",
        [a0, a1, r1, b0, b1, r2],
        [capsule_distance],
        ["a0", "a1", "r1", "b0", "b1", "r2"],
        ["distance"],
    )

    return distance_func


def create_capsule_distance_with_jacobian():
    """
    Create CasADi functions for capsule distance and its Jacobian.

    Returns:
        tuple: (distance_func, jacobian_func)
    """
    # Create base distance function
    distance_func = create_capsule_distance_function()

    # Create Jacobian function
    # Pack all inputs for easier Jacobian computation
    a0 = ca.SX.sym("a0", 3)
    a1 = ca.SX.sym("a1", 3)
    r1 = ca.SX.sym("r1")
    b0 = ca.SX.sym("b0", 3)
    b1 = ca.SX.sym("b1", 3)
    r2 = ca.SX.sym("r2")

    # Compute distance
    dist = distance_func(a0, a1, r1, b0, b1, r2)

    # Compute Jacobians with respect to each input
    jac_a0 = ca.jacobian(dist, a0)
    jac_a1 = ca.jacobian(dist, a1)
    jac_r1 = ca.jacobian(dist, r1)
    jac_b0 = ca.jacobian(dist, b0)
    jac_b1 = ca.jacobian(dist, b1)
    jac_r2 = ca.jacobian(dist, r2)

    # Create Jacobian function
    jacobian_func = ca.Function(
        "capsule_distance_jacobian",
        [a0, a1, r1, b0, b1, r2],
        [jac_a0, jac_a1, jac_r1, jac_b0, jac_b1, jac_r2],
        ["a0", "a1", "r1", "b0", "b1", "r2"],
        [
            "d_dist_d_a0",
            "d_dist_d_a1",
            "d_dist_d_r1",
            "d_dist_d_b0",
            "d_dist_d_b1",
            "d_dist_d_r2",
        ],
    )

    return distance_func, jacobian_func


def create_capsule_distance_vectorized():
    """
    Create a version that takes a single state vector containing all capsule parameters.
    Useful for optimization problems.

    State vector: x = [a0(3), a1(3), r1(1), b0(3), b1(3), r2(1)] (14 elements)

    Returns:
        tuple: (distance_func, jacobian_func)
    """
    # State vector
    x = ca.SX.sym("x", 14)

    # Extract capsule parameters
    a0 = x[0:3]
    a1 = x[3:6]
    r1 = x[6]
    b0 = x[7:10]
    b1 = x[10:13]
    r2 = x[13]

    # Get the base distance function and compute distance
    base_func = create_capsule_distance_function()
    dist = base_func(a0, a1, r1, b0, b1, r2)

    # Compute Jacobian
    jac = ca.jacobian(dist, x)

    # Create functions
    distance_func = ca.Function(
        "capsule_distance_vec", [x], [dist], ["x"], ["distance"]
    )
    jacobian_func = ca.Function(
        "capsule_distance_jacobian_vec", [x], [jac], ["x"], ["jacobian"]
    )

    return distance_func, jacobian_func


def numerical_gradient_check(distance_func, a0, a1, r1, b0, b1, r2, eps=1e-6):
    """
    Verify gradients using finite differences.
    """
    # Get analytical Jacobian
    _, jac_func = create_capsule_distance_with_jacobian()
    jac_a0, jac_a1, jac_r1, jac_b0, jac_b1, jac_r2 = jac_func(a0, a1, r1, b0, b1, r2)

    print("Numerical gradient check:")

    # Check gradient w.r.t. a0
    num_grad_a0 = np.zeros(3)
    for i in range(3):
        a0_plus = a0.copy()
        a0_minus = a0.copy()
        a0_plus[i] += eps
        a0_minus[i] -= eps
        num_grad_a0[i] = (
            float(distance_func(a0_plus, a1, r1, b0, b1, r2))
            - float(distance_func(a0_minus, a1, r1, b0, b1, r2))
        ) / (2 * eps)

    print(f"  d_dist/d_a0 - Analytical: {np.array(jac_a0).flatten()}")
    print(f"  d_dist/d_a0 - Numerical:  {num_grad_a0}")
    print(
        f"  Max error: {np.max(np.abs(np.array(jac_a0).flatten() - num_grad_a0)):.2e}"
    )

    # Check gradient w.r.t. b0
    num_grad_b0 = np.zeros(3)
    for i in range(3):
        b0_plus = b0.copy()
        b0_minus = b0.copy()
        b0_plus[i] += eps
        b0_minus[i] -= eps
        num_grad_b0[i] = (
            float(distance_func(a0, a1, r1, b0_plus, b1, r2))
            - float(distance_func(a0, a1, r1, b0_minus, b1, r2))
        ) / (2 * eps)

    print(f"  d_dist/d_b0 - Analytical: {np.array(jac_b0).flatten()}")
    print(f"  d_dist/d_b0 - Numerical:  {num_grad_b0}")
    print(
        f"  Max error: {np.max(np.abs(np.array(jac_b0).flatten() - num_grad_b0)):.2e}"
    )
