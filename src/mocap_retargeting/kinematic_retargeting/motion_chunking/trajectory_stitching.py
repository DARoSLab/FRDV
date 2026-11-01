"""Blend and combine chunk trajectories, with optional optimization-based stitching."""

import numpy as np

from mocap_retargeting.kinematic_retargeting.optimization.sqp_solver import (
    SQP,
    SQPSolverCfg,
)
from mocap_retargeting.retargeted_motion_io.retargeted_motion_io import (
    validate_trajectory,
)


def blend_trajectories(q_tail, q_head):
    N = q_head.shape[0]
    lin_w = np.linspace(0.0, 1.0, N)
    weights = lin_w * lin_w * (3.0 - 2.0 * lin_w)

    w_col = weights[:, None]
    q_blended = (1.0 - w_col) * q_tail + w_col * q_head

    quat_tail = q_tail[:, 3:7]
    quat_head = q_head[:, 3:7]

    dot = np.sum(quat_tail * quat_head, axis=1, keepdims=True)

    quat_head_corrected = np.where(dot < 0, -quat_head, quat_head)
    dot = np.where(dot < 0, -dot, dot)

    dot = np.clip(dot, -1.0, 1.0)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    threshold = 1e-4

    sin_theta_0_safe = np.where(sin_theta_0 > threshold, sin_theta_0, 1.0)

    fact_tail = np.sin((1.0 - w_col) * theta_0) / sin_theta_0_safe
    fact_head = np.sin(w_col * theta_0) / sin_theta_0_safe

    valid_mask = sin_theta_0 > threshold

    q_slerp = fact_tail * quat_tail + fact_head * quat_head_corrected

    q_linear = (1.0 - w_col) * quat_tail + w_col * quat_head_corrected

    final_quats = np.where(valid_mask, q_slerp, q_linear)

    norm = np.linalg.norm(final_quats, axis=1, keepdims=True)
    final_quats /= norm

    q_blended[:, 3:7] = final_quats

    return q_blended


def combine_chunks(
    results, chunk_size, total_frames, overlap, enable_stitching, verbose=False
):
    """Join exact frame ranges, retaining the earlier contributor until a handoff.

    A worker whose whole assigned section is covered by the active contributor
    does not contribute. For a partially covered section, blend the remaining
    overlap directly into that worker, then make it the active contributor.
    """
    if total_frames < 1 or chunk_size < 1 or overlap < 0:
        raise ValueError("Frame/chunk counts must be positive and overlap nonnegative")
    results = sorted(results, key=lambda result: result[0])
    expected_ids = list(range((total_frames + chunk_size - 1) // chunk_size))
    received_ids = [result[0] for result in results]
    if received_ids != expected_ids:
        raise ValueError(f"Expected chunk IDs {expected_ids}; received {received_ids}")

    checked = []
    nq = joint_shape = None
    for chunk_id, q, mocap in results:
        start = max(0, chunk_id * chunk_size - overlap)
        end = min(total_frames, (chunk_id + 1) * chunk_size + overlap)
        q, mocap = validate_trajectory(
            q, mocap, expected_frames=end - start, nq=nq, context=f"Chunk {chunk_id}"
        )
        nq = q.shape[1]
        if joint_shape is not None and mocap.shape[1:] != joint_shape:
            raise ValueError(f"Chunk {chunk_id}: inconsistent mocap joint shape")
        joint_shape = mocap.shape[1:]
        checked.append((q, mocap, start, end))

    full_q = np.empty((total_frames, nq))
    full_mocap = np.empty((total_frames, *joint_shape))
    active = None
    for chunk_id, current in enumerate(checked):
        q, mocap, start, end = current
        core_start = chunk_id * chunk_size
        core_end = min(core_start + chunk_size, total_frames)
        write_start = core_start

        if enable_stitching and active is not None:
            old_q, old_mocap, old_start, old_end = active
            if old_end >= core_end:
                # The previous result covers this entire section; keep it active.
                full_q[core_start:core_end] = old_q[
                    core_start - old_start : core_end - old_start
                ]
                full_mocap[core_start:core_end] = old_mocap[
                    core_start - old_start : core_end - old_start
                ]
                if verbose:
                    print(
                        f"[INFO] Chunk {chunk_id}: keeping the earlier contributor for frames {core_start}:{core_end}"
                    )
                continue
            if old_end > core_start:
                full_q[core_start:old_end] = blend_trajectories(
                    old_q[core_start - old_start : old_end - old_start],
                    q[core_start - start : old_end - start],
                )
                full_mocap[core_start:old_end] = mocap[
                    core_start - start : old_end - start
                ]
                write_start = old_end

        full_q[write_start:core_end] = q[write_start - start : core_end - start]
        full_mocap[write_start:core_end] = mocap[write_start - start : core_end - start]
        active = current

    validate_trajectory(full_q, full_mocap, expected_frames=total_frames)
    return full_q.tolist(), full_mocap.tolist()


def stitch_chunks_with_factory(problem, q_traj1, q_traj2, overlap_frames=10):
    """
    Stitch two trajectories using Casadi trajectory optimization.

    Args:
        problem: IKRetargetingSolverFactory instance
        q_traj1: First trajectory (list/array of shape [T1, nq])
        q_traj2: Second trajectory (list/array of shape [T2, nq])
        overlap_frames: Number of frames to overlap from each chunk

    Returns:
        stitched_trajectory: Combined trajectory with smooth transition
    """
    import numpy as np

    # Convert to arrays if needed
    q1 = np.array(q_traj1)
    q2 = np.array(q_traj2)

    T1 = q1.shape[0]
    T2 = q2.shape[0]
    nq = q1.shape[1]

    # Take last N frames from first chunk and first N frames from second chunk
    N = min(overlap_frames, T1, T2)
    if N == 0:
        # No overlap possible, just concatenate
        return np.vstack([q1, q2])

    # Extract overlap windows
    window1 = q1[-N:]  # Last N frames of chunk1
    window2 = q2[:N]  # First N frames of chunk2

    # Combined window for optimization: 2N frames
    window = np.vstack([window1, window2])  # Shape (2N, nq)
    T = 2 * N

    # Generate trajectory optimization functions for T frames
    qp_func, ls_func = problem.generate_stitch_solver_functions(T)

    # Create SQP solver
    sqp_cfg = SQPSolverCfg(max_sqp_iter=5, tol=1e-3)  # Fewer iterations for stitching
    solver = SQP(qp_func, ls_func)

    # Prepare parameters: q_first, q_last, q_original
    q_first = window1[0]  # First frame of window1
    q_last = window2[-1]  # Last frame of window2
    q_original = window.flatten()  # Flattened original trajectory

    params = np.concatenate([q_first, q_last, q_original])

    # Initial guess: original window
    x0 = window.flatten()

    # Solve trajectory optimization
    try:
        x_opt, stats = solver.solve(
            x0, params, cfg=sqp_cfg, verbose=False, context="Stitching overlap"
        )
        if not stats.converged and problem.cfg.verbose:
            print(
                f"Stitching optimization didn't converge fully. Final cost: {stats.final_cost:.4f}"
            )
    except Exception as e:
        if problem.cfg.verbose:
            print(f"Stitching optimization failed: {e}. Using original trajectory.")
        return np.vstack([q1, q2])

    # Reshape optimized trajectory
    window_opt = x_opt.reshape(T, nq)

    # Replace the overlapping regions with optimized window
    # Similar to before: first N-1 frames replace last N-1 of q1
    # last N-1 frames replace first N-1 of q2
    q1_stitched = q1.copy()
    q2_stitched = q2.copy()

    if N > 1:
        q1_stitched[-(N - 1) :] = window_opt[1:N]
        q2_stitched[: N - 1] = window_opt[N : 2 * N - 1]

    # Concatenate
    stitched = np.vstack([q1_stitched, q2_stitched])
    return stitched
