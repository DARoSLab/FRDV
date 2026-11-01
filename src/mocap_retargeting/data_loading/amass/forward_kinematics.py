"""NumPy body-model forward kinematics."""

import numpy as np


def _rodrigues_batch(rotvecs):
    """Convert (N, 3) axis-angle vectors to (N, 3, 3) rotation matrices."""
    angles = np.linalg.norm(rotvecs, axis=1, keepdims=True)  # (N, 1)
    angles = np.clip(angles, 1e-8, None)
    axes = rotvecs / angles  # (N, 3)

    cos_a = np.cos(angles)[:, 0]  # (N,)
    sin_a = np.sin(angles)[:, 0]  # (N,)

    K = np.zeros((len(rotvecs), 3, 3), dtype=np.float32)
    K[:, 0, 1] = -axes[:, 2]
    K[:, 0, 2] = axes[:, 1]
    K[:, 1, 0] = axes[:, 2]
    K[:, 1, 2] = -axes[:, 0]
    K[:, 2, 0] = -axes[:, 1]
    K[:, 2, 1] = axes[:, 0]

    eye = np.tile(np.eye(3, dtype=np.float32), (len(rotvecs), 1, 1))
    R = eye + sin_a[:, None, None] * K + (1 - cos_a)[:, None, None] * (K @ K)
    return R


def _lightweight_fk(
    global_orient, body_pose, trans, betas, model_data, num_body_joints
):
    """Compute the SMPL+H body joints without reconstructing the mesh.

    Args:
        global_orient: (N, 3) root orientation in axis-angle
        body_pose: (N, (num_body_joints-1)*3) body joint rotations
        trans: (N, 3) root translation
        betas: (B,) or (N, B) shape parameters
        model_data: dict from _load_smplh_model_data
        num_body_joints: number of body joints to compute (22 for SMPL+H)

    Returns:
        Joint positions array of shape (N, num_body_joints, 3)
    """
    N = global_orient.shape[0]
    J_template = model_data["J_template"]
    J_shapedirs = model_data["J_shapedirs"]
    parents = model_data["parents"]
    num_pose_joints = num_body_joints - 1  # excludes root

    if betas.ndim not in (1, 2) or betas.shape[-1] != model_data["B_max"]:
        raise ValueError(
            f"Expected {model_data['B_max']} shape coefficients; got {betas.shape}"
        )
    if betas.ndim == 2 and betas.shape[0] != N:
        raise ValueError("Per-frame shape coefficients must match the number of frames")
    num_betas = model_data["B_max"]

    # 1. Compute shaped joint locations: J_template + shapedirs @ betas
    if betas.ndim == 1:
        J_shaped = J_template + np.einsum(
            "jcb,b->jc", J_shapedirs[:, :, :num_betas], betas[:num_betas]
        )
        J_shaped = np.tile(J_shaped[None], (N, 1, 1))  # (N, J, 3)
    else:
        J_shaped = J_template[None] + np.einsum(
            "jcb,nb->njc", J_shapedirs[:, :, :num_betas], betas[:, :num_betas]
        )

    # 2. Batch Rodrigues: axis-angle -> rotation matrices
    all_poses = np.concatenate(
        [global_orient[:, None, :], body_pose.reshape(N, num_pose_joints, 3)], axis=1
    )  # (N, num_body_joints, 3)
    R_all = _rodrigues_batch(all_poses.reshape(-1, 3)).reshape(N, num_body_joints, 3, 3)

    # 3. Forward kinematics through kinematic tree
    G = np.zeros((N, num_body_joints, 4, 4), dtype=np.float32)

    # Root joint
    G[:, 0, :3, :3] = R_all[:, 0]
    G[:, 0, :3, 3] = J_shaped[:, 0, :]
    G[:, 0, 3, 3] = 1.0

    for j in range(1, num_body_joints):
        p = parents[j]
        local_T = np.zeros((N, 4, 4), dtype=np.float32)
        local_T[:, :3, :3] = R_all[:, j]
        local_T[:, :3, 3] = J_shaped[:, j, :] - J_shaped[:, p, :]
        local_T[:, 3, 3] = 1.0
        G[:, j] = G[:, p] @ local_T

    # Extract positions and add root translation
    joints = G[:, :num_body_joints, :3, 3] + trans[:, None, :]
    return joints
