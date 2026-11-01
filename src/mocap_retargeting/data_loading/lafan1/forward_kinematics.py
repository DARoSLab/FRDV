"""Scalar and vectorized BVH forward kinematics."""

import numpy as np


def compute_frame_positions_raw(root_joint, motion_data, frame_idx):
    if frame_idx >= len(motion_data):
        return {}

    frame_data = motion_data[frame_idx]
    positions = {}
    data_idx = 0

    def compute_fk(joint, parent_transform):
        nonlocal data_idx

        # Local translation (offset)
        local_translation = joint["offset"].copy()

        # Local rotation
        rotation_matrix = np.eye(3)

        # Apply channel data
        for channel in joint["channels"]:
            val = frame_data[data_idx]
            data_idx += 1

            if channel == "Xposition":
                local_translation[0] += val
            elif channel == "Yposition":
                local_translation[1] += val
            elif channel == "Zposition":
                local_translation[2] += val
            elif channel == "Xrotation":
                rad = np.deg2rad(val)
                c, s = np.cos(rad), np.sin(rad)
                rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
                rotation_matrix = rotation_matrix @ rx
            elif channel == "Yrotation":
                rad = np.deg2rad(val)
                c, s = np.cos(rad), np.sin(rad)
                ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                rotation_matrix = rotation_matrix @ ry
            elif channel == "Zrotation":
                rad = np.deg2rad(val)
                c, s = np.cos(rad), np.sin(rad)
                rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                rotation_matrix = rotation_matrix @ rz

        # Global transform
        # T_global = T_parent * T_local
        # T_local = [R | t]

        global_transform = np.eye(4)
        global_transform[:3, :3] = rotation_matrix
        global_transform[:3, 3] = local_translation

        global_transform = parent_transform @ global_transform

        positions[joint["name"]] = global_transform[:3, 3]

        for child in joint["children"]:
            compute_fk(child, global_transform)

    compute_fk(root_joint, np.eye(4))
    return positions


def map_joint_indices(root_joint):
    """
    Helper to determine the start index in the motion_data array for each joint.
    BVH data is flat, so we traverse the hierarchy to map it.
    """
    curr_idx = 0

    def traverse(joint):
        nonlocal curr_idx
        joint["data_start_idx"] = curr_idx
        curr_idx += len(joint["channels"])

        for child in joint["children"]:
            traverse(child)

    traverse(root_joint)


def compute_fk_vectorized(joint, parent_xform, global_positions, motion_data):
    """
    Recursive vectorized Forward Kinematics.
    Calculates transforms for ALL frames for the current joint at once.
    """
    num_frames = len(motion_data)

    # 1. Prepare Local Translation
    # Shape: (N, 3) - Start with the fixed offset repeated for all frames
    local_translation = np.tile(joint["offset"], (num_frames, 1))

    # 2. Prepare Local Rotation
    # Shape: (N, 3, 3) - Start with Identity
    local_rotation = np.tile(np.eye(3), (num_frames, 1, 1))

    # 3. Extract Channel Data
    start_idx = joint.get("data_start_idx", 0)
    channel_data = motion_data[:, start_idx : start_idx + len(joint["channels"])]

    # 4. Apply Channels
    for i, channel in enumerate(joint["channels"]):
        vals = channel_data[:, i]  # Shape (N,)

        if channel == "Xposition":
            local_translation[:, 0] += vals
        elif channel == "Yposition":
            local_translation[:, 1] += vals
        elif channel == "Zposition":
            local_translation[:, 2] += vals

        elif "rotation" in channel:
            # Convert degrees to radians
            rads = np.deg2rad(vals)
            cos_vals = np.cos(rads)
            sin_vals = np.sin(rads)
            ones = np.ones(num_frames)
            zeros = np.zeros(num_frames)

            # Construct rotation matrices (N, 3, 3)
            if channel == "Xrotation":
                # [[1, 0, 0], [0, c, -s], [0, s, c]]
                rot_mats = np.stack(
                    [
                        np.stack([ones, zeros, zeros], axis=1),
                        np.stack([zeros, cos_vals, -sin_vals], axis=1),
                        np.stack([zeros, sin_vals, cos_vals], axis=1),
                    ],
                    axis=1,
                )  # Stack to (N, 3, 3)

            elif channel == "Yrotation":
                # [[c, 0, s], [0, 1, 0], [-s, 0, c]]
                rot_mats = np.stack(
                    [
                        np.stack([cos_vals, zeros, sin_vals], axis=1),
                        np.stack([zeros, ones, zeros], axis=1),
                        np.stack([-sin_vals, zeros, cos_vals], axis=1),
                    ],
                    axis=1,
                )

            elif channel == "Zrotation":
                # [[c, -s, 0], [s, c, 0], [0, 0, 1]]
                rot_mats = np.stack(
                    [
                        np.stack([cos_vals, -sin_vals, zeros], axis=1),
                        np.stack([sin_vals, cos_vals, zeros], axis=1),
                        np.stack([zeros, zeros, ones], axis=1),
                    ],
                    axis=1,
                )

            # Multiply into current local rotation
            # (N, 3, 3) @ (N, 3, 3)
            local_rotation = local_rotation @ rot_mats

    # 5. Construct Local Transform Matrix (N, 4, 4)
    local_xform = np.tile(np.eye(4), (num_frames, 1, 1))
    local_xform[:, :3, :3] = local_rotation
    local_xform[:, :3, 3] = local_translation

    # 6. Compute Global Transform
    # Global = Parent @ Local
    global_xform = parent_xform @ local_xform

    # Store position
    global_positions[joint["name"]] = global_xform[:, :3, 3]

    # 7. Recurse
    for child in joint["children"]:
        compute_fk_vectorized(child, global_xform, global_positions, motion_data)
