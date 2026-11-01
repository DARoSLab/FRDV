"""LAFAN1 loading and motion alignment."""

import numpy as np
from .bvh_parser import BvhParser
from .forward_kinematics import (
    compute_frame_positions_raw,
    compute_fk_vectorized,
    map_joint_indices,
)
from ..motion_data import MotionData


class Lafan1Loader:
    """Load LAFAN1 BVH data and prepare its joint positions."""

    supports_upper_body_offset = False

    def __init__(self, file_path):
        self.file_path = file_path
        self.joints = {}
        self.root_joint = None
        self.frame_time = 0.0
        self.frames = []
        self.motion_data = None

        # Preprocessing parameters
        self.scale = np.ones(3)
        self.rotation = np.eye(3)
        self.offset_xy = np.zeros(3)
        self.offset_z = np.zeros(3)
        self.joint_names_ordered = []
        self.motion_tensor = None

        self.human_height = None

        parsed = BvhParser(file_path)
        self.joints = parsed.joints
        self.root_joint = parsed.root_joint
        self.frame_time = parsed.frame_time
        self.frames = parsed.frames
        self.motion_data = parsed.motion_data
        self.num_frames = parsed.num_frames
        self._estimate_human_height()

    def _estimate_human_height(self):
        if self.num_frames == 0:
            return
        positions = self._compute_frame_positions_raw(0)
        if not positions:
            return
        all_pos = np.array(list(positions.values()))

        height_cm = np.max(all_pos[:, 1]) - np.min(all_pos[:, 1])
        self.human_height = height_cm / 100.0

    def preprocess(
        self,
        scale=1.0,
        rotation_matrix=None,
        zero_root_xy=True,
        ground=True,
        manual_offset=None,
    ):
        if np.isscalar(scale):
            self.scale = np.array([scale, scale, scale])
        else:
            self.scale = np.array(scale)
        self.rotation = rotation_matrix if rotation_matrix is not None else np.eye(3)
        self.manual_offset = manual_offset if manual_offset is not None else np.zeros(3)

        # Calculate translation offsets
        self.offset_xy = np.zeros(3)
        self.offset_z = np.zeros(3)

        if zero_root_xy:
            # Get frame 0 positions (raw)
            positions_0 = self._compute_frame_positions_raw(0)
            root_name = self.root_joint["name"]
            if root_name in positions_0:
                root_pos = positions_0[root_name] * self.scale
                root_pos = self.rotation @ root_pos
                self.offset_xy = np.array([root_pos[0], root_pos[1], 0])

        if ground:
            min_z = float("inf")
            positions = self._compute_frame_positions_raw(0)
            for pos in positions.values():
                p = self.rotation @ (pos * self.scale)
                p -= self.offset_xy
                if p[2] < min_z:
                    min_z = p[2]
            self.offset_z = np.array([0, 0, min_z])

    def get_frame_positions(self, frame_idx):
        positions = self._compute_frame_positions_raw(frame_idx)
        for name in positions:
            pos = positions[name] * self.scale
            pos = self.rotation @ pos
            pos -= self.offset_xy
            pos -= self.offset_z
            pos += self.manual_offset
            positions[name] = pos
        return positions

    def get_frames(self, start_idx=0, end_idx=None):
        if end_idx is None:
            end_idx = self.num_frames

        # Return a view without copying the positions.
        return self.motion_tensor[start_idx:end_idx]

    def get_skeleton_structure(self):
        # Returns list of (parent_name, child_name) tuples
        bones = []

        def traverse(joint):
            for child in joint["children"]:
                bones.append((joint["name"], child["name"]))
                traverse(child)

        traverse(self.root_joint)
        return bones

    def process_all_frames_vectorized(self):
        """
        Vectorized implementation of Forward Kinematics.
        Processes all frames simultaneously using tensor operations.
        """
        self.joint_names_ordered = list(self.joints.keys())
        num_frames = self.num_frames
        num_joints = len(self.joint_names_ordered)

        # 1. Pre-allocate the result tensor
        # Shape: (Frames, Joints, XYZ)
        self.motion_tensor = np.zeros((num_frames, num_joints, 3))

        # 2. Map data indices
        # The motion_data is a flat array (N, C). We need to know which columns
        # belong to which joint. We calculate this once based on the DFS hierarchy.
        map_joint_indices(self.root_joint)

        # 3. Vectorized Forward Kinematics
        # We start with an Identity matrix for the root's parent for ALL frames
        # Shape: (Num_Frames, 4, 4)
        initial_parent_xform = np.tile(np.eye(4), (num_frames, 1, 1))

        # This dictionary will hold the calculated global positions (N, 3) for each joint
        global_positions = {}

        compute_fk_vectorized(
            self.root_joint, initial_parent_xform, global_positions, self.motion_data
        )

        # 4. Post-processing (Scale, Rotate, Offset)
        # Apply the global scene transformations to the entire tensor at once
        for i, name in enumerate(self.joint_names_ordered):
            if name not in global_positions:
                continue

            # Get raw global position: (N, 3)
            pos = global_positions[name]

            # Apply Scale
            pos = pos * self.scale

            # Apply Global Rotation (Root rotation)
            # (N, 3) @ (3, 3).T -> (N, 3)
            pos = pos @ self.rotation.T

            # Apply Offsets
            pos = pos - self.offset_xy - self.offset_z + self.manual_offset

            self.motion_tensor[:, i, :] = pos

    def _compute_frame_positions_raw(self, frame_idx):
        return compute_frame_positions_raw(self.root_joint, self.motion_data, frame_idx)

    def to_motion_data(self):
        return MotionData(
            self.motion_tensor,
            self.joint_names_ordered,
            self.get_skeleton_structure(),
            1.0 / self.frame_time,
        )
