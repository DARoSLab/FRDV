"""Load original AMASS SMPL+H motions and align their body joints."""

import os
import numpy as np
from .body_model_data import _load_smplh_model_data
from .forward_kinematics import _lightweight_fk
from .joint_definitions import (
    BODY_JOINT_NAMES,
    BODY_PARENTS,
    NUM_BODY_JOINTS,
    UPPER_BODY_JOINT_NAMES,
)
from ..motion_data import MotionData


class AMASSLoader:
    """Reconstruct 22 body joints from original 16-beta SMPL+H AMASS files.

    Requires poses (frames, 156), trans (frames, 3), betas (16,), gender,
    and a source frame rate. Finger poses and DMPL deformations are not applied.
    Model files live at <smplh_model_path>/<gender>/model.npz.
    """

    supports_upper_body_offset = True

    def __init__(self, file_path, smplh_model_path=None):
        self.file_path = file_path
        self.joints = {}
        self.root_joint = None

        # Preprocessing parameters
        self.scale = np.ones(3)
        self.rotation = np.eye(3)
        self.offset_xy = np.zeros(3)
        self.offset_z = np.zeros(3)
        self.ground_enabled = False
        self.ground_clearance = 0.05
        self.joint_names_ordered = []
        self.motion_tensor = None
        self.num_frames = 0
        self.manual_offset = np.zeros(3)

        self.upper_body_offset = None

        self.smplh_model_path = smplh_model_path or os.environ.get("SMPLH_MODEL_PATH")

        self._load()

    def _load(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        if not self.smplh_model_path:
            raise ValueError(
                "SMPL+H model path is required. Set smplh_model_path or "
                "SMPLH_MODEL_PATH to the directory containing male/model.npz, "
                "female/model.npz, and/or neutral/model.npz."
            )

        with np.load(self.file_path, allow_pickle=False) as archive:
            required = {"poses", "trans", "betas", "gender"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(
                    f"AMASS motion {self.file_path} is missing: {sorted(missing)}"
                )
            data = {key: archive[key] for key in required}
            rates = {
                key: archive[key]
                for key in ("mocap_framerate", "mocap_frame_rate")
                if key in archive
            }

        poses, trans, betas = (data[key] for key in ("poses", "trans", "betas"))
        if poses.ndim != 2 or poses.shape[1] != 156 or len(poses) == 0:
            raise ValueError(
                f"AMASS poses must have shape (frames, 156), with at least one frame; "
                f"got {poses.shape}. Expected original SMPL+H motion data."
            )
        if trans.shape != (len(poses), 3):
            raise ValueError(
                f"AMASS trans must have shape ({len(poses)}, 3); got {trans.shape}"
            )
        if betas.shape != (16,):
            raise ValueError(f"AMASS betas must have shape (16,); got {betas.shape}")
        for key, value in (("poses", poses), ("trans", trans), ("betas", betas)):
            if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"AMASS {key} must contain finite numeric values")

        if data["gender"].size != 1:
            raise ValueError(
                "AMASS gender must be a single male, female, or neutral value"
            )
        gender = data["gender"].item()
        if isinstance(gender, bytes):
            gender = gender.decode("utf-8")
        gender = str(gender).strip().lower()
        if gender not in {"male", "female", "neutral"}:
            raise ValueError(f"Unsupported AMASS gender: {gender!r}")
        self.gender = gender

        if not rates:
            raise ValueError(
                "AMASS motion requires mocap_framerate or mocap_frame_rate"
            )
        for key, value in rates.items():
            if value.size != 1 or value.dtype.kind not in "fiu":
                raise ValueError(f"AMASS {key} must be a positive finite frame rate")
            rates[key] = float(value.item())
            if not np.isfinite(rates[key]) or rates[key] <= 0:
                raise ValueError(f"AMASS {key} must be a positive finite frame rate")
        if len(rates) == 2 and not np.isclose(*rates.values()):
            raise ValueError("AMASS frame-rate metadata fields disagree")
        self.frame_time = 1.0 / next(iter(rates.values()))
        self.betas = betas.copy()

        # Finger rotations do not affect the first 22 body joints.
        global_orient = poses[:, :3]
        body_pose = poses[:, 3:66]
        self._model_data = _load_smplh_model_data(self.smplh_model_path, gender)

        # Compute FK using lightweight numpy implementation
        self.motion_tensor = _lightweight_fk(
            global_orient,
            body_pose,
            trans,
            betas,
            self._model_data,
            NUM_BODY_JOINTS,
        )
        self.num_frames = self.motion_tensor.shape[0]
        self.joint_names_ordered = list(BODY_JOINT_NAMES)

        # Populate joints dict for compatibility
        for name in self.joint_names_ordered:
            self.joints[name] = {"name": name}

        if self.joint_names_ordered:
            self.root_joint = self.joints[self.joint_names_ordered[0]]

        self.motion_tensor_raw = self.motion_tensor.copy()

        # Compute T-pose: zero body_pose, zero global_orient, zero trans
        num_pose_params = (NUM_BODY_JOINTS - 1) * 3
        self._tpose_raw = _lightweight_fk(
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, num_pose_params), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            betas,
            self._model_data,
            NUM_BODY_JOINTS,
        )
        # Calibration's synthetic canonical pose is Y-up; rotate it into Z-up.
        R_y_to_z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        self._tpose_raw = np.einsum("ij,tkj->tki", R_y_to_z, self._tpose_raw)

    def preprocess(
        self,
        scale=1.0,
        rotation_matrix=None,
        zero_root_xy=True,
        ground=False,
        manual_offset=None,
        upper_body_offset=None,
        ground_clearance=0.05,
    ):
        if np.isscalar(scale):
            self.scale = np.array([scale, scale, scale])
        else:
            self.scale = np.array(scale)
        self.rotation = rotation_matrix if rotation_matrix is not None else np.eye(3)
        self.manual_offset = manual_offset if manual_offset is not None else np.zeros(3)
        self.ground_enabled = ground
        self.ground_clearance = float(ground_clearance)
        if not np.isfinite(self.ground_clearance) or self.ground_clearance < 0:
            raise ValueError("AMASS ground_clearance must be finite and nonnegative")

        self.motion_tensor = self.motion_tensor_raw.copy()

        # 1. Optional grounding in source meters, before scaling and rotation.
        self.offset_z = np.zeros(3)

        if ground:
            foot_indices = [
                self.joint_names_ordered.index(n) for n in ["L_Ankle", "R_Ankle"]
            ]
            min_z = self.motion_tensor[:, foot_indices, 2].min() - self.ground_clearance
            self.offset_z = np.array([0, 0, min_z])
            self.motion_tensor[:, :, 2] -= min_z

        # 2. Apply axis-specific scaling (coordinate-space correction)
        self.motion_tensor = self.motion_tensor * self.scale

        # 3. Rotate the Z-up motion into the configured output heading.
        self.motion_tensor = np.einsum("ij,tkj->tki", self.rotation, self.motion_tensor)

        # 4. Upper body positional offset (in output frame)
        self.upper_body_offset = upper_body_offset
        if upper_body_offset is not None:
            self._apply_upper_body_offset(upper_body_offset)

        # 5. Zero root XY at first frame
        self.offset_xy = np.zeros(3)
        if zero_root_xy:
            root_pos_0 = self.motion_tensor[0, 0, :].copy()
            self.offset_xy = np.array([root_pos_0[0], root_pos_0[1], 0])
            self.motion_tensor -= self.offset_xy

        # 6. Apply manual offset
        self.motion_tensor += self.manual_offset

    def _apply_upper_body_offset(self, offset):
        """Shift upper body joints by a fixed offset vector (in the output frame).

        Args:
            offset: array-like of shape (3,), e.g. [0.05, 0.0, 0.0] for 5cm forward
        """
        offset = np.array(offset)
        for name in UPPER_BODY_JOINT_NAMES:
            if name in self.joint_names_ordered:
                idx = self.joint_names_ordered.index(name)
                self.motion_tensor[:, idx, :] += offset

    def process_all_frames_vectorized(self):
        """Positions are already computed during loading and preprocessing."""

    def get_frames(self, start_idx=0, end_idx=None):
        if end_idx is None:
            end_idx = self.num_frames
        return self.motion_tensor[start_idx:end_idx]

    def get_frame_positions(self, frame_idx):
        if frame_idx >= self.num_frames:
            return {}

        frame_data = self.motion_tensor[frame_idx]
        positions = {}

        for i, name in enumerate(self.joint_names_ordered):
            if i < len(frame_data):
                positions[name] = frame_data[i]

        return positions

    def get_tpose_positions(self):
        """Return a standing calibration reference using the subject's shape.

        After scale and rotation, place the lowest synthetic ankle above the
        scene floor by ground_clearance * scale[2]. Motion grounding is independent:
        preserving a clip's source height must not leave this zero-translation
        reference below ground. Configured offsets and root XY centering still apply.
        """
        tpose = self._tpose_raw[0].copy()  # (22, 3)

        # 1. Scale
        tpose = tpose * self.scale

        # 2. Rotation
        tpose = np.einsum("ij,kj->ki", self.rotation, tpose)

        # 3. Stand in the scene frame, including when the configured rotation tilts.
        foot_indices = [
            self.joint_names_ordered.index(n) for n in ["L_Ankle", "R_Ankle"]
        ]
        clearance = self.ground_clearance * self.scale[2]
        tpose[:, 2] -= tpose[foot_indices, 2].min() - clearance

        # 4. Upper body offset
        if self.upper_body_offset is not None:
            offset = np.array(self.upper_body_offset)
            for name in UPPER_BODY_JOINT_NAMES:
                if name in self.joint_names_ordered:
                    idx = self.joint_names_ordered.index(name)
                    tpose[idx] += offset

        # 5. Zero root XY (center using T-pose's own root)
        root_pos = tpose[0].copy()
        tpose -= np.array([root_pos[0], root_pos[1], 0])

        # 6. Manual offset
        tpose += self.manual_offset

        return {name: tpose[i] for i, name in enumerate(self.joint_names_ordered)}

    def get_skeleton_structure(self):
        """Returns list of (parent_name, child_name) tuples."""
        bones = []
        for child, parent in BODY_PARENTS.items():
            if parent:
                bones.append((parent, child))
        return bones

    def to_motion_data(self):
        return MotionData(
            self.motion_tensor,
            self.joint_names_ordered,
            self.get_skeleton_structure(),
            1.0 / self.frame_time,
        )
