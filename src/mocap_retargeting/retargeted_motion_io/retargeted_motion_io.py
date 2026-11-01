"""Save and load retargeted robot motions in the existing NumPy dictionary format."""

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TrajectoryData:
    """Robot q (frames, nq), aligned mocap (frames, joints, xyz), and metadata.

    q stores root xyz, root quaternion xyzw, then robot joint positions.
    """

    q: np.ndarray
    fps: float
    mocap: Optional[np.ndarray] = None
    skeleton: Optional[list] = None
    joint_names: Optional[list[str]] = None

    def to_dict(self):
        return {
            "q": self.q,
            "fps": self.fps,
            "mocap": self.mocap,
            "skeleton": self.skeleton,
            "joint_names": self.joint_names,
        }


def load_trajectory(path):
    """Load an existing project trajectory, preserving optional metadata fields."""
    return np.load(path, allow_pickle=True).item()


def validate_trajectory(
    q, mocap, *, expected_frames=None, nq=None, context="Trajectory"
):
    """Reject incomplete or nonfinite retargeting results before joining/saving."""
    q = np.asarray(q, dtype=float)
    mocap = np.asarray(mocap, dtype=float)
    if expected_frames is None:
        expected_frames = len(q)
    if expected_frames < 1:
        raise ValueError(f"{context}: no frames to save")
    if q.ndim != 2 or q.shape[1] < 7 or (nq is not None and q.shape[1] != nq):
        raise ValueError(f"{context}: invalid robot trajectory shape {q.shape}")
    if mocap.ndim != 3 or mocap.shape[1] < 1 or mocap.shape[2] != 3:
        raise ValueError(f"{context}: invalid mocap trajectory shape {mocap.shape}")
    if len(q) != expected_frames or len(mocap) != expected_frames:
        raise ValueError(
            f"{context}: expected {expected_frames} frames; "
            f"received {len(q)} robot and {len(mocap)} mocap frames"
        )
    if not np.isfinite(q).all() or not np.isfinite(mocap).all():
        raise ValueError(f"{context}: trajectory contains nonfinite values")
    return q, mocap


def save_motion(
    cfg,
    q_trajectory,
    mocap_trajectory,
    skeleton,
    start_index,
    bvh_path,
    joint_names=None,
    *,
    fps=None,
):
    q_trajectory, mocap_trajectory = validate_trajectory(q_trajectory, mocap_trajectory)
    fps = float(cfg.mocap.get("fps", 30) if fps is None else fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Trajectory fps must be finite and positive")

    mocap_name = os.path.splitext(os.path.basename(bvh_path))[0]
    robot_name = cfg.get("robot", "unknown")
    date_str = time.strftime("%Y-%m-%d_%H-%M-%S")

    end_index = start_index + len(q_trajectory)

    filename = f"{robot_name}_{mocap_name}_{start_index}_{end_index}_{date_str}.npy"
    output_dir = os.path.join("log", "kin_trajectories", robot_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    data = TrajectoryData(
        q=q_trajectory,
        fps=fps,
        mocap=mocap_trajectory,
        skeleton=skeleton,
        joint_names=joint_names,
    ).to_dict()
    np.save(output_path, data)
    if cfg.verbose:
        print(f"Saved trajectory to {output_path}")

    if cfg.get("wandb", {}).get("enable", False):
        import wandb

        artifact_name = filename
        artifact = wandb.Artifact(
            name=artifact_name,
            type="motion",
            metadata={
                "robot": robot_name,
                "mocap": mocap_name,
                "start_index": start_index,
                "end_index": end_index,
                "fps": fps,
            },
        )
        artifact.add_file(output_path)
        wandb.log_artifact(artifact)
        print(f"Logged artifact to WandB: {artifact_name}")
    return output_path
