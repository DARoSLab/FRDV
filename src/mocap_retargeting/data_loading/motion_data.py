"""Common motion data and the public loader contract."""

from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass
class MotionData:
    """Human joint positions (frames, joints, xyz), with source timing and topology.

    Positions use the loader's current coordinate frame and preprocessing units.
    The retargeting pipeline obtains this view after applying the task alignment.
    """

    positions: np.ndarray
    joint_names: list[str]
    skeleton: list[tuple[str, str]]
    fps: float


class MotionLoader(Protocol):
    num_frames: int
    frame_time: float
    joint_names_ordered: list[str]
    motion_tensor: np.ndarray
    supports_upper_body_offset: bool

    def preprocess(
        self,
        scale=1.0,
        rotation_matrix=None,
        zero_root_xy=True,
        ground=True,
        manual_offset=None,
    ) -> None: ...
    def process_all_frames_vectorized(self) -> None: ...
    def get_frames(self, start_idx=0, end_idx=None) -> np.ndarray: ...
    def get_frame_positions(self, frame_idx) -> dict: ...
    def get_skeleton_structure(self) -> list[tuple[str, str]]: ...
    def to_motion_data(self) -> MotionData: ...
