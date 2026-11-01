"""Dataset hip names and independent robot orientation initialization."""

import numpy as np
import pytest
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from mocap_retargeting.kinematic_retargeting.optimization.sqp_solver import (
    SQPSolverCfg,
    SQPStats,
)
from mocap_retargeting.kinematic_retargeting.optimization.trajectory_solver import (
    RetargetingSolver,
)


def run_motion(data_type, yaw, nominal, *, degenerate=False, missing_hip=False):
    hips = {
        "lafan1": ("LeftUpLeg", "RightUpLeg"),
        "amass": ("L_Hip", "R_Hip"),
    }[data_type]
    names = ["Pelvis", hips[1], hips[0]]
    left = [-0.15 * np.sin(yaw), 0.15 * np.cos(yaw), 0.9]
    right = [0.15 * np.sin(yaw), -0.15 * np.cos(yaw), 0.9]
    if degenerate:
        right = left
    positions = np.array([[[0, 0, 0.9], right, left]])
    mapping = {name: f"mocap_{name}" for name in hips}
    if missing_hip:
        del mapping[hips[0]]
    cfg = OmegaConf.create(
        {
            "data_type": data_type,
            "verbose": False,
            "warmup_iterations": 0,
            "optimizer": {"sqp_boost_frames": 0},
        }
    )

    class Solver:
        def solve(self, q, params, cfg, **kwargs):
            return q.copy(), SQPStats()

    retargeter = RetargetingSolver(
        cfg, Solver(), positions, nominal, mapping, SQPSolverCfg()
    )
    q, _ = retargeter.retarget(names)
    return np.asarray(q)


@pytest.mark.parametrize("data_type", ["lafan1", "amass"])
@pytest.mark.parametrize("yaw", [0.0, np.pi / 2, -np.pi / 2])
def test_hip_names_initialize_yaw_without_modifying_the_nominal_pose(data_type, yaw):
    nominal = np.zeros(10)
    nominal[2] = 0.9
    nominal[3:7] = Rotation.from_euler("xyz", [0.1, -0.05, 0.2]).as_quat()
    original = nominal.copy()
    result = run_motion(data_type, yaw, nominal)
    expected = Rotation.from_euler("z", yaw) * Rotation.from_quat(original[3:7])
    np.testing.assert_allclose(
        Rotation.from_quat(result[0, 3:7]).as_matrix(), expected.as_matrix(), atol=1e-12
    )
    np.testing.assert_array_equal(nominal, original)
    np.testing.assert_array_equal(result[0, :3], original[:3])
    np.testing.assert_array_equal(result[0, 7:], original[7:])


def test_reusing_a_worker_nominal_pose_does_not_accumulate_amass_yaw():
    nominal = np.zeros(10)
    nominal[6] = 1
    first = run_motion("amass", 0.6, nominal)
    run_motion("amass", -1.1, nominal)
    again = run_motion("amass", 0.6, nominal)
    np.testing.assert_array_equal(again, first)
    np.testing.assert_allclose(
        Rotation.from_quat(first[0, 3:7]).as_matrix(),
        Rotation.from_euler("z", 0.6).as_matrix(),
        atol=1e-12,
    )


@pytest.mark.parametrize("case", ["degenerate", "missing_hip"])
def test_unavailable_hip_heading_keeps_the_configured_pose(case):
    nominal = np.zeros(10)
    nominal[3:7] = Rotation.from_euler("z", 0.3).as_quat()
    result = run_motion("amass", 0.8, nominal, **{case: True})
    np.testing.assert_array_equal(result[0], nominal)
