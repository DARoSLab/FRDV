"""Check retargeting targets against the robot's MuJoCo link positions."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from scipy.spatial.transform import Rotation

from mocap_retargeting.kinematic_retargeting.optimization.retargeting_problem import (
    RetargetingProblem,
)
from mocap_retargeting.mujoco_integration.mujoco_interface import pin_to_mj


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("task", ["lafan", "amass"])
def test_retargeting_targets_use_robot_link_origins(task):
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"task=unitree_g1/{task}"])
    cfg.mujoco_dst_xml_path = str(ROOT / cfg.mujoco_dst_xml_path)
    problem = RetargetingProblem(cfg)
    model = mujoco.MjModel.from_xml_path(cfg.mujoco_dst_xml_path)
    data = mujoco.MjData(model)

    for yaw in (0.0, 0.6, -1.2):
        q = problem.robot.q0_robot.copy()
        q[3:7] = (
            Rotation.from_euler("z", yaw) * Rotation.from_quat(q[3:7])
        ).as_quat()
        q[7:] += 0.02 * np.sin(np.arange(len(q) - 7))
        data.qpos[:] = pin_to_mj(q)
        mujoco.mj_forward(model, data)
        targets = problem.kinematics.fn_fk_sites(q=q)

        for item in cfg.mapping:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item.robot)
            assert body_id >= 0
            np.testing.assert_allclose(
                np.asarray(targets[f"mocap_{item.bvh}"]).reshape(3),
                data.xpos[body_id],
                rtol=1e-8,
                atol=1e-8,
            )
