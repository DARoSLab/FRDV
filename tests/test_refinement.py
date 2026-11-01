"""Numerical behavior of spike detection and refinement region patching."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from mocap_retargeting.kinematic_retargeting import retarget_motion

from mocap_retargeting.kinematic_retargeting.motion_chunking.chunk_retargeting import (
    compute_bins,
)

from mocap_retargeting.kinematic_retargeting.refinement.trajectory_refinement import (
    detect_spikes,
)
from mocap_retargeting.kinematic_retargeting.refinement.trajectory_refinement import (
    prepare_group_tasks,
)
from mocap_retargeting.kinematic_retargeting.refinement.trajectory_refinement import (
    apply_group_results,
)


def test_joint_step_flags_the_expected_velocity_and_acceleration_frames():
    q = np.zeros((128, 10))
    q[:, 6] = 1
    q[64:, 7] = 1
    velocity, acceleration, _, _ = detect_spikes(q, fps=30, threshold_multiplier=3)
    np.testing.assert_array_equal(velocity, [63])
    np.testing.assert_array_equal(acceleration, [62, 63])


def test_refinement_replaces_core_and_blends_only_its_extended_region():
    q = np.zeros((80, 10))
    q[:, 0] = np.arange(80)
    q[:, 6] = 1
    mocap = np.zeros((80, 2, 3))
    bins = compute_bins(80, 4)
    (task,) = prepare_group_tasks([[1]], bins, q, mocap, 5, 5, 5, ["a", "b"])
    group_id, group, _, q_start, start, end, left, right, chunk, names, _, _ = task
    assert (start, end, left, right) == (15, 45, 5, 5)
    np.testing.assert_array_equal(q_start, q[14])
    new = q[start:end].copy()
    new[:, 0] += 10
    backup = q.copy()
    apply_group_results(
        [(group_id, group, start, end, left, right, new)], q, backup, bins
    )
    np.testing.assert_array_equal(q[:start], backup[:start])
    np.testing.assert_array_equal(q[end:], backup[end:])
    np.testing.assert_array_equal(q[20:40], new[5:25])
    np.testing.assert_allclose(np.linalg.norm(q[:, 3:7], axis=1), 1)


@pytest.mark.parametrize(
    "workers,enabled,expect_refinement",
    [(1, True, False), (1, False, False), (2, True, True), (2, False, False)],
)
def test_retarget_saves_single_worker_results_without_refinement(
    tmp_path, monkeypatch, capsys, workers, enabled, expect_refinement
):
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "conf"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"n_motion_chunk_workers={workers}",
                f"enable_refine={str(enabled).lower()}",
                "end_index=16",
                "stitch_overlap_frames=2",
            ],
        )
    cfg.mocap_src_path = str(root / cfg.mocap_src_path)
    monkeypatch.chdir(tmp_path)

    # Exercise real loading, chunk planning, stitching, and saved output while
    # replacing the expensive optimization and process boundary.
    problem = SimpleNamespace(
        robot=SimpleNamespace(q0_robot=np.zeros(36), active_mapping={}),
        sqp_cfg=None,
        generate_solver_functions=lambda: (None, None),
    )
    monkeypatch.setattr(retarget_motion, "RetargetingProblem", lambda _: problem)

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["processes"] == workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def starmap(self, function, chunks):
            results = []
            for chunk_id, mocap, *_ in chunks:
                q = np.zeros((len(mocap), 36))
                q[:, :3] = mocap[:, 0]
                q[:, 6] = 1
                results.append((chunk_id, q, mocap))
            return results

    monkeypatch.setattr(retarget_motion.multiprocessing, "Pool", Pool)
    refinement_calls = []

    def refine(cfg, q, mocap, *args, fps):
        # The BVH stores the rounded frame duration 0.033333 seconds.
        assert fps == pytest.approx(1 / 0.033333)
        refinement_calls.append(True)
        refined = np.array(q, copy=True)
        refined[:, 7] = 0.1
        return refined.tolist(), mocap

    monkeypatch.setattr(retarget_motion, "refine_trajectory", refine)
    retarget_motion.run_retargeting(cfg)
    assert len(refinement_calls) == int(expect_refinement)
    (output,) = (tmp_path / "log/kin_trajectories/unitree_g1").glob("*.npy")
    saved = np.load(output, allow_pickle=True).item()
    assert saved["q"].shape == (16, 36)
    assert saved["mocap"].shape[0] == 16
    np.testing.assert_allclose(saved["q"][:, 7], 0.1 if expect_refinement else 0.0)
    if workers == 1 and enabled:
        assert (
            "Skipping refinement because n_motion_chunk_workers=1"
            in capsys.readouterr().out
        )
