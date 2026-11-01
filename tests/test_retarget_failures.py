"""Failed chunks and invalid outputs must fail the motion without inflating counts."""

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from mocap_retargeting.kinematic_retargeting import retarget_motion
from mocap_retargeting.kinematic_retargeting.motion_chunking import chunk_retargeting


@pytest.fixture
def run_case(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("a_bad.npz", "b_good.npz"):
        (inputs / name).touch()
    cfg = OmegaConf.create(
        {
            "robot": "unitree_g1",
            "data_type": "amass",
            "mocap_src_path": str(inputs),
            "mocap": {},
            "n_motion_chunk_workers": 2,
            "stitch_overlap_frames": 1,
            "enable_stitching": True,
            "enable_refine": False,
            "verbose": False,
        }
    )
    problem = SimpleNamespace(
        robot=SimpleNamespace(q0_robot=np.zeros(8), active_mapping={}),
        sqp_cfg=None,
        generate_solver_functions=lambda: (None, None),
    )
    monkeypatch.setattr(retarget_motion, "RetargetingProblem", lambda _: problem)
    positions = np.arange(36, dtype=float).reshape(6, 2, 3)
    motion = SimpleNamespace(
        positions=positions, joint_names=["a", "b"], skeleton=[], fps=120
    )
    loader = SimpleNamespace(num_frames=6, to_motion_data=lambda: motion)
    monkeypatch.setattr(retarget_motion, "prepare_motion", lambda *args: loader)
    monkeypatch.chdir(tmp_path)

    def configure(failure):
        class Pool:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def starmap(self, function, chunks):
                bad = Path(chunks[0][3]).name == "a_bad.npz"
                if bad and failure == "worker_error":
                    raise RuntimeError("Chunk 0 failed: simulated worker exception")
                results = []
                for chunk_id, mocap, *_ in chunks:
                    q = np.zeros((len(mocap), 8))
                    q[:, 6] = 1
                    mocap = mocap.copy()
                    if bad and chunk_id == 0:
                        if failure == "empty":
                            q, mocap = [], []
                        elif failure == "truncated":
                            q, mocap = q[:-1], mocap[:-1]
                        elif failure == "nonfinite":
                            q[0, 7] = np.nan
                        elif failure == "reordered":
                            mocap[[0, 1]] = mocap[[1, 0]]
                    results.append((chunk_id, q, mocap))
                return results

        monkeypatch.setattr(retarget_motion.multiprocessing, "Pool", Pool)
        if failure == "save_error":
            save = retarget_motion.save_motion

            def save_or_fail(*args, **kwargs):
                if Path(args[5]).name == "a_bad.npz":
                    raise OSError("simulated disk write failure")
                return save(*args, **kwargs)

            monkeypatch.setattr(retarget_motion, "save_motion", save_or_fail)
        if failure == "refine_truncated":
            cfg.enable_refine = True

            def refine(cfg, q, mocap, *args, **kwargs):
                return (
                    (q[:-1], mocap[:-1])
                    if Path(args[-1]).name == "a_bad.npz"
                    else (q, mocap)
                )

            monkeypatch.setattr(retarget_motion, "refine_trajectory", refine)
        return cfg

    return configure


@pytest.mark.parametrize(
    "failure",
    [
        "worker_error",
        "empty",
        "truncated",
        "nonfinite",
        "reordered",
        "save_error",
        "refine_truncated",
    ],
)
def test_failed_motion_is_reported_other_motions_continue_and_only_saved_frames_are_counted(
    run_case, tmp_path, capsys, failure
):
    cfg = run_case(failure)
    with pytest.raises(RuntimeError, match="1 of 2 motions failed"):
        retarget_motion.run_retargeting(cfg)
    output = capsys.readouterr().out
    assert "[ERROR] Failed motion" in output and "a_bad.npz" in output
    assert "Successful Motions: 1; Failed Motions: 1" in output
    assert "Total Frames Retargeted: 6" in output
    files = list((tmp_path / "log/kin_trajectories/unitree_g1").glob("*.npy"))
    assert len(files) == 1 and "b_good" in files[0].name
    with (tmp_path / "output/skipped_motions.csv").open() as stream:
        (row,) = list(csv.DictReader(stream))
    assert Path(row["motion_file"]).name == "a_bad.npz"
    assert row["error"]


@pytest.mark.parametrize("failure", ["worker_error", "empty"])
def test_one_worker_failure_produces_no_saved_motion_and_zero_successful_frames(
    run_case, tmp_path, capsys, failure
):
    cfg = run_case(failure)
    cfg.n_motion_chunk_workers = 1
    cfg.mocap_src_path = str(tmp_path / "inputs/a_bad.npz")
    with pytest.raises(RuntimeError, match="1 of 1 motions failed"):
        retarget_motion.run_retargeting(cfg)
    output = capsys.readouterr().out
    assert "Successful Motions: 0; Failed Motions: 1" in output
    assert "Total Frames Retargeted: 0" in output
    assert not (tmp_path / "log/kin_trajectories").exists()


@pytest.mark.parametrize(
    "failure", ["exception", "empty", "truncated", "nonfinite", "reordered"]
)
def test_worker_propagates_context_and_rejects_invalid_results_before_saving(
    monkeypatch, failure
):
    chunk = np.arange(12, dtype=float).reshape(4, 1, 3)
    cfg = OmegaConf.create({"save_chunks": True})
    context = chunk_retargeting.ChunkWorkerContext(None, cfg, np.zeros(8), {}, None)
    monkeypatch.setattr(chunk_retargeting, "_worker_context", context)

    def retarget(names):
        if failure == "exception":
            raise ValueError("simulated retargeting exception")
        q = np.zeros((4, 8))
        q[:, 6] = 1
        mocap = chunk.copy()
        if failure == "empty":
            return [], []
        if failure == "truncated":
            return q[:-1], mocap[:-1]
        if failure == "nonfinite":
            q[0, 7] = np.inf
        if failure == "reordered":
            mocap[[0, 1]] = mocap[[1, 0]]
        return q, mocap

    monkeypatch.setattr(
        chunk_retargeting,
        "RetargetingSolver",
        lambda *args, **kwargs: SimpleNamespace(retarget=retarget),
    )
    monkeypatch.setattr(
        chunk_retargeting,
        "save_motion",
        lambda *args, **kwargs: pytest.fail("Invalid chunk must not be saved"),
    )
    with pytest.raises(RuntimeError, match=r"Chunk 1 of .*motion.npz .*7:11.*failed"):
        chunk_retargeting.solve_chunk(
            1, chunk, 7, "/source/subject/motion.npz", [], ["Pelvis"], 120
        )
