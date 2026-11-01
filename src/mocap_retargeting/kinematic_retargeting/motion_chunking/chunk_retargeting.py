"""Plan motion chunks and retarget them with one solver per worker process."""

import os
from dataclasses import dataclass
from math import ceil
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from mocap_retargeting.kinematic_retargeting.optimization.sqp_solver import (
    SQP,
    SQPSolverCfg,
)
from mocap_retargeting.kinematic_retargeting.optimization.trajectory_solver import (
    RetargetingSolver,
)
from mocap_retargeting.retargeted_motion_io.retargeted_motion_io import (
    save_motion,
    validate_trajectory,
)


def plan_chunks(
    motion_data,
    n_workers,
    overlap,
    start_index,
    bvh_path,
    skeleton,
    joint_names_ordered,
    *,
    fps,
):
    total_frames = len(motion_data)
    if total_frames < 1:
        raise ValueError("The selected motion frame range is empty")
    if n_workers < 1 or overlap < 0:
        raise ValueError(
            "Worker count must be positive and overlap must be nonnegative"
        )
    chunk_size = int(np.ceil(total_frames / n_workers))

    chunks = []
    for k in range(n_workers):
        nom_start = k * chunk_size
        nom_end = min((k + 1) * chunk_size, total_frames)
        if nom_start >= nom_end:
            break

        act_start = max(0, nom_start - overlap)
        act_end = min(total_frames, nom_end + overlap)

        if act_start < act_end:
            chunk_data = motion_data[act_start:act_end]
            chunks.append(
                (
                    k,  # chunk_id
                    chunk_data,  # motion_chunk
                    start_index + act_start,
                    bvh_path,  # bvh_path
                    skeleton,  # skeleton
                    joint_names_ordered,
                    fps,
                )
            )

    return chunks, chunk_size


def compute_bins(total_frames, n_workers):
    chunk_size = int(ceil(total_frames / n_workers))
    bins = []
    for k in range(n_workers):
        start = k * chunk_size
        end = min((k + 1) * chunk_size, total_frames)
        if start < end:
            bins.append((start, end))
    return bins


@dataclass
class ChunkWorkerContext:
    """Solver and configuration owned exclusively by one worker process."""

    solver: SQP
    cfg: DictConfig
    q0_robot: np.ndarray
    active_mapping: dict
    sqp_cfg: SQPSolverCfg


_worker_context: Optional[ChunkWorkerContext] = None


def init_worker(
    cfg,
    qp_func_serialized,
    ls_func_serialized,
    q0_robot,
    active_mapping,
    sqp_cfg,
):
    """
    Initializes the worker process.
    Running this once per worker avoids pickling heavy objects for every task.
    """
    solver = SQP(qp_func_serialized, ls_func_serialized)

    global _worker_context
    _worker_context = ChunkWorkerContext(solver, cfg, q0_robot, active_mapping, sqp_cfg)

    os.environ["OMP_NUM_THREADS"] = "1"


def solve_chunk(
    chunk_id, motion_chunk, start_index, bvh_path, skeleton, joint_names_ordered, fps
):
    """
    Lightweight task function.
    Does not take heavy objects as arguments.
    """
    try:
        if _worker_context is None:
            raise RuntimeError("The chunk worker has not been initialized.")
        cfg = _worker_context.cfg
        solver = _worker_context.solver
        q0_robot = _worker_context.q0_robot
        active_mapping = _worker_context.active_mapping
        sqp_cfg = _worker_context.sqp_cfg

        retargeter = RetargetingSolver(
            cfg, solver, motion_chunk, q0_robot, active_mapping, sqp_cfg,
            context=f"{bvh_path} | chunk {chunk_id}", start_index=start_index,
        )
        q_traj, mocap_traj = retargeter.retarget(joint_names_ordered)
        validate_trajectory(
            q_traj,
            mocap_traj,
            expected_frames=len(motion_chunk),
            nq=len(q0_robot),
            context=f"Chunk {chunk_id}",
        )
        if not np.array_equal(mocap_traj, motion_chunk):
            raise ValueError("Returned mocap frames do not match the input chunk")

        if cfg.get("save_chunks", False):
            # joint_names_ordered is passed to solve_chunk
            save_motion(
                cfg,
                q_traj,
                mocap_traj,
                skeleton,
                start_index,
                bvh_path,
                joint_names_ordered,
                fps=fps,
            )

        return (chunk_id, q_traj, mocap_traj)

    except Exception as e:
        raise RuntimeError(
            f"Chunk {chunk_id} of {bvh_path} "
            f"(source frames {start_index}:{start_index + len(motion_chunk)}) failed: {e}"
        ) from e
