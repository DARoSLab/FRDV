"""Detect and re-solve trajectory spikes, during retargeting or as a standalone command."""

import glob
import multiprocessing
import os
import time
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from math import ceil
from typing import Optional

import hydra
import numpy as np
from omegaconf import DictConfig

from mocap_retargeting import PROJECT_ROOT
from mocap_retargeting.kinematic_retargeting.motion_chunking.chunk_retargeting import (
    compute_bins,
)
from mocap_retargeting.kinematic_retargeting.motion_chunking.trajectory_stitching import (
    blend_trajectories,
)
from mocap_retargeting.kinematic_retargeting.optimization.retargeting_problem import (
    RetargetingProblem,
)
from mocap_retargeting.kinematic_retargeting.optimization.sqp_solver import (
    SQP,
    SQPSolverCfg,
)
from mocap_retargeting.retargeted_motion_io.retargeted_motion_io import load_trajectory


def detect_spikes_adaptive_norm(q, fps, num_worker, cbin=2, threshold_multiplier=5.0):
    dt = 1.0 / fps
    joint_angles = q[:, 7:]

    joint_vel = np.zeros_like(joint_angles)
    joint_vel[:-1] = (joint_angles[1:] - joint_angles[:-1]) / dt
    joint_vel[-1] = joint_vel[-2]
    vel_norm = np.linalg.norm(joint_vel, axis=1)

    joint_acc = np.zeros_like(joint_vel)
    joint_acc[:-1] = (joint_vel[1:] - joint_vel[:-1]) / dt
    joint_acc[-1] = joint_acc[-2]
    acc_norm = np.linalg.norm(joint_acc, axis=1)

    bins = compute_bins(len(q), num_worker)
    n_bins = len(bins)

    vel_threshold = np.zeros(len(q))
    acc_threshold = np.zeros(len(q))

    for bi, (bstart, bend) in enumerate(bins):
        nb_start_idx = max(0, bi - cbin)
        nb_end_idx = min(n_bins - 1, bi + cbin)
        nb_frame_start = bins[nb_start_idx][0]
        nb_frame_end = bins[nb_end_idx][1]

        local_vel = vel_norm[nb_frame_start:nb_frame_end]
        local_acc = acc_norm[nb_frame_start:nb_frame_end]

        vel_threshold[bstart:bend] = np.mean(local_vel) + threshold_multiplier * np.std(
            local_vel
        )
        acc_threshold[bstart:bend] = np.mean(local_acc) + threshold_multiplier * np.std(
            local_acc
        )

    vel_spike_frames = np.where(vel_norm > vel_threshold)[0]
    acc_spike_frames = np.where(acc_norm > acc_threshold)[0]

    print(
        f"     vel_threshold range: [{vel_threshold.min():.2f}, {vel_threshold.max():.2f}]"
    )
    print(
        f"     acc_threshold range: [{acc_threshold.min():.2f}, {acc_threshold.max():.2f}]"
    )

    return vel_spike_frames, acc_spike_frames, vel_norm, acc_norm


def detect_spikes(q, fps, threshold_multiplier=5.0):
    dt = 1.0 / fps
    joint_angles = q[:, 7:]

    joint_vel = np.zeros_like(joint_angles)
    joint_vel[:-1] = (joint_angles[1:] - joint_angles[:-1]) / dt
    joint_vel[-1] = joint_vel[-2]
    vel_norm = np.linalg.norm(joint_vel, axis=1)

    joint_acc = np.zeros_like(joint_vel)
    joint_acc[:-1] = (joint_vel[1:] - joint_vel[:-1]) / dt
    joint_acc[-1] = joint_acc[-2]
    acc_norm = np.linalg.norm(joint_acc, axis=1)

    mean_vel = np.mean(vel_norm)
    std_vel = np.std(vel_norm)
    threshold_vel = mean_vel + threshold_multiplier * std_vel
    print("     threshold_vel: ", threshold_vel)
    vel_spike_frames = np.where(vel_norm > threshold_vel)[0]

    mean_acc = np.mean(acc_norm)
    std_acc = np.std(acc_norm)
    threshold_acc = mean_acc + threshold_multiplier * std_acc
    print("     threshold_acc: ", threshold_acc)
    acc_spike_frames = np.where(acc_norm > threshold_acc)[0]

    return vel_spike_frames, acc_spike_frames, vel_norm, acc_norm


def find_peak_bins(spike_frames, bins):
    peak_bins = set()
    for frame in spike_frames:
        for i, (start, end) in enumerate(bins):
            if start <= frame < end:
                peak_bins.add(i)
                break
    return peak_bins


def group_consecutive_bins(peak_bin_indices):
    if not peak_bin_indices:
        return []
    sorted_indices = sorted(peak_bin_indices)
    groups = [[sorted_indices[0]]]
    for idx in sorted_indices[1:]:
        if idx == groups[-1][-1] + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def prepare_group_tasks(
    groups,
    bins,
    q_full,
    mocap_full,
    overlap,
    refine_boost_frames,
    refine_boost_mult,
    joint_names_ordered,
):
    """
    Prepare task arguments for parallel solving of bin groups.
    Returns list of task_args tuples ready for _solve_group_task.
    """
    total_frames = q_full.shape[0]
    tasks = []

    for gi, group in enumerate(groups):
        group_start = bins[group[0]][0]
        group_end = bins[group[-1]][1]

        # Extend solve region by overlap into good bins on each side
        ext_start = max(0, group_start - overlap)
        ext_end = min(total_frames, group_end + overlap)
        left_overlap = group_start - ext_start
        right_overlap = ext_end - group_end

        # Warm-start: frame just before the extended region
        if ext_start > 0:
            q_start = q_full[ext_start - 1].copy()
        else:
            q_start = q_full[0].copy()

        mocap_chunk = mocap_full[ext_start:ext_end]

        core_frames = group_end - group_start
        ext_frames = ext_end - ext_start
        print(
            f"  Group {gi}: bins {group}, core frames {group_start}-{group_end} ({core_frames}), "
            f"extended {ext_start}-{ext_end} ({ext_frames}), "
            f"left_overlap={left_overlap}, right_overlap={right_overlap}"
        )

        tasks.append(
            (
                gi,
                group,
                bins,
                q_start,
                ext_start,
                ext_end,
                left_overlap,
                right_overlap,
                mocap_chunk,
                joint_names_ordered,
                refine_boost_frames,
                refine_boost_mult,
            )
        )

    return tasks


@dataclass
class RefinementWorkerContext:
    """Persistent solver state for one refinement worker process."""

    solver: SQP
    cfg: DictConfig
    active_mapping: dict
    sqp_cfg: SQPSolverCfg


_worker_context: Optional[RefinementWorkerContext] = None


def _init_refine_worker(cfg, qp_func, ls_func, active_mapping, sqp_cfg):
    """Initialize each worker process with a solver instance."""
    solver = SQP(qp_func, ls_func)

    global _worker_context
    _worker_context = RefinementWorkerContext(solver, cfg, active_mapping, sqp_cfg)

    os.environ["OMP_NUM_THREADS"] = "1"


def _solve_group_task(task_args):
    """
    Worker task: re-solve a single bin group.
    Returns (group_id, group, ext_start, ext_end, left_overlap, right_overlap, q_new).
    """
    (
        group_id,
        group,
        bins,
        q_start,
        ext_start,
        ext_end,
        left_overlap,
        right_overlap,
        mocap_chunk,
        joint_names_ordered,
        refine_boost_frames,
        refine_boost_mult,
    ) = task_args

    if _worker_context is None:
        raise RuntimeError("The refinement worker has not been initialized.")
    solver = _worker_context.solver
    active_mapping = _worker_context.active_mapping
    sqp_cfg = _worker_context.sqp_cfg

    mapping_indices = []
    for bvh_link in active_mapping.keys():
        if bvh_link in joint_names_ordered:
            mapping_indices.append(joint_names_ordered.index(bvh_link))
        else:
            mapping_indices.append(-1)

    q_trajectory = []
    current_q = q_start.copy()
    q_prev = q_start.copy()
    num_frames = mocap_chunk.shape[0]
    frame_times = []

    solve_start = time.perf_counter()
    for i in range(num_frames):
        frame_start = time.perf_counter()
        frame_positions = mocap_chunk[i]

        params = [q_prev]
        for idx in mapping_indices:
            if idx != -1:
                params.append(frame_positions[idx])
            else:
                params.append(np.zeros(3))
        params = np.concatenate(params).reshape(-1)

        # Reduced SQP boost: warm-start is already close, only boost a few frames
        current_sqp_cfg = sqp_cfg
        if i < refine_boost_frames:
            current_sqp_cfg = dc_replace(
                sqp_cfg,
                max_sqp_iter=sqp_cfg.max_sqp_iter * refine_boost_mult,
            )

        try:
            current_q, stats = solver.solve(
                current_q, params, cfg=current_sqp_cfg,
                context=f"Refinement group {group_id} | motion frame {ext_start + i}",
            )
            q_prev = current_q.copy()
            q_trajectory.append(current_q)
        except Exception as e:
            print(f"    [Worker {group_id}] Solver failed at frame {i}: {e}")
            q_trajectory.append(current_q)

        frame_times.append(time.perf_counter() - frame_start)

    total_time = time.perf_counter() - solve_start
    avg_fps = num_frames / total_time if total_time > 0 else 0
    avg_ms = np.mean(frame_times) * 1000 if frame_times else 0
    print(
        f"    [Group {group_id}] {num_frames} frames in {total_time:.2f}s | "
        f"avg {avg_fps:.1f} FPS ({avg_ms:.1f} ms/frame)"
    )

    q_new = np.array(q_trajectory)
    return (group_id, group, ext_start, ext_end, left_overlap, right_overlap, q_new)


def apply_group_results(results, q_full, q_backup, bins):
    """
    Patch re-solved trajectories back into q_full with proper blending.
    """
    for (
        group_id,
        group,
        ext_start,
        ext_end,
        left_overlap,
        right_overlap,
        q_new,
    ) in results:
        group_start = bins[group[0]][0]
        group_end = bins[group[-1]][1]

        # Patch core region directly
        q_full[group_start:group_end] = q_new[
            left_overlap : left_overlap + (group_end - group_start)
        ]

        # Blend left overlap: transition from original (good) to re-solved
        if left_overlap >= 2:
            old_left = q_backup[ext_start:group_start]
            new_left = q_new[:left_overlap]
            blended = blend_trajectories(old_left, new_left)
            q_full[ext_start:group_start] = blended

        # Blend right overlap: transition from re-solved to original (good)
        if right_overlap >= 2:
            new_right = q_new[-right_overlap:]
            old_right = q_backup[group_end:ext_end]
            blended = blend_trajectories(new_right, old_right)
            q_full[group_end:ext_end] = blended


def refine_trajectory(
    cfg,
    q_trajectory,
    mocap_trajectory,
    problem,
    n_workers,
    joint_names_ordered,
    qp_func,
    ls_func,
    bvh_path=None,
    *,
    fps=None,
):
    q_full = np.array(q_trajectory)
    mocap_full = np.array(mocap_trajectory)
    fps = cfg.mocap.get("fps", 30) if fps is None else fps
    total_frames = q_full.shape[0]

    threshold = cfg.get("refine_spike_threshold", 5.0)
    overlap = cfg.get("stitch_overlap_frames", 15)
    refine_boost_frames = cfg.get("refine_boost_frames", 5)
    refine_boost_mult = cfg.get("refine_boost_mult", 5)
    refine_pool_workers = cfg.get("refine_pool_workers") or multiprocessing.cpu_count()

    print(f"[REFINE] Detecting spikes (threshold: mean + {threshold} * std)...")
    vel_spikes, acc_spikes, vel_norm, acc_norm = detect_spikes(q_full, fps, threshold)
    all_spikes = np.union1d(vel_spikes, acc_spikes)

    print(
        f"[REFINE] Velocity spikes: {len(vel_spikes)}, Acceleration spikes: {len(acc_spikes)}"
    )
    print(f"[REFINE] Total unique spike frames: {len(all_spikes)}")

    if len(all_spikes) == 0:
        print("[REFINE] No spikes detected, skipping refinement.")
        return q_trajectory, mocap_trajectory

    bins = compute_bins(total_frames, n_workers)
    peak_bins = find_peak_bins(all_spikes, bins)
    groups = group_consecutive_bins(peak_bins)
    print(f"[REFINE] Peak bins: {sorted(peak_bins)}, Groups: {groups}")

    tasks = prepare_group_tasks(
        groups,
        bins,
        q_full,
        mocap_full,
        overlap,
        refine_boost_frames,
        refine_boost_mult,
        joint_names_ordered,
    )

    init_args = (cfg, qp_func, ls_func, problem.robot.active_mapping, problem.sqp_cfg)

    print(
        f"[REFINE] Solving {len(tasks)} group(s) with {refine_pool_workers} workers..."
    )
    refine_start = time.perf_counter()

    refine_pool = multiprocessing.Pool(
        processes=refine_pool_workers,
        initializer=_init_refine_worker,
        initargs=init_args,
    )
    try:
        results = refine_pool.map(_solve_group_task, tasks)
    finally:
        refine_pool.close()
        refine_pool.join()

    results.sort(key=lambda r: r[0])
    q_backup = q_full.copy()
    apply_group_results(results, q_full, q_backup, bins)

    elapsed = time.perf_counter() - refine_start
    print(f"[REFINE] Refinement completed in {elapsed:.2f}s")

    if cfg.get("refine_save_plot", False) and bvh_path is not None:
        from mocap_retargeting.kinematic_retargeting.refinement.refinement_plots import (
            save_analysis_plot,
        )

        vel_spikes_r, acc_spikes_r, vel_norm_r, acc_norm_r = detect_spikes(
            q_full, fps, threshold
        )
        robot_name = cfg.get("robot", "unknown")
        mocap_name = os.path.splitext(os.path.basename(bvh_path))[0]
        plot_dir = os.path.join("log", "refine_plots", robot_name)
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{mocap_name}_refine_analysis.png")
        save_analysis_plot(
            plot_path,
            vel_norm_r,
            acc_norm_r,
            vel_spikes_r,
            acc_spikes_r,
            bins,
            groups,
            fps,
        )

    return q_full.tolist(), mocap_trajectory


@hydra.main(
    version_base=None,
    config_path=os.path.join(PROJECT_ROOT, "conf"),
    config_name="config",
)
def main(cfg: DictConfig):
    input_dir = cfg.get("refine_input_dir")
    n_workers = cfg.get("refine_n_workers")
    save_plot = cfg.get("refine_save_plot", False)
    threshold = cfg.get("refine_spike_threshold", 5.0)
    overlap = cfg.get("stitch_overlap_frames", 15)
    robot_name = cfg.get("robot", "unknown")
    output_dir = cfg.get(
        "refine_output_dir",
        os.path.join("log", "kin_trajectories_refined", robot_name),
    )

    refine_boost_frames = cfg.get("refine_boost_frames", 5)
    refine_boost_mult = cfg.get("refine_boost_mult", 5)

    refine_pool_workers = cfg.get("refine_pool_workers", multiprocessing.cpu_count())

    if input_dir is None or n_workers is None:
        print("Usage: pixi run retarget_post_process \\")
        print(
            "  +refine_input_dir=<dir> +refine_n_workers=<N> [+refine_save_plot=true]"
        )
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Input directory: {input_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Workers used in original retargeting: {n_workers}")
    print(f"[INFO] Spike threshold: mean + {threshold} * std")
    print(f"[INFO] Blend overlap: {overlap} frames")
    print(
        f"[INFO] Refine SQP boost: {refine_boost_mult}x for first {refine_boost_frames} frames"
    )
    print(f"[INFO] Parallel workers for refinement: {refine_pool_workers}")
    print(f"[INFO] Save plots: {save_plot}")

    print("[INFO] Initializing RetargetingProblem...")
    problem = RetargetingProblem(cfg)
    qp_func, ls_func = problem.generate_solver_functions()

    npy_files = sorted(
        glob.glob(os.path.join(input_dir, "**", "*.npy"), recursive=True)
    )
    npy_files = [f for f in npy_files if "_refined" not in os.path.basename(f)]

    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return

    print(f"[INFO] Found {len(npy_files)} trajectory file(s) to process.")

    init_args = (cfg, qp_func, ls_func, problem.robot.active_mapping, problem.sqp_cfg)
    pool = multiprocessing.Pool(
        processes=refine_pool_workers,
        initializer=_init_refine_worker,
        initargs=init_args,
    )
    print(f"[INFO] Worker pool initialized with {refine_pool_workers} workers.")

    try:
        for file_idx, npy_path in enumerate(npy_files):
            print(f"\n{'=' * 60}")
            print(f"Processing {file_idx + 1}/{len(npy_files)}: {npy_path}")
            print(f"{'=' * 60}")
            motion_start_time = time.perf_counter()

            data = load_trajectory(npy_path)
            q_full = data["q"].copy()
            fps = data.get("fps", 30.0)
            mocap_full = data["mocap"]
            skeleton = data.get("skeleton")
            joint_names = data.get("joint_names")
            total_frames = q_full.shape[0]

            print(
                f"  Trajectory shape: {q_full.shape}, FPS: {fps}, Total frames: {total_frames}"
            )

            bins = compute_bins(total_frames, n_workers)
            chunk_size = int(ceil(total_frames / n_workers))
            print(f"  Bins: {len(bins)}, chunk_size: {chunk_size}")

            vel_spikes, acc_spikes, vel_norm, acc_norm = detect_spikes(
                q_full, fps, threshold
            )
            all_spikes = np.union1d(vel_spikes, acc_spikes)

            print(
                f"  Velocity spikes: {len(vel_spikes)}, Acceleration spikes: {len(acc_spikes)}"
            )
            print(f"  Total unique spike frames: {len(all_spikes)}")

            if len(all_spikes) == 0:
                print("  No spikes detected, skipping.")
                continue

            peak_bins = find_peak_bins(all_spikes, bins)
            print(f"  Peak bins: {sorted(peak_bins)}")

            groups = group_consecutive_bins(peak_bins)
            print(f"  Grouped into {len(groups)} group(s): {groups}")

            tasks = prepare_group_tasks(
                groups,
                bins,
                q_full,
                mocap_full,
                overlap,
                refine_boost_frames,
                refine_boost_mult,
                joint_names,
            )

            q_backup = q_full.copy()

            start_time = time.perf_counter()
            print(f"\n  Solving {len(tasks)} group(s) in parallel...")
            results = pool.map(_solve_group_task, tasks)

            results.sort(key=lambda r: r[0])
            apply_group_results(results, q_full, q_backup, bins)

            elapsed = time.perf_counter() - start_time
            print(f"  Re-solving completed in {elapsed:.2f}s")

            base_name = os.path.basename(npy_path).replace(".npy", "_refined.npy")
            refined_path = os.path.join(output_dir, base_name)
            refined_data = {
                "q": q_full,
                "fps": fps,
                "mocap": mocap_full,
                "skeleton": skeleton,
                "joint_names": joint_names,
            }
            np.save(refined_path, refined_data)
            print(f"  Saved refined trajectory to {refined_path}")

            if save_plot:
                from .refinement_plots import save_analysis_plot

                vel_spikes_r, acc_spikes_r, vel_norm_r, acc_norm_r = detect_spikes(
                    q_full, fps, threshold
                )
                plot_name = os.path.basename(npy_path).replace(
                    ".npy", "_refined_analysis.png"
                )
                plot_path = os.path.join(output_dir, plot_name)
                save_analysis_plot(
                    plot_path,
                    vel_norm_r,
                    acc_norm_r,
                    vel_spikes_r,
                    acc_spikes_r,
                    bins,
                    groups,
                    fps,
                )

            motion_total = time.perf_counter() - motion_start_time
            print(f"  Total time for this motion: {motion_total:.2f}s")
    finally:
        pool.close()
        pool.join()

    print(f"\n{'=' * 60}")
    print("Refinement complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
