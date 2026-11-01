"""Prepare motion and run chunk retargeting, stitching, refinement, and saving."""

import csv
import multiprocessing
import os
import time

import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf

from mocap_retargeting.data_loading.dataset_selection import get_data_type, load_mocap
from mocap_retargeting.data_loading.mocap_transform import resolve_mocap_transform
from mocap_retargeting.kinematic_retargeting.motion_chunking.chunk_retargeting import (
    init_worker,
    plan_chunks,
    solve_chunk,
)
from mocap_retargeting.kinematic_retargeting.motion_chunking.trajectory_stitching import (
    combine_chunks,
)
from mocap_retargeting.kinematic_retargeting.optimization.retargeting_problem import (
    RetargetingProblem,
)
from mocap_retargeting.kinematic_retargeting.refinement.trajectory_refinement import (
    refine_trajectory,
)
from mocap_retargeting.retargeted_motion_io.retargeted_motion_io import (
    save_motion,
    validate_trajectory,
)


def prepare_motion(cfg, bvh_path):
    transform = resolve_mocap_transform(cfg)
    loader = load_mocap(bvh_path, cfg)
    print(f"[INFO] Preprocessing motion data with scale: {transform.scale}")

    preprocess_kwargs = dict(
        scale=transform.scale,
        rotation_matrix=transform.rotation,
        zero_root_xy=True,
        ground=cfg.mocap.get("ground", get_data_type(cfg) == "lafan1"),
        manual_offset=transform.offset,
    )
    if get_data_type(cfg) == "amass":
        preprocess_kwargs["ground_clearance"] = cfg.mocap.get("ground_clearance", 0.05)
    upper_body_offset = transform.upper_body_offset
    if upper_body_offset is not None and loader.supports_upper_body_offset:
        print(f"[INFO] Applying upper body offset: {list(upper_body_offset)}")
        preprocess_kwargs["upper_body_offset"] = list(upper_body_offset)
    loader.preprocess(**preprocess_kwargs)

    if cfg.verbose:
        print(f"[INFO] Processing all frames for {bvh_path} ...")

    process_start = time.perf_counter()
    loader.process_all_frames_vectorized()

    if cfg.verbose:
        print(
            f"[INFO] Processed all frames in {time.perf_counter() - process_start:.4f}s"
        )

    return loader


def run_retargeting(cfg: DictConfig):
    if cfg.get("wandb", {}).get("enable", False):
        project_name = cfg.wandb.get("project") or cfg.get("robot", "mocap_retargeting")
        wandb.init(
            project=project_name,
            entity=cfg.wandb.get("entity"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    problem = RetargetingProblem(cfg)
    qp_func, ls_func = problem.generate_solver_functions()

    mocap_src_path = cfg.mocap_src_path
    if os.path.isdir(mocap_src_path):
        bvh_files = sorted(
            [
                os.path.join(mocap_src_path, f)
                for f in os.listdir(mocap_src_path)
                if f.endswith((".bvh", ".npz"))
            ]
        )
    else:
        bvh_files = [mocap_src_path]

    print(f"Found {len(bvh_files)} motion files to process.")
    if not bvh_files:
        raise ValueError(f"No motion files found in {mocap_src_path}")

    grand_total_frames = 0
    grand_total_time = 0.0
    successful_motions = 0
    batch_start_time = time.perf_counter()
    # timing_records = []

    init_args = (
        cfg,
        qp_func,
        ls_func,
        problem.robot.q0_robot,
        problem.robot.active_mapping,
        problem.sqp_cfg,
    )

    n_workers = cfg.get("n_motion_chunk_workers", multiprocessing.cpu_count())
    refinement_enabled = cfg.get("enable_refine", False) and n_workers > 1
    if n_workers == 1 and cfg.get("enable_refine", False):
        print("[INFO] Skipping refinement because n_motion_chunk_workers=1.")

    print(f"[INFO] Initializing Pool with {n_workers} workers...")
    pool_init_start = time.perf_counter()
    skipped_motions = []
    with multiprocessing.Pool(
        processes=n_workers, initializer=init_worker, initargs=init_args
    ) as pool:
        print(
            f"[INFO] Pool initialized in {time.perf_counter() - pool_init_start:.4f}s"
        )
        for i, bvh_path in enumerate(bvh_files):
            print(f"[INFO] Processing {i + 1}/{len(bvh_files)}: {bvh_path}...")

            try:
                load_motion_start = time.perf_counter()
                loader = prepare_motion(cfg, bvh_path)
                motion = loader.to_motion_data()
                print(
                    f"[INFO] Motion loaded in {time.perf_counter() - load_motion_start:.4f}s"
                )

                start_index = cfg.get("start_index", cfg.mocap.get("start_index", 0))
                end_index = cfg.get("end_index")
                if end_index is None:
                    end_index = cfg.mocap.get("end_index", -1)

                if end_index == -1 or end_index is None:
                    end_index = loader.num_frames

                motion_data = motion.positions[start_index:end_index]
                total_frames = len(motion_data)
                skeleton = motion.skeleton

                overlap = cfg.get("stitch_overlap_frames", 30)
                chunks, chunk_size = plan_chunks(
                    motion_data,
                    n_workers,
                    overlap,
                    start_index,
                    bvh_path,
                    skeleton,
                    motion.joint_names,
                    fps=motion.fps,
                )

                if i == 0:
                    print(
                        f"Running parallel optimization with {len(chunks)} processes..."
                    )

                start_time = time.perf_counter()
                starmap_start = time.perf_counter()
                results = pool.starmap(solve_chunk, chunks)
                print(
                    f"[INFO] Process spawning and execution took {time.perf_counter() - starmap_start:.4f}s"
                )

                full_q_trajectory, full_mocap_trajectory = combine_chunks(
                    results,
                    chunk_size,
                    total_frames,
                    overlap,
                    cfg.get("enable_stitching", False),
                    cfg.verbose,
                )

                if refinement_enabled:
                    full_q_trajectory, full_mocap_trajectory = refine_trajectory(
                        cfg,
                        full_q_trajectory,
                        full_mocap_trajectory,
                        problem,
                        n_workers,
                        motion.joint_names,
                        qp_func,
                        ls_func,
                        bvh_path,
                        fps=motion.fps,
                    )

                validate_trajectory(
                    full_q_trajectory,
                    full_mocap_trajectory,
                    expected_frames=total_frames,
                    nq=len(problem.robot.q0_robot),
                )
                if not np.array_equal(full_mocap_trajectory, motion_data):
                    raise ValueError(
                        "Assembled mocap frames do not match the selected input frames"
                    )

                total_time = time.perf_counter() - start_time
                retarget_fps = total_frames / total_time
                # cumulative_time = time.perf_counter() - batch_start_time
                print(f"\n[INFO] Total Retargeting Time: {total_time:.4f}s")
                print(f"[INFO] Average FPS: {retarget_fps:.2f} fps\n")

                # timing_records.append({
                #     "motion_file": os.path.basename(bvh_path),
                #     "num_frames": total_frames,
                #     "retarget_time_s": round(total_time, 4),
                #     "retarget_fps": round(retarget_fps, 2),
                #     "cumulative_wall_time_s": round(cumulative_time, 4),
                # })

                save_motion(
                    cfg,
                    full_q_trajectory,
                    full_mocap_trajectory,
                    motion.skeleton,
                    start_index,
                    bvh_path,
                    motion.joint_names,
                    fps=motion.fps,
                )
                # Count only complete trajectories whose save finished successfully.
                grand_total_time += total_time
                grand_total_frames += total_frames
                successful_motions += 1
            except Exception as e:
                print(f"[ERROR] Failed motion {bvh_path}: {e}")
                skipped_motions.append(
                    {
                        "motion_file": bvh_path,
                        "error": str(e),
                    }
                )
                continue

    # Save skipped motions to CSV
    if skipped_motions:
        output_dir = cfg.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        skipped_csv_path = os.path.join(output_dir, "skipped_motions.csv")
        with open(skipped_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["motion_file", "error"])
            writer.writeheader()
            writer.writerows(skipped_motions)
        print(f"[INFO] {len(skipped_motions)} motions skipped. See {skipped_csv_path}")

    batch_total_time = time.perf_counter() - batch_start_time
    print("\n" + "=" * 50)
    print("Batch Processing Complete")
    print(
        f"Successful Motions: {successful_motions}; Failed Motions: {len(skipped_motions)}"
    )
    print(f"Total Frames Retargeted: {grand_total_frames}")
    print(f"Total Processing Time (Sum of individual times): {grand_total_time:.4f}s")
    print(f"Total Batch Wall Time: {batch_total_time:.4f}s")
    if grand_total_time > 0:
        print(f"Overall Average FPS: {grand_total_frames / grand_total_time:.2f} fps")
    print("=" * 50 + "\n")
    if skipped_motions:
        raise RuntimeError(
            f"{len(skipped_motions)} of {len(bvh_files)} motions failed. "
            f"See {skipped_csv_path} for filenames and errors."
        )
