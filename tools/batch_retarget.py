"""Batch-retarget LAFAN1 or AMASS motions onto Unitree G1."""

import argparse
import concurrent.futures
from dataclasses import dataclass, replace
import json
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "lafan1": ("lafan", ".bvh"),
    "amass": ("amass", ".npz"),
}


@dataclass(frozen=True)
class RetargetJob:
    motion_path: str
    task: str
    wandb_enable: bool = False
    wandb_project: Optional[str] = None
    start_index: int = 0
    end_index: Optional[int] = None
    smplh_model_path: Optional[str] = None
    jit_compile: bool = True
    jit_recompile: bool = False


def discover_motions(input_dir, dataset, recursive=False):
    """Find only the selected dataset's file format, in a stable order."""
    extension = DATASETS[dataset][1]
    input_dir = Path(input_dir)
    paths = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        path.resolve()
        for path in paths
        if path.is_file() and path.suffix.lower() == extension
    )


def retarget_file(job, *, warmup_dir=None):
    """Run one retarget command and require it to report retargeted frames."""
    print(f"Processing: {job.motion_path}", flush=True)
    cmd = [
        "pixi",
        "run",
        "retarget",
        f"task={job.task}",
        f"mocap_src_path={json.dumps(job.motion_path)}",
        f"wandb.enable={str(job.wandb_enable).lower()}",
        f"start_index={job.start_index}",
        f"jit_compile={str(job.jit_compile).lower()}",
        f"jit_recompile={str(job.jit_recompile).lower()}",
    ]
    if job.end_index is not None:
        cmd.append(f"end_index={job.end_index}")
    if job.wandb_project:
        cmd.append(f"wandb.project={json.dumps(job.wandb_project)}")
    if job.smplh_model_path:
        cmd.append(f"smplh_model_path={json.dumps(job.smplh_model_path)}")
    if warmup_dir is not None:
        cmd.extend(
            [
                "n_motion_chunk_workers=1",
                "enable_refine=false",
                "save_chunks=false",
                f"++output_dir={json.dumps(str(warmup_dir))}",
                f"hydra.run.dir={json.dumps(str(warmup_dir))}",
                "hydra.job.chdir=false",
            ]
        )

    try:
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        output = (getattr(exc, "stdout", None) or "") + (
            getattr(exc, "stderr", None) or ""
        )
        print(f"Error processing {job.motion_path}: {output or exc}", flush=True)
        return False, job.motion_path, output or str(exc)

    output = result.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if result.stderr:
        print(
            result.stderr, end="" if result.stderr.endswith("\n") else "\n", flush=True
        )

    # Require a positive frame count as well as a successful process exit.
    frames = re.search(r"Total Frames Retargeted:\s*(\d+)", output)
    if frames is None or int(frames.group(1)) == 0:
        print(
            f"Error processing {job.motion_path}: no retargeted frames were reported.",
            flush=True,
        )
        return False, job.motion_path, output
    print(f"Successfully processed: {job.motion_path}", flush=True)
    return True, job.motion_path, output


def prepare_jit_cache(first_job, warmup_frames):
    """Complete one short compiled run before starting any parallel motion jobs."""
    end_index = first_job.start_index + warmup_frames
    if first_job.end_index is not None:
        end_index = min(end_index, first_job.end_index)
    print(
        f"Preparing JIT cache using up to {end_index - first_job.start_index} frames "
        f"of {first_job.motion_path}...",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="frdv-jit-warmup-") as directory:
        directory = Path(directory)
        # A unique input alias identifies only this warmup's disposable trajectory.
        source = (
            directory
            / f"jit_warmup_{directory.name}{Path(first_job.motion_path).suffix}"
        )
        source.symlink_to(first_job.motion_path)
        warmup_job = replace(
            first_job,
            motion_path=str(source),
            end_index=end_index,
            wandb_enable=False,
            wandb_project=None,
            jit_compile=True,
            jit_recompile=True,
        )
        try:
            success, _, _ = retarget_file(warmup_job, warmup_dir=directory)
        finally:
            robot = first_job.task.split("/", 1)[0]
            trajectories = PROJECT_ROOT / "log/kin_trajectories" / robot
            for path in trajectories.glob(f"{robot}_{source.stem}_*.npy"):
                path.unlink()
    if success:
        print("JIT cache is ready; all full motion jobs will reuse it.", flush=True)
    return success


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir", type=Path, required=True, help="Directory of source motions"
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="lafan1",
        help="Source dataset (default: lafan1)",
    )
    parser.add_argument("--robot", choices=["unitree_g1"], default="unitree_g1")
    parser.add_argument(
        "--task",
        help="Optional Hydra task, e.g. unitree_g1/subject_scaling/amass/accad/Female1Walking_c3d",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Include motions in subdirectories"
    )
    parser.add_argument(
        "--num_workers",
        type=positive_int,
        default=multiprocessing.cpu_count(),
        help="Concurrent motions",
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument(
        "--smplh_model_path",
        type=Path,
        help="SMPL+H directory containing <gender>/model.npz; otherwise use SMPLH_MODEL_PATH",
    )
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    jit_options = parser.add_mutually_exclusive_group()
    jit_options.add_argument(
        "--jit",
        dest="jit",
        action="store_true",
        help="Compile once, then reuse JIT in all jobs (default)",
    )
    jit_options.add_argument(
        "--no-jit",
        "--no_jit",
        dest="jit",
        action="store_false",
        help="Run without JIT or a compilation warmup",
    )
    parser.set_defaults(jit=True)
    parser.add_argument(
        "--jit_warmup_frames",
        type=positive_int,
        default=3,
        help="Frames to retarget during the one-time JIT warmup (default: 3)",
    )
    args = parser.parse_args(argv)

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")
    if args.start_index < 0 or (
        args.end_index is not None and args.end_index <= args.start_index
    ):
        parser.error("Use start_index >= 0 and end_index greater than start_index.")
    model_path = None
    if args.dataset == "amass":
        model_path = args.smplh_model_path or os.environ.get("SMPLH_MODEL_PATH")
        if not model_path:
            parser.error(
                "AMASS requires --smplh_model_path or the SMPLH_MODEL_PATH environment variable."
            )
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_dir():
            parser.error(f"SMPL+H model directory does not exist: {model_path}")
    elif args.smplh_model_path:
        parser.error("--smplh_model_path is only used with --dataset amass.")

    motion_files = discover_motions(input_dir, args.dataset, args.recursive)
    if not motion_files:
        parser.error(
            f"No {DATASETS[args.dataset][1]} motions found for {args.dataset} in {input_dir}."
        )
    task = args.task or f"{args.robot}/{DATASETS[args.dataset][0]}"
    if args.task:
        from hydra import compose, initialize_config_dir
        from hydra.errors import HydraException

        try:
            with initialize_config_dir(
                config_dir=str(PROJECT_ROOT / "conf"), version_base=None
            ):
                selected = compose(config_name="config", overrides=[f"task={task}"])
        except HydraException as exc:
            parser.error(f"Could not load task {task}: {exc}")
        if selected.data_type != args.dataset or selected.robot != args.robot:
            parser.error("The selected --task must match --dataset and --robot.")
    jobs = [
        RetargetJob(
            motion_path=str(path),
            task=task,
            wandb_enable=not args.no_wandb,
            wandb_project=args.wandb_project,
            start_index=args.start_index,
            end_index=args.end_index,
            smplh_model_path=str(model_path) if model_path else None,
            jit_compile=args.jit,
            jit_recompile=False,
        )
        for path in motion_files
    ]
    print(
        f"Found {len(jobs)} {args.dataset} motions; using {args.num_workers} concurrent jobs.",
        flush=True,
    )
    started = time.perf_counter()
    if args.jit and not prepare_jit_cache(jobs[0], args.jit_warmup_frames):
        print("JIT warmup failed; no full motion jobs were started.", flush=True)
        return 1
    successful = failed = 0
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_workers
        ) as executor:
            futures = [executor.submit(retarget_file, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                success, _, _ = future.result()
                successful += int(success)
                failed += int(not success)
    except KeyboardInterrupt:
        print("\nBatch processing interrupted.")
        return 130
    print(
        f"Batch completed in {time.perf_counter() - started:.2f}s. Successful: {successful}; failed: {failed}."
    )
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
