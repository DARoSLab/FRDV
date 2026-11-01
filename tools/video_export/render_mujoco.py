"""Render saved robot trajectories to MP4 using MuJoCo."""

import argparse
import math
import os
from pathlib import Path
import time

import imageio
import mujoco
import numpy as np
import tqdm

from mocap_retargeting.mujoco_integration.mujoco_interface import resolve_robot_xml
from mocap_retargeting.mujoco_integration.mujoco_interface import pin_to_mj
from mocap_retargeting.retargeted_motion_io.retargeted_motion_io import load_trajectory


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def finite_float(value):
    value = float(value)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be finite")
    return value


def positive_float(value):
    value = finite_float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("log/kin_trajectories/unitree_g1/"), help="Trajectory .npy file or directory")
    parser.add_argument("--config", required=True, help="Matching task YAML; robot XML filenames are never inferred")
    parser.add_argument("--width", type=positive_int, default=1280, help="Video width in pixels (default: 1280)")
    parser.add_argument("--height", type=positive_int, default=720, help="Video height in pixels (default: 720)")
    parser.add_argument("--camera-distance", type=positive_float, default=3.0, help="Distance in meters (default: 3)")
    parser.add_argument("--camera-azimuth-offset", type=finite_float, default=180.0, help="Degrees added to tracking-body yaw (default: 180)")
    parser.add_argument("--camera-elevation", type=finite_float, default=-15.0, help="Camera elevation in degrees (default: -15)")
    parser.add_argument("--track-body", default=None, help="Exact MuJoCo body name; default: auto-select head, face, then torso")
    args = parser.parse_args(argv)
    if args.width % 2 or args.height % 2:
        parser.error("MP4 width and height must be even numbers.")
    try:
        args.xml_path = resolve_robot_xml(args.config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return args


def tracking_body_id(model, body_name=None):
    if body_name is not None:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Tracking body {body_name!r} does not exist in the selected robot XML.")
        return body_id
    for keywords in (("head", "face"), ("torso",)):
        for body_id in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name and any(keyword in name.lower() for keyword in keywords):
                return body_id
    return 1 if model.nbody > 1 else 0


def main(argv=None):
    args = parse_args(argv)
    source = args.path.expanduser()
    if source.is_dir():
        files = sorted(path for path in source.iterdir() if path.is_file() and path.suffix == ".npy")
    elif source.is_file() and source.suffix == ".npy":
        files = [source]
    else:
        raise SystemExit(f"Invalid trajectory path: {source}")
    if not files:
        raise SystemExit(f"No .npy trajectories found in {source}")

    model = mujoco.MjModel.from_xml_path(str(args.xml_path))
    body_id = tracking_body_id(model, args.track_body)
    # MuJoCo's offscreen buffer must accommodate resolutions above the model defaults.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    print(f"Using model: {args.xml_path}")
    print(f"Tracking body: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)}")
    os.makedirs("recordings/mp4", exist_ok=True)

    failed = 0
    for filepath in files:
        renderer = None
        try:
            print(f"Loading {filepath}...")
            motion = load_trajectory(filepath)
            data_mj = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height=args.height, width=args.width)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            scene_option = mujoco.MjvOption()
            mp4_frames = []
            for q in tqdm.tqdm(motion["q"], desc=filepath.name):
                data_mj.qpos = pin_to_mj(q)
                mujoco.mj_forward(model, data_mj)
                camera.lookat = data_mj.xpos[body_id]
                body_rotation = data_mj.xmat[body_id].reshape(3, 3)
                forward = body_rotation @ np.array([1, 0, 0])
                yaw = np.degrees(np.arctan2(forward[1], forward[0]))
                camera.distance = args.camera_distance
                camera.azimuth = yaw + args.camera_azimuth_offset
                camera.elevation = args.camera_elevation
                renderer.update_scene(data_mj, camera=camera, scene_option=scene_option)
                mp4_frames.append(renderer.render().copy())

            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            out_path = f"recordings/mp4/{filepath.stem}_{timestamp}.mp4"
            imageio.mimsave(
                out_path, mp4_frames, fps=motion.get("fps", 30), macro_block_size=1
            )
            print(f"Saved {out_path}")
        except Exception as exc:
            failed += 1
            print(f"Error processing {filepath}: {exc}")
        finally:
            if renderer is not None:
                renderer.close()
    if failed:
        raise SystemExit(f"Rendering failed for {failed} of {len(files)} trajectories.")
    print("All done.")


if __name__ == "__main__":
    main()
