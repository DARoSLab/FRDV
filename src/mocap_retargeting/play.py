"""Command-line entry point for trajectory playback."""

import argparse
from pathlib import Path

from mocap_retargeting import PROJECT_ROOT
from mocap_retargeting.mujoco_integration.mujoco_interface import resolve_robot_xml

from .visualization.playback_app import PlaybackApp


def main():
    parser = argparse.ArgumentParser(description="Playback retargeted motion.")
    parser.add_argument(
        "--path",
        type=str,
        default="log/kin_trajectories/unitree_g1/",
        help="Path to .npy file or directory containing .npy files",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=None,
        help="Number of motions to visualize (optional)",
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help="Disable mocap bone visualization",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(PROJECT_ROOT) / "conf/task/unitree_g1/lafan.yaml"),
        help="Task config (default: bundled Unitree G1 / LAFAN1 configuration)",
    )
    parser.add_argument(
        "--quat-format",
        type=str,
        default="xyzw",
        choices=["xyzw", "wxyz"],
        help="Quaternion format in trajectory data (default: xyzw, use wxyz to skip conversion)",
    )
    args = parser.parse_args()
    try:
        xml_path = str(resolve_robot_xml(args.config))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    return PlaybackApp(args, xml_path).run()


if __name__ == "__main__":
    main()
