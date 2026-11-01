"""Estimate ACCAD subject calibrations using the existing AMASS T-pose transforms.

This offline utility matches mapped human joints to the G1's t_pose link origins.
It estimates XYZ scale, whole-body offset, and upper_body_offset, retaining the
AMASS coordinate rotation. Review the resulting calibration with `pixi run process`.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import yaml
from hydra import compose, initialize_config_dir
from scipy.optimize import least_squares

from mocap_retargeting import PROJECT_ROOT
from mocap_retargeting.data_loading.amass.loader import AMASSLoader
from mocap_retargeting.data_loading.mocap_transform import (
    MocapTransform,
    resolve_mocap_transform,
)
from mocap_retargeting.mocap_calibration.calibration_app import (
    prepare_calibration_positions,
)


def fit_subject_transform(cfg, loader, robot_positions):
    """Fit the GUI's nine scale/translation controls; keep its rotation fixed.

    The objective uses the same mapped joints and weights as retargeting. All
    human coordinates come from prepare_calibration_positions, including subject
    shape, reference grounding, root centering, and upper-body translation.
    """
    initial = resolve_mocap_transform(cfg)
    names = [item.bvh for item in cfg.mapping]
    weights = np.array(
        [
            item.get("weight", cfg.optimizer.tracking_default_weight)
            for item in cfg.mapping
        ],
        dtype=float,
    )
    targets = np.asarray(robot_positions, dtype=float)
    if targets.shape != (len(names), 3) or not np.isfinite(targets).all():
        raise ValueError(
            "Robot positions must provide one finite XYZ target per mapping"
        )
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Calibration mapping weights must be finite and positive")
    if initial.upper_body_offset is None:
        upper = np.zeros(3)
    else:
        upper = initial.upper_body_offset
    x0 = np.concatenate([initial.scale, initial.offset, upper])
    normalized_weights = np.sqrt(weights / weights.mean())[:, None]

    def residual(x):
        positions = prepare_calibration_positions(
            loader,
            x[:3],
            initial.rotation,
            *x[3:6],
            ground=cfg.mocap.get("ground", False),
            upper_body_offset=x[6:9],
            ground_clearance=cfg.mocap.get("ground_clearance", 0.05),
        )
        human = np.array([positions[name] for name in names])
        return ((human - targets) * normalized_weights).reshape(-1)

    before = np.sqrt(np.mean(residual(x0) ** 2) * 3)
    result = least_squares(
        residual,
        x0,
        bounds=(np.r_[[1e-6] * 3, [-np.inf] * 6], np.full(9, np.inf)),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-10,
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Subject calibration failed: {result.message}")
    if np.linalg.matrix_rank(result.jac) < 9:
        raise ValueError(
            "The reference pose cannot determine all XYZ scales and offsets; calibrate it manually"
        )
    after = np.sqrt(np.mean(residual(result.x) ** 2) * 3)
    if after > before + 1e-8:
        raise ValueError("Estimated calibration is worse than the AMASS defaults")
    transform = MocapTransform(
        scale=result.x[:3].copy(),
        rotation=initial.rotation.copy(),
        offset=result.x[3:6].copy(),
        upper_body_offset=result.x[6:9].copy(),
    )
    return transform, float(before), float(after)


def validate_subject_files(files):
    """One calibration requires the same fitted gender and shape in every clip."""
    identity = None
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            gender = archive["gender"].item()
            if isinstance(gender, bytes):
                gender = gender.decode("utf-8")
            gender = str(gender).strip().lower()
            betas = archive["betas"]
        if betas.shape != (16,) or not np.isfinite(betas).all():
            raise ValueError(f"Invalid subject shape in {path}")
        if identity is None:
            identity = (gender, betas.copy())
        elif gender != identity[0] or not np.array_equal(betas, identity[1]):
            raise ValueError(
                f"Multiple body-shape fits in {path.parent}; use separate subject configurations"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="One ACCAD subject folder, or a directory containing subject folders",
    )
    parser.add_argument(
        "--smplh_model_path",
        type=Path,
        help="Original 16-beta SMPL+H root; alternatively set SMPLH_MODEL_PATH",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(PROJECT_ROOT) / "conf/task/unitree_g1/subject_scaling/amass/accad",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing subject YAMLs, including manual calibration edits",
    )
    args = parser.parse_args(argv)
    source = args.input_dir.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"Input directory does not exist: {source}")
    folders = (
        [source]
        if list(source.glob("*.npz"))
        else sorted(
            path
            for path in source.iterdir()
            if path.is_dir() and list(path.glob("*.npz"))
        )
    )
    if not folders:
        parser.error(f"No subject motions found in {source}")
    for folder in folders:
        validate_subject_files(sorted(folder.glob("*.npz")))
        target = args.output_dir / (folder.name + ".yaml")
        if target.exists() and not args.overwrite:
            parser.error(
                f"Subject configuration already exists: {target}; use --overwrite to replace it"
            )
    with initialize_config_dir(
        config_dir=str(Path(PROJECT_ROOT) / "conf"), version_base=None
    ):
        cfg = compose(config_name="config", overrides=["task=unitree_g1/amass"])
    model = mujoco.MjModel.from_xml_path(
        str(Path(PROJECT_ROOT) / cfg.mujoco_src_xml_path)
    )
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "t_pose")
    if key < 0:
        raise ValueError("The robot model requires a t_pose keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    robot_positions = []
    for item in cfg.mapping:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item.robot)
        if body < 0:
            raise ValueError(f"Mapped robot link does not exist: {item.robot}")
        robot_positions.append(data.xpos[body].copy())
    for folder in folders:
        files = sorted(folder.glob("*.npz"))
        loader = AMASSLoader(files[0], smplh_model_path=args.smplh_model_path)
        transform, before, after = fit_subject_transform(cfg, loader, robot_positions)
        document = {
            "defaults": ["/task/unitree_g1/amass", "_self_"],
            "mocap": {
                "scale": transform.scale.round(8).tolist(),
                "rotation": transform.rotation.reshape(-1).tolist(),
                "offset": transform.offset.round(8).tolist(),
            },
            "upper_body_offset": transform.upper_body_offset.round(8).tolist(),
        }
        header = (
            "# @package _global_\n\n"
            f"# ACCAD {folder.name}: {len(files)} clips with one {loader.gender} body-shape fit.\n"
            "# T-pose estimate using the existing calibration transforms and mapped G1 links.\n"
            f"# Weighted joint RMS: defaults {before * 1000:.2f} mm; estimate {after * 1000:.2f} mm.\n"
            "# Review on actual motions with process; rotation retains the AMASS convention.\n"
        )
        target = args.output_dir / (folder.name + ".yaml")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(header + yaml.safe_dump(document, sort_keys=False))
        print(
            f"{folder.name}: scale={transform.scale.round(5).tolist()}, RMS {before * 1000:.2f} -> {after * 1000:.2f} mm; saved {target}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
