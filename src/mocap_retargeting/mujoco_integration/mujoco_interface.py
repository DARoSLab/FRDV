"""Shared MuJoCo integration: configured XML paths, pose conventions, and geometry lookup."""

from pathlib import Path

import mujoco
import numpy as np
import yaml
from hydra import compose, initialize_config_dir
from hydra.errors import HydraException

from mocap_retargeting import PROJECT_ROOT


def resolve_robot_xml(config_path):
    """Read a task config, resolving XML paths beside it or from the repository root."""
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Task configuration does not exist: {config_path}")
    try:
        config = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid task YAML in {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(
            f"Task configuration must contain a YAML mapping: {config_path}"
        )

    # Subject tasks inherit their robot XML from the AMASS task.
    conf_root = Path(PROJECT_ROOT) / "conf"
    if "defaults" in config and config_path.is_relative_to(conf_root):
        name = config_path.relative_to(conf_root).with_suffix("").as_posix()
        try:
            with initialize_config_dir(config_dir=str(conf_root), version_base=None):
                config = compose(config_name=name)
        except HydraException as exc:
            raise ValueError(
                f"Could not compose task configuration {config_path}: {exc}"
            ) from exc

    xml_value = config.get("mujoco_dst_xml_path") or config.get("mujoco_src_xml_path")
    if not isinstance(xml_value, str) or not xml_value.strip():
        raise ValueError(
            f"Task configuration {config_path} must specify mujoco_dst_xml_path "
            "or mujoco_src_xml_path."
        )
    xml_path = Path(xml_value).expanduser()
    candidates = (
        [xml_path]
        if xml_path.is_absolute()
        else [config_path.parent / xml_path, Path(PROJECT_ROOT) / xml_path]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    locations = ", ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(
        f"Configured MuJoCo XML does not exist: {xml_value}. Checked: {locations}. "
        "Generate the processed model with process, or correct --config."
    )


def pin_to_mj(q_pin: np.ndarray) -> np.ndarray:
    q_mujoco = np.zeros(len(q_pin))
    q_mujoco[:3] = q_pin[:3]
    q_mujoco[7:] = q_pin[7:]
    q_mujoco[3:7] = q_pin[3:7][[3, 0, 1, 2]]
    return q_mujoco


def mj_to_pin(q_mujoco: np.ndarray) -> np.ndarray:
    q_pin = np.zeros(len(q_mujoco))
    q_pin[:3] = q_mujoco[:3]
    q_pin[7:] = q_mujoco[7:]
    q_pin[3:7] = q_mujoco[3:7][[1, 2, 3, 0]]
    return q_pin


def get_body_geoms(mj_model, body_id):
    """Return a list of geom IDs belonging to a specific body ID."""
    geom_ids = []
    for geom_id in range(mj_model.ngeom):
        if mj_model.geom_bodyid[geom_id] == body_id:
            geom_ids.append(geom_id)
    return geom_ids


def get_ground_geoms(mj_model):
    """Identify ground geoms (planes)."""
    ground_ids = []
    for geom_id in range(mj_model.ngeom):
        # Check for plane type or specific names
        if mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
            ground_ids.append(geom_id)
            continue

        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name and ("floor" in name.lower() or "ground" in name.lower()):
            ground_ids.append(geom_id)

    return ground_ids
