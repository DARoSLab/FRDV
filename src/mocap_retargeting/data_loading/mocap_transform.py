"""Validate composed motion transforms and save calibration to the selected YAML."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .dataset_selection import get_data_type


@dataclass(frozen=True)
class MocapTransform:
    """Source-axis scale and output-frame rotation, body offset, and upper-body offset."""

    scale: np.ndarray
    rotation: np.ndarray
    offset: np.ndarray
    upper_body_offset: Optional[np.ndarray] = None


def _read_mapping(path):
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _validate_values(values):
    result = {}
    for name, size in (("scale", 3), ("rotation", 9), ("offset", 3)):
        array = np.array(values[name], dtype=float, copy=True)
        if array.size != size or not np.isfinite(array).all():
            raise ValueError(f"mocap.{name} must contain {size} finite numbers")
        result[name] = array.reshape(3, 3) if name == "rotation" else array.reshape(3)
    if np.any(result["scale"] <= 0):
        raise ValueError("mocap.scale must contain positive values")
    rotation = result["rotation"]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError("mocap.rotation must be a proper 3-by-3 rotation matrix")
    if values.get("upper_body_offset") is not None:
        offset = np.array(values["upper_body_offset"], dtype=float, copy=True)
        if offset.size != 3 or not np.isfinite(offset).all():
            raise ValueError("upper_body_offset must contain 3 finite numbers")
        result["upper_body_offset"] = offset.reshape(3)
    return result


def resolve_mocap_transform(cfg):
    """Read the selected configuration after Hydra has merged its defaults.

    Subject YAMLs inherit the AMASS task and override individual fields. Input
    filenames and folder names never select or change calibration values.
    """
    values = {name: cfg.mocap[name] for name in ("scale", "rotation", "offset")}
    if get_data_type(cfg) == "amass":
        values["upper_body_offset"] = cfg.get("upper_body_offset")
    return MocapTransform(**_validate_values(values))


def save_mocap_transform(
    transform, config_path, scale, rotation, offset, *, upper_body_offset=None
):
    """Update only the selected YAML, preserving its inherited defaults."""
    if upper_body_offset is None and transform is not None:
        upper_body_offset = transform.upper_body_offset
    values = _validate_values(
        {
            "scale": scale,
            "rotation": rotation,
            "offset": offset,
            "upper_body_offset": upper_body_offset,
        }
    )
    path = Path(config_path)
    original = path.read_text() if path.is_file() else ""
    document = _read_mapping(path) if path.is_file() else {}
    if document.get("mocap") is None:
        document["mocap"] = {}
    target = document["mocap"]
    if not isinstance(target, dict):
        raise ValueError(f"mocap must be a mapping in {path}")
    for name, value in values.items():
        destination = document if name == "upper_body_offset" else target
        destination[name] = value.reshape(-1).tolist()
    # Keep the leading comments, including Hydra's package directive.
    header = []
    for line in original.splitlines(keepends=True):
        if line.strip() and not line.lstrip().startswith("#"):
            break
        header.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(header) + yaml.safe_dump(document, sort_keys=False))
    return path
