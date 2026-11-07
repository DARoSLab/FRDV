"""Load the original 16-coefficient AMASS SMPL+H body models."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from .amass_joints import BODY_JOINT_NAMES, BODY_PARENTS


def _load_smplh_model_data(model_path, gender):
    """Read <model_path>/<gender>/model.npz, caching unchanged model files."""
    if gender not in {"neutral", "male", "female"}:
        raise ValueError(f"Unsupported SMPL+H gender: {gender!r}")
    path = Path(model_path).expanduser().resolve() / gender / "model.npz"
    if not path.is_file():
        raise FileNotFoundError(f"SMPL+H model not found: {path}")
    stat = path.stat()
    return _read_model(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=6)
def _read_model(path, modified_ns, size):
    required = ("J_regressor", "shapedirs", "v_template", "kintree_table")
    with np.load(path, allow_pickle=False) as archive:
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"SMPL+H model {path} is missing: {sorted(missing)}")
        data = {key: archive[key] for key in required}

    for key, value in data.items():
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"SMPL+H model {path}: {key} must be finite numeric data")

    template = data["v_template"]
    if template.ndim != 2 or template.shape[1] != 3 or len(template) == 0:
        raise ValueError(f"SMPL+H model {path}: v_template must have shape (V, 3)")
    vertices = len(template)
    for key, shape in (
        ("J_regressor", (52, vertices)),
        ("shapedirs", (vertices, 3, 16)),
        ("kintree_table", (2, 52)),
    ):
        if data[key].shape != shape:
            raise ValueError(
                f"SMPL+H model {path}: {key} must have shape {shape}; "
                f"got {data[key].shape}. Use the original 16-coefficient AMASS model."
            )

    tree = data["kintree_table"]
    if tree.dtype.kind not in "iu" or not np.array_equal(tree[1], np.arange(52)):
        raise ValueError(f"SMPL+H model {path}: joints must be ordered by joint ID")
    parents = tree[0].astype(np.int32)
    parents[0] = -1
    if np.any(parents[1:] < 0) or np.any(parents[1:] >= np.arange(1, 52)):
        raise ValueError(f"SMPL+H model {path}: invalid parent ordering")
    expected = [
        -1 if BODY_PARENTS[name] is None else BODY_JOINT_NAMES.index(BODY_PARENTS[name])
        for name in BODY_JOINT_NAMES
    ]
    if not np.array_equal(parents[:22], expected):
        raise ValueError(f"SMPL+H model {path}: incompatible body-joint hierarchy")

    J_regressor = data["J_regressor"][:22].astype(np.float64)
    shapedirs = data["shapedirs"].astype(np.float64)
    return {
        "J_template": J_regressor @ template.astype(np.float64),
        "J_shapedirs": (J_regressor @ shapedirs.reshape(vertices, 48)).reshape(
            22, 3, 16
        ),
        "parents": parents[:22],
        "B_max": 16,
    }
