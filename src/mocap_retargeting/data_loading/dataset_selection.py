"""Motion sources retained for Unitree G1 retargeting."""

from .amass.loader import AMASSLoader
from .lafan1.loader import Lafan1Loader
from .motion_data import MotionLoader


TASK_NAMES = {"lafan1": "lafan", "amass": "amass"}


def get_data_type(cfg):
    data_type = cfg.get("data_type", "lafan1").lower().strip()
    if data_type not in TASK_NAMES:
        supported = ", ".join(TASK_NAMES)
        raise ValueError(
            f"Unsupported data_type {data_type!r}. Choose one of: {supported}."
        )
    return data_type


def load_mocap(file_path, cfg) -> MotionLoader:
    """Load LAFAN1 BVH or AMASS NPZ motions."""
    if get_data_type(cfg) == "amass":
        return AMASSLoader(
            file_path,
            smplh_model_path=cfg.get("smplh_model_path"),
        )
    return Lafan1Loader(file_path)


def task_config_filename(cfg):
    """Select the matching task file when preprocessing saves its alignment."""
    return TASK_NAMES[get_data_type(cfg)] + ".yaml"
