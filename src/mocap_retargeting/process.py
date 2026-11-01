"""Command-line entry point for interactive mocap calibration."""

import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from hydra.plugins.config_source import ConfigSource
from omegaconf import DictConfig

from . import PROJECT_ROOT
from .mocap_calibration.calibration_app import CalibrationApp


def resolve_calibration_config_path():
    """Find the selected YAML in the same source order Hydra used to load it."""
    runtime = HydraConfig.get()
    config_name = runtime.job.config_name.removesuffix(".yaml")
    task = runtime.runtime.choices.get("task")
    selected = (
        "task/" + task
        if config_name == "config" and task is not None
        else config_name
    ).removesuffix(".yaml")
    source_types = {
        source.scheme(): source for source in Plugins.instance().discover(ConfigSource)
    }
    for entry in runtime.runtime.config_sources:
        source_type = source_types[entry.schema]
        source = source_type(
            provider=entry.provider, path=f"{entry.schema}://{entry.path}"
        )
        if source.is_config(selected):
            if entry.schema != "file":
                raise ValueError(
                    f"Calibration cannot save {selected!r} from {source.full_path()}; "
                    "select a YAML in a local config directory."
                )
            return Path(source.path, selected + ".yaml").resolve()
    raise FileNotFoundError(f"Selected calibration YAML {selected!r} was not found.")


@hydra.main(
    config_path=os.path.join(PROJECT_ROOT, "conf"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    config_path = resolve_calibration_config_path()
    return CalibrationApp(cfg, config_path=config_path).run()


if __name__ == "__main__":
    main()
