"""Command-line entry point for kinematic retargeting."""

import os

import hydra
from omegaconf import DictConfig

from mocap_retargeting import PROJECT_ROOT
from mocap_retargeting.kinematic_retargeting.retarget_motion import run_retargeting


@hydra.main(
    version_base=None,
    config_path=os.path.join(PROJECT_ROOT, "conf"),
    config_name="config",
)
def main(cfg: DictConfig):
    return run_retargeting(cfg)


if __name__ == "__main__":
    main()
