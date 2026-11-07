# Python examples and runtime options

Install FRDV as described in the [README](../README.md#install). Run the scripts below from the
repository root with `uv run python <script>.py`. G1 downloads on first use.

The dataset examples use the existing `frdv.cli` configuration helpers. The direct solver classes
are exported by `frdv`. A full pipeline call loads profiles, selects the compiled backend, runs
cleanup, and saves playback metadata; `Retargeter.run` takes already prepared positions.

## LAFAN1: load a real clip and call Retargeter

This standalone script loads the first 300 frames of a bundled BVH clip, applies G1's dataset
settings, constructs an `IKConfig`, solves the motion, and saves a viewer-compatible file.
`ik_config` fills in the full mapping, local link offsets, weights, and constraints from YAML.

```python
from pathlib import Path
import numpy as np
from frdv import IKConfig, Retargeter
from frdv.cli.common import (
    ik_config, per_clip_setup, postprocess, robot_height, to_dict,
)
from frdv.cli.config import compose
from frdv.cli.retarget import prepare

settings = to_dict(compose([
    "robot=g1",
    "dataset=lafan1",
    "data=assets/mocap/sample/walk3_subject5.bvh",
    "end=300",
]))
clip = prepare(settings["data"], settings, robot_height(settings["robot"]))
target_positions = clip.X
cfg: IKConfig = ik_config(
    settings["robot"], settings["solver"], clip.keypoints,
    settings["dataset"], calibration=settings["calibration"],
)
retargeter = Retargeter(cfg)
per_clip_setup(retargeter, settings["solver"], clip.fps)
qpos = retargeter.run(target_positions)
postprocess(retargeter, settings["solver"], target_positions, qpos, clip.fps)

# Preserve the model and point offsets so play can display the robot and targets.
robot_links = [p[0] if isinstance(p, (tuple, list)) else p for p in cfg.targets]
point_offsets = [p[1] if isinstance(p, (tuple, list)) else [0, 0, 0] for p in cfg.targets]
output = Path("outputs/python_direct")
output.mkdir(parents=True, exist_ok=True)
np.savez(
    output / "walk3_subject5.npz",
    qpos=qpos, fps=clip.fps, targets=target_positions,
    robot_xml=cfg.xml_path, robot_links=robot_links,
    robot_link_offsets=point_offsets, keypoints=clip.keypoints,
)
print(qpos.shape)
```

View it with `uv run play outputs/python_direct`.

`Retargeter.run` uses the per-frame solver and keeps the clip sequential. This example explicitly
applies the preset's speed cap and cleanup. The config-based pipeline in the README automatically
selects the compiled whole-clip backend when supported and writes the complete run metadata.

## AMASS: run the full pipeline with a saved profile

This standalone script processes a subject folder, saves its motions, and checks the return
status. Replace the two input paths with your subject and original 16-beta SMPL+H model folders.
Only the model genders present in the motions are required.

```python
from frdv.cli.config import compose
from frdv.cli.retarget import run

def main():
    subject_folder = "/path/to/ACCAD/Male1General_c3d"
    model_folder = "/path/to/smplh"
    settings = compose([
        "robot=g1",
        "dataset=amass",
        "subject=accad/Male1General_c3d",
        f"data={subject_folder}",
        f"dataset.smplh_model_path={model_folder}",
        "out=outputs/python_amass",
    ])
    if run(settings) != 0:
        raise RuntimeError("Retargeting failed; see the terminal output.")

if __name__ == "__main__":
    main()
```

View the saved motions with `uv run play outputs/python_amass`.
For a collection-wide base-profile run, omit the `subject=...` override and set `data` to the
collection root. Saved profiles are selected explicitly and reused. See the [AMASS guide](amass.md).
The main guard also supports the multiprocessing fallback when a compiled clip backend is unavailable.

## Supply your own target positions

`IKConfig.targets` names **robot points**. The array passed to `run` contains their desired
**world positions in metres**, in the same order. For real mocap, load, scale, and align the source
before calling this API.

This self-contained synthetic example starts from G1's own pose and moves three target points
forward by 10 cm over 60 frames. It illustrates the array format; dataset presets track the full
configured body mapping.

```python
from pathlib import Path
import numpy as np
from frdv import IKConfig, Kinematics, Retargeter
from frdv.assets import menagerie

xml_path = menagerie("unitree_g1")
robot_links = ["pelvis", "left_knee_link", "right_knee_link"]
kin = Kinematics(xml_path)
kin.forward(kin.model.qpos0, with_jac=False)
initial_positions = kin.positions(kin.point_set(robot_links))

target_positions = np.repeat(initial_positions[None, :, :], 60, axis=0)
target_positions[:, :, 0] += np.linspace(0.0, 0.1, 60)[:, None]
cfg = IKConfig(
    xml_path=xml_path,
    targets=robot_links,
    target_weights=[1000.0] * len(robot_links),
)
retargeter = Retargeter(cfg)
qpos = retargeter.run(target_positions)

output = Path("outputs/python_targets")
output.mkdir(parents=True, exist_ok=True)
np.savez(
    output / "forward.npz",
    qpos=qpos, fps=30.0, targets=target_positions,
    robot_xml=xml_path, robot_links=robot_links,
)
print(qpos.shape)  # (60, 36)
```

View it with `uv run play outputs/python_targets`.

| Value | Meaning |
| --- | --- |
| `cfg.targets` | Ordered robot body/site names, optionally paired with local XYZ offsets. |
| `target_positions` | Finite NumPy array shaped `(frames, points, 3)`, in world-frame metres. |
| `qpos` | `(frames, nq)` robot poses; G1 has 36 values per frame. |
| `qpos[:, :3]` | Root XYZ position in metres. |
| `qpos[:, 3:7]` | Root quaternion in `wxyz` order. |
| `qpos[:, 7:]` | G1's 29 joint angles in radians. |

`IKConfig` by itself uses class defaults. Robot/dataset YAML presets supply the mapping,
nominal pose, weights, smoothing, limits, and offsets used by the normal dataset pipeline.

## Execution options

Pass these as command-line overrides or strings in `compose([...])`:

| Setting | Behavior |
| --- | --- |
| `workers=1` | Process files with one worker. |
| `workers=8` | Allow up to eight workers across files. |
| `mode=auto` | Default: parallel whole files for multiple clips; sequential for one clip. |
| `mode=sequential` | Process files one at a time. |
| `mode=chunks` | Explicitly split an individual clip over worker processes. |
| `start=0 end=300` | Select frames; `end` is exclusive. |
| `save_skeleton=true` | Include the scaled source skeleton for inspection in the viewer. |

The default worker count is the available CPU count. Within-file chunking is disabled by default.
Chunk mode has a separate execution path: `stitch.burn_in` discards lead-in frames, and the solver
re-solves forward across seams until neighboring results agree within `stitch.seam_tol`.
`threads=false` selects the per-frame execution path; `FRDV_NO_NATIVE=1` disables the C backend.

## Model caching and compatibility

Models are cached under `~/.cache/frdv`; `FRDV_CACHE` changes that location. Set
`FRDV_MENAGERIE_REF` to a Menagerie commit SHA for reproducible models. Each configured ref has a
separate cache. Mutable branches such as `main` stay cached until their copy is removed; old
unversioned caches are retained, and the revision-specific copy is fetched on first use.
Output files record an absolute model path; playback on another machine needs that path updated
to an available copy of the same model and assets.

The original `frdv-*` commands remain aliases; `play` corresponds to `frdv-view`, and `calibrate`
is an alias for `augment`. `uv run retarget --cfg robot=g1 dataset=lafan1` prints the composed settings.
Current outputs use `.npz` and MuJoCo `wxyz` root quaternions. Older `.npy` dictionaries used `xyzw`.
