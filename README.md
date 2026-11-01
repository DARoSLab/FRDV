# FRDV

[![FRDV motion retargeting demo](assets/demo/frdv_preview.webp)](assets/demo/frdv_20_5MB.mp4)

**[Project Page](https://daroslab.github.io/frdv/)** · **[Paper](https://arxiv.org/pdf/2610.28470)**

**Fast Retargeting with Dynamics Validation**

Retarget human motion capture onto **Unitree G1** using constrained kinematic optimization. Calibrate the human-to-robot alignment, retarget a motion, and inspect the saved robot trajectory in your browser.

**Workflow:** `process` (calibrate) → `retarget` (solve and save) → `play` (inspect).
The bundled sample and calibration let you start directly with `retarget`.

| Dataset | Input | Task |
| --- | --- | --- |
| [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) | BVH motion | `unitree_g1/lafan` |
| [AMASS](https://amass.is.tue.mpg.de/) | Original SMPL+H NPZ motion | `unitree_g1/amass` |

> [!NOTE]
> **Only the kinematic retargeting pipeline has been released so far.** The dynamics-validation pipeline described in the paper is not included in the current release.

**Contents**

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Using Your Own Motions](#3-using-your-own-motions)
4. [Calibration and Subject-Specific Scaling](#4-calibration-and-subject-specific-scaling)
5. [Workers and Refinement](#5-workers-and-refinement)
6. [Saved Outputs and Playback](#6-saved-outputs-and-playback)
7. [Optional Tools](#7-optional-tools)
8. [Configuration Reference](#8-configuration-reference)
9. [Contributing](#9-contributing)
10. [Citation and License](#10-citation-and-license)

## 1. Installation

**If Pixi is not installed, follow the [official Pixi installation guide](https://pixi.prefix.dev/latest/installation/) first.** Restart your terminal after installation if the `pixi` command is not available.

Run all commands from the repository root, starting with installation:

```bash
pixi install
```

The environment is configured for Linux x86-64 and macOS Apple Silicon. JIT compilation is enabled by default and requires a C compiler available as `gcc`. To run without JIT, append `jit_compile=false` to a retargeting command.

## 2. Quick Start

Retarget the bundled LAFAN1 sample, then open the result:

```bash
pixi run retarget
pixi run play
```

Open **[localhost:8085](http://localhost:8085)** and click **Play**. The viewer also provides a frame slider, speed control, and looping.

This uses `assets/mocap/sample/walk3_subject5.bvh` and the bundled LAFAN1 calibration. Playback opens the saved G1 trajectories with the LAFAN1 robot configuration.

## 3. Using Your Own Motions

### LAFAN1

Get [lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip) from the [official LAFAN1 dataset folder](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/tree/master/lafan1). On the ZIP page, choose **Download raw file**, then extract the archive. Use the extracted folder containing the `.bvh` files as your motion input.

Retarget a single BVH file:

```bash
pixi run retarget task=unitree_g1/lafan \
  mocap_src_path=/path/to/lafan1/walk.bvh
```

Or process every BVH file in a LAFAN1 folder:

```bash
pixi run retarget task=unitree_g1/lafan \
  mocap_src_path=/path/to/lafan1
```

The task uses the fixed alignment in [lafan.yaml](conf/task/unitree_g1/lafan.yaml). To adjust it, follow [Calibration and Subject-Specific Scaling](#4-calibration-and-subject-specific-scaling).

### AMASS

Download the motion data and body models separately:

| Download | Where to go | What this loader needs |
| --- | --- | --- |
| Motion data | [AMASS homepage](https://amass.is.tue.mpg.de/) | Original SMPL+H `.npz` motions, for example from ACCAD or CMU. |
| Body models | [SMPL+H homepage (MANO / SMPL+H website)](https://mano.is.tue.mpg.de/) | Original AMASS-compatible **16-beta SMPL+H** models in NPZ format. |

From each homepage, open its download section and sign in or register. Select the original **16-beta SMPL+H NPZ** body models.

After extracting the body models, arrange them like this. Only the gender models used by your motions are required:

```text
/path/to/smplh/
├── male/model.npz
├── female/model.npz
└── neutral/model.npz
```

Replace `/path/to/smplh` with your extracted model directory: the folder containing `male/`, `female/`, and/or `neutral/`, not an individual `model.npz`. Pass it with `smplh_model_path` on each AMASS command:

```bash
pixi run retarget task=unitree_g1/amass \
  mocap_src_path=/path/to/motion_poses.npz \
  smplh_model_path=/path/to/smplh
```

To process a folder, use its path instead:

```bash
pixi run retarget task=unitree_g1/amass \
  mocap_src_path=/path/to/AMASS/subject_folder \
  smplh_model_path=/path/to/smplh
```

**Both directory commands scan only files immediately inside the selected folder.** For nested folders, see the [batch utility](#7-optional-tools). General AMASS commands use the AMASS default calibration; [select a subject YAML](#using-an-existing-subject-profile) to use a person's custom values.

## 4. Calibration and Subject-Specific Scaling

> [!WARNING]
> **Calibration is person-dependent.** People have different heights and body proportions. Using another person's calibration can cause poor foot placement, excessive knee bending, or torso lean as the robot tries to match the targets. Check several motions and poses before reusing a calibration for a whole subject.

### Understanding the defaults

The dataset task YAML supplies the default scale, rotation, and offsets:

| Task config | Bundled `mocap.scale` | Use |
| --- | --- | --- |
| [lafan.yaml](conf/task/unitree_g1/lafan.yaml) | `[0.0076, 0.0078, 0.0074]` | Fixed LAFAN1 starting calibration. |
| [amass.yaml](conf/task/unitree_g1/amass.yaml) | `[0.87, 0.83, 0.75]` | AMASS fallback when no subject task is selected. |

Scale values depend on source coordinates and units. Use the matching dataset's calibration as your starting point.

For AMASS, calibrate each person's body to the G1 and reuse that profile across their compatible motions. Keep different fitted body shapes in separate profiles.

### Interactive calibration

For LAFAN1, open the bundled reference motion:

```bash
pixi run process task=unitree_g1/lafan
```

To use your own reference, append `mocap_src_path=/path/to/reference.bvh`. Open **[localhost:8080](http://localhost:8080)**. The UI loads the selected YAML's saved calibration.

1. Adjust **Scale X/Y/Z** to match body proportions to the robot.
2. Adjust **Yaw Adjustment** and **X/Y/Z Offset** to align the motion. Negative **Z Offset** lowers the whole motion in the scene.
3. For AMASS, adjust the **Upper Body Offset** controls if needed.
4. Click **Save All** to update the selected YAML and its configured processed robot XML.
5. Run `retarget` again and inspect several poses. Existing saved trajectories retain their original calibration.

LAFAN1 calibration previews the motion's first frame. AMASS previews a T-pose reconstructed from the selected motion's gender and body shape.

### Using an existing subject profile

Start with one of the **20 ACCAD subject profiles** under [subject_scaling/amass/accad](conf/task/unitree_g1/subject_scaling/amass/accad/).

Select the profile explicitly and provide motions from that subject:

```bash
pixi run retarget \
  task=unitree_g1/subject_scaling/amass/accad/Female1Walking_c3d \
  mocap_src_path=/path/to/ACCAD/Female1Walking_c3d \
  smplh_model_path=/path/to/smplh
```

The same profile applies to every NPZ in that folder. **The folder name alone does not activate the profile.**

### Creating a profile for another person

Create `conf/task/unitree_g1/subject_scaling/amass/my_subject.yaml` with AMASS defaults as an initial guess:

```yaml
# @package _global_
defaults:
  - /task/unitree_g1/amass
  - _self_

mocap:
  scale: [0.87, 0.83, 0.75]
  offset: [0.04, 0.0, 0.0]
upper_body_offset: [0.03, 0.0, 0.065]
```

Open calibration using that profile and one of the person's motions:

```bash
pixi run process \
  task=unitree_g1/subject_scaling/amass/my_subject \
  mocap_src_path=/path/to/my_subject/reference_poses.npz \
  smplh_model_path=/path/to/smplh
```

Adjust the controls and click **Save All** to save the calibration into `my_subject.yaml`. Then retarget the person's folder with the same task:

```bash
pixi run retarget \
  task=unitree_g1/subject_scaling/amass/my_subject \
  mocap_src_path=/path/to/my_subject \
  smplh_model_path=/path/to/smplh
```

Missing fields inherit from AMASS, including its coordinate rotation. The order of precedence is **shared base → dataset task → subject YAML → command-line overrides**.

Calibration also saves the configured robot XML. Set `mujoco_dst_xml_path` in a subject profile to keep a separate processed robot file.

<details>
<summary>Coordinate conventions and ground alignment</summary>

| Setting | Meaning |
| --- | --- |
| `mocap.scale` | Three factors along the **source X/Y/Z axes**, applied before rotation. Unequal factors stretch along those fixed axes. |
| `mocap.rotation` | Source-to-scene rotation, stored as nine row-major values. The UI's yaw control adds an adjustment to the saved rotation. |
| `mocap.offset` | Whole-body translation in **scene X/Y/Z**, in meters. Scene Z points upward. |
| `upper_body_offset` | AMASS-only scene translation of the upper torso, head, and arms. Set `[0, 0, 0]` to disable it. |
| `mocap.ground` | Recorded-motion grounding: `true` for LAFAN1, `false` for AMASS by default. |

</details>

## 5. Workers and Refinement

> [!WARNING]
> **For best motion quality, use one worker (no chunking).** Each motion is solved as one continuous sequence. Multiple workers divide the motion into chunks and can introduce differences where those chunks are joined.

Set `n_motion_chunk_workers` in [conf/base.yaml](conf/base.yaml). The default is:

```yaml
n_motion_chunk_workers: 1
enable_refine: true
```

Refinement is skipped with one worker. With multiple workers, `enable_refine: true` enables refinement after joining the chunks.

For a single run, override the worker count on the command line:

```bash
pixi run retarget task=unitree_g1/lafan \
  mocap_src_path=/path/to/lafan1 n_motion_chunk_workers=4
```

To use multiple workers without refinement, add `enable_refine=false`. CLI overrides take precedence over YAML values for that run and do not rewrite the YAML.

**To process several complete motions concurrently without chunking**, keep `n_motion_chunk_workers: 1` and use [tools/batch_retarget.py](tools/batch_retarget.py):

```bash
pixi run python tools/batch_retarget.py \
  --dataset lafan1 --input_dir /path/to/lafan1 \
  --num_workers 4 --no_wandb
```

`--num_workers 4` runs up to four motion files at once; `n_motion_chunk_workers` controls chunking inside each file. See [Optional Tools](#7-optional-tools) for AMASS batches.

## 6. Saved Outputs and Playback

Trajectories save as timestamped `.npy` files in **`log/kin_trajectories/unitree_g1/`**, preserving the source frame rate.

Run `pixi run play` to open that folder with the default LAFAN1 robot configuration.

View one file by setting `--path /path/to/motion.npy`, or load a folder:

```bash
pixi run play \
  --path log/kin_trajectories/unitree_g1 \
  --num 50
```

Open **[localhost:8085](http://localhost:8085)**. `--num 50` loads up to 50 motions; **Shuffle** selects another group. Omit `--num` to load every `.npy` immediately inside the folder. The viewer does not scan nested directories.

For AMASS, add `--config conf/task/unitree_g1/amass.yaml`, or use the matching subject YAML.

Runs share the G1 output folder, so previous runs and different datasets can coexist. To inspect a fresh run separately, move previous outputs into a backup folder first, or select the new files explicitly.

Check terminal warnings and `output/skipped_motions.csv` for failed motions.

## 7. Optional Tools

<details>
<summary>AMASS batch processing</summary>

AMASS batches can scan nested directories:

```bash
pixi run python tools/batch_retarget.py \
  --dataset amass --input_dir /path/to/AMASS \
  --recursive --num_workers 4 \
  --smplh_model_path /path/to/smplh --no_wandb
```

For custom scaling, add `--task unitree_g1/subject_scaling/amass/accad/Female1Walking_c3d` and use that subject's input folder. One task applies to the whole batch; use separate batches for people with different calibrations.

Batch commands share a compilation cache, so run one batch command at a time. Use `--no-jit` to disable JIT compilation.

List all options with `pixi run python tools/batch_retarget.py --help`.

</details>

<details>
<summary>Estimate AMASS subject profiles automatically</summary>

```bash
pixi run python tools/model_preparation/calibrate_amass_subjects.py \
  --input_dir /path/to/ACCAD \
  --smplh_model_path /path/to/smplh
```

Generate initial calibration YAMLs from one subject folder or a directory of subject folders. Each folder must contain a consistent fitted gender and body shape.

Existing profiles are preserved unless `--overwrite` is supplied; `--output_dir` selects another destination. Review the estimates in `process` and on retargeted motions before adopting them.

</details>

<details>
<summary>Export an MP4 video</summary>

```bash
pixi run python tools/video_export/render_mujoco.py \
  --path /path/to/motion.npy \
  --config conf/task/unitree_g1/lafan.yaml
```

Use the matching dataset or subject config. Videos save to `recordings/mp4/`. MP4 encoding requires an ImageIO video backend such as `imageio-ffmpeg`.

Use `--help` for resolution and camera options.

The playback viewer's **Record** button instead saves a Viser scene recording (`.viser`) to `recordings/private/`.

</details>

<details>
<summary>Robot preparation and dynamics utilities</summary>

| Utility | Purpose |
| --- | --- |
| [capsule_approximation.py](tools/model_preparation/capsule_approximation.py) | Fit capsule geometry to robot meshes and write an XML. |
| [generate_robot_config.py](tools/model_preparation/generate_robot_config.py) | Generate G1 sites and configuration; running it overwrites `conf/task/unitree_g1/lafan.yaml`. |
| [dynfilter.py](tools/dynamics_processing/dynfilter.py) | Smooth saved trajectories and estimate torques and ground-reaction forces. |
| [contact_detect.py](tools/dynamics_processing/contact_detect.py) | Compute robot/mocap foot-contact labels and save mocap-derived labels. |

View the dynamics tools' options with:

```bash
pixi run python tools/dynamics_processing/dynfilter.py --help
pixi run python tools/dynamics_processing/contact_detect.py --help
```

</details>

## 8. Configuration Reference

[conf/config.yaml](conf/config.yaml) selects the task. [conf/base.yaml](conf/base.yaml) supplies shared defaults, and `conf/task/unitree_g1/` contains dataset and subject overrides. Configuration is composed with Hydra; `process` and `retarget` accept `key=value` overrides, while `play` uses `--flag value` arguments.

| Setting | Default / meaning |
| --- | --- |
| `task` | `unitree_g1/lafan`; select AMASS or a subject profile explicitly. |
| `mocap_src_path` | Bundled LAFAN1 sample; replace with a motion file or folder. |
| `start_index`, `end_index` | `0`, `null`: include the first frame through the last. A numeric end is exclusive. |
| `n_motion_chunk_workers` | `1`: solve each motion without chunking. |
| `enable_refine` | `true`, effective only with more than one motion chunk worker. |
| `stitch_overlap_frames` | `15` frames around each chunk. |
| `enable_stitching` | `true`: blend chunk overlaps. |
| `jit_compile`, `jit_recompile` | `true`, `true`: compile and rebuild solver functions on a direct retarget run. |
| `save_chunks` | `false`: save only the assembled trajectory. |
| `wandb.enable` | `false` for direct commands. |
| `verbose` | `false`: enable extra diagnostics when needed. |

## 9. Contributing

Requests and contributions for new robots and datasets are welcome. [Open an issue](https://github.com/DARoSLab/FRDV/issues) to suggest an addition, or [submit a pull request](https://github.com/DARoSLab/FRDV/compare) with your implementation.

## 10. Citation and License

If you use FRDV in your research, please cite [Fast Motion Retargeting Pipeline for Scalable Humanoid Robot Learning](https://arxiv.org/pdf/2610.28470).

```bibtex
@inproceedings{marew2026frdv,
  title = {Fast Motion Retargeting Pipeline for Scalable Humanoid Robot Learning},
  author = {Marew, Daniel and Park, Sangjun and Perera, Nisal and Kim, Donghyun},
  booktitle = {2026 IEEE-RAS International Conference on Humanoid Robots (Humanoids)},
  year = {2026}
}
```

Code is released under the [MIT License](LICENSE). Datasets and body models retain their respective licenses; consult the [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) and [AMASS](https://amass.is.tue.mpg.de/) sources when downloading or citing them.
