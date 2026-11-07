# Calibration belongs to a robot

Body scales, alignment offsets, and optional augmentation-marker fits live under one directory:

```text
src/frdv/conf/calibration/
  unitree_g1/
    lafan1/base.yaml
    amass/
      base.yaml
      subjects/accad/Female1Walking_c3d.yaml
      subjects/accad/...                   # 20 fitted V2 profiles
```

Folder names use `robot.name` (for example `unitree_g1`), while commands retain `robot=g1`.
G1 is the only shipped preset.
Dataset files contain source-format conventions and an automatic scaling policy. They contain
no fixed robot scale or alignment offsets. Each robot/dataset base owns its numerical calibration.

## Inheritance

For the selected robot and dataset, configuration loads in this order:

1. Shared source-format conventions and robot model/link definitions.
2. Shipped robot/dataset `base.yaml`, followed by its user override if present.
3. Selected subject's shipped profile, followed by its user override if present.
4. An explicit `scaling=auto` or `scaling=manual` selection.
5. Explicit command-line values.

User files mirror the same tree under `frdv_conf/calibration/` or
`$FRDV_CONFIG_DIR/calibration/`. Subject files need only contain values that differ from their
robot's base. A profile for another robot is never used as a fallback. Owner metadata is checked
when loading files, and unknown subjects fail with an error.

```bash
uv run retarget robot=g1 dataset=amass subject=accad/Female1Walking_c3d data=/path/to/subject
uv run retarget --cfg robot=g1 dataset=amass subject=accad/Female1Walking_c3d
```

The previous `subject=amass/accad/Female1Walking_c3d` command also works. An explicit owned
selector is `subject=unitree_g1/amass/subjects/accad/Female1Walking_c3d`.
`uv run retarget --help` lists profiles with their robot ownership.

## Saved scales and automatic scales

G1/LAFAN1 has its own manual starting factors and defaults to automatic limb scaling. G1/AMASS
has a manual base alignment and 20 V2 subject profiles fitted using each SMPL+H shape's synthetic
rest pose and V2's G1 reference tracking points. These saved profiles are reused on every run;
retargeting does not refit them. No recorded starting T-pose is required.

Select `scaling=manual` to use the selected profile's saved factors. An ACCAD profile selects its
saved manual factors automatically. Calibration ownership checks remain in place; adding a
custom robot never makes G1's scales its fallback.

`scaling=auto` computes per-clip ratios during target construction. It does not save a new
body-profile YAML. Selecting it with a subject profile retains that profile's alignment offsets.
Review both approaches on actual motions; an improved rest-pose fit alone does not establish
better dynamic motion quality.

## Generate body profiles

The offline generator fits whole-skeleton XYZ scales, a whole-body translation, and an upper-body
translation. SciPy is required for fitting (`pip install scipy`); ordinary retargeting needs no
SciPy. The generated values are reusable starting fits and should be reviewed on moving clips.

Body-profile fitting uses a fixed reference protocol: a 0.10 m height margin, a 0.05 m grounding reference scaled with body size, and a 20 rad/s step cap.

```bash
uv run python -m frdv.cli.fit_body \
  --robots g1 \
  --lafan1 /path/to/lafan1 --amass /path/to/ACCAD \
  --models /path/to/smplh --collection accad \
  --out frdv_conf/calibration --report outputs/body_fit_report.json
```

Use `--overwrite` to replace existing output profiles deliberately. Bases load first and subject
files inherit them. Use a separate output directory to review new fits before replacing G1's
existing profiles.

The offline tool uses a neutral SMPL+H body with zero betas for its AMASS base. Each subject
override uses that folder's actual gender and all sixteen betas, validating a consistent body
fit across its clips. New references are generated with V2 limb scaling and IK; stored reference
qpos and robot-XML hashes record their provenance. The shipped G1 subject profiles retain their
existing fit against G1's configured reference pose.

For LAFAN1, the tool uses V2 automatic retargeting references from twenty evenly spaced clips, sampling
up to 64 frames per clip. This avoids treating BVH zero-channel offsets as an anatomical T-pose.
The tool then fits one XYZ vector and offsets across these motion references. It preserves the
existing default scaling mode and augmentation-marker data. Refitting body alignment does not
refit optional augmentation markers; refit those separately if enabling augmentation.

Upper-body shifts are fitted where observable. A converged reference fit still requires review
on actual motions; it does not establish dynamic balance or correct foot contact.

## File contents

A base contains `dataset.manual_scale`, `dataset.scale`, alignment values such as
`dataset.offset`, and optional `calibration.markers`. Source units and the base source-to-world
rotation remain in `dataset/<name>.yaml`; profiles can override rotation when heading needs tuning.

A subject override can be as small as:

```yaml
# @package _global_
dataset:
  manual_scale: [0.8, 0.72, 0.8]
  scale:
    mode: manual
  offset: [0.03, 0.0, 0.01]
```

Save this under `frdv_conf/calibration/unitree_g1/amass/subjects/cmu/person42.yaml`, tune it for
that subject, and select `robot=g1 dataset=amass subject=cmu/person42`. Other collections use
the same layout without loader changes. LAFAN1 can use the robot's shared base for all performers.

## Fit augmentation markers

Augmentation remains off by default. Previously shipped `calibration/<robot>__lafan1.yaml`
marker offsets are now nested inside that robot's `lafan1/base.yaml` under
`calibration.markers`, including their fit metadata. Marker fitting updates only this section,
preserving existing scales and alignment. With a selected subject, fitting writes that subject's
user override; otherwise it writes the robot/dataset user base.

```bash
uv run augment robot=g1 data=/path/to/lafan1 dataset.augmentation=true
```

Existing custom calibration files must be moved to the robot-owned tree. Convert flat marker
files to `calibration.markers`, and move old `subject/amass/<collection>/<name>.yaml` values to
`calibration/<robot>/amass/subjects/<collection>/<name>.yaml`. Legacy flat files are not loaded.
