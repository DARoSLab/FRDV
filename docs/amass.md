# AMASS with SMPL+H

FRDV reads original AMASS SMPL+H motions with a NumPy-only loader. The normal pipeline applies
the selected G1 calibration, solves the motion, and saves it for playback.

## Models and input

Set `SMPLH_MODEL_PATH` to the folder containing the original 16-beta SMPL+H model files:

```text
smplh/
  female/model.npz
  male/model.npz
  neutral/model.npz
```

Only genders present in your motions need model files. Each motion chooses its own gender's
model; a missing model raises an error. You can also pass
`dataset.smplh_model_path=/path/to/smplh` on the command line. SMPL-X and 10-beta model files
are not interchangeable with these original AMASS files.

Each motion `.npz` must contain:

| Field | Shape / meaning |
| --- | --- |
| `poses` | `(T, 156)` axis-angle SMPL+H joint rotations, at least one frame. |
| `trans` | `(T, 3)` world translations in metres. |
| `betas` | `(16,)` fitted body-shape coefficients. |
| `gender` | One string: `male`, `female`, or `neutral`. |
| `mocap_framerate` | One positive finite frame rate. `mocap_frame_rate` is also accepted. |

Numeric arrays must be finite. If both frame-rate fields exist, they must agree. Missing metadata
is rejected. The loader reconstructs the first 22 body joints; finger poses and DMPL mesh
deformations do not affect these body targets. All 16 betas contribute to the subject skeleton.
Body-model joint regressions are cached while the model file is unchanged, and FK processes all
frames together using rotations and positions rather than per-frame homogeneous matrices.

## Robot-specific base or saved subject profile

The [AMASS dataset preset](../src/frdv/conf/dataset/amass.yaml) contains shared source-format
conventions. Each robot has its own `calibration/<robot>/amass/base.yaml`. G1's base provides
its starting manual alignment. G1 is the only shipped robot preset, and its values are never a
fallback for a custom robot:

```bash
export SMPLH_MODEL_PATH=/path/to/smplh
uv run retarget robot=g1 dataset=amass data=/path/to/AMASS out=outputs/amass_default
```

The input may be one file, a subject folder, or a collection root. The default glob is recursive
`**/*.npz`; point it at motion data rather than models or previously generated outputs. Output
subfolders mirror the input folder structure, preserving repeated clip names across subjects.

Twenty V2 G1/ACCAD profiles live under
`src/frdv/conf/calibration/unitree_g1/amass/subjects/accad/`. Each overrides G1's AMASS base.
Select the profile explicitly and point `data` at its folder:

```bash
uv run retarget robot=g1 dataset=amass subject=accad/Female1Walking_c3d \
  data=/path/to/ACCAD/Female1Walking_c3d out=outputs/female1_walking
uv run play outputs/female1_walking
```

These profiles were fitted against G1's configured V2 reference tracking points, including local
link offsets. Fitting uses
the subject's SMPL+H shape to generate a synthetic rest pose, rather
than the first recorded pose. Scales and whole-/upper-body offsets are saved and reused. The
source-to-world rotation is inherited from the AMASS convention. Review results on moving clips;
reference-pose fit alone does not establish dynamic motion quality.

The ACCAD names are V1 profile keys, including action-specific folders. Folders are not assumed
to be unique physical people, and matching names do not auto-select profiles. A profile run checks
that all selected clips have identical gender and beta coefficients. Use separate runs/profiles
for different body-shape fits. Dataset/robot declarations also prevent using a G1/AMASS profile
with an incompatible selected dataset or robot.

Profiles are read and reused on every run, without fitting or rewriting their YAML. Run
`uv run retarget --help` to list profiles with their robot ownership. The previous
`subject=amass/accad/<name>` command remains supported. LAFAN1 uses each robot's own base;
there is no requirement to create one LAFAN1 profile per person.

## Manual and automatic scaling

G1/AMASS defaults to its own manual source-XYZ factors `[0.87, 0.83, 0.75]`, or the selected
subject's factors. `scaling=manual` uses the selected saved factors, and `scaling=auto` selects
automatic limb scaling. Metres are already the source units, so no
LAFAN1 centimetre conversion applies.
Explicit scaling overrides the selected profile's scaling mode:

```bash
uv run retarget robot=g1 dataset=amass scaling=auto \
  data=/path/to/ACCAD/Female1Walking_c3d out=outputs/auto

uv run retarget robot=g1 dataset=amass subject=accad/Female1Walking_c3d \
  scaling=manual 'dataset.scale.value=[0.75,0.85,0.80]' \
  data=/path/to/ACCAD/Female1Walking_c3d out=outputs/manual
```

Automatic limb scaling measures segment lengths over the clip and uses the shape-derived rest
skeleton for its height fallback. A T-pose or standing first frame is not required. It does not
create a persistent subject profile. If a subject profile is also selected, its rotations and
offsets remain active when choosing `scaling=auto`; omit the subject to start from shared defaults.
For best control over motion quality, tune and review manual profiles across several motions.
The bundled manual alignments are G1-specific starting points; tune another robot or start with
`robot=<name> scaling=auto` and review its alignment.

AMASS preserves V1's default transformation order:

1. Apply manual source-axis scale, then the configured heading rotation.
2. Add `dataset.upper_body_offset` to Spine3 and its descendants, in world metres.
3. Zero the Pelvis's initial XY position, retaining its vertical motion.
4. Add `dataset.offset` to all joints.

G1's AMASS base overrides the extra robot target offset with `[0,0,0]` to avoid adding G1's extra
translation on top of V1's alignment. Synthetic hand/arm augmentation remains off; the AMASS
preset supplies 15 body targets and no synthetic markers.

`dataset.ground.mode=none` preserves source height, including airborne or crouched starts.
Optional `frame0`/`clip` modes use the V2 floor-height heuristic after alignment. They do not
reproduce V1's optional pre-scale grounding operation and do not estimate contacts.

## Add another collection or subject

Create `frdv_conf/calibration/unitree_g1/amass/subjects/cmu/person42.yaml`
(or use `$FRDV_CONFIG_DIR/calibration/`):

```yaml
# @package _global_
dataset:
  manual_scale: [0.87, 0.83, 0.75]
  scale:
    mode: manual
  offset: [0.04, 0.0, 0.0]
  upper_body_offset: [0.03, 0.0, 0.065]
```

Tune these example values rather than assuming they fit that subject. Include `dataset.rotation`
if heading needs correction, following the [scaling guide](scaling.md). Preview a short clip:

```bash
uv run check robot=g1 dataset=amass subject=cmu/person42 data=/path/to/person42/clip.npz
uv run retarget robot=g1 dataset=amass subject=cmu/person42 \
  data=/path/to/person42 end=300 save_skeleton=true out=outputs/preview
uv run play outputs/preview
```

Source conventions load first, followed by this robot's base, the subject override, an explicit
scaling selection, and command-line values. Adding a collection requires only YAML profiles;
no loader changes or ACCAD-specific routing are involved. Put each robot's profiles under its
own directory. See the [calibration layout and inheritance guide](calibration.md).
