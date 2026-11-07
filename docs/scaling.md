# Scaling and alignment for LAFAN1

FRDV provides two scaling workflows. `scaling=auto` estimates limb proportions from each clip.
`scaling=manual` applies fixed XYZ factors that you can tune, save, and reuse. Automatic scaling
remains the LAFAN1 default. G1/AMASS uses its own manual base or selected fitted V2 profile; see the
[AMASS guide](amass.md) for its saved collection/subject profiles.

For best control over motion quality, we recommend a carefully tuned and reviewed manual
profile for each subject/body shape and target robot. Review it on several motions before
reusing it across a collection. Automatic scaling is useful when a reviewed profile is not
available. This recommendation concerns motion quality, not a measured solver-speed advantage;
an unsuitable manual profile can worsen the result.

Edit manual profiles as YAML or command-line values, then inspect them with `uv run check` and
`uv run play`.

## Automatic limb scaling

```bash
uv run retarget robot=g1 scaling=auto data=/path/to/lafan1 out=outputs/auto_g1
```

Automatic mode measures median mapped segment lengths across each clip and adjusts them to the
robot's nominal proportions. Pelvis motion uses a leg-length ratio, and torso placement uses
the robot's nominal offsets. It does not save a per-person body-scaling profile. The resulting
target positions and the run configuration are stored with the output.

The current automatic mode uses the first frame's vertical span for its height fallback and
scale bounds. Crouched or lying starts can affect that estimate. Review unusual starts rather
than assuming automatic scaling is pose-independent.

## Manual scaling and V1 values

The G1/LAFAN1 manual preset starts from V1's calibration:

| Setting | XYZ values | Meaning |
| --- | --- | --- |
| V1 `mocap.scale` | `[0.0076, 0.0078, 0.0074]` | Metres per source centimetre; includes unit conversion. |
| V2 `dataset.units` | `0.01` | Converts source centimetres to metres. |
| V2 `dataset.scale.value` | `[0.76, 0.78, 0.74]` | Dimensionless source-XYZ adjustments after unit conversion. |

For the same source axes and rotation, the effective transformation is:

```text
world_position = dataset.rotation @ (source_position * dataset.units * dataset.scale.value)
```

Manual XYZ factors act along the **source axes before rotation**, preserving V1's convention.
For LAFAN1 the source Y axis is vertical; the configured rotation converts this to world Z.
Do not copy `[0.0076, 0.0078, 0.0074]` directly into V2's scale value: that would apply the unit
conversion twice. The legacy low-level `dataset.scale.mode=fixed` scales world axes after
rotation; use `scaling=manual` for the V1 convention.

```bash
uv run check robot=g1 scaling=manual data=/path/to/lafan1/walk1_subject1.bvh frames=300 \
  'dataset.scale.value=[0.76,0.78,0.74]'

uv run retarget robot=g1 scaling=manual data=/path/to/lafan1/walk1_subject1.bvh \
  'dataset.scale.value=[0.76,0.78,0.74]' out=outputs/manual_preview

uv run play outputs/manual_preview
```

The bundled manual factors are a **G1/LAFAN1 starting point**, not a fit for every performer or
robot. Tune separate profiles for other proportions and target models.

## Tune and save a subject profile

1. Choose representative clips from one person or consistent skeleton/body shape. Include
   several actions and poses. A recorded T-pose is not required; a clear standing reference
   makes visual comparison easier.
2. Start with `scaling=manual`. Adjust `dataset.scale.value` along source X/Y/Z to align body
   proportions, then inspect knees, feet, torso, and arms in the viewer.
3. Adjust `dataset.rotation` if heading or axes need correction. Store a full 3-by-3 rotation
   matrix; apply any yaw adjustment about world Z to the existing source-to-world rotation.
4. Adjust `dataset.offset` for a whole-body translation in world metres. A negative world-Z
   offset lowers the targets. `robot.target_offset` is added separately, so check that setting
   when reproducing an existing alignment.
5. Save the values and inspect several other motions before running the whole subject folder.

Create `frdv_conf/calibration/unitree_g1/lafan1/subjects/people/subject1.yaml`:

```yaml
# @package _global_
dataset:
  manual_scale: [0.76, 0.78, 0.74]
  scale:
    mode: manual
  offset: [0.0, 0.0, -0.02]
```

Include `dataset.rotation` if you changed it. The profile inherits this robot's LAFAN1 base.
User profiles live under `frdv_conf/calibration/` or `$FRDV_CONFIG_DIR/calibration/`:

```bash
uv run retarget robot=g1 subject=people/subject1 data=/path/to/subject1 out=outputs/subject1_g1
```

For a shared LAFAN1 folder, select the subject's filenames:

```bash
uv run retarget robot=g1 subject=people/subject1 data=/path/to/lafan1 \
  'dataset.glob=*_subject1.bvh' out=outputs/subject1_g1
```

The same saved factors apply to every selected motion. Folder names and subject labels do not
automatically select a profile. Configuration order is source format, the selected robot's base,
selected subject override, explicit scaling selection, then command-line values. Saved output retains the configuration used
to generate it. LAFAN1 does not require separate subject profiles.

## Coordinates and grounding

| Setting | Meaning |
| --- | --- |
| `dataset.units` | Metres per raw BVH unit. LAFAN1 uses `0.01`. |
| `dataset.scale.value` | Manual dimensionless XYZ factors in source axes, before rotation. |
| `dataset.scale.per_axis` | Additional axis factors; keep `[1,1,1]` while tuning the manual values. |
| `dataset.rotation` | Source-to-world rotation matrix; world Z points upward. |
| `dataset.offset` | Whole-body translation in world X/Y/Z, in metres. |
| `robot.target_offset` | Additional robot-specific translation of the targets. |
| `dataset.ground.mode` | `frame0` uses the lowest selected joint in frame 0; `clip` uses the lowest over the clip; `none` preserves source height. |

Manual scaling bypasses automatic height-based scale estimates and bounds. Ground alignment is
still a separate operation: the default `frame0` mode can shift a motion incorrectly when it
starts airborne. `clip` can be useful when a later frame reaches the floor, but it is a minimum
height heuristic, not a contact estimator. Inspect grounding on clips with unusual poses.

## Optional hand and arm augmentation

Augmentation is **off by default**. The default G1 mapping tracks 16 body points; it does not
generate or constrain the synthetic hand/arm markers and does not fit their calibration.

Enable the optional 14 points with either scaling mode:

```bash
uv run retarget robot=g1 scaling=auto data=/path/to/lafan1 \
  dataset.augmentation=true out=outputs/augmented_g1
```

These virtual points provide hand-orientation and arm-roll cues. They do not add articulated
fingers and can introduce unwanted motion, so compare the result before enabling them for a
batch. With augmentation enabled, saved marker offsets are reused; missing offsets can be fitted
automatically. This marker calibration is separate from manual body scaling.

When changing a scaling profile with augmentation enabled, explicitly refit the marker offsets
for that setup. Marker fitting writes `calibration.markers` in that robot's base or selected
subject override, preserving the saved body scales and offsets:

```bash
uv run augment robot=g1 subject=people/subject1 data=/path/to/subject1 dataset.augmentation=true
```
