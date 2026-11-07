"""Terminal warnings for questionable, but usable, motion measurements."""

import os
import sys

import numpy as np


def warn_source_overwrite(files, out, data):
    root = os.path.abspath(os.path.expanduser(data))
    collisions = []
    for source in files:
        relative = os.path.relpath(source, root) if os.path.isdir(root) else os.path.basename(source)
        destination = os.path.join(out, os.path.splitext(relative)[0] + '.npz')
        same = os.path.realpath(source) == os.path.realpath(destination)
        if not same and os.path.exists(destination):
            same = os.path.samefile(source, destination)
        if same:
            collisions.append(source)
    if collisions:
        print(f"[frdv] WARNING: output will overwrite {len(collisions)} input motion file(s), including "
              f"'{collisions[0]}'. The original source data will be replaced. Choose a different out directory "
              "if this is unintended.", file=sys.stderr, flush=True)


def warn_measurements(targets, dataset, robot_height, path, start=0):
    issues = []
    height = targets.human_height
    if height < 0.8 or height > 2.5:
        source = (dataset.get('height') or {}).get('source', 'frame0')
        reference = 'the first motion frame' if source == 'frame0' else 'the body rest pose'
        issues.append(f"performer height estimate is {height:.3f} m from {reference}; "
                      "please check that reference, source units and axis rotation (a crouched/lying start can affect scaling)")
    scales = np.asarray(list(targets.segment_scale.values()) if targets.segment_scale else targets.scale)
    if np.any(scales < 0.25) or np.any(scales > 2.5):
        issues.append(f"scale factors range from {scales.min():.3f} to {scales.max():.3f}; "
                      "please review the body proportions, units and keypoint mapping")
    fps = float(targets.fps)
    if fps < 10 or fps > 240:
        issues.append(f"frame rate is {fps:g} Hz; please check the source frame-time metadata")
    if len(targets.X):
        z = targets.X[0, :, 2]
        span = float(np.ptp(z))
        if span < 0.2 * robot_height or span > 2 * robot_height or z.min() < -0.25 * robot_height or z.min() > 2 * robot_height:
            issues.append(f"first target frame spans Z={z.min():.3f}..{z.max():.3f} m for a {robot_height:.3f} m robot; "
                          "please check the starting pose, scales and offsets")
    if len(targets.X) > 1 and 'pelvis' in targets.keypoints:
        root = targets.X[:, targets.keypoints.index('pelvis')]
        distances = np.linalg.norm(np.diff(root, axis=0), axis=1)
        frame = int(np.argmax(distances))
        speed = float(distances[frame]) * fps
        if speed > 15:
            issues.append(f"pelvis moves {distances[frame]:.3f} m between source frames {start + frame} and "
                          f"{start + frame + 1} ({speed:.1f} m/s); please check for a discontinuity or incorrect units")
    if issues:
        print(f"[frdv] WARNING: {path}: " + '; '.join(issues) + '. Continuing with the configured settings.',
              file=sys.stderr, flush=True)
