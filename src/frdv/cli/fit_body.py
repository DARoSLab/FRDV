"""Offline body-profile fitting for robot-specific LAFAN1 and original SMPL+H AMASS.

Run ``python -m frdv.cli.fit_body --help``. SciPy is required only for this offline tool.
Profiles contain whole-body source-XYZ factors and world-space body/upper-body offsets.
The existing augment command fits augmentation markers separately.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from .. import motion as M
from ..motion._smplh import _load_smplh_model_data
from ..motion.amass import validate_subject_files
from ..motion.amass_joints import BODY_JOINT_NAMES
from ..retarget import Retargeter
from .common import ik_config, robot_height, robot_keypoints, robot_points, to_dict
from .config import available, compose


LAFAN_UPPER_JOINTS = ['Spine2', 'Neck', 'Head', 'Head_EndSite', 'LeftShoulder', 'LeftArm',
                    'LeftForeArm', 'LeftHand', 'LeftHand_EndSite', 'RightShoulder', 'RightArm',
                    'RightForeArm', 'RightHand', 'RightHand_EndSite']


def smplh_reference(models, gender='neutral', betas=None):
    model = _load_smplh_model_data(models, gender)
    betas = np.zeros(16) if betas is None else np.asarray(betas)
    joints = model['J_template'] + np.einsum('jcb,b->jc', model['J_shapedirs'], betas)
    rest = (joints - joints[0]) @ np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]]).T
    return M.Motion(list(BODY_JOINT_NAMES), rest[None], 30., rest, model['parents'].tolist())


def solve_affine(predict, expected, weights, initial):
    """Bounded weighted fit; unobservable upper shifts stay zero (e.g. legs-only robots).

    Axis-aligned source rotations and positive scales make the current target transforms
    affine. Verify this property rather than assuming it for a custom tilted rotation.
    """
    from scipy.optimize import lsq_linear
    initial = np.asarray(initial, float)
    f0 = np.asarray(predict(initial), float).ravel()
    expected = np.asarray(expected, float).ravel()
    weight = np.broadcast_to(np.asarray(weights, float), np.asarray(predict(initial)).shape).ravel()
    if f0.shape != expected.shape or not np.isfinite(expected).all() or np.any(weight <= 0):
        raise ValueError('Body fit requires finite matching targets and positive weights')
    A = np.column_stack([np.asarray(predict(initial + np.eye(9)[i])).ravel() - f0 for i in range(9)])
    b = f0 - A @ initial
    check = initial + np.array([.12, .07, .18, .01, -.02, .03, -.01, .02, -.03])
    if not np.allclose(predict(check), (A @ check + b).reshape(np.asarray(predict(check)).shape), atol=1e-9, rtol=1e-9):
        raise ValueError('Non-affine calibration transform; use an axis-aligned source rotation')
    columns = np.flatnonzero(np.linalg.norm(A, axis=0) > 1e-9)
    if not set(range(6)) <= set(columns):
        raise ValueError('Reference does not determine all XYZ scales and whole-body offsets')
    W = np.sqrt(weight / weight.mean())
    Aw, yw = A[:, columns] * W[:, None], (expected - b) * W
    rank = int(np.linalg.matrix_rank(Aw))
    if rank != len(columns):
        raise ValueError(f'Calibration reference is underdetermined: rank {rank}/{len(columns)}')
    lower = np.array([1e-6 if i < 3 else -np.inf for i in columns])
    fitted = lsq_linear(Aw, yw, bounds=(lower, np.full(len(columns), np.inf)), tol=1e-12, max_iter=500)
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError(f'Calibration fitting failed: {fitted.message}')
    result = np.zeros(9); result[columns] = fitted.x
    if np.any(result[:3] < 1e-5):
        raise ValueError('Calibration collapsed a source axis; review the robot reference/mapping')
    def rms(x):
        delta = (np.asarray(predict(x)).ravel() - expected).reshape(-1, 3)
        per_point = weight.reshape(-1, 3)[:, 0]
        return float(np.sqrt(np.average(np.sum(delta ** 2, axis=1), weights=per_point)) * 1000)
    before, after = rms(initial), rms(result)
    if after > before + 1e-6:
        raise ValueError('Fitted calibration is worse than its geometry-derived initialization')
    return result, dict(initial_reference_rms_mm=before, reference_rms_mm=after,
                        parameter_rank=rank, upper_body_fitted=bool(np.any(columns >= 6)))


def apply_values(dataset, values):
    ds = copy.deepcopy(dataset)
    ds.update(scale=dict(mode='manual', value=list(values[:3]), per_axis=[1., 1., 1.]),
              offset=list(values[3:6]), upper_body_offset=list(values[6:9]),
              robot_target_offset=[0., 0., 0.])
    return ds


def _solver(cfg, keys):
    ik = ik_config(cfg['robot'], cfg['solver'], keys, cfg['dataset'], calibration=cfg['calibration'])
    r = Retargeter(ik)
    from .. import _native
    native = _native.Solver(r.problem) if _native.available() and _native.supported(r.problem) else None
    return ik, r, native


def _solve(ik, retargeter, native, targets, fps):
    ik = replace(ik, max_joint_step=20. / fps)
    if native is not None:
        return native.run(ik, targets, native.workspace())
    # Rebuild derived smoothing weights/bounds as well as the public config.
    return Retargeter(ik).run(targets)


def robot_reference(cfg, neutral):
    """Infer a feasible offline reference using V2 limb scaling and robot tracking points.

    Shipped G1 profiles are unchanged unless generation explicitly overwrites their files.
    """
    ds, rb = copy.deepcopy(cfg['dataset']), cfg['robot']
    ratio = robot_height(rb) / (np.ptp(neutral.rest[:, 2]) + .1)
    ds.update(scale=dict(mode='limbs', per_axis=[1., 1., 1.]), offset=[0., 0., 0.],
              upper_body_offset=[0., 0., 0.], robot_target_offset=[0., 0., 0.],
              ground=dict(mode='frame0', joints=['L_Ankle', 'R_Ankle'], margin=.05 * ratio))
    tg = M.build_targets(neutral, ds, robot_height(rb), list(rb['keypoints']), robot_points=robot_points(rb))
    ik, r, native = _solver(cfg, tg.keypoints)
    ik = replace(ik, warmup_iters=500, boost_frames=10, boost_iters=100, accel_weight=0.)
    q = _solve(ik, r, native, np.repeat(tg.X, 24, axis=0), 30.)[-1]
    r.problem.kin.forward(q, with_jac=False)
    keys = [k for k, v in rb['keypoints'].items() if not v.get('calibrate')]
    points = r.problem.kin.point_set([(rb['keypoints'][k]['link'], rb['keypoints'][k].get('offset', [0., 0., 0.])) for k in keys])
    return dict(zip(keys, r.problem.kin.positions(points))), q


def fit_amass(reference, cfg, points):
    rb, ds = cfg['robot'], cfg['dataset']
    ratio = robot_height(rb) / (np.ptp(reference.rest[:, 2]) + .1)
    initial = np.r_[np.full(3, ratio), np.zeros(6)]
    def target(values):
        d = apply_values(ds, values)
        d['ground'] = dict(mode='frame0', joints=['L_Ankle', 'R_Ankle'], margin=.05 * values[2])
        return M.build_targets(reference, d, robot_height(rb), list(rb['keypoints']))
    tg = target(initial)
    expected = np.array([points[k] for k in tg.keypoints])[None]
    kp = robot_keypoints(rb, ds)
    weights = np.array([kp[k]['weight'] for k in tg.keypoints])[None, :, None]
    return solve_affine(lambda x: target(x).X, expected, weights, initial)


def fit_lafan(paths, cfg, clips=20, samples=64):
    """Fit against actual V2 automatic retargeting references across diverse clips.

    Zero-channel BVH offsets are not an anatomical T-pose for this dataset. Use motion
    references instead of treating those offsets as standing human joint positions.
    """
    pick = [paths[i] for i in np.unique(np.linspace(0, len(paths)-1, min(clips, len(paths))).astype(int))]
    rb, ds = cfg['robot'], copy.deepcopy(cfg['dataset'])
    ds.update(scale=dict(mode='limbs', per_axis=[1., 1., 1.]), offset=[0., 0., 0.],
              upper_body_offset=[0., 0., 0.], robot_target_offset=[0., 0., 0.])
    references, expected, heights = [], [], []
    ik = r = native = None
    for path in pick:
        mot = M.load(str(path), 'bvh')
        tg = M.build_targets(mot, ds, robot_height(rb), list(rb['keypoints']), robot_points=robot_points(rb))
        if ik is None:
            ik, r, native = _solver(cfg, tg.keypoints)
        q = _solve(ik, r, native, tg.X, mot.fps)
        sample = np.unique(np.linspace(0, len(q)-1, min(samples, len(q))).astype(int))
        references.append(M.Motion(mot.names, mot.positions[sample], mot.fps, mot.rest, mot.parents))
        positions = []
        for i in sample:
            r.problem.kin.forward(q[i], with_jac=False)
            positions.append(r.problem.kin.positions(r.problem.pts)[:r.problem.S])
        expected.append(np.asarray(positions))
        heights.append(tg.human_height)
    expected = np.concatenate(expected)
    ratio = robot_height(rb) / float(np.median(heights))
    initial = np.r_[np.full(3, ratio), np.zeros(6)]
    def predict(values):
        d = apply_values(ds, values)
        return np.concatenate([M.build_targets(m, d, robot_height(rb), list(rb['keypoints'])).X for m in references])
    kp = robot_keypoints(rb, ds)
    weights = np.array([kp[k]['weight'] for k in tg.keypoints])[None, :, None]
    values, meta = solve_affine(predict, expected, weights, initial)
    meta.update(reference_clips=len(pick), reference_frames=len(expected), source_clips=[p.name for p in pick])
    return values, meta


def save(path, values, metadata, base=None):
    import yaml
    document = copy.deepcopy(base) if base is not None else {}
    ds = document.setdefault('dataset', {})
    ds.update(manual_scale=values[:3].tolist(), offset=values[3:6].tolist(),
              upper_body_offset=values[6:9].tolist(), robot_target_offset=[0., 0., 0.])
    if base is None:
        ds['scale'] = dict(mode='manual')
    document.setdefault('calibration', {})['body_fit'] = metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('# @package _global_\n# Automatically fitted V2 body calibration; review on actual motions.\n'
                    + yaml.safe_dump(document, sort_keys=False))


def main():
    import yaml
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--robots', nargs='+', required=True, choices=available('robot'))
    ap.add_argument('--lafan1', type=Path, required=True)
    ap.add_argument('--amass', type=Path, required=True, help='Collection directory containing subject folders')
    ap.add_argument('--models', required=True, help='Original SMPL+H model root')
    ap.add_argument('--collection', default='accad')
    ap.add_argument('--out', type=Path, required=True, help='Output calibration root, e.g. frdv_conf/calibration')
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--clips', type=int, default=20)
    ap.add_argument('--samples', type=int, default=64)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    if args.clips < 1 or args.samples < 2:
        ap.error('clips must be positive and samples must be at least 2')
    neutral = smplh_reference(args.models)
    lafan = sorted(args.lafan1.glob('*.bvh'))
    if not lafan:
        ap.error('No LAFAN1 BVH clips found')
    subjects = []
    for folder in sorted(args.amass.iterdir()):
        files = sorted(folder.glob('*.npz')) if folder.is_dir() else []
        if files:
            gender, betas = validate_subject_files(files)
            subjects.append((folder.name, gender, betas, len(files)))
    if not subjects:
        ap.error('No AMASS subject folders found')
    from .config import CONF_DIR
    robots = {}
    for robot in args.robots:
        cfg = to_dict(compose([f'robot={robot}', 'dataset=amass', 'data=unused']))
        name = cfg['robot']['name']
        planned = [args.out/name/ds/'base.yaml' for ds in ('amass', 'lafan1')]
        planned += [args.out/name/'amass/subjects'/args.collection/f'{s[0]}.yaml' for s in subjects]
        if not args.overwrite and any(p.exists() for p in planned):
            ap.error(f'Output calibration exists for {name}; use --overwrite to replace')
        robots[robot] = cfg
    report = dict(robots={}, collection=args.collection)
    # Generate every robot's bases before fitting the collection-specific overrides.
    for robot, cfg in robots.items():
        name = cfg['robot']['name']
        points, q = robot_reference(cfg, neutral)
        model_hash = hashlib.sha256(Path(cfg['robot']['xml']).read_bytes()).hexdigest()
        common = dict(robot_model_sha256=model_hash, reference_qpos=q.tolist())
        entry = report['robots'][name] = dict(bases={}, subjects={})
        entry['_points'] = {k: v.tolist() for k, v in points.items()}
        for dataset in ('amass', 'lafan1'):
            c = to_dict(compose([f'robot={robot}', f'dataset={dataset}', 'data=unused']))
            base = yaml.safe_load((CONF_DIR/'calibration'/name/dataset/'base.yaml').read_text())
            if dataset == 'amass':
                values, meta = fit_amass(neutral, c, points)
                meta.update(method='smplh_rest_to_v2_reference', source_reference='neutral SMPL+H, zero 16 betas', **common)
            else:
                c['dataset']['upper_body_joints'] = LAFAN_UPPER_JOINTS
                values, meta = fit_lafan(lafan, c, args.clips, args.samples)
                meta.update(method='v2_automatic_motion_reference', **common)
            save(args.out/name/dataset/'base.yaml', values, meta, base=base)
            entry['bases'][dataset] = dict(values=values.tolist(), **meta)
            print(f'BASE {name}/{dataset}: scales {values[:3].round(4)}, reference RMS {meta["reference_rms_mm"]:.1f} mm', flush=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    for robot, cfg in robots.items():
        name = cfg['robot']['name']; entry = report['robots'][name]
        points = {k: np.array(v) for k, v in entry.pop('_points').items()}
        for subject, gender, betas, n_clips in subjects:
            source = smplh_reference(args.models, gender, betas)
            values, meta = fit_amass(source, cfg, points)
            meta.update(method='smplh_rest_to_v2_reference', gender=gender, clips=n_clips,
                        robot_model_sha256=entry['bases']['amass']['robot_model_sha256'])
            save(args.out/name/'amass/subjects'/args.collection/f'{subject}.yaml', values, meta)
            entry['subjects'][subject] = dict(values=values.tolist(), **meta)
        print(f'SUBJECTS {name}: {len(subjects)} fitted profiles', flush=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
