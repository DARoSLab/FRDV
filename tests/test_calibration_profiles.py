"""Calibration inheritance and isolation between robot morphologies."""
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest
import yaml

from frdv.cli.calibrate import calibrate, save
from frdv.cli.common import calibration_paths, load_calibration, to_dict
from frdv.cli.config import CONF_DIR, available, compose


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setenv('FRDV_CONFIG_DIR', str(tmp_path / 'user'))
    monkeypatch.setenv('FRDV_NO_CONFIG_CACHE', '1')


def write_profile(tmp_path, relative, document):
    path = tmp_path / 'user/calibration' / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('# @package _global_\n' + yaml.safe_dump(document, sort_keys=False))
    return path


def custom_robot(tmp_path):
    path=tmp_path/'user/robot/custom.yaml'
    path.parent.mkdir(parents=True,exist_ok=True)
    doc=yaml.safe_load((CONF_DIR/'robot/g1.yaml').read_text())
    doc['name']='custom_robot'
    path.write_text(yaml.safe_dump(doc))


@pytest.mark.parametrize('robot', ['g1'])
@pytest.mark.parametrize('dataset', ['amass', 'lafan1'])
def test_every_robot_uses_only_its_own_base(robot, dataset):
    cfg = compose([f'robot={robot}', f'dataset={dataset}', 'data=unused'])
    assert cfg.calibration.robot == cfg.robot.name
    assert cfg.calibration.dataset == dataset
    path = Path(cfg.calibration.sources[0])
    assert path.relative_to(CONF_DIR / 'calibration').parts == (cfg.robot.name, dataset, 'base.yaml')
    assert np.isfinite(cfg.dataset.manual_scale).all()
    assert np.all(np.asarray(cfg.dataset.manual_scale) > 0)
    manual = compose([f'robot={robot}', f'dataset={dataset}', 'scaling=manual', 'data=unused'])
    assert manual.dataset.scale.value == cfg.dataset.manual_scale
    assert OmegaConf.is_interpolation(cfg.robot, 'xml')  # No model download during composition.


def test_shared_dataset_presets_have_no_fixed_robot_calibration():
    for name in available('dataset'):
        ds = OmegaConf.load(CONF_DIR / 'dataset' / f'{name}.yaml')
        assert ds.manual_scale is None
        assert ds.scale.mode == 'limbs'
        assert not {'offset', 'upper_body_offset', 'robot_target_offset'} & set(ds)


def test_subject_uses_v2_fit_and_inherits_g1_base(tmp_path):
    cfg = compose(['dataset=amass', 'subject=accad/Female1Walking_c3d', 'data=unused'])
    np.testing.assert_allclose(cfg.dataset.manual_scale, [0.8083732731687319, 0.7201593693841238, 0.8039297835891811])
    assert cfg.calibration.body_fit.method == 'smplh_rest_to_v2_reference'
    assert cfg.dataset.scale.mode == 'manual'
    assert cfg.dataset.scale.value == cfg.dataset.manual_scale
    assert list(cfg.dataset.robot_target_offset) == [0., 0., 0.]
    assert cfg.dataset.weights.left_ankle == 2000
    assert cfg.subject.robot == 'unitree_g1'
    assert cfg.subject.collection == 'accad'
    assert len(cfg.calibration.sources) == 2
    custom_robot(tmp_path)
    for args in (['robot=custom', 'dataset=amass', 'subject=unitree_g1/amass/subjects/accad/Female1Walking_c3d'],):
        with pytest.raises(ValueError, match='No subject calibration|requires robot'):
            compose(args + ['data=unused'])


def test_user_base_subject_and_cli_precedence(tmp_path):
    write_profile(tmp_path, 'unitree_g1/amass/base.yaml', dict(dataset=dict(offset=[.1, .2, .3])))
    write_profile(tmp_path, 'unitree_g1/amass/subjects/cmu/person42.yaml', dict(
        dataset=dict(manual_scale=[.6, .7, .8], offset=[.4, .5, .6])))
    base = compose(['dataset=amass', 'data=unused'])
    assert list(base.dataset.offset) == [.1, .2, .3]
    cfg = compose(['dataset=amass', 'subject=cmu/person42', 'data=unused'])
    assert list(cfg.dataset.offset) == [.4, .5, .6]
    assert list(cfg.dataset.scale.value) == [.6, .7, .8]
    cfg = compose(['dataset=amass', 'subject=cmu/person42', 'scaling=auto',
                   'dataset.offset=[.7,.8,.9]', 'data=unused'])
    assert cfg.dataset.scale.mode == 'limbs'
    assert list(cfg.dataset.offset) == [.7, .8, .9]
    assert cfg.subject.name == 'person42'
    custom_robot(tmp_path)
    other = compose(['robot=custom', 'dataset=amass', 'data=unused'])
    assert list(other.dataset.get('offset', [0., 0., 0.])) != [.7, .8, .9]
    assert other.dataset.manual_scale is None


def test_custom_manual_values_must_be_explicit_or_owned(tmp_path):
    custom_robot(tmp_path)
    cfg = compose(['robot=custom', 'dataset=amass', 'scaling=manual',
                   'dataset.scale.value=[1.01,1.02,1.03]', 'data=unused'])
    assert list(cfg.dataset.scale.value) == [1.01, 1.02, 1.03]
    write_profile(tmp_path, 'custom_robot/amass/base.yaml', dict(dataset=dict(manual_scale=[1.1, 1.2, 1.3])))
    other = compose(['robot=custom', 'dataset=amass', 'scaling=manual', 'data=unused'])
    assert list(other.dataset.scale.value) == [1.1, 1.2, 1.3]
    g1 = compose(['dataset=amass', 'data=unused'])
    assert list(g1.dataset.scale.value) == [.87, .83, .75]


def test_incorrect_owner_metadata_and_unsafe_paths_are_rejected(tmp_path):
    custom_robot(tmp_path)
    write_profile(tmp_path, 'custom_robot/amass/base.yaml', dict(calibration=dict(robot='unitree_g1')))
    with pytest.raises(ValueError, match='requires robot'):
        compose(['robot=custom', 'dataset=amass', 'data=unused'])
    for subject in ('../accad/foo', '/accad/foo', 'accad/../foo'):
        with pytest.raises(ValueError, match='Invalid subject'):
            compose(['dataset=amass', f'subject={subject}', 'data=unused'])


def test_marker_calibration_save_preserves_body_values_and_is_immediately_reused(tmp_path, monkeypatch):
    path = write_profile(tmp_path, 'unitree_g1/lafan1/base.yaml', dict(
        dataset=dict(manual_scale=[.61, .62, .63], offset=[.01, .02, .03])))
    cfg = to_dict(compose(['data=unused', 'dataset.augmentation=true']))
    monkeypatch.setattr('frdv.cli.calibrate.fit_offsets',
        lambda *a, **k: ({'left_thumb': [.1, .2, .3]}, {'left_thumb': .001}, 25))
    result = calibrate(cfg, ['unused'], log=lambda _: None)
    assert result == path
    doc = yaml.safe_load(path.read_text())
    assert doc['dataset'] == dict(manual_scale=[.61, .62, .63], offset=[.01, .02, .03])
    assert doc['calibration']['markers']['offsets']['left_thumb'] == [.1, .2, .3]
    offsets, _ = load_calibration(cfg['robot'], cfg['dataset'], cfg['calibration'])
    assert offsets['left_thumb'] == [.1, .2, .3]
    # On the next run the user's marker override merges with the other shipped offsets.
    next_cfg = to_dict(compose(['data=unused', 'dataset.augmentation=true']))
    offsets, _ = load_calibration(next_cfg['robot'], next_cfg['dataset'], next_cfg['calibration'])
    assert offsets['left_thumb'] == [.1, .2, .3]
    assert 'right_thumb' in offsets


def test_subject_marker_fits_write_to_its_own_profile(tmp_path, monkeypatch):
    profile = 'unitree_g1/lafan1/subjects/people/person1.yaml'
    path = write_profile(tmp_path, profile, dict(dataset=dict(manual_scale=[.65, .75, .85])))
    cfg = to_dict(compose(['subject=people/person1', 'dataset.augmentation=true', 'data=unused']))
    monkeypatch.setattr('frdv.cli.calibrate.fit_offsets',
        lambda *a, **k: ({'left_thumb': [.11, .22, .33]}, {'left_thumb': .001}, 25))
    result = calibrate(cfg, ['unused'], log=lambda _: None)
    assert result == path
    assert yaml.safe_load(path.read_text())['dataset']['manual_scale'] == [.65, .75, .85]
    assert not (path.parents[2] / 'base.yaml').exists()  # No user base was overwritten.
    with pytest.raises(ValueError, match='selected robot'):
        calibration_paths(cfg['robot'], cfg['dataset'], 'custom_robot/lafan1/base.yaml')
