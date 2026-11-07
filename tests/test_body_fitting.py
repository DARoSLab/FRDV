"""Offline fitting must recover observable scales and reject degenerate references."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from frdv.cli.config import CONF_DIR, available, compose
from frdv.cli.fit_body import solve_affine


def test_weighted_fit_recovers_independent_scales_and_offsets():
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(12, 3, 9))
    bias = rng.normal(size=(12, 3))
    truth = np.array([.5, .7, .8, .01, .02, .03, -.01, .04, .05])
    predict = lambda x: np.einsum('ijk,k->ij', matrix, x) + bias
    value, meta = solve_affine(predict, predict(truth), np.linspace(1, 3, 12)[:, None],
                               np.r_[np.ones(3), np.zeros(6)])
    np.testing.assert_allclose(value, truth, atol=1e-10)
    assert meta['parameter_rank'] == 9 and meta['upper_body_fitted']
    assert meta['reference_rms_mm'] < 1e-8


def test_legs_only_reference_does_not_invent_upper_body_shifts():
    rng = np.random.default_rng(9)
    matrix = rng.normal(size=(10, 3, 9)); matrix[:, :, 6:] = 0
    truth = np.array([.3, .4, .5, .01, .02, .03, 0, 0, 0])
    predict = lambda x: np.einsum('ijk,k->ij', matrix, x)
    value, meta = solve_affine(predict, predict(truth), np.ones((10, 1)),
                               np.r_[np.ones(3), np.zeros(6)])
    np.testing.assert_allclose(value, truth, atol=1e-10)
    assert meta['parameter_rank'] == 6 and not meta['upper_body_fitted']


def test_degenerate_and_collapsed_calibrations_are_rejected():
    with pytest.raises(ValueError, match='determine all XYZ'):
        solve_affine(lambda x: np.ones((5, 3)) * x[0], np.ones((5, 3)), np.ones((5, 1)), np.ones(9))
    matrix = np.random.default_rng(10).normal(size=(12, 3, 9))
    predict = lambda x: np.einsum('ijk,k->ij', matrix, x)
    with pytest.raises(ValueError, match='collapsed'):
        solve_affine(predict, predict(np.r_[[-1., .5, .6], np.zeros(6)]), np.ones((12, 1)),
                     np.r_[np.ones(3), np.zeros(6)])


def test_unknown_robot_still_cannot_inherit_a_manual_scale(tmp_path, monkeypatch):
    user = tmp_path / 'config'
    (user / 'robot').mkdir(parents=True)
    doc = yaml.safe_load((CONF_DIR / 'robot/g1.yaml').read_text())
    doc['name'] = 'new_uncalibrated_robot'
    (user / 'robot/custom.yaml').write_text(yaml.safe_dump(doc))
    monkeypatch.setenv('FRDV_CONFIG_DIR', str(user))
    monkeypatch.setenv('FRDV_NO_CONFIG_CACHE', '1')
    auto = compose(['robot=custom', 'dataset=amass', 'data=unused'])
    assert auto.dataset.manual_scale is None
    with pytest.raises(ValueError, match='No manual scaling calibrated'):
        compose(['robot=custom', 'dataset=amass', 'scaling=manual', 'data=unused'])


def test_g1_has_twenty_owned_subject_profiles(monkeypatch, tmp_path):
    monkeypatch.setenv('FRDV_CONFIG_DIR', str(tmp_path / 'config'))
    monkeypatch.setenv('FRDV_NO_CONFIG_CACHE', '1')
    vectors = []
    for robot in available('robot'):
        c = compose([f'robot={robot}', 'dataset=amass', 'subject=accad/Female1Walking_c3d', 'data=unused'])
        files = list((CONF_DIR / 'calibration' / c.robot.name / 'amass/subjects/accad').glob('*.yaml'))
        assert len(files) == 20
        assert c.subject.robot == c.robot.name and c.calibration.robot == c.robot.name
        assert c.dataset.scale.mode == 'manual'
        vectors.append(np.asarray(c.dataset.manual_scale))
    assert len({tuple(v) for v in vectors}) == 1
