from types import SimpleNamespace

import numpy as np

from frdv.cli.diagnostics import warn_measurements, warn_source_overwrite


def motion():
    return SimpleNamespace(human_height=1.75,scale=np.array([0.8]*3),segment_scale=None,fps=30.,
                           keypoints=['pelvis','left_ankle'],
                           X=np.array([[[0.,0.,1.],[0.,0.,0.]],[[0.01,0.,1.],[0.01,0.,0.]]]))


def test_unusual_measurements_warn_without_modifying_targets(capsys):
    tg = motion(); tg.human_height = 0.4; tg.scale[:] = 3.; tg.fps = 500.
    tg.X[1,:,0] += 10
    original = tg.X.copy()
    warn_measurements(tg,{'height':{'source':'frame0'}},1.32,'lying.bvh')
    warning = capsys.readouterr().err
    assert all(text in warning for text in ('WARNING','lying.bvh','0.400 m','first motion frame','scale factors','frame rate','pelvis moves','Continuing'))
    np.testing.assert_array_equal(tg.X,original)


def test_typical_measurements_do_not_warn(capsys):
    warn_measurements(motion(),{},1.32,'walk.bvh')
    assert not capsys.readouterr().err


def test_output_collision_is_a_warning_only(tmp_path,capsys):
    root = tmp_path/'data'; root.mkdir()
    path = root/'clip.npz'; path.write_bytes(b'source archive')
    warn_source_overwrite([str(path)],str(root),str(root))
    assert 'will overwrite 1 input motion' in capsys.readouterr().err
    assert path.read_bytes() == b'source archive'
    warn_source_overwrite([str(path)],str(tmp_path/'out'),str(root))
    assert not capsys.readouterr().err


def test_source_directory_alias_is_warned(tmp_path,capsys):
    import pytest
    root = tmp_path/'data'; root.mkdir()
    path = root/'clip.npz'; path.write_bytes(b'source archive')
    alias = tmp_path/'alias'
    try: alias.symlink_to(root,target_is_directory=True)
    except OSError: pytest.skip('directory symlinks unavailable')
    warn_source_overwrite([str(path)],str(alias),str(root))
    assert 'will overwrite' in capsys.readouterr().err
