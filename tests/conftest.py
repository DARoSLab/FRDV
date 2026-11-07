import os
import pytest

# Toy capsule biped written for the test suite (no third-party assets).
MODEL = os.path.join(os.path.dirname(__file__), "assets", "toy_biped.xml")
TARGETS = ["pelvis", "L_hippitch_link", "L_shank_link", "L_foot_link", "L_toe_link", "R_hippitch_link",
           "R_shank_link", "R_foot_link", "R_toe_link", "torso_link", "l_shoulder_site", "r_shoulder_site"]
FEET = ["R_HEEL", "R_TOE_FL", "R_TOE_FR", "R_TOE_HL", "R_TOE_HR",
        "L_HEEL", "L_TOE_FL", "L_TOE_FR", "L_TOE_HL", "L_TOE_HR"]


@pytest.fixture(scope="session")
def ik_cfg():

    from frdv import IKConfig
    return IKConfig(xml_path=MODEL, targets=TARGETS, target_weights=[1000.0] * len(TARGETS),
                    foot_points=FEET)


def menagerie_or_skip(robot):
    """Path of a MuJoCo Menagerie model (downloaded once); skips the test when offline."""
    from frdv.assets import menagerie
    try:
        return menagerie(robot, quiet=True)
    except OSError as e:
        pytest.skip(f"cannot download {robot} from MuJoCo Menagerie: {e}")
