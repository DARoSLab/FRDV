"""AMASS joint names and parent relationships."""

# The first 22 SMPL+H joints, excluding the finger joints.
BODY_JOINT_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
]

BODY_PARENTS = {
    "Pelvis": None,
    "L_Hip": "Pelvis",
    "R_Hip": "Pelvis",
    "Spine1": "Pelvis",
    "L_Knee": "L_Hip",
    "R_Knee": "R_Hip",
    "Spine2": "Spine1",
    "L_Ankle": "L_Knee",
    "R_Ankle": "R_Knee",
    "Spine3": "Spine2",
    "L_Foot": "L_Ankle",
    "R_Foot": "R_Ankle",
    "Neck": "Spine3",
    "L_Collar": "Spine3",
    "R_Collar": "Spine3",
    "Head": "Neck",
    "L_Shoulder": "L_Collar",
    "R_Shoulder": "R_Collar",
    "L_Elbow": "L_Shoulder",
    "R_Elbow": "R_Shoulder",
    "L_Wrist": "L_Elbow",
    "R_Wrist": "R_Elbow",
}

NUM_BODY_JOINTS = len(BODY_JOINT_NAMES)

# Spine3 and all its descendants (upper body)
UPPER_BODY_JOINT_NAMES = [
    "Spine3",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
]
