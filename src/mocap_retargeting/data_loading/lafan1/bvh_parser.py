"""BVH file parsing used by LAFAN1."""

import numpy as np


class BvhParser:
    """Read BVH hierarchy and channels without changing coordinate conventions."""

    def __init__(self, file_path):
        self.file_path = file_path
        self.joints = {}
        self.root_joint = None
        self.frame_time = 0.0
        self.frames = []
        self._load()

    def _load(self):
        with open(self.file_path, "r") as f:
            content = f.read()

        # Split into Hierarchy and Motion sections
        hierarchy_str, motion_str = content.split("MOTION")

        self._parse_hierarchy(hierarchy_str)
        self._parse_motion(motion_str)

    def _parse_hierarchy(self, hierarchy_str):
        lines = hierarchy_str.split("\n")
        joint_stack = []
        current_joint = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("ROOT"):
                name = line.split()[1]
                joint = {
                    "name": name,
                    "offset": np.zeros(3),
                    "channels": [],
                    "children": [],
                    "parent": None,
                }
                self.joints[name] = joint
                self.root_joint = joint
                current_joint = joint
                joint_stack.append(joint)
            elif line.startswith("JOINT"):
                name = line.split()[1]
                joint = {
                    "name": name,
                    "offset": np.zeros(3),
                    "channels": [],
                    "children": [],
                    "parent": current_joint,
                }
                self.joints[name] = joint
                if current_joint:
                    current_joint["children"].append(joint)
                current_joint = joint
                joint_stack.append(joint)
            elif line.startswith("End Site"):
                name = f"{current_joint['name']}_EndSite"
                joint = {
                    "name": name,
                    "offset": np.zeros(3),
                    "channels": [],
                    "children": [],
                    "parent": current_joint,
                }
                self.joints[name] = joint
                if current_joint:
                    current_joint["children"].append(joint)
                current_joint = joint
                joint_stack.append(joint)
            elif line.startswith("OFFSET"):
                parts = line.split()
                current_joint["offset"] = np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
            elif line.startswith("CHANNELS"):
                parts = line.split()
                current_joint["channels"] = parts[2:]
            elif line.startswith("{"):
                pass
            elif line.startswith("}"):
                joint_stack.pop()
                if joint_stack:
                    current_joint = joint_stack[-1]
                else:
                    current_joint = None

    def _parse_motion(self, motion_str):
        lines = motion_str.strip().split("\n")
        for line in lines:
            if line.startswith("Frames:"):
                self.num_frames = int(line.split()[1])
            elif line.startswith("Frame Time:"):
                self.frame_time = float(line.split()[2])
            else:
                # Data line
                data = [float(x) for x in line.split()]
                self.frames.append(data)

        self.motion_data = np.array(self.frames)
