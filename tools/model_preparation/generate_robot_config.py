import os
import yaml
import mujoco
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets/robots"
CONF_DIR = PROJECT_ROOT / "conf/task"

# BVH Joints to map
BVH_JOINTS = [
    "Hips",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToe",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToe",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
]

# Heuristic keywords for mapping
MAPPING_KEYWORDS = {
    "Hips": ["pelvis", "torso", "base", "hip"],
    "LeftUpLeg": [
        "l_hip",
        "left_hip",
        "l_up_leg",
        "left_up_leg",
        "l_thigh",
        "left_thigh",
        "hip_pitch_l",
        "hip_roll_l",
        "hip_yaw_l",
    ],
    "LeftLeg": [
        "l_knee",
        "left_knee",
        "l_calf",
        "left_calf",
        "l_shin",
        "left_shin",
        "l_lower_leg",
        "left_lower_leg",
        "knee_pitch_l",
    ],
    "LeftFoot": [
        "l_foot",
        "left_foot",
        "l_ankle",
        "left_ankle",
        "ankle_pitch_l",
        "ankle_roll_l",
    ],
    "LeftToe": ["l_toe", "left_toe", "toe_l"],
    "RightUpLeg": [
        "r_hip",
        "right_hip",
        "r_up_leg",
        "right_up_leg",
        "r_thigh",
        "right_thigh",
        "hip_pitch_r",
        "hip_roll_r",
        "hip_yaw_r",
    ],
    "RightLeg": [
        "r_knee",
        "right_knee",
        "r_calf",
        "right_calf",
        "r_shin",
        "right_shin",
        "r_lower_leg",
        "right_lower_leg",
        "knee_pitch_r",
    ],
    "RightFoot": [
        "r_foot",
        "right_foot",
        "r_ankle",
        "right_ankle",
        "ankle_pitch_r",
        "ankle_roll_r",
    ],
    "RightToe": ["r_toe", "right_toe", "toe_r"],
    "Spine": ["spine", "waist", "chest", "trunk", "torso"],
    "LeftArm": [
        "l_shoulder",
        "left_shoulder",
        "l_clavicle",
        "left_clavicle",
        "shoulder_pitch_l",
        "shoulder_roll_l",
        "shoulder_yaw_l",
    ],
    "LeftForeArm": [
        "l_forearm",
        "left_forearm",
        "l_elbow",
        "left_elbow",
        "l_lower_arm",
        "left_lower_arm",
        "elbow_pitch_l",
        "elbow_yaw_l",
    ],
    "LeftHand": ["left_hand", "l_wrist", "left_wrist", "wrist_yaw_l"],
    "RightShoulder": [
        "r_shoulder",
        "right_shoulder",
        "r_clavicle",
        "right_clavicle",
        "shoulder_pitch_r",
        "shoulder_roll_r",
        "shoulder_yaw_r",
    ],
    "RightArm": ["right_arm", "r_upper_arm", "right_upper_arm"],
    "RightForeArm": [
        "r_forearm",
        "right_forearm",
        "r_elbow",
        "right_elbow",
        "r_lower_arm",
        "right_lower_arm",
        "elbow_pitch_r",
        "elbow_yaw_r",
    ],
    "RightHand": ["right_hand", "r_wrist", "right_wrist", "wrist_yaw_r"],
}

# Manual overrides for XML files
XML_OVERRIDES = {
    "unitree_g1": "g1_mocap_29dof.xml",
}


def find_xml_file(robot_dir):
    """Finds the main XML file for the robot."""
    robot_name = robot_dir.name
    if robot_name in XML_OVERRIDES:
        override_path = robot_dir / XML_OVERRIDES[robot_name]
        if override_path.exists():
            return override_path
        print(f"  Warning: Override {override_path} not found.")

    xml_path = robot_dir / f"{robot_name}.xml"
    if xml_path.exists():
        return xml_path

    xml_path = robot_dir / "mujoco" / f"{robot_name}.xml"
    if xml_path.exists():
        return xml_path

    xmls = list(robot_dir.glob("*.xml"))
    if len(xmls) == 1:
        return xmls[0]

    if (robot_dir / "mujoco").exists():
        xmls = list((robot_dir / "mujoco").glob("*.xml"))
        if len(xmls) == 1:
            return xmls[0]

    all_xmls = list(robot_dir.rglob("*.xml"))
    if all_xmls:
        return max(all_xmls, key=lambda p: p.stat().st_size)

    return None


def extract_links(xml_path, visited=None):
    """Extracts all body names (links) from the Mujoco XML, recursively processing includes."""
    if visited is None:
        visited = set()

    xml_path = xml_path.resolve()
    if xml_path in visited:
        return []
    visited.add(xml_path)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        links = []

        for body in root.iter("body"):
            name = body.get("name")
            if name:
                links.append(name)

        for include in root.iter("include"):
            file_attr = include.get("file")
            if file_attr:
                include_path = xml_path.parent / file_attr
                if include_path.exists():
                    links.extend(extract_links(include_path, visited))
                else:
                    print(f"  Warning: Include file not found: {include_path}")

        return links
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return []


def map_links(links):
    """Maps BVH joints to robot links using heuristics."""
    mapping = []
    used_links = set()

    for bvh_joint in BVH_JOINTS:
        keywords = MAPPING_KEYWORDS.get(bvh_joint, [])
        best_match = None

        for link in links:
            if link in used_links:
                continue
            link_lower = link.lower()
            for keyword in keywords:
                if keyword in link_lower:
                    best_match = link
                    break
            if best_match:
                break

        if best_match:
            mapping.append({"bvh": bvh_joint, "robot": best_match})
            used_links.add(best_match)

    return mapping


def get_foot_contacts(model, data):
    """
    Finds the lowest points of the robot (feet) and returns information to create sites.
    Returns: lowest_z, list of dicts {body_name, pos_world, site_name}
    """
    lowest_z = float("inf")
    geom_min_z = []

    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue

        pos = data.geom_xpos[i]
        mat = data.geom_xmat[i].reshape(3, 3)
        gtype = model.geom_type[i]
        size = model.geom_size[i]

        z_min = float("inf")
        # Estimate lowest point of the geom
        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            z_min = pos[2] - size[0]
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            p1_local = np.array([0, 0, -size[1]])
            p2_local = np.array([0, 0, size[1]])
            p1_world = pos + mat @ p1_local
            p2_world = pos + mat @ p2_local
            z_min = min(p1_world[2], p2_world[2]) - size[0]
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            corners_local = []
            for sx in [-1, 1]:
                for sy in [-1, 1]:
                    for sz in [-1, 1]:
                        corners_local.append(
                            np.array([sx * size[0], sy * size[1], sz * size[2]])
                        )
            z_coords = []
            for c in corners_local:
                p_world = pos + mat @ c
                z_coords.append(p_world[2])
            z_min = min(z_coords)
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            z_min = pos[2] - model.geom_rbound[i]
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            # Try to get exact mesh bounds if possible, else rbound
            try:
                mesh_id = model.geom_dataid[i]
                vert_adr = model.mesh_vertadr[mesh_id]
                vert_num = model.mesh_vertnum[mesh_id]
                verts = model.mesh_vert[vert_adr : vert_adr + vert_num]
                verts_world = pos + (mat @ verts.T).T
                z_min = np.min(verts_world[:, 2])
            except:
                z_min = pos[2] - model.geom_rbound[i]
        else:
            z_min = pos[2] - model.geom_rbound[i]

        if z_min < lowest_z:
            lowest_z = z_min

        geom_min_z.append((i, z_min))

    # Find geoms within tolerance of lowest_z
    tolerance = 0.005  # 5mm
    contacts = []

    # Sort by z_min to process lowest first
    geom_min_z.sort(key=lambda x: x[1])

    for i, z_min in geom_min_z:
        if z_min <= lowest_z + tolerance:
            body_id = model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

            pos = data.geom_xpos[i]
            contact_point_world = np.array([pos[0], pos[1], z_min])
            base_name = "foot_contact"
            lower_name = body_name.lower()
            if "left" in lower_name:
                base_name = "L_foot_contact"
            elif "right" in lower_name:
                base_name = "R_foot_contact"
            elif (
                lower_name.startswith("l_")
                or lower_name.endswith("_l")
                or "_l_" in lower_name
            ):
                base_name = "L_foot_contact"
            elif (
                lower_name.startswith("r_")
                or lower_name.endswith("_r")
                or "_r_" in lower_name
            ):
                base_name = "R_foot_contact"

            count = 0
            site_name = f"{base_name}_{count}"
            while any(c["site_name"] == site_name for c in contacts):
                count += 1
                site_name = f"{base_name}_{count}"

            contacts.append(
                {
                    "body_name": body_name,
                    "pos_world": contact_point_world,
                    "site_name": site_name,
                }
            )

    return lowest_z, contacts


def process_robot_xml(xml_path):
    """
    1. Loads model.
    2. Calculates T-pose root height.
    3. Finds foot contacts.
    4. Modifies XML to add keyframe and sites.
    Returns: list of foot site names
    """
    print(f"Processing XML: {xml_path}")
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        modified = False

        worldbodies = root.findall("worldbody")
        if len(worldbodies) > 1:
            print(f"  Found {len(worldbodies)} worldbody tags. Merging...")
            main_worldbody = worldbodies[0]
            for wb in worldbodies[1:]:
                for child in wb:
                    main_worldbody.append(child)
                root.remove(wb)
            modified = True

        for body in root.iter("body"):
            freejoint = body.find("freejoint")
            if freejoint is not None:
                if body.get("pos") != "0 0 0":
                    print(
                        f"  Zeroing root body '{body.get('name')}' position (was '{body.get('pos')}')"
                    )
                    body.set("pos", "0 0 0")
                    modified = True

                if freejoint.get("name") != "floating_base_joint":
                    print(
                        f"  Renaming freejoint '{freejoint.get('name')}' to 'floating_base_joint'"
                    )
                    freejoint.set("name", "floating_base_joint")
                    modified = True
                break

        output_xml_path = xml_path.parent / f"{xml_path.stem}_mocap.xml"
        ET.indent(tree, space="  ", level=0)
        tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
        print(f"  Saved intermediate XML to: {output_xml_path}")

        model = mujoco.MjModel.from_xml_path(str(output_xml_path))
        data = mujoco.MjData(model)

        nq = model.nq
        nv = model.nv
        foot_site_names = []

        if nq >= 7:
            qpos_temp = [0, 0, 0, 1, 0, 0, 0] + [0] * (nq - 7)
            data.qpos[:] = qpos_temp
            mujoco.mj_kinematics(model, data)

            lowest_z, contacts = get_foot_contacts(model, data)
            root_z = -lowest_z
            print(f"  Calculated root height: {root_z:.4f}")
            tree = ET.parse(output_xml_path)
            root = tree.getroot()

            keyframe_section = root.find("keyframe")
            if keyframe_section is None:
                keyframe_section = ET.SubElement(root, "keyframe")

            t_pose_key = None
            for key in keyframe_section.findall("key"):
                if key.get("name") == "t_pose":
                    t_pose_key = key
                    break

            if t_pose_key is None:
                t_pose_key = ET.SubElement(keyframe_section, "key")
                t_pose_key.set("name", "t_pose")

            qpos = [0, 0, root_z, 1, 0, 0, 0] + [0] * (nq - 7)
            t_pose_key.set("qpos", " ".join(map(str, qpos)))
            t_pose_key.set("qvel", " ".join(["0"] * nv))

            for body in root.iter("body"):
                sites_to_remove = []
                for site in body.findall("site"):
                    name = site.get("name", "")
                    if (
                        name.startswith("L_foot_contact")
                        or name.startswith("R_foot_contact")
                        or name.startswith("foot_contact")
                    ):
                        sites_to_remove.append(site)
                for site in sites_to_remove:
                    body.remove(site)

            def find_body_element(element, name):
                if element.tag == "body" and element.get("name") == name:
                    return element
                for child in element:
                    res = find_body_element(child, name)
                    if res is not None:
                        return res
                return None

            for contact in contacts:
                body_name = contact["body_name"]
                site_name = contact["site_name"]
                pos_world = contact["pos_world"]

                body_elem = find_body_element(root, body_name)
                if body_elem is not None:
                    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                    body_pos = data.xpos[bid]
                    body_mat = data.xmat[bid].reshape(3, 3)

                    pos_local = body_mat.T @ (pos_world - body_pos)

                    site_elem = ET.SubElement(body_elem, "body")
                    site_elem.set("name", site_name)
                    site_elem.set(
                        "pos",
                        f"{pos_local[0]:.6f} {pos_local[1]:.6f} {pos_local[2]:.6f}",
                    )
                    # site_elem.set("size", "0.01")
                    # site_elem.set("rgba", "1 0 0 1")
                    # site_elem.set("group", "1")

                    foot_site_names.append(site_name)
                else:
                    print(
                        f"  Warning: Body {body_name} not found in {xml_path.name} (might be in include). Skipping site {site_name}."
                    )

            ET.indent(tree, space="  ", level=0)
            tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
            print(f"  Saved final XML to: {output_xml_path}")

            return foot_site_names, output_xml_path

        else:
            print(f"  Warning: nq={nq} < 7, skipping keyframe/sites.")
            return foot_site_names, None

    except Exception as e:
        print(f"  Error processing XML {xml_path}: {e}")
        return [], None


def generate_config(robot_name, xml_path, mapping, foot_links):
    """Generates the YAML configuration content."""
    try:
        rel_xml_path = xml_path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel_xml_path = xml_path

    config = {
        "defaults": ["/base", "_self_"],
        "robot": robot_name,
        "wandb": {"project": f"{robot_name}_lafan"},
        "mujoco_src_xml_path": str(rel_xml_path),
        "mujoco_dst_xml_path": str(rel_xml_path).replace(".xml", "_processed.xml"),
        "foot_links": foot_links,
        "mapping": mapping,
        "mocap": {
            "offset": [0.0, 0.0, 0.0],
            "rotation": [6.123e-17, 0.0, -1.0, -1.0, 0.0, -6.123e-17, 0.0, 1.0, 0.0],
            "scale": [0.01, 0.01, 0.01],
        },
    }
    return config


def main():
    if not ASSETS_DIR.exists():
        print(f"Assets directory not found: {ASSETS_DIR}")
        return

    robots = [ASSETS_DIR / "unitree_g1"]

    for robot_dir in robots:
        robot_name = robot_dir.name
        print(f"Processing {robot_name}...")

        xml_path = find_xml_file(robot_dir)
        if not xml_path:
            print(f"  No suitable XML found for {robot_name}")
            continue

        print(f"  Found XML: {xml_path}")

        links = extract_links(xml_path)
        if not links:
            print(f"  No links found in {xml_path}")
            continue

        mapping = map_links(links)
        if not mapping:
            print(f"  Could not map any links for {robot_name}")
            continue

        foot_links, processed_xml_path = process_robot_xml(xml_path)

        if processed_xml_path:
            config_content = generate_config(
                robot_name, processed_xml_path, mapping, foot_links
            )
        else:
            config_content = generate_config(robot_name, xml_path, mapping, foot_links)

        output_dir = CONF_DIR / robot_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "lafan.yaml"

        with open(output_file, "w") as f:
            f.write("# @package _global_\n\n")
            yaml.dump(config_content, f, sort_keys=False)

        print(f"  Generated config: {output_file}")


if __name__ == "__main__":
    main()
