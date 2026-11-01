"""
Copyright (c) 2020 INRIA
Inspired from Antonio El Khoury PhD:
https://tel.archives-ouvertes.fr/file/index/docid/833019/filename/thesis.pdf
Section 3.8.1 Computing minimum bounding capsules

Generate capsule collision geometry for FRDV robot models.

make sure to modify link_filter with desired links before running the script
"""

from pathlib import Path

import hppfcl
import numpy as np
import scipy.optimize as optimize
import argparse
import sys

"""
Capsule definition
a, b: the two extremities of the capsule segment
r: radius of the capsule
"""

EPSILON = 1e-8
CONSTRAINT_INFLATION_RATIO = 1e-3

PRESET_LINK_FILTERS = {
    "g1": [
        "left_hip_pitch_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "right_hip_pitch_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "torso_link",
        "waist_yaw_link",
        "left_shoulder_yaw_link",
        "left_wrist_roll_link",
        "left_rubber_hand",
        "right_shoulder_yaw_link",
        "right_wrist_roll_link",
        "right_rubber_hand",
    ],
}


def capsule_volume(a, b, r):
    return np.linalg.norm(b - a) * np.pi * r**2 + 4 / 3 * np.pi * r**3


def distance_points_segment(p, a, b):
    ap = p - a
    ab = b - a
    t = ap.dot(ab) / ab.dot(ab)
    t = np.clip(t, 0, 1)
    p_witness = a[None, :] + (b - a)[None, :] * t[:, None]
    dist = np.linalg.norm(p - p_witness, axis=1).max()
    return dist


def pca_approximation(vertices):
    mean = vertices.mean(axis=0)
    centered_vertices = vertices - mean
    # Use economy SVD to avoid allocating an NxN U matrix for large meshes.
    _, _, vh = np.linalg.svd(centered_vertices, full_matrices=False)
    components = vh
    pca_proj = centered_vertices.dot(components.T)

    a0 = mean + components[0] * (pca_proj[:, 0].min() - EPSILON)
    b0 = mean + components[0] * (pca_proj[:, 0].max() + EPSILON)
    radius = np.linalg.norm(pca_proj[:, 1:], axis=1).max()
    return a0, b0, radius


def capsule_approximation(vertices):
    a0, b0, r0 = pca_approximation(vertices)
    constraint_inflation = CONSTRAINT_INFLATION_RATIO * r0
    x0 = np.array(list(a0) + list(b0) + [r0])

    def constraint_cap(x):
        return distance_points_segment(vertices, x[:3], x[3:6]) - x[6]

    def capsule_vol(x):
        return capsule_volume(x[:3], x[3:6], x[6])

    constraint = optimize.NonlinearConstraint(
        constraint_cap, lb=-np.inf, ub=-constraint_inflation
    )
    res = optimize.minimize(capsule_vol, x0, constraints=constraint)
    res_constraint = constraint_cap(res.x)
    err = (
        "The computed solution is invalid, "
        "a vertex is at a distance {:.5f} of the capsule."
    )
    assert res_constraint <= 1e-4, err.format(res_constraint)
    a, b, r = res.x[:3], res.x[3:6], res.x[6]
    return a, b, r


def approximate_mesh(filename, lMg):
    mesh_loader = hppfcl.MeshLoader()
    mesh = mesh_loader.load(filename, np.ones(3))
    vertices = np.array([lMg * mesh.vertices(i) for i in range(mesh.num_vertices)])
    assert vertices.shape == (mesh.num_vertices, 3)
    a, b, r = capsule_approximation(vertices)
    return a, b, r


def parse_mjcf(infile, outfile, link_filter, geom_group):
    from lxml import etree

    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(infile, parser)
    
    infile_path = Path(infile).absolute()
    model_dir = infile_path.parent

    compiler = tree.find("compiler")
    meshdir = ""
    if compiler is not None:
        meshdir = compiler.attrib.get("meshdir", "")
    
    mesh_map = {}
    assets = tree.find("asset")
    if assets is not None:
        for mesh in assets.findall("mesh"):
            name = mesh.attrib.get("name")
            filename = mesh.attrib.get("file")
            if name and filename:
                mesh_map[name] = filename

    def get_transform(geom):
        from pinocchio import SE3, Quaternion
        
        pos_str = geom.attrib.get("pos", "0 0 0")
        pos = np.array([float(x) for x in pos_str.split()])
        
        quat_str = geom.attrib.get("quat")
        if quat_str:
            w, x, y, z = [float(v) for v in quat_str.split()]
            rot = Quaternion(w, x, y, z).matrix()
        else:
            rot = np.eye(3)
            
        return SE3(rot, pos)

    from tqdm import tqdm
    
    geoms = tree.xpath("//geom[@type='mesh']")
    geoms_processed = []

    for geom in geoms:
        if geom.getparent().get("name", "unknown") in link_filter:
            # print(geom.getparent().get("name", "unknown"))
            geoms_processed.append(geom)

    for geom in tqdm(geoms_processed, desc="Generating capsules (MJCF)"):
        mesh_name = geom.attrib.get("mesh")
        if not mesh_name:
            continue
            
        if mesh_name in mesh_map:
            filename = mesh_map[mesh_name]
        else:
            filename = mesh_name
            
        if meshdir:
            filepath = model_dir / meshdir / filename
        else:
            filepath = model_dir / filename
            
        if not filepath.exists():
             filepath = model_dir / filename
        
        if not filepath.exists():
            print(f"Skipping {mesh_name}: File {filepath} not found")
            continue
            
        lMg = get_transform(geom)
        
        try:
            a, b, radius = approximate_mesh(str(filepath), lMg)
        except Exception as e:
            print(f"Error approximating {mesh_name}: {e}")
            continue
            
        # Create new capsule geom
        capsule = etree.Element("geom")
        capsule.attrib["type"] = "capsule"

        fromto_str = f"{a[0]:.6f} {a[1]:.6f} {a[2]:.6f} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f}"
        
        capsule.attrib["fromto"] = fromto_str
        capsule.attrib["size"] = f"{radius:.6f}"
        capsule.attrib["rgba"] = "0.0 1.0 0.0 0.3"
        capsule.attrib["group"] = str(geom_group)
        
        geom.addnext(capsule)

    tree.write(outfile, pretty_print=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Approximate robot meshes with capsules.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--robot",
        type=str,
        default="g1",
        choices=sorted(PRESET_LINK_FILTERS.keys()),
        help="Robot link-filter preset to use.",
    )
    parser.add_argument(
        "--link-filter",
        type=str,
        default=None,
        help="Optional comma-separated body names. Overrides --robot preset when set.",
    )
    parser.add_argument(
        "--group",
        type=int,
        default=3,
        help="MuJoCo geom group index for generated capsules (0-5 shown in viser).",
    )
    args = parser.parse_args()
    
    infile = Path(args.input)
    outfile = Path(args.output)

    if not infile.exists():
        print(f"Error: Input file '{infile}' not found.")
        sys.exit(1)
        
    ext = infile.suffix.lower()
    
    print(f"Processing {infile} -> {args.output}")

    if args.link_filter:
        link_filter = [x.strip() for x in args.link_filter.split(",") if x.strip()]
    else:
        link_filter = PRESET_LINK_FILTERS[args.robot]

    if len(link_filter) < 1:
        print("Error: Must specify at least one links")
        sys.exit(1)

    parse_mjcf(str(infile), str(outfile), link_filter, args.group)
