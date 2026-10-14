"""Build assets/so101 from TheRobotStudio SO-ARM100 repo (Simulation/SO101).

- copies so101_new_calib.urdf and the meshes it uses
- replaces jaw collision meshes with CoACD convex decompositions: a single convex hull
  of the fixed-jaw body would fill the gap between the jaws and make grasping impossible
- replaces the last FINGER_LEN of each finger with a flat box pad on its inner face:
  with only hull pieces there, a squeezed cube twisted and slipped out of the jaws
- adds Fixed_Jaw_tip / Moving_Jaw_tip links (TCP = their midpoint, as in ManiSkill's SO-100)

Usage: python scripts/prepare_so101_asset.py <path to SO-ARM100 clone>
"""
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import coacd
import numpy as np
import trimesh
from transforms3d.euler import euler2mat

# link, mesh, finger axis (tip at its min), inner-face axis, inner face at max (True) or min
JAWS = [
    ("gripper_link", "wrist_roll_follower_so101_v1.stl", 2, 0, True),
    ("moving_jaw_so101_v1_link", "moving_jaw_so101_v1.stl", 1, 0, False),
]
JAW_MESHES = [j[1] for j in JAWS]
FINGER_LEN = 0.045
PAD = 0.012  # pad thickness behind the inner face
OUT = Path(__file__).resolve().parents[1] / "assets" / "so101"


def origin_tf(el):
    o = el.find("origin")
    xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
    T = np.eye(4)
    T[:3, :3] = euler2mat(*rpy, "sxyz")
    T[:3, 3] = xyz
    return T


def collision_tf(link_el, mesh_name):
    for col in link_el.findall("collision"):
        if mesh_name in col.find("geometry/mesh").get("filename"):
            return origin_tf(col)


def decompose(src, dst, T, finger_axis, finger_end):
    """CoACD parts of the mesh, minus those in the finger region (replaced by a pad)."""
    mesh = trimesh.load(src)
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=0.04, max_convex_hull=16, seed=0)
    keep = lambda v: (np.mean(v, axis=0) @ T[:3, :3].T + T[:3, 3])[finger_axis] >= finger_end
    parts = [p for p in parts if keep(p[0])]
    # One OBJ object per convex part; SAPIEN loads each as its own convex shape.
    with open(dst, "w") as f:
        offset = 1
        for i, (v, fc) in enumerate(parts):
            f.write(f"o part_{i}\n")
            f.writelines(f"v {x} {y} {z}\n" for x, y, z in v)
            f.writelines(f"f {a + offset} {b + offset} {c + offset}\n" for a, b, c in fc)
            offset += len(v)
    return len(parts)


def add_pad(link_el, v, finger_axis, finger_end, inner_axis, inner_at_max):
    """Box over the finger region, flush with the finger's flat inner face."""
    sel = v[v[:, finger_axis] < finger_end]
    lo, hi = sel.min(axis=0), sel.max(axis=0)
    if inner_at_max:
        lo[inner_axis] = hi[inner_axis] - PAD
    else:
        hi[inner_axis] = lo[inner_axis] + PAD
    col = ET.SubElement(link_el, "collision")
    ET.SubElement(col, "origin", xyz=" ".join(f"{x:.6f}" for x in (lo + hi) / 2), rpy="0 0 0")
    ET.SubElement(ET.SubElement(col, "geometry"), "box", size=" ".join(f"{x:.6f}" for x in hi - lo))
    return hi - lo


def tip_in_link(link_el, mesh_name, axis, take_max, depth=0.008):
    """Centroid of the last `depth` metres of a jaw mesh along `axis`, in the link frame."""
    T = collision_tf(link_el, mesh_name)
    v = trimesh.load(SRC / mesh_name).vertices
    v = v @ T[:3, :3].T + T[:3, 3]
    c = v[:, axis]
    sel = v[c >= c.max() - depth] if take_max else v[c <= c.min() + depth]
    return sel.mean(axis=0)


def add_tip(root, name, parent, xyz):
    ET.SubElement(root, "link", name=name)
    j = ET.SubElement(root, "joint", name=f"{name}_joint", type="fixed")
    ET.SubElement(j, "origin", xyz=" ".join(f"{x:.6f}" for x in xyz), rpy="0 0 0")
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=name)


if __name__ == "__main__":
    SRC = Path(sys.argv[1]) / "Simulation" / "SO101" / "assets"
    tree = ET.parse(SRC.parent / "so101_new_calib.urdf")
    root = tree.getroot()
    (OUT / "meshes").mkdir(parents=True, exist_ok=True)

    for mesh in root.iter("mesh"):
        name = Path(mesh.get("filename")).name
        shutil.copy(SRC / name, OUT / "meshes" / name)
        mesh.set("filename", f"meshes/{name}")

    links = {l.get("name"): l for l in root.findall("link")}
    for link, name, finger_axis, inner_axis, inner_at_max in JAWS:
        T = collision_tf(links[link], name)
        v = trimesh.load(SRC / name).vertices @ T[:3, :3].T + T[:3, 3]
        finger_end = v[:, finger_axis].min() + FINGER_LEN
        n = decompose(SRC / name, OUT / "meshes" / f"{name}.coacd.obj", T, finger_axis, finger_end)
        size = add_pad(links[link], v, finger_axis, finger_end, inner_axis, inner_at_max)
        print(f"{name}: {n} convex parts + pad {(size * 100).round(2)} cm")
    hulls = set()
    for col in root.iter("collision"):
        mesh = col.find("geometry/mesh")
        if mesh is None:  # finger pads
            continue
        name = Path(mesh.get("filename")).name
        if name in JAW_MESHES:
            mesh.set("filename", mesh.get("filename") + ".coacd.obj")
        else:
            # Every collision file must be OBJ: the agent loads them as multi-convex files,
            # which SAPIEN does not support for STL.
            if name not in hulls:
                trimesh.load(SRC / name).convex_hull.export(OUT / "meshes" / f"{name}.convex.obj")
                hulls.add(name)
            mesh.set("filename", mesh.get("filename") + ".convex.obj")

    # Fixed jaw tip: far end of the wrist_roll_follower part along gripper_link -z.
    fixed = tip_in_link(links["gripper_link"], JAW_MESHES[0], axis=2, take_max=False)
    # Moving jaw tip: far end of the jaw along its link -y.
    moving = tip_in_link(links["moving_jaw_so101_v1_link"], JAW_MESHES[1], axis=1, take_max=False)
    add_tip(root, "Fixed_Jaw_tip", "gripper_link", fixed)
    add_tip(root, "Moving_Jaw_tip", "moving_jaw_so101_v1_link", moving)
    print("Fixed_Jaw_tip", fixed.round(4), "Moving_Jaw_tip", moving.round(4))

    tree.write(OUT / "so101.urdf", xml_declaration=True, encoding="utf-8")
    print("wrote", OUT / "so101.urdf")
