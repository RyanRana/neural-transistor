"""Read a robot's own URDF and produce a MorphologySpec.

This is what makes "any robotic structure" true rather than a slogan: the toolkit does
not need a hardcoded list of supported robots, it needs the robot's URDF, which every
ROS robot already ships.

Limbs are found by walking the joint tree from the root link and collecting each chain
of consecutive actuated joints. Canonical roles are inferred from joint axis and depth,
which is a heuristic and is reported as such -- ``role_confidence`` says how it was
decided, so a user can override the two or three it gets wrong rather than trusting all
of them blindly.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

from neuraltransistor.morph.spec import Limb, MorphologySpec

ACTUATED = {"revolute", "continuous", "prismatic"}

#: depth within a limb chain -> canonical role, for the common 3-DOF leg
DEPTH_ROLE = {0: "coxa_yaw", 1: "trochanter", 2: "knee", 3: "ankle"}


def _axis(joint) -> np.ndarray:
    a = joint.find("axis")
    if a is None or "xyz" not in a.attrib:
        return np.array([0.0, 0.0, 1.0])
    return np.array([float(x) for x in a.attrib["xyz"].split()])


def parse(path: str | Path, control_hz: float = 200.0,
          name: str | None = None) -> tuple[MorphologySpec, dict]:
    """Parse ``path`` into a MorphologySpec plus a report on what was inferred."""
    root = ET.parse(Path(path)).getroot()
    joints = []
    children = defaultdict(list)
    parent_of = {}
    for j in root.findall("joint"):
        jt = j.get("type", "fixed")
        p = j.find("parent").get("link")
        c = j.find("child").get("link")
        joints.append({"name": j.get("name"), "type": jt, "parent": p,
                       "child": c, "axis": _axis(j)})
        children[p].append(joints[-1])
        parent_of[c] = joints[-1]

    links = {l.get("name") for l in root.findall("link")}
    roots = [l for l in links if l not in parent_of]
    base = roots[0] if roots else next(iter(links))

    # Each actuated chain hanging off the base becomes a limb.
    limbs, unresolved = [], []
    for i, first in enumerate(_branch_starts(children, base)):
        chain = _walk_chain(children, first)
        act = [j for j in chain if j["type"] in ACTUATED]
        if not act:
            continue
        roles = [DEPTH_ROLE.get(d) for d in range(len(act))]
        side = _infer_side(act[0]["name"] + " " + act[0]["child"])
        limbs.append(Limb(name=_limb_name(act[0], i), joints=[j["name"] for j in act],
                          joint_roles=roles, side=side, index=i))
        if len(act) > 4:
            unresolved.append(limbs[-1].name)

    spec = MorphologySpec(
        name=name or Path(path).stem, limbs=limbs, control_hz=control_hz,
        note=f"imported from {Path(path).name}")
    report = {
        "urdf": str(path), "base_link": base,
        "n_joints_total": len(joints),
        "n_actuated": sum(1 for j in joints if j["type"] in ACTUATED),
        "n_limbs": len(limbs),
        "role_confidence": "inferred from chain depth; verify before flying",
        "limbs_over_4dof": unresolved,
        "limbs": [{"name": l.name, "joints": l.joints, "roles": l.joint_roles,
                   "side": l.side} for l in limbs],
    }
    return spec, report


def _branch_starts(children, base):
    return list(children.get(base, []))


def _walk_chain(children, first):
    chain, cur = [first], first
    while True:
        nxt = children.get(cur["child"], [])
        if len(nxt) != 1:
            break
        cur = nxt[0]
        chain.append(cur)
    return chain


def _infer_side(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("_l", "left", "lf", "lh", "l1", "l2", "l3")):
        return "L"
    if any(k in t for k in ("_r", "right", "rf", "rh", "r1", "r2", "r3")):
        return "R"
    return ""


def _limb_name(first_joint, i) -> str:
    n = first_joint["child"]
    return n if n else f"limb{i}"


def write_example(path: str | Path) -> Path:
    """Emit a minimal 4-leg 3-DOF URDF, so the importer is testable without a robot."""
    path = Path(path)
    legs = []
    for leg in ("FL", "FR", "HL", "HR"):
        legs.append(f'  <link name="{leg}_coxa"/>\n'
                    f'  <link name="{leg}_femur"/>\n'
                    f'  <link name="{leg}_tibia"/>\n'
                    f'  <joint name="{leg}_hip_yaw" type="revolute">\n'
                    f'    <parent link="base"/><child link="{leg}_coxa"/>\n'
                    f'    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>\n'
                    f'  </joint>\n'
                    f'  <joint name="{leg}_hip_pitch" type="revolute">\n'
                    f'    <parent link="{leg}_coxa"/><child link="{leg}_femur"/>\n'
                    f'    <axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/>\n'
                    f'  </joint>\n'
                    f'  <joint name="{leg}_knee" type="revolute">\n'
                    f'    <parent link="{leg}_femur"/><child link="{leg}_tibia"/>\n'
                    f'    <axis xyz="0 1 0"/><limit lower="-2" upper="0" effort="1" velocity="1"/>\n'
                    f'  </joint>\n')
    path.write_text('<?xml version="1.0"?>\n<robot name="example_quad">\n'
                    '  <link name="base"/>\n' + "".join(legs) + "</robot>\n")
    return path
