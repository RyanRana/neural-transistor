"""Read a robot's own URDF and produce a MorphologySpec.

This is what makes "any robotic structure" true rather than a slogan: the toolkit does
not need a hardcoded list of supported robots, it needs the robot's URDF, which every
ROS robot already ships.

Limbs are found by walking the joint tree from the root link and collecting each chain
of consecutive actuated joints. Canonical roles are inferred from joint axis and depth,
which is a heuristic and is reported as such -- ``role_confidence`` says how it was
decided, so a user can override the two or three it gets wrong rather than trusting all
of them blindly.

The walk looks *through* fixed joints. Real URDFs are full of them and they are never
limb boundaries: ``base_footprint -> base_link``, an IMU mount, the shell around a hip,
a foot pad. A walk that stops at the first one imports most real robots as zero limbs,
which is the "any robot" promise failing silently -- so ``n_actuated_in_limbs`` is
reported next to ``n_actuated`` and the two are expected to be equal.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from neuraltransistor.morph.spec import Limb, MorphologySpec, name_tokens

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
    work, seen_joints = deque(_next_actuated(children, base)), set()
    while work:
        first = work.popleft()
        if first["name"] in seen_joints:
            continue
        act, forks = _walk_chain(children, first, seen_joints)
        work.extend(forks)
        i = len(limbs)
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
        "n_actuated_in_limbs": sum(len(l.joints) for l in limbs),
        "n_limbs": len(limbs),
        "role_confidence": "inferred from chain depth; verify before flying",
        "limbs_over_4dof": unresolved,
        "limbs": [{"name": l.name, "joints": l.joints, "roles": l.joint_roles,
                   "side": l.side} for l in limbs],
    }
    return spec, report


def _next_actuated(children, link) -> list:
    """The next actuated joints below ``link``, looking through fixed joints.

    Breadth-first so document order survives, which is what makes ``L1, R1, L2, ...``
    come back in the order the URDF author wrote them rather than reversed.
    """
    out, queue, seen = [], deque([link]), set()
    while queue:
        cur = queue.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        for j in children.get(cur, []):
            if j["type"] in ACTUATED:
                out.append(j)
            else:
                queue.append(j["child"])
    return out


def _walk_chain(children, first, seen_joints) -> tuple:
    """Follow one limb down from ``first``; return (chain, forks).

    The chain ends where the structure genuinely forks into two actuated subtrees -- a
    torso into arms, a wrist into gripper fingers. That is a new limb rather than more
    of this one, so the branches come back as ``forks`` and are walked in their turn.
    Nothing actuated is dropped on the floor.
    """
    chain, cur = [first], first
    seen_joints.add(first["name"])
    while True:
        nxt = [j for j in _next_actuated(children, cur["child"])
               if j["name"] not in seen_joints]
        if len(nxt) != 1:
            return chain, nxt
        cur = nxt[0]
        seen_joints.add(cur["name"])
        chain.append(cur)


#: whole tokens that name a side. Two-letter codes are segment+side in either order,
#: so ``rf`` (right front) and ``fr`` (front right) both mean right.
_LEFT = {"l", "left", "lf", "lm", "lr", "lh", "l1", "l2", "l3", "fl", "ml", "hl", "rl"}
_RIGHT = {"r", "right", "rf", "rm", "rr", "rh", "r1", "r2", "r3", "fr", "mr", "hr"}


def _infer_side(text: str) -> str:
    """Left, right, or neither -- from whole tokens, never substrings.

    ``left``/``right`` also match as a prefix or suffix, because ``panda_leftfinger``
    is one token and still says which side it is on.
    """
    for w in name_tokens(text):
        if w in _LEFT or w.startswith("left") or w.endswith("left"):
            return "L"
        if w in _RIGHT or w.startswith("right") or w.endswith("right"):
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
