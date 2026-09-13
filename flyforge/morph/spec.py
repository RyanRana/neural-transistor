"""Morphology: turning fly motor neurons into joint commands on a real robot.

The male-CNS release annotates every motor neuron with the **muscle it innervates**, in
plain anatomical language -- "Ti extensor MN", "Tergopleural/Pleural promotor MN",
"Ta levator MN". That is the piece that makes retargeting principled instead of
hand-waved: a muscle name states a joint and a direction, and both transfer to any
articulated robot regardless of how many legs it has.

The fly leg has five functional joints. Almost every legged robot is a subset:

    coxa_yaw     ThC   protraction / retraction   (promotor  vs remotor)
    coxa_roll    ThC   abduction / adduction      (abductor  vs adductor)
    trochanter   CTr   levation / depression      (Tr extensor vs Tr flexor)
    knee         FTi   extension / flexion        (Ti extensor vs Ti flexor)
    ankle        TiTa  levation / depression      (Ta levator vs Ta depressor)

A 3-DOF robot leg (the common case: hip yaw, hip pitch, knee) takes coxa_yaw,
trochanter and knee. A 2-DOF leg takes trochanter and knee. Nothing is invented: joints
the robot does not have are simply not driven, and the toolkit reports which fly
muscles went unused rather than silently dropping them.

Antagonist pairs matter. Muscles pull; joints are driven by opposing pairs. So a joint
command is ``rate(agonist) - rate(antagonist)``, which is naturally zero-centred and
does the right thing when both are active (co-contraction -> stiffness, not motion).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

#: joint name -> (positive-direction pattern, negative-direction pattern)
#: Patterns match against the motor neuron's ``type`` string.
MUSCLE_MAP = {
    "coxa_yaw": (r"promotor|anterior rotator",
                 r"remotor|posterior rotator"),
    "coxa_roll": (r"abductor",
                  r"adductor"),
    "trochanter": (r"Tr extensor|Tergotr|Sternotrochanter",
                   r"Tr flexor|Fe reductor"),
    "knee": (r"Ti extensor",
             r"Ti flexor|Acc\. ti flexor"),
    "ankle": (r"Ta levator",
              r"Ta depressor|ltm"),
}

#: VNC subclass code -> which leg the motor neuron belongs to
LEG_SUBCLASS = {"fl": "front", "ml": "middle", "hl": "hind"}

#: Wing muscles split into power (amplitude/frequency) and steering (differential).
WING_MAP = {
    "power": r"^DLMn|^DVMn",
    "steering": r"^(?:b1|b2|b3|i1|i2|iii1|iii3|hg[1-4]|tp[12]|ps1)",
}

CANONICAL_JOINTS = list(MUSCLE_MAP)


@dataclass
class Limb:
    """One limb of the target robot."""
    name: str
    joints: list            # names, in the robot's own vocabulary
    joint_roles: list       # one canonical role per joint, or None to leave undriven
    side: str = ""          # "L" | "R" | ""
    index: int = 0

    def role_index(self, role: str) -> Optional[int]:
        return self.joint_roles.index(role) if role in self.joint_roles else None


@dataclass
class MorphologySpec:
    """What the robot is, in the only terms the compiler needs."""
    name: str
    limbs: list = field(default_factory=list)
    wings: int = 0
    control_hz: float = 200.0
    note: str = ""

    @property
    def n_limbs(self) -> int:
        return len(self.limbs)

    @property
    def n_joints(self) -> int:
        return sum(len(l.joints) for l in self.limbs)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_limbs"] = self.n_limbs
        d["n_joints"] = self.n_joints
        return d

    def __str__(self):
        return (f"{self.name}: {self.n_limbs} limbs / {self.n_joints} joints"
                + (f" / {self.wings} wings" if self.wings else "")
                + f" @ {self.control_hz:g} Hz")


def _leg(name, side, idx, dof=3):
    roles = {2: ["trochanter", "knee"],
             3: ["coxa_yaw", "trochanter", "knee"],
             4: ["coxa_yaw", "trochanter", "knee", "ankle"],
             5: CANONICAL_JOINTS}[dof]
    return Limb(name=name, joints=[f"{name}_{r}" for r in roles],
                joint_roles=roles, side=side, index=idx)


def hexapod(dof=3, hz=200.0) -> MorphologySpec:
    names = [("L1", "L"), ("R1", "R"), ("L2", "L"), ("R2", "R"), ("L3", "L"), ("R3", "R")]
    return MorphologySpec("hexapod", [_leg(n, s, i, dof) for i, (n, s) in enumerate(names)],
                          control_hz=hz,
                          note="Six legs. The fly's native layout: no retargeting needed.")


def quadruped(dof=3, hz=200.0) -> MorphologySpec:
    names = [("FL", "L"), ("FR", "R"), ("HL", "L"), ("HR", "R")]
    return MorphologySpec("quadruped", [_leg(n, s, i, dof) for i, (n, s) in enumerate(names)],
                          control_hz=hz,
                          note="Four legs. Middle-leg circuits are merged into the "
                               "front/hind pairs by the retargeter.")


def biped(dof=3, hz=200.0) -> MorphologySpec:
    names = [("L", "L"), ("R", "R")]
    return MorphologySpec("biped", [_leg(n, s, i, dof) for i, (n, s) in enumerate(names)],
                          control_hz=hz,
                          note="Two legs. Only the front-leg circuit pair is used; "
                               "the rest of the coordination graph is dropped.")


def winged(n_wings=2, hz=250.0) -> MorphologySpec:
    return MorphologySpec("winged", [], wings=n_wings, control_hz=hz,
                          note="Flapping flight. Power muscles set amplitude, steering "
                               "muscles set the left/right asymmetry. The control rate "
                               "is set by BODY dynamics, not wingbeat frequency: a "
                               "flapping micro-UAV's unstable body mode needs roughly "
                               "150-300 Hz with <15 ms latency, and the ~200 Hz wingbeat "
                               "is a carrier the controller modulates rather than a rate "
                               "it must resolve. RoboBee's own papers make this point.")


def modular(n=4, dof=2, hz=200.0) -> MorphologySpec:
    """N identical actuated modules -- the '4-piece' case, generalised."""
    return MorphologySpec(f"modular{n}",
                          [_leg(f"M{i+1}", "L" if i % 2 == 0 else "R", i, dof)
                           for i in range(n)],
                          control_hz=hz,
                          note=f"{n} identical modules. Treated as {n} limbs in a "
                               f"symmetric ring for phase assignment.")


BUILTIN = {
    "hexapod": hexapod, "quadruped": quadruped, "biped": biped,
    "winged": winged, "modular": modular,
}


# --------------------------------------------------------------------------- #

@dataclass
class MotorDecode:
    """Which circuit neurons drive which canonical joint, and with what polarity."""
    leg: str                      # "front" | "middle" | "hind" | "wing"
    side: str                     # "L" | "R" | ""
    joint: str                    # canonical joint role
    agonist: np.ndarray           # local neuron indices pushing positive
    antagonist: np.ndarray        # local neuron indices pushing negative

    def command(self, rates: np.ndarray) -> float:
        a = float(rates[self.agonist].mean()) if len(self.agonist) else 0.0
        b = float(rates[self.antagonist].mean()) if len(self.antagonist) else 0.0
        return a - b


def decode_motor(ir, neurons_df) -> list[MotorDecode]:
    """Group a circuit's motor neurons into antagonist pairs per joint.

    ``neurons_df`` is the connectome annotation frame; it is joined by bodyId so the
    muscle names and subclass codes are available even though the IR only keeps types.
    """
    idx = {int(b): i for i, b in enumerate(ir.body_ids)}
    sub = neurons_df[neurons_df.bodyId.isin(idx)]
    sub = sub[sub.superclass.isin(["vnc_motor", "cb_motor"])]
    out: list[MotorDecode] = []
    if len(sub) == 0:
        return out

    for legcode, legname in LEG_SUBCLASS.items():
        for side in ("L", "R"):
            grp = sub[(sub.subclass == legcode) & (sub.somaSide == side)]
            if len(grp) == 0:
                continue
            for joint, (pos_pat, neg_pat) in MUSCLE_MAP.items():
                ag = grp[grp["type"].str.contains(pos_pat, case=False, regex=True, na=False)]
                an = grp[grp["type"].str.contains(neg_pat, case=False, regex=True, na=False)]
                if len(ag) == 0 and len(an) == 0:
                    continue
                out.append(MotorDecode(
                    leg=legname, side=side, joint=joint,
                    agonist=np.array([idx[int(b)] for b in ag.bodyId], dtype=np.int32),
                    antagonist=np.array([idx[int(b)] for b in an.bodyId], dtype=np.int32)))

    for kind, pat in WING_MAP.items():
        for side in ("L", "R"):
            grp = sub[(sub.subclass == "wm") & (sub.somaSide == side)]
            g = grp[grp["type"].str.contains(pat, case=False, regex=True, na=False)]
            if len(g):
                out.append(MotorDecode(
                    leg="wing", side=side, joint=kind,
                    agonist=np.array([idx[int(b)] for b in g.bodyId], dtype=np.int32),
                    antagonist=np.zeros(0, dtype=np.int32)))
    return out


def coverage(decodes: list[MotorDecode], morph: MorphologySpec) -> dict:
    """What fraction of the robot's joints the fly actually supplies a command for."""
    have = {d.joint for d in decodes}
    need = {r for l in morph.limbs for r in l.joint_roles}
    return {"joints_driven": sorted(have & need),
            "joints_undriven": sorted(need - have),
            "fly_joints_unused": sorted(have - need),
            "coverage": len(have & need) / max(len(need), 1)}
