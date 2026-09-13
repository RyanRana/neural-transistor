"""Mapping six fly legs onto a robot that does not have six legs.

The fly walks with a phase pattern, not a leg count. Each leg runs the same local
circuit and the legs are held in a fixed relative phase; the gait *is* that phase
assignment. So retargeting to four legs or two is not a matter of inventing new control
-- it is assigning the same phases to a different number of slots, and choosing which
fly leg's premotor circuit drives each robot limb.

Two things transfer cleanly and one does not, and it is worth being explicit about which
is which.

**Transfers:** the per-leg circuit (swing/stance generation, load and position reflexes)
and the phase relationships between limbs.

**Does not transfer:** absolute timing and joint limits. A robot leg has different mass,
length and torque than a fly leg, so stepping frequency and amplitude must be set from
the robot's own dynamics. The toolkit exposes those as explicit scalars rather than
pretending the fly's values carry over.

Gait phases below are the standard ones from the legged-locomotion literature, expressed
as a fraction of the step cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from neuraltransistor.morph.spec import MorphologySpec, MotorDecode

#: gait name -> {limb name: phase in [0,1)}
GAITS = {
    # six legs
    "tripod":   {"L1": 0.0, "R2": 0.0, "L3": 0.0, "R1": 0.5, "L2": 0.5, "R3": 0.5},
    "tetrapod": {"L1": 0.0, "R2": 0.0, "R1": 1/3, "L3": 1/3, "L2": 2/3, "R3": 2/3},
    "wave":     {"R3": 0.0, "R2": 1/6, "R1": 2/6, "L3": 3/6, "L2": 4/6, "L1": 5/6},
    # four legs
    "trot":     {"FL": 0.0, "HR": 0.0, "FR": 0.5, "HL": 0.5},
    "walk4":    {"FL": 0.0, "HR": 0.25, "FR": 0.5, "HL": 0.75},
    "pace":     {"FL": 0.0, "HL": 0.0, "FR": 0.5, "HR": 0.5},
    "bound":    {"FL": 0.0, "FR": 0.0, "HL": 0.5, "HR": 0.5},
    # two legs
    "alternate": {"L": 0.0, "R": 0.5},
}

DEFAULT_GAIT = {"hexapod": "tripod", "quadruped": "trot", "biped": "alternate"}

#: Which fly leg circuit drives each limb of a non-hexapod robot.
#: Hind legs are the primary propulsors in insects, so they are the default source for
#: reduced-leg-count robots; front legs are the most sensory-dominated and transfer worst.
SOURCE_PREFERENCE = {
    "hexapod":   {"L1": ("front", "L"), "R1": ("front", "R"),
                  "L2": ("middle", "L"), "R2": ("middle", "R"),
                  "L3": ("hind", "L"),  "R3": ("hind", "R")},
    "quadruped": {"FL": ("front", "L"), "FR": ("front", "R"),
                  "HL": ("hind", "L"),  "HR": ("hind", "R")},
    "biped":     {"L": ("hind", "L"),   "R": ("hind", "R")},
}


def infer_sources(morph) -> dict:
    """Work out which fly leg drives each limb, for a morphology we did not author.

    A URDF gives you limb names its author chose -- ``FL_coxa``, ``leg_2``,
    ``front_right_hip`` -- not the tidy keys in SOURCE_PREFERENCE. Without this, importing
    a real robot produces zero driven joints, which is the "any robot" promise failing
    silently at the last step.

    Three attempts, in order:
      1. the builtin table, if the morphology came from BUILTIN
      2. name matching: front/mid/hind and left/right tokens in the limb name
      3. positional fallback: order the limbs, alternate sides, and spread them over the
         fly's three segments. Crude, and reported as such, but it means a robot always
         gets *a* mapping rather than nothing.
    """
    table = SOURCE_PREFERENCE.get(morph.name)
    if table and all(l.name in table for l in morph.limbs):
        return dict(table)

    FRONT = ("front", "fore", "_f", "f_", "fl", "fr", "1")
    HIND = ("hind", "rear", "back", "_h", "h_", "hl", "hr", "3")
    MID = ("mid", "middle", "_m", "m_", "ml", "mr", "2")
    LEFT = ("left", "_l", "l_", "fl", "hl", "ml")
    RIGHT = ("right", "_r", "r_", "fr", "hr", "mr")

    def tok(name, keys):
        n = name.lower()
        return any(k in n for k in keys)

    out, named = {}, True
    for limb in morph.limbs:
        n = limb.name.lower()
        seg = "front" if tok(n, FRONT) else ("hind" if tok(n, HIND) else
                                             ("middle" if tok(n, MID) else None))
        side = limb.side or ("L" if tok(n, LEFT) else ("R" if tok(n, RIGHT) else None))
        if seg is None or side is None:
            named = False
            break
        out[limb.name] = (seg, side)
    if named and out:
        return out

    # positional fallback
    segs = ["front", "middle", "hind"]
    out = {}
    for i, limb in enumerate(morph.limbs):
        side = limb.side or ("L" if i % 2 == 0 else "R")
        n_pairs = max(len(morph.limbs) // 2, 1)
        seg = segs[min(int(i // 2 * 3 / n_pairs), 2)] if n_pairs > 1 else "hind"
        out[limb.name] = (seg, side)
    return out


def infer_phases(morph, gait: Optional[str]) -> tuple:
    """Gait phases for a morphology whose limb names we did not choose."""
    if gait and gait in GAITS:
        table = GAITS[gait]
        if all(l.name in table for l in morph.limbs):
            return table, gait
    n = len(morph.limbs)
    if n == 6:
        base, name = [0.0, .5, .5, 0.0, 0.0, .5], gait or "tripod"
    elif n == 4:
        base, name = [0.0, .5, .5, 0.0], gait or "trot"
    elif n == 2:
        base, name = [0.0, .5], gait or "alternate"
    else:
        base, name = [i / max(n, 1) for i in range(n)], gait or f"wave{n}"
    return {l.name: base[i % len(base)] for i, l in enumerate(morph.limbs)}, name


@dataclass
class LimbBinding:
    limb: str
    phase: float
    source_leg: str
    source_side: str
    joints: dict = field(default_factory=dict)   # canonical role -> MotorDecode

    def commands(self, rates: np.ndarray) -> dict:
        return {role: d.command(rates) for role, d in self.joints.items()}


@dataclass
class Retarget:
    morph: MorphologySpec
    gait: str
    bindings: list
    unused_sources: list = field(default_factory=list)
    note: str = ""

    def commands(self, rates: np.ndarray) -> dict:
        return {b.limb: b.commands(rates) for b in self.bindings}

    def summary(self) -> str:
        lines = [f"{self.morph.name} / {self.gait}: {len(self.bindings)} limbs bound"]
        for b in self.bindings:
            lines.append(f"  {b.limb:4s} phase {b.phase:.3f} <- fly {b.source_leg} "
                         f"{b.source_side} [{', '.join(sorted(b.joints))}]")
        if self.unused_sources:
            lines.append(f"  unused fly sources: {', '.join(self.unused_sources)}")
        return "\n".join(lines)


def retarget(morph: MorphologySpec, decodes: list[MotorDecode],
             gait: Optional[str] = None) -> Retarget:
    """Bind each robot limb to a fly leg circuit and a gait phase."""
    gait = gait or DEFAULT_GAIT.get(morph.name, "")
    if morph.name.startswith("modular"):
        n = len(morph.limbs)
        phases = {l.name: i / n for i, l in enumerate(morph.limbs)}
        pref = {l.name: (["front", "middle", "hind"][i % 3],
                         "L" if i % 2 == 0 else "R")
                for i, l in enumerate(morph.limbs)}
        gait = gait or f"ring{n}"
    else:
        if gait and gait not in GAITS and morph.name in SOURCE_PREFERENCE:
            raise KeyError(f"unknown gait {gait!r}; have {sorted(GAITS)}")
        phases, gait = infer_phases(morph, gait)
        pref = infer_sources(morph)

    by_src: dict = {}
    for d in decodes:
        by_src.setdefault((d.leg, d.side), {})[d.joint] = d

    bindings, used = [], set()
    for limb in morph.limbs:
        src = pref.get(limb.name)
        if src is None:
            continue
        joints = {r: by_src.get(src, {}).get(r) for r in limb.joint_roles}
        joints = {k: v for k, v in joints.items() if v is not None}
        used.add(src)
        bindings.append(LimbBinding(limb=limb.name, phase=phases.get(limb.name, 0.0),
                                    source_leg=src[0], source_side=src[1],
                                    joints=joints))
    unused = [f"{a} {b}" for (a, b) in sorted(by_src) if (a, b) not in used]
    note = ""
    if morph.name == "quadruped":
        note = ("Middle-leg circuits are not bound. Their intersegmental coordination "
                "neurons still run and still shape front/hind phase; only their motor "
                "output is unused.")
    if morph.name == "biped":
        note = ("Only the hind-leg pair is bound. Four of six fly leg circuits go "
                "unused at the motor boundary.")
    return Retarget(morph=morph, gait=gait, bindings=bindings,
                    unused_sources=unused, note=note)


def phase_oscillator(n: int, phases: np.ndarray, freq_hz: float, dt: float,
                     coupling: float = 2.0):
    """A Kuramoto ring that holds the gait phases, for when the circuit needs a clock.

    The fly's inter-leg coordination is distributed and reflex-driven rather than a
    single oscillator, so this is NOT a model of the fly. It is a fallback for robots
    whose mechanics cannot tolerate the transient while a distributed pattern settles.
    Use it, and say you used it.
    """
    theta = np.asarray(phases, dtype=np.float64) * 2 * np.pi
    target = theta.copy()
    omega = 2 * np.pi * freq_hz

    def step():
        nonlocal theta
        d = target[None, :] - target[:, None]
        err = np.sin(theta[None, :] - theta[:, None] - d).sum(axis=1)
        theta = (theta + dt * (omega + coupling / n * err)) % (2 * np.pi)
        return theta / (2 * np.pi)
    return step
