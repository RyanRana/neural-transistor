"""Eight real robots, one connectome circuit, side by side.

Every robot here is somebody else's URDF, fetched from the repository that ships it --
no hand-written models, because the point of the URDF importer is that it reads what a
robot already has rather than what we would have written for it. The eight cover the
shapes that actually turn up: a quadrotor, two quadrupeds with different leg topology, a
hexapod, a biped, an arm, a skid-steer rover and an Ackermann car.

What is real here, and what is not, because the difference is the whole point:

**Real.** The connectome, the circuit, the pruning and int8 quantization, the URDF
import, the muscle-to-joint binding, the gait phase assignment, the emitted ROS 2
package, and the per-joint command ``rate(agonist) - rate(antagonist)`` computed by
ticking the int8 reference runtime. The sign and relative magnitude of every joint
command in the viewer came out of the fly.

**Not real.** The rhythm. The unfitted circuit settles to a steady leg-differentiated
posture and does not oscillate -- the same failure ``compass`` has, and for the same
reason: a rhythm needs a self-sustaining loop and the dynamics that would close it are
not in the wiring (docs/DYNAMICS.md). So the step cycle comes from
``retarget.phase_oscillator``, the Kuramoto ring the library provides for exactly this,
whose docstring says: use it, and say you used it. This is that saying.

**Also not real.** Contact, gravity, mass, torque. This is a kinematic playback of joint
commands -- what ``robot_state_publisher`` shows you in RViz, not what Gazebo shows you.
Nothing here has been simulated dynamically and nothing has touched hardware.

Usage::

    .venv/bin/python demos/formfactors.py            # everything
    .venv/bin/python demos/formfactors.py --no-gifs  # data only, much faster
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

import neuraltransistor as nt
from neuraltransistor.ir.runtime import Reference
from neuraltransistor.morph import retarget as R, spec as M

ROOT = Path(__file__).resolve().parent.parent
URDF_DIR = ROOT / "artifacts" / "urdf"
OUT = ROOT / "artifacts" / "formfactors"

#: Circuit, and how hard it is pruned. `legs` is T1-T3 intrinsic + motor: the six legs
#: and the intersegmental interneurons that couple them.
CIRCUIT, P_REAL, BITS = "legs", 0.95, 8

#: Tonic drive, in Q8.8, applied to every non-motor neuron. The `legs` circuit has no
#: input port -- the descending neurons that would start walking live in `commands`, a
#: different circuit -- so something has to say "go", and this is that something. It is
#: exogenous and it is not from the fly.
DRIVE_Q88 = 256
WARMUP_TICKS, RATE_SMOOTHING = 1200, 0.98

FPS, SECONDS, STEP_HZ = 30, 3.0, 1.0        # 3 s at 30 fps = 3 whole step cycles

ROBOTS = [
    dict(key="cf2x", title="Crazyflie 2.X", kind="quadrotor", gait=None,
         repo="utiasDSL/gym-pybullet-drones",
         url="https://raw.githubusercontent.com/utiasDSL/gym-pybullet-drones/main/"
             "gym_pybullet_drones/assets/cf2x.urdf",
         note="Every joint is fixed, so the leg retargeter binds nothing. A quadrotor "
              "has no joints -- it has four rotors. Driven from the fly's WING motor "
              "neurons instead: power muscles set collective thrust, the left/right "
              "steering-muscle difference sets roll."),
    dict(key="a1", title="Unitree A1", kind="quadruped", gait="trot",
         repo="bulletphysics/bullet3",
         url="https://raw.githubusercontent.com/bulletphysics/bullet3/master/"
             "examples/pybullet/gym/pybullet_data/a1/a1.urdf",
         note="Three-DOF legs, so all three of the fly's transferable joints bind."),
    dict(key="phantomx", title="PhantomX Mark II", kind="hexapod", gait="tripod",
         repo="HumaRobotics/phantomx_description",
         url="https://raw.githubusercontent.com/HumaRobotics/phantomx_description/"
             "master/urdf/phantomx.urdf",
         note="The fly's own layout. Six legs, three segments a side, and the binding "
              "is one-for-one with no leg left over -- the only robot here where the "
              "retargeter has nothing to throw away."),
    dict(key="cassie", title="Agility Cassie", kind="biped", gait="alternate",
         repo="UMich-BipedLab/cassie_description",
         url="https://raw.githubusercontent.com/UMich-BipedLab/cassie_description/"
             "master/urdf/cassie.urdf",
         note="Seven actuated joints a leg, four of which have a fly counterpart; the "
              "other three, and four of the fly's six legs, go unused. Two caveats "
              "show in the picture. Cassie's leg is a closed four-bar that URDF can "
              "only write as an open tree, so the loop is not closed here and the "
              "shank hangs where the knee puts it. And depth-based role inference "
              "binds the fly's coxa_yaw to Cassie's hip ABDUCTION rather than its "
              "flexion, which is the heuristic the importer flags as needing a human."),
    dict(key="minitaur", title="Ghost Minitaur", kind="direct-drive quadruped",
         gait="bound", repo="bulletphysics/bullet3",
         url="https://raw.githubusercontent.com/bulletphysics/bullet3/master/"
             "examples/pybullet/gym/pybullet_data/quadruped/minitaur.urdf",
         note="Five-bar parallel legs: two motors per leg, so the importer reads eight "
              "limbs rather than four, and both motors of a leg bind to the same fly "
              "leg. Honest, and not what a leg-count-based retargeter would assume."),
    dict(key="panda", title="Franka Emika Panda", kind="7-DOF arm", gait=None,
         repo="bulletphysics/bullet3",
         url="https://raw.githubusercontent.com/bulletphysics/bullet3/master/"
             "examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf",
         note="Not a leg, and the mapping says so: chain depth binds the first four "
              "joints and the wrist keeps none. The gripper forks off as two more "
              "limbs. Included because the failure mode should be visible."),
    dict(key="husky", title="Clearpath Husky", kind="skid-steer rover", gait="walk4",
         repo="bulletphysics/bullet3",
         url="https://raw.githubusercontent.com/bulletphysics/bullet3/master/"
             "examples/pybullet/gym/pybullet_data/husky/husky.urdf",
         note="Four continuous wheels, one DOF each, so only coxa_yaw binds and the "
              "command drives wheel speed rather than a joint angle."),
    dict(key="racecar", title="MIT RACECAR", kind="Ackermann car", gait="trot",
         repo="bulletphysics/bullet3",
         url="https://raw.githubusercontent.com/bulletphysics/bullet3/master/"
             "examples/pybullet/gym/pybullet_data/racecar/racecar.urdf",
         note="The steering hinges carry the word 'front' only in their child joint, "
              "which is why segment matching reads joint names and not just the limb."),
]


# --------------------------------------------------------------------------- #
# Recording the API calls
# --------------------------------------------------------------------------- #

class Recorder:
    """Runs a line of API and keeps it, so the transcript cannot drift from the run.

    The string is evaluated, not described: what the viewer prints is the source that
    produced the numbers next to it. A demo that hand-writes its own code listing is one
    edit away from lying about what it ran.
    """

    def __init__(self, env: dict):
        self.env, self.log = env, []

    def __call__(self, code: str, note: str = "") -> object:
        t0 = time.perf_counter()
        value = eval(code, self.env, self.env)          # noqa: S307 - our own source
        ms = (time.perf_counter() - t0) * 1e3
        self.log.append({"code": code, "result": _describe(value),
                         "ms": round(ms, 1), "note": note})
        return value

    def bind(self, name: str, code: str, note: str = "") -> object:
        self.env[name] = value = self(code, note)
        self.log[-1]["code"] = f"{name} = {code}"
        return value


def _describe(v) -> str:
    """A one-line, honest summary of whatever an API call returned.

    Default reprs are no good in a transcript that is meant to be read: a Connectome
    prints its memory address, and a list of MotorDecode prints the first numpy array
    until it runs out of room. Say what came back instead.
    """
    from neuraltransistor.data.source import Connectome
    from neuraltransistor.ir.graph import CircuitIR
    if isinstance(v, Connectome):
        return (f"Connectome {v.meta.dataset}: {len(v):,} annotated bodies, "
                f"{v.meta.n_edges_retained:,} edges indexed")
    if isinstance(v, list) and v and isinstance(v[0], M.MotorDecode):
        legs = sorted({d.leg for d in v})
        joints = sorted({d.joint for d in v})
        return (f"{len(v)} MotorDecode over {', '.join(legs)} "
                f"({len(joints)} joints: {', '.join(joints)})")
    if isinstance(v, CircuitIR):
        return (f"CircuitIR {v.name!r}: {v.n_neurons:,} neurons, {v.n_edges:,} edges, "
                f"{v.n_synapses:,} synapses")
    if isinstance(v, M.MorphologySpec):
        return str(v)
    if isinstance(v, R.Retarget):
        return (f"Retarget: {len(v.bindings)} limbs bound, gait {v.gait!r}, "
                f"{len(v.unused_sources)} fly sources unused")
    if isinstance(v, tuple):
        return " | ".join(_describe(x) for x in v)
    if isinstance(v, dict):
        keep = {k: v[k] for k in list(v)[:6] if not isinstance(v[k], (list, dict))}
        return ", ".join(f"{k}={_fmt(x)}" for k, x in keep.items()) or f"dict({len(v)})"
    if hasattr(v, "drive_rel_err_mean"):
        return (f"drive_err={v.drive_rel_err_mean:.5f}, sign_flips={v.drive_sign_flips}, "
                f"edge_retention={v.edge_retention:.3f}")
    if isinstance(v, np.ndarray):
        return f"ndarray{v.shape} {v.dtype}"
    return _fmt(v)


def _fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.4g}"
    if isinstance(x, int):
        return f"{x:,}"
    return str(x)[:80]


# --------------------------------------------------------------------------- #
# URDF -> drawable scene, and forward kinematics
# --------------------------------------------------------------------------- #

def _nums(s, n=3):
    """Numbers out of a URDF attribute, tolerating unexpanded ROS substitutions.

    Husky's published URDF still contains
    ``rpy="$(optenv HUSKY_IMU_RPY 0 -1.5708 3.1416)"``. With the variable unset, optenv
    means "use the default", which is the numbers that follow the name -- so keeping the
    numeric tokens and dropping the rest resolves it the way roslaunch would, instead of
    crashing on somebody else's file.
    """
    if not s:
        return [0.0] * n
    v = []
    for tok in s.replace(",", " ").replace("$(", " ").replace(")", " ").split():
        try:
            v.append(float(tok))
        except ValueError:
            continue
    return (v + [0.0] * n)[:n]


def _origin(el):
    o = el.find("origin") if el is not None else None
    if o is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return _nums(o.get("xyz")), _nums(o.get("rpy"))


def _rpy(r, p, y) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def _T(xyz, Rm=None) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.eye(3) if Rm is None else Rm
    T[:3, 3] = xyz
    return T


def _axis_rot(axis, q) -> np.ndarray:
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * (K @ K)


def _wire_box(size, T):
    sx, sy, sz = [s / 2 for s in size]
    c = [(x, y, z) for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]
    c = [tuple((T @ np.array([*p, 1.0]))[:3]) for p in c]
    e = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
         (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    return [(c[i], c[j]) for i, j in e]


def _wire_cyl(radius, length, T, n=10):
    segs, rings = [], []
    for sign in (-1, 1):
        ring = [tuple((T @ np.array([radius * math.cos(2 * math.pi * k / n),
                                     radius * math.sin(2 * math.pi * k / n),
                                     sign * length / 2, 1.0]))[:3]) for k in range(n)]
        rings.append(ring)
        segs += [(ring[k], ring[(k + 1) % n]) for k in range(n)]
    segs += [(rings[0][k], rings[1][k]) for k in range(0, n, max(n // 4, 1))]
    return segs


def _wire_sphere(radius, T, n=10):
    segs = []
    for plane in range(2):
        ring = []
        for k in range(n):
            a = 2 * math.pi * k / n
            p = ([radius * math.cos(a), radius * math.sin(a), 0] if plane == 0
                 else [radius * math.cos(a), 0, radius * math.sin(a)])
            ring.append(tuple((T @ np.array([*p, 1.0]))[:3]))
        segs += [(ring[k], ring[(k + 1) % n]) for k in range(n)]
    return segs


def _prop_radius(path: Path) -> float:
    """The Crazyflie URDF carries its own rotor radius in a <properties> tag."""
    el = ET.parse(path).getroot().find("properties")
    try:
        return float(el.get("prop_radius"))
    except (AttributeError, TypeError, ValueError):
        return 0.023


def load_scene(path: Path) -> dict:
    """Link tree, joint frames and drawable wireframe, straight from the URDF.

    Meshes are not fetched, so a link's shape comes from whatever primitives the URDF
    also declares -- visual first, collision as a fallback. Links that only have a mesh
    contribute their frame, and the bone drawn to the next joint, which is what RViz
    shows when you turn the meshes off.
    """
    root = ET.parse(path).getroot()
    links, joints, children, parent_of = {}, [], defaultdict(list), {}

    for l in root.findall("link"):
        prims = []
        for tag in ("visual", "collision"):
            for v in l.findall(tag):
                g = v.find("geometry")
                if g is None:
                    continue
                xyz, rpy = _origin(v)
                T = _T(xyz, _rpy(*rpy))
                for c in g:
                    if c.tag == "box":
                        prims += _wire_box(_nums(c.get("size")), T)
                    elif c.tag == "cylinder":
                        prims += _wire_cyl(float(c.get("radius", 0)),
                                           float(c.get("length", 0)), T)
                    elif c.tag == "sphere":
                        prims += _wire_sphere(float(c.get("radius", 0)), T)
            if prims:
                break
        com, _ = _origin(l.find("inertial"))
        links[l.get("name")] = {"prims": prims, "com": com}

    for j in root.findall("joint"):
        xyz, rpy = _origin(j)
        lim, ax = j.find("limit"), j.find("axis")
        # An ElementTree element with no children is falsy, so `ax or default` silently
        # zeroes every axis and nothing rotates. Test against None.
        axis = _nums(ax.get("xyz")) if ax is not None else [0.0, 0.0, 1.0]
        if not any(axis):
            axis = [0.0, 0.0, 1.0]
        joints.append({
            "name": j.get("name"), "type": j.get("type", "fixed"),
            "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "xyz": xyz, "rpy": rpy, "axis": axis,
            "lower": float(lim.get("lower")) if lim is not None
            and lim.get("lower") is not None else None,
            "upper": float(lim.get("upper")) if lim is not None
            and lim.get("upper") is not None else None,
        })
        children[joints[-1]["parent"]].append(joints[-1])
        parent_of.setdefault(joints[-1]["child"], joints[-1])

    base = next((n for n in links if n not in parent_of), next(iter(links)))

    # A bone per joint: from the parent link's frame out to where the child sits.
    for j in joints:
        if j["parent"] in links:
            links[j["parent"]]["prims"].append(((0.0, 0.0, 0.0), tuple(j["xyz"])))

    # A chassis outline for any link that several limbs hang off. A hexapod whose body
    # is a mesh we did not fetch otherwise renders as six legs meeting at a point; the
    # hull of where its legs actually attach is the body, and it is the URDF's own
    # numbers rather than a shape we made up.
    for name, l in links.items():
        kids = [c for c in children.get(name, []) if any(abs(x) > 1e-9 for x in c["xyz"])]
        if len(kids) < 3:
            continue
        far = max(math.hypot(c["xyz"][0], c["xyz"][1]) for c in kids)
        ring = [c["xyz"] for c in kids
                if math.hypot(c["xyz"][0], c["xyz"][1]) > 0.25 * far]
        if len(ring) < 3:
            continue
        ring.sort(key=lambda p: math.atan2(p[1], p[0]))
        l["prims"] += [(tuple(ring[i]), tuple(ring[(i + 1) % len(ring)]))
                       for i in range(len(ring))]

    # A leaf with nothing but a mesh would end at its own origin, which drops the last
    # segment of every leg. Its centre of mass is roughly mid-segment, so draw to twice
    # that: inferred from the URDF's own inertial data rather than invented. Capped,
    # because a link whose inertial frame sits far off its own origin would otherwise
    # throw a metre-long spike across the picture.
    bones = [math.dist(a, b) for l in links.values() for a, b in l["prims"]]
    cap = 2.5 * (float(np.median(bones)) if bones else 0.1)
    for name, l in links.items():
        if not children.get(name) and len(l["prims"]) == 0:
            c = l["com"]
            if 1e-6 < math.dist((0, 0, 0), c) * 2 <= cap:
                l["prims"].append(((0.0, 0.0, 0.0), (2 * c[0], 2 * c[1], 2 * c[2])))
    return {"links": links, "joints": joints, "children": children, "base": base}


def forward_kinematics(scene: dict, q: dict) -> dict:
    """World transform per link, given a joint-angle dict."""
    out = {scene["base"]: np.eye(4)}
    stack, seen = [scene["base"]], set()
    while stack:
        p = stack.pop()
        if p in seen:                 # closed loops (Cassie) would otherwise not end
            continue
        seen.add(p)
        for j in scene["children"].get(p, []):
            if j["child"] in out:
                continue
            T = out[p] @ _T(j["xyz"], _rpy(*j["rpy"]))
            a = q.get(j["name"], 0.0)
            if j["type"] in ("revolute", "continuous"):
                T = T @ _T([0, 0, 0], _axis_rot(j["axis"], a))
            elif j["type"] == "prismatic":
                T = T @ _T(np.asarray(j["axis"], float) * a)
            out[j["child"]] = T
            stack.append(j["child"])
    return out


def world_segments(scene: dict, T: dict) -> list:
    segs = []
    for name, l in scene["links"].items():
        if name not in T or not l["prims"]:
            continue
        Tm = T[name]
        Rm, t = Tm[:3, :3], Tm[:3, 3]
        for a, b in l["prims"]:
            segs.append((Rm @ np.asarray(a) + t, Rm @ np.asarray(b) + t))
    return segs


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #

def fetch_urdfs() -> None:
    URDF_DIR.mkdir(parents=True, exist_ok=True)
    for r in ROBOTS:
        p = URDF_DIR / f"{r['key']}.urdf"
        if p.exists():
            continue
        print(f"  fetching {r['key']} <- {r['repo']}")
        with urllib.request.urlopen(r["url"], timeout=60) as fh:
            p.write_bytes(fh.read())


def tick_circuit(ir, decodes) -> tuple:
    """Run the int8 reference runtime and read the muscles.

    This is the generated ROS 2 node's ``step()`` with rclpy taken out: tick, low-pass
    the spike train into a rate because a motor neuron drives muscle through a calcium
    filter, then read each joint as rate(agonist) - rate(antagonist).
    """
    rt = Reference(ir, fast=True)
    motor = np.zeros(ir.n_neurons, bool)
    for d in decodes:
        motor[d.agonist] = True
        motor[d.antagonist] = True
    drive = np.zeros(ir.n_neurons, np.int64)
    drive[~motor] = DRIVE_Q88

    rates = np.zeros(ir.n_neurons)
    fired_trace = []
    for _ in range(WARMUP_TICKS):
        f = rt.tick(drive)
        rates = RATE_SMOOTHING * rates + (1 - RATE_SMOOTHING) * f
        fired_trace.append(int(f.sum()))
    return rates, fired_trace


def _neutral(j) -> tuple:
    """Where a joint rests, and how far it can swing either side of that.

    Neutral is the URDF's own zero pose where the joint allows it, and otherwise the
    closest pose that does. Mid-range is not a neutral: Cassie declares knee limits of
    [-2.86, -0.65], whose midpoint is -100 degrees, so resting a leg mid-range folds it
    up over the hip. A fifteen per cent margin at each stop leaves a joint pinned
    against a limit somewhere to move.
    """
    lo, hi = (j["lower"], j["upper"]) if j["lower"] is not None else (-0.5, 0.5)
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo < 1e-9:
        return lo, 0.0
    margin = 0.15 * (hi - lo)
    neutral = min(max(0.0, lo + margin), hi - margin)
    return neutral, min(hi - neutral, neutral - lo)


def joint_trajectory(spec, plan, scene, commands, n_frames) -> tuple:
    """Turn steady joint commands plus a gait clock into an angle per joint per frame.

    Amplitude and sign per joint are the circuit's. The clock is not: it is the library's
    Kuramoto ring holding the retargeter's phases, because the unfitted circuit does not
    oscillate. Within a leg the roles are offset by the textbook swing/stance quarter
    cycle, and the envelope is scaled to the robot's own URDF joint limits -- neither of
    those is from the fly either.
    """
    jt = {j["name"]: j for j in scene["joints"]}
    limbs = [l for l in spec.limbs]
    phases = np.array([next((b.phase for b in plan.bindings if b.limb == l.name), 0.0)
                       for l in limbs])
    control_dt = 1.0 / spec.control_hz
    step = R.phase_oscillator(len(limbs) or 1, phases if len(limbs) else np.zeros(1),
                              freq_hz=STEP_HZ, dt=control_dt, coupling=2.0)
    for _ in range(int(2.0 * spec.control_hz)):        # let the ring settle
        step()

    ROLE_PSI = {"coxa_yaw": 0.0, "coxa_roll": 0.0, "trochanter": math.pi / 2,
                "knee": math.pi, "ankle": 3 * math.pi / 2}

    # Every joint rests at its nearest legal pose, driven or not. Cassie's shin and
    # ankle are passive members of a four-bar and are not bound to any fly muscle, but
    # leaving them at a zero their own limits forbid leaves the foot behind the hip.
    rest = {j["name"]: _neutral(j)[0] for j in scene["joints"]}
    peak = max((abs(c["command"]) for c in commands.values()), default=1.0) or 1.0

    # One entry per joint the circuit drives: which limb's clock it follows, how far it
    # swings, and whether it is a hinge that oscillates or a wheel that turns.
    plan_j = []
    for li, limb in enumerate(limbs):
        # A wheel is a limb that is one continuous joint. "Continuous" alone does not
        # mean wheel -- Cassie's hips, knees and ankles are all continuous, and spinning
        # them through whole turns is how a biped turns into a zigzag.
        wheel = len(limb.joints) == 1
        for jname, role in zip(limb.joints, limb.joint_roles):
            c, j = commands.get(jname), jt.get(jname)
            if c is None or j is None:
                continue
            u = c["command"] / peak
            spins = wheel and j["type"] == "continuous"
            # A wheel gets a whole number of turns over the clip, so the GIF loops.
            turns = max(round(abs(u) * 2.0), 1) * (1 if u >= 0 else -1)
            # Neutral is the URDF's own zero pose where the joint allows it, and
            # otherwise the closest pose that does. Mid-range is not a neutral: Cassie
            # declares its knee limits [-2.86, -0.65], whose midpoint is -100 degrees,
            # so a mid-range "rest" folds the leg up over the hip. Keep a fifteen per
            # cent margin at each stop so a joint pinned at a limit can still move.
            neutral, room = _neutral(j)
            plan_j.append({"name": jname, "limb": li, "u": u, "spin": spins,
                           "omega": 2 * math.pi * turns / SECONDS,
                           "mid": neutral, "amp": 0.5 * room * u,
                           "psi": ROLE_PSI.get(role, 0.0)})

    n_steps = int(SECONDS * spec.control_hz)
    picks = set(np.linspace(0, n_steps - 1, n_frames).round().astype(int).tolist())
    frames, spin = [], defaultdict(float)
    for s in range(n_steps):
        theta = step()
        for pj in plan_j:
            if pj["spin"]:
                spin[pj["name"]] += pj["omega"] * control_dt
        if s not in picks:
            continue
        q = dict(rest)
        for pj in plan_j:
            if pj["spin"]:
                q[pj["name"]] = spin[pj["name"]]
            else:
                ang = 2 * math.pi * theta[pj["limb"]] + pj["psi"]
                q[pj["name"]] = pj["mid"] + pj["amp"] * math.sin(ang)
        frames.append(q)
    return frames, [float(x) for x in (phases if len(limbs) else [])]


def build(robot: dict, conn, ir, decodes, rates) -> dict:
    """Every API call for one robot, recorded as it runs."""
    key = robot["key"]
    urdf_path = URDF_DIR / f"{key}.urdf"
    rel = urdf_path.relative_to(ROOT)
    rec = Recorder({"nt": nt, "M": M, "conn": conn, "ir": ir, "np": np,
                    "OUT": OUT, "urdf": str(rel)})

    spec, report = rec.bind("morph, report", f"nt.morphology(urdf={str(rel)!r})",
                            "the robot's own URDF, unmodified")
    rec.env["morph"], rec.env["report"] = spec, report

    commands, wing = {}, None
    if spec.limbs:
        gait = robot["gait"]
        plan = rec.bind("plan", f"nt.retarget(conn, morph, gait={gait!r})",
                        "binds each limb to a fly leg and a gait phase")
        for b in plan.bindings:
            limb = next(l for l in spec.limbs if l.name == b.limb)
            for role, dec in b.joints.items():
                idx = limb.role_index(role)
                if idx is None:
                    continue
                commands[limb.joints[idx]] = {
                    "limb": b.limb, "role": role, "phase": b.phase,
                    "fly": f"{b.source_leg} {b.source_side}",
                    "agonist": int(len(dec.agonist)),
                    "antagonist": int(len(dec.antagonist)),
                    "command": float(dec.command(rates))}
    else:
        rec.env["plan"] = plan = R.Retarget(morph=spec, gait="none", bindings=[],
                                            unused_sources=[], note=robot["note"])
        dec = rec("M.decode_motor(ir, conn.neurons)",
                  "no joints to bind, so read the wing motor pools directly")
        w = {(d.side, d.joint): float(d.command(rates)) for d in dec if d.leg == "wing"}
        thrust = 0.5 * (w[("L", "power")] + w[("R", "power")])
        wing = {"power_L": w[("L", "power")], "power_R": w[("R", "power")],
                "steer_L": w[("L", "steering")], "steer_R": w[("R", "steering")],
                "thrust": thrust, "roll": w[("R", "steering")] - w[("L", "steering")]}

    emit = rec("nt.emit(ir, str(OUT / 'ros2' / %r), target='ros2', morph=morph, "
               "retarget_plan=plan)" % key,
               "a complete ament_python package: node, launch, package.xml, .fcx")

    scene = load_scene(urdf_path)
    n_frames = int(FPS * SECONDS)
    frames, phases = joint_trajectory(spec, plan, scene, commands, n_frames)

    # Bake forward kinematics once, here, so the GIF and the browser are driven by the
    # same numbers and cannot drift into two slightly different robots.
    # A quadrotor has nothing to articulate, so its one visible degree of freedom is
    # attitude. Bank it by the measured left/right steering-muscle difference -- the
    # mixing from wing muscle to rotor is this demo's, not the library's.
    rotors, W = [], np.eye(4)
    if wing:
        W = _T([0, 0, 0], _rpy(6.0 * wing["roll"], 0.0, 0.0))
        prad = _prop_radius(urdf_path)
        rotors = [{"link": n, "c": l["com"], "r": prad}
                  for n, l in scene["links"].items()
                  if "prop" in n and any(abs(x) > 1e-9 for x in l["com"])]

    drawable = {k for k, v in scene["links"].items() if v["prims"]}
    drawable |= {r["link"] for r in rotors}     # prop links carry no shape, only a frame
    Ts = []
    for q in frames:
        T = forward_kinematics(scene, q)
        Ts.append({k: [round(float(x), 4) for x in (W @ v)[:3, :].ravel()]
                   for k, v in T.items() if k in drawable})

    driven_links = {}
    for jname in commands:
        j = next((x for x in scene["joints"] if x["name"] == jname), None)
        if j:
            driven_links[j["child"]] = commands[jname]["role"]

    return {
        "key": key, "title": robot["title"], "kind": robot["kind"],
        "repo": robot["repo"], "url": robot["url"], "note": robot["note"],
        "gait": plan.gait, "control_hz": spec.control_hz,
        "n_limbs": spec.n_limbs, "n_joints": spec.n_joints,
        "n_actuated": report["n_actuated"],
        "n_actuated_in_limbs": report["n_actuated_in_limbs"],
        "n_joints_total": report["n_joints_total"], "base_link": report["base_link"],
        "limbs_over_4dof": report["limbs_over_4dof"],
        "role_confidence": report["role_confidence"],
        "roles_uncertain": bool(report["limbs_over_4dof"]),
        "plan_summary": plan.summary(), "plan_note": plan.note,
        "unused_sources": plan.unused_sources,
        "commands": commands, "wing": wing, "phases": phases,
        "emit": emit, "api": rec.log,
        "links": {k: [[list(map(lambda z: round(float(z), 5), a)),
                       list(map(lambda z: round(float(z), 5), b))]
                      for a, b in v["prims"]]
                  for k, v in scene["links"].items() if v["prims"]},
        "driven_links": driven_links, "rotors": rotors,
        "frames": Ts,
    }


# --------------------------------------------------------------------------- #
# GIFs
# --------------------------------------------------------------------------- #

PALETTE = {"bg": (14, 17, 23), "grid": (34, 40, 52), "bone": (122, 134, 154),
           "driven": (86, 204, 242), "hot": (255, 138, 76), "text": (226, 232, 240),
           "dim": (120, 132, 150)}


def _project(p, az, el, scale, cx, cy):
    ca, sa, ce, se = math.cos(az), math.sin(az), math.cos(el), math.sin(el)
    x = p[0] * ca + p[1] * sa
    y = -p[0] * sa + p[1] * ca
    return (cx + scale * x, cy - scale * (p[2] * ce - y * se))


def _geometry(d: dict, W: int, H: int, az: float, el: float, pad: int, drop: int) -> dict:
    """Where this robot sits in a W x H tile.

    Fit from the projected outline over every frame, not from a world-space radius: a
    7-DOF arm is half a metre tall and five centimetres wide, and a scale taken from its
    largest world dimension walks it off the bottom of the tile.
    """
    pts = []
    for T in d["frames"]:
        for name, segs in d["links"].items():
            m = T.get(name)
            if m is None:
                continue
            m = np.array(m).reshape(3, 4)
            for a, b in segs:
                pts.append(m[:, :3] @ np.asarray(a) + m[:, 3])
                pts.append(m[:, :3] @ np.asarray(b) + m[:, 3])
    P = np.array(pts) if pts else np.zeros((1, 3))
    ctr = P.mean(axis=0)
    flat = np.array([_project(q - ctr, az, el, 1.0, 0.0, 0.0) for q in P])
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    scale = min((W - 2 * pad) / max(hi[0] - lo[0], 1e-6),
                (H - 2 * pad - drop) / max(hi[1] - lo[1], 1e-6))
    return {"scale": scale, "ctr": ctr,
            "cx": W / 2 - scale * (lo[0] + hi[0]) / 2,
            "cy": H / 2 + drop / 2 - scale * (lo[1] + hi[1]) / 2,
            "span": float(max(np.ptp(P[:, 0]), np.ptp(P[:, 1]), 1e-3))}


def _paint(dr, d: dict, fi: int, geo: dict, az: float, el: float,
           ox: int = 0, oy: int = 0) -> None:
    """Draw one robot, one frame, into an already-open ImageDraw at offset (ox, oy)."""
    T = d["frames"][fi % len(d["frames"])]
    ctr, s = geo["ctr"], geo["scale"]

    def P(v):
        x, y = _project(np.asarray(v) - ctr, az, el, s, geo["cx"], geo["cy"])
        return (x + ox, y + oy)

    g, z0 = geo["span"] * 0.75, ctr[2]
    for i in range(-3, 4):
        u = i * g / 3
        for a, b in (((u, -g, z0), (u, g, z0)), ((-g, u, z0), (g, u, z0))):
            dr.line([P(a), P(b)], fill=PALETTE["grid"], width=1)

    for name, segs in d["links"].items():
        m = T.get(name)
        if m is None:
            continue
        m = np.array(m).reshape(3, 4)
        on = name in d["driven_links"]
        col = PALETTE["driven"] if on else PALETTE["bone"]
        for a, b in segs:
            dr.line([P(m[:, :3] @ np.asarray(a) + m[:, 3]),
                     P(m[:, :3] @ np.asarray(b) + m[:, 3])],
                    fill=col, width=3 if on else 2)

    for name, m in T.items():                 # a dot per link frame: thin, mesh-only
        if name not in d["links"]:            # robots are unreadable as bare bones
            continue
        m = np.array(m).reshape(3, 4)
        x, y = P(m[:, 3])
        on = name in d["driven_links"]
        r = 3 if on else 2
        dr.ellipse([x - r, y - r, x + r, y + r],
                   fill=PALETTE["driven"] if on else PALETTE["bone"])

    if d["wing"] and d["rotors"]:             # the drone: rotor discs, spun by thrust
        spin = 2 * math.pi * fi / 9 * (1 + 4 * d["wing"]["thrust"])
        for rot in d["rotors"]:
            m = T.get(rot["link"])
            if m is None:
                continue
            m = np.array(m).reshape(3, 4)
            c = m[:, :3] @ np.asarray(rot["c"]) + m[:, 3]
            ring = [P(c + m[:, :3] @ np.array(
                [rot["r"] * math.cos(spin + 2 * math.pi * k / 14),
                 rot["r"] * math.sin(spin + 2 * math.pi * k / 14), 0.0]))
                for k in range(14)]
            dr.line(ring + [ring[0]], fill=PALETTE["hot"], width=2)


def _save_gif(frames, path: Path, colors: int = 64) -> None:
    """Write an animated GIF on one shared palette, so frames delta-encode."""
    from PIL import Image
    base = frames[0].quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
    seq = [base] + [f.quantize(palette=base, dither=Image.Dither.NONE)
                    for f in frames[1:]]
    seq[0].save(path, save_all=True, append_images=seq[1:],
                duration=int(1000 / FPS), loop=0, optimize=True, disposal=1)


def render_gif(d: dict, path: Path, size=(460, 360)) -> None:
    from PIL import Image, ImageDraw

    W, H = size
    az, el = math.radians(35), math.radians(22)
    geo = _geometry(d, W, H, az, el, pad=46, drop=28)

    frames = []
    for fi in range(len(d["frames"])):
        img = Image.new("RGB", (W, H), PALETTE["bg"])
        dr = ImageDraw.Draw(img)
        _paint(dr, d, fi, geo, az, el)
        dr.text((14, 12), d["title"], fill=PALETTE["text"])
        dr.text((14, 28), f"{d['kind']} · {d['n_joints']} joints · gait {d['gait']}",
                fill=PALETTE["dim"])
        dr.text((14, H - 22),
                f"{len(d['commands'])} joints driven by fly muscle pairs",
                fill=PALETTE["driven"])
        frames.append(img)
    _save_gif(frames, path)


def render_sheet(robots: list, path: Path, tile=(320, 250), cols: int = 4) -> None:
    """All eight, one loop, one file -- the picture that goes on the README.

    Every panel is redrawn from the same baked kinematics the single tiles use, rather
    than pasted from them, so the sheet is not a second quantization of a first one.
    """
    from PIL import Image, ImageDraw

    tw, th = tile
    rows = (len(robots) + cols - 1) // cols
    W, H = tw * cols, th * rows + 30
    az, el = math.radians(35), math.radians(22)
    geos = [_geometry(d, tw, th, az, el, pad=30, drop=26) for d in robots]

    frames = []
    for fi in range(len(robots[0]["frames"])):
        img = Image.new("RGB", (W, H), PALETTE["bg"])
        dr = ImageDraw.Draw(img)
        dr.text((14, 10), "Eight robots, one fly circuit", fill=PALETTE["text"])
        dr.text((208, 10),
                "cyan = joint driven by a real antagonist muscle pair   ·   "
                "grey = undriven   ·   kinematic playback, no contact model",
                fill=PALETTE["dim"])
        for i, (d, geo) in enumerate(zip(robots, geos)):
            ox, oy = (i % cols) * tw, 30 + (i // cols) * th
            _paint(dr, d, fi, geo, az, el, ox, oy)
            dr.text((ox + 11, oy + 9), d["title"], fill=PALETTE["text"])
            dr.text((ox + 11, oy + 23),
                    f"{d['kind']} · {len(d['commands'])}/{d['n_joints']} driven",
                    fill=PALETTE["dim"])
            if i % cols:
                dr.line([(ox, oy), (ox, oy + th)], fill=PALETTE["grid"], width=1)
            if i >= cols:
                dr.line([(ox, oy), (ox + tw, oy)], fill=PALETTE["grid"], width=1)
        frames.append(img)
    _save_gif(frames, path, colors=32)


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gifs", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("fetching URDFs")
    fetch_urdfs()

    print("compiling the circuit (shared by all eight robots)")
    shared = Recorder({"nt": nt, "M": M})
    conn = shared.bind("conn", "nt.load()", "male-CNS v1.0, from the cached CSR index")
    ir_raw = shared.bind("ir", f"nt.circuit(conn, {CIRCUIT!r})",
                         "T1-T3 intrinsic + motor: six legs and their coupling")
    decodes = shared.bind("decodes", "M.decode_motor(ir, conn.neurons)",
                          "motor neurons grouped into antagonist pairs by muscle name")
    ir, _ = shared.bind("ir, prune_stats", f"nt.prune(ir, p_real={P_REAL})",
                        "drop edges unlikely to be a real pathway")
    shared.env["ir"] = ir
    ir, qrep = shared.bind("ir, quant", f"nt.quantize(ir, bits={BITS})",
                           "log codebook + delta index")
    shared.env["ir"] = ir
    assert np.array_equal(ir_raw.body_ids, ir.body_ids), \
        "prune/quantize reordered neurons; decode indices would be wrong"

    t0 = time.perf_counter()
    rates, fired = tick_circuit(ir, decodes)
    print(f"  ticked {WARMUP_TICKS} x {ir.dynamics.dt * 1e3:g} ms in "
          f"{time.perf_counter() - t0:.2f}s  ({np.mean(fired):.0f} spikes/tick)")

    robots = []
    for r in ROBOTS:
        print(f"  {r['key']:9s}", end=" ", flush=True)
        d = build(r, conn, ir, decodes, rates)
        robots.append(d)
        print(f"{d['n_limbs']} limbs / {d['n_joints']} joints / "
              f"{len(d['commands'])} driven")

    if not args.no_gifs:
        print("rendering GIFs")
        for d in robots:
            p = OUT / f"{d['key']}.gif"
            render_gif(d, p)
            print(f"  {p.name:16s} {p.stat().st_size / 1024:6.0f} KB")
        sheet = ROOT / "docs" / "img" / "form-factors.gif"
        render_sheet(robots, sheet)
        print(f"  {sheet.name:16s} {sheet.stat().st_size / 1024:6.0f} KB  -> {sheet}")

    payload = {
        "generated": time.strftime("%Y-%m-%d"),
        "dataset": conn.meta.dataset,
        "circuit": CIRCUIT, "p_real": P_REAL, "bits": BITS,
        "drive_q88": DRIVE_Q88, "warmup_ticks": WARMUP_TICKS,
        "tick_dt_ms": float(ir.dynamics.dt * 1e3),
        "n_neurons": int(ir.n_neurons), "n_edges": int(ir.n_edges),
        "spikes_per_tick": float(np.mean(fired)),
        "quant": {"drive_err": float(qrep.drive_rel_err_mean),
                  "sign_flips": int(qrep.drive_sign_flips),
                  "edge_retention": float(qrep.edge_retention)},
        "shared_api": shared.log, "fps": FPS,
        "robots": robots,
    }
    # The viewer is checked in and the data is not, so drop a copy of the page beside
    # the data it needs. artifacts/ is generated and ignored; opening
    # artifacts/formfactors/index.html gives a self-contained local viewer.
    (OUT / "index.html").write_text((Path(__file__).parent / "viewer.html").read_text())
    (OUT / "data.json").write_text(json.dumps(payload, separators=(",", ":")))
    (OUT / "data.js").write_text("window.FF = " + json.dumps(payload, separators=(",", ":")) + ";")
    print(f"wrote {OUT/'data.json'}  "
          f"({(OUT/'data.json').stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
