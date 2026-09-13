"""Cutting a runnable circuit out of the whole CNS.

Extraction is three decisions, and the defaults matter:

1. **Which neurons.** A Selector over the annotation table.
2. **What to do with the boundary.** A circuit pulled out of a brain has afferents with
   no source and efferents with no target. ``closure`` decides how much of that
   neighbourhood to pull in with it.
3. **What is a port.** Sensory/descending afferents become input ports; motor/descending
   efferents become output ports. These are the only places a robot touches the circuit.

Modulatory neurons (dopaminergic, octopaminergic, serotonergic) are split out of the
chemical edge list into their own multiplicative edge list here, because compiling them
as ordinary synapses is what destroys the gating behaviour they exist to produce.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from neuraltransistor.circuit.select import Sel, Selector
from neuraltransistor.circuit.sign import assign_signs
from neuraltransistor.data.source import Connectome
from neuraltransistor.ir.graph import CircuitIR, Dynamics, Port

Closure = Literal["none", "afferent", "efferent", "both"]


def extract(
    conn: Connectome,
    sel: Selector,
    name: str = "circuit",
    closure: Closure = "none",
    closure_min_weight: int = 5,
    closure_max_neurons: int = 2000,
    confidence: float = 0.5,
    dt: float = 0.0005,
    split_modulatory: bool = True,
) -> CircuitIR:
    """Extract the subgraph selected by ``sel``.

    Parameters
    ----------
    closure
        ``"none"`` keeps exactly the selected neurons. ``"afferent"`` also pulls in
        strong presynaptic partners (the circuit's drivers), ``"efferent"`` strong
        postsynaptic targets (what it commands), ``"both"`` each.
    closure_min_weight
        A partner must connect with at least this many synapses to be pulled in. The
        default of 5 sits at the 95th percentile of the whole-connectome weight
        distribution, so it admits real pathways and rejects the 62% of edges that are
        a single synapse.
    closure_max_neurons
        Hard cap, so a closure on a hub neuron cannot silently drag in the brain.
    """
    df = conn.neurons
    core = sel.idx(df)
    if len(core) == 0:
        raise ValueError(f"selector {sel.label} matched no neurons")

    keep = set(core.tolist())
    if closure in ("efferent", "both"):
        keep |= _closure_out(conn, core, closure_min_weight, closure_max_neurons)
    if closure in ("afferent", "both"):
        keep |= _closure_in(conn, core, closure_min_weight, closure_max_neurons)

    sel_idx = np.sort(np.fromiter(keep, dtype=np.int32, count=len(keep)))
    n = len(sel_idx)

    # global index -> local index
    member = np.zeros(len(df), dtype=bool)
    member[sel_idx] = True
    local = np.full(len(df), -1, dtype=np.int32)
    local[sel_idx] = np.arange(n, dtype=np.int32)

    sub = df.iloc[sel_idx]
    sign, kind, rep = assign_signs(sub, confidence=confidence)

    # --- gather internal edges -------------------------------------------------
    rows_idx: list[np.ndarray] = []
    rows_w: list[np.ndarray] = []
    counts = np.zeros(n, dtype=np.int64)
    for li, gi in enumerate(sel_idx):
        post, w = conn.out_edges(int(gi))
        m = member[post]
        if not m.any():
            rows_idx.append(np.zeros(0, np.int32)); rows_w.append(np.zeros(0, np.uint16))
            continue
        p = local[post[m]]
        ww = w[m]
        order = np.argsort(p, kind="stable")
        rows_idx.append(p[order]); rows_w.append(ww[order].astype(np.uint16))
        counts[li] = len(p)

    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = (np.concatenate(rows_idx) if n else np.zeros(0, np.int32)).astype(np.int32)
    weight = (np.concatenate(rows_w) if n else np.zeros(0, np.uint16)).astype(np.uint16)

    ir = CircuitIR(
        name=name,
        body_ids=sub["bodyId"].to_numpy().astype(np.int64),
        types=sub["type"].astype(str).tolist(),
        superclasses=sub["superclass"].astype(str).tolist(),
        sign=sign, sign_kind=kind,
        indptr=indptr, indices=indices, weight=weight,
        dynamics=Dynamics.default(n, dt=dt),
        provenance={
            "dataset": conn.meta.dataset, "minconf": conn.meta.minconf,
            "selector": sel.label, "closure": closure,
            "closure_min_weight": closure_min_weight,
            "n_core": int(len(core)), "n_closure": int(n - len(core)),
            "sign_report": str(rep),
            "sign_coverage": round(rep.coverage, 4),
            "nmj_corrections": rep.n_nmj_corrected,
            "index_retained_edges": round(conn.meta.retained, 4),
        },
    )

    if split_modulatory:
        _split_modulatory(ir)
    _attach_ports(ir, sub)
    return ir


def _closure_out(conn: Connectome, core: np.ndarray, minw: int, cap: int) -> set:
    found: dict[int, int] = {}
    for gi in core:
        post, w = conn.out_edges(int(gi))
        m = w >= minw
        for p, ww in zip(post[m], w[m]):
            found[int(p)] = found.get(int(p), 0) + int(ww)
    return _top(found, core, cap)


def _closure_in(conn: Connectome, core: np.ndarray, minw: int, cap: int) -> set:
    found: dict[int, int] = {}
    for gi in core:
        pre, w = conn.in_edges(int(gi))
        m = w >= minw
        for p, ww in zip(pre[m], w[m]):
            found[int(p)] = found.get(int(p), 0) + int(ww)
    return _top(found, core, cap)


def _top(found: dict[int, int], core: np.ndarray, cap: int) -> set:
    for c in core.tolist():
        found.pop(int(c), None)
    if len(found) <= cap:
        return set(found)
    ranked = sorted(found.items(), key=lambda kv: -kv[1])[:cap]
    return {k for k, _ in ranked}


def _split_modulatory(ir: CircuitIR) -> None:
    """Move edges out of modulatory neurons into the multiplicative edge list.

    A dopaminergic terminal scales a target's gain; it does not inject current. Left in
    the chemical list it compiles to an additive kernel, which is the wrong operation
    and quietly removes the circuit's ability to gate.
    """
    mod = np.flatnonzero(ir.sign_kind == 3).astype(np.int32)
    if len(mod) == 0:
        return
    is_mod = np.zeros(ir.n_neurons, dtype=bool)
    is_mod[mod] = True
    src = np.repeat(np.arange(ir.n_neurons, dtype=np.int32), np.diff(ir.indptr))
    take = is_mod[src]
    if not take.any():
        return
    ir.mod_pre = src[take].astype(np.int32)
    ir.mod_post = ir.indices[take].astype(np.int32)
    ir.mod_weight = ir.weight[take].astype(np.uint16)

    keep = ~take
    kept_src = src[keep]
    ir.indices = ir.indices[keep].astype(np.int32)
    ir.weight = ir.weight[keep].astype(np.uint16)
    counts = np.bincount(kept_src, minlength=ir.n_neurons)
    ir.indptr = np.zeros(ir.n_neurons + 1, dtype=np.int64)
    np.cumsum(counts, out=ir.indptr[1:])


def _attach_ports(ir: CircuitIR, sub) -> None:
    supers = np.array(ir.superclasses, dtype=object)
    deg_out = np.diff(ir.indptr)
    deg_in = ir.in_degree()

    def port(mask, nm, role, note=""):
        w = np.flatnonzero(mask).astype(np.int32)
        if len(w):
            return Port(nm, w, role, note)
        return None

    sens = np.isin(supers, ["vnc_sensory", "cb_sensory", "ol_sensory",
                            "sensory_ascending", "vnc_sensory_tbc", "cb_sensory_tbc"])
    desc = supers == "descending_neuron"
    asc = supers == "ascending_neuron"
    motor = np.isin(supers, ["vnc_motor", "cb_motor", "vnc_efferent", "cb_efferent"])
    vis = np.isin(supers, ["visual_projection", "ol_intrinsic"])

    for p in [
        port(sens, "sensory", "sensory", "afferent; a sensor adapter must drive these"),
        port(desc & (deg_in == 0), "descending_in", "descending",
             "command input from the brain, not resolved inside this circuit"),
        port(vis & (deg_in == 0), "visual_in", "sensory", "unresolved visual afferent"),
    ]:
        if p: ir.in_ports.append(p)

    for p in [
        port(motor, "motor", "motor", "efferent; drives actuators"),
        port(desc & (deg_out == 0), "descending_out", "descending",
             "command output toward the nerve cord"),
        port(asc, "ascending", "ascending", "feedback toward the brain"),
    ]:
        if p: ir.out_ports.append(p)


# --------------------------------------------------------------------------- #
# The standard circuit library.
# Each entry is a selector plus the extraction settings that make it runnable.
# --------------------------------------------------------------------------- #

LIBRARY = {
    "compass": dict(
        role="heading estimator",
        does="Gyro in, heading out. Holds a bearing with no GPS and no magnetometer.",
        sel=Sel.type(r"^(EPG|PEN|PEG|Delta7|ER|EL)"),
        doc="Central-complex heading system: EPG bump, PEN angular-velocity "
            "integration, Delta7 global inhibition, ER visual gating.",
        closure="none",
    ),
    "path_integration": dict(
        role="dead reckoning",
        does="Tracks where you are relative to where you started, from self-motion alone.",
        sel=Sel.type(r"^(EPG|PEN|PEG|Delta7|ER|PFN|hDelta|vDelta|FC2|PFL)"),
        doc="Compass plus the vector memory and steering readout: PFN optic-flow "
            "input, hDelta/vDelta accumulation, FC2 goal, PFL steering.",
        closure="none",
    ),
    "steering": dict(
        role="goal steering",
        does="Given a heading and a goal bearing, produce a turn command.",
        sel=Sel.type(r"^(FC2|PFL|hDelta)"),
        doc="The goal-to-turn readout alone. The narrowest useful waist in the "
            "navigation stack.",
        closure="efferent",
    ),
    "optic_motion": dict(
        role="optical flow",
        does="Per-pixel motion direction from a camera. No training data.",
        sel=Sel.type(r"^(T4|T5|Mi1|Mi4|Mi9|Tm1|Tm2|Tm3|Tm4|Tm9|Tm20|C2|C3|L[1-5])"),
        doc="Elementary motion detection: the T4/T5 correlator and its columnar "
            "input pathway.",
        closure="none",
    ),
    "looming": dict(
        role="collision detector",
        does="Fires before you hit something. Scales correctly with approach speed.",
        sel=Sel.type(r"^(LPLC|LC[0-9]|LPi|GF)"),
        doc="Collision avoidance: lobula columnar feature detectors including the "
            "LPLC2 looming pathway.",
        closure="efferent",
    ),
    "optic_flow": dict(
        role="self-motion estimator",
        does="Wide-field flow to ego-motion: are you translating or rotating.",
        sel=Sel.type(r"^(HS|VS|H[12]|CH|LPi|LPT)"),
        doc="Wide-field optic-flow integration: the lobula plate tangential cells "
            "that act as matched filters for self-motion.",
        closure="afferent",
    ),
    "leg_T1": dict(
        role="single-leg controller",
        does="One leg: swing, stance, load and position reflexes.",
        sel=Sel.neuromere("T1") & (Sel.superclass("vnc_intrinsic") | Sel.motor()
                                   | Sel.superclass("vnc_sensory")),
        doc="One front-leg local circuit: premotor interneurons, motor neurons and "
            "leg proprioceptors of the prothoracic neuromere.",
        closure="none",
    ),
    "leg_T2": dict(
        role="single-leg controller",
        does="One leg, middle-segment variant.",
        sel=Sel.neuromere("T2") & (Sel.superclass("vnc_intrinsic") | Sel.motor()
                                   | Sel.superclass("vnc_sensory")),
        doc="One middle-leg local circuit.",
        closure="none",
    ),
    "leg_T3": dict(
        role="single-leg controller",
        does="One leg, rear-segment variant. The propulsive one.",
        sel=Sel.neuromere("T3") & (Sel.superclass("vnc_intrinsic") | Sel.motor()
                                   | Sel.superclass("vnc_sensory")),
        doc="One hind-leg local circuit.",
        closure="none",
    ),
    "legs_all": dict(
        role="gait controller",
        does="Six legs and the coupling that keeps them in phase.",
        sel=Sel.neuromere("T1", "T2", "T3") & (Sel.superclass("vnc_intrinsic")
                                               | Sel.motor() | Sel.superclass("vnc_sensory")),
        doc="All six legs plus the intersegmental interneurons that coordinate them. "
            "This is where inter-leg coordination actually lives.",
        closure="none",
    ),
    "descending": dict(
        role="command bus",
        does="The entire brain-to-body channel: 1,314 command lines.",
        sel=Sel.descending(),
        doc="The entire brain-to-body command bus: 1,314 neurons. Every behavioural "
            "command the brain issues passes through here.",
        closure="none",
    ),
    "looming_pathway": dict(
        role="camera-to-brake pipeline",
        does="Raw camera frames to an escape command, end to end. The only circuit you can drive straight from a sensor.",
        sel=Sel.type(r"^(L[1-5]$|Tm1$|Tm2$|Tm4$|Tm9$|Tm20$|Tm5Y$|T5[a-d]$|T4[a-d]$"
                     r"|LPLC[0-9]|LC4$|LPi|Y3$|DNp0[1-3]$)"),
        doc="The full OFF collision pathway, retina to escape command: lamina monopolar "
            "cells, medulla transmedullary cells, T4/T5 motion detectors, LC4 and LPLC2 "
            "looming detectors, and DNp01 -- the giant fiber itself. The only circuit "
            "here that can be driven directly by a camera. Note DNp01, not GF: the types "
            "named GFC1-4 are giant-fiber-COUPLED interneurons in the nerve cord, and "
            "selecting those instead gives you a circuit whose output is disconnected "
            "from vision.",
        closure="none",
    ),
    "gate": dict(
        role="threat/valence gate",
        does="Learned good/bad signal that multiplies the gain of everything downstream.",
        sel=Sel.type(r"^(KC|MBON|PAM|PPL|PPM|APL|DPM)"),
        doc="Mushroom body: sparse Kenyon-cell context code, 97 MBON valence "
            "readouts, and the dopaminergic neurons that gate them. The fly's "
            "attention/valence mechanism.",
        closure="none",
    ),
    "gate_readout": dict(
        role="valence readout",
        does="The gate without the memory. Kilobyte-scale attention.",
        sel=Sel.type(r"^(MBON|PAM|PPL1)"),
        doc="The gate without the Kenyon cells: valence readout plus dopaminergic "
            "modulation only. Kilobyte-scale.",
        closure="none",
    ),
}


def from_library(conn: Connectome, key: str, **overrides) -> CircuitIR:
    if key not in LIBRARY:
        raise KeyError(f"unknown circuit {key!r}; have {sorted(LIBRARY)}")
    spec = dict(LIBRARY[key])
    doc = spec.pop("doc")
    role = spec.pop("role", "")
    does = spec.pop("does", "")
    sel = spec.pop("sel")
    spec.update(overrides)
    ir = extract(conn, sel, name=key, **spec)
    ir.provenance["doc"] = doc
    ir.provenance["role"] = role
    ir.provenance["does"] = does
    return ir
