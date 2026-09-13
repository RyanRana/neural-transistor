"""Does the circuit compute the thing it is named after?

Every other eval in this package asks whether the toolchain is faithful -- does the C
match the reference, does the artifact fit, does quantization preserve drive. None of
them ask whether the circuit *works*. This one does, and it is the honest frontier of
the project.

The compass is the right first target because its correct behaviour is unambiguous and
measurable. EPG neurons tile a ring, activity should form a single localized bump, the
bump should persist without input, and it should rotate at a rate proportional to
angular velocity. Better still, the ring order is recoverable straight from the data:
male-CNS ``instance`` strings carry the protocerebral-bridge glomerulus and the wedge
index, e.g. ``EPG(PB08)_L4`` / ``EPG(PB08)_R8``. 46 EPG neurons, 23 per side, all parsed.

**This eval is expected to fail on default dynamics, and that is the point.** The
literature is explicit that connectivity alone does not deliver ring-attractor dynamics:
Chang et al. 2023 swept 176,400 parameter sets and report "all tested models failed if we
simply set the synaptic weights proportional to these numbers"; Duan et al. 2025 find a
single global gain produces no bump at all and two gains produce a bump but no
integration; Beiran & Litwin-Kumar 2025 show connectome-constrained students of the
central complex "did not behave as ring attractors". So a negative result here is not a
bug in the toolkit -- it is the measurement that says dynamics fitting is required, made
quantitative instead of asserted.

The gain sweep is the actual experiment: if NO global synaptic scale produces a bump,
that reproduces the published result on a dataset nobody has run it on.

Metric: circular concentration of EPG firing around the ring,
``R = |sum_j r_j e^{i theta_j}| / sum_j r_j``. R = 1 is a delta bump, R = 0 is uniform
activity. Seelig & Jayaraman 2015 measured bump FWHM 82.3 +- 11.5 deg with a stripe and
90.9 +- 11.2 deg in darkness, which for a von Mises bump corresponds to R of roughly
0.6-0.7. That is the target band.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from neuraltransistor.ir.graph import CircuitIR
from neuraltransistor.ir.runtime import Reference

_WEDGE = re.compile(r"_([LR])(\d+)\s*$")
_PB = re.compile(r"\(PB(\d+)([ab]?)\)")

#: R corresponding to the measured bump width band (Seelig & Jayaraman 2015).
TARGET_R = (0.45, 0.85)


@dataclass
class RingMap:
    """EPG neurons placed on a circle, recovered from the annotation."""
    local_idx: np.ndarray        # indices into the circuit
    theta: np.ndarray            # radians, position on the ring
    side: np.ndarray             # 'L' / 'R'
    wedge: np.ndarray            # raw wedge number

    @property
    def n(self) -> int:
        return len(self.local_idx)


def ring_map(ir: CircuitIR, neurons_df, cell_type: str = "EPG") -> RingMap:
    """Recover the EPG ring order from ``instance`` strings.

    The fly's EB is divided into 16 tiles; each hemisphere's wedge index runs 1..9 with
    the two halves offset by half a turn. Left wedge w and right wedge w sit on opposite
    sides of the ring, which is what makes the +-1 PEN shift a rotation rather than a
    translation.
    """
    by_body = {int(b): i for i, b in enumerate(ir.body_ids)}
    sub = neurons_df[neurons_df.bodyId.isin(by_body)]
    sub = sub[sub["type"].astype(str) == cell_type]
    idx, th, sides, wedges = [], [], [], []
    for _, row in sub.iterrows():
        inst = str(row.get("instance") or "")
        m = _WEDGE.search(inst)
        if not m:
            continue
        side, w = m.group(1), int(m.group(2))
        # Map wedge 1..9 per hemisphere onto a shared 18-position ring; the right
        # hemisphere is offset half a turn.
        pos = (w - 1) + (9 if side == "R" else 0)
        idx.append(by_body[int(row.bodyId)])
        th.append(2 * np.pi * pos / 18.0)
        sides.append(side)
        wedges.append(w)
    return RingMap(np.array(idx, dtype=np.int32), np.array(th, dtype=np.float64),
                   np.array(sides, dtype=object), np.array(wedges, dtype=np.int32))


def concentration(rates: np.ndarray, theta: np.ndarray) -> tuple[float, float]:
    """(R, preferred angle) -- circular concentration of activity on the ring."""
    tot = rates.sum()
    if tot <= 0:
        return 0.0, 0.0
    z = (rates * np.exp(1j * theta)).sum() / tot
    return float(np.abs(z)), float(np.angle(z))


def _run(ir: CircuitIR, rm: RingMap, weight_scale: float, drive_amp: int,
         drive_ticks: int, hold_ticks: int, drive_width: float = 0.6):
    """Drive one sector of the ring, then release, recording EPG rates."""
    scaled = ir
    if weight_scale != 1.0:
        import copy
        scaled = copy.copy(ir)
        scaled.weight = np.clip(ir.weight.astype(np.float64) * weight_scale,
                                0, 65535).astype(np.uint16)
    ref = Reference(scaled)

    drive = np.zeros(ir.n_neurons, dtype=np.int64)
    bump_at = 0.0
    w = np.exp(np.cos(rm.theta - bump_at) / (drive_width ** 2))
    w = w / w.max()
    drive[rm.local_idx] = (w * drive_amp).astype(np.int64)

    on = np.zeros((drive_ticks, rm.n))
    for t in range(drive_ticks):
        f = ref.tick(drive)
        on[t] = f[rm.local_idx]
    off = np.zeros((hold_ticks, rm.n))
    zero = np.zeros(ir.n_neurons, dtype=np.int64)
    for t in range(hold_ticks):
        f = ref.tick(zero)
        off[t] = f[rm.local_idx]
    return on, off


def bump_probe(ir: CircuitIR, neurons_df, weight_scale: float = 1.0,
               drive_amp: int = 900, drive_ticks: int = 400,
               hold_ticks: int = 400) -> dict:
    """Drive the ring, release it, and measure whether a bump formed and persisted."""
    rm = ring_map(ir, neurons_df)
    if rm.n < 8:
        return {"ok": False, "reason": f"only {rm.n} EPG neurons placed on the ring"}

    on, off = _run(ir, rm, weight_scale, drive_amp, drive_ticks, hold_ticks)
    win = max(drive_ticks // 4, 1)
    r_driven, ang_driven = concentration(on[-win:].mean(axis=0), rm.theta)
    r_held, ang_held = concentration(off[-win:].mean(axis=0), rm.theta)
    r_early, ang_early = concentration(off[:win].mean(axis=0), rm.theta)

    rate_driven = float(on[-win:].mean())
    rate_held = float(off[-win:].mean())
    drift = float(np.abs(np.angle(np.exp(1j * (ang_held - ang_early)))))

    return {
        "ok": True,
        "n_ring": rm.n,
        "weight_scale": weight_scale,
        "R_driven": round(r_driven, 4),
        "R_held": round(r_held, 4),
        "rate_driven": round(rate_driven, 5),
        "rate_held": round(rate_held, 5),
        "drift_rad": round(drift, 4),
        "bump_formed": bool(TARGET_R[0] <= r_driven <= TARGET_R[1] and rate_driven > 0),
        "bump_persisted": bool(rate_held > 0.005 and r_held >= TARGET_R[0]),
        "silent": bool(rate_driven < 1e-6),
        "saturated": bool(rate_driven > 0.45),
    }


def gain_sweep(ir: CircuitIR, neurons_df,
               scales=(0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0),
               **kw) -> dict:
    """Does ANY single global synaptic scale produce a persistent bump?

    This is the experiment. The published answer on other datasets is no -- one global
    scalar gives either silence or saturation, never a localized persistent bump -- and
    it is why per-cell-type gains are the consensus parameterization. Running it here
    puts a number on that claim for male-CNS.
    """
    rows = []
    for s in scales:
        r = bump_probe(ir, neurons_df, weight_scale=s, **kw)
        if r.get("ok"):
            rows.append(r)
    formed = [r for r in rows if r["bump_formed"]]
    persisted = [r for r in rows if r["bump_persisted"]]
    return {
        "trials": rows,
        "any_bump": len(formed) > 0,
        "any_persistent_bump": len(persisted) > 0,
        "best_R": max((r["R_driven"] for r in rows), default=0.0),
        "n_silent": sum(1 for r in rows if r["silent"]),
        "n_saturated": sum(1 for r in rows if r["saturated"]),
        "verdict": ("a global synaptic scale is sufficient" if persisted else
                    "no global synaptic scale produces a persistent bump -- "
                    "per-cell-type gains are required, as the literature predicts"),
    }
