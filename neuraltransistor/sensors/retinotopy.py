"""Wiring a lattice of ommatidia onto the columnar neurons that read them.

The medulla is retinotopic: each visual column has one copy of each columnar cell type,
and ``assignedOlHex1``/``assignedOlHex2`` say which column a neuron belongs to. That makes
the sensor boundary exact rather than approximate -- ommatidium (h1,h2) drives the L1,
L2, Mi1, Tm9 ... that sit in column (h1,h2), with no interpolation and no guessing.

Measured coverage in male-CNS v1.0: 23,720 neurons carry hex coordinates, all of them
``ol_intrinsic``. T4/T5, LC, LPLC2 and the giant fiber carry none -- correctly, because
they pool across columns rather than belonging to one. So retinotopy is available exactly
at the input layer, which is the only place it is needed: downstream the connectome
carries the signal itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Columnar types that are good input drives, by pathway.
#: L1/L2/L3 are the lamina monopolar cells every medulla pathway starts from.
PATHWAYS = {
    "ON":  ["L1", "Mi1", "Tm3", "Mi4", "Mi9"],
    "OFF": ["L2", "L3", "Tm1", "Tm2", "Tm4", "Tm9"],
    "lamina": ["L1", "L2", "L3", "L4", "L5"],
}


@dataclass
class Retinotopy:
    """Maps ommatidium index -> the circuit-local neurons in that visual column."""
    omm_of_neuron: np.ndarray     # local neuron index -> ommatidium index, or -1
    neurons_of_omm: list          # ommatidium index -> array of local neuron indices
    types: list                   # the columnar types that were matched
    n_mapped: int
    n_ommatidia: int

    @property
    def coverage(self) -> float:
        return float(np.mean([len(x) > 0 for x in self.neurons_of_omm]))

    def drive(self, luminance: np.ndarray, amplitude: float = 600.0,
              baseline: float = 0.0, invert: bool = True) -> np.ndarray:
        """Per-ommatidium luminance -> per-neuron input current (Q8.8).

        ``invert`` because lamina monopolar cells are OFF-responding: L1/L2 hyperpolarize
        to light and depolarize to dark, so a dark looming disc is what excites them.
        Getting this backwards makes a looming detector into a receding detector.
        """
        lum = np.asarray(luminance, dtype=np.float64)
        sig = (1.0 - lum) if invert else lum
        out = np.full(len(self.omm_of_neuron), baseline, dtype=np.int64)
        m = self.omm_of_neuron >= 0
        out[m] = (sig[self.omm_of_neuron[m]] * amplitude + baseline).astype(np.int64)
        return out

    def drive_series(self, luminance: np.ndarray, **kw) -> np.ndarray:
        """(T, n_ommatidia) -> (T, n_neurons)."""
        return np.stack([self.drive(l, **kw) for l in luminance])

    def pool(self, rates: np.ndarray) -> np.ndarray:
        """Per-neuron activity -> per-ommatidium mean, for plotting on the lattice."""
        out = np.zeros(self.n_ommatidia)
        for i, idx in enumerate(self.neurons_of_omm):
            if len(idx):
                out[i] = float(np.mean(rates[idx]))
        return out


def build(conn, ir, retina, types=None, pathway: str = "OFF") -> Retinotopy:
    """Match circuit neurons to ommatidia by hex column.

    ``retina`` must be a connectome-derived lattice (``Retina.from_connectome``) so its
    ommatidia carry the same hex coordinates the neurons do.
    """
    if retina.hex1 is None:
        raise ValueError("needs a connectome-derived retina: Retina.from_connectome(conn)")
    want = types or PATHWAYS[pathway]
    df = conn.with_columns("assignedOlHex1", "assignedOlHex2")

    by_body = {int(b): i for i, b in enumerate(ir.body_ids)}
    sub = df[df.bodyId.isin(by_body)]
    sub = sub[sub["type"].astype(str).isin(want)]
    sub = sub.dropna(subset=["assignedOlHex1", "assignedOlHex2"])

    omm_key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(retina.hex1, retina.hex2))}
    omm_of = np.full(ir.n_neurons, -1, dtype=np.int32)
    buckets = [[] for _ in range(retina.n)]
    matched = set()
    for body, h1, h2, ty in zip(sub.bodyId, sub.assignedOlHex1, sub.assignedOlHex2,
                                sub["type"]):
        o = omm_key.get((int(h1), int(h2)))
        if o is None:
            continue
        li = by_body[int(body)]
        omm_of[li] = o
        buckets[o].append(li)
        matched.add(str(ty))
    return Retinotopy(omm_of_neuron=omm_of,
                      neurons_of_omm=[np.array(b, dtype=np.int32) for b in buckets],
                      types=sorted(matched), n_mapped=int((omm_of >= 0).sum()),
                      n_ommatidia=retina.n)
