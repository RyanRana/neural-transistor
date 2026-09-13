"""How much of the connectome is real, measured without ground truth.

Connectomics prunes weak edges by convention: keep connections of >= 5 synapses, or
>= 10, depending on the paper. The threshold is defensible but arbitrary, and nobody
reports what it costs, because measuring that seems to need ground truth nobody has.

It doesn't. The fly is bilaterally symmetric, and the two hemispheres were
reconstructed independently from the same volume. That makes them a **replicate pair**:
a pathway that is real should appear on both sides, and one that is a reconstruction
artifact has no reason to. Reproducibility across hemispheres is therefore a direct,
label-free measurement of whether a connection is real.

The design is unusually clean in male-CNS v1.0:

    left neurons                75,215
    right neurons               75,119
    types present on both sides 10,940 of 11,230
    median per-type |L-R| cell count difference     0

Measuring it (19,654,874 ipsilateral typed edges):

    weight   edges        reproduced on the other side
    1        7,994,902    83.9%
    2        3,718,546    90.2%
    3        2,042,085    93.3%
    4-5      2,166,674    95.5%
    6-7      1,117,581    97.0%
    8-11     1,097,106    97.9%
    12-19      802,046    98.6%
    20-34      434,915    99.0%
    >=35       281,019    99.1%   <- plateau: the ceiling for a real connection
    shuffled control                28.0%   <- the floor for a fabricated one

Reproducibility saturates at 99.1%, not 100%: some genuine connections are unilateral,
and annotation is imperfect. So an observed rate is a mixture of real connections
reproducing at the ceiling and noise connections reproducing at chance:

    observed(w) = P(real|w) * ceiling + (1 - P(real|w)) * chance

which inverts to a per-edge posterior:

    P(real|w) = (observed(w) - chance) / (ceiling - chance)

Single-synapse edges come out at **P(real) = 0.786** -- about one in five is spurious --
and they are 41% of the graph. By weight 12 the posterior passes 0.99.

**What this does and does not establish.** Reproducibility is measured at the level of a
*type pair*: it asks whether the pathway A->B exists on the other side, not whether this
particular cell-to-cell edge does. So P(real|w) is the probability the pathway is real,
which is an upper bound on the probability the individual edge is. It is the right
quantity for deciding what to compile, because a controller is built from pathways. It
would be the wrong quantity for a claim about an individual synapse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from neuraltransistor.data.source import Connectome, cache_dir

#: Bin edges used for the published curve. Fine where the mass is.
DEFAULT_BINS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 7), (8, 11), (12, 19),
                (20, 34), (35, 59), (60, 99), (100, 1 << 30)]


@dataclass
class NoiseModel:
    """A measured reproducibility curve plus the mixture that inverts it."""
    bins: list                 # [(lo, hi, n_edges, observed_repro)]
    chance: float
    ceiling: float
    n_edges: int
    n_type_pairs: int

    def p_real(self, w) -> np.ndarray:
        """Posterior that an edge of weight ``w`` belongs to a real pathway."""
        w = np.asarray(w, dtype=np.int64)
        out = np.empty(w.shape, dtype=np.float64)
        out.fill(np.nan)
        denom = max(self.ceiling - self.chance, 1e-9)
        for lo, hi, _n, obs in self.bins:
            m = (w >= lo) & (w <= hi)
            if m.any():
                out[m] = np.clip((obs - self.chance) / denom, 0.0, 1.0)
        # anything above the last bin inherits its value
        hi_last = self.bins[-1][1]
        out[w > hi_last] = np.clip((self.bins[-1][3] - self.chance) / denom, 0.0, 1.0)
        return out

    def min_weight_for(self, p: float) -> int:
        """Smallest synapse count whose posterior reaches ``p``.

        This is the principled replacement for a hand-picked threshold: instead of
        "keep >= 5 synapses", say "keep edges that are 95% likely to be real" and let
        the measurement decide what that means.
        """
        denom = max(self.ceiling - self.chance, 1e-9)
        for lo, hi, _n, obs in self.bins:
            if (obs - self.chance) / denom >= p:
                return int(lo)
        return int(self.bins[-1][1])

    def cost_of(self, p: float) -> dict:
        """What thresholding at posterior ``p`` removes."""
        w = self.min_weight_for(p)
        dropped = sum(n for lo, hi, n, _ in self.bins if hi < w)
        kept = sum(n for lo, hi, n, _ in self.bins if hi >= w)
        return {"p": p, "min_weight": w, "edges_dropped": dropped,
                "edges_kept": kept,
                "frac_dropped": dropped / max(dropped + kept, 1)}

    def save(self, path=None) -> Path:
        path = Path(path or (cache_dir() / "noise-model.json"))
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path=None) -> Optional["NoiseModel"]:
        path = Path(path or (cache_dir() / "noise-model.json"))
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        d["bins"] = [tuple(b) for b in d["bins"]]
        return cls(**d)

    def table(self) -> str:
        lines = [f"{'weight':>8} {'edges':>12} {'reproduced':>11} {'P(real)':>9}"]
        for lo, hi, n, obs in self.bins:
            lab = f"{lo}" if lo == hi else (f"{lo}+" if hi >= (1 << 30) else f"{lo}-{hi}")
            p = float(self.p_real(np.array([lo]))[0])
            lines.append(f"{lab:>8} {n:>12,} {obs:>11.3f} {p:>9.3f}")
        lines.append(f"chance {self.chance:.3f} / ceiling {self.ceiling:.3f} "
                     f"over {self.n_edges:,} ipsilateral typed edges")
        return "\n".join(lines)


def measure(conn: Connectome, bins=None, seed: int = 0,
            verbose: bool = True) -> NoiseModel:
    """Measure bilateral reproducibility across the whole connectome."""
    bins = bins or DEFAULT_BINS
    df = conn.neurons
    n = len(df)
    typ = df["type"].astype(str).to_numpy()
    side = df["somaSide"].astype(str).to_numpy()
    tcodes, tuniq = pd.factorize(pd.Series(typ))
    tcodes = tcodes.astype(np.int32)
    T = len(tuniq)
    sid = np.where(side == "L", 0, np.where(side == "R", 1, 2)).astype(np.int8)
    valid = (typ != "") & (sid < 2)

    src = np.repeat(np.arange(n, dtype=np.int32), np.diff(conn.indptr))
    dst = conn.indices
    w = conn.data.astype(np.int64)
    m = valid[src] & valid[dst] & (sid[src] == sid[dst])
    src, dst, w = src[m], dst[m], w[m]
    s = sid[src]
    a = tcodes[src]
    b = tcodes[dst]
    if verbose:
        print(f"[noise] {len(w):,} ipsilateral typed edges, {int(w.sum()):,} synapses")

    key = a.astype(np.int64) * T + b.astype(np.int64)
    present_L = set(np.unique(key[s == 0]).tolist())
    present_R = set(np.unique(key[s == 1]).tolist())

    def repro(keys, sides):
        return np.fromiter(
            ((k in present_R) if ss == 0 else (k in present_L)
             for k, ss in zip(keys.tolist(), sides.tolist())),
            dtype=bool, count=len(keys))

    other = repro(key, s)

    # chance floor: shuffle the postsynaptic type within hemisphere, which preserves
    # the marginal type distribution but destroys the specific pathway
    rng = np.random.default_rng(seed)
    bsh = b.copy()
    for ss in (0, 1):
        i = np.flatnonzero(s == ss)
        bsh[i] = rng.permutation(b[i])
    chance = float(repro(a.astype(np.int64) * T + bsh.astype(np.int64), s).mean())

    rows = []
    for lo, hi in bins:
        sel = (w >= lo) & (w <= hi)
        if not sel.any():
            continue
        rows.append((int(lo), int(hi), int(sel.sum()), float(other[sel].mean())))

    ceiling = max(r[3] for r in rows)
    model = NoiseModel(bins=rows, chance=chance, ceiling=ceiling,
                       n_edges=int(len(w)),
                       n_type_pairs=int(len(present_L | present_R)))
    if verbose:
        print(model.table())
    return model


def get(conn: Connectome, rebuild: bool = False, verbose: bool = False) -> NoiseModel:
    """Cached accessor -- the measurement takes ~60 s, the answer never changes."""
    if not rebuild:
        m = NoiseModel.load()
        if m is not None:
            return m
    m = measure(conn, verbose=verbose)
    m.save()
    return m
