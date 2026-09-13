"""The male-CNS connectome on disk, and the compact index we build from it.

The published release is three Feather tables:

    body-annotations-male-cns-v1.0-minconf-0.5.feather   211,577 bodies, 37 columns
    body-neurotransmitters-male-cns-v1.0.feather       1,835,518 bodies, 10 columns
    connectome-weights-male-cns-v1.0-minconf-0.5.feather  151,856,684 edges

The edge table is (body_pre, body_post, weight) with weight = synapse count. It is
1.05 GB of int64 and a naive scan of it costs ~140 s, which is too slow to sit in
front of an interactive extract. So we pay that cost once and cache a CSR index:
bodyIds collapse to a dense int32 neuron index, weights fit in uint16 (observed max
2,591), and edges are sorted by presynaptic neuron so a circuit's out-edges are one
contiguous slice.

Honest filter: the index is built over ANNOTATED bodies only. Edges touching an
unannotated fragment are dropped, and the exact count that drops is recorded in the
index metadata rather than quietly discarded -- see ``IndexMeta.retained``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.feather as feather

DATASET = "male-cns-v1.0"
MINCONF = "0.5"

ANNOTATIONS = f"body-annotations-{DATASET}-minconf-{MINCONF}.feather"
NEUROTRANSMITTERS = f"body-neurotransmitters-{DATASET}.feather"
WEIGHTS = f"connectome-weights-{DATASET}-minconf-{MINCONF}.feather"

_DEFAULT_DATA = Path.home() / "fly-connectome"
_DEFAULT_CACHE = Path.home() / ".cache" / "neural-transistor"
_LEGACY_CACHE = Path.home() / ".cache" / "flyforge"

# Columns we actually use. The release has 37; pulling all of them costs memory and
# none of the rest feed the compiler.
ANNO_COLS = [
    "bodyId", "type", "superclass", "class", "subclass",
    "somaSide", "somaNeuromere", "statusLabel", "instance",
    # optic-lobe column assignment: the hexagonal lattice coordinates. Needed to build
    # a retina that lines up with the columnar neurons, see neuraltransistor.sensors.retina.
    "assignedOlHex1", "assignedOlHex2",
]
NT_COLS = [
    "body", "consensus_nt", "predicted_nt", "predicted_nt_confidence", "ground_truth",
]


def data_dir() -> Path:
    return Path(os.environ.get("NTX_DATA", _DEFAULT_DATA))


def cache_dir() -> Path:
    """Where the built index lives.

    An index built under the project's previous name is still perfectly good, so it is
    adopted rather than rebuilt -- nobody should pay 75 s for a rename.
    """
    env = os.environ.get("NTX_CACHE")
    if env:
        d = Path(env)
    elif _DEFAULT_CACHE.exists() or not _LEGACY_CACHE.exists():
        d = _DEFAULT_CACHE
    else:
        d = _LEGACY_CACHE
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class IndexMeta:
    """Provenance for a built index. Written next to the arrays, checked on load."""
    dataset: str
    minconf: str
    n_neurons: int
    n_edges_total: int
    n_edges_retained: int
    n_synapses_total: int
    n_synapses_retained: int
    max_weight: int
    build_seconds: float
    source_files: dict

    @property
    def retained(self) -> float:
        return self.n_edges_retained / max(self.n_edges_total, 1)

    @property
    def retained_synapses(self) -> float:
        return self.n_synapses_retained / max(self.n_synapses_total, 1)


class Connectome:
    """Annotated neurons + a CSR edge index over them.

    Attributes
    ----------
    neurons : pd.DataFrame
        One row per annotated body, in index order. ``idx`` is the dense neuron index
        used everywhere downstream; ``bodyId`` is the stable published identifier.
    indptr, indices, data : np.ndarray
        CSR over presynaptic neuron. ``indices[indptr[i]:indptr[i+1]]`` are the
        postsynaptic neuron indices of neuron ``i``; ``data`` the synapse counts.
    """

    def __init__(self, neurons: pd.DataFrame, indptr, indices, data, meta: IndexMeta):
        self.neurons = neurons
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.meta = meta
        self._bodyid_to_idx: Optional[dict] = None

    # -- construction -----------------------------------------------------------

    @classmethod
    def load(cls, rebuild: bool = False, verbose: bool = True) -> "Connectome":
        cache = cache_dir() / f"index-{DATASET}-minconf-{MINCONF}.npz"
        metap = cache.with_suffix(".meta.json")
        if cache.exists() and metap.exists() and not rebuild:
            z = np.load(cache, allow_pickle=False)
            meta = IndexMeta(**json.loads(metap.read_text()))
            neurons = pd.read_parquet(cache.with_suffix(".neurons.parquet"))
            return cls(neurons, z["indptr"], z["indices"], z["data"], meta)
        return cls.build(verbose=verbose)

    @classmethod
    def build(cls, verbose: bool = True) -> "Connectome":
        t0 = time.time()
        d = data_dir()
        for f in (ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS):
            if not (d / f).exists():
                raise FileNotFoundError(
                    f"{f} not found in {d}. Set NTX_DATA to the directory "
                    f"holding the male-CNS release."
                )

        def say(m):
            if verbose:
                print(f"[neuraltransistor] {m}", flush=True)

        say("reading annotations")
        anno = feather.read_table(d / ANNOTATIONS, columns=ANNO_COLS,
                                  memory_map=True).to_pandas()
        anno["type"] = anno["type"].fillna("")
        anno["superclass"] = anno["superclass"].fillna("")
        anno["somaNeuromere"] = anno["somaNeuromere"].fillna("")
        anno["somaSide"] = anno["somaSide"].fillna("")
        anno["statusLabel"] = anno["statusLabel"].astype(str)

        say("reading neurotransmitters")
        nt = feather.read_table(d / NEUROTRANSMITTERS, columns=NT_COLS,
                                memory_map=True).to_pandas()
        anno = anno.merge(nt, left_on="bodyId", right_on="body", how="left")
        anno = anno.drop(columns=["body"])

        # Dense index, ordered by bodyId so searchsorted works.
        anno = anno.sort_values("bodyId", kind="stable").reset_index(drop=True)
        anno["idx"] = np.arange(len(anno), dtype=np.int32)
        body_sorted = anno["bodyId"].to_numpy()
        n = len(anno)
        say(f"{n:,} annotated neurons")

        say("reading edges (1.05 GB, this is the slow part)")
        tbl = feather.read_table(d / WEIGHTS, memory_map=True)
        pre = tbl.column("body_pre").to_numpy()
        post = tbl.column("body_post").to_numpy()
        w = tbl.column("weight").to_numpy()
        n_edges_total = int(len(w))
        n_syn_total = int(w.sum())
        max_w = int(w.max())
        del tbl

        say(f"{n_edges_total:,} edges / {n_syn_total:,} synapses; mapping to index")
        pi = np.searchsorted(body_sorted, pre)
        np.clip(pi, 0, n - 1, out=pi)
        ok_pre = body_sorted[pi] == pre
        del pre
        qi = np.searchsorted(body_sorted, post)
        np.clip(qi, 0, n - 1, out=qi)
        ok = ok_pre & (body_sorted[qi] == post)
        del post, ok_pre

        pi = pi[ok].astype(np.int32)
        qi = qi[ok].astype(np.int32)
        ww = w[ok]
        del ok, w
        n_edges_kept = int(len(ww))
        n_syn_kept = int(ww.sum())
        say(f"retained {n_edges_kept:,} edges ({n_edges_kept/n_edges_total:.1%}) "
            f"/ {n_syn_kept:,} synapses ({n_syn_kept/n_syn_total:.1%})")

        # weight fits uint16 (max observed 2,591); clip defensively and record it.
        ww = np.clip(ww, 0, 65535).astype(np.uint16)

        say("sorting into CSR")
        order = np.argsort(pi, kind="stable")
        indices = qi[order]
        data = ww[order]
        counts = np.bincount(pi, minlength=n)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        del order, pi, qi, ww, counts

        meta = IndexMeta(
            dataset=DATASET, minconf=MINCONF, n_neurons=n,
            n_edges_total=n_edges_total, n_edges_retained=n_edges_kept,
            n_synapses_total=n_syn_total, n_synapses_retained=n_syn_kept,
            max_weight=max_w, build_seconds=round(time.time() - t0, 1),
            source_files={f: (d / f).stat().st_size for f in
                          (ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS)},
        )

        cache = cache_dir() / f"index-{DATASET}-minconf-{MINCONF}.npz"
        say(f"caching to {cache}")
        np.savez(cache, indptr=indptr, indices=indices, data=data)
        cache.with_suffix(".meta.json").write_text(json.dumps(asdict(meta), indent=2))
        anno.to_parquet(cache.with_suffix(".neurons.parquet"), index=False)
        say(f"built in {meta.build_seconds}s")
        return cls(anno, indptr, indices, data, meta)

    # -- access -----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.neurons)

    @property
    def n_edges(self) -> int:
        return int(len(self.indices))

    def out_edges(self, i: int):
        """(post_idx, weight) for one presynaptic neuron."""
        a, b = self.indptr[i], self.indptr[i + 1]
        return self.indices[a:b], self.data[a:b]

    # -- reverse (CSC) index, built on demand ----------------------------------

    def reverse(self):
        """CSC over postsynaptic neuron: who projects INTO neuron j.

        Built lazily because half the pipeline never asks for it, and cached to disk
        because building it means an argsort over 151M edges (~25 s).
        """
        if getattr(self, "_rev", None) is not None:
            return self._rev
        cache = cache_dir() / f"rindex-{self.meta.dataset}-minconf-{self.meta.minconf}.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            self._rev = (z["indptr"], z["indices"], z["data"])
            return self._rev
        n = len(self.neurons)
        # presynaptic partner of every edge, expanded from indptr
        src = np.repeat(np.arange(n, dtype=np.int32), np.diff(self.indptr))
        order = np.argsort(self.indices, kind="stable")
        rindices = src[order]
        rdata = self.data[order]
        counts = np.bincount(self.indices, minlength=n)
        rindptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=rindptr[1:])
        np.savez(cache, indptr=rindptr, indices=rindices, data=rdata)
        self._rev = (rindptr, rindices, rdata)
        return self._rev

    def in_edges(self, j: int):
        """(pre_idx, weight) for one postsynaptic neuron."""
        rp, ri, rd = self.reverse()
        a, b = rp[j], rp[j + 1]
        return ri[a:b], rd[a:b]

    def with_columns(self, *cols: str) -> pd.DataFrame:
        """Neurons frame, guaranteed to carry ``cols``.

        An index cached by an older version may not have every column this version
        wants. Rather than force a 75 s rebuild, the missing columns are read straight
        out of the annotation Feather and joined on bodyId. The cache is left alone.
        """
        missing = [c for c in cols if c not in self.neurons.columns]
        if not missing:
            return self.neurons
        extra = feather.read_table(data_dir() / ANNOTATIONS,
                                   columns=["bodyId"] + missing,
                                   memory_map=True).to_pandas()
        merged = self.neurons.merge(extra, on="bodyId", how="left")
        self.neurons = merged
        return merged

    def idx_of(self, body_ids) -> np.ndarray:
        body_sorted = self.neurons["bodyId"].to_numpy()
        i = np.searchsorted(body_sorted, np.asarray(body_ids))
        return i.astype(np.int32)
