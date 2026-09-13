"""Getting a circuit down to a byte budget, and measuring what that cost.

Two facts about connectome weights drive everything here.

**The weights are already nearly low-precision.** Synapse counts in the raw male-CNS
v1.0 release (151,856,684 edges): 62.0% of edges are a single synapse, 92.2% are <= 3,
99.03% <= 15, max 2,591. Over the retained annotated graph (26,028,386 edges) the
single-synapse share is 40.4%, 94.3% are <= 15 and 98.1% <= 31 -- dropping unannotated
fragments removes weak edges preferentially, so the graph you actually compile wants
about one more bit than the raw distribution suggests. Still: 5 bits covers 98% of it. There is very little precision to throw away --
which means naive weight quantization buys almost nothing, and the real compression has
to come from somewhere else.

**The index dominates the bytes, not the weight.** A sparse layer at int8 weights with
int16 indices spends two thirds of its bytes on addressing. Halving weight precision
from 8 to 4 bits shrinks the artifact by 17%, not 50%. Dropping edges shrinks both the
weight AND its index, so pruning is worth roughly 3x what requantizing is.

So the order of operations is: prune first, then compress the index, then quantize the
weight -- and the budget solver searches in that order.

The error metric is per-neuron **total signed input drive**, sum over incoming edges of
sign(pre) * w. That is the quantity that sets a neuron's operating point; if it survives
compression the circuit still sits where it sat. Per-edge weight error is the wrong
thing to report because errors on single-synapse edges are both the most numerous and
the least consequential.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from typing import Literal, Optional

import numpy as np

from neuraltransistor.ir.graph import CircuitIR

Scheme = Literal["linear", "log", "passthrough"]


@dataclass
class QuantReport:
    """What compression did, measured rather than projected."""
    scheme: str
    weight_bits: int
    index_bits: int
    prune_min_weight: int
    edges_before: int
    edges_after: int
    synapses_before: int
    synapses_after: int
    drive_rel_err_mean: float      # normalised by TOTAL |drive| -- the stable one
    drive_rel_err_p95: float
    drive_rel_err_max: float
    drive_signed_err_mean: float   # normalised by the signed sum -- unstable, kept for comparison
    drive_corr: float              # pearson r between original and compressed drive
    drive_sign_agreement: float
    drive_sign_flips: int
    bytes_before: int
    bytes_after: int

    @property
    def edge_retention(self) -> float:
        return self.edges_after / max(self.edges_before, 1)

    @property
    def synapse_retention(self) -> float:
        return self.synapses_after / max(self.synapses_before, 1)

    @property
    def compression(self) -> float:
        return self.bytes_before / max(self.bytes_after, 1)

    def __str__(self) -> str:
        return (
            f"{self.scheme} w{self.weight_bits}/i{self.index_bits} "
            f"prune>={self.prune_min_weight}: "
            f"{self.bytes_before/1024:.1f}KB -> {self.bytes_after/1024:.1f}KB "
            f"({self.compression:.2f}x) | edges {self.edge_retention:.1%} "
            f"synapses {self.synapse_retention:.1%} | "
            f"drive err mean {self.drive_rel_err_mean:.3%} "
            f"p95 {self.drive_rel_err_p95:.3%} | r={self.drive_corr:.4f} "
            f"sign agree {self.drive_sign_agreement:.3%} | flips {self.drive_sign_flips}"
        )


def input_drive(ir: CircuitIR, weight: Optional[np.ndarray] = None) -> np.ndarray:
    """Total signed synaptic drive arriving at each neuron."""
    w = ir.weight if weight is None else weight
    src = np.repeat(np.arange(ir.n_neurons, dtype=np.int32), np.diff(ir.indptr))
    signed = ir.sign[src].astype(np.float64) * w.astype(np.float64)
    return np.bincount(ir.indices, weights=signed, minlength=ir.n_neurons)


def total_drive(ir: CircuitIR, weight: Optional[np.ndarray] = None) -> np.ndarray:
    """Total UNSIGNED synaptic drive arriving at each neuron.

    The scale of a neuron's input, used as the denominator for drive error. Unlike the
    signed sum this cannot collapse toward zero for a balanced neuron, so it does not
    manufacture enormous relative errors out of small absolute ones.
    """
    w = ir.weight if weight is None else weight
    return np.bincount(ir.indices, weights=np.abs(w.astype(np.float64)),
                       minlength=ir.n_neurons)


def drive_error(d0: np.ndarray, d1: np.ndarray, scale: np.ndarray) -> dict:
    """Compare two drive vectors on the stable normalisation, plus shape metrics."""
    ok = scale > 0
    rel = np.zeros_like(d0)
    rel[ok] = np.abs(d1[ok] - d0[ok]) / scale[ok]
    nzs = np.abs(d0) > 1e-9
    signed_rel = np.abs(d1[nzs] - d0[nzs]) / np.abs(d0[nzs]) if nzs.any() else np.zeros(1)
    corr = float(np.corrcoef(d0, d1)[0, 1]) if d0.std() > 0 and d1.std() > 0 else 1.0
    agree = float((np.sign(d0[nzs]) == np.sign(d1[nzs])).mean()) if nzs.any() else 1.0
    return {
        "mean": float(rel[ok].mean()) if ok.any() else 0.0,
        "p95": float(np.percentile(rel[ok], 95)) if ok.any() else 0.0,
        "max": float(rel[ok].max()) if ok.any() else 0.0,
        "signed_mean": float(signed_rel.mean()),
        "corr": corr,
        "sign_agreement": agree,
        "sign_flips": int(((np.sign(d0) * np.sign(d1)) < 0).sum()),
    }


def prune(ir: CircuitIR, min_weight: int) -> CircuitIR:
    """Drop every edge below ``min_weight`` synapses.

    This is the highest-leverage operation available: it removes the weight AND its
    index, and the connectome's weight distribution means a threshold of 2 already
    removes ~62% of edges while keeping ~70% of synaptic mass.
    """
    if min_weight <= 1:
        return ir
    keep = ir.weight >= min_weight
    src = np.repeat(np.arange(ir.n_neurons, dtype=np.int32), np.diff(ir.indptr))
    out = copy.copy(ir)
    out.indices = ir.indices[keep].astype(np.int32)
    out.weight = ir.weight[keep].astype(np.uint16)
    counts = np.bincount(src[keep], minlength=ir.n_neurons)
    out.indptr = np.zeros(ir.n_neurons + 1, dtype=np.int64)
    np.cumsum(counts, out=out.indptr[1:])
    out.provenance = dict(ir.provenance)
    out.provenance["pruned_min_weight"] = int(min_weight)
    return out


def prune_by_reliability(ir: CircuitIR, p_real: float, model=None) -> tuple[CircuitIR, dict]:
    """Prune to edges at least ``p_real`` likely to be a real pathway.

    The honest version of a weight threshold. Instead of "keep >= 5 synapses" because
    that is what the last paper did, this asks for a confidence and lets the measured
    bilateral-reproducibility curve decide the synapse count that delivers it. See
    neuraltransistor.circuit.noise for how the curve is measured.
    """
    from neuraltransistor.circuit import noise as _noise
    model = model or _noise.NoiseModel.load()
    if model is None:
        raise RuntimeError(
            "no noise model cached; run neuraltransistor.circuit.noise.measure(conn) once")
    w = model.min_weight_for(p_real)
    out = prune(ir, w)
    out.provenance = dict(out.provenance)
    out.provenance.update(prune_p_real=p_real, prune_min_weight_from_model=int(w))
    return out, {"p_real": p_real, "min_weight": int(w),
                 "edges_before": ir.n_edges, "edges_after": out.n_edges,
                 "synapses_before": ir.n_synapses, "synapses_after": out.n_synapses,
                 "edge_retention": out.n_edges / max(ir.n_edges, 1),
                 "synapse_retention": out.n_synapses / max(ir.n_synapses, 1)}


def quantize_weights(w: np.ndarray, bits: int, scheme: Scheme = "log"
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Quantize synapse counts to ``bits``, returning (codes, codebook).

    ``log`` places levels geometrically. Synapse counts are heavy-tailed -- median 1,
    p99 15, max 2,591 -- so linear levels spend almost all their resolution on a range
    that holds almost no edges, and crush the strong connections that carry the
    circuit. Log levels keep relative error roughly constant across four orders of
    magnitude, which is what a multiplicative quantity wants.
    """
    n_levels = 1 << bits
    w = w.astype(np.float64)
    if scheme == "passthrough" or bits >= 16:
        return w.astype(np.uint16), np.arange(int(w.max()) + 1, dtype=np.float32)

    lo, hi = max(float(w.min()), 1.0), float(w.max())
    if scheme == "linear":
        book = np.linspace(lo, hi, n_levels)
    elif scheme == "log":
        book = np.unique(np.round(np.geomspace(lo, max(hi, lo + 1), n_levels)))
        if len(book) < n_levels:  # pad so the codebook is a fixed size on device
            book = np.concatenate([book, np.full(n_levels - len(book), book[-1])])
    else:
        raise ValueError(scheme)

    codes = np.abs(w[:, None] - book[None, :]).argmin(axis=1) if len(w) < 200_000 \
        else _argmin_chunked(w, book)
    return codes.astype(np.uint8 if bits <= 8 else np.uint16), book.astype(np.float32)


def _argmin_chunked(w: np.ndarray, book: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    out = np.empty(len(w), dtype=np.int64)
    for i in range(0, len(w), chunk):
        s = w[i:i + chunk]
        out[i:i + chunk] = np.abs(s[:, None] - book[None, :]).argmin(axis=1)
    return out


def index_bits_needed(ir: CircuitIR, delta: bool = True) -> int:
    """How many bits the CSR index actually needs.

    Indices within a row are sorted, so consecutive deltas are small. If every delta
    fits in 8 bits the index halves -- which on a sparse connectome layer is a bigger
    win than any weight quantization.
    """
    n = ir.n_neurons
    plain = 8 if n <= 256 else (16 if n <= 65536 else 32)
    if not delta or ir.n_edges == 0:
        return plain
    d = np.diff(ir.indices)
    row_start = np.zeros(ir.n_edges, dtype=bool)
    row_start[ir.indptr[:-1][np.diff(ir.indptr) > 0]] = True
    d = d[~row_start[1:]]
    if len(d) == 0:
        return plain
    mx = int(d.max()) if len(d) else 0
    return min(plain, 8 if mx < 256 else (16 if mx < 65536 else 32))


def compress(
    ir: CircuitIR,
    weight_bits: int = 8,
    scheme: Scheme = "log",
    prune_min_weight: int = 1,
    delta_index: bool = True,
) -> tuple[CircuitIR, QuantReport]:
    """Prune + quantize, and measure the damage."""
    before_bytes = ir.footprint(weight_bits=16, index_bits=16)["total_B"]
    d0 = input_drive(ir)

    out = prune(ir, prune_min_weight)
    codes, book = quantize_weights(out.weight, weight_bits, scheme)
    deq = book[codes] if scheme != "passthrough" else out.weight.astype(np.float32)

    d1 = input_drive(out, weight=deq)
    err = drive_error(d0, d1, total_drive(ir))

    ibits = index_bits_needed(out, delta=delta_index)
    after_bytes = out.footprint(weight_bits=weight_bits, index_bits=ibits)["total_B"]

    out.weight = np.round(deq).clip(0, 65535).astype(np.uint16)
    out.provenance = dict(out.provenance)
    out.provenance.update(quant_scheme=scheme, quant_weight_bits=weight_bits,
                          quant_index_bits=ibits)

    rep = QuantReport(
        scheme=scheme, weight_bits=weight_bits, index_bits=ibits,
        prune_min_weight=prune_min_weight,
        edges_before=ir.n_edges, edges_after=out.n_edges,
        synapses_before=ir.n_synapses, synapses_after=int(deq.sum()),
        drive_rel_err_mean=err["mean"], drive_rel_err_p95=err["p95"],
        drive_rel_err_max=err["max"], drive_signed_err_mean=err["signed_mean"],
        drive_corr=err["corr"], drive_sign_agreement=err["sign_agreement"],
        drive_sign_flips=err["sign_flips"],
        bytes_before=before_bytes, bytes_after=after_bytes,
    )
    return out, rep


def fit_budget(
    ir: CircuitIR,
    budget_kb: float,
    max_drive_err: float = 0.10,
    weight_bit_options=(8, 4, 2),
    prune_options=(1, 2, 3, 5, 8, 12, 20, 35, 60, 100),
    scheme: Scheme = "log",
) -> tuple[Optional[CircuitIR], Optional[QuantReport], list[QuantReport]]:
    """Find the least-damaging setting that fits ``budget_kb``.

    Search order follows the economics: prune first (shrinks index + weight), then drop
    weight precision (shrinks weight only). Among settings that fit the budget AND stay
    under ``max_drive_err``, the one with the lowest measured drive error wins. Returns
    ``(None, None, trials)`` when nothing fits, rather than silently returning the least
    bad option -- an over-budget artifact must not look like a success.
    """
    budget_b = budget_kb * 1024
    trials: list[QuantReport] = []
    best = None
    for pmw in prune_options:
        for wb in weight_bit_options:
            cir, rep = compress(ir, weight_bits=wb, scheme=scheme, prune_min_weight=pmw)
            trials.append(rep)
            if rep.bytes_after <= budget_b and rep.drive_rel_err_mean <= max_drive_err:
                if best is None or rep.drive_rel_err_mean < best[1].drive_rel_err_mean:
                    best = (cir, rep)
    if best is None:
        return None, None, trials
    return best[0], best[1], trials
