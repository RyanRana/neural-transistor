"""Reference implementation of the circuit tick, in numpy.

This exists to be compared against, not to be fast. Every operation mirrors the emitted
C exactly -- same Q8.8 fixed point, same saturation points, same refractory handling,
same order -- so that any divergence between this and the device is a real bug in the
compiler rather than a difference of opinion about rounding.

If you change the arithmetic here, change ``flyforge/target/mcu_int8.py`` in the same
commit, and let the equivalence eval decide whether you got it right.
"""

from __future__ import annotations

import numpy as np

from flyforge.ir.graph import CircuitIR
from flyforge.quant.quantize import quantize_weights

REFRACTORY_TICKS = 4
GAIN_UNITY = 128


class Reference:
    """Integer-exact reference runtime for a CircuitIR."""

    def __init__(self, ir: CircuitIR, weight_bits: int = 8, scheme: str = "log"):
        self.ir = ir
        self.n = ir.n_neurons
        codes, book = quantize_weights(ir.weight, weight_bits, scheme)
        self.codes = codes.astype(np.int32)
        self.book = np.round(book).astype(np.int32)
        self.sign = ir.sign.astype(np.int32)
        dyn = ir.dynamics
        self.decay_q8 = np.round(np.exp(-dyn.dt / np.maximum(dyn.tau_m, 1e-6)) * 256
                                 ).astype(np.int32)
        self.thresh = np.round(dyn.threshold * 256).astype(np.int32)
        self.has_mod = ir.n_mod_edges > 0
        self.reset()

    def reset(self):
        self.v = np.zeros(self.n, dtype=np.int32)
        self.fired = np.zeros(self.n, dtype=np.uint8)
        self.refrac = np.zeros(self.n, dtype=np.int32)
        self.gain = np.full(self.n, GAIN_UNITY, dtype=np.int32)

    def tick(self, input_q88: np.ndarray | None = None) -> np.ndarray:
        ir = self.ir
        acc = np.zeros(self.n, dtype=np.int64)

        # event-driven propagation, sign hoisted per row (Dale's law)
        firing = np.flatnonzero(self.fired)
        for i in firing:
            a, b = ir.indptr[i], ir.indptr[i + 1]
            if a == b:
                continue
            np.add.at(acc, ir.indices[a:b],
                      self.sign[i] * self.book[self.codes[a:b]])

        if self.has_mod:
            m = self.fired[ir.mod_pre].astype(bool)
            if m.any():
                np.add.at(self.gain, ir.mod_post[m],
                          ir.mod_weight[m].astype(np.int32))
                np.clip(self.gain, 0, 255, out=self.gain)

        acc = acc.astype(np.int64)
        if input_q88 is not None:
            acc += input_q88.astype(np.int64)
        if self.has_mod:
            acc = (acc * self.gain.astype(np.int64)) >> 8

        in_ref = self.refrac > 0
        v = (self.v.astype(np.int64) * self.decay_q8) >> 8
        v = v + acc
        np.clip(v, -32768, 32767, out=v)

        fired = (~in_ref) & (v >= self.thresh)
        self.v = np.where(in_ref, 0, np.where(fired, 0, v)).astype(np.int32)
        self.refrac = np.where(in_ref, self.refrac - 1,
                               np.where(fired, REFRACTORY_TICKS, 0)).astype(np.int32)
        self.fired = np.where(in_ref, 0, fired).astype(np.uint8)

        if self.has_mod:
            self.gain = (self.gain + ((GAIN_UNITY - self.gain) >> 5)).astype(np.int32)
        return self.fired

    def run(self, n_ticks: int, drive: np.ndarray | None = None,
            record: bool = False):
        """Run ``n_ticks``; ``drive`` is (n_ticks, n) or (n,) Q8.8 input."""
        out = np.zeros((n_ticks, self.n), dtype=np.uint8) if record else None
        rates = np.zeros(self.n, dtype=np.int64)
        for t in range(n_ticks):
            d = None
            if drive is not None:
                d = drive[t] if drive.ndim == 2 else drive
            f = self.tick(d)
            rates += f
            if record:
                out[t] = f
        return (out if record else None), rates / max(n_ticks, 1)
