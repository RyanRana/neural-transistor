"""Fitting the dynamics the connectome does not contain.

The connectome fixes structure and sign. Time constants, thresholds, gains and the
synapse-count-to-strength scale are free, and our own measurement showed a single global
scale cannot produce a working ring attractor at any value (see evals/functional.py).
This package closes that gap.

Two tasks ship:

``RingAttractorTask``  self-supervised, no recorded activity required. Loss terms come
                      from what a heading system must *do*: form one localized bump,
                      hold it when input stops, not drift, and stay off both the silent
                      and saturated rails. Follows Duan et al. 2025, who showed these
                      terms alone recover bump width and linear velocity integration.

``LoomingTask``        supervised against the published giant-fiber model (von Reyn 2017 /
                      Ache 2019), which is itself fitted to real recordings. Distilling a
                      validated model into connectome-constrained dynamics.

Fitting happens in float on the torch backend, and the result is written back into
``CircuitIR.dynamics`` so it can be quantized and emitted like anything else. That order
matters: Biswas 2026 shows synapse counts tolerate +-90% variation if you may re-solve
the scales, while Chang 2023 shows 2% noise on *fixed* weights breaks the circuit. So
quantization must be followed by a re-fit, not applied to a frozen solution.
"""

from neuraltransistor.train.fit import FitResult, fit, refit_after_quantization
from neuraltransistor.train.tasks import LoomingTask, RingAttractorTask

__all__ = ["fit", "refit_after_quantization", "FitResult",
           "RingAttractorTask", "LoomingTask"]
