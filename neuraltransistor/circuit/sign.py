"""Turning a neurotransmitter prediction into a synaptic sign.

The connectome ships a predicted transmitter per body. Downstream everyone wants a
sign, so everyone writes the same three-line dict:

    acetylcholine -> +1,  glutamate -> -1,  GABA -> -1

That dict is wrong in a specific, load-bearing place, and it fails silently.

**Glutamate is inhibitory in the fly CNS but excitatory at the neuromuscular
junction.** Drosophila motor neurons are glutamatergic; glutamate onto muscle opens a
cation channel and contracts it. Apply the naive map to the 708 VNC motor neurons and
you invert the entire motor output layer -- the controller drives every actuator
backwards and nothing in the pipeline complains, because the graph is still perfectly
well-formed. Measured on male-CNS v1.0: 42.9% of motor neurons come back "inhibitory"
under the naive map.

So sign is assigned by (transmitter, target class), not transmitter alone, and the
efferent exception is applied explicitly and counted.

The second trap is coverage. Sign is not uniformly known:

    circuit            signed    ground truth
    central complex    100.0%    71.8%
    optic motion        99.0%    70.4%
    leg intrinsic       99.2%    83.0%
    descending          97.1%     9.8%
    VNC motor           44.9%     0.0%

A neuron with no transmitter call is not silently dropped and not silently assumed
excitatory. It is marked UNKNOWN and its sign becomes a free parameter handed to the
fitting stage, where it is one more thing to be determined by behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

EXCITATORY = ("acetylcholine",)
CNS_INHIBITORY = ("glutamate", "gaba", "histamine")
MODULATORY = ("dopamine", "octopamine", "serotonin")

#: Superclasses whose output leaves the CNS for muscle or gland. Glutamate flips sign
#: here. ``vnc_efferent``/``cb_efferent`` include neurosecretory cells; they are treated
#: as efferent for sign purposes and flagged ``modulatory_target``.
EFFERENT_SUPERCLASSES = frozenset({
    "vnc_motor", "cb_motor", "vnc_efferent", "cb_efferent",
})

UNKNOWN = 0


@dataclass
class SignReport:
    """What sign assignment actually achieved, so a caller can print it honestly."""
    n: int
    n_excitatory: int
    n_inhibitory: int
    n_modulatory: int
    n_unknown: int
    n_nmj_corrected: int
    n_low_confidence: int
    confidence_threshold: float

    @property
    def coverage(self) -> float:
        return 1.0 - self.n_unknown / max(self.n, 1)

    def __str__(self) -> str:
        return (
            f"sign: {self.coverage:.1%} covered on {self.n:,} neurons "
            f"(+{self.n_excitatory:,} / -{self.n_inhibitory:,} / "
            f"mod {self.n_modulatory:,} / unknown {self.n_unknown:,}); "
            f"{self.n_nmj_corrected:,} NMJ sign flips applied; "
            f"{self.n_low_confidence:,} below conf {self.confidence_threshold}"
        )


def assign_signs(
    neurons: pd.DataFrame,
    confidence: float = 0.5,
    trust_ground_truth: bool = True,
) -> tuple[np.ndarray, np.ndarray, SignReport]:
    """Assign a presynaptic sign to every neuron.

    Parameters
    ----------
    neurons
        Frame with ``consensus_nt``, ``predicted_nt``, ``predicted_nt_confidence``,
        ``ground_truth`` and ``superclass``.
    confidence
        Predictions below this confidence are demoted to UNKNOWN unless a ground-truth
        call exists for that body.
    trust_ground_truth
        Ground truth overrides the prediction where present (85,484 bodies have one).

    Returns
    -------
    sign : int8 array in {-1, 0, +1}; 0 means UNKNOWN or modulatory (see ``kind``)
    kind : uint8 array, 0=unknown 1=excitatory 2=inhibitory 3=modulatory
    report : SignReport
    """
    n = len(neurons)
    nt = neurons.get("consensus_nt", pd.Series([None] * n)).astype("string").str.lower()
    gt = neurons.get("ground_truth", pd.Series([None] * n)).astype("string").str.lower()
    pred = neurons.get("predicted_nt", pd.Series([None] * n)).astype("string").str.lower()
    conf = pd.to_numeric(
        neurons.get("predicted_nt_confidence", pd.Series([np.nan] * n)), errors="coerce"
    ).to_numpy()
    sup = neurons.get("superclass", pd.Series([""] * n)).astype("string").fillna("")

    # Resolution order: ground truth > consensus > raw prediction.
    call = pred.copy()
    call = call.where(~nt.isin(EXCITATORY + CNS_INHIBITORY + MODULATORY), nt)
    if trust_ground_truth:
        call = call.where(gt.isna(), gt)
    has_gt = gt.notna().to_numpy()

    low_conf = (~np.isnan(conf)) & (conf < confidence) & (~has_gt)
    call = call.mask(pd.Series(low_conf, index=call.index), other=pd.NA)
    # Collapse the missing-value sentinel to a plain string BEFORE leaving pandas.
    # pd.NA in an object array makes every elementwise == raise, and the failure is a
    # TypeError three frames away from the cause.
    call = call.fillna("__none__")

    sign = np.zeros(n, dtype=np.int8)
    kind = np.zeros(n, dtype=np.uint8)

    is_exc = call.isin(EXCITATORY).to_numpy(dtype=bool)
    is_inh = call.isin(CNS_INHIBITORY).to_numpy(dtype=bool)
    is_mod = call.isin(MODULATORY).to_numpy(dtype=bool)
    is_glu = (call == "glutamate").to_numpy(dtype=bool)

    sign[is_exc] = 1
    kind[is_exc] = 1
    sign[is_inh] = -1
    kind[is_inh] = 2
    kind[is_mod] = 3  # sign stays 0: modulators act multiplicatively, see ir.dynamics

    # --- the NMJ correction ---------------------------------------------------
    efferent = sup.isin(EFFERENT_SUPERCLASSES).to_numpy()
    flip = efferent & is_inh & is_glu
    sign[flip] = 1
    kind[flip] = 1
    n_flipped = int(flip.sum())

    rep = SignReport(
        n=n,
        n_excitatory=int((kind == 1).sum()),
        n_inhibitory=int((kind == 2).sum()),
        n_modulatory=int((kind == 3).sum()),
        n_unknown=int((kind == 0).sum()),
        n_nmj_corrected=n_flipped,
        n_low_confidence=int(low_conf.sum()),
        confidence_threshold=confidence,
    )
    return sign, kind, rep


def naive_signs(neurons: pd.DataFrame) -> np.ndarray:
    """The three-line dict everyone writes, kept so tests can prove it differs.

    Only for demonstrating the motor-neuron inversion. Never use it to build.
    """
    m = {"acetylcholine": 1, "glutamate": -1, "gaba": -1, "histamine": -1}
    return (
        neurons.get("consensus_nt", pd.Series([None] * len(neurons)))
        .astype("string").str.lower().map(m).fillna(0).to_numpy(dtype=np.int8)
    )
