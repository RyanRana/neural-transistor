"""A differentiable, batched, GPU runtime for a CircuitIR.

The integer runtime and the emitted C exist to be *deployed*. This exists to be
*trained*. It is the same circuit, in float, on whatever device you have, with gradients
flowing through it -- which is what the fitting stage needs and what the fixed-point
runtime cannot provide.

Three things it buys:

**Speed.** One sparse matvec per tick on the GPU. The 34,038-neuron looming pathway runs
thousands of trials in parallel instead of one at a time.

**Gradients.** Spiking is a step function with zero derivative everywhere, so the backward
pass uses a surrogate -- a fast sigmoid whose derivative is well-behaved. This is the
standard trick from the SNN training literature and it is the only reason
gradient-descent fitting of a spiking connectome is possible at all.

**The right parameterization.** Free parameters are per CELL TYPE, not per neuron and
certainly not per synapse. That is the consensus across every successful
connectome-constrained model: Lappalainen 2024 fits 734 parameters over 45,669 neurons,
Duan 2025 fits 57 over 439. Per-synapse weights would be 889,288 free parameters for the
looming pathway and would fit anything, which is the same as fitting nothing. Here it is
three scalars per type plus one scale per type-pair-class, and the synapse counts stay
fixed as the measured prior they are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAVE_TORCH = True
except ImportError:                                     # pragma: no cover
    _HAVE_TORCH = False
    nn = object


def best_device() -> str:
    if not _HAVE_TORCH:
        raise ImportError("torch is not installed: uv pip install torch")
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _SurrogateSpike(torch.autograd.Function if _HAVE_TORCH else object):
    """Heaviside forward, fast-sigmoid derivative backward.

    d/dx of a step is zero almost everywhere and infinite at the threshold, so training
    through it is impossible without a stand-in. The fast sigmoid surrogate
    1/(1+|x|*beta)^2 is the usual choice: cheap, bounded, and it does not vanish far from
    threshold the way a Gaussian surrogate does.
    """
    @staticmethod
    def forward(ctx, x, beta: float = 10.0):
        ctx.save_for_backward(x)
        ctx.beta = beta
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + ctx.beta * x.abs()) ** 2
        return grad_out * sg, None


def spike(x, beta: float = 10.0):
    return _SurrogateSpike.apply(x, beta)


@dataclass
class TypeIndex:
    """Maps every neuron to its cell type, and every edge to a type-pair class."""
    type_of_neuron: np.ndarray     # int32, index into `names`
    names: list
    pair_class_of_edge: np.ndarray  # int32, index into `pair_names`
    pair_names: list

    @property
    def n_types(self) -> int:
        return len(self.names)

    @property
    def n_pair_classes(self) -> int:
        return len(self.pair_names)


def build_type_index(ir, group_pairs_by: str = "sign") -> TypeIndex:
    """Group neurons by cell type and edges by type-pair class.

    ``group_pairs_by``:
      ``"sign"``      3 classes (exc / inh / unknown). Fewest parameters, and the
                      literature's minimum viable choice -- Liao & Lo measured distinct
                      count-to-strength mappings per transmitter.
      ``"pre_type"``  one scale per PREsynaptic cell type. The default, and the right
                      biological unit: synaptic strength is a property of the cell type
                      making the synapse, which is exactly what Liao & Lo measure per
                      transmitter and what varies between, say, Delta7 and ER even
                      though both are inhibitory. ``"sign"`` cannot express that
                      difference, and on the compass that is fatal -- ER puts 124k
                      inhibitory synapses onto EPG against Delta7's 4.3k, so a single
                      inhibitory scale has to choose between killing the bump and
                      removing its surround.
      ``"post_type"`` one scale per postsynaptic cell type.
      ``"type_pair"`` one per (pre type, post type) that actually occurs. Most
                      expressive, and for a large circuit it is a lot of parameters --
                      expressive enough to start fitting the task rather than the
                      biology, so prefer ``pre_type`` unless you have a reason.
    """
    names, inv = np.unique(np.array(ir.types, dtype=object), return_inverse=True)
    tid = inv.astype(np.int32)
    src = np.repeat(np.arange(ir.n_neurons, dtype=np.int64), np.diff(ir.indptr))
    dst = ir.indices.astype(np.int64)

    if group_pairs_by == "sign":
        pc = ir.sign_kind[src].astype(np.int32)
        pnames = ["unknown", "excitatory", "inhibitory", "modulatory"]
    elif group_pairs_by == "pre_type":
        pc = tid[src]
        pnames = [str(x) for x in names]
    elif group_pairs_by == "post_type":
        pc = tid[dst]
        pnames = [str(x) for x in names]
    elif group_pairs_by == "type_pair":
        key = tid[src].astype(np.int64) * len(names) + tid[dst]
        uk, pc = np.unique(key, return_inverse=True)
        pc = pc.astype(np.int32)
        pnames = [f"{names[k // len(names)]}->{names[k % len(names)]}" for k in uk]
    else:
        raise ValueError(group_pairs_by)
    return TypeIndex(tid, [str(x) for x in names], pc, pnames)


class TorchCircuit(nn.Module if _HAVE_TORCH else object):
    """A CircuitIR as a trainable spiking network.

    Structure (who connects to whom, and with how many synapses) is FIXED -- it is the
    measurement. Only the biophysics is learned.
    """

    def __init__(self, ir, device: Optional[str] = None, dtype=None,
                 group_pairs_by: str = "pre_type", dt: Optional[float] = None,
                 surrogate_beta: float = 10.0,
                 refractory_ticks: Optional[int] = None):
        if not _HAVE_TORCH:
            raise ImportError("torch is not installed: uv pip install torch")
        super().__init__()
        self.ir = ir
        self.n = ir.n_neurons
        self.device_ = device or best_device()
        self.dtype = dtype or torch.float32
        self.beta = surrogate_beta
        self.dt = dt if dt is not None else (ir.dynamics.dt if ir.dynamics else 5e-4)

        # Refractory period, in ticks, matching the emitted C exactly. The generated
        # kernel forces `refrac` ticks of silence after every spike, which caps any
        # neuron at a 1-in-(refrac+1) duty cycle. Training without it lets the fit buy
        # persistence with firing rates the deployment target cannot produce -- measured:
        # PEG settled at duty cycle 1.0000 and PEN at 0.5145 against a C ceiling of 0.2.
        # A fitted solution that cannot be emitted is not a fitted solution.
        if refractory_ticks is None:
            r = ir.dynamics.refractory if ir.dynamics is not None else 0.0022
            refractory_ticks = int(round(float(np.median(np.atleast_1d(r))) / self.dt))
        self.refrac = max(int(refractory_ticks), 0)

        self.tix = build_type_index(ir, group_pairs_by)
        tid = torch.as_tensor(self.tix.type_of_neuron.astype(np.int64))
        pc = torch.as_tensor(self.tix.pair_class_of_edge.astype(np.int64))
        self.register_buffer("tid", tid.to(self.device_))
        self.register_buffer("pair_class", pc.to(self.device_))

        # fixed structure
        src = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(ir.indptr))
        idx = torch.as_tensor(np.stack([ir.indices.astype(np.int64), src]))  # (post, pre)
        w = torch.as_tensor(ir.weight.astype(np.float32))
        sgn = torch.as_tensor(ir.sign.astype(np.float32))[torch.as_tensor(src)]
        self.register_buffer("edge_index", idx.to(self.device_))
        self.register_buffer("edge_count", (w * sgn).to(self.device_, self.dtype))

        # free parameters, one per cell type
        T = self.tix.n_types
        self.log_tau = nn.Parameter(torch.full((T,), float(np.log(0.020))))
        self.threshold = nn.Parameter(torch.ones(T))
        self.rest = nn.Parameter(torch.zeros(T))
        self.log_gain = nn.Parameter(torch.zeros(T))
        # One synaptic scale per type-pair class, initialised so the network starts at
        # the edge of firing rather than saturated. Shiu et al.'s 0.275 mV is calibrated
        # for their units; applied to raw synapse counts here it puts typical input two
        # orders of magnitude over threshold and every neuron fires every tick, which
        # gives the surrogate gradient nothing to work with. So init from the data:
        # scale such that a median neuron's total excitatory input lands near threshold.
        tot = np.bincount(ir.indices, weights=np.abs(ir.weight.astype(np.float64)),
                          minlength=self.n)
        typical = float(np.median(tot[tot > 0])) if (tot > 0).any() else 1.0
        w0 = 1.0 / max(typical, 1.0)
        self.log_wsyn = nn.Parameter(torch.full((self.tix.n_pair_classes,),
                                                float(np.log(w0))))
        self.w_init = w0
        self.to(self.device_)

    # -- parameter expansion --------------------------------------------------

    def _edge_weights(self):
        """synapse count x per-class scale -> effective weight. Structure stays fixed."""
        return self.edge_count * torch.exp(self.log_wsyn)[self.pair_class]

    def _sparse(self):
        return torch.sparse_coo_tensor(
            self.edge_index, self._edge_weights(), (self.n, self.n)).coalesce()

    @property
    def n_free_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- simulation -----------------------------------------------------------

    def init_state(self, batch: int = 1):
        z = torch.zeros(batch, self.n, device=self.device_, dtype=self.dtype)
        return {"v": z.clone(), "s": z.clone(), "ref": z.clone()}

    def forward(self, drive, state=None, record: bool = True):
        """``drive``: (T, n) or (T, batch, n). Returns (spikes, state)."""
        d = drive if drive.dim() == 3 else drive.unsqueeze(1)
        T, B, _ = d.shape
        st = state or self.init_state(B)
        v, s = st["v"], st["s"]
        ref = st.get("ref")
        if ref is None:
            ref = torch.zeros_like(v)
        W = self._sparse()
        decay = torch.exp(-self.dt / torch.exp(self.log_tau).clamp_min(1e-4))[self.tid]
        thr = self.threshold[self.tid]
        rest = self.rest[self.tid]
        gain = torch.exp(self.log_gain)[self.tid]
        out = []
        for t in range(T):
            inp = torch.sparse.mm(W, s.t()).t() * gain + d[t]
            v = decay * (v - rest) + rest + inp
            if self.refrac:
                # While refractory the C discards input and pins v at 0, so do that here.
                # The mask is discrete and therefore detached; the gradient still flows
                # through v and through the spike that set the counter.
                live = (ref <= 0).to(v.dtype)
                v = v * live
                s = spike(v - thr, self.beta) * live
                ref = (ref - 1.0).clamp_min(0.0) + s.detach() * float(self.refrac)
            else:
                s = spike(v - thr, self.beta)
            v = v * (1.0 - s)                    # reset by subtraction of the whole
            if record:
                out.append(s)
        spikes = torch.stack(out) if record else None
        return spikes, {"v": v, "s": s, "ref": ref}

    # -- interop --------------------------------------------------------------

    def to_dynamics(self):
        """Write fitted parameters back into the IR's Dynamics, ready to quantize."""
        with torch.no_grad():
            tid = self.tix.type_of_neuron
            dyn = self.ir.dynamics
            dyn.tau_m = np.exp(self.log_tau.detach().cpu().numpy())[tid].astype(np.float32)
            dyn.threshold = self.threshold.detach().cpu().numpy()[tid].astype(np.float32)
            dyn.rest = self.rest.detach().cpu().numpy()[tid].astype(np.float32)
            dyn.gain = np.exp(self.log_gain.detach().cpu().numpy())[tid].astype(np.float32)
            dyn.dt = self.dt
        return dyn

    def summary(self) -> str:
        return (f"TorchCircuit[{self.device_}] {self.n:,} neurons / "
                f"{self.ir.n_edges:,} edges\n"
                f"  {self.tix.n_types:,} cell types, "
                f"{self.tix.n_pair_classes} synaptic classes\n"
                f"  {self.n_free_parameters:,} free parameters "
                f"({self.n_free_parameters / max(self.ir.n_edges, 1):.5f} per edge)"
                f"\n  synaptic scale initialised at {self.w_init:.4g} "
                f"(1 / median total input)"
                f"\n  refractory {self.refrac} ticks "
                f"(duty-cycle ceiling {1.0 / (self.refrac + 1):.3f}), matching the emitted C")
