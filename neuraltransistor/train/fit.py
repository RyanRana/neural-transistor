"""The training loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import torch
    _HAVE_TORCH = True
except ImportError:                                     # pragma: no cover
    _HAVE_TORCH = False


@dataclass
class FitResult:
    circuit: object                 # the TorchCircuit, fitted
    history: list = field(default_factory=list)
    best_loss: float = float("inf")
    best_metrics: dict = field(default_factory=dict)
    seconds: float = 0.0
    steps: int = 0

    def summary(self) -> str:
        m = self.best_metrics
        keys = [k for k in ("R_driven", "R_held", "drift_rad", "rate",
                            "trace_corr", "trace_mse") if k in m]
        body = "  ".join(f"{k}={m[k]:.4f}" for k in keys)
        return (f"fit: {self.steps} steps in {self.seconds:.0f}s, "
                f"loss {self.history[0]['loss']:.4f} -> {self.best_loss:.4f}\n  {body}")

    def curve(self) -> dict:
        return {k: [h[k] for h in self.history if k in h]
                for k in self.history[0].keys()} if self.history else {}


def fit(ir, task, steps: int = 300, lr: float = 0.02, device: Optional[str] = None,
        group_pairs_by: str = "pre_type", clip: float = 1.0, log_every: int = 25,
        verbose: bool = True, seed: int = 0, write_back: bool = True,
        curriculum: Optional[tuple] = None) -> FitResult:
    """Fit per-cell-type dynamics so ``ir`` satisfies ``task``.

    Structure is never touched -- synapse counts and signs stay exactly as measured, and
    only the biophysics moves. That is the whole point: a fit that could change who
    connects to whom would not be a connectome-constrained model, it would be an RNN
    with an unusually good initialisation.

    ``curriculum`` is a tuple of ``hold_ticks`` values trained in order, and for the ring
    attractor it is what makes the difference between a fit and a failure. Asked to hold
    a bump for 180 ticks from a cold start, the optimiser cannot get there gradually: the
    only gradient it can follow is "more activity everywhere", which wakes the hold phase
    up and smears the bump out in the same motion (measured: R_driven 0.610 -> 0.198,
    R_held still 0.000). Holding for 20 ticks is nearly free, and each stage starts from
    a solution that already works, so the search never has to trade shape for survival.
    """
    if not _HAVE_TORCH:
        raise ImportError("torch is not installed: uv pip install torch")
    from neuraltransistor.target.torch_backend import TorchCircuit

    torch.manual_seed(seed)
    tc = TorchCircuit(ir, device=device, group_pairs_by=group_pairs_by,
                      dt=ir.dynamics.dt if ir.dynamics else 5e-4)
    stages = tuple(curriculum) if curriculum else (getattr(task, "hold_ticks", None),)
    opt = torch.optim.Adam(tc.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    if verbose:
        print(tc.summary())
        if len(stages) > 1:
            print(f"  task {type(task).__name__}: curriculum hold_ticks {stages}")

    hist, best, best_m, best_state = [], float("inf"), {}, None
    per_stage = max(steps // len(stages), 1)
    drive, stage_i = None, -1
    t0 = time.time()
    for step in range(steps):
        si = min(step // per_stage, len(stages) - 1)
        if si != stage_i:
            stage_i = si
            if stages[si] is not None:
                task.hold_ticks = stages[si]
            drive = task.build_drive(tc.n, tc.device_, tc.dtype)
            # Each stage is a different problem, so a score from an easier one must not
            # be allowed to win the best-checkpoint comparison against a harder one.
            best, best_m, best_state = float("inf"), {}, None
            if verbose:
                print(f"  -- stage {si + 1}/{len(stages)}: "
                      f"hold {task.hold_ticks} ticks, drive {tuple(drive.shape)}")
        opt.zero_grad(set_to_none=True)
        spikes, _ = tc(drive)
        loss, metrics = task.loss(spikes, tc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tc.parameters(), clip)
        opt.step()
        sched.step()

        row = {"step": step, "loss": float(loss.detach()), **metrics}
        hist.append(row)
        if row["loss"] < best:
            best, best_m = row["loss"], metrics
            best_state = {k: v.detach().clone() for k, v in tc.state_dict().items()}
        if verbose and (step % log_every == 0 or step == steps - 1):
            extra = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                              if not k.startswith("l_"))
            print(f"  step {step:4d}  loss {row['loss']:.5f}   {extra}")

    if best_state is not None:
        tc.load_state_dict(best_state)
    if write_back:
        tc.to_dynamics()
    res = FitResult(circuit=tc, history=hist, best_loss=best, best_metrics=best_m,
                    seconds=time.time() - t0, steps=steps)
    if verbose:
        print(res.summary())
    return res


def refit_after_quantization(ir, task, weight_bits: int = 8, steps: int = 150,
                             p_real: Optional[float] = None, verbose: bool = True,
                             **kw) -> dict:
    """Fit, quantize, then fit again on the quantized structure.

    This order is not fussiness. Biswas et al. 2026 find synapse counts tolerate about
    +-90% variation provided the scale factors may be re-solved; Chang et al. 2023 find
    2% noise on *fixed* hand-tuned weights destroys the circuit. Both are true at once:
    the structure is robust, a frozen parameterization is brittle. So quantization -- which
    is a large, structured perturbation of the weights -- has to be followed by a re-fit
    rather than applied to a finished solution and hoped for.

    Returns the before/after metrics so the cost of quantizing is measured, not assumed.
    """
    from neuraltransistor.quant.quantize import compress, prune_by_reliability

    if verbose:
        print("=== fit on full precision ===")
    r1 = fit(ir, task, steps=steps, verbose=verbose, **kw)
    before = dict(r1.best_metrics)

    if verbose:
        print(f"\n=== quantize to int{weight_bits}"
              + (f", prune at P(real)>={p_real}" if p_real else "") + " ===")
    q = ir
    stats = {}
    if p_real:
        q, stats = prune_by_reliability(q, p_real)
    q, qrep = compress(q, weight_bits=weight_bits)
    q.dynamics = ir.dynamics            # carry the fitted dynamics across
    if verbose:
        print(f"  {qrep}")

    if verbose:
        print("\n=== re-fit on the quantized structure ===")
    r2 = fit(q, task, steps=steps, verbose=verbose, **kw)

    return {"before": before, "after_quantization_refit": dict(r2.best_metrics),
            "quant_report": str(qrep), "prune_stats": stats,
            "circuit": q, "fit_full": r1, "fit_quantized": r2}
