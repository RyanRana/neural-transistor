"""Objectives: what a fitted circuit has to actually do."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    _HAVE_TORCH = True
except ImportError:                                     # pragma: no cover
    _HAVE_TORCH = False


def _circ_concentration(rates, theta):
    """R = |sum_j r_j e^{i theta_j}| / sum_j r_j, differentiably.

    R near 1 means one tight bump, R near 0 means activity smeared around the ring.
    Computed from sines and cosines rather than complex numbers because autograd support
    for complex ops is patchier than it looks.
    """
    tot = rates.sum(-1).clamp_min(1e-6)
    c = (rates * torch.cos(theta)).sum(-1) / tot
    s = (rates * torch.sin(theta)).sum(-1) / tot
    return torch.sqrt(c * c + s * s + 1e-12), torch.atan2(s, c)


@dataclass
class RingAttractorTask:
    """Make the compass hold a heading.

    Self-supervised: no recorded neural activity is used, only the requirements a
    heading system must satisfy to be one. Each term below is a thing that is
    *observably* true of a fly's EPG bump.
    """
    ring_theta: np.ndarray          # radians, per ring neuron
    ring_idx: np.ndarray            # circuit-local indices of the ring neurons
    drive_ticks: int = 120
    hold_ticks: int = 180
    drive_amp: float = 1.4
    drive_width: float = 0.6
    n_headings: int = 4             # batch: seed the bump at several angles
    target_R: float = 0.65          # from the measured bump width (Seelig 2015)
    target_activity: float = 0.10   # flies run near 10% active
    w_form: float = 1.0
    w_persist: float = 3.0          # the term that failed without fitting
    w_drift: float = 1.0
    w_rate: float = 0.5
    w_alive: float = 2.0

    def build_drive(self, n_neurons: int, device, dtype):
        """(T, batch, n) input: a bump of drive at several headings, then silence."""
        T = self.drive_ticks + self.hold_ticks
        d = torch.zeros(T, self.n_headings, n_neurons, device=device, dtype=dtype)
        th = torch.as_tensor(self.ring_theta, device=device, dtype=dtype)
        idx = torch.as_tensor(self.ring_idx.astype(np.int64), device=device)
        for b in range(self.n_headings):
            centre = 2 * np.pi * b / self.n_headings
            w = torch.exp(torch.cos(th - centre) / (self.drive_width ** 2))
            w = w / w.max()
            d[:self.drive_ticks, b, idx] = w * self.drive_amp
        return d

    def loss(self, spikes, circuit):
        """spikes: (T, batch, n). Returns (loss, metrics dict)."""
        th = torch.as_tensor(self.ring_theta, device=spikes.device, dtype=spikes.dtype)
        idx = torch.as_tensor(self.ring_idx.astype(np.int64), device=spikes.device)
        ring = spikes[..., idx]

        win = max(self.drive_ticks // 3, 1)
        driven = ring[self.drive_ticks - win:self.drive_ticks].mean(0)
        held_early = ring[self.drive_ticks:self.drive_ticks + win].mean(0)
        held_late = ring[-win:].mean(0)

        R_dr, _ = _circ_concentration(driven, th)
        R_e, a_e = _circ_concentration(held_early, th)
        R_l, a_l = _circ_concentration(held_late, th)

        # 1. a bump forms, of about the measured width
        l_form = ((R_dr - self.target_R) ** 2).mean()
        # 2. and survives the input going away -- the term that fails unfitted
        l_persist = ((R_l - self.target_R) ** 2).mean()
        # 3. without sliding around the ring
        drift = torch.atan2(torch.sin(a_l - a_e), torch.cos(a_l - a_e))
        l_drift = (drift ** 2).mean()
        # 4. at a sane firing rate
        rate = spikes.mean()
        l_rate = (rate - self.target_activity) ** 2
        # 5. and specifically not dead after the input stops. Silence trivially
        #    satisfies "no drift", so it has to be penalised explicitly.
        alive = held_late.mean()
        l_alive = torch.relu(self.target_activity * 0.3 - alive) ** 2

        total = (self.w_form * l_form + self.w_persist * l_persist
                 + self.w_drift * l_drift + self.w_rate * l_rate
                 + self.w_alive * l_alive)
        return total, {
            "R_driven": float(R_dr.mean()), "R_held": float(R_l.mean()),
            "drift_rad": float(drift.abs().mean()), "rate": float(rate),
            "held_rate": float(alive),
            "l_form": float(l_form), "l_persist": float(l_persist),
            "l_drift": float(l_drift), "l_alive": float(l_alive),
        }


@dataclass
class LoomingTask:
    """Make the escape command fire at the right angular size.

    Supervised by the published giant-fiber model, which was fitted to real recordings.
    The target is its membrane trace, normalised, across a batch of approach speeds --
    so the circuit has to reproduce not just "fires eventually" but the l/|v| scaling.
    """
    readout_idx: np.ndarray
    l_over_v: tuple = (0.010, 0.020, 0.040, 0.080)
    dt: float = 0.005
    t_start: float = -0.6
    drive_amp: float = 1.2
    retinotopy: object = None
    retina: object = None
    target_activity: float = 0.05
    w_trace: float = 1.0
    w_rate: float = 0.4
    _cache: dict = field(default_factory=dict)

    def _make(self, n_neurons, device, dtype):
        from neuraltransistor.stimuli import Looming, giant_fiber
        drives, targets = [], []
        for lv in self.l_over_v:
            lo = Looming(l_over_v=lv, dt=self.dt, t_start=self.t_start)
            lum = lo.render(self.retina)
            d = self.retinotopy.drive_series(lum, amplitude=1.0).astype(np.float32)
            drives.append(d * self.drive_amp)
            g = giant_fiber(lo)["v_GF"]
            g = (g - g.min()) / max(g.ptp(), 1e-9)
            targets.append(g.astype(np.float32))
        T = min(len(d) for d in drives)
        dr = np.stack([d[:T] for d in drives], axis=1)
        tg = np.stack([t[:T] for t in targets], axis=1)
        return (torch.as_tensor(dr, device=device, dtype=dtype),
                torch.as_tensor(tg, device=device, dtype=dtype))

    def build_drive(self, n_neurons, device, dtype):
        if "d" not in self._cache:
            d, t = self._make(n_neurons, device, dtype)
            self._cache["d"], self._cache["t"] = d, t
        return self._cache["d"]

    def loss(self, spikes, circuit):
        tgt = self._cache["t"]
        idx = torch.as_tensor(self.readout_idx.astype(np.int64), device=spikes.device)
        out = spikes[..., idx].mean(-1)                      # (T, batch)
        # scale-free comparison: the published model is in mV, ours is a firing rate
        o = (out - out.mean(0, keepdim=True)) / out.std(0, keepdim=True).clamp_min(1e-6)
        t = (tgt - tgt.mean(0, keepdim=True)) / tgt.std(0, keepdim=True).clamp_min(1e-6)
        l_trace = ((o - t) ** 2).mean()
        l_rate = (spikes.mean() - self.target_activity) ** 2
        total = self.w_trace * l_trace + self.w_rate * l_rate
        with torch.no_grad():
            corr = float((o * t).mean())
        return total, {"trace_mse": float(l_trace), "trace_corr": corr,
                       "rate": float(spikes.mean())}
