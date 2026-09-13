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
    drive_amp: float = 0.9
    drive_width: float = 0.6
    n_headings: int = 4             # batch: seed the bump at several angles
    target_R: float = 0.65          # from the measured bump width (Seelig 2015)
    #: Target MEAN FIRING RATE in Hz, converted to a duty cycle against the circuit's dt.
    #: It has to be a rate, not a duty cycle. The previous value, 0.10, was written as
    #: "flies run near 10% active" -- but that is the fraction of neurons active, while
    #: ``spikes.mean()`` is the per-neuron duty cycle. At dt = 0.5 ms a duty cycle of 0.10
    #: is 200 Hz, against a refractory ceiling of 0.20, so the rate target sat at half of
    #: saturation and "hit the target" and "saturate" were nearly the same instruction.
    #: EPG neurons in a fly fire at tens of Hz (Seelig & Jayaraman 2015).
    target_rate_hz: float = 20.0
    hold_windows: int = 4           # score persistence across the hold, not just at its end
    w_form: float = 2.0
    w_persist: float = 3.0          # the term that failed without fitting
    w_drift: float = 1.0
    w_rate: float = 0.5
    w_alive: float = 1.0
    w_sat: float = 6.0              # keeps "stay alive" from being answered by saturating
    sat_frac: float = 0.8           # duty cycles above this fraction of the ceiling cost
    hold_chunk: int = 15            # ticks per window when scoring R across the hold

    @classmethod
    def for_circuit(cls, ir, neurons_df, cell_type: str = "EPG", **kw):
        """Build the task straight from a circuit, recovering the ring from annotation.

            task = RingAttractorTask.for_circuit(ir, conn.with_columns("instance"))

        Saves the caller from importing ``evals.functional.ring_map`` and unpacking it by
        hand, which was the only way to construct this and is not obvious from here.
        """
        from neuraltransistor.evals.functional import ring_map
        rm = ring_map(ir, neurons_df, cell_type=cell_type)
        if rm.n == 0:
            raise ValueError(f"no {cell_type} neurons with a recoverable ring position")
        return cls(ring_theta=rm.theta, ring_idx=rm.local_idx, **kw)

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
        """spikes: (T, batch, n). Returns (loss, metrics dict).

        Everything is scored PER PHASE. Scoring the whole trace at once lets the two
        halves cancel: a fit that saturates while driven and dies once released averages
        out to a healthy-looking mean rate, and that is exactly the solution the
        optimiser kept finding. Persistence is scored across the whole hold rather than
        at the end of it, so a bump that decays is penalised while it is decaying, when
        there is still a gradient to act on, instead of only once it is gone.
        """
        th = torch.as_tensor(self.ring_theta, device=spikes.device, dtype=spikes.dtype)
        idx = torch.as_tensor(self.ring_idx.astype(np.int64), device=spikes.device)
        ring = spikes[..., idx]
        dt = float(getattr(circuit, "dt", 5e-4))
        target_duty = self.target_rate_hz * dt

        D = self.drive_ticks
        win = max(D // 3, 1)
        driven = ring[D - win:D].mean(0)
        R_dr, _ = _circ_concentration(driven, th)

        # the hold, in chunks, so R is measured all the way along it
        hold = ring[D:]
        nchunk = max(int(hold.shape[0]) // max(self.hold_chunk, 1), 2)
        chunks = torch.chunk(hold, nchunk, dim=0)
        Rs, angs = [], []
        for c in chunks:
            r, a = _circ_concentration(c.mean(0), th)
            Rs.append(r); angs.append(a)
        R_chunks = torch.stack(Rs)                       # (nchunk, batch)
        R_l = R_chunks[-1]

        # 1. a bump forms, of about the measured width
        l_form = ((R_dr - self.target_R) ** 2).mean()
        # 2. and is still there at every point of the hold, not just the end
        l_persist = ((R_chunks - self.target_R) ** 2).mean()
        # 3. without sliding around the ring. Weighted by how much of a bump there
        #    actually is -- the preferred angle of a silent ring is not a small drift,
        #    it is undefined, and that noise was measured dominating the loss while the
        #    network sat at exactly zero held activity. Detached so it scales the drift
        #    gradient without paying the network to go quiet; that pressure belongs to
        #    l_persist and l_rate, which are in comparable units.
        drift = torch.atan2(torch.sin(angs[-1] - angs[0]), torch.cos(angs[-1] - angs[0]))
        l_drift = (R_l.detach().clamp(0.0, 1.0) * drift ** 2).mean()
        # 4. at a sane firing rate in BOTH phases, as a relative error so it is
        #    comparable with the R terms rather than 500x smaller than them.
        #    Measured on the RING, not on the whole circuit. "EPG neurons fire at tens of
        #    Hz" is a number someone recorded (Seelig & Jayaraman 2015); "the mean of a
        #    heterogeneous 452-neuron circuit is 20 Hz" is a number nobody has ever
        #    measured, and here it is actively misleading -- 282 of those 452 are ER
        #    cells carrying visual input this isolated compass does not have, so they sit
        #    near silent and drag the mean down, and the optimiser answers by driving
        #    whatever is left far too hard.
        #    As a LOG ratio, not a relative error: squared relative error is unbounded
        #    above and the driven ring starts around 8x the target, which scores 56 on its
        #    own and swamps every other term. A log ratio is symmetric -- twice the target
        #    and half of it cost the same -- and stays O(1) on the overshoot.
        #    Scored on the HOLD ONLY. While the stimulus is on, the firing rate is a
        #    property of the stimulus -- an amplitude this task picked -- not of the
        #    biophysics being fitted, and penalising it just asks the network to fight
        #    the drive. It complies by raising thresholds until no bump forms at all:
        #    measured R_driven 0.610 -> 0.061 while the driven rate climbed 170 -> 377 Hz
        #    anyway. Saturation during drive is still penalised, by l_sat, which is the
        #    part that is actually about the circuit rather than about the stimulus.
        rate_dr = ring[:D].mean()
        rate_ho = ring[D:].mean()
        l_rate = torch.log(rate_ho.clamp_min(1e-6) / target_duty) ** 2
        # 5. specifically not dead once the input stops
        alive = rate_ho
        l_alive = torch.relu(1.0 - alive / (target_duty * 0.3)) ** 2
        # 6. and not alive by the cheap route. The refractory period caps every neuron at
        #    a 1-in-(refrac+1) duty cycle, and a neuron pinned there carries no
        #    information -- the spiking equivalent of a dead ReLU, and a whole ring pinned
        #    there is uniform, which is the opposite of a bump. Measured without this
        #    term: R_driven 0.610 at init collapsing to 0.000 with every neuron at
        #    exactly the 0.200 ceiling. Per phase, because a neuron saturated for the
        #    drive alone only reaches 0.08 of a whole-trace duty cycle and hides.
        #    Scored PER CELL TYPE, not per neuron: a neuron-mean dilutes a saturated
        #    population against its own quiet members, and that is what hid the failure
        #    here. ER is 282 of these 452 neurons; it pinned at the ceiling during the
        #    hold and crushed EPG through its 124k inhibitory synapses, while the
        #    neuron-averaged penalty stayed near zero because most ER cells were quiet.
        refrac = int(getattr(circuit, "refrac", 0) or 0)
        ceiling = 1.0 / (refrac + 1)
        tid = getattr(circuit, "tid", None)
        def sat(x):
            duty = x.mean(0).mean(0)                       # (n,) per-neuron duty
            if tid is None:
                return (torch.relu(duty / ceiling - self.sat_frac) ** 2).mean()
            nt = int(tid.max()) + 1
            tot = torch.zeros(nt, device=duty.device, dtype=duty.dtype).index_add_(0, tid, duty)
            cnt = torch.zeros(nt, device=duty.device, dtype=duty.dtype).index_add_(
                0, tid, torch.ones_like(duty))
            per_type = tot / cnt.clamp_min(1.0)
            return (torch.relu(per_type / ceiling - self.sat_frac) ** 2).mean()
        l_sat = sat(spikes[:D]) + sat(spikes[D:])

        total = (self.w_form * l_form + self.w_persist * l_persist
                 + self.w_drift * l_drift + self.w_rate * l_rate
                 + self.w_alive * l_alive + self.w_sat * l_sat)
        f = lambda t: float(t.detach())
        return total, {
            "R_driven": f(R_dr.mean()), "R_held": f(R_l.mean()),
            "R_hold_mean": f(R_chunks.mean()),
            "drift_rad": f(drift.abs().mean()),
            "hz_drive": f(rate_dr) / dt, "hz_hold": f(rate_ho) / dt,
            "hz_all": f(spikes[D:].mean()) / dt,
            "l_form": f(l_form), "l_persist": f(l_persist), "l_drift": f(l_drift),
            "l_rate": f(l_rate), "l_alive": f(l_alive), "l_sat": f(l_sat),
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
