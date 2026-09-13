"""Physically-parameterized stimuli, and the published models they should reproduce.

These are test vectors, not a simulated world. A looming stimulus is fully described by
two numbers -- object half-size ``l`` and approach speed ``v`` -- and the whole escape
literature is organized around their ratio ``l/|v|``, so it is exactly reproducible and
directly comparable to published recordings.

Having a published reference model alongside the stimulus is the point. It converts
"our circuit produced some output" into "our circuit produced output that does or does
not match a model fitted to real giant-fiber recordings".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# the stimulus

@dataclass
class Looming:
    """An object of half-size l approaching at constant speed v.

    Time is measured toward collision: ``t`` runs negative up to 0, where 0 is the moment
    the object reaches the eye. The half-angle it subtends is

        theta(t) = arctan( l / (v * |t|) )

    and the ratio ``l_over_v = l/|v|`` (in seconds) sets the expansion profile. Larger
    l/|v| means a slower, more gradual loom. Drosophila escape experiments typically span
    l/|v| of roughly 10-80 ms.

    The closed form for the expansion rate is

        dtheta/dt = l*v / ( (v*t)^2 + l^2 )

    which peaks just before contact -- that peak, and its timing, is what looming
    detectors key on.
    """
    l_over_v: float = 0.040          # seconds (the "r/v" ratio; papers quote it in ms)
    dt: float = 0.001                # s per tick
    t_start: float | None = None     # default: the moment it reaches start_size_deg
    start_size_deg: float = 10.0     # FULL angle at stimulus onset
    end_size_deg: float = 90.0       # FULL angle where expansion stops
    hold_s: float = 0.12             # hold at end_size after expansion stops
    hold_ticks: int | None = None    # overrides hold_s if given

    def __post_init__(self):
        tau = self.l_over_v
        # Card-lab convention (von Reyn 2017, Ache 2019): theta is the FULL subtended
        # angle, t < 0 during expansion, and t = 0 is the *theoretical* time of contact,
        # defined as the moment the object would subtend 180 deg -- not the moment it
        # reaches the eye. Inverting theta = 2*arctan(tau/|t|):
        #     t(theta) = -tau / tan(theta/2)
        t_lo = -tau / np.tan(np.radians(self.start_size_deg) / 2)
        t_hi = -tau / np.tan(np.radians(self.end_size_deg) / 2)   # = -tau at 90 deg
        if self.t_start is not None:
            t_lo = min(self.t_start, t_hi - self.dt)
        self.t = np.arange(t_lo, t_hi, self.dt)
        self.theta = np.arctan(tau / np.abs(self.t))              # HALF angle
        self.theta_dot = tau / (self.t ** 2 + tau ** 2)           # d(half)/dt
        if self.hold_ticks is None:
            # Real experiments stop the expansion at 90 deg and HOLD it there. Without a
            # hold, a fast loom (small r/v) ends before the response can peak, because
            # expansion stops at exactly t = -tau while the giant fiber peaks later. The
            # truncation looks like a modelling error and is really a stimulus error.
            self.hold_ticks = int(round(self.hold_s / self.dt))
        if self.hold_ticks:
            # real experiments stop the expansion at 90 deg and hold it there; the
            # 180 deg endpoint is never displayed
            self.t = np.concatenate([self.t, self.t[-1] + self.dt * np.arange(
                1, self.hold_ticks + 1)])
            self.theta = np.concatenate([self.theta,
                                         np.full(self.hold_ticks, self.theta[-1])])
            self.theta_dot = np.concatenate([self.theta_dot,
                                             np.zeros(self.hold_ticks)])

    # --- closed forms worth having ------------------------------------------
    def t_at_size(self, full_angle_deg: float) -> float:
        """Exact time (s, negative) at which the object subtends this FULL angle."""
        return -self.l_over_v / np.tan(np.radians(full_angle_deg) / 2)

    def peak_bracket_ms(self, delay_ms: float = 19.0) -> tuple:
        """Where a real giant fiber's peak must fall, from Ache 2019 Fig 4B.

        A pure angular-SIZE detector peaks when the delayed size crosses 42 deg
        (t = -2.6051*tau + delay); a pure angular-VELOCITY detector peaks when expansion
        peaks, at 90 deg (t = -1.0*tau + delay). A real GF sits between the two, because
        it sums an LC4 velocity term and an LPLC2 size term. Any implementation whose
        peak falls outside this bracket is wrong.
        """
        tau_ms = self.l_over_v * 1000.0
        return (-2.6051 * tau_ms + delay_ms, -1.0 * tau_ms + delay_ms)

    @property
    def angular_size(self) -> np.ndarray:
        """Full subtended angle 2*theta, in radians -- what the literature reports."""
        return 2 * self.theta

    @property
    def t_ms(self) -> np.ndarray:
        return self.t * 1000.0

    def time_at_size(self, full_angle_deg: float) -> float:
        """When (s before contact) the object first subtends this full angle."""
        target = np.radians(full_angle_deg)
        hit = np.flatnonzero(self.angular_size >= target)
        return float(self.t[hit[0]]) if len(hit) else float("nan")

    def render(self, retina, contrast: float = 1.0, background: float = 1.0,
               centre=(0.0, 0.0)) -> np.ndarray:
        """(T, n_ommatidia) luminance of a dark disc on a bright field.

        Dark-on-bright is the standard escape stimulus: it drives the OFF pathway, which
        is where LPLC2 and LC4 live.
        """
        az = retina.azimuth - centre[0]
        el = retina.elevation - centre[1]
        ecc = np.sqrt(az ** 2 + el ** 2)                 # (n,)
        dark = ecc[None, :] <= self.theta[:, None]       # (T, n)
        out = np.full(dark.shape, background, dtype=np.float32)
        out[dark] = background * (1.0 - contrast)
        return out

    def render_image(self, width: int = 128, height: int = 128,
                     hfov_deg: float = 180.0, contrast: float = 1.0,
                     background: float = 1.0) -> np.ndarray:
        """(T, H, W) frames, as a camera would see it. For the sensor path."""
        hf = np.radians(hfov_deg)
        vf = hf * height / width
        px = ((np.arange(width) + 0.5) / width - 0.5) * hf
        py = ((np.arange(height) + 0.5) / height - 0.5) * vf
        AZ, EL = np.meshgrid(px, py)
        ecc = np.sqrt(AZ ** 2 + EL ** 2)
        dark = ecc[None, :, :] <= self.theta[:, None, None]
        out = np.full(dark.shape, background, dtype=np.float32)
        out[dark] = background * (1.0 - contrast)
        return out


# --------------------------------------------------------------------------- #
# the published reference

#: von Reyn et al. 2017 (Neuron 94:1190) + Ache et al. 2019 (Curr Biol 29:1073).
#: A static model: no membrane time constant, only transport delays. Four inputs sum
#: LINEARLY -- which is the interesting part, since the locust LGMD literature
#: (Gabbiani) explicitly rejects additive forms for the analogous neuron.
GF_PARAMS = dict(
    w_LC4=1.62, w_LPLC2=1.45, w_i1=2.27, w_i2=1.0,
    lc4_gain_mv_per_deg_per_s=2.567e-4, lc4_delay_ms=19.0,
    lplc2_peak_mv=1.7, lplc2_mu_deg=42.0, lplc2_sigma_log=0.52, lplc2_delay_ms=19.0,
    i1_offset=-0.53, i1_amp=0.59, i1_mid_deg=66.0, i1_slope=-11.0, i1_delay_ms=37.5,
    i2_amp=-0.52, i2_mu_deg=26.0, i2_sigma_deg=7.8, i2_delay_ms=11.0,
    threshold_size_deg=39.0,   # behavioural GF size threshold (Ache 2019)
)


def _delay(x: np.ndarray, ms: float, dt: float) -> np.ndarray:
    """Shift a signal later in time by ``ms``, holding the first value."""
    k = int(round(ms / 1000.0 / dt))
    if k <= 0:
        return x
    return np.concatenate([np.full(k, x[0]), x[:-k]])


def giant_fiber(loom: Looming, params: dict = None) -> dict:
    """The published GF model's response to a looming stimulus.

    Returns each component plus the summed membrane voltage, so you can see that LC4
    contributes a term linear in angular VELOCITY while LPLC2 contributes a log-Gaussian
    in angular SIZE -- the asymmetry that makes the pair a size-and-speed detector rather
    than either alone.
    """
    p = dict(GF_PARAMS, **(params or {}))
    dt = loom.dt
    size_deg = np.degrees(loom.angular_size)
    rate_deg = np.degrees(2 * loom.theta_dot)

    v_lc4 = p["lc4_gain_mv_per_deg_per_s"] * _delay(rate_deg, p["lc4_delay_ms"], dt)

    s = np.clip(_delay(size_deg, p["lplc2_delay_ms"], dt), 1e-3, None)
    v_lplc2 = p["lplc2_peak_mv"] * np.exp(
        -((np.log(s) - np.log(p["lplc2_mu_deg"])) ** 2) / (2 * p["lplc2_sigma_log"] ** 2))

    s1 = _delay(size_deg, p["i1_delay_ms"], dt)
    v_i1 = p["i1_offset"] + p["i1_amp"] / (1 + np.exp(-(s1 - p["i1_mid_deg"]) / p["i1_slope"]))

    s2 = _delay(size_deg, p["i2_delay_ms"], dt)
    v_i2 = p["i2_amp"] * np.exp(-((s2 - p["i2_mu_deg"]) ** 2) / (2 * p["i2_sigma_deg"] ** 2))

    v_gf = (p["w_LC4"] * v_lc4 + p["w_LPLC2"] * v_lplc2
            + p["w_i1"] * v_i1 + p["w_i2"] * v_i2)

    peak = int(np.argmax(v_gf))
    return {
        "t_ms": loom.t_ms, "size_deg": size_deg, "rate_deg_s": rate_deg,
        "v_LC4": v_lc4, "v_LPLC2": v_lplc2, "v_i1": v_i1, "v_i2": v_i2, "v_GF": v_gf,
        "peak_idx": peak,
        "peak_t_ms": float(loom.t_ms[peak]),
        "peak_size_deg": float(size_deg[peak]),
        "peak_mv": float(v_gf[peak]),
        "t_at_threshold_ms": loom.time_at_size(p["threshold_size_deg"]) * 1000.0,
    }


def sweep_l_over_v(values=(0.010, 0.020, 0.040, 0.080), **kw) -> list:
    """The standard experimental sweep: peak timing should scale with l/|v|."""
    out = []
    for lv in values:
        loom = Looming(l_over_v=lv, **kw)
        r = giant_fiber(loom)
        out.append({"l_over_v_ms": lv * 1000, "peak_t_ms": r["peak_t_ms"],
                    "peak_size_deg": r["peak_size_deg"], "peak_mv": r["peak_mv"],
                    "response": r, "loom": loom})
    return out
