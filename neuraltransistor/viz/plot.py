"""Light-theme plotting for neuraltransistor."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

PALETTE = {
    "bg": "#ffffff", "panel": "#fafaf9", "fg": "#18181b", "dim": "#71717a",
    "faint": "#a1a1aa", "line": "#e7e5e4",
    "accent": "#b45309", "ok": "#15803d", "bad": "#b91c1c",
    "cool": "#1d4ed8", "violet": "#7c3aed",
}
SERIES = [PALETTE["accent"], PALETTE["ok"], PALETTE["cool"],
          PALETTE["bad"], PALETTE["violet"], "#0891b2"]


def _rc():
    return {
        "figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["bg"],
        "savefig.facecolor": PALETTE["bg"], "text.color": PALETTE["fg"],
        "axes.labelcolor": PALETTE["fg"], "xtick.color": PALETTE["dim"],
        "ytick.color": PALETTE["dim"], "axes.edgecolor": PALETTE["line"],
        "grid.color": PALETTE["line"], "font.family": "monospace",
        "font.monospace": ["Menlo", "DejaVu Sans Mono"], "font.size": 9,
        "axes.titlesize": 10, "figure.dpi": 170, "axes.grid": True,
        "grid.linewidth": 0.5, "grid.alpha": 0.7, "legend.frameon": False,
    }


@contextmanager
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context(_rc()):
        yield plt


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def _marker_size(fig, ax, az, el, figsize) -> float:
    """Marker area in points^2 such that hexagons tile instead of speckling.

    Derived from the actual median nearest-neighbour spacing rather than guessed, so the
    same call looks right for a 40-ommatidium patch and a full 887-ommatidium eye.
    """
    if len(az) < 3:
        return 40.0
    pts = np.stack([az, el], axis=1)
    k = min(len(pts), 400)
    sel = pts[np.linspace(0, len(pts) - 1, k).astype(int)]
    d = np.sqrt(((sel[:, None, :] - sel[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    spacing = float(np.median(d.min(axis=1)))
    span = max(np.ptp(az), np.ptp(el)) or 1.0
    inches = min(figsize) * 0.78          # axes are smaller than the figure
    pts_per_deg = inches * 72.0 / span
    diam = spacing * pts_per_deg
    return float(np.clip(np.pi * (diam / 2) ** 2 * 0.95, 4.0, 4000.0))


def _fig(ax, figsize):
    import matplotlib.pyplot as plt
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)


# --------------------------------------------------------------------------- #
# the eye

def lattice(retina, values=None, ax=None, title=None, cmap="magma",
            size=None, show_axes=False, figsize=(5.2, 5.0)):
    """Draw the ommatidial lattice, optionally coloured by a per-ommatidium value.

    Pass ``values`` to show what the eye is seeing, or an activity vector to show what
    a columnar cell type is doing.
    """
    with style() as plt:
        fig, ax = _fig(ax, figsize)
        az = np.degrees(retina.azimuth)
        el = np.degrees(retina.elevation)
        if size is None:
            size = _marker_size(fig, ax, az, el, figsize)
        if values is None:
            ax.scatter(az, el, s=size, c=PALETTE["accent"], alpha=.75,
                       edgecolors="none", marker="h")
        else:
            v = np.asarray(values, dtype=float)
            sc = ax.scatter(az, el, s=size, c=v, cmap=cmap, marker="h",
                            edgecolors="none")
            cb = fig.colorbar(sc, ax=ax, fraction=.042, pad=.03)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=7, color=PALETTE["line"])
        if show_axes and retina.hex1 is not None:
            from neuraltransistor.sensors.retina import HEX_AXES
            nb = retina.neighbours()
            c = len(az) // 2
            for (off, name), col in zip(HEX_AXES, SERIES):
                j = nb[off][c]
                if j >= 0:
                    ax.plot([az[c], az[j]], [el[c], el[j]], color=col, lw=2,
                            label=name, zorder=5)
            ax.legend(fontsize=6.5, loc="upper right", labelcolor=PALETTE["fg"])
        ax.set_aspect("equal")
        ax.set_xlabel("azimuth (deg)"); ax.set_ylabel("elevation (deg)")
        ax.set_title(title or f"{retina.n} ommatidia · {retina.source}",
                     loc="left", pad=9)
        ax.grid(False)
        _clean(ax)
        if ax.figure is fig and len(fig.axes) <= 2:
            fig.tight_layout()
    return fig


def coverage(retina, resampler, ax=None, figsize=(5.2, 5.0), title=None):
    """Which ommatidia the sensor actually reaches, and which it leaves blind."""
    with style() as plt:
        fig, ax = _fig(ax, figsize)
        az, el = np.degrees(retina.azimuth), np.degrees(retina.elevation)
        s = _marker_size(fig, ax, az, el, figsize)
        ax.scatter(az[~resampler.covered], el[~resampler.covered], s=s, marker="h",
                   c=PALETTE["line"], edgecolors="none", label="no sensor data")
        ax.scatter(az[resampler.covered], el[resampler.covered], s=s, marker="h",
                   c=PALETTE["ok"], alpha=.85, edgecolors="none", label="covered")
        ax.set_aspect("equal"); ax.grid(False); _clean(ax)
        ax.set_xlabel("azimuth (deg)"); ax.set_ylabel("elevation (deg)")
        ax.legend(fontsize=7, loc="lower right", labelcolor=PALETTE["fg"])
        ax.set_title(title or f"{resampler.coverage:.1%} of the eye covered "
                              f"({resampler.shape[1]}x{resampler.shape[0]})",
                     loc="left", pad=9)
        fig.tight_layout()
    return fig


def sample(retina, resampler, image, figsize=(9.0, 4.2), title=None, cmap="gray"):
    """Camera frame beside what the fly's eye makes of it."""
    with style() as plt:
        fig = plt.figure(figsize=figsize)
        a = fig.add_axes([0.02, 0.04, 0.40, 0.82])
        b = fig.add_axes([0.50, 0.04, 0.40, 0.82])
        a.imshow(image, cmap=cmap, interpolation="nearest")
        a.set_title(f"sensor · {image.shape[1]}x{image.shape[0]}", loc="left",
                    pad=8, fontsize=9)
        a.axis("off")
        vals = resampler.apply(image)
        lattice(retina, values=vals, ax=b, cmap=cmap + ("_r" if cmap == "gray" else ""),
                title=f"eye · {retina.n} ommatidia, {resampler.coverage:.0%} covered")
        b.set_xlabel(""); b.set_ylabel("")
        b.set_xticks([]); b.set_yticks([])
        for sp in b.spines.values():
            sp.set_visible(False)
        if title:
            fig.suptitle(title, x=0.02, y=0.98, ha="left", fontsize=10.5)
    return fig


# --------------------------------------------------------------------------- #
# circuits

def ring(theta, rates, ax=None, figsize=(4.6, 4.6), title=None, label=None):
    """Activity on a circular ring, e.g. the compass bump across EPG neurons."""
    with style() as plt:
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(projection="polar")
        else:
            fig = ax.figure
        th = np.asarray(theta); r = np.asarray(rates, dtype=float)
        o = np.argsort(th)
        th, r = th[o], r[o]
        thc = np.concatenate([th, th[:1]]); rc = np.concatenate([r, r[:1]])
        ax.plot(thc, rc, color=PALETTE["accent"], lw=1.8, label=label)
        ax.fill(thc, rc, color=PALETTE["accent"], alpha=.13)
        tot = r.sum()
        if tot > 0:
            z = (r * np.exp(1j * th)).sum() / tot
            ax.annotate("", xy=(np.angle(z), np.abs(z) * r.max()), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=PALETTE["fg"], lw=1.4))
            ax.set_title(title or f"R = {np.abs(z):.3f}", loc="left", pad=14)
        ax.set_yticklabels([]); ax.grid(alpha=.4)
    return fig


def trace(t, series: dict, ax=None, figsize=(7.2, 3.4), xlabel="time (ms)",
          ylabel="", title=None, vlines=None):
    """Time series, one line per key."""
    with style() as plt:
        fig, ax = _fig(ax, figsize)
        for (name, y), col in zip(series.items(), SERIES):
            ax.plot(t, y, color=col, lw=1.6, label=name)
        for v, lab, col in (vlines or []):
            ax.axvline(v, color=col, lw=1.0, ls=(0, (4, 3)))
            if lab:
                ax.text(v, ax.get_ylim()[1], f" {lab}", color=col, fontsize=7,
                        va="top")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, loc="left", pad=9)
        if len(series) > 1:
            ax.legend(fontsize=7.5, labelcolor=PALETTE["fg"])
        _clean(ax); fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# analysis figures

def noise_curve(model, figsize=(7.2, 5.4)):
    """Bilateral reproducibility and the posterior it implies."""
    with style() as plt:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                     gridspec_kw={"height_ratios": [1.35, 1],
                                                  "hspace": .14})
        lo = np.array([b[0] for b in model.bins])
        n = np.array([b[2] for b in model.bins])
        obs = np.array([b[3] for b in model.bins])
        p = np.array([float(model.p_real(np.array([l]))[0]) for l in lo])

        a1.axhline(model.ceiling, color=PALETTE["ok"], lw=1, ls=(0, (4, 3)))
        a1.axhline(model.chance, color=PALETTE["bad"], lw=1, ls=(0, (4, 3)))
        a1.text(1.05, model.ceiling + .014, f"ceiling {model.ceiling:.3f}  (a real pathway)",
                color=PALETTE["ok"], fontsize=7.5)
        a1.text(1.05, model.chance + .014, f"chance {model.chance:.3f}  (post-type shuffled)",
                color=PALETTE["bad"], fontsize=7.5)
        a1.plot(lo, obs, color=PALETTE["accent"], lw=1.8, marker="o", ms=4, zorder=3)
        a1.fill_between(lo, model.chance, obs, color=PALETTE["accent"], alpha=.07)
        a1.set_ylim(.20, 1.05); a1.set_ylabel("reproduced on the\nother hemisphere")
        a1.set_title(f"bilateral reproducibility measures which edges are real  ·  "
                     f"{model.n_edges:,} ipsilateral typed edges", loc="left", pad=10)
        _clean(a1)

        a2.plot(lo, p, color=PALETTE["fg"], lw=1.8, marker="o", ms=4, zorder=3)
        a2.axhline(.95, color=PALETTE["faint"], lw=.9, ls=":")
        w95 = model.min_weight_for(.95)
        a2.axvline(w95, color=PALETTE["ok"], lw=1, ls=(0, (4, 3)))
        a2.text(w95 * 1.16, .655,
                f"P≥0.95 ⇒ weight ≥ {w95}\ndrops {model.cost_of(.95)['frac_dropped']:.0%} of edges",
                color=PALETTE["ok"], fontsize=7.5)
        a2.scatter([1], [p[0]], s=62, facecolor="none", edgecolor=PALETTE["bad"], lw=1.5,
                   zorder=4)
        a2.text(1.35, p[0] - .05,
                f"1 synapse → P(real)={p[0]:.3f}\n{n[0]/1e6:.1f}M edges, ~1 in 5 spurious",
                color=PALETTE["bad"], fontsize=7.5)
        a2.set_xscale("log"); a2.set_xlabel("synapse count on the edge")
        a2.set_ylabel("P(real | w)"); a2.set_ylim(.55, 1.04)
        a2.set_xticks([1, 2, 3, 5, 10, 20, 50, 100])
        a2.set_xticklabels(["1", "2", "3", "5", "10", "20", "50", "100+"])
        _clean(a2)
    return fig


def compression(data: dict, figsize=(9.4, 3.9)):
    """Size vs fidelity, and where polarity breaks. ``data`` from artifacts/pareto.json."""
    with style() as plt:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=figsize,
                                      gridspec_kw={"wspace": .26})
        for (key, d), col in zip(data.items(), SERIES):
            ax.plot(d["kb"], d["err"], color=col, lw=1.5, marker="o", ms=3.6, label=key)
            ax2.plot(d["p"], d["agree"], color=col, lw=1.5, marker="o", ms=3.6)
        for kb, name in [(128, "STM32F401"), (1024, "STM32H743"), (2048, "GAP9")]:
            ax.axvline(kb, color=PALETTE["faint"], lw=.8, ls=":")
            ax.text(kb * 1.05, 24, name, color=PALETTE["faint"], fontsize=6.6,
                    rotation=90, va="top")
        ax.set_xscale("log"); ax.set_xlabel("flash on device (KB)")
        ax.set_ylabel("drive error (%)"); ax.set_ylim(0, 26)
        ax.set_title("size vs fidelity", loc="left", pad=8)
        ax.legend(fontsize=6.6, loc="upper right", labelcolor=PALETTE["fg"])
        ax2.set_xlabel("P(real) threshold"); ax2.set_ylabel("sign agreement (%)")
        ax2.set_ylim(55, 102)
        ax2.set_title("where the polarity breaks", loc="left", pad=8)
        _clean(ax); _clean(ax2)
    return fig


def waist(stages: list, figsize=(7.2, 3.4), title=None):
    """Neuron counts through the sensory-to-motor hierarchy."""
    with style() as plt:
        fig, ax = plt.subplots(figsize=figsize)
        xs = np.arange(len(stages))
        vals = np.array([s[1] for s in stages])
        ax.bar(xs, vals, width=.56, zorder=3,
               color=[PALETTE["accent"] if v > 5000 else PALETTE["ok"] for v in vals])
        for x, (lab, v) in zip(xs, stages):
            ax.text(x, v * 1.32, f"{v:,}", ha="center", color=PALETTE["fg"], fontsize=9)
        ax.set_yscale("log"); ax.set_xticks(xs)
        ax.set_xticklabels([s[0] for s in stages], fontsize=7.5, color=PALETTE["fg"])
        ax.set_ylim(300, 2.4e5); ax.set_ylabel("neurons")
        ax.set_title(title or "the fly narrows to a 1,314-wire bus, then 708 actuator lines",
                     loc="left", pad=10)
        ax.grid(axis="x", visible=False); _clean(ax)
        fig.tight_layout()
    return fig


def bump_sweep(sweep: dict, figsize=(7.2, 3.7)):
    """The compass gain sweep: forms a bump, never holds it."""
    with style() as plt:
        fig, ax = plt.subplots(figsize=figsize)
        tr = sweep["trials"]
        xs = [t["weight_scale"] for t in tr]
        ax.axhspan(0.45, 0.85, color=PALETTE["ok"], alpha=.08)
        ax.text(min(xs) * 1.05, .83, "measured bump width band\n(Seelig & Jayaraman 2015)",
                color=PALETTE["ok"], fontsize=7, va="top")
        ax.plot(xs, [t["R_driven"] for t in tr], color=PALETTE["accent"], lw=1.8,
                marker="o", ms=4.5, label="under drive", zorder=3)
        ax.plot(xs, [t["R_held"] for t in tr], color=PALETTE["bad"], lw=1.8,
                marker="o", ms=4.5, label="after input removed", zorder=3)
        ax.annotate("R = 0.000 everywhere", xy=(0.35, 0.0), xytext=(0.5, 0.22),
                    color=PALETTE["bad"], fontsize=7.5,
                    arrowprops=dict(arrowstyle="->", color=PALETTE["bad"], lw=1))
        ax.set_xscale("log"); ax.set_xlabel("global synaptic scale  (×250 range swept)")
        ax.set_ylabel("bump concentration  R"); ax.set_ylim(-.04, 1.0)
        ax.set_title("the compass forms a bump and cannot hold it, at any global gain",
                     loc="left", pad=10)
        ax.legend(fontsize=7.5, labelcolor=PALETTE["fg"], loc="center right")
        _clean(ax); fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# form factors

#: Nominal limb positions per morphology, purely for drawing. (x, y) in body frame.
_LAYOUT = {
    "hexapod":   {"L1": (-.55, .85), "R1": (.55, .85), "L2": (-.7, 0), "R2": (.7, 0),
                  "L3": (-.55, -.85), "R3": (.55, -.85)},
    "quadruped": {"FL": (-.6, .7), "FR": (.6, .7), "HL": (-.6, -.7), "HR": (.6, -.7)},
    "biped":     {"L": (-.4, 0), "R": (.4, 0)},
}
_SEG_COLOR = {"front": PALETTE["accent"], "middle": PALETTE["cool"],
              "hind": PALETTE["ok"], "wing": PALETTE["violet"]}


def morphology(morph, plan=None, ax=None, figsize=(4.6, 4.6), title=None):
    """Draw a robot's limbs, coloured by which fly leg circuit drives each one.

    Phase is printed on each limb: two limbs sharing a phase step together. For a
    hexapod tripod that is the classic two alternating triangles, and you can read it
    straight off the picture.
    """
    with style() as plt:
        fig, ax = _fig(ax, figsize)
        lay = _LAYOUT.get(morph.name)
        if lay is None:                      # URDF or modular: ring them
            n = max(morph.n_limbs, 1)
            lay = {l.name: (0.8 * np.cos(2 * np.pi * i / n + np.pi / 2),
                            0.8 * np.sin(2 * np.pi * i / n + np.pi / 2))
                   for i, l in enumerate(morph.limbs)}
        bind = {b.limb: b for b in (plan.bindings if plan else [])}

        ax.add_patch(plt.Rectangle((-.28, -.95), .56, 1.9, fc=PALETTE["panel"],
                                   ec=PALETTE["line"], lw=1.2, zorder=1))
        for name, (x, y) in lay.items():
            b = bind.get(name)
            col = _SEG_COLOR.get(b.source_leg, PALETTE["faint"]) if b else PALETTE["faint"]
            ax.plot([0, x], [y * .55, y], color=col, lw=2.4, zorder=2,
                    solid_capstyle="round")
            ax.scatter([x], [y], s=190, color=col, zorder=3, edgecolors=PALETTE["bg"],
                       linewidths=1.6)
            lab = name if b is None else f"{name}\n{b.phase:.2f}"
            ax.annotate(lab, (x, y), color=PALETTE["fg"], fontsize=7,
                        ha="center", va="center", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.18", fc=PALETTE["bg"],
                                  ec="none", alpha=.82))
        if plan:
            seen = []
            for b in plan.bindings:
                if b.source_leg not in seen:
                    seen.append(b.source_leg)
            for i, seg in enumerate(seen):
                ax.scatter([], [], s=70, color=_SEG_COLOR.get(seg, PALETTE["faint"]),
                           label=f"fly {seg} leg")
            ax.legend(fontsize=6.8, loc="upper center", labelcolor=PALETTE["fg"],
                      bbox_to_anchor=(0.5, -0.02), ncol=len(seen), handletextpad=.3,
                      columnspacing=1.1)
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.3, 1.15)
        ax.set_aspect("equal"); ax.axis("off")
        t = title or f"{morph.name} \u00b7 {morph.n_joints} joints @ {morph.control_hz:g} Hz"
        if plan:
            t += f" \u00b7 {plan.gait}"
        ax.set_title(t, loc="left", pad=8, fontsize=9.5)
    return fig


def recipes(reports: dict, figsize=(9.6, 4.4)):
    """Every form factor on one chart: what it costs and what it costs you."""
    with style() as plt:
        fig, (a, b) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={"wspace": .3})
        names = list(reports)
        kb = [reports[n].flash_kb for n in names]
        err = [reports[n].drive_err * 100 for n in names]
        agree = [reports[n].sign_agreement * 100 for n in names]
        y = np.arange(len(names))[::-1]

        cols = [PALETTE["ok"] if reports[n].fits_target else PALETTE["bad"] for n in names]
        a.barh(y, kb, color=cols, height=.6, zorder=3)
        for yy, k, n in zip(y, kb, names):
            a.text(k * 1.06, yy, f"{k:,.0f} KB", va="center", fontsize=7.5,
                   color=PALETTE["fg"])
        for x, lab in [(512, "512 KB"), (2048, "2 MB")]:
            a.axvline(x, color=PALETTE["faint"], lw=.8, ls=":")
            a.text(x, len(names) - .3, lab, fontsize=6.6, color=PALETTE["faint"],
                   rotation=90, va="top")
        a.set_yticks(y); a.set_yticklabels(names, fontsize=8)
        a.set_xscale("log"); a.set_xlim(20, 9000)
        a.set_xlabel("flash on device (KB)")
        a.set_title("what each form factor costs", loc="left", pad=8)
        a.grid(axis="y", visible=False); _clean(a)

        b.scatter(err, agree, s=90, c=cols, zorder=3, edgecolors=PALETTE["bg"], lw=1.4)
        for e, g, n in zip(err, agree, names):
            b.annotate(n, (e, g), fontsize=6.8, color=PALETTE["dim"],
                       xytext=(5, 4), textcoords="offset points")
        b.axhline(95, color=PALETTE["faint"], lw=.8, ls=":")
        b.text(0.4, 95.6, "95% sign agreement", fontsize=6.8, color=PALETTE["faint"])
        b.set_xlabel("drive error (%)"); b.set_ylabel("sign agreement (%)")
        b.set_title("what it costs you", loc="left", pad=8)
        _clean(b)
    return fig
