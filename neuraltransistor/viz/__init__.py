"""Plots for looking at what the toolkit is doing.

Light theme by default, because these end up in READMEs and on screens that are usually
white. Every function returns the matplotlib Figure so you can keep editing it, and takes
``ax=`` so you can compose panels.

    import neuraltransistor as ff
    from neuraltransistor import viz

    r = ff.Retina.from_connectome(ff.load())
    viz.lattice(r).savefig("eye.png")
"""

from neuraltransistor.viz.plot import (PALETTE, bump_sweep, compression, coverage,
                               lattice, morphology, noise_curve, recipes, ring,
                               sample, style, trace, waist)

__all__ = ["style", "PALETTE", "lattice", "coverage", "sample", "ring", "trace",
           "noise_curve", "compression", "waist", "bump_sweep", "morphology",
           "recipes"]
