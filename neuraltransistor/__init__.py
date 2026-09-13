"""neuraltransistor -- compile fly connectome circuits into quantized robot controllers.

    import neuraltransistor as nt

    conn = nt.load()                      # cached CSR index over male-CNS v1.0
    ir   = nt.circuit(conn, "compass")    # 452 neurons, 54,290 edges
    ir, _   = nt.prune(ir, p_real=0.95)   # keep edges 95% likely to be real
    ir, rep = nt.quantize(ir, bits=8)     # log codebook + delta index
    nt.emit(ir, "out/")                   # freestanding C for any MCU

Everything is measured rather than estimated: `rep` carries the drive error, the
correlation and the sign agreement that compression actually cost.

Circuits are named for what they do -- `compass`, `collision`, `legs`, `commands` --
and `nt.circuits()` lists them. The anatomical names this library used first
(`legs_all`, `gate_readout`, `optic_motion`, ...) still resolve.
"""

from neuraltransistor.circuit import noise
from neuraltransistor.circuit.extract import (ALIASES, LIBRARY, extract,
                                              from_library, resolve)
from neuraltransistor.circuit.select import Sel, Selector
from neuraltransistor.circuit.sign import assign_signs
from neuraltransistor.data.source import Connectome
from neuraltransistor.ir.graph import CircuitIR, Dynamics, Port
from neuraltransistor.ir.runtime import Reference
from neuraltransistor.quant.quantize import (compress, fit_budget, input_drive,
                                     prune_by_reliability, total_drive)
from neuraltransistor.quant.quantize import prune as prune_weights
from neuraltransistor.target.devices import DEVICES, Device, fit_report
from neuraltransistor.target.mcu_int8 import emit_c

__version__ = "0.1.0"

__all__ = [
    # the short path
    "load", "circuits", "circuit", "prune", "quantize", "budget", "emit",
    "morphology", "retarget", "sensor", "evaluate", "fit",
    # emit targets, by name
    "emit_c", "emit_ros2",
    # types
    "Connectome", "CircuitIR", "Dynamics", "Port", "Reference", "Sel", "Selector",
    # the rest
    "LIBRARY", "ALIASES", "resolve", "extract", "from_library", "assign_signs",
    "compress", "fit_budget", "prune_by_reliability", "prune_weights",
    "input_drive", "total_drive", "noise", "DEVICES", "Device", "fit_report",
    "retina",
]


# --- the short path ---------------------------------------------------------

def load(rebuild: bool = False, verbose: bool = False) -> Connectome:
    """The connectome, from cache. First call builds the index (~75 s)."""
    return Connectome.load(rebuild=rebuild, verbose=verbose)


def circuits() -> dict:
    """Every circuit in the library: ``{name: what it does}``."""
    return {k: v.get("does", v.get("role", "")) for k, v in sorted(LIBRARY.items())}


def circuit(conn: Connectome, which, **kw) -> CircuitIR:
    """A circuit from the library, or pass a Selector for your own.

        ff.circuit(conn, "compass")
        ff.circuit(conn, ff.Sel.type(r"^MBON") | ff.Sel.type(r"^PAM"), name="valence")

    ``which`` is deliberately not called ``name``: a custom selector usually wants to
    pass its own ``name=`` through to extract(), and two parameters called name is a
    TypeError waiting to happen.
    """
    if isinstance(which, Selector):
        kw.setdefault("name", "custom")
        return extract(conn, which, **kw)
    return from_library(conn, which, **kw)


def prune(ir: CircuitIR, p_real: float = 0.95):
    """Drop edges less than ``p_real`` likely to be a real pathway.

    Returns (circuit, stats). The threshold comes from measured bilateral
    reproducibility, not convention -- see neuraltransistor.circuit.noise.
    """
    return prune_by_reliability(ir, p_real)


def quantize(ir: CircuitIR, bits: int = 8, scheme: str = "log", min_weight: int = 1):
    """Quantize weights to ``bits``. Returns (circuit, QuantReport)."""
    return compress(ir, weight_bits=bits, scheme=scheme, prune_min_weight=min_weight)


def budget(ir: CircuitIR, kb: float, max_drive_err: float = 0.10):
    """Fit the circuit into ``kb`` kilobytes, or return None rather than overshoot."""
    return fit_budget(ir, kb, max_drive_err=max_drive_err)


def emit(ir: CircuitIR, outdir: str, target: str = "c", **kw):
    """Compile ``ir`` to ``outdir``.

        nt.emit(ir, "out/")                                  # freestanding C99
        nt.emit(ir, "out/", target="ros2", morph=my_morph)   # a ROS 2 package

    One entry point with a ``target``, matching ``ntx emit --target``, rather than one
    function per backend.
    """
    if target in ("c", "mcu", "int8"):
        return emit_c(ir, outdir, **kw)
    if target == "ros2":
        morph = kw.pop("morph", None) or kw.pop("morphology", None)
        if morph is None:
            raise TypeError("target='ros2' needs morph= (see nt.morphology)")
        return emit_ros2(ir, morph, outdir, **kw)
    raise ValueError(f"unknown target {target!r}; have 'c' and 'ros2'")


def fit(ir: CircuitIR, task, **kw):
    """Fit the dynamics the connectome does not contain. See neuraltransistor.train."""
    from neuraltransistor.train import fit as _f
    return _f(ir, task, **kw)


def emit_ros2(ir: CircuitIR, morph, outdir: str,
              package: str = "neuraltransistor_controller", retarget_plan=None):
    from neuraltransistor.target.ros2 import emit_ros2 as _e
    return _e(ir, morph, outdir, package=package, retarget_plan=retarget_plan)


def morphology(name: str = None, urdf: str = None, **kw):
    """A MorphologySpec, from the builtin library or read from a URDF.

        ff.morphology("hexapod")
        ff.morphology(urdf="my_robot.urdf")     # -> (spec, report)
    """
    from neuraltransistor.morph import spec as _m, urdf as _u
    if urdf:
        return _u.parse(urdf, **kw)
    return _m.BUILTIN[name or "hexapod"](**kw)


def retarget(conn: Connectome, morph, gait: str = None, circuit_name: str = "legs"):
    """Bind a robot's limbs to fly leg circuits and a gait phase."""
    from neuraltransistor.morph import retarget as _r, spec as _m
    ir = from_library(conn, circuit_name)
    decodes = _m.decode_motor(ir, conn.neurons)
    return _r.retarget(morph, decodes, gait=gait)


def sensor(n: int = 886, **kw):
    """A hexagonal ommatidial lattice matching the fly eye.

    ``n`` is the ommatidia count; the fly has ~886 per eye. Pair with
    ``neuraltransistor.sensors.retina.build_resampler`` to map a camera frame onto it.
    """
    from neuraltransistor.sensors.retina import hex_lattice
    return hex_lattice(n_target=n, **kw)


#: Previous name for :func:`sensor`.
retina = sensor


def evaluate(conn: Connectome, circuits=("valence", "compass", "commands"),
             verbose: bool = True):
    """Run the eval suite."""
    from neuraltransistor.evals import suite
    return suite.run_all(conn, circuits=tuple(circuits), verbose=verbose)
