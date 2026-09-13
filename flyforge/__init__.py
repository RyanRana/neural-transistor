"""flyforge -- compile fly connectome circuits into quantized robot controllers.

    import flyforge as ff

    conn = ff.load()                            # cached CSR index over male-CNS v1.0
    ir   = ff.circuit(conn, "compass")          # 452 neurons, 54,290 edges
    ir, _ = ff.prune(ir, p_real=0.95)           # keep edges 95% likely to be real
    ir, rep = ff.quantize(ir, bits=8)           # log codebook + delta index
    ff.emit_c(ir, "out/")                       # freestanding C for any MCU

Everything is measured rather than estimated: `rep` carries the drive error, the
correlation and the sign agreement that compression actually cost.
"""

from flyforge.circuit import noise
from flyforge.circuit.extract import LIBRARY, extract, from_library
from flyforge.circuit.select import Sel, Selector
from flyforge.circuit.sign import assign_signs
from flyforge.data.source import Connectome
from flyforge.ir.graph import CircuitIR, Dynamics, Port
from flyforge.ir.runtime import Reference
from flyforge.quant.quantize import (compress, fit_budget, input_drive,
                                     prune_by_reliability, total_drive)
from flyforge.quant.quantize import prune as prune_weights
from flyforge.target.devices import DEVICES, Device, fit_report
from flyforge.target.mcu_int8 import emit_c

__version__ = "0.1.0"

__all__ = [
    "load", "circuit", "prune", "quantize", "budget", "emit_c", "emit_ros2",
    "morphology", "retarget", "retina", "evaluate",
    "Connectome", "CircuitIR", "Dynamics", "Port", "Reference",
    "Sel", "Selector", "LIBRARY", "extract", "from_library", "assign_signs",
    "compress", "fit_budget", "prune_by_reliability", "prune_weights",
    "input_drive", "total_drive", "noise", "DEVICES", "Device", "fit_report",
]


# --- the short path ---------------------------------------------------------

def load(rebuild: bool = False, verbose: bool = False) -> Connectome:
    """The connectome, from cache. First call builds the index (~75 s)."""
    return Connectome.load(rebuild=rebuild, verbose=verbose)


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
    reproducibility, not convention -- see flyforge.circuit.noise.
    """
    return prune_by_reliability(ir, p_real)


def quantize(ir: CircuitIR, bits: int = 8, scheme: str = "log", min_weight: int = 1):
    """Quantize weights to ``bits``. Returns (circuit, QuantReport)."""
    return compress(ir, weight_bits=bits, scheme=scheme, prune_min_weight=min_weight)


def budget(ir: CircuitIR, kb: float, max_drive_err: float = 0.10):
    """Fit the circuit into ``kb`` kilobytes, or return None rather than overshoot."""
    return fit_budget(ir, kb, max_drive_err=max_drive_err)


def emit_ros2(ir: CircuitIR, morph, outdir: str, package: str = "flyforge_controller"):
    from flyforge.target.ros2 import emit_ros2 as _e
    return _e(ir, morph, outdir, package=package)


def morphology(name: str = None, urdf: str = None, **kw):
    """A MorphologySpec, from the builtin library or read from a URDF.

        ff.morphology("hexapod")
        ff.morphology(urdf="my_robot.urdf")     # -> (spec, report)
    """
    from flyforge.morph import spec as _m, urdf as _u
    if urdf:
        return _u.parse(urdf, **kw)
    return _m.BUILTIN[name or "hexapod"](**kw)


def retarget(conn: Connectome, morph, gait: str = None, circuit_name: str = "legs_all"):
    """Bind a robot's limbs to fly leg circuits and a gait phase."""
    from flyforge.morph import retarget as _r, spec as _m
    ir = from_library(conn, circuit_name)
    decodes = _m.decode_motor(ir, conn.neurons)
    return _r.retarget(morph, decodes, gait=gait)


def retina(n: int = 886, **kw):
    """A hexagonal ommatidial lattice matching the fly eye."""
    from flyforge.sensors.retina import hex_lattice
    return hex_lattice(n_target=n, **kw)


def evaluate(conn: Connectome, circuits=("gate_readout", "compass", "descending"),
             verbose: bool = True):
    """Run the eval suite."""
    from flyforge.evals import suite
    return suite.run_all(conn, circuits=tuple(circuits), verbose=verbose)
