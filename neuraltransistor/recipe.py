"""A recipe is one complete build: connectome in, deployable artifacts out.

Everything the toolkit does is a pipeline with the same shape -- pick a circuit, decide
how much of it is real, shrink it to a budget, bind it to a body, emit code. A Recipe
writes that down declaratively so a form factor is a config rather than a script, and so
two robots can be compared on the same page.

    from neuraltransistor.recipe import RECIPES
    report = RECIPES["quadruped-walker"].build(conn, "out/")
    print(report.summary())

Every recipe reports honestly whether it fits its target, and `build` does not fail if it
does not -- an over-budget build is a result, and hiding it would defeat the purpose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class BuildReport:
    recipe: str
    circuit: str
    neurons: int
    edges_full: int
    edges_kept: int
    synapses_kept_frac: float
    flash_b: int
    ram_b: int
    weight_bits: int
    p_real: float
    drive_err: float
    drive_corr: float
    sign_agreement: float
    target: str
    fits_target: bool
    fits_devices: list
    morphology: str
    joints: int
    joints_driven: int
    control_hz: float
    macs_per_tick: int
    files: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def flash_kb(self) -> float:
        return round(self.flash_b / 1024, 1)

    @property
    def ram_kb(self) -> float:
        return round(self.ram_b / 1024, 1)

    def summary(self) -> str:
        ok = "fits" if self.fits_target else "DOES NOT FIT"
        lines = [
            f"{self.recipe}",
            f"  circuit    {self.circuit}: {self.neurons:,} neurons, "
            f"{self.edges_kept:,}/{self.edges_full:,} edges kept "
            f"({self.synapses_kept_frac:.0%} of synapses)",
            f"  body       {self.morphology}: {self.joints_driven}/{self.joints} joints "
            f"driven @ {self.control_hz:g} Hz",
            f"  artifact   int{self.weight_bits}, P(real)>={self.p_real}: "
            f"{self.flash_kb} KB flash / {self.ram_kb} KB RAM, "
            f"{self.macs_per_tick:,} MAC/tick",
            f"  fidelity   drive error {self.drive_err:.2%}, r={self.drive_corr:.4f}, "
            f"sign agreement {self.sign_agreement:.1%}",
            f"  target     {self.target}: {ok}  "
            f"(fits {len(self.fits_devices)} devices)",
        ]
        lines += [f"  note       {n}" for n in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["flash_kb"] = self.flash_kb
        d["ram_kb"] = self.ram_kb
        return d


@dataclass
class Recipe:
    """One robot, end to end."""
    name: str
    what: str                       # one line: what this robot does
    circuit: str
    morphology: str = "hexapod"     # a BUILTIN name, or a path to a .urdf
    gait: Optional[str] = None
    sensor: str = "none"            # "imu" | "camera" | "event" | "proprioception" | "none"
    target: str = "STM32H743"
    p_real: float = 0.95
    weight_bits: int = 8
    budget_kb: Optional[float] = None
    control_hz: Optional[float] = None
    notes: list = field(default_factory=list)

    def build(self, conn, outdir: str | Path = "out", emit: bool = True,
              verbose: bool = False) -> BuildReport:
        import tempfile
        from neuraltransistor.circuit.extract import from_library
        from neuraltransistor.morph import retarget as R, spec as M, urdf as U
        from neuraltransistor.quant.quantize import compress, prune_by_reliability
        from neuraltransistor.target.devices import DEVICES
        from neuraltransistor.target.mcu_int8 import emit_c
        from neuraltransistor.target.ros2 import emit_ros2

        outdir = Path(outdir) / self.name
        notes = list(self.notes)

        from neuraltransistor.quant.quantize import (drive_error, input_drive,
                                                     total_drive)
        ir = from_library(conn, self.circuit)
        full_edges = ir.n_edges
        small, stats = prune_by_reliability(ir, self.p_real)
        small, rep = compress(small, weight_bits=self.weight_bits)
        # Report fidelity against the ORIGINAL circuit, not against the already-pruned
        # one. compress() measures only the quantization step; the honest number for a
        # deployment is what pruning AND quantizing together cost.
        err = drive_error(input_drive(ir), input_drive(small), total_drive(ir))

        # body
        if str(self.morphology).endswith(".urdf"):
            morph, urep = U.parse(self.morphology,
                                  control_hz=self.control_hz or 200.0)
            notes.append(f"morphology imported from {Path(self.morphology).name}: "
                         f"{urep['n_actuated']} actuated joints")
        else:
            kw = {"hz": self.control_hz} if self.control_hz else {}
            morph = M.BUILTIN[self.morphology](**kw)

        # Only claim driven joints if THIS circuit actually contains motor neurons.
        # Retargeting off legs_all regardless of the circuit would report a compass as
        # driving twelve joints, which it emphatically does not.
        plan = None
        joints_driven = 0
        own_motor = M.decode_motor(small, conn.neurons)
        if own_motor:
            try:
                plan = R.retarget(morph, own_motor, gait=self.gait)
                joints_driven = sum(len(b.joints) for b in plan.bindings)
                if plan.note:
                    notes.append(plan.note)
            except Exception as e:
                notes.append(f"no motor retarget: {type(e).__name__}: {e}")
        else:
            notes.append(f"{self.circuit} contains no motor neurons: this circuit is a "
                         f"sensor or a command stage, and its output port drives "
                         f"whatever consumes it rather than joints directly.")

        with tempfile.TemporaryDirectory() as td:
            er = emit_c(small, td, weight_bits=self.weight_bits)
        dev = DEVICES.get(self.target)
        fits_t = bool(dev and dev.fits(er.flash_B, er.ram_B))
        fits = [d.name for d in DEVICES.values()
                if d.available and d.fits(er.flash_B, er.ram_B)]
        if self.budget_kb and er.flash_B / 1024 > self.budget_kb:
            notes.append(f"over the stated {self.budget_kb:.0f} KB budget by "
                         f"{er.flash_B/1024 - self.budget_kb:.0f} KB")

        files = []
        if emit:
            outdir.mkdir(parents=True, exist_ok=True)
            er = emit_c(small, outdir / "firmware", weight_bits=self.weight_bits)
            files += [str(Path(f).relative_to(outdir)) for f in er.files]
            r2 = emit_ros2(small, morph, outdir / "ros2", retarget_plan=plan)
            files.append(str(Path(r2["package"]).relative_to(outdir)))
            small.save(outdir / f"{self.circuit}.fcx")
            files.append(f"{self.circuit}.fcx")

        report = BuildReport(
            recipe=self.name, circuit=self.circuit, neurons=small.n_neurons,
            edges_full=full_edges, edges_kept=small.n_edges,
            synapses_kept_frac=stats["synapse_retention"],
            flash_b=er.flash_B, ram_b=er.ram_B, weight_bits=self.weight_bits,
            p_real=self.p_real, drive_err=err["mean"],
            drive_corr=err["corr"], sign_agreement=err["sign_agreement"],
            target=self.target, fits_target=fits_t, fits_devices=fits,
            morphology=morph.name, joints=morph.n_joints, joints_driven=joints_driven,
            control_hz=morph.control_hz, macs_per_tick=small.macs_per_tick(),
            files=files, notes=notes)
        if emit:
            (outdir / "build.json").write_text(json.dumps(report.to_dict(), indent=2))
        if verbose:
            print(report.summary())
        return report


#: Worked form factors. Each is a real build with measured numbers, not an illustration.
RECIPES = {
    "hexapod-walker": Recipe(
        name="hexapod-walker",
        what="Six legs, tripod gait. The fly's native layout, so no retargeting loss.",
        circuit="legs_all", morphology="hexapod", gait="tripod",
        sensor="proprioception", target="ESP32-S3", p_real=0.95, control_hz=200,
        notes=["All six leg circuits bind, including the intersegmental interneurons "
               "that coordinate them."]),

    "quadruped-walker": Recipe(
        name="quadruped-walker",
        what="Four legs, trot. Middle-leg circuits still run and still shape phase; "
             "only their motor output goes unused.",
        circuit="legs_all", morphology="quadruped", gait="trot",
        sensor="proprioception", target="ESP32-S3", p_real=0.97, control_hz=200),

    "biped-walker": Recipe(
        name="biped-walker",
        what="Two legs from the hind-leg pair, which are the fly's propulsive legs.",
        circuit="leg_T3", morphology="biped", gait="alternate",
        sensor="proprioception", target="STM32H743", p_real=0.95, control_hz=200),

    "microuav-collision": Recipe(
        name="microuav-collision",
        what="Collision avoidance for a 27 g quadrotor. Camera to escape command, "
             "no training data anywhere in the loop.",
        circuit="looming", morphology="winged", sensor="event",
        target="STM32F405", p_real=0.97, control_hz=250,
        notes=["STM32F405 is the Crazyflie 2.x flight controller -- the only MCU in "
               "the device table proven airborne on a sub-30 g robot.",
               "The full retina-to-DNp01 pathway is `looming_pathway` and is far "
               "larger; this recipe compiles the lobula stage only."]),

    "heading-hold": Recipe(
        name="heading-hold",
        what="GPS-denied heading. An IMU drives the compass; the bump is the estimate.",
        circuit="compass", morphology="quadruped", sensor="imu",
        target="STM32F401", p_real=0.95, control_hz=200,
        notes=["Requires fitted dynamics: unfitted, the bump forms and does not "
               "persist. See docs/FINDINGS.md section 3."]),

    "valence-gate": Recipe(
        name="valence-gate",
        what="The smallest useful thing here: 97 MBONs plus their dopaminergic gate, "
             "as a learned good/bad signal that modulates everything downstream.",
        circuit="gate_readout", morphology="modular", sensor="none",
        target="nRF52840", p_real=0.90, weight_bits=8, control_hz=100,
        notes=["Kilobyte-scale. Fits every device in the table, including the "
               "96 KB-SRAM STM32F401."]),
}


def build_all(conn, outdir: str | Path = "out", emit: bool = False,
              verbose: bool = True) -> dict:
    out = {}
    for name, r in RECIPES.items():
        out[name] = r.build(conn, outdir, emit=emit)
        if verbose:
            print(out[name].summary(), "\n")
    return out
