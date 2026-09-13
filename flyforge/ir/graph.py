"""CircuitIR -- the neutral representation everything else compiles through.

The pipeline is deliberately hourglass-shaped:

    connectome  ->  [ CircuitIR ]  ->  int8 MCU / int4 MCU / SNN / ONNX
    morphology  ->       ^
    sensor      ->       |
    fitted dynamics -----+

Everything upstream (which dataset, which neurons, which morphology, which sensor)
and everything downstream (which chip, which bit width) is replaceable. The IR is
the contract in the middle, and it is the only thing a new backend has to understand.

Three decisions in here matter more than the rest:

**Structure and dynamics are separate.** ``weight`` is the synapse count -- an integer
the connectome measured. ``gain``, ``tau``, ``threshold``, ``rest`` are dynamics the
connectome does NOT contain and something else must supply. Keeping them in different
arrays means you can always answer "is this number measured or fitted?", which is the
question that decides whether a result is a finding or a guess.

**Modulatory edges are multiplicative, not additive.** A dopaminergic or octopaminergic
terminal does not add current, it scales the gain of a target population. They live in
their own edge list and compile to a different kernel. This is what makes the fly's
attention/gating behaviour expressible instead of being flattened into more synapses.

**Ports are explicit.** A circuit cut out of a brain has dangling edges. Rather than
pretend otherwise, the boundary is named: ``in_ports`` are afferents whose activity must
be supplied by a sensor adapter, ``out_ports`` efferents a morphology adapter consumes.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

FCX_MAGIC = "flyforge-circuit"
FCX_VERSION = 1


@dataclass
class Port:
    """A named boundary of the circuit."""
    name: str
    neurons: np.ndarray          # dense-local indices into this circuit
    role: str                    # "sensory" | "motor" | "descending" | "ascending" | "modulatory"
    note: str = ""

    def to_meta(self):
        return {"name": self.name, "role": self.role, "note": self.note,
                "n": int(len(self.neurons))}


@dataclass
class Dynamics:
    """Per-neuron dynamics. Every field here is FITTED, never measured.

    Defaults are the leaky-integrate-and-fire values used by Shiu et al. for the
    whole-brain fly model; they are a starting point for the fitter, not a result.
    """
    tau_m: np.ndarray            # membrane time constant, seconds
    threshold: np.ndarray        # spike/activation threshold
    rest: np.ndarray             # resting level
    gain: np.ndarray             # per-neuron output gain
    refractory: np.ndarray       # refractory period, seconds
    synaptic_gain: float = 1.0   # global scale from synapse count -> conductance
    dt: float = 0.0005           # integration step (2 kHz default)

    @staticmethod
    def default(n: int, dt: float = 0.0005) -> "Dynamics":
        return Dynamics(
            tau_m=np.full(n, 0.020, dtype=np.float32),
            threshold=np.full(n, 1.0, dtype=np.float32),
            rest=np.zeros(n, dtype=np.float32),
            gain=np.ones(n, dtype=np.float32),
            refractory=np.full(n, 0.0022, dtype=np.float32),
            synaptic_gain=1.0,
            dt=dt,
        )

    @property
    def n_free_params(self) -> int:
        return sum(a.size for a in (self.tau_m, self.threshold, self.rest,
                                    self.gain, self.refractory)) + 1


@dataclass
class CircuitIR:
    """A subgraph of the connectome, plus everything needed to run it."""

    name: str
    body_ids: np.ndarray         # int64, the published identifier, length n
    types: list[str]
    superclasses: list[str]
    sign: np.ndarray             # int8 in {-1,0,+1}, presynaptic
    sign_kind: np.ndarray        # uint8 0=unknown 1=exc 2=inh 3=modulatory

    # chemical synapses, CSR over presynaptic local index
    indptr: np.ndarray           # int64, n+1
    indices: np.ndarray          # int32, n_edges
    weight: np.ndarray           # uint16, synapse counts (MEASURED)

    # modulatory edges, COO (they are sparse and few)
    mod_pre: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    mod_post: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    mod_weight: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint16))

    dynamics: Optional[Dynamics] = None
    in_ports: list[Port] = field(default_factory=list)
    out_ports: list[Port] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    # -- shape ------------------------------------------------------------------

    @property
    def n_neurons(self) -> int:
        return int(len(self.body_ids))

    @property
    def n_edges(self) -> int:
        return int(len(self.indices))

    @property
    def n_mod_edges(self) -> int:
        return int(len(self.mod_pre))

    @property
    def n_synapses(self) -> int:
        return int(self.weight.sum())

    def degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    def in_degree(self) -> np.ndarray:
        return np.bincount(self.indices, minlength=self.n_neurons)

    # -- accounting -------------------------------------------------------------

    def footprint(self, weight_bits: int = 8, index_bits: int = 16,
                  state_bits: int = 16) -> dict:
        """Bytes on the target, counted the way the device actually stores it.

        A sparse connectome layer is dominated by the INDEX, not the weight -- at int8
        weights with int16 indices, two thirds of the bytes are addressing. Any size
        claim that counts only weights is off by 3x, so this counts both.
        """
        w = self.n_edges * weight_bits / 8
        idx = self.n_edges * index_bits / 8
        ptr = (self.n_neurons + 1) * 4
        state = self.n_neurons * state_bits / 8 * 2      # membrane + refractory timer
        gain = self.n_neurons * weight_bits / 8          # per-neuron gain
        mod = self.n_mod_edges * (weight_bits + 2 * index_bits) / 8
        total = w + idx + ptr + state + gain + mod
        return {
            "weights_B": int(w), "indices_B": int(idx), "indptr_B": int(ptr),
            "state_B": int(state), "gain_B": int(gain), "mod_B": int(mod),
            "total_B": int(total), "total_KB": round(total / 1024, 2),
            "weight_bits": weight_bits, "index_bits": index_bits,
            "flash_B": int(w + idx + ptr + gain + mod),   # read-only, lives in flash
            "ram_B": int(state),                          # mutable, must be in RAM
        }

    def macs_per_tick(self) -> int:
        """One synaptic accumulate per edge per tick, plus per-neuron update."""
        return self.n_edges + self.n_neurons * 4 + self.n_mod_edges

    # -- provenance -------------------------------------------------------------

    def fingerprint(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        for a in (self.body_ids, self.indptr, self.indices, self.weight,
                  self.sign, self.mod_pre, self.mod_post):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def summary(self) -> str:
        f = self.footprint()
        known = float((self.sign_kind != 0).mean())
        return (
            f"{self.name}: {self.n_neurons:,} neurons, {self.n_edges:,} edges, "
            f"{self.n_synapses:,} synapses, {self.n_mod_edges:,} modulatory\n"
            f"  sign known {known:.1%} | in-ports {len(self.in_ports)} "
            f"out-ports {len(self.out_ports)}\n"
            f"  int8 footprint {f['total_KB']} KB "
            f"(flash {f['flash_B']/1024:.1f} KB, ram {f['ram_B']/1024:.1f} KB) | "
            f"{self.macs_per_tick():,} MAC/tick"
        )

    # -- serialization ----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write a .fcx: a zip of npy arrays plus a JSON manifest."""
        path = Path(path)
        arrays = {
            "body_ids": self.body_ids, "sign": self.sign, "sign_kind": self.sign_kind,
            "indptr": self.indptr, "indices": self.indices, "weight": self.weight,
            "mod_pre": self.mod_pre, "mod_post": self.mod_post,
            "mod_weight": self.mod_weight,
        }
        if self.dynamics is not None:
            for k in ("tau_m", "threshold", "rest", "gain", "refractory"):
                arrays[f"dyn_{k}"] = getattr(self.dynamics, k)
        for p in self.in_ports:
            arrays[f"inport_{p.name}"] = p.neurons
        for p in self.out_ports:
            arrays[f"outport_{p.name}"] = p.neurons

        manifest = {
            "magic": FCX_MAGIC, "version": FCX_VERSION, "name": self.name,
            "n_neurons": self.n_neurons, "n_edges": self.n_edges,
            "n_synapses": self.n_synapses, "fingerprint": self.fingerprint(),
            "types": self.types, "superclasses": self.superclasses,
            "in_ports": [p.to_meta() for p in self.in_ports],
            "out_ports": [p.to_meta() for p in self.out_ports],
            "dynamics": ({"synaptic_gain": self.dynamics.synaptic_gain,
                          "dt": self.dynamics.dt} if self.dynamics else None),
            "provenance": self.provenance,
            "footprint_int8": self.footprint(),
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", json.dumps(manifest, indent=2))
            for k, v in arrays.items():
                b = io.BytesIO()
                np.save(b, np.ascontiguousarray(v))
                z.writestr(f"arrays/{k}.npy", b.getvalue())
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CircuitIR":
        path = Path(path)
        with zipfile.ZipFile(path) as z:
            man = json.loads(z.read("manifest.json"))
            if man.get("magic") != FCX_MAGIC:
                raise ValueError(f"{path} is not a flyforge circuit")

            def arr(k):
                return np.load(io.BytesIO(z.read(f"arrays/{k}.npy")))

            dyn = None
            if man.get("dynamics"):
                dyn = Dynamics(
                    tau_m=arr("dyn_tau_m"), threshold=arr("dyn_threshold"),
                    rest=arr("dyn_rest"), gain=arr("dyn_gain"),
                    refractory=arr("dyn_refractory"),
                    synaptic_gain=man["dynamics"]["synaptic_gain"],
                    dt=man["dynamics"]["dt"])
            ir = cls(
                name=man["name"], body_ids=arr("body_ids"),
                types=man["types"], superclasses=man["superclasses"],
                sign=arr("sign"), sign_kind=arr("sign_kind"),
                indptr=arr("indptr"), indices=arr("indices"), weight=arr("weight"),
                mod_pre=arr("mod_pre"), mod_post=arr("mod_post"),
                mod_weight=arr("mod_weight"),
                dynamics=dyn, provenance=man.get("provenance", {}),
            )
            ir.in_ports = [Port(p["name"], arr(f"inport_{p['name']}"), p["role"],
                                p.get("note", "")) for p in man["in_ports"]]
            ir.out_ports = [Port(p["name"], arr(f"outport_{p['name']}"), p["role"],
                                 p.get("note", "")) for p in man["out_ports"]]
        return ir
