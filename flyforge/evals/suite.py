"""Practical evals: does this toolkit produce artifacts that actually work.

These are not benchmarks against a simulated world. Each one asks a question whose
answer decides whether the tool is usable on real hardware, and answers it by running
the real thing:

  c_equivalence     does the emitted C reproduce the reference bit for bit
  compiles          does the emitted C build clean with -Wall -Werror
  throughput        how many ticks per second, measured, on this host
  quant_fidelity    what does each bit width cost in synaptic drive
  budget_fit        which real chips does each circuit fit on
  determinism       does the same selector give the same artifact twice
  roundtrip         does .fcx survive save/load unchanged
  nmj_guard         does the motor-neuron sign correction actually fire
  noise_monotonic   is bilateral reproducibility monotone in synapse count

Every eval returns a pass/fail plus the measurement behind it, so a failure says what
the number was rather than only that it was wrong.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from flyforge.circuit.extract import from_library
from flyforge.circuit.sign import assign_signs, naive_signs
from flyforge.data.source import Connectome
from flyforge.ir.graph import CircuitIR
from flyforge.ir.runtime import Reference
from flyforge.quant.quantize import compress, input_drive
from flyforge.target.mcu_int8 import emit_c

#: Real devices, from datasheets. (sram_KB, flash_KB, clock_MHz, note)
DEVICES = {
    "STM32F411":  (128,   512,  100, "Cortex-M4F, common on nano-quad flight controllers"),
    "STM32H743":  (1024,  2048, 480, "Cortex-M7, DSP + double FPU"),
    "STM32U585":  (786,   2048, 160, "Cortex-M33, ultra-low-power line"),
    "ESP32-S3":   (512,   8192, 240, "Xtensa LX7 dual core, vector extensions"),
    "RP2350":     (520,   4096, 150, "dual Cortex-M33 / RISC-V"),
    "nRF52840":   (256,   1024, 64,  "Cortex-M4F, BLE, very low power"),
    "GAP9":       (1536,  2048, 370, "RISC-V PULP cluster + NE16 accelerator"),
    "Apollo4":    (2048,  2048, 192, "Cortex-M4F, sub-mW class"),
}


def _sha(x) -> str:
    return hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=8).hexdigest()


@dataclass
class EvalResult:
    name: str
    passed: bool
    detail: dict = field(default_factory=dict)
    note: str = ""

    def __str__(self):
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.note}"


# --------------------------------------------------------------------------- #

_HARNESS = r"""
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "%(p)s.h"
static %(p)s_state_t st;
int main(int argc, char **argv) {
    int T = argc > 1 ? atoi(argv[1]) : 64;
    int quiet = argc > 2;   /* timing mode: skip I/O so it measures the kernel */
    static int16_t drive[%(P)s_N_NEURONS];
    for (int i = 0; i < %(P)s_N_NEURONS; ++i)
        drive[i] = (int16_t)((i * 37) %% 400 - 100);
    %(p)s_reset(&st);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int t = 0; t < T; ++t) {
        %(p)s_tick(&st, drive);
        if (quiet) continue;
        uint64_t h = 1469598103934665603ULL;
        long n = 0;
        for (int i = 0; i < %(P)s_N_NEURONS; ++i) {
            h ^= st.fired[i]; h *= 1099511628211ULL; n += st.fired[i];
        }
        printf("%%d %%ld %%llu\n", t, n, (unsigned long long)h);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double el = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    fprintf(stderr, "ELAPSED %%.9f\n", el);
    return 0;
}
"""


def eval_c_equivalence(ir: CircuitIR, ticks: int = 64, weight_bits: int = 8
                       ) -> EvalResult:
    """Emit C, compile it, run it, and diff against the numpy reference."""
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        return EvalResult("c_equivalence", False, note="no C compiler found")
    prefix = ir.name.replace("-", "_")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        emit_c(ir, td, weight_bits=weight_bits, prefix=prefix)
        (td / "harness.c").write_text(_HARNESS % {"p": prefix, "P": prefix.upper()})
        exe = td / "run"
        cp = subprocess.run(
            [cc, "-O2", "-std=c99", "-o", str(exe), str(td / "harness.c"),
             str(td / f"{prefix}.c"), str(td / f"{prefix}_runtime.c"), f"-I{td}"],
            capture_output=True, text=True)
        if cp.returncode != 0:
            return EvalResult("c_equivalence", False,
                              detail={"stderr": cp.stderr[-1500:]},
                              note="compile failed")
        out = subprocess.run([str(exe), str(ticks)], capture_output=True, text=True)
        c_rows = [l.split() for l in out.stdout.strip().splitlines() if l.strip()]

    ref = Reference(ir, weight_bits=weight_bits)
    drive = np.array([((i * 37) % 400) - 100 for i in range(ir.n_neurons)],
                     dtype=np.int64)
    mism, first_bad = 0, None
    for t in range(ticks):
        f = ref.tick(drive)
        n = int(f.sum())
        h = np.uint64(1469598103934665603)
        for b in f:
            h = np.uint64((int(h) ^ int(b)) & 0xFFFFFFFFFFFFFFFF)
            h = np.uint64((int(h) * 1099511628211) & 0xFFFFFFFFFFFFFFFF)
        if t >= len(c_rows):
            mism += 1
            continue
        cn, ch = int(c_rows[t][1]), int(c_rows[t][2])
        if cn != n or ch != int(h):
            mism += 1
            if first_bad is None:
                first_bad = {"tick": t, "c_fired": cn, "ref_fired": n,
                             "c_hash": ch, "ref_hash": int(h)}
    return EvalResult(
        "c_equivalence", mism == 0,
        detail={"ticks": ticks, "mismatched_ticks": mism, "first": first_bad},
        note=f"{ticks - mism}/{ticks} ticks bit-identical to reference")


def eval_compiles(ir: CircuitIR) -> EvalResult:
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        return EvalResult("compiles", False, note="no C compiler")
    prefix = ir.name.replace("-", "_")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        emit_c(ir, td, prefix=prefix)
        cp = subprocess.run(
            [cc, "-c", "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2",
             str(td / f"{prefix}.c"), str(td / f"{prefix}_runtime.c"), f"-I{td}"],
            capture_output=True, text=True, cwd=td)
    ok = cp.returncode == 0
    return EvalResult("compiles", ok, detail={"stderr": cp.stderr[-1200:]},
                      note="clean under -Wall -Wextra -Werror" if ok
                           else "warnings/errors under -Werror")


def eval_throughput(ir: CircuitIR, ticks: int = 200) -> EvalResult:
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        return EvalResult("throughput", False, note="no C compiler")
    prefix = ir.name.replace("-", "_")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        emit_c(ir, td, prefix=prefix)
        (td / "harness.c").write_text(_HARNESS % {"p": prefix, "P": prefix.upper()})
        exe = td / "run"
        subprocess.run([cc, "-O2", "-std=c99", "-o", str(exe), str(td / "harness.c"),
                        str(td / f"{prefix}.c"), str(td / f"{prefix}_runtime.c"),
                        f"-I{td}"], capture_output=True)
        if not exe.exists():
            return EvalResult("throughput", False, note="harness did not build")
        cp = subprocess.run([str(exe), str(ticks), "quiet"],
                            capture_output=True, text=True)
    el = None
    for line in cp.stderr.splitlines():
        if line.startswith("ELAPSED"):
            el = float(line.split()[1])
    if el is None or el <= 0:
        return EvalResult("throughput", False, note="no timing returned")
    tps = ticks / el
    # Per-tick cost normalised by the work actually done, so circuits of different
    # size are comparable and an MCU estimate is a simple clock-ratio away.
    ns_per_edge = 1e9 * el / max(ticks * ir.n_edges, 1)
    return EvalResult("throughput", tps > 0,
                      detail={"ticks_per_s_host": round(tps, 1),
                              "us_per_tick_host": round(1e6 / tps, 2),
                              "ns_per_edge_host": round(ns_per_edge, 3),
                              "edges": ir.n_edges,
                              "realtime_200hz_ok": bool(tps > 200)},
                      note=f"{tps:,.0f} tick/s ({1e6/tps:.1f} us/tick, "
                           f"{ns_per_edge:.2f} ns/edge) on host")


def eval_quant_fidelity(ir: CircuitIR, bits=(8, 4, 2)) -> EvalResult:
    rows = {}
    ok = True
    for b in bits:
        _, rep = compress(ir, weight_bits=b, scheme="log", prune_min_weight=1)
        rows[f"w{b}"] = {"drive_err_mean": round(rep.drive_rel_err_mean, 5),
                         "drive_err_p95": round(rep.drive_rel_err_p95, 5),
                         "sign_flips": rep.drive_sign_flips,
                         "KB": round(rep.bytes_after / 1024, 1)}
        if b == 8 and rep.drive_rel_err_mean > 0.02:
            ok = False
    return EvalResult("quant_fidelity", ok, detail=rows,
                      note=f"int8 drive error {rows['w8']['drive_err_mean']:.3%}, "
                           f"{rows['w8']['sign_flips']} sign flips")


def eval_budget_fit(ir: CircuitIR, weight_bits: int = 8) -> EvalResult:
    with tempfile.TemporaryDirectory() as td:
        rep = emit_c(ir, td, weight_bits=weight_bits)
    fits = {}
    for dev, (sram, flash, mhz, _n) in DEVICES.items():
        fits[dev] = {"flash_ok": rep.flash_B <= flash * 1024,
                     "ram_ok": rep.ram_B <= sram * 1024,
                     "fits": rep.flash_B <= flash * 1024 and rep.ram_B <= sram * 1024}
    n = sum(1 for v in fits.values() if v["fits"])
    return EvalResult("budget_fit", n > 0,
                      detail={"flash_KB": round(rep.flash_B / 1024, 1),
                              "ram_KB": round(rep.ram_B / 1024, 1), "devices": fits},
                      note=f"fits {n}/{len(DEVICES)} devices "
                           f"(flash {rep.flash_B/1024:.0f}KB ram {rep.ram_B/1024:.0f}KB)")


def eval_determinism(conn: Connectome, key: str) -> EvalResult:
    a = from_library(conn, key)
    b = from_library(conn, key)
    same = a.fingerprint() == b.fingerprint()
    return EvalResult("determinism", same,
                      detail={"fingerprint": a.fingerprint()},
                      note="identical fingerprint across two extractions" if same
                           else "extraction is not reproducible")


def eval_roundtrip(ir: CircuitIR) -> EvalResult:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.fcx"
        ir.save(p)
        back = CircuitIR.load(p)
        same = back.fingerprint() == ir.fingerprint()
        size = p.stat().st_size
    return EvalResult("roundtrip", same,
                      detail={"fcx_KB": round(size / 1024, 1),
                              "fingerprint": ir.fingerprint()},
                      note=f".fcx {size/1024:.0f}KB, fingerprint preserved" if same
                           else "fingerprint changed across save/load")


def eval_nmj_guard(conn: Connectome) -> EvalResult:
    """The motor-neuron sign correction must actually change something."""
    mn = conn.neurons[conn.neurons.superclass == "vnc_motor"]
    sign, kind, rep = assign_signs(mn)
    naive = naive_signs(mn)
    disagree = int((sign != naive).sum())
    naive_inh = int((naive == -1).sum())
    fixed_inh = int((sign == -1).sum())
    ok = rep.n_nmj_corrected > 0 and disagree > 0
    return EvalResult("nmj_guard", ok,
                      detail={"motor_neurons": len(mn),
                              "naive_inhibitory": naive_inh,
                              "corrected_inhibitory": fixed_inh,
                              "sign_flips_applied": rep.n_nmj_corrected,
                              "disagreements_with_naive": disagree},
                      note=f"{rep.n_nmj_corrected} motor neurons rescued from a "
                           f"spurious inhibitory sign "
                           f"({naive_inh}->{fixed_inh} inhibitory)")


def eval_noise_monotonic(conn: Connectome) -> EvalResult:
    from flyforge.circuit import noise as _n
    m = _n.get(conn)
    obs = [b[3] for b in m.bins]
    # allow a little wobble in the sparse tail bins
    viol = sum(1 for a, b in zip(obs, obs[1:]) if b < a - 0.005)
    ok = viol == 0 and m.chance < 0.5 < m.ceiling
    return EvalResult("noise_monotonic", ok,
                      detail={"chance": round(m.chance, 4),
                              "ceiling": round(m.ceiling, 4),
                              "violations": viol,
                              "p_real_w1": round(float(m.p_real(np.array([1]))[0]), 4),
                              "curve": [(b[0], b[1], b[2], round(b[3], 4)) for b in m.bins]},
                      note=f"monotone; chance {m.chance:.3f} ceiling {m.ceiling:.3f}, "
                           f"P(real|w=1)={float(m.p_real(np.array([1]))[0]):.3f}")


# --------------------------------------------------------------------------- #

def run_all(conn: Connectome, circuits=("gate_readout", "compass", "descending"),
            verbose: bool = True) -> dict:
    results: list[EvalResult] = []
    results.append(eval_nmj_guard(conn))
    results.append(eval_noise_monotonic(conn))
    results.append(eval_determinism(conn, circuits[0]))
    per_circuit = {}
    for key in circuits:
        ir = from_library(conn, key)
        rs = [eval_roundtrip(ir), eval_compiles(ir), eval_c_equivalence(ir),
              eval_throughput(ir), eval_quant_fidelity(ir), eval_budget_fit(ir)]
        per_circuit[key] = [asdict(r) for r in rs]
        if verbose:
            print(f"\n--- {key} ({ir.n_neurons:,} neurons, {ir.n_edges:,} edges) ---")
            for r in rs:
                print("   ", r)
        results.extend(rs)
    if verbose:
        print()
        for r in results[:3]:
            print("   ", r)
    n_pass = sum(1 for r in results if r.passed)
    summary = {"passed": n_pass, "total": len(results),
               "global": [asdict(r) for r in results[:3]],
               "per_circuit": per_circuit}
    if verbose:
        print(f"\n{n_pass}/{len(results)} evals passed")
    return summary
