"""neuraltransistor command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _conn(args):
    from neuraltransistor.data.source import Connectome
    return Connectome.load(rebuild=getattr(args, "rebuild", False), verbose=True)


def cmd_list(args):
    from neuraltransistor.circuit.extract import LIBRARY
    print(f"{'circuit':18s} {'does':26s} what you get")
    print("-" * 100)
    for k, v in LIBRARY.items():
        print(f"{k:18s} {v.get('role', ''):26s} {v.get('does', v['doc'])}")
    if args.biology:
        print("\n--- biological detail ---")
        for k, v in LIBRARY.items():
            print(f"\n{k}\n  {v['doc']}")


def cmd_index(args):
    c = _conn(args)
    m = c.meta
    print(f"{m.dataset} minconf {m.minconf}")
    print(f"  neurons            {m.n_neurons:,}")
    print(f"  edges     total    {m.n_edges_total:,}")
    print(f"            retained {m.n_edges_retained:,} ({m.retained:.1%})")
    print(f"  synapses  total    {m.n_synapses_total:,}")
    print(f"            retained {m.n_synapses_retained:,} ({m.retained_synapses:.1%})")
    print(f"  max weight         {m.max_weight:,}")
    print(f"  built in           {m.build_seconds}s")


def cmd_noise(args):
    from neuraltransistor.circuit import noise
    c = _conn(args)
    m = noise.get(c, rebuild=args.rebuild, verbose=True)
    print(m.table())
    print()
    for p in (0.90, 0.95, 0.99, 0.999):
        d = m.cost_of(p)
        print(f"  P(real) >= {p:<6} -> keep weight >= {d['min_weight']:<3} "
              f"drops {d['frac_dropped']:.1%} of edges")


def cmd_extract(args):
    from neuraltransistor.circuit.extract import from_library
    c = _conn(args)
    ir = from_library(c, args.circuit)
    print(ir.summary())
    if args.out:
        p = ir.save(args.out)
        print(f"wrote {p} ({Path(p).stat().st_size/1024:.0f} KB)")


def cmd_info(args):
    from neuraltransistor.ir.graph import CircuitIR
    ir = CircuitIR.load(args.fcx)
    print(ir.summary())
    print("  provenance:")
    for k, v in ir.provenance.items():
        print(f"    {k}: {v}")


def cmd_compress(args):
    from neuraltransistor.circuit.extract import from_library
    from neuraltransistor.ir.graph import CircuitIR
    from neuraltransistor.quant.quantize import compress, fit_budget, prune_by_reliability
    ir = (CircuitIR.load(args.circuit) if args.circuit.endswith(".fcx")
          else from_library(_conn(args), args.circuit))
    if args.p_real:
        ir, stats = prune_by_reliability(ir, args.p_real)
        print(f"reliability prune p>={args.p_real}: weight>={stats['min_weight']}, "
              f"edges {stats['edge_retention']:.1%} synapses {stats['synapse_retention']:.1%}")
    if args.budget_kb:
        out, rep, trials = fit_budget(ir, args.budget_kb, max_drive_err=args.max_err)
        if out is None:
            print(f"nothing fits {args.budget_kb} KB at drive error <= {args.max_err:.0%}")
            best = min(trials, key=lambda t: t.bytes_after)
            print(f"  smallest reachable: {best}")
            sys.exit(1)
        print(rep)
        ir = out
    else:
        ir, rep = compress(ir, weight_bits=args.bits, prune_min_weight=args.min_weight)
        print(rep)
    if args.out:
        print(f"wrote {ir.save(args.out)}")


def cmd_emit(args):
    from neuraltransistor.circuit.extract import from_library
    from neuraltransistor.ir.graph import CircuitIR
    ir = (CircuitIR.load(args.circuit) if args.circuit.endswith(".fcx")
          else from_library(_conn(args), args.circuit))
    if args.target == "mcu":
        from neuraltransistor.target.mcu_int8 import emit_c
        r = emit_c(ir, args.out, weight_bits=args.bits)
        print(r)
        for f in r.files:
            print("   ", f)
    elif args.target == "ros2":
        from neuraltransistor.morph import spec as M, urdf
        from neuraltransistor.target.ros2 import emit_ros2
        if args.urdf:
            morph, rep = urdf.parse(args.urdf)
            print(f"urdf: {morph} ({rep['n_actuated']} actuated joints)")
        else:
            morph = M.BUILTIN[args.morph]()
        plan = None
        try:
            from neuraltransistor.morph import retarget as _R, spec as _M
            conn = _conn(args)
            legs = from_library(conn, "legs")
            plan = _R.retarget(morph, _M.decode_motor(legs, conn.neurons), gait=args.gait)
        except Exception as e:
            print(f"  (no motor retarget: {type(e).__name__}: {e})")
        r = emit_ros2(ir, morph, args.out, retarget_plan=plan)
        print(f"ros2 package: {r['package']} "
              f"({r['joints_driven']}/{r['joints']} joints driven @ {r['control_hz']}Hz)")
        for f in r["files"]:
            print("   ", f)


def cmd_morph(args):
    from neuraltransistor.circuit.extract import from_library
    from neuraltransistor.morph import spec as M, retarget as R, urdf
    c = _conn(args)
    ir = from_library(c, "legs")
    d = M.decode_motor(ir, c.neurons)
    if args.urdf:
        morph, rep = urdf.parse(args.urdf)
        print(json.dumps({k: v for k, v in rep.items() if k != "limbs"}, indent=1))
    else:
        morph = M.BUILTIN[args.morph]()
    print(morph)
    print(M.coverage(d, morph))
    rt = R.retarget(morph, d, gait=args.gait)
    print(rt.summary())
    if rt.note:
        print("note:", rt.note)


def cmd_eval(args):
    from neuraltransistor.evals import suite
    c = _conn(args)
    r = suite.run_all(c, circuits=tuple(args.circuits))
    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=1, default=str))
        print(f"wrote {args.out}")
    sys.exit(0 if r["passed"] == r["total"] else 1)


_TICK, _CROSS, _DOT = "\u2713", "\u2717", "\u00b7"


def _say(step, msg):
    print(f"\n[{step}] {msg}")


def cmd_doctor(args):
    """Tell the user exactly what is and is not ready, and what to do about it."""
    import shutil
    from neuraltransistor.data import fetch
    from neuraltransistor.data.source import cache_dir, data_dir
    ok = True
    print("neural transistor \u2014 environment check\n")

    print(f"  {_TICK} python {sys.version.split()[0]}")
    for mod in ("numpy", "pandas", "pyarrow", "scipy"):
        try:
            m = __import__(mod)
            print(f"  {_TICK} {mod} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"  {_CROSS} {mod} missing        \u2192 uv pip install -e .")
            ok = False
    for mod, why in (("matplotlib", "plots (ntx demo)"),):
        try:
            __import__(mod)
            print(f"  {_TICK} {mod}  ({why})")
        except ImportError:
            print(f"  {_DOT} {mod} missing    \u2192 optional, needed for {why}")
    cc = shutil.which("cc") or shutil.which("gcc")
    print(f"  {_TICK if cc else _DOT} C compiler {cc or 'not found (only needed to build emitted C)'}")

    print(f"\n  data directory: {data_dir()}")
    st = fetch.status(data_dir())
    for name, f in st["files"].items():
        if f["complete"]:
            print(f"  {_TICK} {name[:56]:56s} {f['expect']/1e6:8.1f} MB")
        else:
            have = f"{f['have']/1e6:.1f}/{f['expect']/1e6:.1f} MB" if f["present"] else "absent"
            print(f"  {_CROSS} {name[:56]:56s} {have}")
            ok = False
    if not st["ready"]:
        print(f"\n  \u2192 run:  ntx quickstart      (downloads "
              f"{st['missing_bytes']/1e9:.1f} GB, public, no account)")

    idx = cache_dir() / "index-male-cns-v1.0-minconf-0.5.npz"
    if idx.exists():
        print(f"\n  {_TICK} index built  ({idx.stat().st_size/1e6:.0f} MB at {idx.parent})")
    else:
        print(f"\n  {_CROSS} index not built     \u2192 ntx index   (about 75 s, once)")
        ok = False
    print(f"\n{'everything is ready.' if ok else 'not ready yet \u2014 see the arrows above.'}")
    sys.exit(0 if ok else 1)


def cmd_quickstart(args):
    """Go from a fresh clone to a compiled circuit in one command."""
    import time
    from neuraltransistor.data import fetch
    from neuraltransistor.data.source import data_dir
    t0 = time.time()
    print("neural transistor \u2014 quickstart")
    print("turns the fly connectome into C you can put on a microcontroller.\n")

    _say(1, f"connectome data \u2192 {data_dir()}")
    st = fetch.status(data_dir())
    if st["ready"]:
        print("  already here, skipping the download")
    else:
        print(f"  need {st['missing_bytes']/1e9:.1f} GB. Public release data, no account, resumable.")
        if not args.yes:
            r = input("  download now? [Y/n] ").strip().lower()
            if r and r not in ("y", "yes"):
                print("  stopped. Set NTX_DATA to a directory that already has the "
                      "three .feather files, or rerun with --yes.")
                sys.exit(1)
        fetch.ensure(data_dir())

    _say(2, "building the index (one time, about 75 s)")
    from neuraltransistor.data.source import Connectome
    conn = Connectome.load(verbose=True)
    m = conn.meta
    print(f"  {m.n_neurons:,} neurons, {m.n_edges_retained:,} edges, "
          f"{m.n_synapses_retained:,} synapses")

    _say(3, "measuring which connections are real (bilateral reproducibility)")
    from neuraltransistor.circuit import noise
    nm = noise.get(conn, verbose=False)
    print(f"  chance {nm.chance:.3f}, ceiling {nm.ceiling:.3f} \u2192 "
          f"P(real | 1 synapse) = {float(nm.p_real(__import__('numpy').array([1]))[0]):.3f}")
    print(f"  asking for 95% confidence means keeping edges of "
          f"{nm.min_weight_for(0.95)}+ synapses")

    _say(4, f"extracting the {args.circuit} circuit")
    from neuraltransistor.circuit.extract import from_library
    ir = from_library(conn, args.circuit)
    for line in ir.summary().splitlines():
        print("  " + line)

    _say(5, "pruning and quantizing to int8")
    from neuraltransistor.quant.quantize import compress, prune_by_reliability
    small, stats = prune_by_reliability(ir, 0.95)
    small, rep = compress(small, weight_bits=8)
    print(f"  kept {stats['edge_retention']:.0%} of edges carrying "
          f"{stats['synapse_retention']:.0%} of the synapses")
    print(f"  {rep}")

    _say(6, f"emitting C into {args.out}/")
    from neuraltransistor.target.mcu_int8 import emit_c
    er = emit_c(small, args.out, weight_bits=8)
    print(f"  {er}")
    for f in er.files:
        print(f"    {f}")

    _say(7, "which chips it fits")
    from neuraltransistor.target.devices import DEVICES
    fits = [d.name for d in DEVICES.values() if d.fits(er.flash_B, er.ram_B) and d.available]
    print(f"  {len(fits)} of {sum(1 for d in DEVICES.values() if d.available)}: "
          + ", ".join(fits))

    print(f"\ndone in {time.time()-t0:.0f}s.\n")
    print("next:")
    print("  ntx ui                      an interactive page on :8765")
    print("  ntx demo                    write the figures into docs/img/")
    print("  ntx list                    every circuit you can compile")
    print("  ntx eval                    prove the emitted C matches the reference")
    print("  ntx emit legs_all -o out/ --target ros2 --urdf my_robot.urdf")


def cmd_fetch(args):
    from neuraltransistor.data import fetch
    from neuraltransistor.data.source import data_dir
    fetch.ensure(args.dest or data_dir())


def cmd_demo(args):
    """Write every figure in the docs, from live data."""
    from neuraltransistor import viz
    from neuraltransistor.circuit import noise
    from neuraltransistor.sensors.retina import Retina, build_resampler
    from neuraltransistor.circuit.extract import from_library
    from neuraltransistor.stimuli import Looming, giant_fiber
    import os
    conn = _conn(args)
    out = args.out
    os.makedirs(out, exist_ok=True)
    print(f"writing figures to {out}/")
    viz.noise_curve(noise.get(conn)).savefig(f"{out}/noise-curve.png", bbox_inches="tight")
    print("  noise-curve.png")
    df = conn.neurons
    stages = [("optic lobe\nintrinsic", "ol_intrinsic"), ("visual\nprojection", "visual_projection"),
              ("central brain\nintrinsic", "cb_intrinsic"), ("descending", "descending_neuron"),
              ("VNC\nintrinsic", "vnc_intrinsic"), ("motor", "vnc_motor")]
    viz.waist([(l, int((df.superclass == k).sum())) for l, k in stages]).savefig(
        f"{out}/narrow-waist.png", bbox_inches="tight")
    print("  narrow-waist.png")
    r = Retina.from_connectome(conn, "R")
    viz.lattice(r, show_axes=True).savefig(f"{out}/eye-lattice.png", bbox_inches="tight")
    print("  eye-lattice.png")
    lo = Looming(l_over_v=0.040)
    frames = lo.render_image(128, 128, hfov_deg=180)
    rs = build_resampler(r, 128, 128, hfov_deg=180)
    import numpy as _np
    i = int(_np.argmin(_np.abs(lo.t_ms + 60)))
    viz.sample(r, rs, frames[i], title="looming disc, l/|v| = 40 ms, 60 ms before contact"
               ).savefig(f"{out}/eye-sample.png", bbox_inches="tight")
    print("  eye-sample.png")
    gf = giant_fiber(lo)
    viz.trace(lo.t_ms, {"LC4 \u00b7 approach speed": gf["v_LC4"] * 1.62,
                        "LPLC2 \u00b7 object size": gf["v_LPLC2"] * 1.45,
                        "escape command": gf["v_GF"]},
              xlabel="time before impact (ms)", ylabel="mV",
              title="collision detector, object closing at r/v = 40 ms",
              vlines=[(gf["peak_t_ms"],
                       f"fires, {gf['peak_size_deg']:.0f}\u00b0 wide", "#b91c1c")]
              ).savefig(f"{out}/gf-model.png", bbox_inches="tight")
    print("  gf-model.png")
    sw = {f"r/v = {lv*1000:.0f} ms": (Looming(l_over_v=lv, dt=0.0005).t_ms,
                                      giant_fiber(Looming(l_over_v=lv, dt=0.0005))["v_GF"])
          for lv in (0.010, 0.020, 0.040, 0.080)}
    viz.trace(None, sw, xlabel="time before impact (ms)",
              ylabel="escape command (mV)",
              title="fires earlier for slower approaches \u2014 every peak inside the "
                    "published bracket").savefig(f"{out}/gf-sweep.png", bbox_inches="tight")
    print("  gf-sweep.png")
    from neuraltransistor.morph import retarget as _R, spec as _M
    from neuraltransistor.recipe import build_all
    legs = from_library(conn, "legs"); dec = _M.decode_motor(legs, conn.neurons)
    with viz.style() as _p:
        f2, axes = _p.subplots(1, 3, figsize=(12.6, 4.8))
    for ax, (nm, g) in zip(axes, [("hexapod", "tripod"), ("quadruped", "trot"),
                                  ("biped", "alternate")]):
        m = _M.BUILTIN[nm](); viz.morphology(m, _R.retarget(m, dec, gait=g), ax=ax)
    f2.savefig(f"{out}/form-factors.png", bbox_inches="tight")
    print("  form-factors.png")
    viz.recipes(build_all(conn, "out", emit=False, verbose=False)).savefig(
        f"{out}/recipes.png", bbox_inches="tight")
    print("  recipes.png")


def cmd_fit(args):
    """Fit the dynamics the connectome does not contain."""
    from neuraltransistor.circuit.extract import from_library
    from neuraltransistor.train import RingAttractorTask, fit

    conn = _conn(args)
    ir = from_library(conn, args.circuit)
    print(f"{ir.name}: {ir.n_neurons:,} neurons / {ir.n_edges:,} edges")

    task = RingAttractorTask.for_circuit(ir, conn.with_columns("instance"))
    print(f"  ring: {len(task.ring_idx)} neurons at "
          f"{len(set(task.ring_theta.round(6)))} positions")
    res = fit(ir, task, steps=args.steps, lr=args.lr, device=args.device,
              curriculum=tuple(args.curriculum) if args.curriculum else None,
              verbose=True, log_every=max(args.steps // 12, 1))
    m = res.best_metrics
    print()
    print(f"  R while driven  {m.get('R_driven', 0):.3f}")
    print(f"  R once released {m.get('R_held', 0):.3f}   (target {task.target_R})")
    print(f"  drift           {m.get('drift_rad', 0):.3f} rad over the hold")
    print(f"  rate            {m.get('hz_drive', 0):.0f} Hz driven / "
          f"{m.get('hz_hold', 0):.0f} Hz held")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"circuit": ir.name, "steps": res.steps, "seconds": res.seconds,
             "best_loss": res.best_loss, "metrics": m,
             "curve": res.curve()}, indent=1))
        print(f"  wrote {args.out}")
    return res


def cmd_firmware(args):
    """Cross-build a bootable firmware image around an emitted circuit."""
    from neuraltransistor.circuit.extract import from_library
    from neuraltransistor.ir.graph import CircuitIR
    from neuraltransistor.quant.quantize import prune_by_reliability
    from neuraltransistor.target.devices import DEVICES
    from neuraltransistor.target.firmware import emit_firmware, energy_estimate

    ir = (CircuitIR.load(args.circuit) if str(args.circuit).endswith(".fcx")
          else from_library(_conn(args), args.circuit))
    if args.p_real:
        ir, _ = prune_by_reliability(ir, args.p_real)
    rep = emit_firmware(ir, args.out, device=args.device, machine=args.machine,
                        ticks=args.ticks, build=not args.no_build)

    print(f"{ir.name}: {ir.n_neurons:,} neurons / {ir.n_edges:,} edges")
    print(f"  wrote {len(rep.files) + 3} files into {args.out}/")
    print(f"  expected digest  DIGEST {rep.expect}   ({rep.ticks} ticks vs the numpy reference)")
    if rep.built:
        print(f"  linked           .text {rep.text_b:,}  .data {rep.data_b:,}  "
              f".bss {rep.bss_b:,}")
        print(f"  flash {rep.flash_b / 1024:.1f} KB   ram {rep.ram_b / 1024:.1f} KB"
              "   (measured by the linker, not estimated)")
        if args.device and args.device in DEVICES:
            d = DEVICES[args.device]
            ok = rep.flash_b <= d.flash_b and rep.ram_b <= d.sram_b
            print(f"  {args.device}: {'fits' if ok else 'DOES NOT FIT'} "
                  f"({d.flash_kb} KB flash / {d.sram_kb} KB sram)")
            if args.volts:
                e = energy_estimate(args.device, args.hz, voltage=args.volts)
                print(f"  energy at {args.volts} V: {e['active_mw_at_voltage']} mW active "
                      f"-- DERIVED from {e['basis']}, not measured")
    else:
        print(f"  not built: {rep.build_error.splitlines()[-1] if rep.build_error else '?'}")
        print("  (install arm-none-eabi-gcc, or run cilicon.yml in CI)")
    return rep


def cmd_ui(args):
    from neuraltransistor.api.server import serve
    serve(port=args.port, open_browser=not args.no_open)


def main(argv=None):
    p = argparse.ArgumentParser("ntx",
                                description="Compile fly connectome circuits into "
                                            "quantized controllers for small robots.")
    p.add_argument("--rebuild", action="store_true", help="rebuild cached index")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quickstart", help="fresh clone -> compiled C, one command")
    q.add_argument("--circuit", default="compass")
    q.add_argument("-o", "--out", default="out")
    q.add_argument("--yes", action="store_true", help="don't ask before downloading")
    q.set_defaults(fn=cmd_quickstart)

    sub.add_parser("doctor", help="check what is ready and what is not").set_defaults(fn=cmd_doctor)

    fe = sub.add_parser("fetch", help="download the connectome release (1.0 GB, public)")
    fe.add_argument("--dest"); fe.set_defaults(fn=cmd_fetch)

    de = sub.add_parser("demo", help="write the figures from live data")
    de.add_argument("-o", "--out", default="docs/img"); de.set_defaults(fn=cmd_demo)

    li = sub.add_parser("list", help="list the circuit library")
    li.add_argument("--biology", action="store_true",
                    help="also print the underlying neuroanatomy")
    li.set_defaults(fn=cmd_list)
    sub.add_parser("index", help="show connectome index stats").set_defaults(fn=cmd_index)
    sub.add_parser("noise", help="bilateral reproducibility / noise model").set_defaults(fn=cmd_noise)

    e = sub.add_parser("extract", help="extract a circuit to .fcx")
    e.add_argument("circuit"); e.add_argument("-o", "--out")
    e.set_defaults(fn=cmd_extract)

    i = sub.add_parser("info", help="describe a .fcx")
    i.add_argument("fcx"); i.set_defaults(fn=cmd_info)

    c = sub.add_parser("compress", help="prune + quantize to a budget")
    c.add_argument("circuit")
    c.add_argument("--budget-kb", type=float)
    c.add_argument("--p-real", type=float, help="keep edges this likely to be real")
    c.add_argument("--bits", type=int, default=8)
    c.add_argument("--min-weight", type=int, default=1)
    c.add_argument("--max-err", type=float, default=0.10)
    c.add_argument("-o", "--out"); c.set_defaults(fn=cmd_compress)

    m = sub.add_parser("emit", help="emit C or a ROS 2 package")
    m.add_argument("circuit"); m.add_argument("-o", "--out", required=True)
    m.add_argument("--target", choices=["mcu", "ros2"], default="mcu")
    m.add_argument("--bits", type=int, default=8)
    m.add_argument("--morph", default="hexapod")
    m.add_argument("--urdf"); m.add_argument("--gait"); m.set_defaults(fn=cmd_emit)

    mo = sub.add_parser("morph", help="inspect morphology retargeting")
    mo.add_argument("--morph", default="hexapod")
    mo.add_argument("--urdf"); mo.add_argument("--gait")
    mo.set_defaults(fn=cmd_morph)

    ev = sub.add_parser("eval", help="run the eval suite")
    ev.add_argument("--circuits", nargs="+",
                    default=["valence", "compass", "commands"])
    ev.add_argument("-o", "--out"); ev.set_defaults(fn=cmd_eval)

    ft = sub.add_parser("fit", help="fit the dynamics the connectome does not contain")
    ft.add_argument("circuit", nargs="?", default="compass")
    ft.add_argument("--steps", type=int, default=600)
    ft.add_argument("--lr", type=float, default=0.02)
    ft.add_argument("--device", default=None)
    ft.add_argument("--curriculum", type=int, nargs="*", default=[20, 45, 90, 180],
                    help="hold_ticks per stage; empty for a single stage")
    ft.add_argument("-o", "--out", default=None)
    ft.set_defaults(fn=cmd_fit)

    fw = sub.add_parser("firmware", help="cross-build a bootable image (arm-none-eabi + QEMU)")
    fw.add_argument("circuit", nargs="?", default="compass")
    fw.add_argument("-o", "--out", default="out/firmware")
    fw.add_argument("--device", default=None, help="gate size against this part")
    fw.add_argument("--machine", default=None, help="emulated core to link and boot for")
    fw.add_argument("--ticks", type=int, default=64)
    fw.add_argument("--p-real", type=float, default=None)
    fw.add_argument("--volts", type=float, default=None,
                    help="derived energy estimate at this rail (not a measurement)")
    fw.add_argument("--hz", type=float, default=200.0)
    fw.add_argument("--no-build", action="store_true")
    fw.set_defaults(fn=cmd_firmware)

    u = sub.add_parser("ui", help="serve the local UI")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--no-open", action="store_true"); u.set_defaults(fn=cmd_ui)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
