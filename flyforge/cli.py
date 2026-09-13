"""flyforge command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _conn(args):
    from flyforge.data.source import Connectome
    return Connectome.load(rebuild=getattr(args, "rebuild", False), verbose=True)


def cmd_list(args):
    from flyforge.circuit.extract import LIBRARY
    for k, v in LIBRARY.items():
        print(f"{k:18s} {v['doc']}")


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
    from flyforge.circuit import noise
    c = _conn(args)
    m = noise.get(c, rebuild=args.rebuild, verbose=True)
    print(m.table())
    print()
    for p in (0.90, 0.95, 0.99, 0.999):
        d = m.cost_of(p)
        print(f"  P(real) >= {p:<6} -> keep weight >= {d['min_weight']:<3} "
              f"drops {d['frac_dropped']:.1%} of edges")


def cmd_extract(args):
    from flyforge.circuit.extract import from_library
    c = _conn(args)
    ir = from_library(c, args.circuit)
    print(ir.summary())
    if args.out:
        p = ir.save(args.out)
        print(f"wrote {p} ({Path(p).stat().st_size/1024:.0f} KB)")


def cmd_info(args):
    from flyforge.ir.graph import CircuitIR
    ir = CircuitIR.load(args.fcx)
    print(ir.summary())
    print("  provenance:")
    for k, v in ir.provenance.items():
        print(f"    {k}: {v}")


def cmd_compress(args):
    from flyforge.circuit.extract import from_library
    from flyforge.ir.graph import CircuitIR
    from flyforge.quant.quantize import compress, fit_budget, prune_by_reliability
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
    from flyforge.circuit.extract import from_library
    from flyforge.ir.graph import CircuitIR
    ir = (CircuitIR.load(args.circuit) if args.circuit.endswith(".fcx")
          else from_library(_conn(args), args.circuit))
    if args.target == "mcu":
        from flyforge.target.mcu_int8 import emit_c
        r = emit_c(ir, args.out, weight_bits=args.bits)
        print(r)
        for f in r.files:
            print("   ", f)
    elif args.target == "ros2":
        from flyforge.morph import spec as M, urdf
        from flyforge.target.ros2 import emit_ros2
        if args.urdf:
            morph, rep = urdf.parse(args.urdf)
            print(f"urdf: {morph} ({rep['n_actuated']} actuated joints)")
        else:
            morph = M.BUILTIN[args.morph]()
        r = emit_ros2(ir, morph, args.out)
        print(f"ros2 package: {r['package']} ({r['joints']} joints @ {r['control_hz']}Hz)")
        for f in r["files"]:
            print("   ", f)


def cmd_morph(args):
    from flyforge.circuit.extract import from_library
    from flyforge.morph import spec as M, retarget as R, urdf
    c = _conn(args)
    ir = from_library(c, "legs_all")
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
    from flyforge.evals import suite
    c = _conn(args)
    r = suite.run_all(c, circuits=tuple(args.circuits))
    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=1, default=str))
        print(f"wrote {args.out}")
    sys.exit(0 if r["passed"] == r["total"] else 1)


def cmd_ui(args):
    from flyforge.api.server import serve
    serve(port=args.port, open_browser=not args.no_open)


def main(argv=None):
    p = argparse.ArgumentParser("flyforge",
                                description="Compile fly connectome circuits into "
                                            "quantized controllers for small robots.")
    p.add_argument("--rebuild", action="store_true", help="rebuild cached index")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list the circuit library").set_defaults(fn=cmd_list)
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
    m.add_argument("--urdf"); m.set_defaults(fn=cmd_emit)

    mo = sub.add_parser("morph", help="inspect morphology retargeting")
    mo.add_argument("--morph", default="hexapod")
    mo.add_argument("--urdf"); mo.add_argument("--gait")
    mo.set_defaults(fn=cmd_morph)

    ev = sub.add_parser("eval", help="run the eval suite")
    ev.add_argument("--circuits", nargs="+",
                    default=["gate_readout", "compass", "descending"])
    ev.add_argument("-o", "--out"); ev.set_defaults(fn=cmd_eval)

    u = sub.add_parser("ui", help="serve the local UI")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--no-open", action="store_true"); u.set_defaults(fn=cmd_ui)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
