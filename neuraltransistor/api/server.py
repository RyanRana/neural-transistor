"""A small local UI for poking at circuits without writing code.

Deliberately stdlib-only -- no FastAPI, no uvicorn, no install step. ``neuraltransistor ui``
should work on a fresh clone with numpy and pyarrow and nothing else, because the point
of it is to lower the bar to trying this, and a dependency wall is the opposite of that.

Extracted circuits are cached in memory, so the first click on a circuit costs an
extraction (0.02-0.3 s) and every later one is instant.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).parent / "static"

_conn = None
_cache: dict = {}
_lock = threading.Lock()


def _connectome():
    global _conn
    if _conn is None:
        from neuraltransistor.data.source import Connectome
        _conn = Connectome.load(verbose=False)
    return _conn


def _circuit(key: str):
    with _lock:
        if key not in _cache:
            from neuraltransistor.circuit.extract import from_library
            _cache[key] = from_library(_connectome(), key)
        return _cache[key]


def _compute(key: str, p_real: float, bits: int) -> dict:
    from neuraltransistor.circuit import noise
    from neuraltransistor.target.devices import DEVICES
    from neuraltransistor.quant.quantize import compress
    from neuraltransistor.target.mcu_int8 import emit_c
    import tempfile

    ir = _circuit(key)
    model = noise.get(_connectome())
    min_w = model.min_weight_for(p_real)
    out, rep = compress(ir, weight_bits=bits, scheme="log", prune_min_weight=min_w)

    with tempfile.TemporaryDirectory() as td:
        er = emit_c(out, td, weight_bits=bits)

    devices = []
    for dev, d in DEVICES.items():
        devices.append({"name": dev, "fits": d.fits(er.flash_B, er.ram_B),
                        "flash_kb": d.flash_kb, "sram_kb": d.sram_kb,
                        "mw": d.active_mw, "available": d.available, "note": d.note})
    return {
        "circuit": key,
        "doc": ir.provenance.get("doc", ""),
        "neurons": out.n_neurons,
        "edges_full": ir.n_edges,
        "edges": out.n_edges,
        "synapses": out.n_synapses,
        "mod_edges": out.n_mod_edges,
        "min_weight": int(min_w),
        "edge_retention": rep.edge_retention,
        "synapse_retention": rep.synapse_retention,
        "kb": round(er.flash_B / 1024, 1),
        "ram_kb": round(er.ram_B / 1024, 1),
        "drive_err": rep.drive_rel_err_mean,
        "sign_flips": rep.drive_sign_flips,
        "sign_coverage": float((out.sign_kind != 0).mean()),
        "devices": devices,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html")
        if self.path == "/api/circuits":
            from neuraltransistor.circuit.extract import LIBRARY
            from neuraltransistor.circuit import noise
            m = noise.get(_connectome())
            c = _connectome()
            return self._send(200, json.dumps({
                "circuits": [{"key": k, "doc": v["doc"]} for k, v in LIBRARY.items()],
                "dataset": c.meta.dataset,
                "n_neurons": c.meta.n_neurons,
                "n_edges": c.meta.n_edges_retained,
                "noise": {"chance": m.chance, "ceiling": m.ceiling,
                          "curve": [[b[0], b[1], b[2], b[3]] for b in m.bins]},
            }))
        if self.path.startswith("/api/compute"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            try:
                out = _compute(q.get("circuit", ["compass"])[0],
                               float(q.get("p_real", ["0.9"])[0]),
                               int(q.get("bits", ["8"])[0]))
                return self._send(200, json.dumps(out))
            except Exception as e:
                return self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))
        self._send(404, json.dumps({"error": "not found"}))


def serve(port: int = 8765, open_browser: bool = True):
    _connectome()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"neuraltransistor ui -> {url}   (ctrl-c to stop)")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
