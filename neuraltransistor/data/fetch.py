"""Fetch the male-CNS release. No account, no token, no cloud SDK.

The three tables this project needs are served as plain public HTTPS objects out of the
Janelia release bucket. That means a first-time user needs exactly one command and no
credentials, which is the difference between a project people try and a project people
bounce off.

Downloads resume: interrupt it, run it again, it picks up where it stopped. Files are
verified by content length against the published sizes, so a truncated download is
detected rather than silently producing a corrupt index later.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BASE = ("https://storage.googleapis.com/flyem-male-cns/v1.0"
        "/connectome-data/flat-connectome")

#: name -> (expected bytes, what it is). Sizes verified against the served objects.
FILES = {
    "body-annotations-male-cns-v1.0-minconf-0.5.feather": (
        14_483_314, "neuron annotations: type, class, side, neuromere, hex column"),
    "body-neurotransmitters-male-cns-v1.0.feather": (
        43_282_834, "per-neuron neurotransmitter predictions"),
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather": (
        1_051_241_946, "the connection graph: 151,856,684 edges"),
}

TOTAL_BYTES = sum(v[0] for v in FILES.values())


@dataclass
class FetchResult:
    path: Path
    downloaded: int
    skipped: bool
    seconds: float


def _human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:,.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024


def _bar(done: int, total: int, width: int = 28) -> str:
    frac = 0.0 if not total else min(done / total, 1.0)
    fill = int(frac * width)
    return "[" + "#" * fill + "." * (width - fill) + f"] {frac*100:5.1f}%"


def fetch_one(name: str, dest_dir: Path, progress: bool = True,
              chunk: int = 1 << 20) -> FetchResult:
    """Download one file, resuming a partial transfer if present."""
    if name not in FILES:
        raise KeyError(f"unknown file {name!r}; have {sorted(FILES)}")
    expect, _desc = FILES[name]
    dest = Path(dest_dir) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if dest.exists() and dest.stat().st_size == expect:
        if progress:
            print(f"  ok    {name}  ({_human(expect)}, already complete)")
        return FetchResult(dest, 0, True, 0.0)

    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > expect:
        part.unlink(); have = 0

    req = urllib.request.Request(f"{BASE}/{name}")
    if have:
        req.add_header("Range", f"bytes={have}-")
    mode = "ab" if have else "wb"

    if progress:
        verb = "resume" if have else "get   "
        print(f"  {verb} {name}  ({_human(expect)})")

    got = have
    last = 0.0
    with urllib.request.urlopen(req, timeout=60) as r, open(part, mode) as f:
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)
            got += len(b)
            now = time.time()
            if progress and (now - last > 0.25):
                rate = got / max(now - t0, 1e-6)
                sys.stdout.write(f"\r        {_bar(got, expect)}  "
                                 f"{_human(got)} / {_human(expect)}  "
                                 f"{_human(rate)}/s   ")
                sys.stdout.flush()
                last = now
    if progress:
        sys.stdout.write("\r" + " " * 78 + "\r")

    if got != expect:
        raise IOError(f"{name}: expected {expect:,} bytes, got {got:,}. "
                      f"Partial file kept at {part} -- rerun to resume.")
    shutil.move(str(part), str(dest))
    dt = time.time() - t0
    if progress:
        print(f"  done  {name}  in {dt:.0f}s ({_human(expect/max(dt,1e-6))}/s)")
    return FetchResult(dest, got - have, False, dt)


def status(dest_dir: Path) -> dict:
    """What is present, what is missing, what is truncated."""
    dest_dir = Path(dest_dir)
    out = {"dir": str(dest_dir), "files": {}, "ready": True, "missing_bytes": 0}
    for name, (expect, desc) in FILES.items():
        p = dest_dir / name
        have = p.stat().st_size if p.exists() else 0
        ok = have == expect
        out["files"][name] = {"present": p.exists(), "complete": ok,
                              "have": have, "expect": expect, "what": desc}
        if not ok:
            out["ready"] = False
            out["missing_bytes"] += expect - have
    return out


def ensure(dest_dir: Path | str | None = None, progress: bool = True) -> Path:
    """Make sure all three tables are present and complete. Returns the directory."""
    from neuraltransistor.data.source import data_dir
    dest_dir = Path(dest_dir) if dest_dir else data_dir()
    st = status(dest_dir)
    if st["ready"]:
        if progress:
            print(f"  all three tables present in {dest_dir}")
        return dest_dir
    if progress:
        print(f"  fetching {_human(st['missing_bytes'])} into {dest_dir}")
        print(f"  (public release data, no account needed; resumable)")
    for name in FILES:
        fetch_one(name, dest_dir, progress=progress)
    return dest_dir
