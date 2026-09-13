"""Mapping a real camera onto the fly's eye.

The fly eye is a hexagonal lattice of ommatidia, each with a Gaussian acceptance
function. Drosophila: roughly 750-800 ommatidia per eye, interommatidial angle
(delta-phi) about 5 degrees, acceptance angle (delta-rho) about 5 degrees, covering
close to a full hemisphere per eye. Every camera you can actually bolt to a micro-robot
is a rectangular array behind a lens with a much narrower field of view.

So the optic-lobe circuits cannot be driven by raw pixels. Something has to resample the
sensor into the lattice the circuit was wired for, and that something is a sparse matrix
computed once here and then quantized and shipped like any other weight.

The resampler is deliberately honest about the mismatch it is papering over: a 70-degree
camera covers a fraction of the fly's visual field, so most ommatidia have no sensor
data at all. ``coverage`` reports exactly which ones, instead of feeding them zeros and
letting a motion detector interpret that as darkness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DROSOPHILA = dict(n_ommatidia=800, delta_phi_deg=5.0, delta_rho_deg=5.0,
                  fov_azimuth_deg=180.0, fov_elevation_deg=140.0)


@dataclass
class Retina:
    """A hex lattice of viewing directions with Gaussian acceptance."""
    azimuth: np.ndarray        # radians, per ommatidium
    elevation: np.ndarray
    delta_rho: float           # acceptance angle, radians
    eye: str = "L"

    @property
    def n(self) -> int:
        return len(self.azimuth)


def hex_lattice(n_target: int = 400, delta_phi_deg: float = 5.0,
                fov_az_deg: float = 180.0, fov_el_deg: float = 140.0,
                eye: str = "L", delta_rho_deg: float = 5.0) -> Retina:
    """Build a hexagonal lattice of viewing directions for one eye.

    Rows are offset by half a column spacing, which is what makes the packing hexagonal
    rather than square; this is not cosmetic, it is why neighbouring columns come in the
    three axes the T4/T5 circuit is wired to compare along.
    """
    dphi = np.radians(delta_phi_deg)
    row_h = dphi * np.sqrt(3) / 2
    el_max = np.radians(fov_el_deg) / 2
    az_max = np.radians(fov_az_deg) / 2
    az, el = [], []
    r = 0
    e = -el_max
    while e <= el_max:
        offset = (dphi / 2) if (r % 2) else 0.0
        a = -az_max + offset
        while a <= az_max:
            az.append(a)
            el.append(e)
            a += dphi
        e += row_h
        r += 1
    az = np.array(az)
    el = np.array(el)
    if len(az) > n_target:  # thin uniformly to hit the requested count
        keep = np.linspace(0, len(az) - 1, n_target).astype(int)
        az, el = az[keep], el[keep]
    if eye == "R":
        az = -az
    return Retina(azimuth=az, elevation=el,
                  delta_rho=np.radians(delta_rho_deg), eye=eye)


@dataclass
class Resampler:
    """Sparse camera-pixel -> ommatidium matrix, plus what it could not cover."""
    indptr: np.ndarray
    indices: np.ndarray
    weight: np.ndarray
    n_ommatidia: int
    n_pixels: int
    covered: np.ndarray        # bool per ommatidium
    shape: tuple

    @property
    def coverage(self) -> float:
        return float(self.covered.mean())

    def apply(self, image: np.ndarray) -> np.ndarray:
        flat = image.reshape(-1).astype(np.float32)
        out = np.zeros(self.n_ommatidia, dtype=np.float32)
        for i in range(self.n_ommatidia):
            a, b = self.indptr[i], self.indptr[i + 1]
            if a == b:
                continue
            out[i] = float(flat[self.indices[a:b]] @ self.weight[a:b])
        return out

    def footprint(self, weight_bits: int = 8) -> dict:
        nnz = len(self.indices)
        return {"nnz": nnz,
                "bytes": int(nnz * (weight_bits / 8 + 4) + (self.n_ommatidia + 1) * 4),
                "KB": round((nnz * (weight_bits / 8 + 4)
                             + (self.n_ommatidia + 1) * 4) / 1024, 2)}


def build_resampler(retina: Retina, width: int, height: int,
                    hfov_deg: float = 70.0, vfov_deg: float | None = None,
                    truncate_sigma: float = 2.0) -> Resampler:
    """Gaussian-acceptance resampling from a pinhole camera into the lattice."""
    vfov_deg = vfov_deg if vfov_deg is not None else hfov_deg * height / width
    hf, vf = np.radians(hfov_deg), np.radians(vfov_deg)

    px = (np.arange(width) + 0.5) / width - 0.5
    py = (np.arange(height) + 0.5) / height - 0.5
    pix_az = px * hf
    pix_el = -py * vf
    AZ, EL = np.meshgrid(pix_az, pix_el)
    AZ = AZ.reshape(-1)
    EL = EL.reshape(-1)

    sigma = retina.delta_rho / 2.355        # FWHM -> sigma
    cutoff = truncate_sigma * sigma
    rows_i, rows_w, counts = [], [], np.zeros(retina.n, dtype=np.int64)
    covered = np.zeros(retina.n, dtype=bool)

    for i in range(retina.n):
        da = AZ - retina.azimuth[i]
        de = EL - retina.elevation[i]
        d2 = da * da + de * de
        m = d2 <= cutoff * cutoff
        if not m.any():
            rows_i.append(np.zeros(0, np.int32))
            rows_w.append(np.zeros(0, np.float32))
            continue
        idx = np.flatnonzero(m).astype(np.int32)
        w = np.exp(-0.5 * d2[m] / (sigma * sigma)).astype(np.float32)
        s = w.sum()
        if s > 0:
            w /= s
            covered[i] = True
        rows_i.append(idx)
        rows_w.append(w)
        counts[i] = len(idx)

    indptr = np.zeros(retina.n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return Resampler(indptr=indptr,
                     indices=np.concatenate(rows_i) if retina.n else np.zeros(0, np.int32),
                     weight=np.concatenate(rows_w) if retina.n else np.zeros(0, np.float32),
                     n_ommatidia=retina.n, n_pixels=width * height,
                     covered=covered, shape=(height, width))
