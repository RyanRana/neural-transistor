"""The fly eye: a real hexagonal lattice, and how to get a camera onto it.

Optic-lobe circuits are wired for a hex lattice of ommatidia with Gaussian acceptance.
Every camera you can bolt to a micro-robot is a rectangular array behind a lens. This
module builds the lattice -- preferably from the connectome itself rather than a
synthetic approximation -- and the sparse matrix that resamples a sensor into it.

**The lattice is in the data.** ``assignedOlHex1``/``assignedOlHex2`` give each optic-lobe
neuron its column. Measured here: 892 distinct columns on the right eye, 879 on the left,
886 Mi1 cells (Mi1 tiles 1:1 with columns). Building from those coordinates means the
lattice matches the circuit you are about to drive, column for column.

**The neighbour convention is not the usual axial one.** Neighbours are ±(1,0), ±(0,1)
and ±(1,1), so the basis vectors sit at 120 degrees and ±(1,-1) is the long diagonal.
Both readings look perfectly regular and nothing catches the mistake; only the resulting
footprint aspect ratio discriminates. Getting this wrong silently rotates every motion
detector by 30 degrees.

**The acceptance angle everyone quotes is wrong.** Textbooks give delta-rho ~4.5-5.7 deg,
which is a prediction of the Snyder diffraction formula, not a measurement. Intracellular
recordings give 7.7-9.5 deg (Gonzalez-Bellido 2011: 8.23; Juusola 2017: 9.47 dark /
7.70 light). So delta-rho / delta-phi is about 1.8, not ~1 -- the fly is a heavily
*blurred* sampler whose acceptance functions overlap their neighbours substantially.
Using 5 degrees makes the eye look far sharper than it is. Default here is 8.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

#: Measured, not predicted. Column counts read directly from male-CNS v1.0.
DROSOPHILA = dict(
    n_columns_right=892, n_columns_left=879, n_mi1_right=886,
    delta_phi_deg=4.63,        # interommatidial angle
    delta_rho_deg=8.2,         # acceptance angle, INTRACELLULAR (not Snyder)
    rho_over_phi=1.77,
    fov_azimuth_deg=180.0, fov_elevation_deg=140.0,
    blind_spot_rear_deg=50.0,
)

#: Hex neighbour offsets in (hex1, hex2). The basis is at 120 degrees, so these six are
#: equidistant and ±(1,-1) is NOT a neighbour.
HEX_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1))

#: The three axes of the lattice, as (offset, name). T4/T5 subtypes compare along these.
HEX_AXES = (((1, 0), "horizontal"), ((0, 1), "oblique_down"), ((1, 1), "oblique_up"))


def hex_to_xy(h1: np.ndarray, h2: np.ndarray, pitch: float = 1.0) -> tuple:
    """Hex coordinates -> planar cartesian, using the 120-degree basis."""
    h1 = np.asarray(h1, dtype=np.float64)
    h2 = np.asarray(h2, dtype=np.float64)
    x = (h1 - 0.5 * h2) * pitch
    y = (np.sqrt(3) / 2 * h2) * pitch
    return x, y


@dataclass
class Retina:
    """A lattice of viewing directions with Gaussian acceptance.

    ``azimuth``/``elevation`` are radians. ``hex1``/``hex2`` are present when the lattice
    came from the connectome, and are what lets you line an ommatidium up with the
    columnar neuron that reads it.
    """
    azimuth: np.ndarray
    elevation: np.ndarray
    delta_rho: float
    eye: str = "R"
    hex1: Optional[np.ndarray] = None
    hex2: Optional[np.ndarray] = None
    source: str = "synthetic"

    @property
    def n(self) -> int:
        return len(self.azimuth)

    @property
    def sigma(self) -> float:
        """Gaussian sigma of the acceptance function (delta_rho is the FWHM)."""
        return self.delta_rho / 2.3548

    # -- neighbour structure -------------------------------------------------

    def neighbours(self) -> dict:
        """{offset: int array of partner index, or -1} for the six hex neighbours.

        Only available for a connectome-derived lattice, because it needs the integer
        hex coordinates. This is what an elementary motion detector consumes: each
        ommatidium paired with its neighbour along a given axis.
        """
        if self.hex1 is None:
            raise ValueError("neighbours() needs a connectome-derived lattice; "
                             "use Retina.from_connectome()")
        key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(self.hex1, self.hex2))}
        out = {}
        for d in HEX_NEIGHBOURS:
            arr = np.full(self.n, -1, dtype=np.int32)
            for i, (a, b) in enumerate(zip(self.hex1, self.hex2)):
                j = key.get((int(a) + d[0], int(b) + d[1]))
                if j is not None:
                    arr[i] = j
            out[d] = arr
        return out

    def axis_pairs(self, axis: str = "horizontal") -> np.ndarray:
        """(N,2) array of (here, neighbour) index pairs along one lattice axis."""
        off = dict((n, o) for o, n in HEX_AXES)[axis]
        nb = self.neighbours()[off]
        ok = nb >= 0
        return np.stack([np.flatnonzero(ok), nb[ok]], axis=1)

    def interior(self) -> np.ndarray:
        """Indices of ommatidia with all six neighbours present (no edge artifacts)."""
        nb = self.neighbours()
        good = np.ones(self.n, dtype=bool)
        for d in HEX_NEIGHBOURS:
            good &= nb[d] >= 0
        return np.flatnonzero(good)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_connectome(cls, conn, eye: str = "R", cell_type: str = "Mi1",
                        delta_phi_deg: float = None,
                        delta_rho_deg: float = None) -> "Retina":
        """Build the lattice from the connectome's own column assignment.

        Uses one cell per column (Mi1 by default -- it tiles the medulla 1:1 and is the
        main ON-pathway input to T4), so the resulting lattice is in exact correspondence
        with the columnar neurons the optic circuits are built from.
        """
        df = conn.with_columns("assignedOlHex1", "assignedOlHex2")
        m = df[(df["type"].astype(str) == cell_type) & (df["somaSide"] == eye)]
        m = m.dropna(subset=["assignedOlHex1", "assignedOlHex2"])
        if len(m) == 0:
            raise ValueError(f"no {cell_type} on side {eye} with hex coordinates")
        h1 = m["assignedOlHex1"].to_numpy().astype(np.int32)
        h2 = m["assignedOlHex2"].to_numpy().astype(np.int32)

        dphi = np.radians(delta_phi_deg or DROSOPHILA["delta_phi_deg"])
        drho = np.radians(delta_rho_deg or DROSOPHILA["delta_rho_deg"])
        x, y = hex_to_xy(h1, h2, pitch=dphi)
        # Centre the lattice, then treat the plane as an angular map. This is an
        # equidistant-projection approximation of a curved eye: fine near the centre,
        # progressively wrong toward the rim. Real per-ommatidium viewing vectors exist
        # (reiserlab/eyemap_T4, from microCT) and would replace this.
        x = x - x.mean()
        y = y - y.mean()
        if eye == "L":
            x = -x
        return cls(azimuth=x, elevation=y, delta_rho=drho, eye=eye,
                   hex1=h1, hex2=h2, source=f"male-cns {cell_type} {eye}")

    @classmethod
    def synthetic(cls, n_target: int = 886, delta_phi_deg: float = None,
                  fov_az_deg: float = 180.0, fov_el_deg: float = 140.0,
                  eye: str = "R", delta_rho_deg: float = None) -> "Retina":
        """A regular hex lattice, when you have no connectome loaded."""
        dphi = np.radians(delta_phi_deg or DROSOPHILA["delta_phi_deg"])
        drho = np.radians(delta_rho_deg or DROSOPHILA["delta_rho_deg"])
        row_h = dphi * np.sqrt(3) / 2
        el_max, az_max = np.radians(fov_el_deg) / 2, np.radians(fov_az_deg) / 2
        az, el = [], []
        r, e = 0, -el_max
        while e <= el_max:
            a = -az_max + ((dphi / 2) if (r % 2) else 0.0)
            while a <= az_max:
                az.append(a); el.append(e); a += dphi
            e += row_h; r += 1
        az, el = np.array(az), np.array(el)
        if len(az) > n_target:
            keep = np.linspace(0, len(az) - 1, n_target).astype(int)
            az, el = az[keep], el[keep]
        if eye == "L":
            az = -az
        return cls(azimuth=az, elevation=el, delta_rho=drho, eye=eye, source="synthetic")

    def extent_deg(self) -> dict:
        return {"azimuth_deg": float(np.degrees(np.ptp(self.azimuth))),
                "elevation_deg": float(np.degrees(np.ptp(self.elevation))),
                "delta_rho_deg": float(np.degrees(self.delta_rho)),
                "n": self.n, "source": self.source}


# convenience alias kept for the earlier API
def hex_lattice(n_target: int = 886, **kw) -> Retina:
    return Retina.synthetic(n_target=n_target, **kw)


@dataclass
class Resampler:
    """Sparse camera-pixel -> ommatidium matrix, plus what it could not cover."""
    indptr: np.ndarray
    indices: np.ndarray
    weight: np.ndarray
    n_ommatidia: int
    n_pixels: int
    covered: np.ndarray
    shape: tuple

    def __post_init__(self):
        self._m = None

    @property
    def matrix(self):
        """scipy CSR view, built once. Makes apply() a single sparse matvec."""
        if self._m is None:
            from scipy.sparse import csr_matrix
            self._m = csr_matrix((self.weight, self.indices, self.indptr),
                                 shape=(self.n_ommatidia, self.n_pixels))
        return self._m

    @property
    def coverage(self) -> float:
        return float(self.covered.mean())

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Resample one frame (H,W) into per-ommatidium intensity."""
        return self.matrix @ image.reshape(-1).astype(np.float32)

    def apply_batch(self, frames: np.ndarray) -> np.ndarray:
        """Resample (T,H,W) -> (T,n_ommatidia) in one multiply."""
        f = frames.reshape(len(frames), -1).astype(np.float32)
        return (self.matrix @ f.T).T

    def footprint(self, weight_bits: int = 8) -> dict:
        nnz = len(self.indices)
        b = nnz * (weight_bits / 8 + 4) + (self.n_ommatidia + 1) * 4
        return {"nnz": nnz, "bytes": int(b), "KB": round(b / 1024, 2),
                "mean_pixels_per_ommatidium": round(nnz / max(self.n_ommatidia, 1), 1)}


def build_resampler(retina: Retina, width: int, height: int,
                    hfov_deg: float = 70.0, vfov_deg: float | None = None,
                    truncate_sigma: float = 2.5) -> Resampler:
    """Gaussian-acceptance resampling from a pinhole camera into the lattice.

    Vectorised over ommatidia: for a 180-degree fisheye into 886 ommatidia this is a
    fraction of a second, where the naive per-ommatidium loop took minutes.
    """
    vfov_deg = vfov_deg if vfov_deg is not None else hfov_deg * height / width
    hf, vf = np.radians(hfov_deg), np.radians(vfov_deg)
    px = (np.arange(width) + 0.5) / width - 0.5
    py = (np.arange(height) + 0.5) / height - 0.5
    AZ, EL = np.meshgrid(px * hf, -py * vf)
    AZ, EL = AZ.reshape(-1), EL.reshape(-1)

    sigma = retina.sigma
    cutoff = truncate_sigma * sigma
    rows_i, rows_w = [], []
    counts = np.zeros(retina.n, dtype=np.int64)
    covered = np.zeros(retina.n, dtype=bool)

    # bucket pixels by azimuth so each ommatidium only scans a slab
    order = np.argsort(AZ)
    AZs, ELs = AZ[order], EL[order]
    for i in range(retina.n):
        a0, e0 = retina.azimuth[i], retina.elevation[i]
        lo = np.searchsorted(AZs, a0 - cutoff)
        hi = np.searchsorted(AZs, a0 + cutoff)
        if hi <= lo:
            rows_i.append(np.zeros(0, np.int32)); rows_w.append(np.zeros(0, np.float32))
            continue
        sl = slice(lo, hi)
        de = ELs[sl] - e0
        da = AZs[sl] - a0
        d2 = da * da + de * de
        m = d2 <= cutoff * cutoff
        if not m.any():
            rows_i.append(np.zeros(0, np.int32)); rows_w.append(np.zeros(0, np.float32))
            continue
        idx = order[sl][m].astype(np.int32)
        w = np.exp(-0.5 * d2[m] / (sigma * sigma)).astype(np.float32)
        s = w.sum()
        if s > 0:
            w /= s
            covered[i] = True
        rows_i.append(idx); rows_w.append(w); counts[i] = len(idx)

    indptr = np.zeros(retina.n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return Resampler(
        indptr=indptr,
        indices=np.concatenate(rows_i) if retina.n else np.zeros(0, np.int32),
        weight=np.concatenate(rows_w) if retina.n else np.zeros(0, np.float32),
        n_ommatidia=retina.n, n_pixels=width * height, covered=covered,
        shape=(height, width))
