# Mapping a camera onto a fly eye

Optic-lobe circuits are wired for a hexagonal lattice of ommatidia with Gaussian
acceptance. Every camera you can bolt to a micro-robot is a rectangular array behind a
lens. Something has to resample between them, and that something is a sparse matrix
computed once and then quantized and shipped like any other weight.

## The eye, measured from this dataset

Counts read directly out of male-CNS v1.0 rather than quoted:

| | |
|---|--:|
| distinct hex columns, right eye | **892** |
| distinct hex columns, left eye | 879 |
| Mi1 cells, right eye (Mi1 tiles 1:1 with columns) | **886** |
| interommatidial angle Δφ | 4.63° |

Column coordinates live in `assignedOlHex1` / `assignedOlHex2`. A caution if you use
them: **the neighbour convention is not the usual axial one.** Neighbours are ±(1,0),
±(0,1) and ±(1,1) — the basis is at 120°, so ±(1,−1) is the long diagonal. Both readings
look perfectly regular geometrically and nothing catches the error; only the resulting
footprint aspect ratio discriminates.

## The acceptance angle everyone quotes is wrong

Textbooks give Δρ ≈ 4.5–5.7°. That is a prediction of the Snyder diffraction formula,
not a measurement. Every intracellular recording disagrees:

| source | Δρ |
|---|--:|
| Gonzalez-Bellido 2011 | 8.23° |
| Juusola 2017 (dark) | 9.47° |
| Juusola 2017 (light) | 7.70° |

So **Δρ/Δφ ≈ 1.7–2.0, not ~1**: the fly is a heavily *blurred* sampler whose acceptance
functions overlap their neighbours substantially. Using 5° makes the eye look far sharper
than it is, and makes a camera look worse at matching it than it really is. neuraltransistor
defaults to the measured 8.2°.

This was a real bug here — the first version of `retina.py` used 5.0° and reported
coverage numbers that were too pessimistic.

(Related: flyvis's `BoxEye` is a flat 13×13 **box** at 5.8°/column, not a Gaussian, and
its lattice is 11.8% anisotropic with the isotropic fix commented out in source. Worth
knowing before comparing against it.)

## Pixel count is solved; field of view is not

Because Δρ is ~8°, σ = 3.27°, and even a 128×128 sensor behind a 180° fisheye resolves
the fly's own sampling at 2.3 px/σ. Resolution is not the constraint. **Field of view is.**

Coverage of the fly's visual field, computed by `build_resampler`:

| optic | sensor | FOV | coverage | resampler |
|---|---|--:|--:|--:|
| SilkyEvCam GenX320, stock lens | 320×320 | 46° | **13.3%** | 546k nnz |
| GenX320, wider lens | 320×320 | 70° | 26.1% | 544k nnz |
| wide lens | 160×120 | 120° | 54.1% | 102k nnz |
| fisheye | 128×128 | 180° | **100%** | 67k nnz, 331 KB |

The fisheye row is the interesting one: a *lower* resolution sensor with a wide enough
lens both covers the whole eye and produces a **smaller** resampler, because each
ommatidium integrates fewer pixels. For driving fly optic circuits, spend your budget on
the lens, not the sensor.

The obstacle is optical, not computational: the GenX320 array is 2.016 mm wide, so 180°
f-θ needs a 0.64 mm effective focal length, and stock micro-optics bottom out near
1.1 mm. Three ways out — tile several modules, commission a wafer-level fisheye, or model
only a frontal acute zone.

## Mass is the real budget

| sensor | mass | note |
|---|--:|---|
| DAVIS346 | 100 g | |
| Prophesee EVK4 | 40 g | |
| DVXplorer Micro | 16 g | still over a Crazyflie's 15 g payload |
| **SilkyEvCam GenX320 4×5 flex module** | **0.13 g** | *same silicon as the EVK4* |
| Floreano lab artificial elementary eye | **2 mg** | 3 photoreceptors, 1 lens, on-die ADC, Δρ 4.78° / Δφ 5.52°, 444 µA — essentially one ommatidium in 0.9 mm² |

A ~120× mass gap for identical silicon, depending on whether it is sold as a camera or a
module. And **nobody has flown an event camera on a sub-30 g aircraft** — the lightest
event-camera drone is ~800 g. The budget closes on paper: a Crazyflie build lands at
5.63 g of a 15 g payload and ~1.5% of flight power.

## Status

`neuraltransistor/sensors/retina.py` builds the lattice and the sparse Gaussian resampler, and
reports which ommatidia the sensor cannot cover instead of feeding them zeros — a motion
detector reads zeros as darkness, which is a different thing from no data.

What does **not** exist yet: an adapter that consumes a real event stream, any
calibration against a physical sensor, and any validation that resampled input drives the
optic circuits correctly. This is the thinnest layer in the repo and it is untested
beyond "it runs and the geometry is right."
