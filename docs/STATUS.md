# STATUS

Source of truth for what exists. Distinguishes **verified** (ran it, read the output)
from **code exists** (written, imports, never proved anything) from **not built**.

Last updated 2026-09-13.

## Milestones

| # | milestone | state |
|---|---|---|
| F1 | Load male-CNS v1.0, build a cached CSR index | **verified** |
| F2 | Selection DSL + circuit extraction into a portable IR | **verified** |
| F3 | Sign assignment with the NMJ exception | **verified** |
| F4 | Bilateral-reproducibility noise model → P(real \| w) | **verified** |
| F5 | Prune / quantize / budget-solve to a byte target | **verified** |
| F6 | int8 C emitter, bit-identical to reference | **verified** |
| F7 | Morphology: muscle→joint map, URDF import, gait retarget | **verified** |
| F8 | ROS 2 package emitter | code exists |
| F9 | Hex-lattice retina resampler | code exists |
| F10 | Local UI | **verified** |
| F11 | Eval suite | **verified** (33 evals) |
| F12 | Dynamics fitting against behaviour | **not built** |
| F13 | Functional validation (does the compass hold a bump) | **measured — negative** |
| F14 | SNN backend (Loihi / Speck) | **not built** |
| F15 | On-hardware flash + power measurement | **not built** |

## Verified numbers

Everything here was produced by running the thing, on `male-cns-v1.0 minconf 0.5`.

**Index (F1).** 211,577 annotated bodies. Raw table 151,856,684 edges /
311,833,243 synapses. Retained over annotated bodies: 26,028,386 edges (17.1%) /
125,365,933 synapses (40.2%). Max weight 2,591. Build 75.4 s, cached. Extraction after
that is 0.02–0.32 s per circuit, against ~140 s for a naive scan.

**Noise model (F4).** 19,654,874 ipsilateral typed edges. Chance floor 0.280
(post-type shuffled within hemisphere), ceiling 0.992. P(real|w=1) = 0.786, crosses
0.99 at w=12. P≥0.95 ⇒ weight ≥ 6, drops 81.0% of edges.

**Sign (F3).** 77.4% of annotated bodies get a sign. Per circuit: central complex 100%,
optic motion 99.0%, leg intrinsic 99.2%, descending 97.1%, VNC motor 44.9%. NMJ
exception rescues 166 motor neurons; naive map says 304/708 inhibitory, corrected says 10.

**Quantization (F5).** int8 mean drive error 0.11% (gate_readout), 0.22% (compass),
0.37% (descending); zero sign flips in all three. Compass at P≥0.95 + delta-indexed
int8: 188 KB → 73.6 KB at 0.000% drive error.

**C emitter (F6).** 64/64 ticks bit-identical to the numpy reference on gate_readout,
compass and descending. Clean under `-Wall -Wextra -Werror`. Throughput on host:
96,618 tick/s (gate_readout), 40,445 (compass), 41,246 (descending) — 0.34–2.50 ns/edge.

**Morphology (F7).** 34 motor decodes from `legs_all`: 3 leg pairs × 5 joints + 4 wing
channels. 100% joint coverage on hexapod, quadruped and biped at 3 DOF. URDF import
reads 12 actuated joints / 4 limbs from the example quadruped.

**Gating (F3/F6).** Silencing the dopaminergic pathway moves the circuit, so the
multiplicative compilation is load-bearing rather than decorative: `gate` 3,524 neurons
/ 27.3% mean rate change, `gate_readout` 209 / 11.1%, `compass` 229 / 16.3%,
`descending` 323 / 4.7%, `legs_all` 1,011 / 1.0%.

**Scale (F6).** `legs_all` — 11,790 neurons, 1,569,085 edges — is bit-identical to the
reference for 64/64 ticks and runs 3,076 tick/s (325 µs/tick) on host. int8 costs 0.611%
drive error and 2 sign flips out of 11,790 neurons. It fits 1 of 8 reference devices
(ESP32-S3, on flash).

**Functional probe (F13).** The compass ring is recoverable from `instance` strings: 46
EPG neurons at 16 distinct ring positions. Swept over 250× of global synaptic gain, a
bump forms (R up to 0.582, inside the measured band) and **never persists** — R_held =
0.000 at every scale, with nothing silent and nothing saturated. This reproduces Chang
2023 / Duan 2025 / Beiran & Litwin-Kumar 2025 on a dataset none of them used. The wiring
gives spatial structure, not an attractor.

**Retina (F9).** Using the *measured* acceptance angle Δρ = 8.2° (not the textbook 5°,
which is a Snyder-formula prediction): a 46° lens covers 13.3% of the fly's visual field,
70° covers 26.1%, 120° covers 54.1%, and a 128×128 fisheye at 180° covers 100% with a
*smaller* resampler than the 320×320. Computed, not measured on a sensor.

## Not built — be explicit

| thing | why it matters |
|---|---|
| Dynamics fitting | Every time constant, threshold and gain is a default, not a result. The connectome does not contain them. Until something fits them against behaviour, no circuit here is claimed to *work*, only to run. |
| Functional validation | Nobody has shown the compass holds a bump, that T4/T5 reports motion direction, or that leg circuits produce a gait. Compiling is not working. |
| Hardware | Every device fit is datasheet arithmetic. Nothing has been flashed. There is no measured power number anywhere in this repo. |
| SNN backend | The IR separates additive and multiplicative edges partly so a spiking backend is possible. None is written. |
| Sensor adapters | The ROS 2 node wires the IMU path and stubs the rest. No event-stream reader, no calibration against a physical sensor. |
| Actuator model | `MotorDecode.command()` is `mean(agonist) − mean(antagonist)`. There is no force-per-spike, no calcium filter, no torque model, and the ROS 2 node publishes one value to every joint. This is a placeholder, not a controller. |
| `optic_motion` | 4.9 MB. Fits nothing. Needs pruning that has not been validated. |

## Known sharp edges

- The 62% single-synapse figure is the **raw release table**. The graph you compile is
  40.4% single-synapse, 94.3% ≤ 15, 98.1% ≤ 31. Quoting one for the other understates
  the bit width you need by about one bit. This was caught by a test after being wrong
  in prose first.
- Reproducibility is a **type-pair** measurement, so P(real|w) bounds the pathway, not
  the individual synapse.
- `fit_budget` returns `None` when nothing fits. Do not "fix" this to return the least
  bad option; an over-budget artifact must not look like a success.
- Drive error must be normalised by **total** input magnitude, not the signed sum. The
  signed version explodes on E/I-balanced neurons and overstated damage fourfold before
  it was caught. Report correlation and sign agreement alongside it.
- `winged()` runs at 250 Hz, set by body dynamics, NOT by the ~200 Hz wingbeat. The
  wingbeat is a carrier the controller modulates, not a rate it must resolve.
- Modulatory edges must stay multiplicative. Compiling them as ordinary synapses leaves
  a well-formed graph with the gating silently removed.
