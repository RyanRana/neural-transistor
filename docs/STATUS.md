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
| F9 | Hex-lattice retina resampler | **verified** (exact retinotopy, 99.9%) |
| F10 | Local UI | **verified** |
| F11 | Eval suite | **verified** (31 evals) |
| F12 | Dynamics fitting against behaviour | **built — has not closed F13** |
| F13 | Functional validation: compass holds a bump | **measured — negative** |
| F13b | Functional validation: looming detector | **verified** |
| F14 | SNN backend (Loihi / Speck) | **not built** |
| F15 | On-silicon flash + power measurement | **not built** |
| F16 | Cross-compile + boot on an emulated Cortex-M | **verified** |

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

**Morphology (F7).** 34 motor decodes from `legs`: 3 leg pairs × 5 joints + 4 wing
channels. 100% joint coverage on hexapod, quadruped and biped at 3 DOF. URDF import
reads 12 actuated joints / 4 limbs from the example quadruped.

**Gating (F3/F6).** Silencing the dopaminergic pathway moves the circuit, so the
multiplicative compilation is load-bearing rather than decorative: `gate` 3,524 neurons
/ 27.3% mean rate change, `valence` 209 / 11.1%, `compass` 229 / 16.3%,
`commands` 323 / 4.7%, `legs` 1,011 / 1.0%.

**Scale (F6).** `legs` — 11,790 neurons, 1,569,085 edges — is bit-identical to the
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

**Cross-compile and boot (F16).** Every circuit below is built by `arm-none-eabi-gcc`
for Cortex-M, linked, and booted under `qemu-system-arm`, where it recomputes the FNV-1a
digest of 64 ticks of spikes derived from the numpy reference. Sizes are
`arm-none-eabi-size` on the linked ELF — the linker's, not arithmetic.

| circuit | P(real) | core | flash | RAM | digest |
|---|--:|---|--:|--:|---|
| `valence` | 0.90 | Cortex-M3 | 51.2 KB | 4.6 KB | matches |
| `commands` | 0.95 | Cortex-M3 | 79.8 KB | 14.1 KB | matches |
| `compass` | 0.95 | Cortex-M3 | 111.4 KB | 4.9 KB | matches |
| `leg3` | 0.95 | Cortex-M7 | 289.9 KB | 34.9 KB | matches |
| `collision` | 0.97 | Cortex-M3 | 763.2 KB | 80.1 KB | matches |
| `legs` | 0.95 | Cortex-M3 | 1190.3 KB | 126.7 KB | matches |

The linker also corrected a headline number. For `compass` the emitter estimated 4.0 KB
of RAM and the linker reports 4.9 KB. The estimate was right about the kernel — state
struct 2,260 B plus accumulator 1,808 B = 4,068 B, exactly — and omitted the 904-byte
input buffer the *caller* must supply, one int16 of drive per neuron. That is 18% of
total RAM on this circuit, and the published figure should have included it.

Removing the last libc dependency was a prerequisite: the runtime used to
`#include <string.h>` for three `memset` calls, which is free on a hosted build and does
not compile against a bare-metal toolchain shipping no newlib — the exact configuration
the code exists for.

**Looming (F13b).** The connectome-derived giant fiber (DNp01, *not* the GF* types, which
are giant-fiber-coupled interneurons in the nerve cord with no visual input) fires 56 ms
before contact. Onset scales with l/|v| at r = −0.9999 and angular size at onset holds at
19.9 ± 2.7° across an 8× range of approach speeds — linear l/|v| scaling plus a constant
angular threshold are the two defining signatures of a biological looming detector, and
both fall out of unfitted connectivity. The threshold is ~20° where the published giant
fiber is ~40°, a gain calibration rather than a structural failure. This works where the
compass does not because it is feed-forward: no attractor is required.

**Dynamics fitting (F12).** `neuraltransistor.train` fits per-cell-type biophysics on a
differentiable spiking backend; structure and sign are never touched. `ntx fit compass`
runs it. It has **not** made the compass hold a heading, and the honest account of why is
below.

## The compass, in detail

The negative result stands, but its cause is now measured rather than assumed, and it is
not the one the earlier framing implied.

**The ring-attractor kernel is present in the connectome.** Measuring the two-hop
effective weight between EPG neurons as a function of their separation on the ring:
EPG→PEG→EPG peaks sharply at Δθ = +10° (658 against ~120 elsewhere), and EPG→Δ7→EPG is
inhibitory everywhere *except* near zero (−212 at ±10° against −1300 further out). The
net kernel is **positive only at +10° and negative at every other angle** — a Mexican
hat, which is exactly what a ring attractor requires. "Connectivity is not dynamics" was
never a claim that the wiring lacks the structure. The structure is there.

**What is missing is the biophysics, and fitting it has not yet succeeded.** Six defects
in the fitting setup were found and fixed, each of which was independently preventing
convergence:

1. The trainable model had **no refractory period** while the emitted C enforces four
   ticks. That let the fit buy persistence with duty cycles of 1.0 against a hardware
   ceiling of 0.2. It produced R_held = 0.614 that way, and that number was not real:
   the solution could not be emitted. A fitted solution that cannot be emitted is not a
   fitted solution, so the trainable model now enforces the same refractory the kernel does.
2. `target_activity = 0.10`, commented "flies run near 10% active", conflated the
   *fraction of neurons active* with the *per-neuron duty cycle* the loss actually
   measures. At dt = 0.5 ms that target is 200 Hz against a refractory ceiling of 0.2, so
   "hit the rate target" and "saturate" were nearly the same instruction. It is now a
   firing rate in Hz.
3. The anti-silence term was `relu(0.3·target − rate)²` in squared-rate units, maxing out
   at 9e-4 against a persistence term of 0.42 — about 470× too weak to prevent the
   collapse it existed to prevent.
4. Drift was computed on a silent ring, where the preferred angle is undefined. That
   noise was the single largest term in the loss (1.10 of 2.39) while held activity was
   exactly zero.
5. All inhibition shared **one** synaptic scale. ER puts 124k inhibitory synapses onto
   EPG against Δ7's 4.3k, so a single knob had to choose between killing the bump and
   deleting its surround. Scales are now per presynaptic cell type.
6. Saturation was penalised per neuron, so ER — 282 of the circuit's 452 cells — could
   pin at the ceiling while the neuron-mean stayed near zero, because most ER cells were
   quiet. It is now penalised per cell type.

After all six, the fit reduces its loss by more than an order of magnitude and still does
not produce a persistent bump: R_held sits around 0.05–0.12 against a target of 0.65, and
R_driven degrades below its *unfitted* value of 0.61 in the process. The remaining
difficulty is optimisation, not structure — 300 ticks of backpropagation through a
spiking network whose activity dies early gives the persistence term almost no gradient
to act on, and a hold-length curriculum was not enough to fix it. A search that does not
rely on long-horizon gradients is the obvious next thing to try.

## Not built — be explicit

| thing | why it matters |
|---|---|
| Dynamics fitting | Every time constant, threshold and gain is a default, not a result. The connectome does not contain them. Until something fits them against behaviour, no circuit here is claimed to *work*, only to run. |
| Functional validation | The looming detector is validated against its published signatures. Nobody has shown the compass holds a bump, that T4/T5 reports motion direction, or that leg circuits produce a gait. Compiling is not working. |
| On-silicon execution | Cross-compile, link and boot are real and checked in CI, but the boot is QEMU: it models the ISA, not flash timing, peripherals or the clock tree. Nothing has run on a physical part. |
| Power | No measured figure exists. `energy_estimate` is datasheet arithmetic and says so in every field it returns. An emulator cannot measure energy; a shunt (`ina219`/`ina226`) on real hardware can. |
| SNN backend | The IR separates additive and multiplicative edges partly so a spiking backend is possible. None is written. |
| Sensor adapters | The ROS 2 node wires the IMU path and stubs the rest. No event-stream reader, no calibration against a physical sensor. |
| Actuator model | `MotorDecode.command()` is `mean(agonist) − mean(antagonist)`. There is no force-per-spike, no calcium filter, no torque model, and the ROS 2 node publishes one value to every joint. This is a placeholder, not a controller. |
| `motion` | 4.9 MB. Fits nothing. Needs pruning that has not been validated. |

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
