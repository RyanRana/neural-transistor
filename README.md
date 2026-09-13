# flyforge

Compile *Drosophila* connectome circuits into quantized controllers that run on
milliwatt hardware.

The male-CNS connectome is brain **and** ventral nerve cord in one volume: 211,577
annotated bodies, 151,856,684 edges, 311,833,243 synapses. That means the leg circuits,
the compass, the visual system and the command bus between them are all in the same
coordinate space, and you can cut a runnable controller out of any of them.

flyforge is the toolchain for doing that: select a circuit, decide how much of it is
real, compile it to int8 C or a ROS 2 package, and get told what it costs.

```
connectome ──▶ select ──▶ sign ──▶ [ CircuitIR ] ──▶ prune ──▶ quantize ──▶ int8 C
                                        ▲                                └▶ ROS 2 pkg
              morphology (URDF) ────────┤
              sensor (hex retina) ──────┘
```

## Why this fits on a microcontroller at all

The fly's own architecture is a stack of narrow waists, and those waists are the
controller. 89,403 optic-lobe neurons compress to 9,201 visual projection neurons. The
entire brain commands the entire body through **1,314 descending neurons**. The whole
animal acts through **708 motor neurons**. You do not need 180,000 neurons on the robot;
you need the circuit that produces the right command vector.

Measured, from the real data:

| circuit | neurons | edges | int8 flash | RAM |
|---|--:|--:|--:|--:|
| `gate_readout` — MBON valence + dopaminergic gate | 429 | 4,144 | **57 KB** | 4 KB |
| `compass` — EPG/PEN ring attractor | 452 | 54,290 | 188 KB | 4 KB |
| `descending` — the whole brain→body bus | 1,314 | 71,762 | 239 KB | 12 KB |
| `path_integration` — compass + PFN + FC2 + PFL | 1,651 | 125,198 | 381 KB | 5 KB |
| `leg_T1` — one front leg, premotor + MN + proprioceptors | 3,553 | 311,915 | 949 KB | 11 KB |
| `legs_all` — six legs and their coordination | 11,790 | 1,569,085 | 4.7 MB | 104 KB |

`gate_readout` and `compass` fit all eight reference devices, down to an nRF52840.

## The part that is actually new

Connectomics prunes weak edges by convention — keep ≥5 synapses, or ≥10, depending on
the paper. Nobody reports what that costs, because measuring it looks like it needs
ground truth nobody has.

It doesn't. **The fly is bilaterally symmetric, and the two hemispheres were
reconstructed independently.** They are a replicate pair. A real pathway should appear on
both sides; a reconstruction artifact has no reason to. So reproducibility across
hemispheres measures whether a connection is real, with no labels at all.

The design is unusually clean here: 75,215 left neurons vs 75,119 right, 10,940 of
11,230 types present on both sides, and a median per-type cell-count difference of
exactly **0**.

Measured over 19,654,874 ipsilateral typed edges:

| weight | edges | reproduced | P(real) |
|--:|--:|--:|--:|
| 1 | 7,994,902 | 83.9% | **0.786** |
| 2 | 3,718,546 | 90.2% | 0.875 |
| 3 | 2,042,085 | 93.3% | 0.918 |
| 4–5 | 2,166,674 | 95.5% | 0.949 |
| 6–7 | 1,117,581 | 97.0% | 0.970 |
| 8–11 | 1,097,106 | 97.9% | 0.983 |
| 12–19 | 802,046 | 98.6% | 0.993 |
| ≥20 | 715,934 | 99.0% | ~0.999 |
| *shuffled control* | | *28.0%* | |

Reproducibility saturates at 99.2%, not 100% — some real connections are genuinely
unilateral. So an observed rate is a mixture of real pathways reproducing at the ceiling
and noise reproducing at chance, and inverting it gives a per-edge posterior:

```
P(real | w) = (observed(w) − chance) / (ceiling − chance)
```

**About one in five single-synapse edges is spurious, and they are 40% of the compiled
graph.** That turns an arbitrary threshold into a stated confidence:

```bash
flyforge compress compass --p-real 0.95     # → weight ≥ 6, keeps 51% of edges, 90% of synapses
```

Caveat, stated plainly: reproducibility is measured at the level of a *type pair*, so
this is the probability the **pathway** is real, an upper bound on the probability the
individual cell-to-cell edge is. Right quantity for deciding what to compile; wrong
quantity for a claim about one synapse.

## The trap that silently breaks everything

Everyone writes this dict:

```python
{"acetylcholine": +1, "glutamate": -1, "gaba": -1}
```

**Glutamate is inhibitory in the fly CNS but excitatory at the neuromuscular junction.**
Fly motor neurons are glutamatergic. Apply the naive map and you invert the entire motor
output layer — the robot drives every actuator backwards, and nothing in the pipeline
complains, because the graph is still perfectly well-formed.

Measured: the naive map calls **304 of 708** motor neurons inhibitory. flyforge's
`assign_signs` applies the efferent exception and brings that to **10**, reporting all
166 corrections it made.

## Quickstart

```bash
uv venv --python 3.12 && uv pip install -e .
export FLYFORGE_DATA=~/fly-connectome     # the three .feather files

flyforge index                            # build the CSR index (75 s, once)
flyforge noise                            # the reproducibility curve above
flyforge list                             # the circuit library
flyforge extract compass -o compass.fcx
flyforge compress compass --p-real 0.95 --budget-kb 120
flyforge emit compass -o out/ --target mcu
flyforge emit legs_all -o out/ --target ros2 --urdf my_robot.urdf
flyforge morph --urdf my_robot.urdf
flyforge eval
flyforge ui                               # minimal local UI on :8765
```

## Morphology

Motor neurons are annotated by **target muscle** — `Ti extensor MN`,
`Tergopleural/Pleural promotor MN`, `Ta levator MN` — and a muscle name states a joint
and a direction, both of which transfer to any articulated robot. That makes retargeting
principled rather than hand-waved.

```
coxa_yaw     ThC    promotor / remotor          hip yaw
coxa_roll    ThC    abductor / adductor         hip roll
trochanter   CTr    Tr extensor / Tr flexor     hip pitch
knee         FTi    Ti extensor / Ti flexor     knee
ankle        TiTa   Ta levator / Ta depressor   ankle
```

Joints are driven by antagonist pairs (`rate(agonist) − rate(antagonist)`), so
co-contraction produces stiffness rather than motion, as it does in the animal. Pass a
URDF and the limbs and joint roles are read off the robot itself. Hexapod, quadruped,
biped, winged and N-module layouts are built in, and the retargeter reports what it could
not use — a biped leaves four of six leg circuits unbound at the motor boundary, and says
so.

## What is verified, and what is not

**Verified by measurement, under test (`pytest tests/`, 15 passing):**

- The emitted C is **bit-identical to the numpy reference for 64/64 ticks**, on every
  circuit tested — same Q8.8 fixed point, same saturation, same refractory handling
- Emitted C compiles clean under `-Wall -Wextra -Werror`
- int8 costs **0.11–0.37% mean synaptic drive error and zero sign flips**
- Bilateral reproducibility is monotone in synapse count; chance 0.280, ceiling 0.992
- The NMJ correction fires on 166 motor neurons and changes the naive answer
- Extraction is deterministic: identical fingerprint across runs
- `.fcx` survives save/load with its fingerprint intact
- Compass runs **40,445 tick/s (24.7 µs/tick)** on host — the fly's own loop is ~200 Hz
- Motor decoding covers all six legs × five joints, 100% joint coverage on
  hexapod/quadruped/biped
- The budget solver returns `None` rather than an over-budget artifact when nothing fits

**Not yet true:**

- **Dynamics are not fitted.** `Dynamics` defaults are LIF starting points, not results.
  The connectome gives structure; time constants, thresholds and gains are free
  parameters and nothing has tuned them against behaviour yet. Every artifact keeps
  measured and fitted quantities in separate arrays so you can always tell which is which.
- **No circuit has been shown to do its job.** The compass compiles and runs; whether it
  holds a heading bump is untested. That is the next milestone, not a claim.
- **Nothing has run on real hardware.** All device fits are computed against datasheet
  flash/SRAM, not flashed and measured. No power number here is measured.
- The ROS 2 node's sensor adapters are stubs past the IMU path.
- The SNN backend (Loihi / Speck) does not exist; the IR is shaped for it, that is all.
- `optic_motion` at 4.9 MB does not fit any target device without pruning that has not
  been validated.

## Layout

```
flyforge/data       male-CNS loader, CSR + reverse index, cached
flyforge/circuit    selection DSL, extraction, sign assignment, noise model
flyforge/ir         CircuitIR, .fcx container, integer-exact reference runtime
flyforge/quant      pruning, log quantization, delta-indexing, budget solver
flyforge/target     int8 C emitter, ROS 2 package emitter
flyforge/morph      muscle→joint map, URDF import, gait retargeting
flyforge/sensors    hex-lattice retina resampler
flyforge/evals      the claims above, executed
flyforge/api        the local UI
```

## Data

male-CNS v1.0 (`minconf 0.5`), three Feather tables in `$FLYFORGE_DATA`. The index is
built over **annotated bodies only**, which keeps 17.1% of edges and 40.2% of synapses —
the dropped material is overwhelmingly single-synapse fragment noise, and retained edges
average 4.82 synapses against 2.05 across the raw table. That filter is recorded in
every artifact's provenance rather than applied quietly.
