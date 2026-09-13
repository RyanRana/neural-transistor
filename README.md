# Neural Transistor

Compile *Drosophila* connectome circuits into quantized controllers that run on
milliwatt hardware.

Google and Janelia built the map. The query tools let you look at it. This is the
toolchain that takes a piece of it, tells you which parts are real, shrinks it until it
fits in kilobytes, tells you what the shrinking cost, and hands you C you can flash.
See [what already exists vs what this adds](docs/WHERE-THIS-FITS.md).

<img src="docs/img/ui.png" width="100%" alt="neuraltransistor UI">

---

## Install

```bash
git clone <repo> && cd neuraltransistor
uv venv --python 3.12 && uv pip install -e .
export NTX_DATA=~/fly-connectome          # the three male-CNS .feather files
neuraltransistor index                                 # builds the CSR index, ~75 s, once
```

No account, no token. The male-CNS release is public:
`gs://flyem-male-cns/` reads anonymously.

## API

```python
import neuraltransistor as ff

conn = ff.load()                          # 211,577 neurons, 26,028,386 edges (cached)
ir   = ff.circuit(conn, "compass")        # 452 neurons, 54,290 edges, 188 KB

ir, stats = ff.prune(ir, p_real=0.95)     # keep edges 95% likely to be real
ir, rep   = ff.quantize(ir, bits=8)       # log codebook + delta-encoded index
print(rep)   # 188KB -> 74KB (2.6x) | drive err 0.02% | r=1.0000 | 0 sign flips

ff.emit_c(ir, "out/")                     # freestanding C99, no malloc, no libc
```

Cut your own circuit with a selector instead of the library:

```python
valence = ff.circuit(conn, ff.Sel.type(r"^MBON") | ff.Sel.type(r"^PAM"), name="valence")
# valence: 413 neurons, 3,725 edges, 7,408 modulatory
```

Fit a byte budget — it returns `None` rather than hand you an over-budget artifact:

```python
ir, rep, trials = ff.budget(ir, kb=80)
# log w4/i8 prune>=2: 247KB -> 66KB (3.8x) | drive err 6.6% | r=0.9986 | sign agree 98.7%
```

Target a real robot by reading its URDF:

```python
spec, report = ff.morphology(urdf="my_robot.urdf")   # 4 limbs / 12 joints
plan = ff.retarget(conn, spec, gait="trot")          # binds fly leg circuits to limbs
ff.emit_ros2(ir, spec, "out/")                       # ament_python package
```

Run it in Python to check against the device:

```python
rt = ff.Reference(ir)                     # integer-exact, matches the emitted C bit for bit
rt.run(200, drive)
```

## CLI

```bash
neuraltransistor list                             # the circuit library
neuraltransistor noise                            # the reproducibility curve
neuraltransistor extract compass -o compass.fcx
neuraltransistor compress compass --p-real 0.95 --budget-kb 120
neuraltransistor emit compass -o out/ --target mcu
neuraltransistor emit legs_all -o out/ --target ros2 --urdf my_robot.urdf
neuraltransistor eval                             # 33 evals
neuraltransistor ui                               # the page above, on :8765
```

## The library

| circuit | neurons | edges | int8 flash | fits |
|---|--:|--:|--:|---|
| `gate_readout` — MBON valence + dopaminergic gate | 429 | 4,144 | **57 KB** | all 10 |
| `compass` — EPG/PEN ring attractor | 452 | 54,290 | 188 KB | all 10 |
| `descending` — the entire brain→body bus | 1,314 | 71,762 | 239 KB | all 10 |
| `path_integration` — + PFN, FC2, PFL | 1,651 | 125,198 | 381 KB | all 10 |
| `leg_T1` — one front leg, premotor + MN + proprioceptors | 3,553 | 311,915 | 949 KB | 8 |
| `legs_all` — six legs and their coordination | 11,790 | 1,569,085 | 4.7 MB | 1 |

Plus `steering`, `optic_motion`, `looming`, `optic_flow`, `leg_T2/T3`, `gate`.

It fits because the fly's own architecture is a stack of narrow waists: 89,403
optic-lobe neurons compress to 9,201 projection neurons, the whole brain commands the
whole body through **1,314 descending neurons**, and the animal acts through **708 motor
neurons**.

---

## Two things worth knowing

**Bilateral symmetry is a free replicate experiment.** The hemispheres were reconstructed
independently, so cross-hemisphere reproducibility measures whether an edge is real with
no ground truth. That turns the field's arbitrary "keep ≥5 synapses" convention into a
stated confidence.

<img src="docs/img/noise-curve.png" width="100%" alt="bilateral reproducibility">

**The compass forms a bump and cannot hold it.** Swept across 250× of global synaptic
gain, the connectome alone produces a correctly-sized bump under drive and zero
persistence without it. Connectivity is not dynamics.

<img src="docs/img/bump-sweep.png" width="100%" alt="bump gain sweep">

Both in detail, with the numbers and the code: **[docs/FINDINGS.md](docs/FINDINGS.md)**.

## Docs

| | |
|---|---|
| [FINDINGS.md](docs/FINDINGS.md) | the noise model, the motor-neuron sign trap, the bump result |
| [DYNAMICS.md](docs/DYNAMICS.md) | what the connectome does *not* contain, and who has fitted it |
| [TARGETS.md](docs/TARGETS.md) | real chips, sourced power numbers, what has actually flown |
| [SENSORS.md](docs/SENSORS.md) | mapping a camera onto a 886-ommatidium eye |
| [WHERE-THIS-FITS.md](docs/WHERE-THIS-FITS.md) | what Google/Janelia already published, and what this adds |
| [STATUS.md](docs/STATUS.md) | verified vs code-exists vs not built |
| [NOTES.md](NOTES.md) | running log |

## Honesty

`pytest tests/` — 15 passing. `neuraltransistor eval` — 33 passing.

**Verified:** emitted C is bit-identical to the numpy reference for 64/64 ticks on every
circuit up to 11,790 neurons · int8 costs 0.02–0.6% drive error at r ≥ 0.999 · the
motor-neuron sign guard rescues 166 of 708 neurons · extraction is deterministic ·
compass runs 40,445 tick/s on host.

**Not true yet:** no dynamics are fitted, so no circuit is claimed to *work* — the
compass bump result above is the measurement of that gap, not a workaround for it.
Nothing has been flashed to hardware; every device fit is datasheet arithmetic and no
power number here is measured. Sensor and actuator adapters are the thinnest layer in
the repo. Full ledger in [STATUS.md](docs/STATUS.md).
