# Deployment targets

Sourced numbers, and two availability corrections that matter more than any spec.

## Conventional MCUs

| part | SRAM | flash | clock | active | int8 contract | note |
|---|--:|--:|--:|--:|---|---|
| **STM32F405** | 196 KB | 1 MB | 168 MHz | 132 mW | CMSIS-NN | the Crazyflie 2.x flight controller — the only MCU here proven airborne on a sub-30 g robot |
| STM32F401 | 96 KB | 512 KB | 84 MHz | 41 mW | CMSIS-NN | SRAM is the real constraint |
| STM32H743 | 1060 KB | 2 MB | 480 MHz | 363 mW | CMSIS-NN | fast, power-hungry for this class |
| **STM32U575** | 786 KB | 2 MB | 160 MHz | **22.1 mW** | CMSIS-NN | best mW-per-neuron here |
| nRF52840 | 256 KB | 1 MB | 64 MHz | ~15 mW | CMSIS-NN | chosen for the radio, not the compute |
| RP2350 | 520 KB | ext | 150 MHz | 38.7 mW | CMSIS-NN | has DSP but **not** MVE; SMLAD needs adjacent halfwords, so gathers defeat it — budget ~1 MAC/cycle on CSR |
| ESP32-S3 | 416 KB | 8 MB | 240 MHz | ~330 mW | **per-tensor pow2** | breaks the contract: per-channel exists on P4, not plain S3 |
| **Apollo510** | 3750 KB | 4 MB | 250 MHz | **11.7 mW** | CMSIS-NN | most SRAM, lowest power; Helium gives 8 int8 MAC/cycle; no NPU |
| Apollo4 Plus | 2750 KB | 2 MB | 192 MHz | 7.6 mW | CMSIS-NN | sub-10 mW with 2.7 MB SRAM |
| ~~GAP9~~ | 1728 KB | 2 MB | 370 MHz | 20–70 mW | CMSIS-NN | **GreenWaves entered judicial liquidation 14 Jan 2025.** Architecturally ideal, not purchasable. NE16 RTL survives at `pulp-platform/ne16` |

**One contract covers nearly all of them.** CMSIS-NN is bit-exact with the TFLite Micro
reference kernels — int8 weights in [−127,127] with zero-point 0, per-axis on conv; int8
activations per-tensor with a zero-point; int32 per-axis bias — and ST Edge AI, Ambiq's
neuralSPOT, ExecuTorch's Cortex-M backend and GAP9's NE16 all speak it. One emitter
covers STM32, Ambiq, RP2350 and GAP9. Sub-8-bit is a widening inside that contract, not a
new one.

## Neuromorphic, and why none of it is a target yet

| part | weights on silicon | capacity | power | verdict |
|---|---|---|--:|---|
| Loihi 2 | 8 bit max (Loihi 1 was 9 — this went *down*) | 1M neurons | 0.45 W/chip | **closed.** No public SKU; every `lava-nc` repo archived 2026-05-13. Smallest form factor is 108 g — 3.6× an entire sub-30 g mass budget |
| Speck 2e/2f | 8-bit signed | 327,680 neurons, 272 KB kernel memory | 0.5–47 mW | **structurally incompatible.** Fan-out is 2 destinations *per core*, not per neuron — all neurons in a core share them. Per-neuron connectome routing is inexpressible, independent of memory |
| Akida AKD1000 | 1/2/4 bit | 78 NPs, 8 MB | ~905 mW board floor | the only viable SNN target. Real problem is block-sparse partitioning, not fan-in (its 57,334 cap clears our max of 11,526 easily) |
| Xylo | 8 bit | **~31,744 total weights** | 578 µW | true arbitrary-graph engine — and smaller than our smallest circuit |
| Innatera Pulsar | not published | not published | 0.5 mW typ | black-box SDK; licence forbids publishing benchmarks |

**No neuromorphic chip has ever run onboard a sub-30 g robot.** The lightest neuromorphic
flight on record is 856 g.

## The workload mismatch

Every toolchain above is a dense conv/GEMM engine. Ours is sparse CSR SpMV. Densifying
costs, measured against our own circuit sizes:

| circuit | density | cost of densifying |
|---|--:|--:|
| `compass` | 26.6% | **1.06×** |
| `leg_T1` | — | 13× |
| `optic_motion` | — | **502×** |

So `compass` runs on stock dense CMSIS-NN kernels with zero new kernel work, and
everything else needs hand-written sparse kernels that bypass the vendor stack entirely
— at which point SRAM size and gather-load support matter far more than advertised TOPS.

## Power budgets, measured on real platforms

| platform | mass | total power | compute | share |
|---|--:|--:|--:|--:|
| RoboBee | 80 mg | 19 mW | ~10 mW budgeted | ~53% — and it still does not fit |
| Crazyflie 2.1+ | 29 g | ~1.83 W | GAP8 at 64 mW | **3.5%** |

The binding constraint is **mass, not milliwatts**. Palossi's endurance breakdown: 440 s
→ 350 s carrying the AI-deck unpowered → 340 s running it. The 4.4 g shield costs 0.8% of
power and 20% of endurance. Battery specific energy also collapses with scale — 142 → 111
→ 79 Wh/kg as cells go 5 g → 0.38 g — which is why everything below ~1 g runs on laser,
solar, RF or tether.

## What has actually flown

| chip | platform | mass | evidence |
|---|---|--:|---|
| STM32F405 | Crazyflie 2.x | 27–29 g | product |
| **GAP8** | PULP-Dronet on Crazyflie, 64 mW, 3.5% of budget | 27 g | arXiv:1805.01831 — the only sub-30 g onboard DNN inference on record |
| Speck | JetHexa hexapod, 2.7 mW | kg-class | *Sci. Robotics* 10.1126/scirobotics.ads3968 |
| Speck | quadrotor, event camera, 2.3 ms latency | 856 g | NeurIPS 2025 — lightest neuromorphic flight |
| Loihi 2 | — | — | **nothing verified onboard** |

## Recommendation

One int8 per-channel affine emitter (covers STM32 / Ambiq / RP2350), one sparse-CSR
kernel emitter that ignores vendor toolchains entirely, and Akida as the single SNN
target behind a block-sparse partitioner. Loihi 2, Speck and GAP9 should not be backend
targets in 2026 — respectively closed, structurally incompatible, and liquidated.

---

*Numbers tagged in the source research as datasheet / vendor-claim / measured-in-paper.
Anything vendor-claimed should be re-checked against the primary PDF before it hardens
into a design assumption — the literature sweep that produced this table caught automated
PDF summarisation fabricating two plausible-looking power figures.*
