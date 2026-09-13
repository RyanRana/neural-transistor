# The biology, for people who want it

Everything else in these docs is written for engineers. This page is the neuroscience,
kept here so it stays out of the way.

You do not need any of this to use the toolkit. `ntx list` tells you what each circuit
does; this tells you what it *is*.

## The dataset

Male CNS v1.0 — the fruit fly's brain **and** ventral nerve cord reconstructed in one
volume, from Janelia FlyEM with Google Research's segmentation. 211,577 annotated bodies,
151,856,684 connections, 311,833,243 synapses. Public, no account required.

This matters over the better-known FlyWire because FlyWire is brain-only. Leg circuits
live in the nerve cord, so anything about walking needs this dataset.

## What each circuit actually is

| toolkit name | anatomy |
|---|---|
| `compass` | Central complex heading system. EPG neurons tile a ring in the ellipsoid body and carry a single activity bump encoding heading; PEN neurons rotate the bump in proportion to angular velocity; Delta7 provides the inhibition that keeps it to one bump; ER (ring) neurons gate visual input in. |
| `odometry` | The above, plus PFN neurons carrying optic-flow and wind vectors, hΔ/vΔ columnar cells accumulating them, FC2 holding a goal direction, and PFL neurons reading out a turn. |
| `steering` | FC2 → PFL3 → descending. The goal-to-turn readout on its own. |
| `motion` | The elementary motion detector: lamina monopolar cells L1–L5, medulla transmedullary cells Mi/Tm, and the T4 (ON) and T5 (OFF) direction-selective cells that correlate neighbouring columns. |
| `collision` | Lobula columnar feature detectors. LC4 responds to angular velocity, LPLC2 to angular size; both converge on the giant fiber. |
| `eye` | The above wired to its input: lamina → medulla → T4/T5 → LC4/LPLC2 → **DNp01**, the giant fiber itself. |
| `flow` | Lobula plate tangential cells (HS/VS), which act as matched filters for particular self-motion patterns. |
| `leg_T1/T2/T3` | One thoracic neuromere: premotor interneurons, ~86 leg motor neurons, and the chordotonal/campaniform proprioceptors that feed them back. |
| `legs` | All three neuromeres plus the intersegmental interneurons that couple them. |
| `commands` | All 1,314 descending neurons — the entire brain-to-nerve-cord channel. |
| `gate` | Mushroom body. Kenyon cells carry a sparse odour/context code; 97 MBONs read out learned valence; PAM/PPL1 dopaminergic neurons set the gain of the KC→MBON synapses; APL (two cells, one per side) provides the global inhibition that keeps the code sparse. |

## Two naming traps that cost real time here

**`GF*` is not the giant fiber.** The types named GFC1–GFC4 are giant-fiber-*coupled*
interneurons in the nerve cord. Their inputs are VNC interneurons; they receive nothing
visual. The giant fiber is **DNp01** — two cells, one per hemisphere. Wiring the wrong
one gives a circuit whose output has no connection to vision and no error to tell you so.

Once corrected, the connectome independently confirms the published model: DNp01's two
largest inputs are **LC4 (6,362 synapses)** and **LPLC2 (4,710)**, exactly the two terms
in von Reyn's `v_GF = 1.62·LC4 + 1.45·LPLC2 + …`.

**Glutamate flips sign at the muscle.** Glutamate is inhibitory in the fly's central
nervous system and *excitatory* at the neuromuscular junction, and fly motor neurons are
glutamatergic. The usual `{acetylcholine: +1, glutamate: −1, GABA: −1}` map therefore
inverts the entire motor output layer — 304 of 708 motor neurons come back labelled
inhibitory — and nothing errors, because the graph stays well-formed. `assign_signs`
applies the efferent exception and reports its 166 corrections.

## Why sizes are what they are

The nervous system is a series of narrowing stages, and that is why a useful circuit fits
in kilobytes rather than megabytes:

- 89,403 optic-lobe neurons compress to 9,201 projection neurons leaving the eye
- the entire brain commands the entire body through **1,314** descending neurons
- the animal acts on the world through **708** motor neurons
- the complete learned good/bad signal is **97** MBONs
- global inhibition for sparse coding is **2** cells

You never need to run 180,000 neurons. You need the stage that produces the command.

## The eye

~892 ommatidia per eye in a hexagonal lattice, interommatidial angle 4.63°. The
acceptance angle is **8.2°**, not the 4.5–5.7° textbooks quote — that figure is a
prediction of the Snyder diffraction formula, while every intracellular recording gives
7.7–9.5°. So the fly is a heavily *blurred* sampler and its acceptance functions overlap
their neighbours substantially, which makes a modest camera a better match than you would
expect. See [SENSORS.md](SENSORS.md).

## Reading list

The models this toolkit validates against, and where its default parameters come from:

- **von Reyn et al. 2017** *Neuron* 94:1190 and **Ache et al. 2019** *Curr Biol* 29:1073 — the giant-fiber escape model, reproduced in `neuraltransistor.stimuli.giant_fiber`
- **Shiu et al. 2024** *Nature* 634:210 — whole-brain leaky integrate-and-fire; source of the default membrane parameters
- **Lappalainen et al. 2024** *Nature* 634:1132 — connectome-constrained network for the optic lobe; source of the per-cell-type parameterization
- **Chang, Huang & Lo 2023** *J Comp Physiol A* 209:721 — swept 176,400 parameter sets and found weights proportional to synapse count never work
- **Eckstein et al. 2024** *Cell* 187:2574 — the neurotransmitter predictions, 87% per synapse
- **Seelig & Jayaraman 2015** *Nature* 521:186 — the heading bump, and the width our compass probe is measured against
