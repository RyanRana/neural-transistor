# What already exists, and what this adds

A fair question to ask of any project sitting on top of someone else's dataset. Short
answer: the reconstruction, the storage, the viewers and the query APIs are all solved
and published, and this project uses them rather than rebuilding them. What is missing
is everything between "you can query the wiring diagram" and "it is running on a chip."

## Already published — used here, not rebuilt

| what | who | this project's relationship |
|---|---|---|
| **The connectome itself** — male-CNS v1.0, FlyWire, hemibrain, MANC | Janelia FlyEM, Google Research, Princeton | consumed as-is. `ntx fetch` pulls the released Feather tables |
| **EM → wiring diagram** — flood-filling networks, SegCLR, the `connectomics` library | Google Research | never touched. We start where they finish |
| **TensorStore, neuroglancer** | Google | not needed for connectivity work |
| **CAVE / Codex, neuPrint, navis, neuprint-python** | Princeton, Janelia, Cambridge | these are the *query* layer. We read the bulk release instead because a compiler wants the whole graph at once, not per-neuron API calls |
| **Neurotransmitter predictions** (87% per synapse, 94% per neuron) | Eckstein et al. 2024 | consumed directly. Our contribution is the efferent-exception correction on top |
| **Cell type / muscle / column annotations** | Janelia + community curation | consumed directly, and they carry more than people use — the muscle names are what make morphology retargeting principled |
| **Published dynamics** — Shiu LIF params, the giant-fiber model, Borst's temporal filters | Scott, von Reyn, Ache, Borst labs | used as defaults and as **validation targets**. `stimuli.giant_fiber` is their model, implemented faithfully so ours can be checked against it |

## Already published — simulation, on big computers

Two landmark papers run connectomes as models. Both are excellent and neither is trying
to do what this does.

- **Shiu et al. 2024** (*Nature*) — leaky integrate-and-fire over the whole fly brain,
  127,400 neurons in Brian2. A single global synaptic scale, float, workstation-class.
- **Lappalainen et al. 2024** (*Nature*, `flyvis`) — a connectome-constrained deep network
  for the optic lobe. 734 fitted parameters over 45,669 neurons, trained by backprop,
  PyTorch on a GPU.

Both are **simulation studies**: the artifact is a scientific result, and the compute
budget is a workstation. Neither produces something you can put on a robot, and neither
claims to.

## Not published by anyone — what this adds

1. **A compiler.** Connectome subgraph → signed sparse CSR → pruned → log-quantized →
   freestanding C99 with no malloc and no libc, bit-identical to a numpy reference.
   Nothing else takes a connectome to a deployable artifact.
2. **A measurement of what compression costs.** Drive error, correlation and sign
   agreement per bit-width and per pruning threshold, on real circuits. Previously
   nobody needed this number because nobody was shrinking these graphs.
3. **A label-free noise model.** Using bilateral symmetry as a replicate pair to get
   P(real | synapse count) with no ground truth. This is the piece most likely to be
   useful to connectomics people who care nothing about robots — it replaces the field's
   arbitrary "keep ≥5 synapses" convention with a stated confidence.
4. **Morphology retargeting from annotation.** Motor neurons are labelled by the muscle
   they innervate, and a muscle name states a joint and a direction. That transfers to
   any URDF, which is what makes "any robot" more than a slogan.
5. **Functional probes that are allowed to fail.** The compass gain sweep is a
   measurement, not a demo. It reports that the wiring alone produces a bump and no
   attractor.

## Who is closest

- **Pugliese, Tuthill & Brunton (2025, bioRxiv)** found a three-neuron rhythm generator
  in the nerve cord — connectome-derived, and the closest thing to a controller. Their
  own caveat: *"phase coupling was absent across the six leg CPGs."* Rhythm, no
  coordination, no body.
- **Nobody has driven a physical robot from a fly connectome.** Verified four ways
  during this project's research pass, including a full-text sweep of Europe PMC for
  "connectome-constrained" (25 results, zero robots).

So the gap is real, and it is a *systems* gap rather than a science gap: the biology and
the data are published, and what is missing is the boring engineering that turns them
into bytes on a microcontroller.

## The one-paragraph version

Google and Janelia built the map. The query tools let you look at the map. The
simulation papers show the map can be run as a model on a workstation. This project is
the toolchain that takes a piece of that map, tells you honestly which parts of it are
real, shrinks it until it fits in kilobytes, tells you what the shrinking cost, and hands
you C you can flash.
