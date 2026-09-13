# What the connectome does not contain

A connectome is a strong prior, not a specification. It fixes the coupling *pattern* and
— at roughly 87% per-synapse, 94% per-neuron accuracy (Eckstein et al. 2024) — the
*sign*. Everything with units of time, voltage or gain is free.

This page is the list of what is free, how much it matters, and who has pinned it down.
flyforge keeps measured and fitted quantities in separate arrays precisely so this
boundary stays visible: `CircuitIR.weight` is measured, `CircuitIR.dynamics` is not.

## The parameterization that works

The naive framing is one weight per edge: **151.9M free parameters**. Hopeless. One scale
per type-pair: 934,196. Still bad.

Every successful published model instead parameterizes **per cell type** — one membrane
τ, one resting level, one gain each, plus a unitary synaptic scale per type-pair:

| model | scope | free parameters |
|---|---|--:|
| Lappalainen et al. 2024 (flyvis) | 45,669 optic-lobe neurons, 65 types | **734** |
| Borst 2025 | same 65 types | 130 |
| Duan et al. 2025 | 439 CX neurons, 6 types | 57 |
| Biswas et al. 2026 | EPG–Δ7 ring | **4** |

Litwin-Kumar & Turaga 2019 state the principle directly: infer *"gains, thresholds, and
time constants… rather than connection strengths"*, which *"reduces the number of
unconstrained parameters from O(N²) synaptic weights to O(N) biophysical parameters."*

An unconstrained RNN over flyvis's 45,669 neurons would need 2.2×10⁹ parameters. The
connectome buys six orders of magnitude. It does not buy the last one.

## The result that should stop you

Chang, Huang & Lo (2023) swept **176,400 parameter sets** over conductance-based models
of the central complex:

> *"Although it is tempting to use such information to indicate the synaptic weights, the
> synaptic numbers revealed in the connectomic data are highly variable… **All tested
> models failed if we simply set the synaptic weights proportional to these numbers.**"*

We reproduced the shape of this on male-CNS independently — see
[FINDINGS.md §3](FINDINGS.md#3-the-compass-forms-a-bump-and-cannot-hold-it). Across 250×
of global synaptic gain the compass forms a bump and never holds it.

Two results that look contradictory and are not:

- **Biswas et al.**: synapse counts can vary ±90% and valid scale factors can almost
  always be re-found.
- **Chang et al.**: 2% Gaussian noise on *hand-tuned* weights breaks the circuit.

The structure is robust *if you are allowed to re-solve the scales*; a fixed
parameterization is fragile. **Consequence for this toolkit: a quantizer must re-fit
per-type scale factors after quantizing, not quantize a fixed solution.** That is not yet
implemented and is the main reason F12 is open.

## The free parameters, ranked by how much they change behaviour

| # | parameter | count here | range in the literature | quantization note |
|--:|---|--:|---|---|
| 1 | **sign** of ambiguous connections | 38,012 neurons (23.1%) are glutamate/histamine/monoamine | binary | 1 bit — but Δ7 (conf 0.64) and Mi9 (conf 0.75) are the two most consequential and least certain calls in the fly |
| 2 | **synapse count → weight scale** | 1 global, 3 per-NT, or ~934k per type-pair | Shiu 0.275 mV/syn; Liao & Lo ACh 0.2 / GABA 0.09 / Glu 0.04 | a single global scalar demonstrably fails; log-quantize |
| 3 | membrane τ | 1 per cell type | converges tightly: **20 ms** (Shiu, Pisokas, Kakaria), 15 (Chang), 40 (Borst), 50 (Lappalainen init) | narrow range, 4–6 bits ample |
| 4 | synaptic τ | 1 per transmitter | GABA_A 5, ACh 20, NMDA 100 ms | sets max integrable angular velocity |
| 5 | input filter τ (optic lobe) | 2 per type | τ_LP 14–107 ms, τ_HP 127–391 ms | **four types have no highpass at all, and that bit is invisible in the wiring** |
| 6 | transfer function | 2–4 per type | ReLU / sigmoid / softplus / ELU | **categorical, not scalar** — Westeinde's steering result does not exist if f is linear |
| 7 | reversal potentials | 3–4 global | measured: **E_ACh = −21 mV**, E_Glu −71, E_GABA −68, E_leak −65 (Groschner 2022) | most models wrongly assume 0 or +60 for ACh |
| 8 | E/I balance, self-excitation | ~2 per recurrent circuit | Kim 2017 α=+3 vs Kim 2019 α=−7.76 — same lab, same circuit, both make bumps | ring attractors have **N−3 discrete optimal J_E values**, i.e. natively quantized |
| 9 | motor / steering gain | 1 per output | spans two orders of magnitude, 0.03 → 25 | log-quantize, always paired with a clip |
| 10 | noise amplitude | 1–3 global | ≤1% fine, **10% is the breaking point** | sets the drift rate, which *is* the headline metric, and is free in every model |

The single clearest illustration of how loosely these are pinned: Kim 2017 uses α = +3,
D = 0.1, β = 20. Kim 2019 uses α = **−7.76**, D = 5.19, β = 1.96. Same lab, same circuit.
Both work.

## Defaults used here, and where they come from

`Dynamics.default()` uses the Shiu et al. 2024 whole-brain LIF values, which are the only
complete published parameter set for a fly-brain-scale model:

```
V_rest = V_reset = −52 mV      τ_membrane   = 20 ms
V_threshold     = −45 mV       τ_refractory = 2.2 ms
R = 10 MΩ, C = 2 nF            τ_synaptic   = 5 ms
                               W_syn = 0.275 mV   ← their single free parameter
```

Shiu's own robustness check: ±30% on W_syn preserves the qualitative prediction in
90.2% of 164 experiments, and a shuffled connectome scores 1/100 against 100/100. That
model was validated on taste→motor and grooming — **never on a central-complex task**,
and its uniform W_syn plus binary E/I are exactly the two assumptions Chang showed break
ring attractors.

So: these defaults are a starting point for a fitter. They are not a result, and nothing
in this repo claims a circuit works because it ran with them.

## The empirical warrant for weight ∝ synapse count

Liu et al. 2022 (*Curr Biol* 32:559) measured it: synapse **density** predicts somatic
uEPSP amplitude at **r² = 0.77** overall, 0.84 for dendritic connections. So 16–23% of
physiological strength is unexplained by count, and the relationship is linear in
*density*, not raw count. Axo-axonic connections deviate badly — compartment matters.

## Open ground

Nobody has published a dynamical model of the male-CNS central complex, optic lobe or
VNC. Both connectome-constrained optic-lobe models run on a 65-type, 7-column FIB-25
medulla reconstruction — a small fraction of what this dataset holds. flyvis contains no
LC type at all; this release has **4,253 LC neurons across 48 types**.

For walking specifically, the gap has a crisp statement. Pugliese et al. (2025) found a
three-neuron rhythm generator in the nerve cord — and reported that *"phase coupling was
absent across the six leg CPGs in the full connectome simulation."* **Connectome plus
generic biophysics gives rhythm but not inter-leg phase.**
