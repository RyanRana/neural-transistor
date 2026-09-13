# notes

running log of things that turned out to be true. one line each.

- the fly is bilaterally symmetric so left vs right is basically a free replicate experiment, you can measure connectome noise with no ground truth at all
- single synapse edges are 41% of the graph and about 21% of them are junk, that is the entire noise story in one number
- shuffling post types within a hemisphere gives a 28.0% chance floor and reproducibility tops out at 99.2%, everything between those two is a mixture you can just invert
- so p(real | weight) falls straight out, 0.786 at one synapse, 0.949 at four, past 0.99 by twelve
- asking for 95% confidence means weight >= 6 which throws away 81% of edges, that is a 5x compression knob with an actual number attached instead of a vibe
- the naive glutamate equals inhibitory map silently flips every motor neuron because glutamate is excitatory at the insect nmj, 42.9% of vnc motor neurons come back wrong and nothing errors
- indices dominate a sparse connectome layer not weights, int8 to int4 only buys 17% but dropping an edge kills its index too so pruning is worth about 3x what requantizing is
- 62% of edges in the raw release are a single synapse, 40% once you keep only annotated bodies, and the compiled graph wants about one more bit than the raw numbers suggest, 4 bits covers 94% and 5 bits covers 98%
- careful quoting that 62% number, it is the raw table not the graph you actually compile, they are different populations and i mixed them up once already
- the entire brain to body command bus is 1314 descending neurons, that is the whole api surface between the brain and the legs
- the mbon valence readout is 97 neurons, the fly's complete learned good/bad signal fits in about 57kb with its dopaminergic gate attached
- apl is literally 2 neurons, one per hemisphere, doing global inhibition for sparse coding
- building the csr index costs 75s once and turns a 140s scan into a 0.02s extract, should have done it first
- restricting to annotated bodies drops 83% of edges but keeps 40% of synapses, what you lose is almost entirely single synapse fragments
- dopaminergic edges have to be multiplicative not additive, compile them as ordinary synapses and the gating behaviour just quietly disappears
- one front leg is 3553 neurons and fits in 949kb before any pruning, the compass is 452 neurons and 188kb
- best way to prove the gating actually compiles is to silence it and see if anything moves, 3524 neurons shift in the mushroom body so it is real
- emitted c matches the numpy reference bit for bit for 64 ticks which is the only reason i trust any of the size numbers
- a 70 degree camera only sees 22.8% of what a fly sees, you need like 120 degrees minimum before the optic circuits make sense
