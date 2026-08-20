# Results

This document records the controlled experiments in [METHOD.md](METHOD.md).
It reports teacher-forced held-out likelihood only. Generation samples remain
useful qualitative evidence, but they are not substituted for BPB.

## Reading The Tables

Lower bits per UTF-8 byte (BPB) is better. Values are the mean over three
independently trained seeds, `13`, `29`, and `47`. `+/-` denotes the sample
standard deviation. All reported models were trained from scratch for 6,000
updates on the frozen corpus splits.

Tiny Shakespeare uses full-context teacher-forced evaluation. enwik8 uses
block-reset teacher-forced evaluation because its test split contains five
million bytes. BPB values are not comparable across the two corpora. The
lossless enwik8 tokenizer represents all source bytes before evaluation. Raw
JSON reports retain each evaluator's scored-byte denominator, which differs
from the complete test size only at an initial or incomplete context boundary.

The 6.5M Character GPT and Byte GPT rows below use the nanoGPT architecture
with the historical ILM AdamW protocol: all-parameter weight decay, constant
learning rate `1e-3`, AdamW betas `(0.9, 0.999)`, and no gradient clipping.
The full optimizer crossover is reported separately. It shows why direct
cross-implementation differences should be read as reference comparisons,
rather than component-level causal estimates.

## Optimizer Crossover

The 6.5M crossover evaluates Character or Byte GPT and Flat ILM under both
optimizer protocols. The ILM protocol uses all-parameter decay and a constant
learning rate. The nanoGPT protocol decays matrices only, uses a 100-step
warmup followed by cosine decay from `1e-3` to `1e-4`, uses betas `(0.9,
0.99)`, and clips gradients at `1.0`.

| Corpus | Reference GPT, ILM AdamW | Flat ILM, ILM AdamW | Reference GPT, nanoGPT AdamW | Flat ILM, nanoGPT AdamW |
| --- | ---: | ---: | ---: | ---: |
| Tiny Shakespeare | 2.650571 +/- 0.037308 | 2.150793 +/- 0.007531 | 2.716750 +/- 0.016752 | 2.122007 +/- 0.004429 |
| enwik8 | 2.479660 +/- 0.016049 | 2.333131 +/- 0.004498 | 2.640124 +/- 0.003919 | 2.308238 +/- 0.001876 |

Flat ILM is lower than the corresponding reference GPT under both protocols.
The mean matched-protocol differences are `0.499778` and `0.594743` BPB on
Tiny Shakespeare, and `0.146529` and `0.331886` BPB on enwik8, for the ILM and
nanoGPT protocols respectively. The magnitude is therefore optimizer
sensitive, particularly on enwik8. The crossover supports a stable ordering in
these conditions, but it does not make Byte or Character GPT versus ILM a
pure representation-only ablation.

The semantic-code permutation and Flat-to-Full comparisons remain more direct
controls because they use the same ILM implementation and optimizer protocol.

## 6.5M Results

The Tiny Shakespeare tokenizer maps 15,030 lexical entries to three base-64
coordinates. Permuted Flat uses the same final code set and lexical segmentation
as Flat ILM, but applies the fixed code permutation with seed `314159`.

| Tiny Shakespeare family | Parameters | Mean BPB +/- SD | Seed BPB values |
| --- | ---: | ---: | --- |
| Character GPT | 6,525,600 | 2.650571 +/- 0.037308 | 2.665362, 2.608136, 2.678216 |
| Atomic Lexical | 6,469,374 | 2.320493 +/- 0.024002 | 2.305153, 2.308174, 2.348153 |
| Permuted Flat | 6,555,064 | 2.296561 +/- 0.011518 | 2.289941, 2.309861, 2.289881 |
| Flat ILM | 6,555,064 | 2.150793 +/- 0.007531 | 2.159404, 2.147534, 2.145441 |
| Full ILM | 6,631,992 | **2.120122 +/- 0.008608** | 2.123232, 2.126744, 2.110392 |

| enwik8 family | Parameters | Mean BPB +/- SD | Seed BPB values |
| --- | ---: | ---: | --- |
| Byte GPT | 6,567,600 | 2.479660 +/- 0.016049 | 2.496671, 2.477523, 2.464786 |
| Permuted Flat | 6,561,064 | 2.504098 +/- 0.009986 | 2.493390, 2.505747, 2.513156 |
| Flat ILM | 6,561,064 | 2.333131 +/- 0.004498 | 2.336511, 2.334855, 2.328025 |
| Full ILM | 6,676,456 | **2.235895 +/- 0.006256** | 2.235785, 2.242206, 2.229695 |

At this tier, embedding-derived Flat ILM improves over Permuted Flat by
`0.145768` BPB on Tiny Shakespeare and `0.170967` BPB on enwik8. Full ILM
improves over Flat ILM by `0.030671` and `0.097236` BPB respectively.

## 15.5M Results

The larger tier increases width while retaining six layers, six heads, the
same dropout, batch size, corpus splits, and 6,000-update horizon. The main
comparison at this tier is within the ILM family. Atomic Lexical remains
available on Tiny Shakespeare. We do not foreground the Byte or Character GPT
rows because an optimizer crossover was completed only at the 6.5M tier.

| Tiny Shakespeare family | Parameters | Mean BPB +/- SD | Seed BPB values |
| --- | ---: | ---: | --- |
| Atomic Lexical | 15,537,630 | 2.384029 +/- 0.006054 | 2.377300, 2.385753, 2.389034 |
| Flat ILM | 15,483,532 | 2.140613 +/- 0.009786 | 2.135988, 2.133997, 2.151854 |
| Full ILM | 15,601,932 | **2.135076 +/- 0.002098** | 2.132676, 2.136556, 2.135997 |

| enwik8 family | Parameters | Mean BPB +/- SD | Seed BPB values |
| --- | ---: | ---: | --- |
| Flat ILM | 15,492,772 | 2.292757 +/- 0.009464 | 2.300834, 2.282343, 2.295094 |
| Full ILM | 15,670,372 | **2.181544 +/- 0.001983** | 2.182295, 2.179294, 2.183041 |

Full ILM remains lower than Flat ILM by `0.005537` BPB on Tiny Shakespeare and
`0.111213` BPB on enwik8. The different changes from 6.5M to 15.5M are
fixed-horizon observations. They do not establish a scaling law or a claim
that the two corpora have the same optimization behavior.

The Permuted Flat control is reported only at 6.5M. It establishes that code
organization affects the observed outputs at a controlled parameter scale on
both corpora. It does not establish that the exact effect size must persist at
all parameter scales.

Atomic lexical modeling is not included on enwik8 because its lossless lexical
vocabulary contains 519,955 entries. Learned atomic input and output
interfaces would dominate a 6.5M parameter budget, and reducing the
Transformer width enough to fit that budget would make the control less
informative.

## What The Current Evidence Supports

The repeated semantic-versus-permuted-code differences support the claim that
the final embedding-cluster code assignment contributes beyond coordinate
factorization alone. The Atomic Lexical control on Tiny Shakespeare also shows
that replacing the same lexical vocabulary with atomic learned IDs does not
recover Flat ILM's BPB at either tested parameter tier.

Full ILM improves likelihood over Flat ILM in every completed setting. This
supports the combined coordinate-role embeddings, coordinate-role output heads,
and word-prefix objective. The present matrix does not isolate the contribution
of each component.

## Limits

- Three seeds provide replication, but are too few for strong claims about
  small differences or broad statistical generalization.
- All models use a 6,000-step horizon. The width comparisons measure
  fixed-budget performance rather than fully converged scaling behavior.
- Flat coordinate models have more sequential positions than Atomic lexical
  models for a similar lexical span. The parameter-matched Atomic control does
  not equalize attention FLOPs.
- The 6.5M optimizer crossover leaves the same ordering under both protocols,
  but it shows that cross-implementation BPB gaps are optimizer sensitive.
- These results are from-scratch small-model comparisons. They do not establish
  state-of-the-art performance or a comparison with large pretrained systems.

## Source Artifacts

Seed-level reports are stored in [evaluation/results/](evaluation/results/).
Commands for every family and the optimizer crossover are recorded in
[METHOD.md](METHOD.md). Checkpoints are local generated artifacts and are not
part of the repository release.
