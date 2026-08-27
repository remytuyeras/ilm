# Results

This document records the controlled experiments in [METHOD.md](METHOD.md).
It reports teacher-forced held-out likelihood only. Generation samples remain
useful qualitative evidence, but they are not substituted for BPB.

## Reading The Tables

Lower bits per UTF-8 byte (BPB) is better. Unless noted otherwise, values are
the mean over independently trained seeds `13`, `29`, and `47`, and `+/-`
denotes their sample standard deviation. The replicated Permuted Flat control
has two levels of variation: three assignment seeds, each evaluated with the
same three model-training seeds. All reported models were trained from scratch
for 6,000 updates on the frozen corpus splits.

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
coordinates. The replicated Permuted Flat control uses the same final code set
and lexical segmentation as Flat ILM while varying only the lexical-entry-to-code
assignment. Its assignment-level results appear after the primary model-family
tables.

| Tiny Shakespeare family | Parameters | Mean BPB +/- SD | Seed BPB values |
| --- | ---: | ---: | --- |
| Character GPT | 6,525,600 | 2.650571 +/- 0.037308 | 2.665362, 2.608136, 2.678216 |
| Atomic Lexical | 6,469,374 | 2.320493 +/- 0.024002 | 2.305153, 2.308174, 2.348153 |
| Flat ILM | 6,555,064 | 2.150793 +/- 0.007531 | 2.159404, 2.147534, 2.145441 |
| Full ILM | 6,631,992 | **2.120122 +/- 0.008608** | 2.123232, 2.126744, 2.110392 |

| enwik8 family | Parameters | Mean BPB +/- SD | Seed BPB values |
| --- | ---: | ---: | --- |
| Byte GPT | 6,567,600 | 2.479660 +/- 0.016049 | 2.496671, 2.477523, 2.464786 |
| Flat ILM | 6,561,064 | 2.333131 +/- 0.004498 | 2.336511, 2.334855, 2.328025 |
| Full ILM | 6,676,456 | **2.235895 +/- 0.006256** | 2.235785, 2.242206, 2.229695 |

Full ILM improves over Flat ILM by `0.030671` BPB on Tiny Shakespeare and by
`0.097236` BPB on enwik8.

### Replicated Permuted Flat Control

Each row below averages model-training seeds `13`, `29`, and `47` under one
fixed code assignment. `Delta` is the assignment-level mean BPB minus the
corresponding Flat ILM mean. The aggregate row is the mean across the three
independently sampled assignment-level means. It must not be interpreted as a
mean and standard deviation over nine exchangeable model seeds.

| Assignment seed | Tiny Shakespeare BPB / s.d. | Tiny Delta | enwik8 BPB / s.d. | enwik8 Delta |
| ---: | ---: | ---: | ---: | ---: |
| `314159` | 2.296561 +/- 0.011518 | +0.145768 | 2.504098 +/- 0.009986 | +0.170967 |
| `271828` | 2.293996 +/- 0.016085 | +0.143203 | 2.521013 +/- 0.001390 | +0.187883 |
| `161803` | 2.294466 +/- 0.004873 | +0.143673 | 2.519387 +/- 0.013009 | +0.186257 |
| **Mean across assignments** | **2.295008 +/- 0.001366** | **+0.144215** | **2.514833 +/- 0.009332** | **+0.181702** |

The first three rows report the sample standard deviation across model-training
seeds. The aggregate row reports the sample standard deviation across the three
assignment-level means.

All nine matched model-seed comparisons are worse under the unrestricted
permuted map on each corpus. The type-level code set is preserved, while corpus
frequency is redistributed across occupied coordinate paths. The
exact-frequency control below addresses this frequency-marginal dimension
directly.

### Exact-Frequency Permuted Flat Control

Exact-frequency reassignment permutes lexical entries only within strata whose
integer training frequencies agree. It therefore preserves every
training-frequency-weighted coordinate marginal exactly. Each row averages the
three model-training seeds under one fixed reassignment. The aggregate row is
the mean across assignment-level means.

| Assignment seed | Tiny Shakespeare BPB / s.d. | Tiny Delta | enwik8 BPB / s.d. | enwik8 Delta |
| ---: | ---: | ---: | ---: | ---: |
| `314159` | 2.253048 +/- 0.002506 | +0.102255 | 2.427861 +/- 0.005512 | +0.094730 |
| `271828` | 2.252124 +/- 0.007565 | +0.101330 | 2.428106 +/- 0.007727 | +0.094975 |
| `161803` | 2.266179 +/- 0.003992 | +0.115386 | 2.430606 +/- 0.000575 | +0.097475 |
| **Mean across assignments** | **2.257117 +/- 0.007861** | **+0.106324** | **2.428857 +/- 0.001519** | **+0.095727** |

All nine matched model-seed differences are positive for Exact-frequency
Permuted Flat on both corpora. The table below records the associated type,
mass, and marginal diagnostics, averaged across assignment seeds. `Moved types`
is the fraction of lexical types whose assigned path changes. `rho_train` and
`rho_test` are the fractions of training and test lexical-entry occurrences
whose assigned path moves. `TV` is the maximum total-variation distance across
coordinate roles for the corresponding frequency-weighted marginals.

| Corpus | Control | Moved types | rho_train | rho_test | TV_train | TV_test |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Tiny Shakespeare | Exact-frequency | 0.981 | 0.313 | 0.314 | 0.000 | 0.049 |
| Tiny Shakespeare | Unrestricted | 1.000 | 1.000 | 1.000 | 0.626 | 0.638 |
| enwik8 | Exact-frequency | 0.996 | 0.235 | 0.228 | 0.000 | 0.021 |
| enwik8 | Unrestricted | 1.000 | 1.000 | 1.000 | 0.729 | 0.741 |

Exact-frequency reassignment moves less training-token mass than unrestricted
reassignment while moving approximately 98% of Tiny Shakespeare types and more
than 99% of enwik8 types. It leaves the training coordinate marginals unchanged
and still raises BPB by `0.106324` on Tiny Shakespeare and `0.095727` on enwik8
relative to the original Flat ILM assignment. The remaining token mass includes
high-frequency entries in singleton frequency strata, which cannot exchange
paths under the exact constraint. This shows that changes in
training-frequency-weighted coordinate marginals are not necessary for the
permutation penalty, while keeping the claim separate from an isolated
attribution to embedding geometry.

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

The replicated semantic-versus-permuted-code differences support the claim that
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
