# Paper Artifacts

`build_artifacts.py` reads the completed seed-level JSON reports in
`experiments/evaluation/results/` and regenerates every file in this directory
and in `../figures/`.

- `seed_bpb.csv` contains one held-out BPB result per model seed.
- `bpb_summary.csv` contains the paper means and sample standard deviations.
- `crossover_summary.csv` and `optimizer_crossover.tex` record the 6.5M optimizer crossover.
- `results_6m.tex` and `results_15m.tex` are the corpus-paired result tables included by `main.tex`.
- `pairwise_deltas.csv` records the planned contrasts used in the text.

The current study has three independently trained seeds: 13, 29, and 47.
Parameter counts are the exact total trainable counts reported in
`experiments/RESULTS.md`.

## Current Rows

| Corpus | Tier | Family | Mean BPB | Sample SD |
| --- | --- | --- | ---: | ---: |
| Tiny Shakespeare | 6.5M | Character GPT | 2.650571 | 0.037308 |
| Tiny Shakespeare | 6.5M | Atomic Lexical | 2.320493 | 0.024002 |
| Tiny Shakespeare | 6.5M | Permuted Flat | 2.296561 | 0.011518 |
| Tiny Shakespeare | 6.5M | Flat ILM | 2.150793 | 0.007531 |
| Tiny Shakespeare | 6.5M | Full ILM | 2.120122 | 0.008608 |
| Tiny Shakespeare | 15.5M | Atomic Lexical | 2.384029 | 0.006054 |
| Tiny Shakespeare | 15.5M | Flat ILM | 2.140613 | 0.009786 |
| Tiny Shakespeare | 15.5M | Full ILM | 2.135076 | 0.002098 |
| enwik8 | 6.5M | Byte GPT | 2.479660 | 0.016049 |
| enwik8 | 6.5M | Permuted Flat | 2.504098 | 0.009986 |
| enwik8 | 6.5M | Flat ILM | 2.333131 | 0.004498 |
| enwik8 | 6.5M | Full ILM | 2.235895 | 0.006256 |
| enwik8 | 15.5M | Flat ILM | 2.292757 | 0.009464 |
| enwik8 | 15.5M | Full ILM | 2.181544 | 0.001983 |
