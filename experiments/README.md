# Experiments

This directory records the controlled experiments behind the ILM study. It is
separate from exploratory sandbox checkpoints. Corpus sources remain in
`data/corpora/`, reusable general-purpose tokenizer assets remain in
`data/tokenizers/`, and local checkpoints remain ignored unless deliberately
released.

## Controlled Evaluation

[evaluation/](evaluation/README.md) contains the fixed corpus splits, frozen
experiment tokenizers, model configurations, seed-level held-out reports, and
result aggregation inputs for the completed Tiny Shakespeare and enwik8 study.

- [METHOD.md](METHOD.md) is the executable protocol, including tokenizer
  construction, split preparation, model commands, baseline commands, and
  optimizer crossover.
- [RESULTS.md](RESULTS.md) reports the completed three-seed BPB measurements
  and their scope.

## Directory Policy

| Path | Contents | Version-control policy |
| --- | --- | --- |
| `evaluation/configs/` | Locked nanoGPT reference configurations | Track |
| `evaluation/prompts/` | Reusable qualitative prompt sets | Track |
| `evaluation/splits/` | Split manifests and documentation | Track source metadata, not generated corpora |
| `evaluation/tokenizers/` | Frozen experiment mappings | Track mappings and manifests |
| `evaluation/results/` | Seed-level BPB reports and summaries | Track selected reports |
| `evaluation/runs/` | Local checkpoints and logs | Ignore |

Start with [docs/reproducibility.md](../docs/reproducibility.md) for a compact
overview, then use [METHOD.md](METHOD.md) for the complete command sequence.
