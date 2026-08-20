# Controlled Evaluation

This directory contains the controlled evaluation records used by the ILM
study. It includes completed three-seed Tiny Shakespeare and enwik8 runs as
well as the commands and configurations needed to reproduce their setup.

- `configs/`: locked nanoGPT reference configurations.
- `patches/`: versioned changes required by the controlled nanoGPT reference.
- `prompts/`: reusable qualitative prompt definitions.
- `splits/`: split-generation metadata, byte ranges, and checksums. Generated
  corpus text is local.
- `tokenizers/`: frozen experiment mappings built from the complete frozen
  corpus under the closed-corpus representation protocol.
- `runs/`: ignored local checkpoints and logs.
- `results/`: seed-level BPB reports and selected aggregate records.

The primary tokenizer is built and frozen from the complete corpus. Then
`experiments/prepare_text_split.py` creates line-aligned model-data files and
their manifest. The sandbox accepts the files through `--train-text`,
`--validation-text`, and `--test-text`. Use
`experiments/check_tokenizer_coverage.py --oov-policy error` to confirm that
the frozen tokenizer covers every split before training.

`experiments/evaluate_ilm.py` writes held-out BPB reports. The from-scratch
nanoGPT reference uses `experiments/prepare_nanogpt_char.py` to export the same
fixed split. Do not use nanoGPT's default Shakespeare download or preparation
script for controlled comparisons.

The reference is pinned to nanoGPT commit
`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`. Apply the repository patch before
running a nanoGPT command. It adds a configurable seed and the all-parameter
AdamW profile used by the optimizer crossover:

```bash
git -C baselines/nanoGPT apply --check \
  ../../experiments/evaluation/patches/nanogpt_controlled_baseline.patch
git -C baselines/nanoGPT apply \
  ../../experiments/evaluation/patches/nanogpt_controlled_baseline.patch
```

Read [../METHOD.md](../METHOD.md) for the executable protocol and
[../RESULTS.md](../RESULTS.md) for completed measurements. The method includes
the optimizer crossover required to interpret cross-implementation reference
comparisons.

The historical package snapshot for the completed runs is
[`paper_environment_2026-08.txt`](paper_environment_2026-08.txt). It documents
provenance only. Use the root `requirements.txt` for a supported environment.
