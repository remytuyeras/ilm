# Controlled Evaluation

This directory defines the planned controlled evaluation of ILM. It is a
scaffold, not a completed result set.

- `configs/`: locked model and optimizer configurations.
- `prompts/`: held-out prompt definitions once the split is frozen.
- `splits/`: split generation metadata, byte ranges, and checksums.
- `tokenizers/`: frozen tokenizer mappings fitted only on a training split.
- `runs/`: ignored local checkpoints and logs.
- `results/`: generated metrics and selected aggregate tables.

Follow [metric_evidence_plan.md](metric_evidence_plan.md) before treating a
comparison as paper evidence.

Build and freeze the primary tokenizer from the full source corpus, then use
`experiments/prepare_text_split.py` to create line-aligned model-data files and
their manifest. The sandbox accepts the files through `--train-text`,
`--validation-text`, and `--test-text`.
`experiments/evaluate_ilm.py` writes held-out BPB reports, while
`experiments/aggregate_results.py` creates seed-level summaries and figures.
Use `experiments/check_tokenizer_coverage.py` with `--oov-policy error` to
confirm fixed-split coverage before the first model run.

`experiments/prepare_nanogpt_char.py` exports these exact fixed splits for the
from-scratch nanoGPT character baseline. Do not use nanoGPT's default
Shakespeare download or preparation script for controlled comparisons.
