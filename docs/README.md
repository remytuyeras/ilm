# ILM Documentation

These guides describe the maintained SDK, model variants, and reproducible
experiments. Historical notes and discarded design paths remain in the ignored
`reflect/` directory.

- [Tokenization](tokenization.md): tokenizer methods, released artifacts,
  semantic-label sidecars, diagnostics, and embedding caches.
- [Architecture](architecture.md): flattened coordinate-time Transformer,
  coordinate-role interfaces, and the word-prefix objective.
- [Training](training.md): sandbox checkpoints, architecture flags, and model
  metadata, with practical guidance for interpreting training runs.
- [Decoding](decoding.md): scalar and coordinate-aware sampling during
  generation.
- [Evaluation](evaluation.md): held-out BPB evaluation, generation comparisons,
  and external references.
- [Reproducibility](reproducibility.md): the controlled experiment layout and
  evidence plan, deterministic splits, and result aggregation.

[HOWTO.md](../HOWTO.md) remains the command-oriented guide for the interactive
scripts.
