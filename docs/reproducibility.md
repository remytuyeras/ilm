# Reproducibility

`experiments/evaluation/` is the home for the controlled evaluation study. It
contains fixed configurations, prompt sets, split manifests, experiment-local
tokenizers, and the evidence plan. It does not duplicate the source corpus in
version control or commit checkpoints and embedding caches.

Generated train, validation, and test files may be created locally from the
source corpus. Their byte ranges and checksums belong under `splits/`. Frozen
tokenizers fitted to the frozen corpus belong under the experiment's
`tokenizers/` directory because they are part of that experiment's definition.

Create deterministic, byte-disjoint split files with a context gap at each
boundary:

```bash
python experiments/prepare_text_split.py \
  --source-file data/corpora/training_old_english.txt \
  --output-dir experiments/evaluation/splits/tinyshakespeare \
  --max-context-bytes 60
```

The command writes `train.txt`, `validation.txt`, `test.txt`, and
`manifest.json`. Build and freeze the primary tokenizer from the full source
corpus before splitting model data. A semantic spelling sidecar and PCA plot may
be generated afterwards for analysis, but are not inputs to primary training or
evaluation.

Verify that the frozen tokenizer covers each fixed split before training:

```bash
python experiments/check_tokenizer_coverage.py \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --text experiments/evaluation/splits/tinyshakespeare/train.txt \
  --text experiments/evaluation/splits/tinyshakespeare/validation.txt \
  --text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --oov-policy error
```

The complete training matrix and completed metrics are recorded in
[experiments/METHOD.md](../experiments/METHOD.md) and
[experiments/RESULTS.md](../experiments/RESULTS.md). Run outputs should preserve their seed, configuration, corpus checksum,
tokenizer checksum, and generation settings. Paper-local tables and figures are
derived from the tracked result records rather than edited by hand.

## Historical Environment

The completed paper runs used Python 3.9.6 and PyTorch 2.6.0. These versions
are retained as provenance for the frozen metrics and artifacts. They are not
the supported development environment because they have known security
advisories. The repository's current `requirements.txt` specifies the patched
runtime for new tests and reruns. A rerun under that environment reproduces the
protocol, not a claim of bitwise-identical historical metrics.

The complete package snapshot is recorded in
[`experiments/evaluation/paper_environment_2026-08.txt`](../experiments/evaluation/paper_environment_2026-08.txt).
It is a historical manifest, not an installable requirements file.

See [training.md](training.md) and [evaluation.md](evaluation.md) for the
training, teacher-forced BPB, generation, and aggregation commands.
