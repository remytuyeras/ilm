# Training

`sandbox/sandbox.py` is the interactive training and sampling entry point. It
creates a checkpoint and metadata JSON on `create`, then appends curriculum
events to the original metadata file on `improve`.

## Default Assets

The sandbox uses:

```text
data/tokenizers/tokenizer_embedding_cluster_v1.json
data/corpora/training_old_english.txt
```

Override either path with `--tokenizer-json` or `--training-text`. The
checkpoint metadata records the paths used by that run. Older release metadata
may name the pre-reorganization `data/` paths and should be treated as a
historical record.

## Configuration Layers

Training parameters such as dropout, learning rate, and epoch count can change
without changing checkpoint tensor shapes. Architecture parameters such as
embedding dimension, layer count, head count, ILM input embeddings, and ILM
output heads alter model compatibility. The ILM objective changes the training
criterion and requires the same window geometry during any continued training.

The active model uses `head_num` heads and derives
`head_size = embedding_dim // head_num`. Both values are recorded in model
metadata. See [HOWTO.md](../HOWTO.md) for the create, improve, load, streaming,
and plot workflows.

## Reproducible Splits

The default `--training-text` path retains the original implicit 80/20
coordinate split for quick local experiments. A publication run should instead
pass all three files with `--train-text`, `--validation-text`, and `--test-text`.
Each file is encoded independently, so a sampled context window cannot cross a
split boundary. Use `--seed` to reset Python, NumPy, and PyTorch before model
construction, dropout, and batch sampling. The seed, split mode, OOV policy,
encoding statistics, and periodic validation history are saved in metadata.

Prepare deterministic corpus files first:

```bash
python experiments/prepare_text_split.py \
  --source-file data/corpora/training_old_english.txt \
  --output-dir experiments/evaluation/splits/tinyshakespeare \
  --max-context-bytes 60
```

Then train with frozen tokenizer and explicit split inputs:

```bash
python sandbox/sandbox.py create models/evaluation/c_full_seed13.pth \
  --seed 13 \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --train-text experiments/evaluation/splits/tinyshakespeare/train.txt \
  --validation-text experiments/evaluation/splits/tinyshakespeare/validation.txt \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --oov-policy error \
  --validation-interval 500 \
  --ilm-input-embeddings --ilm-output-heads --ilm-objective
```

## Unknown Tokens

For the primary corpus-level experiment, build and freeze the tokenizer from
the full fixed corpus before creating model-data splits. `--oov-policy error`
then verifies that all split text is covered. An unseen token stops encoding and
reports its count, rate, and representative examples.

Use `--oov-policy fallback` only when a new document is introduced to probe
transfer coverage. It replaces every unseen token with one fixed valid code,
the tokenizer's numerically smallest code unless `--oov-fallback-code` is set.
For example, if the fallback is `0:0:0`, both an unseen `" astrophysics"` and
an unseen `" telescope"` are encoded as `0:0:0`. The document can then be
processed, but the model cannot distinguish those words. Metadata and
teacher-forced evaluation reports record `oov_token_count`, `oov_rate`, and the
selected fallback code.

Check coverage without starting training:

```bash
python experiments/check_tokenizer_coverage.py \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --text experiments/evaluation/splits/tinyshakespeare/train.txt \
  --text experiments/evaluation/splits/tinyshakespeare/validation.txt \
  --text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --oov-policy error \
  --output-file experiments/evaluation/results/tokenizer_coverage.json
```

## Independent ILM Options

`--ilm-input-embeddings` gives the same coordinate value a separate input
embedding for each coordinate role. `--ilm-output-heads` selects a separate
language-model head for each predicted coordinate role. `--ilm-objective`
excludes an incomplete word suffix at the left boundary of a sampled training
window from the cross-entropy average. The options can be enabled separately or
together. The objective requires `word_block_size == block_size // syllable_num`.

## Released Checkpoints

`models/release/` contains selected sanitized checkpoints and metadata. Local
checkpoints and experimental runs remain ignored because they can be large and
are tied to a specific training run.
