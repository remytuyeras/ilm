# HOWTO

This is the practical runbook for the project scripts. The file is organized by source file so each script's commands, arguments, and behavior stay together.

## Setup

<details>
<summary>Environment and dependency setup</summary>
<br>


Install dependencies:

```bash
pip install -r requirements.txt
```

For the embedding-cluster tokenizer, create a `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
```

The tokenizer code calls `load_dotenv()`, so `.env` is read automatically when the OpenAI embedding client is created.

</details>

## `tests/quickstart.py`

<details>
<summary>Build, load, and inspect tokenizers</summary>
<br>


`tests/quickstart.py` is the tokenizer playground. It either loads an existing tokenizer JSON or builds a new one, then tokenizes one sample line from the source text and detokenizes it back.

### Purpose

Use this file when you want to:

- load a tokenizer JSON and verify it works
- build the original `relative-position` tokenizer
- build the `embedding-cluster` tokenizer
- test tokenizer options without training a model
- inspect one tokenized line before using the tokenizer in `sandbox/sandbox.py`

### Default Behavior

```bash
python tests/quickstart.py
```

This runs in load mode. It loads:

```text
data/tokenizer_v2.json
```

and reads line `20` from:

```text
data/training_old_english.txt
```

It prints:

```text
token codes
detokenized tokens
```

### General Arguments

```text
--mode load|build
--source-file PATH
--target-file PATH
--method relative-position|embedding-cluster
--line-index N
```

`--mode load` reads an existing tokenizer JSON from `--target-file`.

`--mode build` creates a tokenizer from `--source-file` and writes it to `--target-file`.

`--source-file` is the text corpus used for building the tokenizer and selecting the sample line.

`--target-file` is the tokenizer JSON to read or write.

`--line-index` selects which source-file line is used for the smoke test after loading or building.

### Load A Tokenizer

Load the current relative-position tokenizer:

```bash
python tests/quickstart.py \
  --mode load \
  --source-file data/training_old_english.txt \
  --target-file data/tokenizer_v2.json
```

Load an embedding-cluster tokenizer:

```bash
python tests/quickstart.py \
  --mode load \
  --source-file data/training_old_english.txt \
  --target-file data/tokenizer_embedding_cluster_v1.json
```

Loading does not care which method created the JSON. It only needs the saved `direct` and `reverse` mappings.

### Build With `relative-position`

The `relative-position` method is the original tokenizer method. It computes where tokens tend to appear inside nested text segments, then assigns hierarchical codes.

Build it with:

```bash
python tests/quickstart.py \
  --mode build \
  --method relative-position \
  --source-file data/training_old_english.txt \
  --target-file data/tokenizer_v2.json
```

SDK equivalent:

```python
from ilm.tokenizer import create_tokenizer

tokenizer, detokenizer = create_tokenizer(
    source_file="data/training_old_english.txt",
    target_file="data/tokenizer_v2.json",
    method="relative-position",
)
```

### Build With `embedding-cluster`

The `embedding-cluster` method does:

```text
unique text tokens
-> OpenAI embeddings
-> optional normalization
-> PCA
-> first centroid assignment
-> residual clustering for later code coordinates
-> cosine renaming of residual centroids against the first centroid basis
-> tokenizer JSON
```

The residual-centroid idea is:

```text
E_0(w) = PCA(embedding(w))
token_0 = nearest first-level centroid C
E_1(w) = E_0(w) - C
token_1 = nearest residual centroid, renamed by cosine similarity to the first centroid basis
E_2(w) = E_1(w) - C'
token_2 = nearest residual centroid, again renamed against the first centroid basis
```

This makes later token coordinates behave like residual correction terms in the same global semantic basis, instead of recursively creating unrelated local cluster labels.

Build it with:

```bash
python tests/quickstart.py \
  --mode build \
  --method embedding-cluster \
  --source-file data/training_old_english.txt \
  --target-file data/tokenizer_embedding_cluster_v1.json \
  --cluster-method spherical-kmeans \
  --reduced-dim 10 \
  --embedding-batch-size 512 \
  --collision-report-limit 200 \
  --semantic-spelling-file \
  --plot-pca-3d \
  --plot-clusters \
  --centroid-label-method llm \
  --centroid-label-model gpt-5.6-terra
```

Suggested default parameters for the current residual-centroid method:

```text
--cluster-method spherical-kmeans
--reduced-dim 10
--depth 3
--embedding-batch-size 512
--collision-report-limit 200
```

Earlier experiments used `balanced-kmeans` for the older recursive tokenizer because that method needed help keeping child clusters from overflowing the remaining code capacity. That is no longer the recommendation. In the residual-centroid tokenizer, the mathematical principle is nearest-centroid assignment.

`spherical-kmeans` is now the recommended clustering mode because it assigns clusters by cosine direction. That better matches embedding geometry: words with similar meaning usually point in similar directions, even when their vector magnitudes differ. The tokenizer still subtracts the residual-space mean of each assigned cluster, so the residual code remains an approximation process:

```text
E_{n+1}(w) = E_n(w) - C_n
```

SDK equivalent:

```python
from ilm.tokenizer import create_tokenizer

tokenizer, detokenizer = create_tokenizer(
    source_file="data/training_old_english.txt",
    target_file="data/tokenizer_embedding_cluster_v1.json",
    method="embedding-cluster",
    cluster_method="spherical-kmeans",
    reduced_dim=10,
    embedding_batch_size=512,
    semantic_spelling_file="data/tokenizer_embedding_cluster_v1.semantic.json",
)
```

### Semantic Spelling Export

Embedding-cluster builds can also write a human-readable JSON file that expresses each source token as a chain of centroid labels:

```bash
--semantic-spelling-file
```

With no path, this derives the file name from `--target-file`. For:

```text
data/tokenizer_embedding_cluster_v1.json
```

the semantic spelling file is:

```text
data/tokenizer_embedding_cluster_v1.semantic.json
```

The file is a plain mapping, not a tokenizer object with `direct` and `reverse`:

```json
{
    " apple": " fruit:sweet: red",
    " king": " ruler:man:crown"
}
```

By default, each part is the closest source token to the centroid assigned at that residual level. This is cheap, deterministic, and requires no extra model calls. It is meant for interpretation: it lets you inspect whether the learned code behaves like a compositional semantic alphabet.

You can also choose the output path explicitly:

```bash
--semantic-spelling-file data/old_english_semantic_spelling.json
```

Use LLM centroid labels when closest-token labels are too literal or awkward:

```bash
--semantic-spelling-file \
--centroid-label-method llm
```

This uses `AsyncOpenAI` to label one residual level at a time. With the default `--depth 3`, that is:

```text
3 level calls
```

Each call receives all `64` clusters for that level together, with up to `20` randomly sampled words per cluster by default. The model returns a coordinated JSON object with one semantic atom per cluster. Seeing the full level at once helps avoid duplicate labels and generic labels such as `old-english-vocabulary`. These LLM labels are used by both the semantic spelling JSON and the PCA centroid plot labels.

Useful LLM label options:

```bash
--centroid-label-model gpt-5.6-terra
--centroid-label-concurrency 8
--centroid-label-examples 20
--centroid-label-max-output-tokens 4096
```

`--centroid-label-model` chooses the OpenAI model used only for centroid naming. `gpt-5.6-terra` is the default because centroid naming is a taxonomy task: the model sees all `64` clusters for a level at once and needs to produce coordinated, distinct labels.

LLM labels are validated before they are used. If a label is missing, duplicated, generic such as `old-english-vocabulary`, or longer than two words, the builder keeps the good labels and sends one targeted repair request for only the bad cluster ids.

The label requests do not pass `temperature`, because some models reject that sampling parameter. For `gpt-5.6-*` models, the request uses low reasoning effort so the model has some room to compare clusters without turning this into an expensive reasoning task.

`--centroid-label-concurrency` controls how many residual-level labeling calls run at the same time.

`--centroid-label-examples` controls how many random words are sampled from each cluster for the level labeling prompt.

`--centroid-label-max-output-tokens` controls the output budget for the level's JSON label object. If a model returns invalid or empty text, the builder asks for a targeted repair response before using any closest-token fallback.

If every LLM label still fails after repair, the build raises an error instead of silently writing closest-token labels. If only a few individual labels fail, those labels fall back to closest-token labels and the terminal prints the first errors. The terminal also reports how many labels were repaired without fallback.

When `--plot-pca-3d` is also enabled, the semantic spelling file is saved before plot windows open.

### Embedding Options

```text
--embedding-model text-embedding-3-small
--embedding-batch-size 512
--cache-file PATH
--refresh-cache
--keep-token-spacing
```

`--embedding-model` selects the OpenAI embedding model.

`--embedding-batch-size` controls how many unique tokens are sent per embedding API request. A good starting value is `512`. You can try `1000` or `2048`, but do not use `10000` for the synchronous embeddings call.

`--cache-file` lets you choose where the embedding matrix cache is stored.

By default, embeddings are cached next to the target JSON. For:

```text
data/tokenizer_embedding_cluster_v1.json
```

the default cache is:

```text
data/tokenizer_embedding_cluster_v1.embeddings.npz
```

The `.json` file is the actual tokenizer. The `.npz` file is only a numeric embedding cache so you can retry PCA/clustering choices without paying for embeddings again.

`--refresh-cache` ignores the existing `.npz` cache and calls the embedding API again.

`--keep-token-spacing` preserves leading whitespace when embedding tokens. By default, tokens are stripped before embedding so `" king"` and `"king"` embed like the same word.

### PCA Options

```text
--reduced-dim 10
--no-normalize
```

`--reduced-dim` controls the PCA dimension. It defaults to `10`, and it does not need to match the base-64 tokenizer coordinate size.

For the residual-centroid method, `10` is the recommended starting point. In current experiments it gives the most legible ideogram-like decompositions: compressed enough to force semantic basis atoms, but detailed enough to separate axes like verbs, plurals, negation, necessity, growth, possession, and social roles.

The earlier `64` setting kept much more embedding information, but it could make the tokenizer behave too numerically: clusters can become excellent at separating points while being less legible as semantic building blocks.

Useful experiments:

```bash
--reduced-dim 3
--reduced-dim 8
--reduced-dim 10
--reduced-dim 16
--reduced-dim 32
--reduced-dim 64
```

Use `10` for the main semantic tokenizer. Use `8` as a slightly more compressed baseline, use `3` when you want an aggressively compressed symbolic basis, and use `16`, `32`, or `64` when you want to test whether extra detail improves collision behavior or generation quality. Higher dimensions are no longer the first recommendation for ILM's semantic-symbol goal.

Normalization is on by default. `--no-normalize` skips the standardization step before PCA.

### Clustering Options

```text
--cluster-method spherical-kmeans
--depth 3
--max-tokens 5000
--random-state 42
```

`spherical-kmeans` assigns each token to the centroid direction with the highest cosine similarity. This is the recommended mode for embedding-cluster tokenizers because it treats the cluster basis as semantic directions.

`cosine-kmeans` is accepted as an alias for `spherical-kmeans`.

`kmeans` uses ordinary Euclidean K-Means at each residual level. It is still available for comparison, but it is no longer the main recommendation for ILM's semantic basis.

Recommended:

```text
--cluster-method spherical-kmeans
```

`--depth 3` means codes look like:

```text
a:b:c
```

The total code capacity is:

```text
64^3 = 262,144
```

### Collision Repair Report

Embedding-cluster builds print a live collision repair table by default. A collision happens when two tokens receive the same preferred residual code, even though the total code space has enough capacity.

The table is printed as collisions are found and repaired:

```text
Residual code collision repairs:
+--------+-----------+------------------------------+-------------+-------------+-------------+
| repair | token_idx | token                        | preferred   | repaired    | kept_prefix |
+--------+-----------+------------------------------+-------------+-------------+-------------+
| 1      | 8412      | 'king'                       | 12:4:9      | 12:4:10     | 2/3         |
+--------+-----------+------------------------------+-------------+-------------+-------------+
Residual collision repair summary: 1 repaired.
```

Columns:

- `preferred`: the code produced directly by the residual-centroid method
- `repaired`: the nearest unused fallback code
- `kept_prefix`: how much of the original code prefix was preserved

Use this to avoid a huge terminal report:

```bash
--collision-report-limit 100
```

Use this to silence the report:

```bash
--no-collision-report
```

The final JSON also stores the total count as `residual_collision_repairs`.

### PCA/Centroid Plot

Embedding-cluster builds can display 3D matplotlib plots of the first three PCA dimensions for every residual level:

```bash
--plot-pca-3d
```

The plot shows:

- sampled token/residual vectors projected onto `PC1`, `PC2`, and `PC3`
- that residual level's KMeans centroids as black `x` markers
- each centroid labeled with the closest token to that centroid

For the default `--depth 3`, this opens three plots:

```text
Residual Level 0: E_0(w)
Residual Level 1: E_1(w)
Residual Level 2: E_2(w)
```

Color points by cluster at each residual level:

```bash
--plot-pca-3d --plot-clusters
```

By default, each plot samples up to `20000` token points so the figures remain responsive. Centroid labels are still computed from the full vector set for that residual level, not only the plotted sample.

Useful options:

```bash
--plot-sample-size 5000
--plot-sample-size 0
--no-plot-centroid-labels
```

Use `--plot-sample-size 0` to plot every token point.

`--max-tokens` is useful for cheap experiments before embedding the full vocabulary:

```bash
python tests/quickstart.py \
  --mode build \
  --method embedding-cluster \
  --source-file data/training_old_english.txt \
  --target-file data/tokenizer_embedding_cluster_test.json \
  --max-tokens 5000
```

### Recommended Quickstart Workflows

Build a normal tokenizer:

```bash
python tests/quickstart.py --mode build --method relative-position --source-file data/training_old_english.txt --target-file data/tokenizer_v2.json
```

Try embedding-cluster on a small vocabulary:

```bash
python tests/quickstart.py \
  --mode build \
  --method embedding-cluster \
  --source-file data/training_old_english.txt \
  --target-file data/tokenizer_embedding_cluster_test.json \
  --cluster-method spherical-kmeans \
  --reduced-dim 10 \
  --embedding-batch-size 512 \
  --max-tokens 5000
```

Build the full embedding-cluster tokenizer:

```bash
python tests/quickstart.py \
  --mode build \
  --method embedding-cluster \
  --source-file data/training_old_english.txt \
  --target-file data/tokenizer_embedding_cluster_v1.json \
  --cluster-method spherical-kmeans \
  --reduced-dim 10 \
  --embedding-batch-size 512
```

</details>

## `sandbox/sandbox.py`

<details>
<summary>Train, improve, load, and sample models</summary>
<br>


`sandbox/sandbox.py` is the model experiment runner. It creates, improves, loads, and then interactively samples from an `IntuinisticLanguageModel`.

For practical guidance on learning rate, dropout, checkpoint branching, loss interpretation, and repetition loops, see [docs/training_strategy.md](docs/training_strategy.md). For ideas on making deterministic `temperature=0` generation less repetitive, see [docs/greedy_decoding_strategy.md](docs/greedy_decoding_strategy.md).

### Purpose

Use this file when you want to:

- train a fresh model
- continue training a checkpoint
- load a checkpoint and generate text
- compare checkpoint quality by trying prompts
- inspect model weights through the UI plot mode

### CLI Shape

The script uses argparse subcommands:

```bash
python sandbox/sandbox.py create models/m2.v0.0.0.pth
python sandbox/sandbox.py improve models/m2.v0.0.0.pth --patch
python sandbox/sandbox.py load models/m2.v0.0.0.pth
```

Use `create`, `improve`, or `load` without `--` because they are the action the script performs. Options such as `--dropout`, `--lr`, and `--patch` modify that action.

### Shared Arguments

Shared arguments can go after the model path, which is often the easiest form to read:

```text
--tokenizer-json PATH
--training-text PATH
--vocab-size N
--block-size N
--batch-size N
--embedding-dim N
--head-size N
--layer-num N
--dropout FLOAT
--epoch-num N
--lr FLOAT
```

Example:

```bash
python sandbox/sandbox.py \
  improve models/m2.v0.2.1.pth \
  --patch \
  --dropout 0.4 \
  --epoch-num 2000 \
  --lr 1e-3
```

They can also go before the subcommand if you prefer argparse's global-option style.

### Defaults

Dataset/tokenizer defaults:

```text
--tokenizer-json data/tokenizer_embedding_cluster_v1.json
--training-text data/training_old_english.txt
```

Relative-position tokenizer path that may still be useful for comparison:

```text
data/tokenizer_v2.json
```

Older small-data defaults that may still be useful for comparison:

```text
data/tokenizer_v1.json
data/training_input.txt
```

Architecture defaults:

```text
--vocab-size 64
--block-size 60
--batch-size 32
--embedding-dim 80
--head-num 4
--layer-num 6
--coordinate-token-embeddings off
```

Training defaults:

```text
--dropout 0.5
--epoch-num 4000
--lr 1e-3
```

It is usually more typical to tune `dropout`, `epoch_num`, and `lr` first. Those change training behavior without changing checkpoint tensor shapes.

Change architecture parameters more carefully. If you change `vocab_size`,
`block_size`, `embedding_dim`, `head_num`, `layer_num`, or architecture flags
such as `--coordinate-token-embeddings`, existing checkpoint files may no longer
load because the saved tensor shapes must match the model architecture.

Each attention head has `embedding_dim // head_num` channels. `head_size` is
derived and recorded in model metadata rather than configured independently.

### Model Metadata JSON

`create` starts a model curriculum record by writing a JSON file next to the first checkpoint:

```text
models/m3.v0.0.0.pth
models/m3.v0.0.0.json
```

That JSON records the tokenizer path, training text path, model architecture, checkpoint list, final train/validation losses, and training curriculum. Later `improve` commands do not create a new JSON for each new checkpoint. They append a new entry to the existing curriculum JSON and update `latest_model_path`.

For example:

```bash
python sandbox/sandbox.py create models/m3.v0.0.0.pth --dropout 0.4
python sandbox/sandbox.py improve models/m3.v0.0.0.pth --patch --epoch-num 2000
python sandbox/sandbox.py improve models/m3.v0.0.1.pth --minor --lr 5e-4
```

All three runs are recorded in `models/m3.v0.0.0.json`.

If you try to improve a checkpoint that has no matching curriculum JSON, the script stops before training. This keeps `improve` from silently creating a second experiment record.

To train with the relative-position tokenizer:

```bash
python sandbox/sandbox.py \
  create models/m2.v0.0.0.pth \
  --tokenizer-json data/tokenizer_v2.json
```

### `create`

Use `create` with a `.pth` model path:

```bash
python sandbox/sandbox.py create models/m2.v0.0.0.pth
```

Behavior:

1. Loads `--tokenizer-json`.
2. Reads `--training-text`.
3. Creates a fresh `IntuinisticLanguageModel`.
4. Trains it for `--epoch-num` steps.
5. Saves weights to the path you gave.
6. Creates the matching model metadata JSON.
7. Opens the interactive user interface.

If no `.pth` path is provided, argparse prints usage and reports that `model_path` is required.

### `improve`

Use `improve` with an existing checkpoint:

```bash
python sandbox/sandbox.py improve models/m2.v0.0.0.pth --patch
```

Behavior:

1. Loads the model checkpoint.
2. Reads the training text.
3. Trains more using the current `--dropout`, `--epoch-num`, and `--lr`.
4. Saves a new versioned checkpoint.
5. Appends this run to the existing model metadata JSON.
6. Opens the interactive user interface.

Version flags:

```text
--patch  m2.v0.0.0.pth -> m2.v0.0.1.pth
--minor  m2.v0.0.0.pth -> m2.v0.1.0.pth
--major  m2.v0.0.0.pth -> m2.v1.0.0.pth
```

If you omit the version flag, the script defaults to patch:

```text
No semantic versioning given. Creating patch.
```

In this message, "semantic versioning" means checkpoint versioning, not the embedding-cluster tokenizer.

Example:

```bash
python sandbox/sandbox.py improve models/m2.v0.2.1.pth --minor
```

### `load`

Use `load` for generation only:

```bash
python sandbox/sandbox.py load models/m2.v0.2.1.pth
```

Behavior:

1. Loads `--tokenizer-json`.
2. Creates the model architecture using the CLI/default model dimensions.
3. Loads the checkpoint weights.
4. Opens the interactive user interface.

The architecture settings must match the checkpoint. If you trained a model with different `embedding_dim`, `block_size`, or `layer_num`, pass those values correctly before the `load` subcommand.

### Recommended Sandbox Workflows

Build a tokenizer, then train a model:

```bash
python tests/quickstart.py --mode build --method relative-position --source-file data/training_old_english.txt --target-file data/tokenizer_v2.json
python sandbox/sandbox.py create models/m2.v0.0.0.pth
```

Continue training a promising model:

```bash
python sandbox/sandbox.py improve models/m2.v0.2.1.pth --patch
```

Compare generation only:

```bash
python sandbox/sandbox.py load models/m2.v0.1.1.pth
python sandbox/sandbox.py load models/m2.v0.2.1.pth
```

### Useful Commands

The commands below intentionally focus on experiment mode, training knobs, and sampling knobs. Architecture dimensions such as `embedding_dim`, `layer_num`, `block_size`, and `batch_size` are usually easier to keep in `sandbox/sandbox.py` so the checkpoint family has one clear source of truth.

Create a coordinate-head model with ordinary coordinate CE. This is the clean baseline for testing whether coordinate-specific heads help:

```bash
python sandbox/sandbox.py create models/m5.v0.0.0.pth \
  --coordinate-lm-heads \
  --dropout 0.4 \
  --epoch-num 4000 \
  --lr 2e-4
```

`--coordinate-lm-heads` is the exception here because it changes which architecture family is being trained. Use it consistently when creating, improving, or loading that checkpoint family.

Create a word-row transformer model. This keeps standard coordinate-time
attention, uses coordinate-role LM heads, and trains on selected word-row
prefix positions:

```bash
python sandbox/sandbox.py create models/m11.v0.0.0.pth \
  --word-row-transformer \
  --dropout 0.5 \
  --epoch-num 4000 \
  --lr 0.001
```

`--word-row-transformer` implies `--coordinate-lm-heads`. Use it consistently
when creating, improving, or loading that checkpoint family.

Create a word-row transformer with coordinate-specific token embeddings. This
uses separate input embedding rows for each coordinate role, so the embedding
table has `syllable_num * vocab_size` rows instead of `vocab_size` rows:

```bash
python sandbox/sandbox.py create models/m12.v0.0.0.pth \
  --word-row-transformer \
  --coordinate-token-embeddings \
  --dropout 0.5 \
  --epoch-num 4000 \
  --lr 0.001
```

Use `--coordinate-token-embeddings` consistently when creating, improving, or
loading that checkpoint family.

Continue a coordinate-head model:

```bash
python sandbox/sandbox.py improve models/m5.v0.0.0.pth \
  --coordinate-lm-heads \
  --patch \
  --dropout 0.4 \
  --epoch-num 2000 \
  --lr 1e-4
```

Load a coordinate-head model with the current coordinate-aware decoding defaults:

```bash
python sandbox/sandbox.py load models/m4.v0.1.0.pth \
  --coordinate-lm-heads \
  --stream
```

Load a coordinate-head model with explicit coordinate-specific sampling:

```bash
python sandbox/sandbox.py load models/m4.v0.1.0.pth \
  --coordinate-lm-heads \
  --top-k-by-coordinate 3,4,6 \
  --temperature-by-coordinate 1,0.95,0.8 \
  --stream
```

Load a word-row transformer model:

```bash
python sandbox/sandbox.py load models/m11.v0.0.0.pth \
  --word-row-transformer \
  --stream
```

Load a model while forcing scalar sampling instead of coordinate-specific sampling. The `none` values are needed here because the coordinate-specific defaults are active unless they are explicitly turned off:

```bash
python sandbox/sandbox.py load models/m4.v0.1.0.pth \
  --coordinate-lm-heads \
  --top-k-by-coordinate none \
  --temperature-by-coordinate none \
  --top-k 3 \
  --temperature 1 \
  --stream
```

Create a standard model without coordinate-specific LM heads:

```bash
python sandbox/sandbox.py create models/m3.v0.0.0.pth \
  --dropout 0.4 \
  --epoch-num 4000 \
  --lr 2e-4
```

</details>

## `comparisons/compare_generation.py`

<details>
<summary>Compare ILM checkpoints with Hugging Face references</summary>
<br>


`comparisons/compare_generation.py` is a qualitative comparison harness. It
runs fixed prompts through an ILM checkpoint, a Hugging Face causal language
model, or both, then writes side-by-side reports to:

```text
comparisons/outputs/
```

Prompt generation shows a progress bar by default. Use `--no-progress` when
you want clean logs.

Install the optional Hugging Face dependencies only when you need the reference
backend:

```bash
pip install -r comparisons/requirements.txt
```

Compare a word-row transformer checkpoint with a reference model:

```bash
python comparisons/compare_generation.py \
  --backend both \
  --ilm-model-path models/m4.v0.0.1.pth \
  --word-row-transformer \
  --hf-reference karpathy-gpt2 \
  --prompts-file comparisons/prompts/ilm_quality.txt \
  --temperature 1 \
  --top-k 3 \
  --completed-words 300 \
  --hf-max-new-tokens 300
```

Run ILM only:

```bash
python comparisons/compare_generation.py \
  --backend ilm \
  --ilm-model-path models/m4.v0.0.1.pth \
  --word-row-transformer \
  --prompt "The queen" \
  --prompt "We will go battle against our enemies"
```

Run Hugging Face only:

```bash
python comparisons/compare_generation.py \
  --backend hf \
  --hf-reference sadia-gpt2-shakespeare \
  --prompt "The queen" \
  --hf-temperature 1 \
  --hf-top-k 3 \
  --hf-max-new-tokens 300
```

Open an interactive prompt loop for the Hugging Face reference model:

```bash
python comparisons/hf_sandbox.py \
  --hf-reference tinyshakespeare-42m \
  --temperature 1 \
  --top-k 3 \
  --max-new-tokens 300
```

Named reference presets:

```text
--hf-reference karpathy-gpt2             # OpenAI GPT-2 small, official GPT-2 path used by nanoGPT
--hf-reference shakespeare-gpt2          # community Shakespeare-tuned GPT-2-style model
--hf-reference sadia-gpt2-shakespeare    # smaller Shakespeare-tuned GPT-2 reference
--hf-reference tinyshakespeare-42m       # stronger 30.5M Tiny Shakespeare reference
--hf-reference fawern-gpt2-shakespeare   # stronger GPT-2 small Shakespeare fine-tune
```

Use `--hf-model MODEL_ID` for any custom Hugging Face model. It overrides
`--hf-reference`. Use `--hf-tokenizer TOKENIZER_ID` if the model should be
decoded with a base-model tokenizer. The `fawern-gpt2-shakespeare` preset uses
the GPT-2 tokenizer for this reason.

Inside the reference prompt loop:

```text
!config
!exit
```

Streaming is on by default. Use `--no-stream` to print only after generation is
complete.

The Hugging Face model must be loadable by `AutoModelForCausalLM`. Some nanoGPT
repositories publish raw training checkpoints rather than Transformers-compatible
models. If one reference does not load, try another reference model. Use
`--trust-remote-code` only for repositories you trust.

This harness is not a full benchmark. It is meant for reading outputs side by
side with fixed prompts and recorded sampling settings. Loss values are not
automatically comparable across tokenizers.

Prompt files:

```text
comparisons/prompts/shakespeare.txt
comparisons/prompts/ilm_quality.txt
```

Use `shakespeare.txt` for a short smoke test. Use `ilm_quality.txt` for a
broader qualitative pass over stage formatting, royal/family attractors,
action prompts, negation prompts, and plain-English stress tests.

</details>

## `ilm/utils/user_interface.py`

<details>
<summary>Interactive prompts, streaming, and plots</summary>
<br>


`ilm/utils/user_interface.py` is the interactive prompt used by `sandbox/sandbox.py` after creating, improving, or loading a model.

### Purpose

Use this interface to:

- type prompts and generate continuations
- exit the sandbox cleanly
- inspect model weights with plots

### Main Prompt

After `sandbox/sandbox.py` finishes setup, it enters:

```text
>>> 
```

Type a prompt:

```text
>>> Where is the queen
```

The UI tokenizes your prompt, generates `completed_words * syllable_num` coordinate tokens, detokenizes the result, and prints the continuation.

Current generation settings are passed from `sandbox/sandbox.py`:

```python
completed_words = 100
syllable_num = 3
temperature = 1
top_k = 3
top_k_by_coordinate = 3,4,6
temperature_by_coordinate = 1,0.95,0.8
stream = False
```

Set them from the command line:

```bash
python sandbox/sandbox.py load models/m3.v0.0.3.pth \
  --temperature 1 \
  --top-k-by-coordinate 3,4,6 \
  --completed-words 100
```

`completed_words=100` and `syllable_num=3` means generation asks for:

```text
300 coordinate tokens
```

Since each text token is represented by 3 coordinates, this is roughly 100 text tokens.

`temperature=1` controls randomness when coordinate-specific temperature is off. Lower values are more conservative and repetitive. Higher values are more surprising and more error-prone.

`top_k=3` means each next coordinate is sampled from only the 3 most likely coordinate values when coordinate-specific `top_k` is off.

For ILM, one text token is usually a three-coordinate code. You can therefore give each coordinate its own `top_k`:

```bash
python sandbox/sandbox.py load models/m3.v0.0.3.pth \
  --temperature 1 \
  --top-k-by-coordinate 3,5,8
```

With `syllable_num=3`, this means:

| Coordinate | Role | `top_k` |
| --- | --- | --- |
| 0 | broad semantic region | 3 |
| 1 | semantic refinement | 5 |
| 2 | local neighbor choice | 8 |

`--top-k-by-coordinate` overrides `--top-k` during generation when it is not `None`. The number of values must match `--syllable-num`.

`--temperature-by-coordinate` works the same way. It overrides `--temperature` when it is not `None`.

Use `none` to make the scalar values active:

```bash
python sandbox/sandbox.py load models/m3.v0.0.3.pth \
  --top-k-by-coordinate none \
  --temperature-by-coordinate none \
  --top-k 3 \
  --temperature 1
```

The load table marks scalar values as `overridden` whenever the coordinate-specific values are active.

Use `--stream` to print decoded words as soon as each coordinate group completes:

```bash
python sandbox/sandbox.py load models/m3.v0.0.3.pth \
  --stream
```

Streaming disables the inference progress bar so the generated text can appear cleanly in the terminal.

### `!exit`

Exit the main UI:

```text
>>> !exit
```

### `!plot`

Open model weight plotting mode:

```text
>>> !plot
```

The UI prints indexed model tensors:

```text
0) token_embedding_table.weight torch.Size(...)
1) pos_embedding_table.weight torch.Size(...)
...
```

Then it shows a nested prompt:

```text
~ >>> 
```

Type an index to plot that tensor:

```text
~ >>> 0
```

Exit plot mode:

```text
~ >>> !exit
```

This returns to the main generation prompt.

### What The Plots Show

For a 1D tensor, `!plot` draws a line plot of the parameter values.

For a 2D tensor, `!plot` shows two grayscale images:

```text
Original Image
Hierarchically Ordered Image
```

The original image is the raw matrix. Examples:

- token embedding table
- positional embedding table
- linear layer weights

The hierarchically ordered image reorders the matrix rows using Ward hierarchical clustering. This is an inspection tool: it tries to place similar rows near each other so structure can become visible.

Useful things to look for:

- Are embedding rows noisy or do they form bands/clusters?
- Are some dimensions much brighter/darker than others?
- Do layer weights show block-like structure?
- Does a trained model look more organized than an untrained model?

For tensors with 3 dimensions, the UI plots each first-axis slice side by side.
For tensors with more than 3 dimensions, the UI prints:

```text
Tensor is not 1D, 2D, or 3D, unable to plot.
```

### `!plot`

Use `!plot` to inspect model tensors:

```text
>>> !plot
```

The plot menu includes normal `state_dict` tensors. For 3D tensors, the first
axis is shown as side-by-side heatmaps. This is useful for future word-row
experiments that expose a bank of matrices in the checkpoint.

### Common UI Problems

`Language not recongized!`

The prompt contains a token that is not in the tokenizer JSON. Try words/capitalization that appear in the training text, or build a tokenizer with broader source data.

Generation loops:

Try changing `--temperature` or `--top-k` when running `sandbox/sandbox.py`. Lower temperature is more stable but more repetitive. Higher temperature is more creative but noisier.

</details>

## Troubleshooting

<details>
<summary>Common errors and fixes</summary>
<br>


Checkpoint load error:

The architecture settings in `sandbox/sandbox.py` probably do not match the checkpoint. Check `embedding_dim`, `block_size`, `layer_num`, and `vocab_size`.

Embedding API error:

Check that `.env` contains `OPENAI_API_KEY`, dependencies are installed, and the batch size is not too large.

</details>
