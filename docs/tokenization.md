# Tokenization

ILM exposes tokenization through `ilm.tokenizer.create_tokenizer` and
`ilm.tokenizer.load_tokenizer`. The SDK supports a relative-position builder
and an embedding-cluster builder. Both save a JSON mapping that is sufficient
to encode and decode text.

## Artifact Policy

| Artifact | Location | Required at Runtime | Git Policy |
| --- | --- | --- | --- |
| Training corpus | `data/corpora/` | Only for builds and training | Track selected sources |
| Tokenizer mapping | `data/tokenizers/*.json` | Yes | Track released mappings |
| Semantic labels | `data/tokenizers/*.semantic.json` | No | Optional analysis artifact |
| Build manifest | `data/tokenizers/*.manifest.json` | No | Track |
| Embedding cache | `data/cache/*.embeddings.npz` | No | Ignore |

The semantic-label sidecar translates a code into three human-readable centroid
labels. It supports analysis and plots, but the language model consumes only
the numeric tokenizer mapping. An embedding cache stores numerical vectors from
the configured embedding model so PCA and clustering can be rerun without a
new API request.

## Build a Tokenizer

`tests/quickstart.py` is the command-line entry point for building and loading
tokenizers without writing Python. It exposes the same builder settings as the
SDK and prints a tokenization round trip after completing the requested action.

```bash
python tests/quickstart.py \
  --mode build \
  --method embedding-cluster \
  --source-file data/corpora/training_old_english.txt \
  --target-file data/tokenizers/my_tokenizer.json \
  --cluster-method spherical-kmeans \
  --reduced-dim 10 \
  --embedding-batch-size 512 \
  --cache-file data/cache/my_tokenizer.embeddings.npz \
  --collision-report-limit 50
```

Embedding-cluster builds require `OPENAI_API_KEY`. The build writes a portable
JSON mapping and, when requested, a local embedding cache. `--reduced-dim 10`
is the maintained starting point for the embedding-cluster method.

The equivalent SDK call is:

```python
from ilm.tokenizer import create_tokenizer

tokenizer, detokenizer = create_tokenizer(
    source_file="data/corpora/training_old_english.txt",
    target_file="data/tokenizers/my_tokenizer.json",
    method="embedding-cluster",
    cluster_method="spherical-kmeans",
    reduced_dim=10,
    cache_file="data/cache/my_tokenizer.embeddings.npz",
)
```

Use an explicit `cache_file` for a released tokenizer. The SDK otherwise puts
the cache next to the target JSON, which is convenient for a one-off local
build but mixes a large regenerable file with portable artifacts.

Use `--lossless-tokenization` with `tests/quickstart.py`, or
`lossless_tokenization=True` in the SDK, when every source character must be
represented. This is required for byte-normalized evaluation on arbitrary text
such as enwik8.

For a semantic-label sidecar, add:

```python
semantic_spelling_file="data/tokenizers/my_tokenizer.semantic.json"
```

The optional `centroid_label_method="llm"` improves human-readable labels but
does not affect the tokenizer mapping or model training.

### Lossless Source Coverage

The default lexical splitter preserves the historical ILM word and punctuation
behavior. For a raw-byte benchmark such as enwik8, use
`lossless_tokenization=True` when building an embedding-cluster tokenizer. It
keeps recognized lexical tokens and emits each otherwise-unmatched character as
its own token. The selected mode is saved in the tokenizer JSON and is restored
by `load_tokenizer`, so training and evaluation use the same splitter.

```python
tokenizer, detokenizer = create_tokenizer(
    source_file="data/corpora/enwik8",
    target_file="experiments/evaluation/tokenizers/enwik8_lossless.json",
    method="embedding-cluster",
    depth=4,
    lossless_tokenization=True,
)
```

Use `experiments/check_tokenizer_coverage.py --require-lossless-encoding` to
verify that the frozen mapping represents every source byte before reporting a
raw-byte likelihood comparison. This mode changes the tokenizer vocabulary and
therefore requires a newly built tokenizer and newly trained model.

### Atomic Lexical Control

`load_atomic_lexical_tokenizer()` keeps a frozen tokenizer's lexical boundaries
but replaces each hierarchical code with one contiguous vocabulary ID. It is a
conventional learned input-table and output-head control for testing whether a
coordinate representation adds value beyond lexical segmentation alone. The
experiment sandbox exposes it through `--atomic-lexical` and infers the atomic
vocabulary size from the tokenizer JSON.

## Frozen-Tokenizer Diagnostics

After a tokenizer is frozen, do not rerun its build command merely to obtain a
sidecar or plots. `experiments/tokenizer_diagnostics.py` reads the tokenizer
JSON and its NPZ embedding cache without changing the mapping. It reconstructs
centroid diagnostics from the final assigned codes, which is especially useful
after collision repair.

```bash
python experiments/tokenizer_diagnostics.py \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --cache-file data/cache/evaluation_semantic_d10.embeddings.npz \
  --semantic-spelling-file \
    experiments/evaluation/tokenizers/semantic_d10.semantic.json \
  --centroid-label-method llm \
  --centroid-label-model gpt-4.1-mini \
  --plot-pca-2d --plot-pca-3d --plot-clusters \
  --plot-output-dir experiments/evaluation/tokenizers/figures
```

`closest-token` is the no-API default, but it uses a nearby vocabulary item as
the label and can produce artifacts such as `"provision:avail:badest"`. Use the
explicit LLM mode above for semantic labels. It makes one batched labeling call
per residual level, normally three calls for a depth-three tokenizer.

The figures open interactively and are also saved as one PDF per residual level
and dimension when `--plot-output-dir` is set. Add `--no-show-plots` to save
without opening plot windows. The script writes only the requested sidecar and
figures, never the tokenizer JSON. Its reconstructed residual centroids describe
the final frozen code assignment. They are not an archived copy of any
intermediate K-Means centers from the original build.

## Released Mappings

The primary current mapping is
`data/tokenizers/tokenizer_embedding_cluster_v1.json`. Its adjacent manifest
records the corpus checksum, residual-centroid settings, and local cache path.
The `d10` variants are retained as alternative semantic-clustering runs.

See [HOWTO.md](../HOWTO.md) for the complete CLI surface.
