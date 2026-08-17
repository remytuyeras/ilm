# Released Tokenizers

Each `*.json` file is a portable tokenizer mapping accepted by
`ilm.load_tokenizer`. The embedding-cluster mappings also carry build metadata
under their `metadata` field.

Files named `*.semantic.json` are optional semantic-label sidecars. Files named
`*.manifest.json` record the corpus checksum and the associated local cache
path. The cache itself belongs in `data/cache/` and is not committed.

To rebuild an embedding-cluster tokenizer while keeping the cache outside this
directory:

```bash
python tests/quickstart.py --mode build \
  --method embedding-cluster \
  --source-file data/corpora/training_old_english.txt \
  --target-file data/tokenizers/tokenizer_embedding_cluster_v1.json \
  --cache-file data/cache/tokenizer_embedding_cluster_v1.embeddings.npz \
  --cluster-method spherical-kmeans \
  --reduced-dim 10
```
