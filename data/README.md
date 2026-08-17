# Data Assets

`data/corpora/` contains source text used by examples and local experiments.
`data/tokenizers/` contains portable tokenizer mappings that can be loaded by
the SDK. `data/cache/` contains local embedding caches and is intentionally
ignored by Git.

The core runtime artifact is a tokenizer JSON file. Semantic spelling files are
optional interpretive sidecars. They help inspect centroid labels but are not
read by `ilm.load_tokenizer` or the Transformer.

For the public artifact policy and rebuild commands, see
[docs/tokenization.md](../docs/tokenization.md).
