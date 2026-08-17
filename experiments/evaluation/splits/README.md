# Split Manifests

Generate train, validation, and test text locally from the source corpus. Keep
the split byte ranges, context-gap policy, and SHA-256 hashes here. The generated
text files themselves need not be committed when they can be reconstructed from
the tracked source corpus and manifest.

Create them with:

```bash
python experiments/prepare_text_split.py \
  --source-file data/corpora/training_old_english.txt \
  --output-dir experiments/evaluation/splits/tinyshakespeare \
  --max-context-bytes 60
```

The `max-context-bytes` gap is at least the largest model context measured in
raw bytes. It keeps examples around a split boundary out of every split file.
