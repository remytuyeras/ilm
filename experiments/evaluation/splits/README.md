# Split Manifests

Generate train, validation, and test text locally from the source corpus. Keep
the split byte ranges, context-gap policy, and SHA-256 hashes here. The generated
text files themselves need not be committed when they can be reconstructed from
the tracked source corpus and manifest.

The Tiny Shakespeare experiment uses three non-overlapping source-ordered
regions. It requests a 2,048-byte gap before each held-out region and aligns the
held-out boundaries to line starts. The frozen manifest records the resulting
excluded spans and exact byte ranges. Recreate that split with:

```bash
python experiments/prepare_text_split.py \
  --source-file data/corpora/training_old_english.txt \
  --output-dir experiments/evaluation/splits/tinyshakespeare \
  --max-context-bytes 2048
```

The training, validation, and test files are encoded separately by the training
code. Consequently, no sampled context crosses a split boundary. The enwik8
manifest records its canonical adjacent 90M/5M/5M byte partition.
