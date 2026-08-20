# Generation Comparison Tools

This directory contains the qualitative generation-comparison scripts and prompt
sets. The maintained usage guide is [docs/evaluation.md](../docs/evaluation.md).

- `compare_generation.py` runs ILM, Hugging Face, or paired completions and
  writes JSON and Markdown reports under `comparisons/outputs/`.
- `hf_sandbox.py` opens an interactive prompt loop for a configured Hugging Face
  reference.
- `prompts/` contains reusable prompt sets.
- `requirements.txt` contains optional Transformers dependencies.

Generated reports and Hugging Face caches are intentionally ignored by Git.
