# Evaluation

ILM has two complementary evaluation paths. Teacher-forced evaluation measures
held-out negative log-likelihood in bits per UTF-8 byte (BPB). Generation
comparison produces readable completions and repetition diagnostics. The former
is the primary quantitative measure for controlled studies. The latter is a
qualitative inspection tool.

## Teacher-Forced BPB

Evaluate a frozen ILM checkpoint on an explicit held-out split with
teacher-forced bits per UTF-8 byte (BPB):

```bash
python experiments/evaluate_ilm.py \
  --model-path models/evaluation/c_full_6m_seed13.pth \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --oov-policy error \
  --ilm-input-embeddings --ilm-output-heads --ilm-objective \
  --output-file experiments/evaluation/results/c_full_6m_seed13.test_metrics.json
```

Teacher forcing supplies each true preceding coordinate and measures the
negative log-probability of the next true coordinate. BPB divides the total
negative log-likelihood by the original UTF-8 bytes scored, which permits
comparison across tokenizers with different coordinate counts.

Use `--oov-policy error` for fixed-corpus evaluation, where an OOV indicates a
tokenizer/data mismatch. For a new transfer document, use
`--oov-policy fallback` to complete the run and interpret the reported OOV rate
as coverage. All unseen words share one fallback code, so BPB from a high-OOV
document measures performance after a lossy replacement rather than fully
represented text.

Aggregate generation and held-out reports into CSV files, bootstrap intervals,
and PDF figures:

```bash
python experiments/aggregate_results.py \
  --generation-report 'comparisons/outputs/evaluation/*.json' \
  --evaluation-report 'experiments/evaluation/results/*.test_metrics.json' \
  --output-dir experiments/evaluation/results/aggregate
```

The completed controlled metrics and their interpretation are recorded in
[experiments/RESULTS.md](../experiments/RESULTS.md).

## Generation Comparisons

`comparisons/compare_generation.py` compares an ILM checkpoint with configured
Hugging Face references. It writes JSON and Markdown reports under
`comparisons/outputs/`, which are local generated artifacts by default. Prompt
generation displays a progress bar unless `--no-progress` is passed.

Install the optional reference-model dependencies first:

```bash
pip install -r comparisons/requirements.txt
```

Use the same prompt file and sampling settings for both backends:

```bash
python comparisons/compare_generation.py \
  --backend both \
  --ilm-model-path path/to/model.pth \
  --hf-reference fawern-gpt2-shakespeare \
  --prompts-file comparisons/prompts/ilm_quality.txt \
  --temperature 1 \
  --top-k 3 \
  --completed-words 300 \
  --hf-max-new-tokens 300 \
  --samples-per-prompt 5 \
  --generation-seed 13
```

Add `--ilm-input-embeddings`, `--ilm-output-heads`, and `--ilm-objective` when
the ILM checkpoint was trained with those architecture options. Check its JSON
metadata rather than assuming all three are enabled.

Reference aliases include `karpathy-gpt2`, `shakespeare-gpt2`,
`sadia-gpt2-shakespeare`, `tinyshakespeare-42m`, and
`fawern-gpt2-shakespeare`. Pass `--hf-model MODEL_ID` to load another
Transformers-compatible Hugging Face model. `--hf-tokenizer TOKENIZER_ID` is
available when a checkpoint requires a separate tokenizer.

Run only one backend with `--backend ilm` or `--backend hf`. The interactive
reference prompt loop is also available:

```bash
python comparisons/hf_sandbox.py \
  --hf-reference fawern-gpt2-shakespeare \
  --temperature 1 \
  --top-k 3 \
  --max-new-tokens 300
```

## Interpreting Generation Reports

Reports store ANSI-free completions, configured sampling values, invalid-code
counts, and repeated n-gram metrics. They are useful for finding repetition
attractors, format breakdowns, and invalid lexical codes. They do not by
themselves form a controlled benchmark: reference models can differ in
tokenizer, training corpus, context length, parameter count, and optimization.
Use the experiment protocol in [reproducibility.md](reproducibility.md) for
paper evidence.
