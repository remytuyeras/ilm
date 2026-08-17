# Evaluation

`comparisons/compare_generation.py` compares an ILM checkpoint with configured
Hugging Face references. It writes JSON and Markdown reports under
`comparisons/outputs/`, which are local generated artifacts by default.

External Shakespeare GPT-2 models are useful qualitative references, but they
are not controlled baselines when their pretraining and fine-tuning data differ
from ILM's corpus. The controlled evaluation therefore treats parameter-matched
from-scratch BPE and character Transformers as primary baselines.

Use a fixed prompt file for a comparison:

```bash
python comparisons/compare_generation.py \
  --backend both \
  --ilm-model-path models/release/m5.v0.0.1.pth \
  --ilm-objective \
  --ilm-output-heads \
  --ilm-input-embeddings \
  --prompts-file comparisons/prompts/shakespeare.txt \
  --temperature 1 --top-k 3 \
  --samples-per-prompt 5 \
  --generation-seed 13
```

The comparison report stores ANSI-free completions, one deterministic seed per
prompt/sample pair, invalid-code counts, and repeated n-gram metrics. A fixed
`--generation-seed` is reset before every completion. It makes the sampling
protocol reproducible without making the five generated samples identical.

Evaluate a frozen ILM checkpoint on an explicit held-out split with
teacher-forced bits per UTF-8 byte (BPB):

```bash
python experiments/evaluate_ilm.py \
  --model-path models/evaluation/c_full_seed13.pth \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --oov-policy error \
  --ilm-input-embeddings --ilm-output-heads --ilm-objective
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
  --evaluation-report 'models/evaluation/*.test_metrics.json' \
  --output-dir experiments/evaluation/results/aggregate
```

The planned primary metrics are held-out bits per UTF-8 byte, invalid-code rate,
repetition measures, and blinded prompt-level quality comparisons. See
[metric_evidence_plan.md](../experiments/evaluation/metric_evidence_plan.md).
