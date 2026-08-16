# Model Comparisons

This folder is for comparing ILM generations against reference models without
mixing those experiments into `sandbox/`.

The comparison script can run:

- an ILM checkpoint from `models/`
- a Hugging Face causal language model
- both backends on the same prompts

Generated reports are written to:

```text
comparisons/outputs/
```

Those reports are experiment artifacts and are ignored by git.

Prompt generation shows a progress bar by default. Use `--no-progress` for
clean logs or when redirecting output.

## Setup

The ILM backend uses the repository dependencies.

The Hugging Face backend needs the optional comparison dependencies:

```bash
pip install -r comparisons/requirements.txt
```

## Compare ILM Against A Reference

Example using a word-row transformer checkpoint:

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
  --hf-max-new-tokens 300 \
  --progress
```

`karpathy-gpt2` resolves to the OpenAI GPT-2 small checkpoint loaded through
Transformers, matching the official nanoGPT path for GPT-2 checkpoints.

Use one of the Shakespeare-tuned community references with:

```bash
--hf-reference shakespeare-gpt2
--hf-reference sadia-gpt2-shakespeare
```

Use a stronger challenge reference with:

```bash
--hf-reference tinyshakespeare-42m
--hf-reference fawern-gpt2-shakespeare
```

Use any custom Hugging Face model that works with `AutoModelForCausalLM`:

```bash
--hf-model Esmaelmoat/SHAKESPEARE_GPT2
```

`--hf-model` overrides `--hf-reference`. Use `--hf-tokenizer TOKENIZER_ID` when
a fine-tuned checkpoint does not publish tokenizer files cleanly or should use
its base model tokenizer. The `fawern-gpt2-shakespeare` preset uses the GPT-2
tokenizer for this reason. Some nanoGPT checkpoints on Hugging Face are raw
training checkpoints rather than Transformers-compatible models. If one model
does not load, try another reference model or use `--trust-remote-code` only for
repositories you trust.

## ILM Only

```bash
python comparisons/compare_generation.py \
  --backend ilm \
  --ilm-model-path models/m4.v0.0.1.pth \
  --word-row-transformer \
  --prompt "The queen" \
  --prompt "We will go battle against our enemies"
```

If the ILM checkpoint was trained with coordinate-specific input embeddings,
load it with the matching architecture flag:

```bash
--coordinate-token-embeddings
```

## Prompt Sets

Use the small default Shakespeare prompt set:

```bash
--prompts-file comparisons/prompts/shakespeare.txt
```

Use the broader ILM quality prompt set:

```bash
--prompts-file comparisons/prompts/ilm_quality.txt
```

`ilm_quality.txt` includes royal/family attractor tests, dialogue-format tests,
action prompts, negation prompts, and plain-English stress tests.

## Hugging Face Only

```bash
python comparisons/compare_generation.py \
  --backend hf \
  --hf-reference karpathy-gpt2 \
  --prompt "The queen" \
  --hf-temperature 1 \
  --hf-top-k 3 \
  --hf-max-new-tokens 300
```

## Interactive Reference Sandbox

Use `hf_sandbox.py` when you want to interact with the reference model directly,
without running a comparison report:

```bash
python comparisons/hf_sandbox.py \
  --hf-reference tinyshakespeare-42m \
  --temperature 1 \
  --top-k 3 \
  --max-new-tokens 300
```

The prompt loop supports:

```text
!config
!exit
```

Streaming is on by default. Disable it with:

```bash
--no-stream
```

## Notes On Fairness

This is a qualitative comparison harness. It is useful for reading outputs
side by side, counting obvious invalid tokens, and testing repeated prompts.

It is not automatically a fair benchmark because ILM and reference models may
use different tokenizers, training data, context windows, and loss units. For a
paper-like comparison, keep prompts fixed, run multiple seeds or samples, record
all sampling settings, and compare both text quality and measurable artifacts
such as repetition rate and invalid-token rate.
