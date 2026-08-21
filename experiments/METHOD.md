# Method

This file records the executable protocol for the controlled ILM experiments.
The completed measurements are reported in [RESULTS.md](RESULTS.md).

## Experimental Protocol At A Glance

All reported models are trained from scratch for 6,000 updates with seeds
`13`, `29`, and `47`. The semantic tokenizer is frozen before model training.
It is fitted once from the complete frozen corpus, while train, validation,
and test splits are used only for language-model fitting and evaluation. This
is a closed-corpus tokenizer protocol, not a train-only vocabulary protocol.

| Corpus | Model tiers | Teacher-forced metric | Evaluation mode |
| --- | --- | --- | --- |
| Tiny Shakespeare | approximately 6.5M and 15.5M parameters | BPB | full context |
| enwik8 | approximately 6.5M and 15.5M parameters | BPB | block reset |

The principal controls retain the same corpus split and training horizon.
Atomic Lexical preserves ILM's lexical segmentation while replacing coordinate
codes with atomic IDs. Random Flat preserves the same final coordinate code
set while permuting word-to-code assignments. Byte or character nanoGPT is a
from-scratch reference baseline. Exact parameter counts and completed outcomes
are maintained in [RESULTS.md](RESULTS.md).

## Frozen Corpus And Splits

The first corpus is `data/corpora/training_old_english.txt`, the local Tiny
Shakespeare source. Its semantic tokenizer is fitted once from the **complete,
unchanged corpus**. The train, validation, and test files are then used only
for language-model fitting and evaluation.

The split helper requests a 2,048-byte context separation at each boundary,
then begins held-out files at line boundaries. For the frozen split, this
excludes 2,038 bytes at the first boundary and 2,060 bytes at the second. Line
alignment matters because the tokenizer distinguishes tokens such as
`" bloody"` and `"bloody"`.

```bash
python experiments/prepare_text_split.py \
  --source-file data/corpora/training_old_english.txt \
  --output-dir experiments/evaluation/splits/tinyshakespeare \
  --max-context-bytes 2048 \
  --overwrite
```

Current lexical coverage result for `semantic_d10.json`:

| Split | Tokens | OOV tokens |
| --- | ---: | ---: |
| Train | 241,433 | 0 |
| Validation | 29,642 | 0 |
| Test | 30,766 | 0 |

Recheck coverage after recreating a split. The primary experiments use
`--oov-policy error`. `fallback` is reserved for an explicitly documented
transfer or coverage probe on a new document.

Teacher-forced Tiny Shakespeare evaluations additionally use
`--require-lossless-encoding` on the held-out test split. That check verifies
source reconstruction, rather than only lexical OOV coverage.

```bash
python experiments/check_tokenizer_coverage.py \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --text experiments/evaluation/splits/tinyshakespeare/train.txt \
  --text experiments/evaluation/splits/tinyshakespeare/validation.txt \
  --text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --oov-policy error \
  --output-file experiments/evaluation/results/tokenizer_coverage.json
```

## Frozen Semantic Tokenizer

`semantic_d10.json` is the primary tokenizer. It uses
`text-embedding-3-small`, featurewise standardization before PCA, and
spherical k-means to construct a three-coordinate base-64 code. For
non-whitespace tokens, surrounding whitespace is stripped for the embedding
request only. Standalone whitespace uses a tagged placeholder. The original
lexical spelling remains the tokenizer key.

| Property | Value |
| --- | ---: |
| Vocabulary entries | 15,030 |
| Unique direct and reverse codes | 15,030 |
| Code depth | 3 |
| Coordinate base | 64 |
| PCA dimension | 10 |
| Clustering seed | 13 |
| Residual collision repairs | 3,063 |
| Frozen tokenizer SHA-256 | `419e1d614d35dba1cfcd60c1a8045f6c5b40688b2aaf5bdb1633b0e39c3f6c80` |

The repairs preserve a bijective tokenizer mapping. They are a material
diagnostic, not a reason to claim that every final path is an unmodified
clustering assignment.

```bash
mkdir -p experiments/evaluation/tokenizers

python tests/quickstart.py \
  --mode build \
  --method embedding-cluster \
  --source-file data/corpora/training_old_english.txt \
  --target-file experiments/evaluation/tokenizers/semantic_d10.json \
  --cluster-method spherical-kmeans \
  --reduced-dim 10 \
  --embedding-batch-size 512 \
  --cache-file data/cache/evaluation_semantic_d10.embeddings.npz \
  --random-state 13 \
  --collision-report-limit 50
```

### Diagnostics And Interpretation Sidecars

The semantic spelling sidecar and PCA plots are generated from the frozen
mapping. They do not change tokenizer codes or participate in model training.
The LLM labels are an interpretation aid, not supervision and not evidence of
semantic correctness.

```bash
python experiments/tokenizer_diagnostics.py \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --cache-file data/cache/evaluation_semantic_d10.embeddings.npz \
  --semantic-spelling-file \
    experiments/evaluation/tokenizers/semantic_d10.semantic.json \
  --centroid-label-method llm \
  --centroid-label-model gpt-4.1-mini \
  --random-state 13 \
  --plot-pca-2d \
  --plot-pca-3d \
  --plot-clusters \
  --plot-output-dir experiments/evaluation/tokenizers/figures \
  --no-show-plots
```

## ILM Model Families

All three ILM options must be independently switchable:

| Option | Controlled change |
| --- | --- |
| `--ilm-input-embeddings` | Uses a separate embedding map for each coordinate role. |
| `--ilm-output-heads` | Uses a separate language-model output head for each coordinate role. |
| `--ilm-objective` | Applies the word-prefix objective, which excludes suffix fragments at a batch boundary. |

The first controlled pair is:

```text
C-Flat-6M = semantic tokenizer + ordinary coordinate Transformer
C-Full-6M = semantic tokenizer + all three ILM options
```

## Optimizer Crossover

### enwik8

The original Byte GPT and ILM runs used different AdamW protocols. The
permutation and Flat-to-Full controls remain optimizer matched because they
stay within the ILM implementation. The completed crossover resolves the
optimizer mismatch without turning Byte-to-Flat into a representation-only
causal comparison.

| Model | Optimizer profile | Learning-rate schedule | Status |
| --- | --- | --- | --- |
| Byte GPT | nanoGPT matrix-only decay | cosine, 100-step warmup, `1e-3` to `1e-4` | completed native reference |
| Flat ILM | all-parameter decay | constant `1e-3` | completed primary ILM row |
| Byte GPT | all-parameter decay | constant `1e-3` | completed crossover and primary reference row |
| Flat ILM | nanoGPT matrix-only decay | cosine, 100-step warmup, `1e-3` to `1e-4` | completed crossover |

The completed crossover retains the enwik8 6.5M architectures, data, context,
batch size, dropout, and 6,000-update horizon. `all_parameters` applies weight decay to every
trainable tensor, which is the historical ILM behavior. `nanogpt` applies
weight decay only to tensors with at least two dimensions. The protocols also
match `weight_decay`, AdamW betas, gradient clipping, and the learning-rate
schedule.

### Byte GPT With the ILM Optimizer Protocol

```bash
mkdir -p experiments/evaluation/runs experiments/evaluation/results

for seed in 13 29 47; do
  (
    cd baselines/nanoGPT
    ../../venv/bin/python train.py \
      ../../experiments/evaluation/configs/nanogpt_enwik8_byte_6m.py \
      --seed=$seed \
      --out_dir=../../experiments/evaluation/runs/enwik8_byte_gpt_6m_all_parameters_constant_seed$seed \
      --optimizer_profile='all_parameters' \
      --decay_lr=False \
      --learning_rate=0.001 \
      --weight_decay=0.01 \
      --beta1=0.9 --beta2=0.999 --grad_clip=0.0
  )

  venv/bin/python experiments/evaluate_nanogpt_char.py \
    --checkpoint-path experiments/evaluation/runs/enwik8_byte_gpt_6m_all_parameters_constant_seed$seed/ckpt.pt \
    --data-dir baselines/nanoGPT/data/ilm_enwik8_byte \
    --test-text experiments/evaluation/splits/enwik8/test.txt \
    --evaluation-mode block-reset --evaluation-batch-size 128 \
    --output-file experiments/evaluation/results/enwik8_byte_gpt_6m_all_parameters_constant_seed$seed.test_metrics.json
done
```

### Flat ILM With the nanoGPT Optimizer Protocol

```bash
mkdir -p models/evaluation experiments/evaluation/results

for seed in 13 29 47; do
  venv/bin/python sandbox/sandbox.py create \
    models/evaluation/enwik8_lossless_s4_c_flat_6m_nanogpt_cosine_seed$seed.pth \
    --seed $seed \
    --tokenizer-json experiments/evaluation/tokenizers/enwik8_lossless_semantic_d10_s4.json \
    --train-text experiments/evaluation/splits/enwik8/train.txt \
    --validation-text experiments/evaluation/splits/enwik8/validation.txt \
    --test-text experiments/evaluation/splits/enwik8/test.txt \
    --oov-policy error \
    --syllable-num 4 --word-block-size 20 --block-size 80 \
    --embedding-dim 300 --head-num 6 --layer-num 6 --batch-size 32 \
    --dropout 0.5 --epoch-num 6000 --lr 0.001 \
    --optimizer-profile nanogpt \
    --lr-schedule cosine --warmup-iters 100 --lr-decay-iters 6000 --min-lr 0.0001 \
    --weight-decay 0.1 --beta1 0.9 --beta2 0.99 --grad-clip 1.0 \
    --validation-interval 200 --no-interactive

  venv/bin/python experiments/evaluate_ilm.py \
    --model-path models/evaluation/enwik8_lossless_s4_c_flat_6m_nanogpt_cosine_seed$seed.pth \
    --tokenizer-json experiments/evaluation/tokenizers/enwik8_lossless_semantic_d10_s4.json \
    --test-text experiments/evaluation/splits/enwik8/test.txt \
    --seed $seed --oov-policy error --require-lossless-encoding \
    --evaluation-mode block-reset --evaluation-batch-size 128 \
    --syllable-num 4 --word-block-size 20 --block-size 80 \
    --embedding-dim 300 --head-num 6 --layer-num 6 --dropout 0.5 \
    --output-file experiments/evaluation/results/enwik8_lossless_s4_c_flat_6m_nanogpt_cosine_seed$seed.test_metrics.json
done
```

Interpret the four cells jointly. Flat ILM is lower than Byte GPT under both
protocols, while the size of the gap is optimizer sensitive. The within-ILM
permutation and Flat-to-Full conclusions do not depend on this crossover.

### Tiny Shakespeare

The 6.5M Tiny Shakespeare Character GPT and Flat ILM runs had the same optimizer
mismatch. The completed crossover retains the split, architectures, approximate
source context, dropout, batch size, and 6,000-update horizon.

| Model | Optimizer profile | Learning-rate schedule | Status |
| --- | --- | --- | --- |
| Character GPT | nanoGPT matrix-only decay | cosine, 100-step warmup, `1e-3` to `1e-4` | completed native reference |
| Flat ILM | all-parameter decay | constant `1e-3` | completed primary ILM row |
| Character GPT | all-parameter decay | constant `1e-3` | completed crossover and primary reference row |
| Flat ILM | nanoGPT matrix-only decay | cosine, 100-step warmup, `1e-3` to `1e-4` | completed crossover |

#### Character GPT With the ILM Optimizer Protocol

```bash
mkdir -p experiments/evaluation/runs experiments/evaluation/results

for seed in 13 29 47; do
  (
    cd baselines/nanoGPT
    ../../venv/bin/python train.py \
      ../../experiments/evaluation/configs/nanogpt_char_6m.py \
      --seed=$seed \
      --out_dir=../../experiments/evaluation/runs/char_gpt_6m_all_parameters_constant_seed$seed \
      --optimizer_profile='all_parameters' \
      --decay_lr=False \
      --learning_rate=0.001 \
      --weight_decay=0.01 \
      --beta1=0.9 --beta2=0.999 --grad_clip=0.0
  )

  venv/bin/python experiments/evaluate_nanogpt_char.py \
    --checkpoint-path experiments/evaluation/runs/char_gpt_6m_all_parameters_constant_seed$seed/ckpt.pt \
    --data-dir baselines/nanoGPT/data/ilm_tinyshakespeare \
    --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
    --evaluation-batch-size 32 \
    --output-file experiments/evaluation/results/char_gpt_6m_all_parameters_constant_seed$seed.test_metrics.json
done
```

#### Flat ILM With the nanoGPT Optimizer Protocol

```bash
mkdir -p models/evaluation experiments/evaluation/results

for seed in 13 29 47; do
  venv/bin/python sandbox/sandbox.py create \
    models/evaluation/c_flat_6m_nanogpt_cosine_seed$seed.pth \
    --seed $seed \
    --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
    --train-text experiments/evaluation/splits/tinyshakespeare/train.txt \
    --validation-text experiments/evaluation/splits/tinyshakespeare/validation.txt \
    --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
    --oov-policy error \
    --word-block-size 20 --block-size 60 \
    --embedding-dim 300 --head-num 6 --layer-num 6 --batch-size 32 \
    --dropout 0.5 --epoch-num 6000 --lr 0.001 \
    --optimizer-profile nanogpt \
    --lr-schedule cosine --warmup-iters 100 --lr-decay-iters 6000 --min-lr 0.0001 \
    --weight-decay 0.1 --beta1 0.9 --beta2 0.99 --grad-clip 1.0 \
    --validation-interval 200 --no-interactive

  venv/bin/python experiments/evaluate_ilm.py \
    --model-path models/evaluation/c_flat_6m_nanogpt_cosine_seed$seed.pth \
    --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
    --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
    --seed $seed --oov-policy error --require-lossless-encoding \
    --evaluation-batch-size 32 \
    --word-block-size 20 --block-size 60 \
    --embedding-dim 300 --head-num 6 --layer-num 6 --dropout 0.5 \
    --output-file experiments/evaluation/results/c_flat_6m_nanogpt_cosine_seed$seed.test_metrics.json
done
```

As with enwik8, report the Tiny Shakespeare crossover as a four-cell result.
The schedule sensitivity of the Character-GPT-to-Flat-ILM comparison is
determined by the two crossover cells, while the permutation and Flat-to-Full
controls remain internally matched.

### Fixed Code-Permutation Control

`Random-Flat-6M` uses the same lexical vocabulary, final set of 15,030 code
strings, coordinate base, code depth, and source segmentation as
`C-Flat-6M`. Each assignment applies one fixed random permutation to the
assignment from lexical entries to final code strings. This removes the semantic association
between words and their codes without changing code occupancy or type-level
coordinate marginals. Corpus-frequency-weighted coordinate marginals are not
preserved. Each fixed permutation is shared across model seeds `13`, `29`, and
`47`.

The permutation index is an independent representation replicate. It is not a
model-training seed. The original completed control remains index `1` and
keeps its legacy filenames. Indices `2` and `3` add twelve new training and
evaluation runs across the two corpora.

| Permutation index | Permutation seed | Tiny Shakespeare result prefix | enwik8 result prefix | Status |
| ---: | ---: | --- | --- | --- |
| `1` | `314159` | `random_flat_6m` | `enwik8_lossless_s4_random_flat_6m` | completed |
| `2` | `271828` | `random_2_flat_6m` | `enwik8_lossless_s4_random_2_flat_6m` | completed |
| `3` | `161803` | `random_3_flat_6m` | `enwik8_lossless_s4_random_3_flat_6m` | completed |

The final analysis should summarize each permutation index over its three
model seeds, then compare the three permutation-level effects with Flat ILM.
It should not present the nine Permuted Flat runs as nine exchangeable model
seeds.

```bash
python experiments/create_permuted_tokenizer.py \
  --source-tokenizer experiments/evaluation/tokenizers/semantic_d10.json \
  --target-tokenizer experiments/evaluation/tokenizers/control_permuted_codes_s3_seed314159.json \
  --permutation-seed 314159
```

Train and evaluate `Random-Flat-6M` with model seeds `13`, `29`, and `47`.
Use the same architecture and optimization settings as `C-Flat-6M`.

```bash
for seed in 13 29 47; do
  python sandbox/sandbox.py create models/evaluation/random_flat_6m_seed$seed.pth \
    --seed $seed \
    --tokenizer-json experiments/evaluation/tokenizers/control_permuted_codes_s3_seed314159.json \
    --train-text experiments/evaluation/splits/tinyshakespeare/train.txt \
    --validation-text experiments/evaluation/splits/tinyshakespeare/validation.txt \
    --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
    --oov-policy error \
    --word-block-size 20 --block-size 60 \
    --embedding-dim 300 --head-num 6 --layer-num 6 --batch-size 32 \
    --dropout 0.5 --epoch-num 6000 --lr 0.001 \
    --validation-interval 200 \
    --no-interactive

  python experiments/evaluate_ilm.py \
    --model-path models/evaluation/random_flat_6m_seed$seed.pth \
    --tokenizer-json experiments/evaluation/tokenizers/control_permuted_codes_s3_seed314159.json \
    --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
    --seed $seed --oov-policy error \
    --require-lossless-encoding \
    --evaluation-batch-size 32 \
    --word-block-size 20 --block-size 60 \
    --embedding-dim 300 --head-num 6 --layer-num 6 --dropout 0.5 \
    --output-file experiments/evaluation/results/random_flat_6m_seed$seed.test_metrics.json
done
```

### Additional permutation replicates

Create, train, and evaluate the two additional Tiny Shakespeare code
assignments. The completed index-`1` files are not overwritten. Indices `2`
and `3` are now completed and retained as rerunnable commands.

```bash
for index in 2 3; do
  case $index in
    2) permutation_seed=271828 ;;
    3) permutation_seed=161803 ;;
  esac
  tokenizer=experiments/evaluation/tokenizers/control_permuted_codes_s3_seed$permutation_seed.json

  python experiments/create_permuted_tokenizer.py \
    --source-tokenizer experiments/evaluation/tokenizers/semantic_d10.json \
    --target-tokenizer $tokenizer \
    --permutation-seed $permutation_seed

  for seed in 13 29 47; do
    python sandbox/sandbox.py create models/evaluation/random_${index}_flat_6m_seed$seed.pth \
      --seed $seed \
      --tokenizer-json $tokenizer \
      --train-text experiments/evaluation/splits/tinyshakespeare/train.txt \
      --validation-text experiments/evaluation/splits/tinyshakespeare/validation.txt \
      --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
      --oov-policy error \
      --syllable-num 3 --word-block-size 20 --block-size 60 \
      --embedding-dim 300 --head-num 6 --layer-num 6 --batch-size 32 \
      --dropout 0.5 --epoch-num 6000 --lr 0.001 \
      --validation-interval 500 --no-interactive

    python experiments/evaluate_ilm.py \
      --model-path models/evaluation/random_${index}_flat_6m_seed$seed.pth \
      --tokenizer-json $tokenizer \
      --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
      --seed $seed --oov-policy error --require-lossless-encoding \
      --evaluation-batch-size 32 \
      --syllable-num 3 --word-block-size 20 --block-size 60 \
      --embedding-dim 300 --head-num 6 --layer-num 6 --dropout 0.5 \
      --output-file experiments/evaluation/results/random_${index}_flat_6m_seed$seed.test_metrics.json
  done
done
```

All Tiny Shakespeare model families use explicit train, validation, and test
files, a shared 6,000-step horizon, and held-out test BPB. Completed outcomes
are recorded in [RESULTS.md](RESULTS.md).

### Seed 13

<details open>
<summary>Completed: character GPT, C-Flat-6M, and C-Full-6M</summary>

#### Character GPT

The seed-13 character baseline uses the shared nanoGPT configuration described
in [Character nanoGPT Baseline](#character-nanogpt-baseline).

```bash
cd baselines/nanoGPT

../../venv/bin/python train.py \
  ../../experiments/evaluation/configs/nanogpt_char_6m.py

cd ../..
```

```bash
python experiments/evaluate_nanogpt_char.py \
  --checkpoint-path experiments/evaluation/runs/char_gpt_6m_seed13/ckpt.pt \
  --data-dir baselines/nanoGPT/data/ilm_tinyshakespeare \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --output-file experiments/evaluation/results/char_gpt_6m_seed13.test_metrics.json
```

**Observed character seed-13 pilot.** The 6.50M-parameter checkpoint reached
step 6,000 with a best sampled validation loss of `1.735141`. Exact held-out
evaluation reported `2.734766` BPB over 109,477 scored UTF-8 bytes. The report
is `experiments/evaluation/results/char_gpt_6m_seed13.test_metrics.json`.

#### C-Flat-6M

The completed flat control keeps all ILM architecture options off. It matches
the character baseline with six layers, width 300, six attention heads, roughly
74 bytes of context, and 6,000 training steps.

```bash
python sandbox/sandbox.py create models/evaluation/c_flat_6m_seed13.pth \
  --seed 13 \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --train-text experiments/evaluation/splits/tinyshakespeare/train.txt \
  --validation-text experiments/evaluation/splits/tinyshakespeare/validation.txt \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --oov-policy error \
  --word-block-size 20 --block-size 60 \
  --embedding-dim 300 --head-num 6 --layer-num 6 --batch-size 32 \
  --dropout 0.5 --epoch-num 6000 --lr 0.001 \
  --validation-interval 200
```

```bash
python experiments/evaluate_ilm.py \
  --model-path models/evaluation/c_flat_6m_seed13.pth \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --seed 13 \
  --oov-policy error \
  --evaluation-batch-size 32 \
  --word-block-size 20 --block-size 60 \
  --embedding-dim 300 --head-num 6 --layer-num 6 \
  --dropout 0.5 \
  --output-file experiments/evaluation/results/c_flat_6m_seed13.test_metrics.json
```

**Observed flat seed-13 pilot.** `C-Flat-6M` contains 6,555,064 parameters and
reached step 6,000 after recording 14,200,730 source bytes observed during
training. Its exact held-out result is `2.159404` BPB over 109,477.67 scored
UTF-8 bytes. The report is
`experiments/evaluation/results/c_flat_6m_seed13.test_metrics.json`.

#### C-Full-6M

The next run adds all three independently implemented ILM changes. Keep every
other setting identical to the flat control.

```bash
python sandbox/sandbox.py create models/evaluation/c_full_6m_seed13.pth \
  --seed 13 \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --train-text experiments/evaluation/splits/tinyshakespeare/train.txt \
  --validation-text experiments/evaluation/splits/tinyshakespeare/validation.txt \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --oov-policy error \
  --ilm-objective \
  --ilm-input-embeddings \
  --ilm-output-heads \
  --word-block-size 20 --block-size 60 \
  --embedding-dim 300 --head-num 6 --layer-num 6 --batch-size 32 \
  --dropout 0.5 --epoch-num 6000 --lr 0.001 \
  --validation-interval 200
```

```bash
python experiments/evaluate_ilm.py \
  --model-path models/evaluation/c_full_6m_seed13.pth \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --seed 13 \
  --oov-policy error \
  --evaluation-batch-size 32 \
  --word-block-size 20 --block-size 60 \
  --embedding-dim 300 --head-num 6 --layer-num 6 \
  --dropout 0.5 \
  --ilm-objective \
  --ilm-input-embeddings \
  --ilm-output-heads \
  --output-file experiments/evaluation/results/c_full_6m_seed13.test_metrics.json
```

**Observed full seed-13 pilot.** `C-Full-6M` contains 6,631,992 parameters and
reached step 6,000 after recording 14,198,586 source bytes observed during
training. Its exact held-out result is `2.123232` BPB over 109,477.67 scored
UTF-8 bytes. The report is
`experiments/evaluation/results/c_full_6m_seed13.test_metrics.json`.

</details>

### Seed 29

<details>
<summary>Completed: character GPT, C-Flat-6M, and C-Full-6M</summary>

#### Character GPT

```bash
cd baselines/nanoGPT
../../venv/bin/python train.py \
  ../../experiments/evaluation/configs/nanogpt_char_6m.py \
  --seed=29 \
  --out_dir=../../experiments/evaluation/runs/char_gpt_6m_seed29
cd ../..
```

```bash
python experiments/evaluate_nanogpt_char.py \
  --checkpoint-path experiments/evaluation/runs/char_gpt_6m_seed29/ckpt.pt \
  --data-dir baselines/nanoGPT/data/ilm_tinyshakespeare \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --output-file experiments/evaluation/results/char_gpt_6m_seed29.test_metrics.json
```

#### C-Flat-6M

```bash
python sandbox/sandbox.py create models/evaluation/c_flat_6m_seed29.pth \
  --seed 29 \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --train-text experiments/evaluation/splits/tinyshakespeare/train.txt \
  --validation-text experiments/evaluation/splits/tinyshakespeare/validation.txt \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --oov-policy error \
  --word-block-size 20 --block-size 60 \
  --embedding-dim 300 --head-num 6 --layer-num 6 --batch-size 32 \
  --dropout 0.5 --epoch-num 6000 --lr 0.001 \
  --validation-interval 200
```

```bash
python experiments/evaluate_ilm.py \
  --model-path models/evaluation/c_flat_6m_seed29.pth \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --seed 29 --oov-policy error --evaluation-batch-size 32 \
  --word-block-size 20 --block-size 60 \
  --embedding-dim 300 --head-num 6 --layer-num 6 --dropout 0.5 \
  --output-file experiments/evaluation/results/c_flat_6m_seed29.test_metrics.json
```

#### C-Full-6M

```bash
python sandbox/sandbox.py create models/evaluation/c_full_6m_seed29.pth \
  --seed 29 \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --train-text experiments/evaluation/splits/tinyshakespeare/train.txt \
  --validation-text experiments/evaluation/splits/tinyshakespeare/validation.txt \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --oov-policy error \
  --ilm-objective --ilm-input-embeddings --ilm-output-heads \
  --word-block-size 20 --block-size 60 \
  --embedding-dim 300 --head-num 6 --layer-num 6 --batch-size 32 \
  --dropout 0.5 --epoch-num 6000 --lr 0.001 \
  --validation-interval 200
```

```bash
python experiments/evaluate_ilm.py \
  --model-path models/evaluation/c_full_6m_seed29.pth \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --seed 29 --oov-policy error --evaluation-batch-size 32 \
  --word-block-size 20 --block-size 60 \
  --embedding-dim 300 --head-num 6 --layer-num 6 --dropout 0.5 \
  --ilm-objective --ilm-input-embeddings --ilm-output-heads \
  --output-file experiments/evaluation/results/c_full_6m_seed29.test_metrics.json
```

</details>

### Seed 47

<details>
<summary>Completed: character GPT, C-Flat-6M, and C-Full-6M</summary>

#### Character GPT

```bash
cd baselines/nanoGPT
../../venv/bin/python train.py \
  ../../experiments/evaluation/configs/nanogpt_char_6m.py \
  --seed=47 \
  --out_dir=../../experiments/evaluation/runs/char_gpt_6m_seed47
cd ../..
```

```bash
python experiments/evaluate_nanogpt_char.py \
  --checkpoint-path experiments/evaluation/runs/char_gpt_6m_seed47/ckpt.pt \
  --data-dir baselines/nanoGPT/data/ilm_tinyshakespeare \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --output-file experiments/evaluation/results/char_gpt_6m_seed47.test_metrics.json
```

#### C-Flat-6M

```bash
python sandbox/sandbox.py create models/evaluation/c_flat_6m_seed47.pth \
  --seed 47 \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --train-text experiments/evaluation/splits/tinyshakespeare/train.txt \
  --validation-text experiments/evaluation/splits/tinyshakespeare/validation.txt \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --oov-policy error \
  --word-block-size 20 --block-size 60 \
  --embedding-dim 300 --head-num 6 --layer-num 6 --batch-size 32 \
  --dropout 0.5 --epoch-num 6000 --lr 0.001 \
  --validation-interval 200
```

```bash
python experiments/evaluate_ilm.py \
  --model-path models/evaluation/c_flat_6m_seed47.pth \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --seed 47 --oov-policy error --evaluation-batch-size 32 \
  --word-block-size 20 --block-size 60 \
  --embedding-dim 300 --head-num 6 --layer-num 6 --dropout 0.5 \
  --output-file experiments/evaluation/results/c_flat_6m_seed47.test_metrics.json
```

#### C-Full-6M

```bash
python sandbox/sandbox.py create models/evaluation/c_full_6m_seed47.pth \
  --seed 47 \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --train-text experiments/evaluation/splits/tinyshakespeare/train.txt \
  --validation-text experiments/evaluation/splits/tinyshakespeare/validation.txt \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --oov-policy error \
  --ilm-objective --ilm-input-embeddings --ilm-output-heads \
  --word-block-size 20 --block-size 60 \
  --embedding-dim 300 --head-num 6 --layer-num 6 --batch-size 32 \
  --dropout 0.5 --epoch-num 6000 --lr 0.001 \
  --validation-interval 200
```

```bash
python experiments/evaluate_ilm.py \
  --model-path models/evaluation/c_full_6m_seed47.pth \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --seed 47 --oov-policy error --evaluation-batch-size 32 \
  --word-block-size 20 --block-size 60 \
  --embedding-dim 300 --head-num 6 --layer-num 6 --dropout 0.5 \
  --ilm-objective --ilm-input-embeddings --ilm-output-heads \
  --output-file experiments/evaluation/results/c_full_6m_seed47.test_metrics.json
```

</details>

## Atomic Lexical Control

The atomic control uses the same frozen lexical boundaries as ILM, but assigns
each lexical type one ordinary vocabulary ID. It has a standard learned input
embedding table and standard learned lexical output projection. It therefore
tests lexical atomization without coordinate factorization or any ILM option.

The frozen `semantic_d10.json` mapping contains 15,030 lexical entries. With
six layers, six heads, width 156, and a 20-word context, the atomic model has
6,469,374 parameters. This is within 1.4% of `C-Flat-6M` while preserving a
conventional atomic vocabulary interface. BPB uses the same held-out corpus and
distributes the NLL of a predicted lexical token over that token's UTF-8 bytes.

Run the following commands for `seed=13`, then repeat them with `seed=29` and
`seed=47`. `--no-interactive` skips the post-training chat UI.

```bash
seed=13

python sandbox/sandbox.py create models/evaluation/atomic_lexical_6m_seed$seed.pth \
  --atomic-lexical \
  --seed $seed \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --train-text experiments/evaluation/splits/tinyshakespeare/train.txt \
  --validation-text experiments/evaluation/splits/tinyshakespeare/validation.txt \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --oov-policy error \
  --syllable-num 1 --word-block-size 20 --block-size 20 \
  --embedding-dim 156 --head-num 6 --layer-num 6 --batch-size 32 \
  --dropout 0.5 --epoch-num 6000 --lr 0.001 \
  --validation-interval 200 \
  --no-interactive

python experiments/evaluate_ilm.py \
  --model-path models/evaluation/atomic_lexical_6m_seed$seed.pth \
  --atomic-lexical \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --seed $seed --oov-policy error \
  --evaluation-batch-size 32 \
  --syllable-num 1 --word-block-size 20 --block-size 20 \
  --embedding-dim 156 --head-num 6 --layer-num 6 --dropout 0.5 \
  --output-file experiments/evaluation/results/atomic_lexical_6m_seed$seed.test_metrics.json
```

### Backbone-Width-Matched Atomic Control

This secondary control keeps the same atomic lexical vocabulary and 20-word
context, but increases the hidden width from 156 to 300. It therefore matches
the Transformer width of `C-Flat-6M` and `C-Full-6M`, while retaining the
larger atomic input and output tables. The resulting model has 15,537,630
trainable parameters. It is not a parameter-matched comparison and should be
reported separately from `Atomic-Lexical-6M`.

Run the following commands for `seed=13`, then repeat with `seed=29` and
`seed=47`.

```bash
seed=13

python sandbox/sandbox.py create models/evaluation/atomic_lexical_15m_seed$seed.pth \
  --atomic-lexical \
  --seed $seed \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --train-text experiments/evaluation/splits/tinyshakespeare/train.txt \
  --validation-text experiments/evaluation/splits/tinyshakespeare/validation.txt \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --oov-policy error \
  --syllable-num 1 --word-block-size 20 --block-size 20 \
  --embedding-dim 300 --head-num 6 --layer-num 6 --batch-size 32 \
  --dropout 0.5 --epoch-num 6000 --lr 0.001 \
  --validation-interval 200 \
  --no-interactive

python experiments/evaluate_ilm.py \
  --model-path models/evaluation/atomic_lexical_15m_seed$seed.pth \
  --atomic-lexical \
  --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --seed $seed --oov-policy error \
  --require-lossless-encoding \
  --evaluation-batch-size 32 \
  --syllable-num 1 --word-block-size 20 --block-size 20 \
  --embedding-dim 300 --head-num 6 --layer-num 6 --dropout 0.5 \
  --output-file experiments/evaluation/results/atomic_lexical_15m_seed$seed.test_metrics.json
```

## 15.5M Scaling Tier

The second Tiny Shakespeare tier asks whether the relative behavior of the
representation families persists at a larger shared parameter scale. It uses
six layers, six heads, the same dropout, batch size, context settings, and
6,000-step horizon as the 6M tier. Width `462` yields 15,438,192 total
trainable parameters for character nanoGPT, 15,483,532 for `C-Flat`, and
15,601,932 for `C-Full`.
`Atomic-Lexical-15M` uses width `300` and has 15,537,630 parameters.

Run all commands for model seeds `13`, `29`, and `47`. The models are within
about 1.1% of the Atomic parameter count. The character tokenizer is an ASCII
character tokenizer on Tiny Shakespeare, so it is also a byte-level baseline
for this corpus.

### Character/Byte GPT-15M

```bash
for seed in 13 29 47; do
  (
    cd baselines/nanoGPT
    ../../venv/bin/python train.py \
      ../../experiments/evaluation/configs/nanogpt_char_15m.py \
      --seed=$seed \
      --out_dir=../../experiments/evaluation/runs/char_gpt_15m_seed$seed
  )

  python experiments/evaluate_nanogpt_char.py \
    --checkpoint-path experiments/evaluation/runs/char_gpt_15m_seed$seed/ckpt.pt \
    --data-dir baselines/nanoGPT/data/ilm_tinyshakespeare \
    --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
    --output-file experiments/evaluation/results/char_gpt_15m_seed$seed.test_metrics.json
done
```

### C-Flat-15M And C-Full-15M

```bash
for seed in 13 29 47; do
  python sandbox/sandbox.py create models/evaluation/c_flat_15m_seed$seed.pth \
    --seed $seed \
    --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
    --train-text experiments/evaluation/splits/tinyshakespeare/train.txt \
    --validation-text experiments/evaluation/splits/tinyshakespeare/validation.txt \
    --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
    --oov-policy error \
    --word-block-size 20 --block-size 60 \
    --embedding-dim 462 --head-num 6 --layer-num 6 --batch-size 32 \
    --dropout 0.5 --epoch-num 6000 --lr 0.001 \
    --validation-interval 200 \
    --no-interactive

  python experiments/evaluate_ilm.py \
    --model-path models/evaluation/c_flat_15m_seed$seed.pth \
    --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
    --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
    --seed $seed --oov-policy error --require-lossless-encoding \
    --evaluation-batch-size 32 \
    --word-block-size 20 --block-size 60 \
    --embedding-dim 462 --head-num 6 --layer-num 6 --dropout 0.5 \
    --output-file experiments/evaluation/results/c_flat_15m_seed$seed.test_metrics.json

  python sandbox/sandbox.py create models/evaluation/c_full_15m_seed$seed.pth \
    --seed $seed \
    --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
    --train-text experiments/evaluation/splits/tinyshakespeare/train.txt \
    --validation-text experiments/evaluation/splits/tinyshakespeare/validation.txt \
    --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
    --oov-policy error \
    --ilm-objective --ilm-input-embeddings --ilm-output-heads \
    --word-block-size 20 --block-size 60 \
    --embedding-dim 462 --head-num 6 --layer-num 6 --batch-size 32 \
    --dropout 0.5 --epoch-num 6000 --lr 0.001 \
    --validation-interval 200 \
    --no-interactive

  python experiments/evaluate_ilm.py \
    --model-path models/evaluation/c_full_15m_seed$seed.pth \
    --tokenizer-json experiments/evaluation/tokenizers/semantic_d10.json \
    --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
    --seed $seed --oov-policy error --require-lossless-encoding \
    --evaluation-batch-size 32 \
    --word-block-size 20 --block-size 60 \
    --embedding-dim 462 --head-num 6 --layer-num 6 --dropout 0.5 \
    --ilm-objective --ilm-input-embeddings --ilm-output-heads \
    --output-file experiments/evaluation/results/c_full_15m_seed$seed.test_metrics.json
done
```

`Atomic-Lexical-15M` is already specified above. Run the fixed code-permutation
control at 6M before interpreting a Flat-versus-Full scaling difference as a
semantic-tokenizer effect. A 15.5M permutation control is a useful subsequent
ablation, but not required to establish the primary scaling tier.

## Character nanoGPT Baseline

The character baseline uses Karpathy's nanoGPT training implementation, pinned
locally at commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`. It does **not**
use nanoGPT's downloaded Shakespeare split. `prepare_nanogpt_char.py` writes
the project’s fixed train, validation, and test files as the expected binary
format and constructs a 65-character vocabulary from the complete corpus.

```bash
git clone https://github.com/karpathy/nanoGPT.git baselines/nanoGPT
git -C baselines/nanoGPT checkout 3adf61e154c3fe3fca428ad6bc3818b27a3b8291
git -C baselines/nanoGPT apply \
  ../../experiments/evaluation/patches/nanogpt_controlled_baseline.patch

python experiments/prepare_nanogpt_char.py \
  --corpus-file data/corpora/training_old_english.txt \
  --train-text experiments/evaluation/splits/tinyshakespeare/train.txt \
  --validation-text experiments/evaluation/splits/tinyshakespeare/validation.txt \
  --test-text experiments/evaluation/splits/tinyshakespeare/test.txt \
  --output-dir baselines/nanoGPT/data/ilm_tinyshakespeare
```

`experiments/evaluation/configs/nanogpt_char_6m.py` defines a 6.50M-parameter
baseline with six layers, width 300, six attention heads, 74-character context,
batch size 32, dropout 0.5, and seed 13. The repository patch adds the
configurable seed used by the controlled runs.

For a later seed `N`, override both the random seed and output directory while
using the same configuration file:

```bash
cd baselines/nanoGPT

../../venv/bin/python train.py \
  ../../experiments/evaluation/configs/nanogpt_char_6m.py \
  --seed=N \
  --out_dir=../../experiments/evaluation/runs/char_gpt_6m_seedN

cd ../..
```

Then evaluate `char_gpt_6m_seedN/ckpt.pt` into
`experiments/evaluation/results/char_gpt_6m_seedN.test_metrics.json` with the
same evaluation command recorded under seed 13.

The 100-step smoke test confirms data loading and configuration but does not
write a checkpoint because evaluation happens at step 200.

```bash
cd baselines/nanoGPT

../../venv/bin/python train.py \
  ../../experiments/evaluation/configs/nanogpt_char_6m.py \
  --max_iters=100

cd ../..
```

Generate fixed-prompt qualitative samples with an explicit generation seed:

```bash
python experiments/sample_nanogpt_char.py \
  --checkpoint-path experiments/evaluation/runs/char_gpt_6m_seed13/ckpt.pt \
  --data-dir baselines/nanoGPT/data/ilm_tinyshakespeare \
  --prompt "The queen" \
  --max-new-characters 300 \
  --temperature 1 \
  --top-k 3 \
  --generation-seed 13 \
  --output-file experiments/evaluation/results/char_gpt_6m_seed13.the_queen.json
```

## enwik8 Replication

The second-corpus replication uses enwik8, a 100M-byte raw Wikipedia/XML
benchmark. Its conventional split is the first 90M bytes for training, the
next 5M for validation, and the final 5M for test. The registered models are
a 6M byte-level GPT, C-Flat-6M, and C-Full-6M run with seeds
`13`, `29`, and `47`.

Download the canonical archive from the Hutter Prize data host, extract it,
and verify its published size and checksums before creating any splits:

```bash
mkdir -p data/corpora

curl --fail --location \
  --output data/corpora/enwik8.zip \
  https://mattmahoney.net/dc/enwik8.zip

unzip -t data/corpora/enwik8.zip
unzip -p data/corpora/enwik8.zip enwik8 > data/corpora/enwik8

wc -c < data/corpora/enwik8
md5 -q data/corpora/enwik8
shasum -a 1 data/corpora/enwik8
```

The expected values are `100000000` bytes, MD5
`a1fa5ffddb56f4953e226637dabbb36a`, and SHA-1
`57b8363b814821dc9d47aa4d41f58733519076b2`. The source is UTF-8 and consists
primarily of English text, but it also retains XML markup, URLs, tables,
names, and non-English characters.

Create the conventional split with the dedicated helper. Do not use
`prepare_text_split.py` here because it intentionally moves boundaries to
whitespace, whereas enwik8 uses fixed byte boundaries.

```bash
python experiments/prepare_enwik8_split.py \
  --source-file data/corpora/enwik8 \
  --output-dir experiments/evaluation/splits/enwik8
```

The three-coordinate semantic tokenizer is not feasible on this corpus. The
lossless tokenization count is 519,955 unique types, exceeding the `64^3 =
262144` available codes. The recorded result is
`experiments/evaluation/tokenizers/enwik8_lossless_capacity.json`. Do not truncate the
vocabulary or use a fallback code to force a depth-three primary result.

The enwik8 replication therefore uses four coordinates per lexical token,
labelled `s4` in every artifact name. `64^4 = 16777216` codes cover the same
519,955 token types without altering the corpus. The recorded capacity report
is `experiments/evaluation/tokenizers/enwik8_lossless_capacity_depth4.json`.

Confirm the depth-four capacity before making embedding API calls.

```bash
python experiments/check_tokenizer_capacity.py \
  --source-file data/corpora/enwik8 \
  --base 64 \
  --depth 4 \
  --lossless-tokenization \
  --output-file experiments/evaluation/tokenizers/enwik8_lossless_capacity_depth4.json
```

If it fits, build and freeze the semantic tokenizer from the complete fixed
corpus. This follows the project’s closed-corpus tokenizer protocol.

```bash
python tests/quickstart.py \
  --mode build \
  --method embedding-cluster \
  --source-file data/corpora/enwik8 \
  --target-file experiments/evaluation/tokenizers/enwik8_lossless_semantic_d10_s4.json \
  --cluster-method spherical-kmeans \
  --reduced-dim 10 \
  --depth 4 \
  --embedding-batch-size 512 \
  --cache-file data/cache/evaluation_enwik8_semantic_d10_s4.embeddings.npz \
  --lossless-tokenization \
  --random-state 13
```

The existing embedding cache is extended in place. It reuses lexical-token
embeddings and requests vectors only for the newly introduced fallback character
tokens. The new tokenizer name prevents confusion with the earlier non-lossless
artifact.

Create frozen-tokenizer diagnostics after the build. The following command
writes an LLM-labelled semantic sidecar and sampled two- and
three-dimensional PCA figures. These labels are optional interpretation
artifacts and do not participate in tokenizer construction or model training.
The full embedding cache is large, so keep the displayed point sample at
20,000.

```bash
python experiments/tokenizer_diagnostics.py \
  --tokenizer-json experiments/evaluation/tokenizers/enwik8_lossless_semantic_d10_s4.json \
  --cache-file data/cache/evaluation_enwik8_semantic_d10_s4.embeddings.npz \
  --semantic-spelling-file \
    experiments/evaluation/tokenizers/enwik8_lossless_semantic_d10_s4.semantic.json \
  --centroid-label-method llm \
  --centroid-label-model gpt-4.1-mini \
  --random-state 13 \
  --plot-pca-2d \
  --plot-pca-3d \
  --plot-clusters \
  --plot-sample-size 20000 \
  --plot-output-dir experiments/evaluation/tokenizers/figures/enwik8_lossless_s4 \
  --no-show-plots
```

LLM centroid labels are optional interpretation artifacts and are not part of
the primary experiment. With depth four, the implementation makes four grouped
calls, one for each set of 64 centroids, rather than 256 separate calls. Add
the following options to the diagnostic command only when those labels are
useful for inspection:

```text
--centroid-label-method llm
--centroid-label-model <chosen-model>
--centroid-label-examples 20
```

Check strict coverage and export the corresponding character-level data for
nanoGPT. These paths are enwik8-specific and do not touch the Shakespeare
artifacts. `--require-lossless-encoding` is stricter than an OOV check. It
also requires every raw UTF-8 byte to be represented by the ILM token stream.
This check must pass before an ILM enwik8 result can be compared with the
byte-level baseline.

```bash
python experiments/check_tokenizer_coverage.py \
  --tokenizer-json experiments/evaluation/tokenizers/enwik8_lossless_semantic_d10_s4.json \
  --text experiments/evaluation/splits/enwik8/train.txt \
  --text experiments/evaluation/splits/enwik8/validation.txt \
  --text experiments/evaluation/splits/enwik8/test.txt \
  --oov-policy error \
  --require-lossless-encoding \
  --output-file experiments/evaluation/results/enwik8_lossless_tokenizer_coverage.json

python experiments/prepare_nanogpt_char.py \
  --corpus-file data/corpora/enwik8 \
  --train-text experiments/evaluation/splits/enwik8/train.txt \
  --validation-text experiments/evaluation/splits/enwik8/validation.txt \
  --test-text experiments/evaluation/splits/enwik8/test.txt \
  --unit utf8-byte \
  --output-dir baselines/nanoGPT/data/ilm_enwik8_byte
```

The `utf8-byte` setting is deliberate. enwik8 is conventionally evaluated as
a byte-level benchmark and the frozen corpus contains 205 observed byte values.
Treating the same UTF-8 file as Unicode characters would create 6,064 code
points, changing the tied input/output vocabulary matrix and producing an
unmatched 8.30M-parameter nanoGPT. The byte configuration has 6,567,600 total
parameters, including position embeddings, versus 6,561,064 for the enwik8
`s4` flat ILM control. nanoGPT's startup line excludes position embeddings, so
it displays this configuration as `6.55M`.

For each seed, train and evaluate the byte-level baseline, flat coordinate
control, and full ILM model in that order. Every ILM output path begins with
`enwik8_lossless_s4_` to distinguish this four-coordinate corpus replication from the
three-coordinate Tiny Shakespeare experiments. The enwik8 test set contains
five million bytes, so all three models use `block-reset` teacher-forced BPB:
each non-overlapping context window is scored in one forward pass. Do not
compare these values numerically with the earlier Tiny Shakespeare
`full-context` reports.

```bash
for seed in 13 29 47; do
  cd baselines/nanoGPT
  ../../venv/bin/python train.py \
    ../../experiments/evaluation/configs/nanogpt_enwik8_byte_6m.py \
    --seed=$seed \
    --out_dir=../../experiments/evaluation/runs/enwik8_byte_gpt_6m_seed$seed
  cd ../..

  python experiments/evaluate_nanogpt_char.py \
    --checkpoint-path experiments/evaluation/runs/enwik8_byte_gpt_6m_seed$seed/ckpt.pt \
    --data-dir baselines/nanoGPT/data/ilm_enwik8_byte \
    --test-text experiments/evaluation/splits/enwik8/test.txt \
    --evaluation-mode block-reset \
    --evaluation-batch-size 128 \
    --output-file experiments/evaluation/results/enwik8_byte_gpt_6m_seed$seed.test_metrics.json

  python sandbox/sandbox.py create models/evaluation/enwik8_lossless_s4_c_flat_6m_seed$seed.pth \
    --seed $seed \
    --tokenizer-json experiments/evaluation/tokenizers/enwik8_lossless_semantic_d10_s4.json \
    --train-text experiments/evaluation/splits/enwik8/train.txt \
    --validation-text experiments/evaluation/splits/enwik8/validation.txt \
    --test-text experiments/evaluation/splits/enwik8/test.txt \
    --oov-policy error \
    --syllable-num 4 --word-block-size 20 --block-size 80 \
    --embedding-dim 300 --head-num 6 --layer-num 6 --batch-size 32 \
    --dropout 0.5 --epoch-num 6000 --lr 0.001 \
    --validation-interval 200

  python experiments/evaluate_ilm.py \
    --model-path models/evaluation/enwik8_lossless_s4_c_flat_6m_seed$seed.pth \
    --tokenizer-json experiments/evaluation/tokenizers/enwik8_lossless_semantic_d10_s4.json \
    --test-text experiments/evaluation/splits/enwik8/test.txt \
    --seed $seed --oov-policy error \
    --require-lossless-encoding \
    --evaluation-mode block-reset --evaluation-batch-size 128 \
    --syllable-num 4 --word-block-size 20 --block-size 80 \
    --embedding-dim 300 --head-num 6 --layer-num 6 --dropout 0.5 \
    --output-file experiments/evaluation/results/enwik8_lossless_s4_c_flat_6m_seed$seed.test_metrics.json

  python sandbox/sandbox.py create models/evaluation/enwik8_lossless_s4_c_full_6m_seed$seed.pth \
    --seed $seed \
    --tokenizer-json experiments/evaluation/tokenizers/enwik8_lossless_semantic_d10_s4.json \
    --train-text experiments/evaluation/splits/enwik8/train.txt \
    --validation-text experiments/evaluation/splits/enwik8/validation.txt \
    --test-text experiments/evaluation/splits/enwik8/test.txt \
    --oov-policy error \
    --ilm-objective --ilm-input-embeddings --ilm-output-heads \
    --syllable-num 4 --word-block-size 20 --block-size 80 \
    --embedding-dim 300 --head-num 6 --layer-num 6 --batch-size 32 \
    --dropout 0.5 --epoch-num 6000 --lr 0.001 \
    --validation-interval 200

  python experiments/evaluate_ilm.py \
    --model-path models/evaluation/enwik8_lossless_s4_c_full_6m_seed$seed.pth \
    --tokenizer-json experiments/evaluation/tokenizers/enwik8_lossless_semantic_d10_s4.json \
    --test-text experiments/evaluation/splits/enwik8/test.txt \
    --seed $seed --oov-policy error \
    --require-lossless-encoding \
    --evaluation-mode block-reset --evaluation-batch-size 128 \
    --syllable-num 4 --word-block-size 20 --block-size 80 \
    --embedding-dim 300 --head-num 6 --layer-num 6 --dropout 0.5 \
    --ilm-objective --ilm-input-embeddings --ilm-output-heads \
    --output-file experiments/evaluation/results/enwik8_lossless_s4_c_full_6m_seed$seed.test_metrics.json
done
```

### Fixed Code-Permutation Control

`Random-Flat-6M` preserves enwik8's lossless lexical segmentation and the exact final set of occupied
four-coordinate codes. One fixed permutation changes only the assignment from
lexical entries to those codes. The permutation seed is independent of the
three model seeds and must remain fixed for the full ablation. The shared
permutation-index mapping appears in the Tiny Shakespeare permutation-control
section above. The same index and permutation seed are used for both corpora.

```bash
python experiments/create_permuted_tokenizer.py \
  --source-tokenizer experiments/evaluation/tokenizers/enwik8_lossless_semantic_d10_s4.json \
  --target-tokenizer experiments/evaluation/tokenizers/enwik8_lossless_permuted_codes_s4_seed314159.json \
  --permutation-seed 314159
```

```bash
for seed in 13 29 47; do
  python sandbox/sandbox.py create models/evaluation/enwik8_lossless_s4_random_flat_6m_seed$seed.pth \
    --seed $seed \
    --tokenizer-json experiments/evaluation/tokenizers/enwik8_lossless_permuted_codes_s4_seed314159.json \
    --train-text experiments/evaluation/splits/enwik8/train.txt \
    --validation-text experiments/evaluation/splits/enwik8/validation.txt \
    --test-text experiments/evaluation/splits/enwik8/test.txt \
    --oov-policy error \
    --syllable-num 4 --word-block-size 20 --block-size 80 \
    --embedding-dim 300 --head-num 6 --layer-num 6 --batch-size 32 \
    --dropout 0.5 --epoch-num 6000 --lr 0.001 \
    --validation-interval 200 \
    --no-interactive

  python experiments/evaluate_ilm.py \
    --model-path models/evaluation/enwik8_lossless_s4_random_flat_6m_seed$seed.pth \
    --tokenizer-json experiments/evaluation/tokenizers/enwik8_lossless_permuted_codes_s4_seed314159.json \
    --test-text experiments/evaluation/splits/enwik8/test.txt \
    --seed $seed --oov-policy error \
    --require-lossless-encoding \
    --evaluation-mode block-reset --evaluation-batch-size 128 \
    --syllable-num 4 --word-block-size 20 --block-size 80 \
    --embedding-dim 300 --head-num 6 --layer-num 6 --dropout 0.5 \
    --output-file experiments/evaluation/results/enwik8_lossless_s4_random_flat_6m_seed$seed.test_metrics.json
done
```

### Additional permutation replicates

Run the same two representation replicates on enwik8. Together with the Tiny
Shakespeare commands above, this produces twelve new train-and-evaluate runs:
two permutation indices, two corpora, and three model seeds. All twelve runs
are completed. The original index-`1` reports remain unchanged.

```bash
for index in 2 3; do
  case $index in
    2) permutation_seed=271828 ;;
    3) permutation_seed=161803 ;;
  esac
  tokenizer=experiments/evaluation/tokenizers/enwik8_lossless_permuted_codes_s4_seed$permutation_seed.json

  python experiments/create_permuted_tokenizer.py \
    --source-tokenizer experiments/evaluation/tokenizers/enwik8_lossless_semantic_d10_s4.json \
    --target-tokenizer $tokenizer \
    --permutation-seed $permutation_seed

  for seed in 13 29 47; do
    python sandbox/sandbox.py create models/evaluation/enwik8_lossless_s4_random_${index}_flat_6m_seed$seed.pth \
      --seed $seed \
      --tokenizer-json $tokenizer \
      --train-text experiments/evaluation/splits/enwik8/train.txt \
      --validation-text experiments/evaluation/splits/enwik8/validation.txt \
      --test-text experiments/evaluation/splits/enwik8/test.txt \
      --oov-policy error \
      --syllable-num 4 --word-block-size 20 --block-size 80 \
      --embedding-dim 300 --head-num 6 --layer-num 6 --batch-size 32 \
      --dropout 0.5 --epoch-num 6000 --lr 0.001 \
      --validation-interval 500 --no-interactive

    python experiments/evaluate_ilm.py \
      --model-path models/evaluation/enwik8_lossless_s4_random_${index}_flat_6m_seed$seed.pth \
      --tokenizer-json $tokenizer \
      --test-text experiments/evaluation/splits/enwik8/test.txt \
      --seed $seed --oov-policy error --require-lossless-encoding \
      --evaluation-mode block-reset --evaluation-batch-size 128 \
      --syllable-num 4 --word-block-size 20 --block-size 80 \
      --embedding-dim 300 --head-num 6 --layer-num 6 --dropout 0.5 \
      --output-file experiments/evaluation/results/enwik8_lossless_s4_random_${index}_flat_6m_seed$seed.test_metrics.json
  done
done
```

### 15.5M Scaling Tier

This tier repeats the enwik8 ByteGPT, Flat, and Full comparison at width `462`, while preserving six
layers, six heads, batch size 32, the 6,000-step horizon, and the existing
context settings. Total trainable parameter counts are 15,502,872 for ByteGPT,
15,492,772 for Flat, and 15,670,372 for Full. These counts are within 1.2% of
one another. nanoGPT's startup count omits position embeddings, so its printed
number will be slightly smaller than the total reported here.

All three families use the existing canonical enwik8 split. Evaluate with
`block-reset` BPB, as in the 6M tier.

```bash
for seed in 13 29 47; do
  (
    cd baselines/nanoGPT
    ../../venv/bin/python train.py \
      ../../experiments/evaluation/configs/nanogpt_enwik8_byte_15m.py \
      --seed=$seed \
      --out_dir=../../experiments/evaluation/runs/enwik8_byte_gpt_15m_seed$seed
  )

  python experiments/evaluate_nanogpt_char.py \
    --checkpoint-path experiments/evaluation/runs/enwik8_byte_gpt_15m_seed$seed/ckpt.pt \
    --data-dir baselines/nanoGPT/data/ilm_enwik8_byte \
    --test-text experiments/evaluation/splits/enwik8/test.txt \
    --evaluation-mode block-reset --evaluation-batch-size 128 \
    --output-file experiments/evaluation/results/enwik8_byte_gpt_15m_seed$seed.test_metrics.json

  python sandbox/sandbox.py create models/evaluation/enwik8_lossless_s4_c_flat_15m_seed$seed.pth \
    --seed $seed \
    --tokenizer-json experiments/evaluation/tokenizers/enwik8_lossless_semantic_d10_s4.json \
    --train-text experiments/evaluation/splits/enwik8/train.txt \
    --validation-text experiments/evaluation/splits/enwik8/validation.txt \
    --test-text experiments/evaluation/splits/enwik8/test.txt \
    --oov-policy error \
    --syllable-num 4 --word-block-size 20 --block-size 80 \
    --embedding-dim 462 --head-num 6 --layer-num 6 --batch-size 32 \
    --dropout 0.5 --epoch-num 6000 --lr 0.001 \
    --validation-interval 200 \
    --no-interactive

  python experiments/evaluate_ilm.py \
    --model-path models/evaluation/enwik8_lossless_s4_c_flat_15m_seed$seed.pth \
    --tokenizer-json experiments/evaluation/tokenizers/enwik8_lossless_semantic_d10_s4.json \
    --test-text experiments/evaluation/splits/enwik8/test.txt \
    --seed $seed --oov-policy error --require-lossless-encoding \
    --evaluation-mode block-reset --evaluation-batch-size 128 \
    --syllable-num 4 --word-block-size 20 --block-size 80 \
    --embedding-dim 462 --head-num 6 --layer-num 6 --dropout 0.5 \
    --output-file experiments/evaluation/results/enwik8_lossless_s4_c_flat_15m_seed$seed.test_metrics.json

  python sandbox/sandbox.py create models/evaluation/enwik8_lossless_s4_c_full_15m_seed$seed.pth \
    --seed $seed \
    --tokenizer-json experiments/evaluation/tokenizers/enwik8_lossless_semantic_d10_s4.json \
    --train-text experiments/evaluation/splits/enwik8/train.txt \
    --validation-text experiments/evaluation/splits/enwik8/validation.txt \
    --test-text experiments/evaluation/splits/enwik8/test.txt \
    --oov-policy error \
    --ilm-objective --ilm-input-embeddings --ilm-output-heads \
    --syllable-num 4 --word-block-size 20 --block-size 80 \
    --embedding-dim 462 --head-num 6 --layer-num 6 --batch-size 32 \
    --dropout 0.5 --epoch-num 6000 --lr 0.001 \
    --validation-interval 200 \
    --no-interactive

  python experiments/evaluate_ilm.py \
    --model-path models/evaluation/enwik8_lossless_s4_c_full_15m_seed$seed.pth \
    --tokenizer-json experiments/evaluation/tokenizers/enwik8_lossless_semantic_d10_s4.json \
    --test-text experiments/evaluation/splits/enwik8/test.txt \
    --seed $seed --oov-policy error --require-lossless-encoding \
    --evaluation-mode block-reset --evaluation-batch-size 128 \
    --syllable-num 4 --word-block-size 20 --block-size 80 \
    --embedding-dim 462 --head-num 6 --layer-num 6 --dropout 0.5 \
    --ilm-objective --ilm-input-embeddings --ilm-output-heads \
    --output-file experiments/evaluation/results/enwik8_lossless_s4_c_full_15m_seed$seed.test_metrics.json
done
```

The character baseline remains directly applicable because enwik8 is a
byte-level benchmark. The semantic ILM result uses four coordinate roles and
must be reported separately from the three-coordinate Tiny Shakespeare result.

## Result Discipline

Use validation data for horizon and decoding decisions. Keep test likelihood
and fixed-prompt generation for final evaluation. Hugging Face Shakespeare
checkpoints such as fawern and Esmaelmoat are qualitative references only
because their pretraining, fine-tuning data, and parameter budgets are not
controlled here.
