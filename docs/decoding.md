# Decoding

ILM generates one coordinate at a time. A depth-three lexical code therefore
requires three coordinate predictions before the detokenizer can recover one
lexical entry. The model forward path remains the ordinary causal Transformer
path used by Flat ILM and Full ILM. Coordinate-aware decoding changes sampling
settings, not model weights or attention.

## Scalar Sampling

`--temperature` scales logits before sampling. Lower values concentrate
probability mass on the highest-scoring coordinates. `--top-k` restricts
sampling to the highest-scoring coordinate values.

```bash
python sandbox/sandbox.py load models/MODEL.pth \
  --temperature 1 \
  --top-k 3 \
  --stream
```

Use `--generation-seed` when a completion needs to be reproducible:

```bash
--generation-seed 13
```

## Coordinate-Aware Sampling

Coordinate roles can use different temperatures and candidate-set sizes. The
number of values must equal `--syllable-num`.

```bash
python sandbox/sandbox.py load models/MODEL.pth \
  --top-k-by-coordinate 3,4,6 \
  --temperature-by-coordinate 1,0.95,0.8 \
  --stream
```

When either coordinate-specific option is present, it overrides its scalar
counterpart. For the example above, scalar `--top-k` and `--temperature` are
shown as overridden in the inference configuration table.

The intended interpretation is hierarchical: early coordinates select broad
regions, while later coordinates select increasingly local alternatives. A
conservative starting point for a depth-three tokenizer is `3,4,6` for top-k
and `1,0.95,0.8` for temperature. These are exploratory generation settings,
not part of the teacher-forced BPB protocol.

To use scalar sampling after a configuration file or source default defines
coordinate-specific values, pass:

```bash
--top-k-by-coordinate none \
--temperature-by-coordinate none
```

## Streaming

`--stream` prints decoded lexical entries as each full coordinate code is
completed. It does not change the generated coordinate sequence or sampling
distribution. Omit it when collecting a single final completion or redirecting
output to a file.
