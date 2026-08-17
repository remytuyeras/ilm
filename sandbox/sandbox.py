"""
Model experiment sandbox.

This file is intentionally more like an experiment notebook than a polished
training CLI. It wires together:

- a tokenizer JSON
- a training text file
- an IntuinisticLanguageModel configuration
- create/improve/load checkpoint workflows
- the interactive `ilm.user_interface`

Dataset/tokenizer defaults:

    tokenizer_json = "data/tokenizers/tokenizer_embedding_cluster_v1.json"
    training_text = "data/corpora/training_old_english.txt"

Relative-position tokenizer path that may still be useful for comparison:

    tokenizer_json = "data/tokenizers/tokenizer_v2.json"

Older small-data defaults that may still be useful for comparison:

    tokenizer_json = "data/tokenizers/tokenizer_v1.json"
    training_text = "data/corpora/training_input.txt"

The knobs you usually change first are optimization/training parameters:

    dropout = 0.5
    epoch_num = 4000
    lr = 1e-3

These are cheaper to sweep and are less likely to break checkpoint loading.

Sampling parameters control the interactive output without changing checkpoint
compatibility:

    completed_words = 100
    syllable_num = 3
    temperature = 0.8
    top_k = 10
    top_k_by_coordinate = 3,5,8
    temperature_by_coordinate = 1,0.95,0.8
    stream = False

When coordinate-specific sampling is not `None`, it overrides the scalar
sampling value. For example, `top_k_by_coordinate = 3,4,6` overrides `top_k`
at generation time. Use `--top-k-by-coordinate none` or
`--temperature-by-coordinate none` to force the scalar value to be active.

Architecture parameters are usually changed more carefully:

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    vocab_size = 64
    syllable_num = 3
    word_block_size = 20
    block_size = syllable_num * word_block_size
    batch_size = 32
    embedding_dim = 80
    head_num = 4
    head_size = embedding_dim // head_num
    layer_num = 6
    ilm_input_embeddings = False
    ilm_output_heads = False
    ilm_objective = False

If you change architecture parameters, existing checkpoint files may no longer
load because the saved tensor shapes must match the model architecture.

`ilm_input_embeddings` changes the checkpoint architecture by assigning the
same coordinate value a different input embedding at each coordinate role.

`ilm_output_heads` changes the checkpoint architecture by using one output head
per predicted coordinate role.

`ilm_objective` keeps standard coordinate-time attention and changes only the
training loss. It excludes the incomplete word suffix at a sampled window's
left boundary. Generation does not use the prefix mask.

Current usage:

    python sandbox/sandbox.py create models/m2.v0.0.0.pth
    python sandbox/sandbox.py improve models/m2.v0.0.0.pth --patch
    python sandbox/sandbox.py load models/m2.v0.0.0.pth

The `create` command writes one JSON metadata file next to the first model
checkpoint. Later `improve` commands keep appending to that same JSON file so
the model's training curriculum stays together instead of splitting across one
metadata file per checkpoint.

For publication-style runs, provide `--train-text`, `--validation-text`, and
`--test-text` together. The sandbox encodes each file independently, so sample
windows cannot cross split boundaries. `--seed` resets Python, NumPy, and
PyTorch before model construction and training. `--generation-seed` resets the
same generators before every interactive completion. A tokenizer frozen from
the full corpus uses `--oov-policy error` to verify split coverage, while
`--oov-policy fallback` reports coverage on a newly introduced document.

For a fixed-corpus experiment, use `--oov-policy error` and expect no OOVs.
For a new document, `--oov-policy fallback` is a transfer-coverage probe: every
unknown word receives the same valid code, so the model can process the text
but cannot distinguish the unknown words from one another.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
sys.path.insert(1, "./")
import ilm
import torch

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_CYAN = "\033[36m"
ANSI_GREEN = "\033[32m"

DEFAULT_TOKENIZER_JSON = "data/tokenizers/tokenizer_embedding_cluster_v1.json"
DEFAULT_TRAINING_TEXT = "data/corpora/training_old_english.txt"

DEFAULT_SYLLABLE_NUM = 3
DEFAULT_WORD_BLOCK_SIZE = 25
DEFAULT_VOCAB_SIZE = 64
DEFAULT_BLOCK_SIZE = DEFAULT_SYLLABLE_NUM * DEFAULT_WORD_BLOCK_SIZE
DEFAULT_BATCH_SIZE = 32
DEFAULT_EMBEDDING_DIM = 600
DEFAULT_HEAD_NUM = 4
DEFAULT_HEAD_SIZE = DEFAULT_EMBEDDING_DIM // DEFAULT_HEAD_NUM
DEFAULT_LAYER_NUM = 8
DEFAULT_ILM_INPUT_EMBEDDINGS = False
DEFAULT_ILM_OUTPUT_HEADS = False
DEFAULT_ILM_OBJECTIVE = False
DEFAULT_ATOMIC_LEXICAL = False

DEFAULT_DROPOUT = 0.5
DEFAULT_EPOCH_NUM = 4000
DEFAULT_LR = 1e-3
DEFAULT_VALIDATION_INTERVAL = 500
DEFAULT_SEED = 42
DEFAULT_OPTIMIZER_PROFILE = "all-parameters"
DEFAULT_LR_SCHEDULE = "constant"
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999
DEFAULT_GRAD_CLIP = 0.0
DEFAULT_WARMUP_ITERS = 0
DEFAULT_LR_DECAY_ITERS = None
DEFAULT_MIN_LR = None

DEFAULT_COMPLETED_WORDS = 300
DEFAULT_GENERATION_SEED = None
DEFAULT_OOV_POLICY = "error"
DEFAULT_OOV_FALLBACK_CODE = None

'''
DEFAULT_TEMPERATURE_BY_COORDINATE and --temperature-by-coordinate
0.8 = cleaner, more repetitive
0.9 = likely best stability
1.0 = current balanced baseline
1.1 = more expressive, more grammar errors
'''
DEFAULT_TEMPERATURE = 1
DEFAULT_TEMPERATURE_BY_COORDINATE = None # (1,0.95,0.8) # None or tuple; e.g. (1,0.95,0.8)

'''
DEFAULT_TOP_K_BY_COORDINATE or --top-k-by-coordinate
3,5,8 = most varied, but noisier
3,4,6 = best balance
2,4,6 = cleanest, but flatter
'''
DEFAULT_TOP_K = 3
DEFAULT_TOP_K_BY_COORDINATE = None # (3,4,6) # None or tuple; e.g. (3,4,6)

DEFAULT_STREAM = True


# ==================================================================
# ==================================================================


def parse_int_sequence(value):
    if value.strip().lower() in {"none", "off"}:
        return None
    try:
        items = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers, like 3,5,8, or none") from exc
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return items


def parse_float_sequence(value):
    if value.strip().lower() in {"none", "off"}:
        return None
    try:
        items = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers, like 0.8,1.0,1.1, or none") from exc
    if not items:
        raise argparse.ArgumentTypeError("expected at least one number")
    if any(item < 0 for item in items):
        raise argparse.ArgumentTypeError("values must be zero or positive")
    return items


def add_shared_arguments(parser, use_defaults=True):
    def default(value):
        if use_defaults:
            return {"default": value}
        return {"default": argparse.SUPPRESS}

    parser.add_argument("--tokenizer-json", **default(DEFAULT_TOKENIZER_JSON))
    parser.add_argument("--training-text", **default(DEFAULT_TRAINING_TEXT))
    parser.add_argument("--train-text", **default(None))
    parser.add_argument("--validation-text", **default(None))
    parser.add_argument("--test-text", **default(None))
    parser.add_argument(
        "--oov-policy",
        choices=["error", "fallback"],
        help="Stop on unknown words, or replace each with one deterministic code for transfer coverage.",
        **default(DEFAULT_OOV_POLICY),
    )
    parser.add_argument(
        "--oov-fallback-code",
        help="Explicit coordinate code used by --oov-policy fallback; defaults to the smallest tokenizer code.",
        **default(DEFAULT_OOV_FALLBACK_CODE),
    )

    parser.add_argument("--vocab-size", type=int, **default(DEFAULT_VOCAB_SIZE))
    parser.add_argument("--block-size", type=int, **default(DEFAULT_BLOCK_SIZE))
    parser.add_argument("--word-block-size", type=int, **default(DEFAULT_WORD_BLOCK_SIZE))
    parser.add_argument("--batch-size", type=int, **default(DEFAULT_BATCH_SIZE))
    parser.add_argument("--embedding-dim", type=int, **default(DEFAULT_EMBEDDING_DIM))
    parser.add_argument("--head-num", type=int, **default(DEFAULT_HEAD_NUM))
    parser.add_argument("--layer-num", type=int, **default(DEFAULT_LAYER_NUM))
    parser.add_argument(
        "--atomic-lexical",
        action="store_true",
        default=DEFAULT_ATOMIC_LEXICAL if use_defaults else argparse.SUPPRESS,
        help="Use frozen lexical tokenizer entries as ordinary atomic vocabulary IDs.",
    )
    parser.add_argument(
        "--ilm-input-embeddings",
        action="store_true",
        default=DEFAULT_ILM_INPUT_EMBEDDINGS if use_defaults else argparse.SUPPRESS,
        help="Use coordinate-role-conditioned input embeddings.",
    )
    parser.add_argument(
        "--ilm-output-heads",
        action="store_true",
        default=DEFAULT_ILM_OUTPUT_HEADS if use_defaults else argparse.SUPPRESS,
        help="Use coordinate-role-conditioned output heads.",
    )
    parser.add_argument(
        "--ilm-objective",
        action="store_true",
        default=DEFAULT_ILM_OBJECTIVE if use_defaults else argparse.SUPPRESS,
        help="Train with the word-prefix loss that excludes left-boundary suffix fragments.",
    )
    parser.add_argument("--dropout", type=float, **default(DEFAULT_DROPOUT))
    parser.add_argument("--epoch-num", type=int, **default(DEFAULT_EPOCH_NUM))
    parser.add_argument("--lr", type=float, **default(DEFAULT_LR))
    parser.add_argument(
        "--optimizer-profile",
        choices=["all-parameters", "nanogpt"],
        **default(DEFAULT_OPTIMIZER_PROFILE),
        help="AdamW parameter-decay grouping. all-parameters preserves the historical ILM optimizer.",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine"],
        **default(DEFAULT_LR_SCHEDULE),
    )
    parser.add_argument("--weight-decay", type=float, **default(DEFAULT_WEIGHT_DECAY))
    parser.add_argument("--beta1", type=float, **default(DEFAULT_BETA1))
    parser.add_argument("--beta2", type=float, **default(DEFAULT_BETA2))
    parser.add_argument("--grad-clip", type=float, **default(DEFAULT_GRAD_CLIP))
    parser.add_argument("--warmup-iters", type=int, **default(DEFAULT_WARMUP_ITERS))
    parser.add_argument("--lr-decay-iters", type=int, **default(DEFAULT_LR_DECAY_ITERS))
    parser.add_argument("--min-lr", type=float, **default(DEFAULT_MIN_LR))
    parser.add_argument("--validation-interval", type=int, **default(DEFAULT_VALIDATION_INTERVAL))
    parser.add_argument("--seed", type=int, **default(DEFAULT_SEED))

    parser.add_argument("--completed-words", type=int, **default(DEFAULT_COMPLETED_WORDS))
    parser.add_argument("--generation-seed", type=int, **default(DEFAULT_GENERATION_SEED))
    parser.add_argument("--syllable-num", type=int, **default(DEFAULT_SYLLABLE_NUM))
    parser.add_argument("--temperature", type=float, **default(DEFAULT_TEMPERATURE))
    parser.add_argument("--top-k", type=int, **default(DEFAULT_TOP_K))
    parser.add_argument("--top-k-by-coordinate", type=parse_int_sequence, **default(DEFAULT_TOP_K_BY_COORDINATE))
    parser.add_argument(
        "--temperature-by-coordinate",
        type=parse_float_sequence,
        **default(DEFAULT_TEMPERATURE_BY_COORDINATE),
    )
    parser.add_argument("--stream", action="store_true", default=DEFAULT_STREAM if use_defaults else argparse.SUPPRESS)
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Finish after the requested create, improve, or load operation without opening the chat UI.",
    )


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def metadata_path_for(model_path):
    root, _ = os.path.splitext(model_path)
    return root + ".json"


def normalized_path(path):
    return os.path.normpath(os.path.abspath(path))


def file_descriptor(path):
    if path is None:
        return None
    resolved_path = normalized_path(path)
    digest = hashlib.sha256()
    with open(resolved_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": resolved_path,
        "sha256": digest.hexdigest(),
        "bytes": os.path.getsize(resolved_path),
    }


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def architecture_config(args):
    return {
        "vocab_size": args.vocab_size,
        "block_size": args.block_size,
        "word_block_size": args.word_block_size,
        "syllable_num": args.syllable_num,
        "embedding_dim": args.embedding_dim,
        "head_num": args.head_num,
        "head_size": args.embedding_dim // args.head_num,
        "layer_num": args.layer_num,
        "atomic_lexical": args.atomic_lexical,
        "ilm_input_embeddings": args.ilm_input_embeddings,
        "ilm_output_heads": args.ilm_output_heads,
        "ilm_objective": args.ilm_objective,
    }


def training_config(args):
    return {
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "epoch_num": args.epoch_num,
        "lr": args.lr,
        "optimizer_profile": args.optimizer_profile,
        "lr_schedule": args.lr_schedule,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "grad_clip": args.grad_clip,
        "warmup_iters": args.warmup_iters,
        "lr_decay_iters": args.lr_decay_iters,
        "min_lr": args.min_lr,
        "validation_interval": args.validation_interval,
        "seed": args.seed,
    }


def data_config(args):
    explicit_splits = all([args.train_text, args.validation_text, args.test_text])
    training_path = args.train_text or args.training_text
    return {
        "mode": "explicit_train_validation_test" if explicit_splits else "implicit_80_20",
        "training_text": file_descriptor(training_path),
        "train_text": file_descriptor(args.train_text),
        "validation_text": file_descriptor(args.validation_text),
        "test_text": file_descriptor(args.test_text),
        "tokenizer_json": file_descriptor(args.tokenizer_json),
        "oov_policy": args.oov_policy,
        "oov_fallback_code": args.oov_fallback_code,
    }


def sampling_config(args):
    return {
        "completed_words": args.completed_words,
        "syllable_num": args.syllable_num,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_k_by_coordinate": (
            list(args.top_k_by_coordinate)
            if args.top_k_by_coordinate is not None
            else None
        ),
        "temperature_by_coordinate": (
            list(args.temperature_by_coordinate)
            if args.temperature_by_coordinate is not None
            else None
        ),
        "stream": args.stream,
        "generation_seed": args.generation_seed,
    }


def format_setting(value):
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def color_text(text, color):
    return f"{color}{text}{ANSI_RESET}"


def print_inference_config(args):
    top_k_mode = "overridden" if args.top_k_by_coordinate is not None else "active"
    top_k_by_coordinate_mode = "active" if args.top_k_by_coordinate is not None else "off"
    temperature_mode = "overridden" if args.temperature_by_coordinate is not None else "active"
    temperature_by_coordinate_mode = "active" if args.temperature_by_coordinate is not None else "off"

    settings = [
        ("Completed words", args.completed_words, "active"),
        ("Syllables per token", args.syllable_num, "active"),
        ("Temperature", args.temperature, temperature_mode),
        ("Temperature by coordinate", args.temperature_by_coordinate, temperature_by_coordinate_mode),
        ("Top-k", args.top_k, top_k_mode),
        ("Top-k by coordinate", args.top_k_by_coordinate, top_k_by_coordinate_mode),
        ("Atomic lexical control", args.atomic_lexical, "active" if args.atomic_lexical else "off"),
        ("ILM input embeddings", args.ilm_input_embeddings, "active" if args.ilm_input_embeddings else "off"),
        ("ILM output heads", args.ilm_output_heads, "active" if args.ilm_output_heads else "off"),
        ("ILM word-prefix objective", args.ilm_objective, "active" if args.ilm_objective else "off"),
        ("Generation seed", args.generation_seed, "active" if args.generation_seed is not None else "off"),
        ("Streaming", args.stream, "active"),
    ]
    rows = [(title, format_setting(value), mode) for title, value, mode in settings]
    title_width = max(len("Setting"), *(len(title) for title, _, _ in rows))
    value_width = max(len("Value"), *(len(value) for _, value, _ in rows))
    mode_width = max(len("Use"), *(len(mode) for _, _, mode in rows))
    rule = f"+-{'-' * title_width}-+-{'-' * value_width}-+-{'-' * mode_width}-+"
    header = (
        f"| {'Setting'.ljust(title_width)} "
        f"| {'Value'.ljust(value_width)} "
        f"| {'Use'.ljust(mode_width)} |"
    )

    print(color_text("Inference configuration", ANSI_BOLD + ANSI_CYAN))
    print(color_text(rule, ANSI_CYAN))
    print(color_text(header, ANSI_BOLD))
    print(color_text(rule, ANSI_CYAN))
    for title, value, mode in rows:
        formatted_value = color_text(value.ljust(value_width), ANSI_GREEN)
        print(f"| {title.ljust(title_width)} | {formatted_value} | {mode.ljust(mode_width)} |")
    print(color_text(rule, ANSI_CYAN))


def run_record(args, command, device, output_model_path, input_model_path=None, semver=None, model=None, losses=None):
    record = {
        "command": command,
        "created_at": now_utc(),
        "input_model_path": input_model_path,
        "output_model_path": output_model_path,
        "semver_increment": semver,
        "tokenizer_json": args.tokenizer_json,
        "training_text": args.training_text,
        "data": data_config(args),
        "device": str(device),
        "architecture": architecture_config(args),
        "training": training_config(args),
        "sampling": sampling_config(args),
    }
    if model is not None:
        record["parameter_count"] = parameter_count(model)
    if losses is not None:
        record["losses"] = losses
    return record


def write_metadata(metadata_path, metadata):
    parent_dir = os.path.dirname(metadata_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    print(f"Model metadata saved to {metadata_path}")


def read_metadata(metadata_path):
    with open(metadata_path, "r") as f:
        return json.load(f)


def metadata_references_model(metadata, model_path):
    target = normalized_path(model_path)
    candidates = [
        metadata.get("root_model_path"),
        metadata.get("latest_model_path"),
    ]
    candidates.extend(metadata.get("checkpoints", []))
    for record in metadata.get("training_curriculum", []):
        candidates.append(record.get("input_model_path"))
        candidates.append(record.get("output_model_path"))

    return any(path and normalized_path(path) == target for path in candidates)


def find_metadata_path(model_path):
    direct_path = metadata_path_for(model_path)
    if os.path.exists(direct_path):
        return direct_path

    model_dir = os.path.dirname(model_path) or "."
    if not os.path.isdir(model_dir):
        return None

    for name in sorted(os.listdir(model_dir)):
        if not name.endswith(".json"):
            continue
        candidate_path = os.path.join(model_dir, name)
        try:
            metadata = read_metadata(candidate_path)
        except (json.JSONDecodeError, OSError):
            continue
        if metadata_references_model(metadata, model_path):
            return candidate_path

    return None


def create_metadata(args, device, model, losses, manager):
    timestamp = now_utc()
    metadata = {
        "schema_version": 1,
        "created_at": timestamp,
        "updated_at": timestamp,
        "root_model_path": args.model_path,
        "latest_model_path": args.model_path,
        "tokenizer_json": args.tokenizer_json,
        "training_text": args.training_text,
        "data": data_config(args),
        "data_statistics": manager.data_statistics(),
        "architecture": architecture_config(args),
        "checkpoints": [args.model_path],
        "training_curriculum": [
            run_record(
                args=args,
                command="create",
                device=device,
                output_model_path=args.model_path,
                model=model,
                losses=losses,
            )
        ],
    }
    write_metadata(metadata_path_for(args.model_path), metadata)


def append_improvement_metadata(metadata_path, args, device, model, improved_model, semver, losses, manager):
    metadata = read_metadata(metadata_path)
    checkpoints = metadata.setdefault("checkpoints", [])
    for checkpoint in [args.model_path, improved_model]:
        if checkpoint not in checkpoints:
            checkpoints.append(checkpoint)

    metadata["updated_at"] = now_utc()
    metadata["latest_model_path"] = improved_model
    metadata["data"] = data_config(args)
    metadata["data_statistics"] = manager.data_statistics()
    metadata.setdefault("training_curriculum", []).append(
        run_record(
            args=args,
            command="improve",
            device=device,
            input_model_path=args.model_path,
            output_model_path=improved_model,
            semver=semver,
            model=model,
            losses=losses,
        )
    )
    write_metadata(metadata_path, metadata)


def build_parser():
    parser = argparse.ArgumentParser(description="Create, improve, load, and sample ILM checkpoints.")
    add_shared_arguments(parser)

    subcommand_options = argparse.ArgumentParser(add_help=False)
    add_shared_arguments(subcommand_options, use_defaults=False)

    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser(
        "create",
        parents=[subcommand_options],
        help="Train a fresh model checkpoint.",
    )
    create_parser.add_argument("model_path")

    improve_parser = subparsers.add_parser(
        "improve",
        parents=[subcommand_options],
        help="Load a checkpoint, train more, and save a new version.",
    )
    improve_parser.add_argument("model_path")
    version_group = improve_parser.add_mutually_exclusive_group()
    version_group.add_argument("--major", action="store_const", const="major", dest="semver")
    version_group.add_argument("--minor", action="store_const", const="minor", dest="semver")
    version_group.add_argument("--patch", action="store_const", const="patch", dest="semver")

    load_parser = subparsers.add_parser(
        "load",
        parents=[subcommand_options],
        help="Load a checkpoint and open the interactive UI.",
    )
    load_parser.add_argument("model_path")

    return parser


def validate_sampling_args(parser, args):
    split_paths = [args.train_text, args.validation_text, args.test_text]
    if any(split_paths) and not all(split_paths):
        parser.error("--train-text, --validation-text, and --test-text must be provided together")
    if args.syllable_num <= 0:
        parser.error("--syllable-num must be positive")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.temperature < 0:
        parser.error("--temperature must be zero or positive")
    if args.validation_interval <= 0:
        parser.error("--validation-interval must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    if not 0 <= args.beta1 < 1 or not 0 <= args.beta2 < 1:
        parser.error("--beta1 and --beta2 must be in [0, 1)")
    if args.grad_clip < 0:
        parser.error("--grad-clip must be non-negative")
    if args.warmup_iters < 0:
        parser.error("--warmup-iters must be non-negative")
    if args.lr_schedule == "cosine":
        if args.lr_decay_iters is None or args.lr_decay_iters <= args.warmup_iters:
            parser.error("--lr-schedule cosine requires --lr-decay-iters > --warmup-iters")
        if args.min_lr is None or args.min_lr < 0:
            parser.error("--lr-schedule cosine requires a non-negative --min-lr")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.generation_seed is not None and args.generation_seed < 0:
        parser.error("--generation-seed must be non-negative")
    if args.oov_fallback_code is not None and args.oov_policy != "fallback":
        parser.error("--oov-fallback-code requires --oov-policy fallback")

    coordinate_settings = [
        ("--top-k-by-coordinate", args.top_k_by_coordinate),
        ("--temperature-by-coordinate", args.temperature_by_coordinate),
    ]
    for argument_name, values in coordinate_settings:
        if values is None:
            continue
        if len(values) != args.syllable_num:
            parser.error(
                f"{argument_name} must contain exactly {args.syllable_num} values "
                f"when --syllable-num is {args.syllable_num}"
            )
    if args.word_block_size <= 0:
        parser.error("--word-block-size must be positive")
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.head_num <= 0:
        parser.error("--head-num must be positive")
    if args.embedding_dim % args.head_num != 0:
        parser.error("--embedding-dim must be divisible by --head-num")
    expected_block_size = args.syllable_num * args.word_block_size
    if args.block_size != expected_block_size:
        parser.error(
            "--block-size must equal --syllable-num * --word-block-size "
            f"({args.syllable_num} * {args.word_block_size} = {expected_block_size})"
        )
    if args.ilm_objective and args.word_block_size != args.block_size // args.syllable_num:
        parser.error(
            "--ilm-objective requires --word-block-size to equal "
            "--block-size // --syllable-num"
        )
    if args.atomic_lexical:
        if args.syllable_num != 1:
            parser.error("--atomic-lexical requires --syllable-num 1")
        if args.block_size != args.word_block_size:
            parser.error("--atomic-lexical requires --block-size == --word-block-size")
        if args.ilm_input_embeddings or args.ilm_output_heads or args.ilm_objective:
            parser.error(
                "--atomic-lexical cannot be combined with ILM architecture flags"
            )


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_model(args, device):
    return ilm.IntuinisticLanguageModel(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        block_size=args.block_size,
        layer_num=args.layer_num,
        device=device,
        dropout=args.dropout,
        syllable_num=args.syllable_num,
        word_block_size=args.word_block_size,
        head_num=args.head_num,
        ilm_input_embeddings=args.ilm_input_embeddings,
        ilm_output_heads=args.ilm_output_heads,
        ilm_objective=args.ilm_objective,
    )


def build_manager(args, tokenizer, device):
    train_path = args.train_text or args.training_text
    with open(train_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    validation_text = None
    test_text = None
    if args.train_text:
        with open(args.validation_text, "r", encoding="utf-8") as f:
            validation_text = f.read()
        with open(args.test_text, "r", encoding="utf-8") as f:
            test_text = f.read()

    return ilm.TrainingManager(
        raw_text,
        tokenizer,
        device=device,
        batch_size=args.batch_size,
        block_size=args.block_size,
        syllable_num=args.syllable_num,
        return_start_positions=(
            args.ilm_input_embeddings
            or args.ilm_output_heads
            or args.ilm_objective
        ),
        validation_text=validation_text,
        test_text=test_text,
        oov_policy=args.oov_policy,
        fallback_code=args.oov_fallback_code,
    )


def train_model_with_args(model, manager, args):
    return model.train_model(
        manager,
        epoch_num=args.epoch_num,
        lr=args.lr,
        validation_interval=args.validation_interval,
        optimizer_profile=args.optimizer_profile,
        lr_schedule=args.lr_schedule,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        grad_clip=args.grad_clip,
        warmup_iters=args.warmup_iters,
        lr_decay_iters=args.lr_decay_iters,
        min_lr=args.min_lr,
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    validate_sampling_args(parser, args)

    # Apply the run seed before model construction, batch sampling, and dropout.
    ilm.set_seed(args.seed)
    if args.atomic_lexical:
        tokenizer, detokenizer = ilm.load_atomic_lexical_tokenizer(args.tokenizer_json)
        args.vocab_size = len(tokenizer.direct_mapping)
        print(f"Atomic lexical vocabulary inferred from tokenizer: {args.vocab_size}")
    else:
        tokenizer, detokenizer = ilm.load_tokenizer(args.tokenizer_json)
    device = get_device()

    ilmodel = build_model(args, device)

    if args.command == "create":
        manager = build_manager(args, tokenizer, device)
        losses = train_model_with_args(ilmodel, manager, args)

        os.makedirs("models", exist_ok=True)
        ilmodel.save_model(model_path=args.model_path)
        create_metadata(args, device, ilmodel, losses, manager)

    elif args.command == "improve":
        metadata_path = find_metadata_path(args.model_path)
        if metadata_path is None:
            parser.error(
                "no model metadata JSON found for this checkpoint; "
                "run `create` to start a recorded model curriculum"
            )
        ilmodel.load_model(model_path=args.model_path)
        manager = build_manager(args, tokenizer, device)
        losses = train_model_with_args(ilmodel, manager, args)

        if args.semver is None:
            print("No semantic versioning given. Creating patch.")
        semver = args.semver or "patch"
        improved_model = ilm.increment_version(name=args.model_path, semver_increment=semver)
        ilmodel.save_model(model_path=improved_model)
        append_improvement_metadata(
            metadata_path,
            args,
            device,
            ilmodel,
            improved_model,
            semver,
            losses,
            manager,
        )

    elif args.command == "load":
        ilmodel.load_model(model_path=args.model_path)

    print(f"Model has {parameter_count(ilmodel)/1e3}k parameters")
    print_inference_config(args)
    if not args.no_interactive:
        ilm.user_interface(
            ilmodel,
            tokenizer,
            detokenizer,
            completed_words=args.completed_words,
            syllable_num=args.syllable_num,
            temperature=args.temperature,
            top_k=args.top_k,
            top_k_by_coordinate=args.top_k_by_coordinate,
            temperature_by_coordinate=args.temperature_by_coordinate,
            stream=args.stream,
            generation_seed=args.generation_seed,
            oov_policy=args.oov_policy,
            oov_fallback_code=args.oov_fallback_code,
        )


if __name__ == "__main__":
    main()
