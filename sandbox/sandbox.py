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

    tokenizer_json = "data/tokenizer_embedding_cluster_v1.json"
    training_text = "data/training_old_english.txt"

Relative-position tokenizer path that may still be useful for comparison:

    tokenizer_json = "data/tokenizer_v2.json"

Older small-data defaults that may still be useful for comparison:

    tokenizer_json = "data/tokenizer_v1.json"
    training_text = "data/training_input.txt"

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
    coordinate_token_embeddings = False
    coordinate_lm_heads = False
    word_row_transformer = False

If you change architecture parameters, existing checkpoint files may no longer
load because the saved tensor shapes must match the model architecture.

`coordinate_lm_heads` changes the checkpoint architecture by using one language
model head per coordinate role.

`word_row_transformer` keeps standard coordinate-time attention, implies
`coordinate_lm_heads`, and changes training to use a word-row prefix loss.
Generation does not use the row map.

Current usage:

    python sandbox/sandbox.py create models/m2.v0.0.0.pth
    python sandbox/sandbox.py improve models/m2.v0.0.0.pth --patch
    python sandbox/sandbox.py load models/m2.v0.0.0.pth

The `create` command writes one JSON metadata file next to the first model
checkpoint. Later `improve` commands keep appending to that same JSON file so
the model's training curriculum stays together instead of splitting across one
metadata file per checkpoint.
"""

import argparse
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

DEFAULT_TOKENIZER_JSON = "data/tokenizer_embedding_cluster_v1.json"
DEFAULT_TRAINING_TEXT = "data/training_old_english.txt"

DEFAULT_SYLLABLE_NUM = 3
DEFAULT_WORD_BLOCK_SIZE = 25
DEFAULT_VOCAB_SIZE = 64
DEFAULT_BLOCK_SIZE = DEFAULT_SYLLABLE_NUM * DEFAULT_WORD_BLOCK_SIZE
DEFAULT_BATCH_SIZE = 32
DEFAULT_EMBEDDING_DIM = 600
DEFAULT_HEAD_NUM = 4
DEFAULT_HEAD_SIZE = DEFAULT_EMBEDDING_DIM // DEFAULT_HEAD_NUM
DEFAULT_LAYER_NUM = 8
DEFAULT_COORDINATE_TOKEN_EMBEDDINGS = False
DEFAULT_COORDINATE_LM_HEADS = False
DEFAULT_WORD_ROW_TRANSFORMER = False

DEFAULT_DROPOUT = 0.5
DEFAULT_EPOCH_NUM = 4000
DEFAULT_LR = 1e-3

DEFAULT_COMPLETED_WORDS = 300

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

    parser.add_argument("--vocab-size", type=int, **default(DEFAULT_VOCAB_SIZE))
    parser.add_argument("--block-size", type=int, **default(DEFAULT_BLOCK_SIZE))
    parser.add_argument("--word-block-size", type=int, **default(DEFAULT_WORD_BLOCK_SIZE))
    parser.add_argument("--batch-size", type=int, **default(DEFAULT_BATCH_SIZE))
    parser.add_argument("--embedding-dim", type=int, **default(DEFAULT_EMBEDDING_DIM))
    parser.add_argument("--head-num", type=int, **default(DEFAULT_HEAD_NUM))
    parser.add_argument("--layer-num", type=int, **default(DEFAULT_LAYER_NUM))
    parser.add_argument(
        "--coordinate-token-embeddings",
        action="store_true",
        default=DEFAULT_COORDINATE_TOKEN_EMBEDDINGS if use_defaults else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--coordinate-lm-heads",
        action="store_true",
        default=DEFAULT_COORDINATE_LM_HEADS if use_defaults else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--word-row-transformer",
        action="store_true",
        default=DEFAULT_WORD_ROW_TRANSFORMER if use_defaults else argparse.SUPPRESS,
    )
    parser.add_argument("--dropout", type=float, **default(DEFAULT_DROPOUT))
    parser.add_argument("--epoch-num", type=int, **default(DEFAULT_EPOCH_NUM))
    parser.add_argument("--lr", type=float, **default(DEFAULT_LR))

    parser.add_argument("--completed-words", type=int, **default(DEFAULT_COMPLETED_WORDS))
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


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def metadata_path_for(model_path):
    root, _ = os.path.splitext(model_path)
    return root + ".json"


def normalized_path(path):
    return os.path.normpath(os.path.abspath(path))


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
        "coordinate_token_embeddings": args.coordinate_token_embeddings,
        "coordinate_lm_heads": args.coordinate_lm_heads,
        "word_row_transformer": args.word_row_transformer,
    }


def training_config(args):
    return {
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "epoch_num": args.epoch_num,
        "lr": args.lr,
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
        ("Coordinate token embeddings", args.coordinate_token_embeddings, "active" if args.coordinate_token_embeddings else "off"),
        ("Word-row transformer", args.word_row_transformer, "active" if args.word_row_transformer else "off"),
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


def create_metadata(args, device, model, losses):
    timestamp = now_utc()
    metadata = {
        "schema_version": 1,
        "created_at": timestamp,
        "updated_at": timestamp,
        "root_model_path": args.model_path,
        "latest_model_path": args.model_path,
        "tokenizer_json": args.tokenizer_json,
        "training_text": args.training_text,
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


def append_improvement_metadata(metadata_path, args, device, model, improved_model, semver, losses):
    metadata = read_metadata(metadata_path)
    checkpoints = metadata.setdefault("checkpoints", [])
    for checkpoint in [args.model_path, improved_model]:
        if checkpoint not in checkpoints:
            checkpoints.append(checkpoint)

    metadata["updated_at"] = now_utc()
    metadata["latest_model_path"] = improved_model
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
    if args.syllable_num <= 0:
        parser.error("--syllable-num must be positive")
    if args.word_row_transformer:
        args.coordinate_lm_heads = True
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.temperature < 0:
        parser.error("--temperature must be zero or positive")

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
        coordinate_token_embeddings=args.coordinate_token_embeddings,
        coordinate_lm_heads=args.coordinate_lm_heads,
        word_row_transformer=args.word_row_transformer,
    )


def build_manager(args, tokenizer, device):
    with open(args.training_text, "r") as f:
        raw_text = f.read()

    return ilm.TrainingManager(
        raw_text,
        tokenizer,
        device=device,
        batch_size=args.batch_size,
        block_size=args.block_size,
        syllable_num=args.syllable_num,
        return_start_positions=(
            args.coordinate_token_embeddings
            or args.coordinate_lm_heads
            or args.word_row_transformer
        ),
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    validate_sampling_args(parser, args)

    tokenizer, detokenizer = ilm.load_tokenizer(args.tokenizer_json)
    device = get_device()

    ilmodel = build_model(args, device)

    if args.command == "create":
        manager = build_manager(args, tokenizer, device)
        losses = ilmodel.train_model(
            manager,
            epoch_num=args.epoch_num,
            lr=args.lr,
        )

        os.makedirs("models", exist_ok=True)
        ilmodel.save_model(model_path=args.model_path)
        create_metadata(args, device, ilmodel, losses)

    elif args.command == "improve":
        metadata_path = find_metadata_path(args.model_path)
        if metadata_path is None:
            parser.error(
                "no model metadata JSON found for this checkpoint; "
                "run `create` to start a recorded model curriculum"
            )
        ilmodel.load_model(model_path=args.model_path)
        manager = build_manager(args, tokenizer, device)
        losses = ilmodel.train_model(
            manager,
            epoch_num=args.epoch_num,
            lr=args.lr,
        )

        if args.semver is None:
            print("No semantic versioning given. Creating patch.")
        semver = args.semver or "patch"
        improved_model = ilm.increment_version(name=args.model_path, semver_increment=semver)
        ilmodel.save_model(model_path=improved_model)
        append_improvement_metadata(metadata_path, args, device, ilmodel, improved_model, semver, losses)

    elif args.command == "load":
        ilmodel.load_model(model_path=args.model_path)

    print(f"Model has {parameter_count(ilmodel)/1e3}k parameters")
    print_inference_config(args)
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
    )


if __name__ == "__main__":
    main()
