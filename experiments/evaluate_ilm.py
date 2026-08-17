"""Teacher-forced held-out likelihood evaluation for an ILM checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(1, ROOT_DIR)

import ilm


def get_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute teacher-forced ILM likelihood in bits per scored UTF-8 byte."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument("--test-text", required=True)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument(
        "--evaluation-mode",
        choices=["full-context", "block-reset"],
        default="full-context",
        help="Score with rolling full context or with non-overlapping training-style context blocks.",
    )
    parser.add_argument(
        "--oov-policy",
        choices=["error", "fallback"],
        default="error",
        help="Stop on OOVs, or use a deterministic fallback to measure new-document coverage.",
    )
    parser.add_argument(
        "--oov-fallback-code",
        default=None,
        help="Explicit code for fallback OOV handling; defaults to the smallest tokenizer code.",
    )
    parser.add_argument(
        "--require-lossless-encoding",
        action="store_true",
        help="Fail unless the tokenizer represents every UTF-8 byte of the test source.",
    )

    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument(
        "--atomic-lexical",
        action="store_true",
        help="Evaluate a checkpoint trained on one atomic ID per frozen lexical token.",
    )
    parser.add_argument("--syllable-num", type=int, default=3)
    parser.add_argument("--word-block-size", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=300)
    parser.add_argument("--head-num", type=int, default=4)
    parser.add_argument("--layer-num", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--ilm-input-embeddings", action="store_true")
    parser.add_argument("--ilm-output-heads", action="store_true")
    parser.add_argument("--ilm-objective", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.block_size is None:
        args.block_size = args.syllable_num * args.word_block_size
    if args.block_size != args.syllable_num * args.word_block_size:
        parser.error("--block-size must equal --syllable-num * --word-block-size")
    if args.embedding_dim % args.head_num != 0:
        parser.error("--embedding-dim must be divisible by --head-num")
    if args.ilm_objective and args.word_block_size != args.block_size // args.syllable_num:
        parser.error("--ilm-objective requires --word-block-size == --block-size // --syllable-num")
    if args.evaluation_batch_size <= 0:
        parser.error("--evaluation-batch-size must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.oov_fallback_code is not None and args.oov_policy != "fallback":
        parser.error("--oov-fallback-code requires --oov-policy fallback")
    if args.atomic_lexical:
        if args.syllable_num != 1:
            parser.error("--atomic-lexical requires --syllable-num 1")
        if args.block_size != args.word_block_size:
            parser.error("--atomic-lexical requires --block-size == --word-block-size")
        if args.ilm_input_embeddings or args.ilm_output_heads or args.ilm_objective:
            parser.error("--atomic-lexical cannot be combined with ILM architecture flags")


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    ilm.set_seed(args.seed)
    device = get_device(args.device)
    if args.atomic_lexical:
        tokenizer, _ = ilm.load_atomic_lexical_tokenizer(args.tokenizer_json)
        args.vocab_size = len(tokenizer.direct_mapping)
    else:
        tokenizer, _ = ilm.load_tokenizer(args.tokenizer_json)
    raw_test_text = Path(args.test_text).read_text(encoding="utf-8")
    coordinates, encoding_stats, byte_weights = ilm.encode_context(
        raw_test_text,
        tokenizer,
        oov_policy=args.oov_policy,
        fallback_code=args.oov_fallback_code,
    )
    if (
        args.require_lossless_encoding
        and encoding_stats.tokenized_source_bytes != encoding_stats.source_bytes
    ):
        raise SystemExit(
            "Lossless encoding required, but the tokenizer represented "
            f"{encoding_stats.tokenized_source_bytes}/"
            f"{encoding_stats.source_bytes} UTF-8 source bytes."
        )
    model = ilm.IntuinisticLanguageModel(
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
    model.load_model(args.model_path)
    metrics = model.evaluate_teacher_forced(
        coordinates=coordinates,
        coordinate_byte_weights=byte_weights,
        batch_size=args.evaluation_batch_size,
        evaluation_mode=args.evaluation_mode,
    )

    report = {
        "schema_version": 1,
        "model_path": args.model_path,
        "tokenizer_json": args.tokenizer_json,
        "tokenizer_json_sha256": hashlib.sha256(
            Path(args.tokenizer_json).read_bytes()
        ).hexdigest(),
        "test_text": args.test_text,
        "test_text_sha256": hashlib.sha256(raw_test_text.encode("utf-8")).hexdigest(),
        "seed": args.seed,
        "device": str(device),
        "oov_policy": args.oov_policy,
        "oov_fallback_code": args.oov_fallback_code,
        "require_lossless_encoding": args.require_lossless_encoding,
        "evaluation_mode": args.evaluation_mode,
        "event_unit": "lexical-token" if args.atomic_lexical else "coordinate",
        "encoding": encoding_stats.to_dict(),
        "architecture": {
            "vocab_size": args.vocab_size,
            "atomic_lexical": args.atomic_lexical,
            "syllable_num": args.syllable_num,
            "word_block_size": args.word_block_size,
            "block_size": args.block_size,
            "embedding_dim": args.embedding_dim,
            "head_num": args.head_num,
            "layer_num": args.layer_num,
            "ilm_input_embeddings": args.ilm_input_embeddings,
            "ilm_output_heads": args.ilm_output_heads,
            "ilm_objective": args.ilm_objective,
        },
        "teacher_forced": metrics,
    }
    output_path = Path(args.output_file) if args.output_file else Path(args.model_path).with_suffix(".test_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Test BPB: {metrics['bits_per_utf8_byte']:.6f}")
    print(f"Scored coordinate events: {metrics['coordinate_events']}")
    print(f"Scored UTF-8 bytes: {metrics['scored_source_bytes']:.3f}")
    print(f"OOV rate: {encoding_stats.to_dict()['oov_rate']:.4%}")
    print(f"Wrote evaluation report to {output_path}")


if __name__ == "__main__":
    main()
