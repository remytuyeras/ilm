"""Check whether a hierarchical ILM tokenizer has enough unique codes for a corpus."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(1, ROOT_DIR)

from ilm.tokenizer.core import collect_unique_tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count ILM token types and compare them with base**depth code capacity."
    )
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lossless-tokenization", action="store_true")
    parser.add_argument("--output-file", default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.base <= 1 or args.depth <= 0:
        raise SystemExit("--base must exceed one and --depth must be positive")

    tokens = collect_unique_tokens(
        args.source_file,
        lossless_tokenization=args.lossless_tokenization,
    )
    capacity = args.base ** args.depth
    report = {
        "source_file": args.source_file,
        "base": args.base,
        "depth": args.depth,
        "lossless_tokenization": args.lossless_tokenization,
        "capacity": capacity,
        "unique_tokens": len(tokens),
        "remaining_codes": capacity - len(tokens),
        "fits": len(tokens) <= capacity,
    }
    print(json.dumps(report, indent=2))
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote capacity report to {output_path}")
    if not report["fits"]:
        raise SystemExit(
            f"{report['unique_tokens']} unique tokens exceed {args.base}^{args.depth}={capacity} codes"
        )


if __name__ == "__main__":
    main()
