"""Check whether a frozen ILM tokenizer covers one or more text files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(1, ROOT_DIR)

import ilm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report frozen-tokenizer coverage for fixed splits or new transfer documents."
    )
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument(
        "--text",
        action="append",
        required=True,
        help="Text file to check. Provide once per split or transfer document.",
    )
    parser.add_argument("--oov-policy", choices=["error", "fallback"], default="error")
    parser.add_argument("--oov-fallback-code", default=None)
    parser.add_argument(
        "--require-lossless-encoding",
        action="store_true",
        help="Fail unless the tokenizer represents every UTF-8 source byte in each file.",
    )
    parser.add_argument("--output-file", default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.oov_fallback_code is not None and args.oov_policy != "fallback":
        parser.error("--oov-fallback-code requires --oov-policy fallback")

    tokenizer, _ = ilm.load_tokenizer(args.tokenizer_json)
    reports = []
    for text_path in args.text:
        raw_text = Path(text_path).read_text(encoding="utf-8")
        try:
            _, stats, _ = ilm.encode_context(
                raw_text,
                tokenizer,
                oov_policy=args.oov_policy,
                fallback_code=args.oov_fallback_code,
            )
        except ilm.UnknownTokenError as exc:
            stats = exc.stats
            reports.append({
                "path": str(Path(text_path)),
                "covered": False,
                "lossless_source_coverage": (
                    stats.tokenized_source_bytes == stats.source_bytes
                ),
                "examples": exc.examples,
                **stats.to_dict(),
            })
            print(
                f"OOV: {text_path}: {stats.oov_token_count}/{stats.token_count} "
                f"({stats.to_dict()['oov_rate']:.4%}) examples={exc.examples!r}"
            )
            continue

        lossless_source_coverage = (
            stats.tokenized_source_bytes == stats.source_bytes
        )
        covered = stats.oov_token_count == 0 and (
            not args.require_lossless_encoding or lossless_source_coverage
        )
        reports.append({
            "path": str(Path(text_path)),
            "covered": covered,
            "lossless_source_coverage": lossless_source_coverage,
            "examples": [],
            **stats.to_dict(),
        })
        print(
            f"Coverage: {text_path}: {stats.token_count - stats.oov_token_count}/"
            f"{stats.token_count} ({1.0 - stats.to_dict()['oov_rate']:.4%}); "
            f"represented UTF-8 bytes: {stats.tokenized_source_bytes}/"
            f"{stats.source_bytes}"
        )

    report = {
        "tokenizer_json": args.tokenizer_json,
        "oov_policy": args.oov_policy,
        "oov_fallback_code": args.oov_fallback_code,
        "require_lossless_encoding": args.require_lossless_encoding,
        "files": reports,
    }
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote coverage report to {output_path}")

    if (
        args.oov_policy == "error" or args.require_lossless_encoding
    ) and any(not item["covered"] for item in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
