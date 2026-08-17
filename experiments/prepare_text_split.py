"""Create reproducible contiguous train, validation, and test text splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple


WHITESPACE_BYTES = {ord(" "), ord("\n"), ord("\r"), ord("\t")}
NEWLINE_BYTES = {ord("\n"), ord("\r")}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def boundary_after_whitespace(data: bytes, target: int, lower: int, upper: int) -> int:
    """Choose a nearby UTF-8-safe boundary immediately after whitespace."""
    target = min(max(target, lower), upper)
    max_distance = max(target - lower, upper - target)
    for distance in range(max_distance + 1):
        for index in (target + distance, target - distance):
            if lower <= index < upper and data[index] in WHITESPACE_BYTES:
                return index + 1
    raise ValueError("could not find a whitespace boundary for the requested split")


def boundary_at_line_start(data: bytes, target: int, lower: int, upper: int) -> int:
    """Choose a nearby newline and retain it as the first byte of the next split.

    ILM's tokenizer distinguishes ``" word"`` from ``"word"``. Starting a
    held-out split immediately after a space would therefore manufacture an
    artificial OOV token. Keeping a newline at the beginning of a split
    preserves the source tokenization while still leaving the requested gap.
    """
    target = min(max(target, lower), upper)
    max_distance = max(target - lower, upper - target)
    for distance in range(max_distance + 1):
        for index in (target + distance, target - distance):
            if lower <= index < upper and data[index] in NEWLINE_BYTES:
                return index
    raise ValueError("could not find a line boundary for the requested split")


def make_split_ranges(
        source: bytes,
        train_fraction: float,
        validation_fraction: float,
        test_fraction: float,
        context_gap_bytes: int,
        ) -> Dict[str, Tuple[int, int]]:
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("all split fractions must be positive")
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    if context_gap_bytes < 0:
        raise ValueError("context_gap_bytes must be non-negative")

    total = len(source)
    train_target = int(total * train_fraction)
    validation_target = int(total * (train_fraction + validation_fraction))

    train_end = boundary_after_whitespace(source, train_target, 0, total)
    validation_start = boundary_at_line_start(
        source,
        train_end + context_gap_bytes,
        train_end,
        total,
    )
    validation_end = boundary_after_whitespace(
        source,
        max(validation_target, validation_start + 1),
        validation_start,
        total,
    )
    test_start = boundary_at_line_start(
        source,
        validation_end + context_gap_bytes,
        validation_end,
        total,
    )
    if validation_end <= validation_start or test_start >= total:
        raise ValueError("source is too small for the requested fractions and context gap")
    return {
        "train": (0, train_end),
        "validation": (validation_start, validation_end),
        "test": (test_start, total),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create contiguous text splits with context gaps.")
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-context-bytes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_path = Path(args.source_file)
    output_dir = Path(args.output_dir)
    source = source_path.read_bytes()
    source.decode("utf-8")  # Fail early instead of creating invalid text files.
    ranges = make_split_ranges(
        source,
        args.train_fraction,
        args.validation_fraction,
        args.test_fraction,
        args.max_context_bytes,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    paths = {name: output_dir / f"{name}.txt" for name in ranges}
    if not args.overwrite and (manifest_path.exists() or any(path.exists() for path in paths.values())):
        raise SystemExit(f"split outputs already exist in {output_dir}; pass --overwrite to replace them")

    split_manifest = {}
    for name, (start, end) in ranges.items():
        content = source[start:end]
        paths[name].write_bytes(content)
        split_manifest[name] = {
            "path": str(paths[name]),
            "byte_range": [start, end],
            "bytes": len(content),
            "sha256": sha256_bytes(content),
        }

    manifest = {
        "source_file": str(source_path),
        "source_bytes": len(source),
        "source_sha256": sha256_bytes(source),
        "fractions": {
            "train": args.train_fraction,
            "validation": args.validation_fraction,
            "test": args.test_fraction,
        },
        "context_gap_bytes": args.max_context_bytes,
        "splits": split_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote split manifest to {manifest_path}")
    for name, details in split_manifest.items():
        print(f"{name}: {details['bytes']} bytes, sha256={details['sha256']}")


if __name__ == "__main__":
    main()
