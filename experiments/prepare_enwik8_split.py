"""Write the conventional enwik8 90M/5M/5M contiguous byte split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional


TOTAL_BYTES = 100_000_000
TRAIN_END = 90_000_000
VALIDATION_END = 95_000_000


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write the standard enwik8 90M/5M/5M byte split without moving boundaries."
    )
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    source_path = Path(args.source_file)
    output_dir = Path(args.output_dir)
    source = source_path.read_bytes()
    if len(source) != TOTAL_BYTES:
        raise SystemExit(
            f"expected {TOTAL_BYTES} bytes for enwik8, found {len(source)} in {source_path}"
        )

    ranges = {
        "train": (0, TRAIN_END),
        "validation": (TRAIN_END, VALIDATION_END),
        "test": (VALIDATION_END, TOTAL_BYTES),
    }
    outputs = {name: output_dir / f"{name}.txt" for name in ranges}
    manifest_path = output_dir / "manifest.json"
    if not args.overwrite and (manifest_path.exists() or any(path.exists() for path in outputs.values())):
        raise SystemExit(f"split outputs already exist in {output_dir}; pass --overwrite to replace them")

    # ILM and the semantic tokenizer operate on UTF-8 text, so fail explicitly
    # if a canonical byte boundary were to cut an encoded character.
    split_manifest = {}
    for name, (start, end) in ranges.items():
        split = source[start:end]
        split.decode("utf-8")
        split_manifest[name] = {
            "path": str(outputs[name]),
            "byte_range": [start, end],
            "bytes": len(split),
            "sha256": sha256_bytes(split),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, details in split_manifest.items():
        outputs[name].write_bytes(source[slice(*details["byte_range"])])

    manifest = {
        "source_file": str(source_path),
        "source_bytes": len(source),
        "source_sha256": sha256_bytes(source),
        "protocol": "canonical contiguous 90M/5M/5M enwik8 byte split",
        "splits": split_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote canonical enwik8 split manifest to {manifest_path}")
    for name, details in split_manifest.items():
        print(f"{name}: {details['bytes']} bytes, sha256={details['sha256']}")


if __name__ == "__main__":
    main()
