"""Export frozen text splits in nanoGPT character or UTF-8-byte format."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create train.bin, val.bin, test.bin, and meta.pkl for a nanoGPT baseline."
    )
    parser.add_argument("--corpus-file", required=True, help="Full frozen corpus used to define the character set.")
    parser.add_argument("--train-text", required=True)
    parser.add_argument("--validation-text", required=True)
    parser.add_argument("--test-text", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--unit",
        choices=["character", "utf8-byte"],
        default="character",
        help="Character events for text corpora, or raw UTF-8 bytes for byte-level benchmarks.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    source_paths = {
        "train": Path(args.train_text),
        "validation": Path(args.validation_text),
        "test": Path(args.test_text),
    }
    if args.unit == "character":
        corpus = Path(args.corpus_file).read_text(encoding="utf-8")
        splits = {name: path.read_text(encoding="utf-8") for name, path in source_paths.items()}
        corpus_bytes = corpus.encode("utf-8")
    else:
        corpus = Path(args.corpus_file).read_bytes()
        splits = {name: path.read_bytes() for name, path in source_paths.items()}
        corpus_bytes = bytes(corpus)

    units = sorted(set(corpus))
    stoi = {unit: index for index, unit in enumerate(units)}
    itos = {index: unit for unit, index in stoi.items()}

    output_dir = Path(args.output_dir)
    outputs = {
        "train": output_dir / "train.bin",
        "validation": output_dir / "val.bin",
        "test": output_dir / "test.bin",
        "meta": output_dir / "meta.pkl",
        "manifest": output_dir / "manifest.json",
    }
    if not args.overwrite and any(path.exists() for path in outputs.values()):
        raise SystemExit(f"outputs already exist in {output_dir}; pass --overwrite to replace them")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_splits = {}
    for name, values in splits.items():
        unknown = sorted(set(values) - set(stoi))
        if unknown:
            raise ValueError(f"{name} contains units absent from the frozen corpus: {unknown!r}")
        ids = np.asarray([stoi[value] for value in values], dtype=np.uint16)
        output_path = outputs[name]
        ids.tofile(output_path)
        raw_bytes = values.encode("utf-8") if isinstance(values, str) else bytes(values)
        manifest_splits[name] = {
            "text_path": str(source_paths[name]),
            "events": len(values),
            "utf8_bytes": len(raw_bytes),
            "sha256": sha256_bytes(raw_bytes),
            "binary_path": str(output_path),
            "binary_dtype": "uint16",
        }

    with outputs["meta"].open("wb") as handle:
        pickle.dump({"vocab_size": len(units), "stoi": stoi, "itos": itos, "unit": args.unit}, handle)
    manifest = {
        "corpus_file": str(args.corpus_file),
        "corpus_sha256": sha256_bytes(corpus_bytes),
        "unit": args.unit,
        "vocab_size": len(units),
        "splits": manifest_splits,
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote nanoGPT {args.unit} data to {output_dir}")
    print(f"Vocabulary: {len(units)}")


if __name__ == "__main__":
    main()
