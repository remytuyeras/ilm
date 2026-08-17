"""Create a fixed word-to-code permutation control from an ILM tokenizer JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve an ILM tokenizer's lexical vocabulary and final code set, "
            "but randomly permute code assignments across lexical tokens."
        )
    )
    parser.add_argument("--source-tokenizer", required=True)
    parser.add_argument("--target-tokenizer", required=True)
    parser.add_argument("--permutation-seed", type=int, required=True)
    return parser


def load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        mapping = json.load(source)
    if not isinstance(mapping.get("direct"), dict) or not isinstance(mapping.get("reverse"), dict):
        raise ValueError(f"{path} must contain object-valued 'direct' and 'reverse' mappings")
    return mapping


def permute_mapping(
    mapping: dict[str, Any],
    permutation_seed: int,
) -> dict[str, Any]:
    tokens = list(mapping["direct"])
    codes = list(mapping["direct"].values())
    if len(tokens) != len(codes) or len(set(codes)) != len(codes):
        raise ValueError("Tokenizer direct mapping must assign one unique code to each token")

    shuffled_codes = list(codes)
    random.Random(permutation_seed).shuffle(shuffled_codes)
    direct = dict(zip(tokens, shuffled_codes))
    reverse = {code: token for token, code in direct.items()}

    metadata = dict(mapping.get("metadata", {}))
    metadata.update(
        {
            "control": "fixed_code_permutation",
            "code_permutation_seed": permutation_seed,
            "code_permutation_preserves_final_code_set": True,
        }
    )
    return {"direct": direct, "reverse": reverse, "metadata": metadata}


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.permutation_seed < 0:
        raise SystemExit("--permutation-seed must be non-negative")

    source_path = Path(args.source_tokenizer)
    target_path = Path(args.target_tokenizer)
    source_bytes = source_path.read_bytes()
    source_mapping = load_mapping(source_path)
    control_mapping = permute_mapping(source_mapping, args.permutation_seed)
    control_mapping["metadata"]["source_tokenizer_sha256"] = hashlib.sha256(source_bytes).hexdigest()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(control_mapping, indent=4) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote fixed code-permutation tokenizer to {target_path}")
    print(f"Lexical entries: {len(control_mapping['direct'])}")
    print(f"Permutation seed: {args.permutation_seed}")


if __name__ == "__main__":
    main()
