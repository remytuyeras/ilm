"""Create a fixed word-to-code permutation control from an ILM tokenizer JSON."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ilm.tokenizer.core import find_tokens


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
    parser.add_argument(
        "--frequency-control",
        choices=("global", "exact"),
        default="global",
        help=(
            "Constraint on code reassignment. 'global' reproduces the existing "
            "unrestricted permutation; 'exact' only exchanges lexical entries "
            "with equal training frequency."
        ),
    )
    parser.add_argument(
        "--frequency-text",
        help=(
            "Text used to count lexical frequencies for 'exact'. "
            "Use the training split to avoid consulting held-out frequencies."
        ),
    )
    return parser


def load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        mapping = json.load(source)
    if not isinstance(mapping.get("direct"), dict) or not isinstance(mapping.get("reverse"), dict):
        raise ValueError(f"{path} must contain object-valued 'direct' and 'reverse' mappings")
    return mapping


def count_token_frequencies(
    frequency_text: Path,
    mapping: dict[str, Any],
) -> tuple[dict[str, int], int]:
    """Count tokenizer events in a text under the frozen tokenization mode."""
    tokenization_mode = mapping.get("metadata", {}).get("tokenization_mode", "legacy")
    if tokenization_mode not in {"legacy", "lossless"}:
        raise ValueError(f"Unknown tokenizer tokenization mode: {tokenization_mode!r}")

    counts: Counter[str] = Counter()
    with frequency_text.open("r", encoding="utf-8") as source:
        for line in source:
            counts.update(find_tokens(line, lossless=tokenization_mode == "lossless"))

    unknown = sorted(set(counts).difference(mapping["direct"]))
    if unknown:
        preview = ", ".join(repr(token) for token in unknown[:5])
        raise ValueError(
            f"{frequency_text} contains {len(unknown)} tokens absent from the frozen mapping: {preview}"
        )
    return ({token: int(counts[token]) for token in mapping["direct"]}, sum(counts.values()))


def frequency_stratum(frequency: int, frequency_control: str) -> int:
    if frequency < 0:
        raise ValueError("Frequencies must be non-negative")
    if frequency_control == "global":
        return 0
    if frequency_control == "exact":
        return frequency
    raise ValueError(f"Unknown frequency control: {frequency_control!r}")


def frequency_count_digest(counts: dict[str, int]) -> str:
    payload = "".join(f"{token}\t{count}\n" for token, count in sorted(counts.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def direct_mapping_digest(direct_mapping: dict[str, str]) -> str:
    payload = "".join(f"{token}\t{code}\n" for token, code in sorted(direct_mapping.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def permute_mapping(
    mapping: dict[str, Any],
    permutation_seed: int,
    *,
    frequency_control: str = "global",
    frequencies: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    tokens = list(mapping["direct"])
    codes = list(mapping["direct"].values())
    if len(tokens) != len(codes) or len(set(codes)) != len(codes):
        raise ValueError("Tokenizer direct mapping must assign one unique code to each token")
    if frequency_control != "global" and frequencies is None:
        raise ValueError(f"Frequency control {frequency_control!r} requires token frequencies")
    if frequencies is not None and set(frequencies) != set(tokens):
        raise ValueError("Frequency counts must contain exactly the frozen lexical vocabulary")

    codes_by_token = dict(zip(tokens, codes))
    tokens_by_stratum: dict[int, list[str]] = defaultdict(list)
    for token in tokens:
        frequency = 0 if frequencies is None else frequencies[token]
        tokens_by_stratum[frequency_stratum(frequency, frequency_control)].append(token)

    rng = random.Random(permutation_seed)
    direct: dict[str, str] = {}
    for stratum in sorted(tokens_by_stratum):
        stratum_tokens = tokens_by_stratum[stratum]
        shuffled_codes = [codes_by_token[token] for token in stratum_tokens]
        rng.shuffle(shuffled_codes)
        direct.update(zip(stratum_tokens, shuffled_codes))
    reverse = {code: token for token, code in direct.items()}

    metadata = dict(mapping.get("metadata", {}))
    eligible_entries = sum(len(group) for group in tokens_by_stratum.values() if len(group) > 1)
    moved_entries = sum(mapping["direct"][token] != code for token, code in direct.items())
    metadata.update(
        {
            "control": "fixed_code_permutation",
            "code_permutation_seed": permutation_seed,
            "code_permutation_preserves_final_code_set": True,
            "code_permutation_frequency_control": frequency_control,
            "code_permutation_stratum_count": len(tokens_by_stratum),
            "code_permutation_eligible_entries": eligible_entries,
            "code_permutation_moved_entries": moved_entries,
            "code_permutation_direct_mapping_sha256": direct_mapping_digest(direct),
        }
    )
    return {"direct": direct, "reverse": reverse, "metadata": metadata}


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.permutation_seed < 0:
        raise SystemExit("--permutation-seed must be non-negative")
    if args.frequency_control != "global" and args.frequency_text is None:
        raise SystemExit("--frequency-text is required for --frequency-control exact")
    if args.frequency_control == "global" and args.frequency_text is not None:
        raise SystemExit("--frequency-text is only valid with --frequency-control exact")

    source_path = Path(args.source_tokenizer)
    target_path = Path(args.target_tokenizer)
    source_bytes = source_path.read_bytes()
    source_mapping = load_mapping(source_path)
    frequencies: Optional[dict[str, int]] = None
    frequency_event_count: Optional[int] = None
    frequency_text_path: Optional[Path] = None
    if args.frequency_text is not None:
        frequency_text_path = Path(args.frequency_text)
        frequencies, frequency_event_count = count_token_frequencies(frequency_text_path, source_mapping)

    control_mapping = permute_mapping(
        source_mapping,
        args.permutation_seed,
        frequency_control=args.frequency_control,
        frequencies=frequencies,
    )
    control_mapping["metadata"]["source_tokenizer_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    if frequencies is not None and frequency_text_path is not None and frequency_event_count is not None:
        control_mapping["metadata"].update(
            {
                "code_permutation_frequency_source_sha256": hashlib.sha256(
                    frequency_text_path.read_bytes()
                ).hexdigest(),
                "code_permutation_frequency_count_sha256": frequency_count_digest(frequencies),
                "code_permutation_frequency_event_count": frequency_event_count,
                "code_permutation_frequency_source": str(frequency_text_path),
            }
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(control_mapping, indent=4) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote fixed code-permutation tokenizer to {target_path}")
    print(f"Lexical entries: {len(control_mapping['direct'])}")
    print(f"Permutation seed: {args.permutation_seed}")
    print(f"Frequency control: {args.frequency_control}")
    if frequencies is not None:
        print(f"Frequency strata: {control_mapping['metadata']['code_permutation_stratum_count']}")
        print(f"Eligible lexical entries: {control_mapping['metadata']['code_permutation_eligible_entries']}")
        print(f"Moved lexical entries: {control_mapping['metadata']['code_permutation_moved_entries']}")


if __name__ == "__main__":
    main()
