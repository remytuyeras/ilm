"""Summarize lexical and training-event displacement for permutation controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.create_permuted_tokenizer import count_token_frequencies, load_mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure type and training-event displacement under selected code permutations."
    )
    parser.add_argument("--source-tokenizer", required=True)
    parser.add_argument("--frequency-text", required=True)
    parser.add_argument(
        "--test-text",
        help="Optional held-out text used for test-event displacement and marginal diagnostics.",
    )
    parser.add_argument(
        "--permuted-tokenizer",
        action="append",
        required=True,
        help="Permutation tokenizer to analyze. Repeat this option for multiple maps.",
    )
    parser.add_argument("--top-tokens", type=int, default=20)
    parser.add_argument("--output-file")
    return parser


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinate_marginal_metrics(
    source_direct: dict[str, str],
    target_direct: dict[str, str],
    frequencies: dict[str, int],
) -> dict[str, Any]:
    source_codes = {
        token: tuple(int(value) for value in code.split(":"))
        for token, code in source_direct.items()
    }
    target_codes = {
        token: tuple(int(value) for value in code.split(":"))
        for token, code in target_direct.items()
    }
    depth = len(next(iter(source_codes.values())))
    if any(len(code) != depth for code in target_codes.values()):
        raise ValueError("Source and target mappings must use one common coordinate depth")

    total_events = sum(frequencies.values())
    role_total_variation: list[float] = []
    role_l1_difference: list[int] = []
    maximum_cell_difference = 0
    for role in range(depth):
        source_counts: dict[int, int] = {}
        target_counts: dict[int, int] = {}
        for token, frequency in frequencies.items():
            source_value = source_codes[token][role]
            target_value = target_codes[token][role]
            source_counts[source_value] = source_counts.get(source_value, 0) + frequency
            target_counts[target_value] = target_counts.get(target_value, 0) + frequency
        l1_difference = 0
        for value in set(source_counts).union(target_counts):
            difference = abs(source_counts.get(value, 0) - target_counts.get(value, 0))
            l1_difference += difference
            maximum_cell_difference = max(maximum_cell_difference, difference)
        role_l1_difference.append(l1_difference)
        role_total_variation.append(l1_difference / (2 * total_events) if total_events else 0.0)

    return {
        "coordinate_depth": depth,
        "frequency_weighted_marginals_match": all(value == 0 for value in role_l1_difference),
        "frequency_weighted_marginal_l1_by_role": role_l1_difference,
        "frequency_weighted_marginal_total_variation_by_role": role_total_variation,
        "frequency_weighted_marginal_max_total_variation": max(role_total_variation, default=0.0),
        "frequency_weighted_marginal_max_cell_difference": maximum_cell_difference,
    }


def event_mass_metrics(
    moved: list[str],
    frequencies: dict[str, int],
) -> dict[str, Any]:
    total_events = sum(frequencies.values())
    moved_events = sum(frequencies[token] for token in moved)
    return {
        "event_count": total_events,
        "moved_event_count": moved_events,
        "moved_event_rate": moved_events / total_events if total_events else 0.0,
        "rho_tokens": moved_events / total_events if total_events else 0.0,
        "unmoved_event_count": total_events - moved_events,
    }


def permutation_metrics(
    source_mapping: dict[str, Any],
    target_mapping: dict[str, Any],
    training_frequencies: dict[str, int],
    top_tokens: int,
    test_frequencies: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    source_direct = source_mapping["direct"]
    target_direct = target_mapping["direct"]
    if set(source_direct) != set(target_direct):
        raise ValueError("Source and target mappings must contain the same lexical vocabulary")
    if set(source_direct.values()) != set(target_direct.values()):
        raise ValueError("Source and target mappings must occupy the same final code set")

    moved = [token for token in source_direct if source_direct[token] != target_direct[token]]
    unmoved = [token for token in source_direct if source_direct[token] == target_direct[token]]

    descending_training_frequency = lambda token: (-training_frequencies[token], token)
    report = {
        "lexical_entry_count": len(source_direct),
        "moved_type_count": len(moved),
        "moved_type_rate": len(moved) / len(source_direct),
        "training": {
            **event_mass_metrics(moved, training_frequencies),
            **coordinate_marginal_metrics(source_direct, target_direct, training_frequencies),
        },
        "top_unmoved_tokens_by_training_frequency": [
            {"token": token, "training_frequency": training_frequencies[token]}
            for token in sorted(unmoved, key=descending_training_frequency)[:top_tokens]
        ],
        "top_moved_tokens_by_training_frequency": [
            {"token": token, "training_frequency": training_frequencies[token]}
            for token in sorted(moved, key=descending_training_frequency)[:top_tokens]
        ],
    }
    if test_frequencies is not None:
        report["test"] = {
            **event_mass_metrics(moved, test_frequencies),
            **coordinate_marginal_metrics(source_direct, target_direct, test_frequencies),
        }
    return report


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.top_tokens < 0:
        raise SystemExit("--top-tokens must be non-negative")

    source_path = Path(args.source_tokenizer)
    frequency_path = Path(args.frequency_text)
    source_mapping = load_mapping(source_path)
    training_frequencies, training_event_count = count_token_frequencies(frequency_path, source_mapping)
    test_path = Path(args.test_text) if args.test_text else None
    test_frequencies: Optional[dict[str, int]] = None
    test_event_count: Optional[int] = None
    if test_path is not None:
        test_frequencies, test_event_count = count_token_frequencies(test_path, source_mapping)
    reports = []

    for tokenizer_name in args.permuted_tokenizer:
        target_path = Path(tokenizer_name)
        target_mapping = load_mapping(target_path)
        report = permutation_metrics(
            source_mapping,
            target_mapping,
            training_frequencies,
            args.top_tokens,
            test_frequencies=test_frequencies,
        )
        report.update(
            {
                "source_tokenizer": str(source_path),
                "source_tokenizer_sha256": file_sha256(source_path),
                "frequency_text": str(frequency_path),
                "frequency_text_sha256": file_sha256(frequency_path),
                "test_text": str(test_path) if test_path else None,
                "test_text_sha256": file_sha256(test_path) if test_path else None,
                "permuted_tokenizer": str(target_path),
                "permuted_tokenizer_sha256": file_sha256(target_path),
                "permutation_seed": target_mapping.get("metadata", {}).get("code_permutation_seed"),
                "frequency_control": target_mapping.get("metadata", {}).get(
                    "code_permutation_frequency_control", "global"
                ),
            }
        )
        reports.append(report)

    payload = {
        "training_event_count": training_event_count,
        "test_event_count": test_event_count,
        "reports": reports,
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"Wrote permutation displacement report to {output_path}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
