#!/usr/bin/env python3
"""Build BPB and displacement artifacts for frequency-controlled permutations."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_permutation_controls import (  # noqa: E402
    count_token_frequencies,
    load_mapping,
    permutation_metrics,
)


RESULTS_DIR = ROOT / "experiments" / "evaluation" / "results"
PAPER_DIR = Path(__file__).resolve().parent
SEEDS = (13, 29, 47)
ASSIGNMENTS = (314159, 271828, 161803)
MODEL_OFFSETS = (-0.12, 0.0, 0.12)
ASSIGNMENT_OFFSETS = (-0.035, 0.0, 0.035)
CONDITION_X = {"Flat ILM": 0.0, "Exact-frequency": 1.65, "Unrestricted": 3.30}
COLORS = {"Flat ILM": "#2563eb", "Exact-frequency": "#b45309", "Unrestricted": "#a855f7"}
EDGE_COLOR = "#1f2937"


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    source_tokenizer: str
    train_text: str
    test_text: str
    flat_prefix: str
    exact_prefixes: tuple[str, str, str]
    random_prefixes: tuple[str, str, str]
    exact_tokenizers: tuple[str, str, str]
    random_tokenizers: tuple[str, str, str]


CORPORA = (
    CorpusSpec(
        "Tiny Shakespeare",
        "experiments/evaluation/tokenizers/semantic_d10.json",
        "experiments/evaluation/splits/tinyshakespeare/train.txt",
        "experiments/evaluation/splits/tinyshakespeare/test.txt",
        "c_flat_6m",
        ("frequency_exact_flat_6m", "frequency_2_exact_flat_6m", "frequency_3_exact_flat_6m"),
        ("random_flat_6m", "random_2_flat_6m", "random_3_flat_6m"),
        (
            "experiments/evaluation/tokenizers/frequency_exact_codes_s3_train_seed314159.json",
            "experiments/evaluation/tokenizers/frequency_exact_codes_s3_train_seed271828.json",
            "experiments/evaluation/tokenizers/frequency_exact_codes_s3_train_seed161803.json",
        ),
        (
            "experiments/evaluation/tokenizers/control_permuted_codes_s3_seed314159.json",
            "experiments/evaluation/tokenizers/control_permuted_codes_s3_seed271828.json",
            "experiments/evaluation/tokenizers/control_permuted_codes_s3_seed161803.json",
        ),
    ),
    CorpusSpec(
        "enwik8",
        "experiments/evaluation/tokenizers/enwik8_lossless_semantic_d10_s4.json",
        "experiments/evaluation/splits/enwik8/train.txt",
        "experiments/evaluation/splits/enwik8/test.txt",
        "enwik8_lossless_s4_c_flat_6m",
        (
            "enwik8_lossless_s4_frequency_exact_flat_6m",
            "enwik8_lossless_s4_frequency_2_exact_flat_6m",
            "enwik8_lossless_s4_frequency_3_exact_flat_6m",
        ),
        (
            "enwik8_lossless_s4_random_flat_6m",
            "enwik8_lossless_s4_random_2_flat_6m",
            "enwik8_lossless_s4_random_3_flat_6m",
        ),
        (
            "experiments/evaluation/tokenizers/enwik8_lossless_frequency_exact_codes_s4_train_seed314159.json",
            "experiments/evaluation/tokenizers/enwik8_lossless_frequency_exact_codes_s4_train_seed271828.json",
            "experiments/evaluation/tokenizers/enwik8_lossless_frequency_exact_codes_s4_train_seed161803.json",
        ),
        (
            "experiments/evaluation/tokenizers/enwik8_lossless_permuted_codes_s4_seed314159.json",
            "experiments/evaluation/tokenizers/enwik8_lossless_permuted_codes_s4_seed271828.json",
            "experiments/evaluation/tokenizers/enwik8_lossless_permuted_codes_s4_seed161803.json",
        ),
    ),
)


def load_bpb(path: Path) -> float:
    report = json.loads(path.read_text(encoding="utf-8"))
    return float(report["teacher_forced"]["bits_per_utf8_byte"])


def load_result_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corpus in CORPORA:
        flat = {
            seed: load_bpb(RESULTS_DIR / f"{corpus.flat_prefix}_seed{seed}.test_metrics.json")
            for seed in SEEDS
        }
        for assignment_index, assignment_seed in enumerate(ASSIGNMENTS):
            assignment_rows: list[dict[str, Any]] = []
            missing: list[Path] = []
            for condition, prefixes in (
                ("Exact-frequency", corpus.exact_prefixes),
                ("Unrestricted", corpus.random_prefixes),
            ):
                for seed in SEEDS:
                    path = RESULTS_DIR / f"{prefixes[assignment_index]}_seed{seed}.test_metrics.json"
                    if not path.exists():
                        missing.append(path)
                        continue
                    bpb = load_bpb(path)
                    assignment_rows.append(
                        {
                            "corpus": corpus.name,
                            "assignment_seed": assignment_seed,
                            "model_seed": seed,
                            "condition": condition,
                            "flat_bpb": flat[seed],
                            "bpb": bpb,
                            "delta_bpb": bpb - flat[seed],
                            "report": str(path.relative_to(ROOT)),
                        }
                    )
            if missing:
                raise FileNotFoundError(f"Missing expected report: {missing[0]}")
            rows.extend(assignment_rows)
    return rows


def load_displacement_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corpus in CORPORA:
        source = load_mapping(ROOT / corpus.source_tokenizer)
        train_frequencies, _ = count_token_frequencies(ROOT / corpus.train_text, source)
        test_frequencies, _ = count_token_frequencies(ROOT / corpus.test_text, source)
        for assignment_index, assignment_seed in enumerate(ASSIGNMENTS):
            for condition, tokenizers in (
                ("Exact-frequency", corpus.exact_tokenizers),
                ("Unrestricted", corpus.random_tokenizers),
            ):
                metrics = permutation_metrics(
                    source,
                    load_mapping(ROOT / tokenizers[assignment_index]),
                    train_frequencies,
                    top_tokens=0,
                    test_frequencies=test_frequencies,
                )
                rows.append(
                    {
                        "corpus": corpus.name,
                        "assignment_seed": assignment_seed,
                        "condition": condition,
                        "moved_type_rate": metrics["moved_type_rate"],
                        "train_moved_token_mass": metrics["training"]["rho_tokens"],
                        "test_moved_token_mass": metrics["test"]["rho_tokens"],
                        "train_max_role_tv": metrics["training"]["frequency_weighted_marginal_max_total_variation"],
                        "test_max_role_tv": metrics["test"]["frequency_weighted_marginal_max_total_variation"],
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_mass_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Corpus & Control & Moved types & $\\rho_{\\mathrm{train}}$ & $\\rho_{\\mathrm{test}}$ & "
        "TV$_{\\mathrm{train}}$ & TV$_{\\mathrm{test}}$ \\\\",
        "\\midrule",
    ]
    for corpus in CORPORA:
        for condition in ("Exact-frequency", "Unrestricted"):
            group = [row for row in rows if row["corpus"] == corpus.name and row["condition"] == condition]
            lines.append(
                f"{corpus.name if condition == 'Exact-frequency' else ''} & {condition} & "
                f"{statistics.mean(float(row['moved_type_rate']) for row in group):.3f} & "
                f"{statistics.mean(float(row['train_moved_token_mass']) for row in group):.3f} & "
                f"{statistics.mean(float(row['test_moved_token_mass']) for row in group):.3f} & "
                f"{statistics.mean(float(row['train_max_role_tv']) for row in group):.3f} & "
                f"{statistics.mean(float(row['test_max_role_tv']) for row in group):.3f} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bpb_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Corpus & Control & BPB (mean $\\pm$ s.d.) & BPB increase over Flat (mean $\\pm$ s.d.) \\\\",
        "\\midrule",
    ]
    for corpus in CORPORA:
        flat_rows = [row for row in rows if row["corpus"] == corpus.name]
        flat_mean = statistics.mean(float(row["flat_bpb"]) for row in flat_rows)
        for condition in ("Exact-frequency", "Unrestricted"):
            group = [
                row for row in rows
                if row["corpus"] == corpus.name and row["condition"] == condition
            ]
            assignment_means = [
                statistics.mean(
                    float(row["bpb"]) for row in group
                    if int(row["assignment_seed"]) == assignment_seed
                )
                for assignment_seed in ASSIGNMENTS
            ]
            mean_bpb = statistics.mean(assignment_means)
            assignment_sd = statistics.stdev(assignment_means)
            delta = mean_bpb - flat_mean
            delta_means = [
                assignment_mean - flat_mean for assignment_mean in assignment_means
            ]
            delta_sd = statistics.stdev(delta_means)
            lines.append(
                f"{corpus.name if condition == 'Exact-frequency' else ''} & {condition} & "
                f"${mean_bpb:.4f} \\pm {assignment_sd:.4f}$ & "
                f"${delta:+.4f} \\pm {delta_sd:.4f}$ \\\\")
        lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_seed_bpb_tex(path: Path, rows: list[dict[str, Any]], condition: str) -> None:
    lines = [
        "\\begin{tabular}{lrrr|rrr}",
        "\\toprule",
        " & \\multicolumn{3}{c|}{Tiny Shakespeare} & \\multicolumn{3}{c}{enwik8} \\\\",
        "Assignment & $s=13$ & $s=29$ & $s=47$ & $s=13$ & $s=29$ & $s=47$ \\\\",
        "\\midrule",
    ]
    flat_values: dict[str, list[float]] = {}
    for corpus in CORPORA:
        corpus_rows = [row for row in rows if row["corpus"] == corpus.name]
        flat_values[corpus.name] = [
            float(next(row["flat_bpb"] for row in corpus_rows if row["model_seed"] == seed))
            for seed in SEEDS
        ]
    lines.append(
        "Flat \\ilm{} & "
        + " & ".join(f"{value:.6f}" for value in flat_values["Tiny Shakespeare"])
        + " & "
        + " & ".join(f"{value:.6f}" for value in flat_values["enwik8"])
        + " \\\\"
    )
    lines.append("\\midrule")
    for assignment_seed in ASSIGNMENTS:
        values_by_corpus: dict[str, list[float]] = {}
        for corpus in CORPORA:
            values_by_corpus[corpus.name] = [
                float(
                    next(
                        row["bpb"]
                        for row in rows
                        if row["corpus"] == corpus.name
                        and row["condition"] == condition
                        and row["assignment_seed"] == assignment_seed
                        and row["model_seed"] == seed
                    )
                )
                for seed in SEEDS
            ]
        lines.append(
            f"{assignment_seed} & "
            + " & ".join(f"{value:.6f}" for value in values_by_corpus["Tiny Shakespeare"])
            + " & "
            + " & ".join(f"{value:.6f}" for value in values_by_corpus["enwik8"])
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_progression(rows: list[dict[str, Any]], output_dir: Path) -> None:
    flat_values = [float(row["flat_bpb"]) for row in rows]
    condition_values = [float(row["bpb"]) for row in rows]
    lower, upper = min(flat_values + condition_values), max(flat_values + condition_values)
    padding = max((upper - lower) * 0.12, 0.012)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.45), sharey=False)

    for ax, corpus in zip(axes, CORPORA):
        subset = [row for row in rows if row["corpus"] == corpus.name]
        flat_by_seed = {
            seed: float(next(row["flat_bpb"] for row in subset if row["model_seed"] == seed))
            for seed in SEEDS
        }
        for assignment_index, assignment_seed in enumerate(ASSIGNMENTS):
            assignment_rows = [
                row for row in subset if row["assignment_seed"] == assignment_seed
            ]
            if len(assignment_rows) != 2 * len(SEEDS):
                raise ValueError(
                    f"Expected complete result group for {corpus.name}, assignment {assignment_seed}"
                )
            for model_index, seed in enumerate(SEEDS):
                x_offset = MODEL_OFFSETS[model_index] + ASSIGNMENT_OFFSETS[assignment_index]
                exact = next(
                    row for row in subset
                    if row["assignment_seed"] == assignment_seed
                    and row["model_seed"] == seed
                    and row["condition"] == "Exact-frequency"
                )
                random = next(
                    row for row in subset
                    if row["assignment_seed"] == assignment_seed
                    and row["model_seed"] == seed
                    and row["condition"] == "Unrestricted"
                )
                ax.plot(
                    [
                        CONDITION_X["Flat ILM"] + x_offset,
                        CONDITION_X["Exact-frequency"] + x_offset,
                        CONDITION_X["Unrestricted"] + x_offset,
                    ],
                    [flat_by_seed[seed], float(exact["bpb"]), float(random["bpb"])],
                    color="#dbe3ee",
                    linewidth=0.35,
                    alpha=0.55,
                    zorder=1,
                )
        for condition in ("Flat ILM", "Exact-frequency", "Unrestricted"):
            x = CONDITION_X[condition]
            values = (
                list(flat_by_seed.values())
                if condition == "Flat ILM"
                else [float(row["bpb"]) for row in subset if row["condition"] == condition]
            )
            offsets = MODEL_OFFSETS if condition == "Flat ILM" else [
                MODEL_OFFSETS[SEEDS.index(int(row["model_seed"]))] + ASSIGNMENT_OFFSETS[ASSIGNMENTS.index(int(row["assignment_seed"]))]
                for row in subset if row["condition"] == condition
            ]
            ax.scatter(
                [x + offset for offset in offsets], values,
                s=24, color=COLORS[condition], edgecolor=EDGE_COLOR, linewidth=0.45, zorder=3,
            )
            ax.scatter(x, statistics.mean(values), marker="x", s=35, color="#111827", linewidth=1.1, zorder=4)
        ax.set_xticks(
            list(CONDITION_X.values()),
            ["Original\nassignment", "Exact-frequency\npermutation", "Unrestricted\npermutation"],
            fontsize=7.5,
        )
        ax.set_xlim(-0.42, 3.72)
        ax.set_title(corpus.name)
        ax.grid(axis="y", color="#d1d5db", linewidth=0.7)
        ax.set_axisbelow(True)
        corpus_values = [float(row["flat_bpb"]) for row in subset] + [float(row["bpb"]) for row in subset]
        corpus_padding = max((max(corpus_values) - min(corpus_values)) * 0.13, padding)
        ax.set_ylim(min(corpus_values) - corpus_padding, max(corpus_values) + corpus_padding)
    axes[0].set_ylabel("Test BPB (lower is better)")
    fig.tight_layout(w_pad=1.25)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(figures_dir / f"frequency_permutation_progression.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    artifacts_dir = PAPER_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    result_rows = load_result_rows()
    displacement_rows = load_displacement_rows()
    write_csv(artifacts_dir / "frequency_permutation_progression.csv", result_rows)
    write_csv(artifacts_dir / "frequency_permutation_mass_displacement.csv", displacement_rows)
    write_mass_tex(artifacts_dir / "frequency_permutation_mass_summary.tex", displacement_rows)
    write_bpb_tex(artifacts_dir / "frequency_permutation_bpb_summary.tex", result_rows)
    write_seed_bpb_tex(
        artifacts_dir / "frequency_permutation_exact_raw.tex",
        result_rows,
        "Exact-frequency",
    )
    plot_progression(result_rows, PAPER_DIR)
    print(f"Wrote frequency-permutation progression artifacts to {PAPER_DIR}")


if __name__ == "__main__":
    main()
