#!/usr/bin/env python3
"""Build tables and figures for the controlled ILM evaluation paper.

The script deliberately reads the seed-level teacher-forced reports in
``experiments/evaluation/results``.  It writes paper-local CSV, TeX, Markdown,
PDF, and PNG assets, so the manuscript does not depend on superseded
generation-comparison artifacts.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "evaluation" / "results"
PAPER_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PAPER_DIR / "artifacts"
FIGURES_DIR = PAPER_DIR / "figures"
SEEDS = (13, 29, 47)
SUPERSEDED_ASSETS = (
    ARTIFACTS_DIR / "tinyshakespeare_results.tex",
    ARTIFACTS_DIR / "enwik8_results.tex",
    FIGURES_DIR / "tinyshakespeare_bpb.pdf",
    FIGURES_DIR / "tinyshakespeare_bpb.png",
    FIGURES_DIR / "enwik8_bpb.pdf",
    FIGURES_DIR / "enwik8_bpb.png",
)


@dataclass(frozen=True)
class Family:
    key: str
    corpus: str
    tier: str
    label: str
    parameters: int
    prefix: str


FAMILIES = (
    Family("tiny_char_6", "Tiny Shakespeare", "6.5M", "Character GPT", 6_525_600, "char_gpt_6m_all_parameters_constant"),
    Family("tiny_atomic_6", "Tiny Shakespeare", "6.5M", "Atomic Lexical", 6_469_374, "atomic_lexical_6m"),
    Family("tiny_random_6", "Tiny Shakespeare", "6.5M", "Permuted Flat", 6_555_064, "random_flat_6m"),
    Family("tiny_flat_6", "Tiny Shakespeare", "6.5M", "Flat ILM", 6_555_064, "c_flat_6m"),
    Family("tiny_full_6", "Tiny Shakespeare", "6.5M", "Full ILM", 6_631_992, "c_full_6m"),
    Family("tiny_char_15", "Tiny Shakespeare", "15.5M", "Character GPT", 15_438_192, "char_gpt_15m"),
    Family("tiny_atomic_15", "Tiny Shakespeare", "15.5M", "Atomic Lexical", 15_537_630, "atomic_lexical_15m"),
    Family("tiny_flat_15", "Tiny Shakespeare", "15.5M", "Flat ILM", 15_483_532, "c_flat_15m"),
    Family("tiny_full_15", "Tiny Shakespeare", "15.5M", "Full ILM", 15_601_932, "c_full_15m"),
    Family("enwik_byte_6", "enwik8", "6.5M", "Byte GPT", 6_567_600, "enwik8_byte_gpt_6m_all_parameters_constant"),
    Family("enwik_random_6", "enwik8", "6.5M", "Permuted Flat", 6_561_064, "enwik8_lossless_s4_random_flat_6m"),
    Family("enwik_flat_6", "enwik8", "6.5M", "Flat ILM", 6_561_064, "enwik8_lossless_s4_c_flat_6m"),
    Family("enwik_full_6", "enwik8", "6.5M", "Full ILM", 6_676_456, "enwik8_lossless_s4_c_full_6m"),
    Family("enwik_byte_15", "enwik8", "15.5M", "Byte GPT", 15_502_872, "enwik8_byte_gpt_15m"),
    Family("enwik_flat_15", "enwik8", "15.5M", "Flat ILM", 15_492_772, "enwik8_lossless_s4_c_flat_15m"),
    Family("enwik_full_15", "enwik8", "15.5M", "Full ILM", 15_670_372, "enwik8_lossless_s4_c_full_15m"),
)

CROSSOVER_FAMILIES = (
    Family("tiny_char_ilm", "Tiny Shakespeare", "6.5M", "Character GPT", 6_525_600, "char_gpt_6m_all_parameters_constant"),
    Family("tiny_flat_ilm", "Tiny Shakespeare", "6.5M", "Flat ILM", 6_555_064, "c_flat_6m"),
    Family("tiny_char_nanogpt", "Tiny Shakespeare", "6.5M", "Character GPT", 6_525_600, "char_gpt_6m"),
    Family("tiny_flat_nanogpt", "Tiny Shakespeare", "6.5M", "Flat ILM", 6_555_064, "c_flat_6m_nanogpt_cosine"),
    Family("enwik_byte_ilm", "enwik8", "6.5M", "Byte GPT", 6_567_600, "enwik8_byte_gpt_6m_all_parameters_constant"),
    Family("enwik_flat_ilm", "enwik8", "6.5M", "Flat ILM", 6_561_064, "enwik8_lossless_s4_c_flat_6m"),
    Family("enwik_byte_nanogpt", "enwik8", "6.5M", "Byte GPT", 6_567_600, "enwik8_byte_gpt_6m"),
    Family("enwik_flat_nanogpt", "enwik8", "6.5M", "Flat ILM", 6_561_064, "enwik8_lossless_s4_c_flat_6m_nanogpt_cosine"),
)


def teacher_fields(report: dict[str, Any]) -> tuple[float, float, int]:
    teacher = report["teacher_forced"]
    bpb = float(teacher["bits_per_utf8_byte"])
    source_bytes = float(teacher.get("scored_source_bytes", teacher.get("scored_utf8_bytes")))
    events = int(teacher.get("coordinate_events", teacher.get("character_events", teacher.get("scored_events"))))
    return bpb, source_bytes, events


def load_rows(families: tuple[Family, ...] = FAMILIES) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in families:
        for seed in SEEDS:
            path = RESULTS_DIR / f"{family.prefix}_seed{seed}.test_metrics.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing expected report: {path}")
            report = json.loads(path.read_text())
            bpb, source_bytes, events = teacher_fields(report)
            rows.append(
                {
                    "family_key": family.key,
                    "corpus": family.corpus,
                    "tier": family.tier,
                    "family": family.label,
                    "seed": seed,
                    "bpb": bpb,
                    "parameters": family.parameters,
                    "evaluation_mode": report.get("evaluation_mode", report["teacher_forced"].get("evaluation_mode", "full-context")),
                    "scored_source_bytes": source_bytes,
                    "scored_events": events,
                    "report": str(path.relative_to(ROOT)),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]], families: tuple[Family, ...] = FAMILIES) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["family_key"]].append(row)

    summary: list[dict[str, Any]] = []
    order = {family.key: index for index, family in enumerate(families)}
    for key, group in grouped.items():
        bpbs = [row["bpb"] for row in sorted(group, key=lambda item: item["seed"])]
        first = group[0]
        summary.append(
            {
                "family_key": key,
                "corpus": first["corpus"],
                "tier": first["tier"],
                "family": first["family"],
                "parameters": first["parameters"],
                "mean_bpb": statistics.mean(bpbs),
                "sample_sd_bpb": statistics.stdev(bpbs),
                "seed13_bpb": bpbs[0],
                "seed29_bpb": bpbs[1],
                "seed47_bpb": bpbs[2],
                "evaluation_mode": first["evaluation_mode"],
            }
        )
    return sorted(summary, key=lambda item: order[item["family_key"]])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def format_bpb(row: dict[str, Any] | None) -> str:
    if row is None:
        return "--"
    return f"{row['mean_bpb']:.4f} $\\pm$ {row['sample_sd_bpb']:.4f}"


def write_scale_tex_table(path: Path, summary: list[dict[str, Any]], tier: str) -> None:
    by_key = {(row["corpus"], row["family"]): row for row in summary if row["tier"] == tier}
    if tier == "6.5M":
        tiny_families = ["Character GPT", "Atomic Lexical", "Permuted Flat", "Flat ILM", "Full ILM"]
        enwik_families = ["Byte GPT", None, "Permuted Flat", "Flat ILM", "Full ILM"]
    else:
        tiny_families = ["Atomic Lexical", "Flat ILM", "Full ILM"]
        enwik_families = [None, "Flat ILM", "Full ILM"]

    lines = [
        "\\begin{tabular}{lr|lr}",
        "\\toprule",
        "\\multicolumn{2}{c|}{Tiny Shakespeare} & \\multicolumn{2}{c}{enwik8} \\\\",
        "Family & BPB (mean $\\pm$ s.d.) & Family & BPB (mean $\\pm$ s.d.) \\\\",
        "\\midrule",
    ]
    for tiny, enwik in zip(tiny_families, enwik_families):
        tiny_row = by_key.get(("Tiny Shakespeare", tiny)) if tiny is not None else None
        enwik_row = by_key.get(("enwik8", enwik)) if enwik is not None else None
        lines.append(
            f"{tex_escape(tiny) if tiny else '--'} & {format_bpb(tiny_row)} & "
            f"{tex_escape(enwik) if enwik else '--'} & {format_bpb(enwik_row)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))


def write_crossover_tex_table(path: Path, summary: list[dict[str, Any]]) -> None:
    values = {row["family_key"]: row for row in summary}
    rows = (
        ("Tiny Shakespeare", "tiny_char_ilm", "tiny_flat_ilm", "tiny_char_nanogpt", "tiny_flat_nanogpt"),
        ("enwik8", "enwik_byte_ilm", "enwik_flat_ilm", "enwik_byte_nanogpt", "enwik_flat_nanogpt"),
    )
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "& \\multicolumn{4}{c}{\\textbf{Optimizer protocol}} \\\\",
        "\\cmidrule(lr){2-5}",
        "& \\multicolumn{2}{c}{ILM AdamW} & \\multicolumn{2}{c}{nanoGPT AdamW} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        "Corpus & Reference GPT & Flat ILM & Reference GPT & Flat ILM \\\\",
        "\\midrule",
    ]
    for corpus, ref_ilm, flat_ilm, ref_nanogpt, flat_nanogpt in rows:
        lines.append(
            f"{corpus} & {format_bpb(values[ref_ilm])} & {format_bpb(values[flat_ilm])} & "
            f"{format_bpb(values[ref_nanogpt])} & {format_bpb(values[flat_nanogpt])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))


def figure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, stem: str) -> None:
    figure.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_bpb_by_scale(rows: list[dict[str, Any]], tier: str, stem: str) -> None:
    family_order = {
        "6.5M": {
            "Tiny Shakespeare": ["Character GPT", "Atomic Lexical", "Permuted Flat", "Flat ILM", "Full ILM"],
            "enwik8": ["Byte GPT", "Permuted Flat", "Flat ILM", "Full ILM"],
        },
        "15.5M": {
            "Tiny Shakespeare": ["Atomic Lexical", "Flat ILM", "Full ILM"],
            "enwik8": ["Flat ILM", "Full ILM"],
        },
    }[tier]
    colors = {
        "Character GPT": "#6b7280",
        "Byte GPT": "#6b7280",
        "Atomic Lexical": "#b45309",
        "Permuted Flat": "#a855f7",
        "Flat ILM": "#2563eb",
        "Full ILM": "#059669",
    }
    selected_rows = [
        row
        for row in rows
        if row["tier"] == tier
        and row["corpus"] in family_order
        and row["family"] in family_order[row["corpus"]]
    ]
    lower = min(row["bpb"] for row in selected_rows)
    upper = max(row["bpb"] for row in selected_rows)
    padding = max((upper - lower) * 0.06, 0.01)

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.25), sharey=True)
    for ax, corpus in zip(axes, ("Tiny Shakespeare", "enwik8")):
        subset = [row for row in rows if row["corpus"] == corpus and row["tier"] == tier]
        available = [family for family in family_order[corpus] if any(row["family"] == family for row in subset)]
        for x, family in enumerate(available):
            values = sorted(row["bpb"] for row in subset if row["family"] == family)
            ax.scatter([x - 0.10, x, x + 0.10], values, s=26, color=colors[family], zorder=3)
        ax.set_xticks(range(len(available)), available, rotation=24, ha="right")
        ax.set_title(corpus)
        ax.set_ylim(lower - padding, upper + padding)
        ax.grid(axis="y", color="#d1d5db", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Test BPB (lower is better)")
    fig.tight_layout(w_pad=1.0)
    save_figure(fig, stem)


def plot_permutation_control(rows: list[dict[str, Any]]) -> None:
    comparisons = (
        ("Tiny Shakespeare", "Permuted Flat", "Flat ILM", "6.5M"),
        ("enwik8", "Permuted Flat", "Flat ILM", "6.5M"),
    )
    colors = {"Permuted Flat": "#a855f7", "Flat ILM": "#2563eb"}
    selected_rows = [
        row
        for row in rows
        if row["tier"] == "6.5M" and row["family"] in {"Permuted Flat", "Flat ILM"}
    ]
    lower = min(row["bpb"] for row in selected_rows)
    upper = max(row["bpb"] for row in selected_rows)
    padding = max((upper - lower) * 0.06, 0.01)

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.15), sharey=True)
    for ax, (corpus, control, semantic, tier) in zip(axes, comparisons):
        subset = [row for row in rows if row["corpus"] == corpus and row["tier"] == tier]
        by_key = {(row["family"], row["seed"]): row["bpb"] for row in subset}
        for seed in SEEDS:
            values = [by_key[(control, seed)], by_key[(semantic, seed)]]
            ax.plot([0, 1], values, color="#9ca3af", linewidth=1, zorder=1)
            ax.scatter([0, 1], values, color=[colors[control], colors[semantic]], s=32, zorder=2)
        control_values = [by_key[(control, seed)] for seed in SEEDS]
        semantic_values = [by_key[(semantic, seed)] for seed in SEEDS]
        delta = statistics.mean(control_values) - statistics.mean(semantic_values)
        ax.text(
            0.5,
            0.04,
            f"$\\Delta$ = {delta:.3f} BPB",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
        )
        ax.set_xticks([0, 1], [control, semantic])
        ax.set_title(corpus)
        ax.set_ylim(lower - padding, upper + padding)
        ax.grid(axis="y", color="#d1d5db", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.35, 1.35)
    axes[0].set_ylabel("Test BPB (lower is better)")
    fig.tight_layout(w_pad=1.1)
    save_figure(fig, "semantic_code_permutation_control")


def write_readme(summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Paper Artifacts",
        "",
        "`build_artifacts.py` reads the completed seed-level JSON reports in",
        "`experiments/evaluation/results/` and regenerates every file in this directory",
        "and in `../figures/`.",
        "",
        "- `seed_bpb.csv` contains one held-out BPB result per model seed.",
        "- `bpb_summary.csv` contains the paper means and sample standard deviations.",
        "- `crossover_summary.csv` and `optimizer_crossover.tex` record the 6.5M optimizer crossover.",
        "- `results_6m.tex` and `results_15m.tex` are the corpus-paired result tables included by `main.tex`.",
        "- `pairwise_deltas.csv` records the planned contrasts used in the text.",
        "",
        "The current study has three independently trained seeds: 13, 29, and 47.",
        "Parameter counts are the exact total trainable counts reported in",
        "`experiments/RESULTS.md`.",
        "",
        "## Current Rows",
        "",
        "| Corpus | Tier | Family | Mean BPB | Sample SD |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    displayed_keys = {
        "tiny_char_6", "tiny_atomic_6", "tiny_random_6", "tiny_flat_6", "tiny_full_6",
        "enwik_byte_6", "enwik_random_6", "enwik_flat_6", "enwik_full_6",
        "tiny_atomic_15", "tiny_flat_15", "tiny_full_15",
        "enwik_flat_15", "enwik_full_15",
    }
    for row in summary:
        if row["family_key"] not in displayed_keys:
            continue
        lines.append(
            f"| {row['corpus']} | {row['tier']} | {row['family']} | "
            f"{row['mean_bpb']:.6f} | {row['sample_sd_bpb']:.6f} |"
        )
    (ARTIFACTS_DIR / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for asset in SUPERSEDED_ASSETS:
        asset.unlink(missing_ok=True)
    figure_style()
    rows = load_rows()
    summary = summarize(rows)
    write_csv(ARTIFACTS_DIR / "seed_bpb.csv", rows)
    write_csv(ARTIFACTS_DIR / "bpb_summary.csv", summary)

    summary_by_key = {row["family_key"]: row for row in summary}
    pair_specs = (
        ("Tiny Shakespeare", "6.5M", "Flat ILM", "Full ILM", "tiny_flat_6", "tiny_full_6"),
        ("Tiny Shakespeare", "6.5M", "Permuted Flat", "Flat ILM", "tiny_random_6", "tiny_flat_6"),
        ("Tiny Shakespeare", "15.5M", "Flat ILM", "Full ILM", "tiny_flat_15", "tiny_full_15"),
        ("enwik8", "6.5M", "Flat ILM", "Full ILM", "enwik_flat_6", "enwik_full_6"),
        ("enwik8", "6.5M", "Permuted Flat", "Flat ILM", "enwik_random_6", "enwik_flat_6"),
        ("enwik8", "15.5M", "Flat ILM", "Full ILM", "enwik_flat_15", "enwik_full_15"),
    )
    deltas = []
    for corpus, tier, left_label, right_label, left_key, right_key in pair_specs:
        left = summary_by_key[left_key]["mean_bpb"]
        right = summary_by_key[right_key]["mean_bpb"]
        deltas.append(
            {
                "corpus": corpus,
                "tier": tier,
                "higher_bpb_family": left_label,
                "lower_bpb_family": right_label,
                "mean_bpb_difference": left - right,
            }
        )
    write_csv(ARTIFACTS_DIR / "pairwise_deltas.csv", deltas)
    crossover_rows = load_rows(CROSSOVER_FAMILIES)
    crossover_summary = summarize(crossover_rows, CROSSOVER_FAMILIES)
    write_csv(ARTIFACTS_DIR / "crossover_seed_bpb.csv", crossover_rows)
    write_csv(ARTIFACTS_DIR / "crossover_summary.csv", crossover_summary)
    write_scale_tex_table(ARTIFACTS_DIR / "results_6m.tex", summary, "6.5M")
    write_scale_tex_table(ARTIFACTS_DIR / "results_15m.tex", summary, "15.5M")
    write_crossover_tex_table(ARTIFACTS_DIR / "optimizer_crossover.tex", crossover_summary)
    write_readme(summary)
    plot_bpb_by_scale(rows, "6.5M", "bpb_6m_by_corpus")
    plot_bpb_by_scale(rows, "15.5M", "bpb_15m_by_corpus")
    plot_permutation_control(rows)


if __name__ == "__main__":
    main()
