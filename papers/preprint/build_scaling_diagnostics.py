#!/usr/bin/env python3
"""Build fixed-budget capacity diagnostics for Flat and Full ILM.

These figures show outcomes at a common 6,000-update horizon, not fitted
scaling laws or convergence-adjusted scaling results.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "evaluation" / "results"
PAPER_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PAPER_DIR / "artifacts"
FIGURES_DIR = PAPER_DIR / "figures"
SEED_PATTERN = re.compile(r"_seed(\d+)\.test_metrics\.json$")
COLORS = {"Flat ILM": "#2563eb", "Full ILM": "#059669"}
CAPTIONS = r"""% Reusable captions for the fixed-budget capacity diagnostics.
\paragraph{Fixed-budget capacity diagnostic.}
Seed-level held-out BPB for Flat and Full \ilm{} at the reported parameter tiers.
Solid segments connect the 6.5M and 15.5M tiers trained under the shared
compact-model protocol. Dashed colored segments connect the 15.5M tier to the
approximately 100M enwik8 results, which use a distinct large-scale
cosine-decay optimization regime. They therefore indicate persistence across
regimes rather than a homogeneous scaling curve. Points show individual
model-training seeds, and black crosses show seed means. Blue and green encode
Flat and Full \ilm{}, respectively. The gray dashed sample in the enwik8 legend
encodes only the line-style convention for the 100M regime. It does not denote
a third model condition.

\paragraph{Full-minus-Flat diagnostic.}
Seed-paired $\Delta$ BPB, defined as Full minus Flat, across the completed
parameter tiers. Negative values favor Full \ilm{}. Solid and dashed segments
have the same compact-protocol and 100M-cosine-regime meanings as above.
"""


@dataclass(frozen=True)
class Condition:
    corpus: str
    tier: str
    parameters_m: float
    family: str
    prefix: str


CONDITIONS = (
    Condition("Tiny Shakespeare", "6.5M", 6.5, "Flat ILM", "c_flat_6m"),
    Condition("Tiny Shakespeare", "6.5M", 6.5, "Full ILM", "c_full_6m"),
    Condition("Tiny Shakespeare", "15.5M", 15.5, "Flat ILM", "c_flat_15m"),
    Condition("Tiny Shakespeare", "15.5M", 15.5, "Full ILM", "c_full_15m"),
    Condition("enwik8", "6.5M", 6.5, "Flat ILM", "enwik8_lossless_s4_c_flat_6m"),
    Condition("enwik8", "6.5M", 6.5, "Full ILM", "enwik8_lossless_s4_c_full_6m"),
    Condition("enwik8", "15.5M", 15.5, "Flat ILM", "enwik8_lossless_s4_c_flat_15m"),
    Condition("enwik8", "15.5M", 15.5, "Full ILM", "enwik8_lossless_s4_c_full_15m"),
    Condition("enwik8", "100M", 100.0, "Flat ILM", "enwik8_lossless_s4_c_flat_100m_cosine"),
    Condition("enwik8", "100M", 100.0, "Full ILM", "enwik8_lossless_s4_c_full_100m_cosine"),
)


def completed_paths(prefix: str) -> list[tuple[int, Path]]:
    paths = []
    for path in RESULTS_DIR.glob(f"{prefix}_seed*.test_metrics.json"):
        match = SEED_PATTERN.search(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    return sorted(paths)


def load_values(condition: Condition) -> dict[int, float]:
    values = {}
    for seed, path in completed_paths(condition.prefix):
        report = json.loads(path.read_text(encoding="utf-8"))
        values[seed] = float(report["teacher_forced"]["bits_per_utf8_byte"])
    if not values:
        raise FileNotFoundError(f"no completed reports for {condition.prefix}")
    return values


def centered_offsets(count: int, spacing: float = 0.018) -> list[float]:
    return [(index - (count - 1) / 2) * spacing for index in range(count)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_protocol_segments(
    axis: plt.Axes,
    mean_x: list[float],
    mean_y: list[float],
    *,
    color: str,
    label: str | None = None,
) -> None:
    """Use a dashed segment when the 100M cosine protocol begins."""
    compact = [(x, y) for x, y in zip(mean_x, mean_y) if x < 100]
    if len(compact) >= 2:
        axis.plot(
            [x for x, _ in compact],
            [y for _, y in compact],
            color=color,
            linewidth=1.4,
            label=label,
            zorder=2,
        )
    elif compact and label:
        axis.plot([], [], color=color, linewidth=1.4, label=label)

    large = [(x, y) for x, y in zip(mean_x, mean_y) if x >= 100]
    if compact and large:
        axis.plot(
            [compact[-1][0], large[0][0]],
            [compact[-1][1], large[0][1]],
            color=color,
            linewidth=1.4,
            linestyle="--",
            zorder=2,
        )


def plot_bpb(values_by_condition: dict[Condition, dict[int, float]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(6.4, 2.75), constrained_layout=True)
    for axis, corpus in zip(axes, ("Tiny Shakespeare", "enwik8")):
        for family in ("Flat ILM", "Full ILM"):
            family_conditions = [
                condition for condition in CONDITIONS if condition.corpus == corpus and condition.family == family
            ]
            family_conditions.sort(key=lambda condition: condition.parameters_m)
            mean_x, mean_y = [], []
            for condition in family_conditions:
                values = values_by_condition[condition]
                seeds = sorted(values)
                x_values = [condition.parameters_m * (10 ** offset) for offset in centered_offsets(len(seeds))]
                y_values = [values[seed] for seed in seeds]
                axis.scatter(
                    x_values,
                    y_values,
                    s=26,
                    color=COLORS[family],
                    edgecolor="#1f2937",
                    linewidth=0.45,
                    zorder=3,
                )
                average = statistics.mean(y_values)
                axis.scatter(
                    condition.parameters_m,
                    average,
                    marker="x",
                    s=30,
                    color="#111827",
                    linewidth=1.1,
                    zorder=5,
                )
                mean_x.append(condition.parameters_m)
                mean_y.append(average)
            plot_protocol_segments(axis, mean_x, mean_y, color=COLORS[family], label=family)

        sizes = sorted({condition.parameters_m for condition in CONDITIONS if condition.corpus == corpus})
        axis.set_xscale("log")
        axis.set_xticks(sizes)
        axis.set_xticklabels([f"{value:g}M" for value in sizes])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_title(corpus)
        axis.set_xlabel("Approximate trainable parameters")
        axis.grid(axis="y", color="#d1d5db", linewidth=0.7)
        axis.set_axisbelow(True)
        legend_handles = [
            Line2D([], [], color=COLORS["Flat ILM"], linewidth=1.4, label="Flat ILM"),
            Line2D([], [], color=COLORS["Full ILM"], linewidth=1.4, label="Full ILM"),
        ]
        if corpus == "enwik8":
            legend_handles.append(
                Line2D([], [], color="#4b5563", linewidth=1.2, linestyle="--", label="100M large-scale cosine regime")
            )
        if corpus == "enwik8":
            axis.legend(
                handles=legend_handles,
                frameon=False,
                fontsize=8,
                loc="lower left",
            )
        else:
            axis.legend(handles=legend_handles, frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylabel("Test BPB (lower is better)")
    figure.savefig(FIGURES_DIR / "scaling_diagnostic_bpb.pdf", bbox_inches="tight")
    figure.savefig(FIGURES_DIR / "scaling_diagnostic_bpb.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_delta(values_by_condition: dict[Condition, dict[int, float]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(6.4, 2.75), constrained_layout=True)
    for axis, corpus in zip(axes, ("Tiny Shakespeare", "enwik8")):
        conditions = sorted(
            {condition.parameters_m for condition in CONDITIONS if condition.corpus == corpus}
        )
        mean_x, mean_y = [], []
        for parameters_m in conditions:
            flat = next(condition for condition in CONDITIONS if condition.corpus == corpus and condition.parameters_m == parameters_m and condition.family == "Flat ILM")
            full = next(condition for condition in CONDITIONS if condition.corpus == corpus and condition.parameters_m == parameters_m and condition.family == "Full ILM")
            shared_seeds = sorted(set(values_by_condition[flat]) & set(values_by_condition[full]))
            deltas = [values_by_condition[full][seed] - values_by_condition[flat][seed] for seed in shared_seeds]
            x_values = [parameters_m * (10 ** offset) for offset in centered_offsets(len(deltas))]
            axis.scatter(x_values, deltas, s=26, color="#7c3aed", edgecolor="#1f2937", linewidth=0.45, zorder=3)
            average = statistics.mean(deltas)
            axis.scatter(parameters_m, average, marker="x", s=30, color="#111827", linewidth=1.1, zorder=5)
            mean_x.append(parameters_m)
            mean_y.append(average)
        plot_protocol_segments(axis, mean_x, mean_y, color="#7c3aed")
        axis.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--")
        axis.set_xscale("log")
        axis.set_xticks(conditions)
        axis.set_xticklabels([f"{value:g}M" for value in conditions])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_title(corpus)
        axis.set_xlabel("Approximate trainable parameters")
        axis.grid(axis="y", color="#d1d5db", linewidth=0.7)
        axis.set_axisbelow(True)
        if corpus == "enwik8":
            axis.legend(
                handles=[
                    Line2D([], [], color="#7c3aed", linewidth=1.4, label="shared compact protocol"),
                    Line2D([], [], color="#4b5563", linewidth=1.2, linestyle="--", label="100M large-scale cosine regime"),
                ],
                frameon=False,
                fontsize=8,
                loc="upper right",
            )
    axes[0].set_ylabel(r"$\Delta$ test BPB (Full minus Flat)" "\n(negative favors Full)")
    figure.savefig(FIGURES_DIR / "scaling_diagnostic_delta.pdf", bbox_inches="tight")
    figure.savefig(FIGURES_DIR / "scaling_diagnostic_delta.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    values_by_condition = {condition: load_values(condition) for condition in CONDITIONS}
    seed_rows = []
    summary_rows = []
    for condition, values in values_by_condition.items():
        for seed, bpb in sorted(values.items()):
            seed_rows.append(
                {
                    "corpus": condition.corpus,
                    "tier": condition.tier,
                    "parameters_m": condition.parameters_m,
                    "family": condition.family,
                    "seed": seed,
                    "bpb": bpb,
                }
            )
        bpb_values = list(values.values())
        summary_rows.append(
            {
                "corpus": condition.corpus,
                "tier": condition.tier,
                "parameters_m": condition.parameters_m,
                "family": condition.family,
                "completed_seeds": len(bpb_values),
                "mean_bpb": statistics.mean(bpb_values),
                "sample_sd_bpb": statistics.stdev(bpb_values) if len(bpb_values) > 1 else "",
            }
        )
    write_csv(ARTIFACTS_DIR / "scaling_diagnostic_seed_bpb.csv", seed_rows)
    write_csv(ARTIFACTS_DIR / "scaling_diagnostic_summary.csv", summary_rows)
    (ARTIFACTS_DIR / "scaling_diagnostic_captions.tex").write_text(CAPTIONS, encoding="utf-8")
    plot_bpb(values_by_condition)
    plot_delta(values_by_condition)


if __name__ == "__main__":
    main()
