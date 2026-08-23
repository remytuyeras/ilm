#!/usr/bin/env python3
"""Plot archived native validation-objective trajectories for the preprint.

Flat and Full ILM use different native training objectives: Full applies the
word-prefix target mask.  These curves are therefore within-family optimization
diagnostics, not a cross-family likelihood comparison.  Teacher-forced BPB is
the shared comparison metric.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models" / "evaluation"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
SEEDS = (13, 29, 47)


@dataclass(frozen=True)
class Family:
    corpus: str
    tier: str
    label: str
    prefix: str
    color: str


FAMILIES = (
    Family("Tiny Shakespeare", "6.5M", "Flat ILM", "c_flat_6m", "#2563eb"),
    Family("Tiny Shakespeare", "6.5M", "Full ILM", "c_full_6m", "#059669"),
    Family("Tiny Shakespeare", "15.5M", "Flat ILM", "c_flat_15m", "#2563eb"),
    Family("Tiny Shakespeare", "15.5M", "Full ILM", "c_full_15m", "#059669"),
    Family("enwik8", "6.5M", "Flat ILM", "enwik8_lossless_s4_c_flat_6m", "#2563eb"),
    Family("enwik8", "6.5M", "Full ILM", "enwik8_lossless_s4_c_full_6m", "#059669"),
    Family("enwik8", "15.5M", "Flat ILM", "enwik8_lossless_s4_c_flat_15m", "#2563eb"),
    Family("enwik8", "15.5M", "Full ILM", "enwik8_lossless_s4_c_full_15m", "#059669"),
)


def load_history(prefix: str, seed: int) -> tuple[list[int], list[float]]:
    path = MODELS_DIR / f"{prefix}_seed{seed}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    history = data["training_curriculum"][-1]["losses"]["history"]
    return (
        [int(point["step"]) for point in history],
        [float(point["validation_loss"]) for point in history],
    )


def mean_series(series: list[list[float]]) -> list[float]:
    return [sum(values) / len(values) for values in zip(*series)]


def write_tail_summary(rows: list[dict[str, float | str]]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "validation_loss_tail_summary.csv"
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    panels = {
        ("Tiny Shakespeare", "6.5M"): (0, 0),
        ("enwik8", "6.5M"): (0, 1),
        ("Tiny Shakespeare", "15.5M"): (1, 0),
        ("enwik8", "15.5M"): (1, 1),
    }
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), constrained_layout=True)
    summary_rows: list[dict[str, float | str]] = []
    series_by_family: dict[Family, tuple[list[int], list[list[float]]]] = {}

    for family in FAMILIES:
        axis = axes[panels[(family.corpus, family.tier)]]
        trajectories = []
        steps = None
        for seed in SEEDS:
            current_steps, losses = load_history(family.prefix, seed)
            steps = current_steps
            trajectories.append(losses)
            axis.plot(current_steps, losses, color=family.color, alpha=0.22, linewidth=0.9)

        assert steps is not None
        series_by_family[family] = (steps, trajectories)
        average = mean_series(trajectories)
        axis.plot(steps, average, color=family.color, linewidth=2.0, label=family.label)
        axis.scatter([steps[-1]], [average[-1]], color=family.color, edgecolor="#1f2937", linewidth=0.45, s=22, zorder=3)
        axis.set_title(f"{family.corpus}, {family.tier}", fontsize=10)
        axis.set_xlim(0, steps[-1])
        axis.grid(axis="y", color="#d1d5db", linewidth=0.7)
        summary_rows.append(
            {
                "corpus": family.corpus,
                "tier": family.tier,
                "family": family.label,
                "mean_validation_loss_step_5600": mean(losses[-3] for losses in trajectories),
                "mean_validation_loss_step_6000": average[-1],
                "mean_change_5600_to_6000": average[-1] - mean(losses[-3] for losses in trajectories),
            }
        )

    for row in axes:
        for axis in row:
            axis.set_xlabel("Optimization step")
            axis.legend(frameon=False, fontsize=8, loc="upper right")
    figure.supylabel("Native validation objective (nats/selected event)", fontsize=9)

    figure.savefig(FIGURES_DIR / "validation_loss_trajectories.pdf", bbox_inches="tight")
    figure.savefig(FIGURES_DIR / "validation_loss_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    tail_figure, tail_axes = plt.subplots(2, 2, figsize=(7.0, 5.0), constrained_layout=True)
    for family in FAMILIES:
        axis = tail_axes[panels[(family.corpus, family.tier)]]
        steps, trajectories = series_by_family[family]
        tail_start = next(index for index, step in enumerate(steps) if step >= 4000)
        tail_steps = steps[tail_start:]
        tail_trajectories = [losses[tail_start:] for losses in trajectories]
        for losses in tail_trajectories:
            axis.plot(tail_steps, losses, color=family.color, alpha=0.25, linewidth=0.9)
        average = mean_series(tail_trajectories)
        axis.plot(tail_steps, average, color=family.color, linewidth=2.0, label=family.label)
        axis.scatter([tail_steps[-1]], [average[-1]], color=family.color, edgecolor="#1f2937", linewidth=0.45, s=22, zorder=3)
        axis.set_title(f"{family.corpus}, {family.tier}", fontsize=10)
        axis.set_xlim(4000, tail_steps[-1])
        axis.grid(axis="y", color="#d1d5db", linewidth=0.7)

    for row in tail_axes:
        for axis in row:
            axis.set_xlabel("Optimization step")
            axis.legend(frameon=False, fontsize=8, loc="upper right")
    tail_figure.supylabel("Native validation objective (nats/selected event)", fontsize=9)

    tail_figure.savefig(FIGURES_DIR / "validation_loss_tail.png", dpi=220, bbox_inches="tight")
    tail_figure.savefig(FIGURES_DIR / "validation_loss_tail.pdf", bbox_inches="tight")
    plt.close(tail_figure)
    write_tail_summary(summary_rows)


if __name__ == "__main__":
    main()
