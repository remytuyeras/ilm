#!/usr/bin/env python3
"""Plot completed large-scale enwik8 ILM validation trajectories.

The plots use the same convention as ``build_learning_curves.py``: faint lines
are individual model-training seeds and the darker line is their arithmetic
mean. Flat and Full use different native training objectives, so the figure is
an optimization diagnostic rather than a cross-family likelihood comparison.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models" / "evaluation"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


@dataclass(frozen=True)
class Family:
    label: str
    prefix: str
    color: str


FAMILIES = (
    Family("Flat ILM", "enwik8_lossless_s4_c_flat_100m_cosine", "#2563eb"),
    Family("Full ILM", "enwik8_lossless_s4_c_full_100m_cosine", "#059669"),
)
SEED_PATTERN = re.compile(r"_seed(\d+)\.json$")


def completed_paths(prefix: str) -> list[tuple[int, Path]]:
    paths = []
    for path in MODELS_DIR.glob(f"{prefix}_seed*.json"):
        match = SEED_PATTERN.search(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    return sorted(paths)


def load_history(path: Path) -> tuple[list[int], list[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    history = data["training_curriculum"][-1]["losses"]["history"]
    return (
        [int(point["step"]) for point in history],
        [float(point["validation_loss"]) for point in history],
    )


def mean_series(series: list[list[float]]) -> list[float]:
    return [sum(values) / len(values) for values in zip(*series)]


def save_plot(path_stem: str, start_step: int) -> None:
    figure, axis = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    latest_step = None

    for family in FAMILIES:
        trajectories = []
        reference_steps = None
        for _, path in completed_paths(family.prefix):
            steps, losses = load_history(path)
            start = next(index for index, step in enumerate(steps) if step >= start_step)
            steps, losses = steps[start:], losses[start:]
            if reference_steps is None:
                reference_steps = steps
            elif steps != reference_steps:
                raise ValueError(f"inconsistent validation steps for {family.label}")
            trajectories.append(losses)
            axis.plot(steps, losses, color=family.color, alpha=0.22, linewidth=0.9)

        if not trajectories:
            continue

        assert reference_steps is not None
        average = mean_series(trajectories)
        axis.plot(reference_steps, average, color=family.color, linewidth=2.0, label=family.label)
        axis.scatter(
            [reference_steps[-1]],
            [average[-1]],
            color=family.color,
            edgecolor="#1f2937",
            linewidth=0.45,
            s=28,
            zorder=3,
        )
        latest_step = reference_steps[-1]

    if latest_step is None:
        raise FileNotFoundError("no completed 100M cosine histories found")

    axis.set_title("enwik8, approximately 100M parameters", fontsize=11)
    axis.set_xlim(start_step, latest_step)
    axis.set_xlabel("Optimization step")
    axis.set_ylabel("Native validation objective (nats/selected event)")
    axis.grid(axis="y", color="#d1d5db", linewidth=0.7)
    axis.legend(frameon=False, fontsize=9)
    figure.savefig(FIGURES_DIR / f"{path_stem}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES_DIR / f"{path_stem}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_plot("large_scale_validation_loss_trajectories", start_step=0)
    save_plot("large_scale_validation_loss_tail", start_step=4000)


if __name__ == "__main__":
    main()
