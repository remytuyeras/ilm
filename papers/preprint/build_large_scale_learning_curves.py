#!/usr/bin/env python3
"""Plot archived native validation-objective trajectories for 100M ILM runs.

The Flat and Full objectives differ because Full masks left-edge targets.  The
plots therefore diagnose optimization within each family; BPB remains the
shared held-out comparison metric.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models" / "evaluation"
PAPER_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PAPER_DIR / "figures"
ARTIFACTS_DIR = PAPER_DIR / "artifacts"
SEEDS = (13, 29, 47)


@dataclass(frozen=True)
class Family:
    label: str
    prefix: str
    color: str


FAMILIES = (
    Family("Flat ILM", "enwik8_lossless_s4_c_flat_100m_cosine", "#2563eb"),
    Family("Full ILM", "enwik8_lossless_s4_c_full_100m_cosine", "#059669"),
)


def load_history(prefix: str, seed: int) -> list[dict[str, float | int]]:
    path = MODELS_DIR / f"{prefix}_seed{seed}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    history = data["training_curriculum"][-1]["losses"]["history"]
    return [
        {"step": int(point["step"]), "validation_loss": float(point["validation_loss"])}
        for point in history
    ]


def mean_series(series: list[list[float]]) -> list[float]:
    return [sum(values) / len(values) for values in zip(*series)]


def plot_axis(
    axis: plt.Axes,
    histories: dict[str, list[list[dict[str, float | int]]]],
    *,
    start_step: int,
) -> None:
    for family in FAMILIES:
        runs = histories[family.label]
        steps = [int(point["step"]) for point in runs[0]]
        start = next(index for index, step in enumerate(steps) if step >= start_step)
        selected_steps = steps[start:]
        trajectories = [
            [float(point["validation_loss"]) for point in run[start:]]
            for run in runs
        ]
        for values in trajectories:
            axis.plot(selected_steps, values, color=family.color, alpha=0.24, linewidth=0.9)
        average = mean_series(trajectories)
        axis.plot(selected_steps, average, color=family.color, linewidth=2.0, label=family.label)
        axis.scatter(
            [selected_steps[-1]],
            [average[-1]],
            color=family.color,
            edgecolor="#1f2937",
            linewidth=0.45,
            s=22,
            zorder=3,
        )
    axis.set_xlim(start_step, 6000)
    axis.set_xlabel("Optimization step")
    axis.grid(axis="y", color="#d1d5db", linewidth=0.7)
    axis.legend(frameon=False, fontsize=8, loc="upper right")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    histories = {
        family.label: [load_history(family.prefix, seed) for seed in SEEDS]
        for family in FAMILIES
    }

    archive = {
        "description": (
            "Native validation-objective histories for the reported 100M enwik8 "
            "Flat and Full ILM runs. Absolute values are not cross-family comparable."
        ),
        "model_training_seeds": list(SEEDS),
        "series": {
            f"{family.prefix}_seed{seed}": {"history": history}
            for family in FAMILIES
            for seed, history in zip(SEEDS, histories[family.label])
        },
    }
    (ARTIFACTS_DIR / "large_scale_validation_loss_histories.json").write_text(
        json.dumps(archive, indent=2) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, float | int | str]] = []
    for family in FAMILIES:
        for seed, history in zip(SEEDS, histories[family.label]):
            rows.append(
                {
                    "family": family.label,
                    "seed": seed,
                    "validation_loss_step_4000": next(
                        float(point["validation_loss"]) for point in history if point["step"] == 4000
                    ),
                    "validation_loss_step_6000": float(history[-1]["validation_loss"]),
                }
            )
    with (ARTIFACTS_DIR / "large_scale_validation_loss_tail_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(6.4, 2.7))
    figure.subplots_adjust(left=0.16, right=0.98, bottom=0.22, top=0.84)
    plot_axis(axis, histories, start_step=0)
    axis.set_title("enwik8, 100M parameter tier", fontsize=10)
    axis.set_ylabel("Native validation objective (nats/event)", fontsize=8, labelpad=8)
    figure.savefig(FIGURES_DIR / "large_scale_validation_loss_trajectories.pdf", bbox_inches="tight")
    figure.savefig(
        FIGURES_DIR / "large_scale_validation_loss_trajectories.png", dpi=220, bbox_inches="tight"
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 2.7))
    figure.subplots_adjust(left=0.16, right=0.98, bottom=0.22, top=0.84)
    plot_axis(axis, histories, start_step=4000)
    axis.set_title("enwik8, 100M parameter tier: final 2,000 updates", fontsize=10)
    axis.set_ylabel("Native validation objective (nats/event)", fontsize=8, labelpad=8)
    figure.savefig(FIGURES_DIR / "large_scale_validation_loss_tail.pdf", bbox_inches="tight")
    figure.savefig(FIGURES_DIR / "large_scale_validation_loss_tail.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
