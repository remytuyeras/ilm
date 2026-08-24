#!/usr/bin/env python3
"""Build seed-level BPB assets for completed 100M enwik8 ILM runs."""

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


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "evaluation" / "results"
PAPER_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PAPER_DIR / "artifacts"
FIGURES_DIR = PAPER_DIR / "figures"
SEED_PATTERN = re.compile(r"_seed(\d+)\.test_metrics\.json$")


@dataclass(frozen=True)
class Family:
    label: str
    prefix: str
    parameters: int
    color: str


FAMILIES = (
    Family("Flat ILM", "enwik8_lossless_s4_c_flat_100m_cosine", 99_891_856, "#2563eb"),
    Family("Full ILM", "enwik8_lossless_s4_c_full_100m_cosine", 100_343_632, "#059669"),
)


def completed_paths(prefix: str) -> list[tuple[int, Path]]:
    paths = []
    for path in RESULTS_DIR.glob(f"{prefix}_seed*.test_metrics.json"):
        match = SEED_PATTERN.search(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    return sorted(paths)


def load_rows() -> list[dict[str, object]]:
    rows = []
    for family in FAMILIES:
        for seed, path in completed_paths(family.prefix):
            report = json.loads(path.read_text(encoding="utf-8"))
            teacher = report["teacher_forced"]
            rows.append(
                {
                    "family": family.label,
                    "seed": seed,
                    "parameters": family.parameters,
                    "bpb": float(teacher["bits_per_utf8_byte"]),
                    "evaluation_mode": report.get("evaluation_mode", teacher.get("evaluation_mode")),
                    "report": str(path.relative_to(ROOT)),
                }
            )
    if not rows:
        raise FileNotFoundError("no completed 100M cosine test-metric reports found")
    return rows


def centered_offsets(count: int, spacing: float = 0.14) -> list[float]:
    return [(index - (count - 1) / 2) * spacing for index in range(count)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    summaries = []
    for family in FAMILIES:
        values = sorted(float(row["bpb"]) for row in rows if row["family"] == family.label)
        if not values:
            continue
        summaries.append(
            {
                "family": family.label,
                "parameters": family.parameters,
                "completed_seeds": len(values),
                "mean_bpb": statistics.mean(values),
                "sample_sd_bpb": statistics.stdev(values) if len(values) > 1 else "",
            }
        )
    write_csv(ARTIFACTS_DIR / "large_scale_100m_seed_bpb.csv", rows)
    write_csv(ARTIFACTS_DIR / "large_scale_100m_summary.csv", summaries)

    figure, axis = plt.subplots(figsize=(3.7, 2.65))
    values_by_family = {
        family.label: sorted(float(row["bpb"]) for row in rows if row["family"] == family.label)
        for family in FAMILIES
    }
    all_values = [value for values in values_by_family.values() for value in values]
    lower, upper = min(all_values), max(all_values)
    padding = max((upper - lower) * 0.12, 0.01)

    for position, family in enumerate(FAMILIES):
        values = values_by_family[family.label]
        if not values:
            continue
        axis.scatter(
            [position + offset for offset in centered_offsets(len(values))],
            values,
            s=30,
            color=family.color,
            edgecolor="#1f2937",
            linewidth=0.45,
            zorder=3,
        )
        axis.scatter(
            position,
            statistics.mean(values),
            marker="x",
            s=30,
            color="#111827",
            linewidth=1.1,
            zorder=4,
        )

    axis.set_title("enwik8, approximately 100M parameters")
    axis.set_ylabel("Test BPB (lower is better)")
    axis.set_xticks(range(len(FAMILIES)), [family.label for family in FAMILIES], rotation=18, ha="right")
    axis.set_xlim(-0.4, len(FAMILIES) - 0.6)
    axis.set_ylim(lower - padding, upper + padding)
    axis.grid(axis="y", color="#d1d5db", linewidth=0.7)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "large_scale_100m_bpb.pdf", bbox_inches="tight")
    figure.savefig(FIGURES_DIR / "large_scale_100m_bpb.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
