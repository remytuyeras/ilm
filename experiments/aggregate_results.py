"""Aggregate ILM generation and teacher-forced evaluation reports."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib
import numpy as np

# Result aggregation is a noninteractive command and must work in CI/headless shells.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


GENERATION_METRICS = (
    "word_count",
    "invalid_code_count",
    "max_repeated_bigram",
    "max_repeated_trigram",
    "repeated_bigram_fraction",
    "repeated_trigram_fraction",
)


def expand_patterns(patterns: Sequence[str]) -> List[Path]:
    paths = []
    for pattern in patterns:
        matches = [Path(path) for path in glob.glob(pattern)]
        paths.extend(matches or [Path(pattern)])
    return [Path(path) for path in sorted({str(path) for path in paths})]


def bootstrap_interval(values: Sequence[float], seed: int, samples: int = 2000) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return float(array[0]), float(array[0])
    generator = np.random.default_rng(seed)
    draws = generator.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def model_label(record: Dict[str, Any]) -> str:
    config = record.get("config", {})
    if record.get("backend") == "ilm":
        return str(config.get("ilm_model_path", "ilm"))
    return str(config.get("resolved_hf_model") or config.get("hf_reference") or "huggingface")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generation_rows(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data.get("records", []) if isinstance(data, dict) else data
        for record in records:
            if record.get("error"):
                continue
            metrics = record.get("metrics", {})
            row = {
                "source_report": str(path),
                "backend": record.get("backend"),
                "model": model_label(record),
                "prompt": record.get("prompt"),
                "sample_index": record.get("sample_index", 0),
                "generation_seed": record.get("generation_seed"),
                "elapsed_seconds": record.get("elapsed_seconds"),
                **metrics,
            }
            rows.append(row)
    return rows


def summarize_rows(rows: Sequence[Dict[str, Any]], metrics: Sequence[str], seed: int) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["backend"]), str(row["model"]))].append(row)

    summary = []
    for group_index, ((backend, model), group_rows) in enumerate(sorted(grouped.items())):
        for metric_index, metric in enumerate(metrics):
            values = [float(row[metric]) for row in group_rows if metric in row]
            if not values:
                continue
            lower, upper = bootstrap_interval(values, seed + 1009 * group_index + metric_index)
            summary.append({
                "backend": backend,
                "model": model,
                "metric": metric,
                "n": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci95_low": lower,
                "ci95_high": upper,
            })
    return summary


def test_rows(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get("teacher_forced", {})
        if not metrics:
            continue
        rows.append({
            "source_report": str(path),
            "model": data.get("model_path"),
            "seed": data.get("seed"),
            "test_text_sha256": data.get("test_text_sha256"),
            "oov_rate": data.get("encoding", {}).get("oov_rate"),
            **metrics,
        })
    return rows


def plot_metric(summary: Sequence[Dict[str, Any]], metric: str, output_path: Path, title: str) -> None:
    selected = [row for row in summary if row["metric"] == metric]
    if not selected:
        return
    labels = [Path(str(row["model"])).name for row in selected]
    means = [row["mean"] for row in selected]
    errors = [
        [row["mean"] - row["ci95_low"] for row in selected],
        [row["ci95_high"] - row["mean"] for row in selected],
    ]
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 1.6), 4.5))
    axis.errorbar(range(len(labels)), means, yerr=errors, fmt="o", capsize=5)
    axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axis.set_ylabel(metric)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate reproducible ILM result reports.")
    parser.add_argument("--generation-report", action="append", default=[])
    parser.add_argument("--evaluation-report", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generation_paths = expand_patterns(args.generation_report)
    evaluation_paths = expand_patterns(args.evaluation_report)
    if not generation_paths and not evaluation_paths:
        raise SystemExit("provide at least one --generation-report or --evaluation-report")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_lines = ["# Aggregated Results", ""]

    if generation_paths:
        rows = generation_rows(generation_paths)
        summary = summarize_rows(rows, GENERATION_METRICS, args.bootstrap_seed)
        write_csv(output_dir / "generation_records.csv", rows)
        write_csv(output_dir / "generation_summary.csv", summary)
        plot_metric(
            summary,
            "max_repeated_bigram",
            output_dir / "generation_max_repeated_bigram.pdf",
            "Generation Repetition with 95% Bootstrap Intervals",
        )
        markdown_lines.extend([
            "## Generation", "",
            f"Records: {len(rows)}", "",
            "| Backend | Model | Metric | N | Mean | 95% CI |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ])
        for row in summary:
            markdown_lines.append(
                f"| {row['backend']} | {row['model']} | {row['metric']} | {row['n']} | "
                f"{row['mean']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |"
            )
        markdown_lines.append("")

    if evaluation_paths:
        rows = test_rows(evaluation_paths)
        grouped = [
            {**row, "backend": "ilm"}
            for row in rows
        ]
        summary = summarize_rows(grouped, ["bits_per_utf8_byte"], args.bootstrap_seed)
        write_csv(output_dir / "test_metrics.csv", rows)
        write_csv(output_dir / "test_bpb_summary.csv", summary)
        plot_metric(summary, "bits_per_utf8_byte", output_dir / "test_bpb.pdf", "Teacher-Forced Test BPB")
        markdown_lines.extend([
            "## Teacher-Forced Test Likelihood", "",
            "| Model | N | Mean BPB | 95% CI |",
            "| --- | ---: | ---: | --- |",
        ])
        for row in summary:
            markdown_lines.append(
                f"| {row['model']} | {row['n']} | {row['mean']:.6f} | "
                f"[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] |"
            )
        markdown_lines.append("")

    report_path = output_dir / "summary.md"
    report_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    print(f"Wrote aggregate results to {output_dir}")


if __name__ == "__main__":
    main()
