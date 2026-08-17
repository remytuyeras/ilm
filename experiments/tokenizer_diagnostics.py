"""Generate sidecars and plots from a frozen embedding-cluster tokenizer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(1, ROOT_DIR)

from ilm.tokenizer.core import force_json_extension, load_dictionary
from ilm.tokenizer.embedding_cluster import (
    label_centroids,
    plot_residual_levels_2d,
    plot_residual_levels_3d,
    reconstruct_frozen_mapping_diagnostics,
    save_semantic_spelling_mapping,
)


def default_semantic_spelling_path(tokenizer_json: str) -> str:
    path = Path(force_json_extension(tokenizer_json))
    return str(path.with_suffix(".semantic.json"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create diagnostics from a frozen tokenizer without rebuilding its mapping."
    )
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument("--cache-file", required=True)
    parser.add_argument("--semantic-spelling-file", nargs="?", const="auto", default=None)
    parser.add_argument("--centroid-label-method", choices=["closest-token", "llm"], default="closest-token")
    parser.add_argument("--centroid-label-model", default="gpt-5.6-terra")
    parser.add_argument("--centroid-label-concurrency", type=int, default=8)
    parser.add_argument("--centroid-label-examples", type=int, default=20)
    parser.add_argument("--centroid-label-max-output-tokens", type=int, default=4096)
    parser.add_argument("--plot-pca-3d", action="store_true")
    parser.add_argument("--plot-pca-2d", action="store_true")
    parser.add_argument("--plot-clusters", action="store_true")
    parser.add_argument("--no-plot-centroid-labels", action="store_true")
    parser.add_argument("--plot-sample-size", type=int, default=20000)
    parser.add_argument("--plot-output-dir", default=None)
    parser.add_argument("--no-show-plots", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.plot_pca_3d and not args.plot_pca_2d and args.semantic_spelling_file is None:
        raise SystemExit("provide --semantic-spelling-file, --plot-pca-2d, and/or --plot-pca-3d")

    tokenizer_mapping = load_dictionary(args.tokenizer_json)
    direct_mapping = tokenizer_mapping["direct"]
    metadata = tokenizer_mapping.get("metadata", {})
    if metadata.get("method") != "embedding-cluster":
        raise SystemExit("tokenizer diagnostics require an embedding-cluster tokenizer JSON")

    cached = np.load(args.cache_file, allow_pickle=False)
    tokens = cached["tokens"].tolist()
    embeddings = cached["embeddings"].astype(np.float32)
    embedding_model = str(cached["model"])
    if embedding_model != metadata.get("embedding_model"):
        raise SystemExit(
            f"cache uses {embedding_model!r}, tokenizer metadata requires "
            f"{metadata.get('embedding_model')!r}"
        )

    levels = reconstruct_frozen_mapping_diagnostics(
        tokens=tokens,
        embeddings=embeddings,
        direct_mapping=direct_mapping,
        reduced_dim=int(metadata["reduced_dim"]),
        normalize=bool(metadata["normalize"]),
        depth=int(metadata["depth"]),
        cluster_method=str(metadata["cluster_method"]),
        base=int(metadata.get("base", 64)),
        random_state=int(metadata.get("random_state", args.random_state)),
    )
    label_centroids(
        tokens=tokens,
        levels=levels,
        method=args.centroid_label_method,
        model=args.centroid_label_model,
        concurrency=args.centroid_label_concurrency,
        examples_per_cluster=args.centroid_label_examples,
        max_output_tokens=args.centroid_label_max_output_tokens,
        random_state=args.random_state,
    )

    semantic_spelling_file = args.semantic_spelling_file
    if semantic_spelling_file == "auto":
        semantic_spelling_file = default_semantic_spelling_path(args.tokenizer_json)
    if semantic_spelling_file is not None:
        save_semantic_spelling_mapping(
            tokens=tokens,
            levels=levels,
            filename=semantic_spelling_file,
            token_order=direct_mapping.keys(),
        )
        print(f"Semantic spelling saved to {force_json_extension(semantic_spelling_file)}")

    if args.plot_pca_3d:
        plot_residual_levels_3d(
            tokens=tokens,
            levels=levels,
            color_clusters=args.plot_clusters,
            label_centroids=not args.no_plot_centroid_labels,
            sample_size=args.plot_sample_size,
            random_state=args.random_state,
            output_dir=args.plot_output_dir,
            show=not args.no_show_plots,
        )
    if args.plot_pca_2d:
        plot_residual_levels_2d(
            tokens=tokens,
            levels=levels,
            color_clusters=args.plot_clusters,
            label_centroids=not args.no_plot_centroid_labels,
            sample_size=args.plot_sample_size,
            random_state=args.random_state,
            output_dir=args.plot_output_dir,
            show=not args.no_show_plots,
        )


if __name__ == "__main__":
    main()
