"""
Tokenizer SDK quickstart playground.

Default load test:
    python tests/quickstart.py

Load a specific tokenizer:
    python tests/quickstart.py --target-file data/tokenizer_v2.json

Build the original relative-position tokenizer:
    python tests/quickstart.py --mode build \
        --method relative-position \
        --source-file data/training_old_english.txt \
        --target-file data/tokenizer_v2.json

Build the embedding-cluster tokenizer:
    python tests/quickstart.py --mode build \
        --method embedding-cluster \
        --source-file data/training_old_english.txt \
        --target-file data/tokenizer_embedding_cluster_v1.json \
        --cluster-method spherical-kmeans \
        --reduced-dim 10 \
        --embedding-batch-size 512

Embedding-cluster builds call the OpenAI embeddings API. Put OPENAI_API_KEY in
`.env` before running, or export it in your shell. The SDK caches embeddings next
to the target JSON by default, so later clustering experiments can reuse them.

Useful experiments:
    --reduced-dim 3
    --reduced-dim 8
    --reduced-dim 10
    --reduced-dim 16
    --reduced-dim 32
    --reduced-dim 64
    --max-tokens 5000
    --collision-report-limit 100
    --no-collision-report
    --semantic-spelling-file
    --centroid-label-method llm
    --centroid-label-model gpt-5.6-terra
    --plot-pca-3d
    --plot-clusters
"""

import argparse
import sys

sys.path.insert(1, "./")

from ilm.tokenizer import (
    TOKENIZER_METHODS,
    create_tokenizer,
    default_semantic_spelling_file,
    load_tokenizer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Try ILM tokenizer SDK methods.")
    parser.add_argument("--mode", choices=["load", "build"], default="load")
    parser.add_argument("--source-file", default="data/training_old_english.txt")
    parser.add_argument("--target-file", default="data/tokenizer_v2.json")
    parser.add_argument("--method", choices=TOKENIZER_METHODS, default="relative-position")
    parser.add_argument("--line-index", type=int, default=20)

    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--reduced-dim", type=int, default=10)
    parser.add_argument(
        "--cluster-method",
        choices=["kmeans", "spherical-kmeans", "cosine-kmeans"],
        default="spherical-kmeans",
    )
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--cache-file", default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--keep-token-spacing", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-collision-report", action="store_true")
    parser.add_argument("--collision-report-limit", type=int, default=None)
    parser.add_argument(
        "--semantic-spelling-file",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Write a human-readable JSON mapping each token to centroid labels. "
            "If no path is given, derive one from --target-file."
        ),
    )
    parser.add_argument(
        "--centroid-label-method",
        choices=["closest-token", "closest-word", "nearest-token", "llm"],
        default="closest-token",
    )
    parser.add_argument("--centroid-label-model", default="gpt-5.6-terra")
    parser.add_argument("--centroid-label-concurrency", type=int, default=8)
    parser.add_argument("--centroid-label-examples", type=int, default=20)
    parser.add_argument("--centroid-label-max-output-tokens", type=int, default=4096)
    parser.add_argument("--plot-pca-3d", action="store_true")
    parser.add_argument("--plot-clusters", action="store_true")
    parser.add_argument("--no-plot-centroid-labels", action="store_true")
    parser.add_argument("--plot-sample-size", type=int, default=20000)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "build":
        semantic_spelling_file = args.semantic_spelling_file
        if semantic_spelling_file == "auto":
            semantic_spelling_file = default_semantic_spelling_file(args.target_file)

        tokenizer, detokenizer = create_tokenizer(
            source_file=args.source_file,
            target_file=args.target_file,
            method=args.method,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            reduced_dim=args.reduced_dim,
            cluster_method=args.cluster_method,
            depth=args.depth,
            normalize=not args.no_normalize,
            cache_file=args.cache_file,
            refresh_cache=args.refresh_cache,
            strip_embedding_text=not args.keep_token_spacing,
            max_tokens=args.max_tokens,
            random_state=args.random_state,
            report_collisions=not args.no_collision_report,
            collision_report_limit=args.collision_report_limit,
            plot_pca_3d=args.plot_pca_3d,
            plot_clusters=args.plot_clusters,
            plot_centroid_labels=not args.no_plot_centroid_labels,
            plot_sample_size=args.plot_sample_size,
            semantic_spelling_file=semantic_spelling_file,
            centroid_label_method=args.centroid_label_method,
            centroid_label_model=args.centroid_label_model,
            centroid_label_concurrency=args.centroid_label_concurrency,
            centroid_label_examples=args.centroid_label_examples,
            centroid_label_max_output_tokens=args.centroid_label_max_output_tokens,
        )
    else:
        tokenizer, detokenizer = load_tokenizer(args.target_file)

    sample_line = ""
    with open(args.source_file, "r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index == args.line_index:
                sample_line = line
                break

    tokens = tokenizer(sample_line)
    print(tokens)
    print(detokenizer(tokens))


if __name__ == "__main__":
    main()
