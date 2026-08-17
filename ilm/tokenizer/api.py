import os
from typing import Callable, List, Optional, Tuple

from .core import force_json_extension, load_tokenizer
from .relative_position import create_relative_position_tokenizer


TOKENIZER_METHODS = ("relative-position", "embedding-cluster")
DEFAULT_CENTROID_LABEL_MODEL = "gpt-5.6-terra"


def _normalize_method(method: str) -> str:
    method = method.replace("_", "-").lower()
    aliases = {
        "relative": "relative-position",
        "relative-position": "relative-position",
        "embedding": "embedding-cluster",
        "embedding-cluster": "embedding-cluster",
    }
    if method not in aliases:
        allowed = ", ".join(TOKENIZER_METHODS)
        raise ValueError(f"Unknown tokenizer method {method!r}. Choose one of: {allowed}.")
    return aliases[method]


def default_embedding_cache_file(target_file: str) -> str:
    json_path = force_json_extension(target_file)
    base, _ = os.path.splitext(json_path)
    return f"{base}.embeddings.npz"


def default_semantic_spelling_file(target_file: str) -> str:
    json_path = force_json_extension(target_file)
    base, _ = os.path.splitext(json_path)
    return f"{base}.semantic.json"


def create_tokenizer(
    source_file: str,
    target_file: Optional[str] = None,
    method: str = "relative-position",
    embedding_model: str = "text-embedding-3-small",
    embedding_batch_size: int = 512,
    reduced_dim: int = 10,
    cluster_method: str = "spherical-kmeans",
    depth: int = 3,
    normalize: bool = True,
    cache_file: Optional[str] = None,
    refresh_cache: bool = False,
    strip_embedding_text: bool = True,
    max_tokens: Optional[int] = None,
    random_state: int = 42,
    report_collisions: bool = False,
    collision_report_limit: Optional[int] = None,
    plot_pca_3d: bool = False,
    plot_pca_2d: bool = False,
    plot_clusters: bool = False,
    plot_centroid_labels: bool = True,
    plot_sample_size: Optional[int] = 20000,
    plot_output_dir: Optional[str] = None,
    show_plots: bool = True,
    semantic_spelling_file: Optional[str] = None,
    centroid_label_method: str = "closest-token",
    centroid_label_model: str = DEFAULT_CENTROID_LABEL_MODEL,
    centroid_label_concurrency: int = 8,
    centroid_label_examples: int = 20,
    centroid_label_max_output_tokens: int = 4096,
    lossless_tokenization: bool = False,
) -> Tuple[Callable[[str], List[Optional[str]]], Callable[[List[str]], List[Optional[str]]]]:
    """
    Create a tokenizer using one of ILM's tokenizer-building methods.

    method="relative-position" keeps the original ILM builder based on average
    relative token position inside nested text segments.

    method="embedding-cluster" builds residual centroid token codes from
    embedding vectors reduced with PCA.
    """
    method = _normalize_method(method)

    if method == "relative-position":
        if semantic_spelling_file is not None:
            raise ValueError("semantic_spelling_file is only available with method='embedding-cluster'.")
        if lossless_tokenization:
            raise ValueError("lossless_tokenization is only available with method='embedding-cluster'.")
        return create_relative_position_tokenizer(
            source_file=source_file,
            target_file=target_file,
        )

    if cache_file is None and target_file is not None:
        cache_file = default_embedding_cache_file(target_file)

    from .embedding_cluster import create_embedding_cluster_tokenizer

    return create_embedding_cluster_tokenizer(
        source_file=source_file,
        target_file=target_file,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        reduced_dim=reduced_dim,
        cluster_method=cluster_method,
        depth=depth,
        normalize=normalize,
        cache_file=cache_file,
        refresh_cache=refresh_cache,
        strip_embedding_text=strip_embedding_text,
        max_tokens=max_tokens,
        random_state=random_state,
        report_collisions=report_collisions,
        collision_report_limit=collision_report_limit,
        plot_pca_3d=plot_pca_3d,
        plot_pca_2d=plot_pca_2d,
        plot_clusters=plot_clusters,
        plot_centroid_labels=plot_centroid_labels,
        plot_sample_size=plot_sample_size,
        plot_output_dir=plot_output_dir,
        show_plots=show_plots,
        semantic_spelling_file=semantic_spelling_file,
        centroid_label_method=centroid_label_method,
        centroid_label_model=centroid_label_model,
        centroid_label_concurrency=centroid_label_concurrency,
        centroid_label_examples=centroid_label_examples,
        centroid_label_max_output_tokens=centroid_label_max_output_tokens,
        lossless_tokenization=lossless_tokenization,
    )
