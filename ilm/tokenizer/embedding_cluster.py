"""
Embedding-cluster hierarchical tokenizer.

This module builds "a:b:c" tokenizer codes from embedding vectors reduced with
PCA and residual centroid codes. The first token id names the nearest centroid
in the reduced embedding space. Later token ids are learned from residuals and
renamed by cosine similarity against the first centroid basis.

The recommended cluster method is spherical K-Means: assignments are made by
cosine direction, while the residual subtraction still uses each assigned
cluster's mean vector in the current residual space.
"""

import asyncio
import itertools
import json
import os
import re
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .core import (
    collect_unique_tokens,
    force_json_extension,
    generate_detokenizer,
    generate_tokenizer,
    save_dictionary,
)


ClusterMethod = str
CentroidLabelMethod = str

CLUSTER_METHODS = ("kmeans", "spherical-kmeans")
CENTROID_LABEL_METHODS = ("closest-token", "llm")
DEFAULT_CENTROID_LABEL_MODEL = "gpt-5.6-terra"
DEFAULT_CENTROID_LABEL_MAX_OUTPUT_TOKENS = 4096
GENERIC_CENTROID_LABEL_PARTS = {
    "cluster",
    "clusters",
    "english",
    "generic",
    "misc",
    "miscellaneous",
    "mixed",
    "semantic",
    "token",
    "tokens",
    "unknown",
    "vocabulary",
    "word",
    "words",
}
GENERIC_CENTROID_LABELS = {
    "general",
    "language",
    "old-english",
    "old-english-vocabulary",
    "other",
    "various",
}


def _normalize_cluster_method(method: ClusterMethod) -> ClusterMethod:
    normalized = method.replace("_", "-").lower()
    aliases = {
        "kmeans": "kmeans",
        "cosine": "spherical-kmeans",
        "cosine-kmeans": "spherical-kmeans",
        "spherical": "spherical-kmeans",
        "spherical-kmeans": "spherical-kmeans",
    }
    if normalized not in aliases:
        allowed = ", ".join(CLUSTER_METHODS)
        raise ValueError(f"Unknown cluster_method {method!r}. Choose one of: {allowed}.")
    return aliases[normalized]


def _normalize_centroid_label_method(method: CentroidLabelMethod) -> CentroidLabelMethod:
    normalized = method.replace("_", "-").lower()
    aliases = {
        "closest": "closest-token",
        "closest-word": "closest-token",
        "closest-token": "closest-token",
        "nearest": "closest-token",
        "nearest-word": "closest-token",
        "nearest-token": "closest-token",
        "llm": "llm",
        "openai": "llm",
        "gpt": "llm",
    }
    if normalized not in aliases:
        allowed = ", ".join(CENTROID_LABEL_METHODS)
        raise ValueError(f"Unknown centroid_label_method {method!r}. Choose one of: {allowed}.")
    return aliases[normalized]


def embedding_text_for_token(token: str, strip_embedding_text: bool = True) -> str:
    """
    Convert a tokenizer token into non-empty text suitable for embedding.
    """
    special_tokens = {
        "\n": "<newline>",
        "\r": "<carriage return>",
        "\t": "<tab>",
    }
    if token in special_tokens:
        return special_tokens[token]

    if token.strip() == "":
        return f"<whitespace length={len(token)}>"

    return token.strip() if strip_embedding_text else token


def fetch_openai_embeddings(
    tokens: Sequence[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 512,
    strip_embedding_text: bool = True,
) -> np.ndarray:
    """
    Embed tokens with the OpenAI embeddings API.

    The synchronous embeddings endpoint accepts many input strings per request,
    but keep batch_size comfortably below the API array limit.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if batch_size > 2048:
        raise ValueError("OpenAI embeddings input arrays support at most 2048 strings per request.")

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI()

    embedding_rows: List[List[float]] = []
    for start in tqdm(range(0, len(tokens), batch_size), desc="Embedding tokens"):
        batch_tokens = tokens[start:start + batch_size]
        batch_inputs = [embedding_text_for_token(token, strip_embedding_text) for token in batch_tokens]
        response = client.embeddings.create(
            model=model,
            input=batch_inputs,
            encoding_format="float",
        )
        for row in sorted(response.data, key=lambda item: item.index):
            embedding_rows.append(row.embedding)

    return np.asarray(embedding_rows, dtype=np.float32)


def load_or_create_embeddings(
    tokens: Sequence[str],
    cache_file: Optional[str],
    model: str,
    batch_size: int,
    strip_embedding_text: bool,
    refresh_cache: bool = False,
) -> np.ndarray:
    """
    Load embeddings from cache or call the OpenAI embeddings API and cache them.
    """
    if cache_file is not None and os.path.exists(cache_file) and not refresh_cache:
        cached = np.load(cache_file, allow_pickle=False)
        cached_tokens = cached["tokens"].tolist()
        cached_model = str(cached["model"])
        if cached_tokens != list(tokens):
            raise ValueError(
                "Embedding cache token order does not match the source file. "
                "Use --refresh-cache or choose a different --cache-file."
            )
        if cached_model != model:
            raise ValueError(
                f"Embedding cache was built with {cached_model!r}, not {model!r}. "
                "Use --refresh-cache or choose a different --cache-file."
            )
        return cached["embeddings"].astype(np.float32)

    embeddings = fetch_openai_embeddings(
        tokens=tokens,
        model=model,
        batch_size=batch_size,
        strip_embedding_text=strip_embedding_text,
    )

    if cache_file is not None:
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        np.savez(
            cache_file,
            tokens=np.asarray(tokens, dtype=str),
            model=np.asarray(model),
            embeddings=embeddings,
        )

    return embeddings


def reduce_embeddings(
    embeddings: np.ndarray,
    reduced_dim: int = 10,
    normalize: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    """
    Normalize embeddings if requested, then reduce them with PCA.
    """
    if reduced_dim < 1:
        raise ValueError("reduced_dim must be >= 1")

    max_dim = min(embeddings.shape[0], embeddings.shape[1])
    n_components = min(reduced_dim, max_dim)
    matrix = embeddings
    if normalize:
        matrix = StandardScaler().fit_transform(matrix)

    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(matrix).astype(np.float32)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _centers_from_labels(
    vectors: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    fallback_centers: Optional[np.ndarray] = None,
) -> np.ndarray:
    centers = np.zeros((n_clusters, vectors.shape[1]), dtype=np.float32)
    counts = np.bincount(labels, minlength=n_clusters).astype(np.float32)
    np.add.at(centers, labels, vectors.astype(np.float32))

    non_empty = counts > 0
    centers[non_empty] /= counts[non_empty, None]
    if fallback_centers is not None:
        centers[~non_empty] = fallback_centers[~non_empty]
    return centers


def _fit_spherical_kmeans(
    vectors: np.ndarray,
    n_clusters: int,
    random_state: int,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster by cosine direction, then return residual-space cluster means.
    """
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")

    if n_clusters == 1:
        labels = np.zeros(len(vectors), dtype=np.int64)
        return vectors.mean(axis=0, keepdims=True).astype(np.float32), labels

    normalized_vectors = _l2_normalize(vectors.astype(np.float32))
    initial = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state).fit(normalized_vectors)
    direction_centers = _l2_normalize(initial.cluster_centers_.astype(np.float32))
    labels = initial.labels_.astype(np.int64)

    for _ in range(max_iter):
        similarities = normalized_vectors @ direction_centers.T
        new_labels = np.argmax(similarities, axis=1).astype(np.int64)
        new_direction_centers = _centers_from_labels(
            normalized_vectors,
            new_labels,
            n_clusters,
            fallback_centers=direction_centers,
        )
        new_direction_centers = _l2_normalize(new_direction_centers)

        center_shift = np.max(np.linalg.norm(new_direction_centers - direction_centers, axis=1))
        converged = np.array_equal(new_labels, labels) and center_shift < tol
        labels = new_labels
        direction_centers = new_direction_centers
        if converged:
            break

    residual_centers = _centers_from_labels(
        vectors,
        labels,
        n_clusters,
        fallback_centers=direction_centers,
    )
    return residual_centers.astype(np.float32), labels.astype(np.int64)


def _fit_cluster_level(
    vectors: np.ndarray,
    method: ClusterMethod,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit one residual clustering level and return centers plus row labels.
    """
    method = _normalize_cluster_method(method)
    k = min(64, len(vectors))
    if method == "spherical-kmeans":
        return _fit_spherical_kmeans(vectors, n_clusters=k, random_state=random_state)

    fitted = KMeans(n_clusters=k, n_init="auto", random_state=random_state).fit(vectors)
    return fitted.cluster_centers_.astype(np.float32), fitted.labels_.astype(np.int64)


def _complete_rankings(rankings: np.ndarray, base: int = 64) -> np.ndarray:
    if rankings.shape[1] == base:
        return rankings.astype(np.int16)

    full_rankings = []
    all_values = list(range(base))
    for row in rankings.tolist():
        missing = [value for value in all_values if value not in row]
        full_rankings.append(row + missing)
    return np.asarray(full_rankings, dtype=np.int16)


def _cosine_rankings(vectors: np.ndarray, reference_centers: np.ndarray, base: int = 64) -> np.ndarray:
    """
    Rank reference centroid ids for each vector by descending cosine similarity.
    """
    vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    reference_norms = np.linalg.norm(reference_centers, axis=1, keepdims=True).T
    denom = np.maximum(vector_norms * reference_norms, 1e-12)
    similarities = vectors @ reference_centers.T / denom
    return _complete_rankings(np.argsort(-similarities, axis=1), base=base)


def _euclidean_rankings(vectors: np.ndarray, reference_centers: np.ndarray, base: int = 64) -> np.ndarray:
    """
    Rank reference centroid ids for each vector by increasing Euclidean distance.
    """
    distances = np.sum((vectors[:, None, :] - reference_centers[None, :, :]) ** 2, axis=2)
    return _complete_rankings(np.argsort(distances, axis=1), base=base)


def _rankings_for_cluster_method(
    vectors: np.ndarray,
    reference_centers: np.ndarray,
    method: ClusterMethod,
    base: int = 64,
) -> np.ndarray:
    if _normalize_cluster_method(method) == "spherical-kmeans":
        return _cosine_rankings(vectors, reference_centers, base=base)
    return _euclidean_rankings(vectors, reference_centers, base=base)


def _nearest_reference_indices(centers: np.ndarray, reference_centers: np.ndarray) -> np.ndarray:
    """
    Name residual centroids by the closest first-level centroid in cosine space.
    """
    return _cosine_rankings(centers, reference_centers)[:, 0].astype(np.int64)


def _nearest_token_indices_to_centers(
    vectors: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> List[Optional[int]]:
    """
    Find the closest token vector to each centroid within its assigned cluster.
    """
    nearest_indices: List[Optional[int]] = []
    for center_idx, center in enumerate(centers):
        cluster_indices = np.where(labels == center_idx)[0]
        if len(cluster_indices) == 0:
            nearest_indices.append(None)
            continue
        cluster_vectors = vectors[cluster_indices]
        if metric == "cosine":
            center_norm = center / max(float(np.linalg.norm(center)), 1e-12)
            cluster_norms = _l2_normalize(cluster_vectors)
            similarities = cluster_norms @ center_norm
            nearest_indices.append(int(cluster_indices[int(np.argmax(similarities))]))
        else:
            distances = np.sum((cluster_vectors - center) ** 2, axis=1)
            nearest_indices.append(int(cluster_indices[int(np.argmin(distances))]))
    return nearest_indices


def _centroid_token_labels(
    tokens: Sequence[str],
    vectors: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> List[str]:
    nearest_token_indices = _nearest_token_indices_to_centers(
        vectors=vectors,
        centers=centers,
        labels=labels,
        metric=metric,
    )
    return [
        tokens[token_idx] if token_idx is not None else f"<centroid:{center_idx}>"
        for center_idx, token_idx in enumerate(nearest_token_indices)
    ]


def _sample_token_indices_by_cluster(
    labels: np.ndarray,
    n_clusters: int,
    limit: int,
    random_state: int,
) -> List[List[int]]:
    if limit < 1:
        raise ValueError("centroid_label_examples must be >= 1")

    rng = np.random.default_rng(random_state)
    sampled_indices: List[List[int]] = []
    for center_idx in range(n_clusters):
        cluster_indices = np.where(labels == center_idx)[0]
        if len(cluster_indices) > limit:
            cluster_indices = rng.choice(cluster_indices, size=limit, replace=False)
        sampled_indices.append(np.sort(cluster_indices).astype(int).tolist())
    return sampled_indices


def _format_centroid_example_tokens(tokens: Sequence[str], token_indices: Sequence[int]) -> List[str]:
    examples = []
    seen = set()
    for token_idx in token_indices:
        text = embedding_text_for_token(tokens[token_idx])
        if text not in seen:
            seen.add(text)
            examples.append(text)
    return examples


def _clean_llm_centroid_label(raw_label: str) -> str:
    label = raw_label.strip().splitlines()[0] if raw_label.strip() else ""
    label = re.sub(r"^(label|centroid|name)\s*[:=-]\s*", "", label, flags=re.IGNORECASE)
    label = label.strip(" \t\r\n\"'`.,;")
    label = label.replace(":", " ")
    label = re.sub(r"[^A-Za-z0-9\-\s]", "", label)
    label = re.sub(r"\s+", "-", label).strip("-").lower()
    label = re.sub(r"-+", "-", label)
    if not label:
        raise ValueError("empty LLM centroid label")
    return label[:64]


def _llm_centroid_label_quality_error(label: str) -> Optional[str]:
    parts = [part for part in label.split("-") if part]
    if label in GENERIC_CENTROID_LABELS or any(part in GENERIC_CENTROID_LABEL_PARTS for part in parts):
        return "too generic"
    if len(parts) > 2:
        return "too long"
    return None


def _validate_llm_centroid_labels(
    labels: Sequence[str],
    n_clusters: int,
    taken_labels: Optional[Dict[str, int]] = None,
) -> Tuple[List[Optional[str]], Dict[int, str]]:
    accepted: List[Optional[str]] = [None for _ in range(n_clusters)]
    issues: Dict[int, str] = {}
    seen_labels = dict(taken_labels or {})

    for label_idx in range(n_clusters):
        if label_idx >= len(labels):
            issues[label_idx] = "missing"
            continue

        try:
            label = _clean_llm_centroid_label(labels[label_idx])
        except ValueError as error:
            issues[label_idx] = str(error)
            continue

        quality_error = _llm_centroid_label_quality_error(label)
        if quality_error is not None:
            issues[label_idx] = f"{quality_error}: {label!r}"
            continue

        if label in seen_labels:
            issues[label_idx] = f"duplicates label {seen_labels[label]}: {label!r}"
            continue

        accepted[label_idx] = label
        seen_labels[label] = label_idx

    return accepted, issues


def _raise_for_llm_centroid_label_issues(issues: Dict[int, str]) -> None:
    if not issues:
        return
    label_idx = next(iter(issues))
    raise ValueError(f"LLM centroid label {label_idx} is {issues[label_idx]}")


def _reasoning_effort_for_label_model(model: str) -> Optional[str]:
    normalized = model.lower()
    if normalized.startswith("gpt-5.6"):
        return "low"
    if normalized.startswith("gpt-5"):
        return "minimal"
    return None


def _format_llm_error(error: Exception, max_width: int = 240) -> str:
    message = f"{type(error).__name__}: {error}"
    message = re.sub(r"\s+", " ", message).strip()
    if len(message) <= max_width:
        return message
    return message[:max_width - 3] + "..."


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM response did not contain a JSON object")
        parsed = json.loads(stripped[start:end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object")
    return parsed


def _parse_level_centroid_label_candidates(response_text: str) -> Dict[int, str]:
    parsed = _extract_json_object(response_text)
    labels_data = parsed.get("labels", parsed)

    if isinstance(labels_data, list):
        labels_by_id: Dict[int, str] = {}
        for item in labels_data:
            if not isinstance(item, dict):
                raise ValueError("LLM labels list must contain objects")
            labels_by_id[int(item["id"])] = str(item["label"])
    elif isinstance(labels_data, dict):
        labels_by_id = {int(key): str(value) for key, value in labels_data.items()}
    else:
        raise ValueError("LLM labels must be a JSON object or list")

    return labels_by_id


def _parse_level_centroid_labels(response_text: str, n_clusters: int) -> List[str]:
    labels_by_id = _parse_level_centroid_label_candidates(response_text)
    labels = [labels_by_id[idx] if idx in labels_by_id else "" for idx in range(n_clusters)]
    accepted, issues = _validate_llm_centroid_labels(labels, n_clusters)
    _raise_for_llm_centroid_label_issues(issues)
    return [str(label) for label in accepted]


def _apply_llm_centroid_label_candidates(
    labels_by_id: Dict[int, str],
    n_clusters: int,
    taken_labels: Optional[Dict[str, int]] = None,
    required_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[Optional[str]], Dict[int, str]]:
    accepted: List[Optional[str]] = [None for _ in range(n_clusters)]
    issues: Dict[int, str] = {}
    seen_labels = dict(taken_labels or {})
    label_indices = range(n_clusters) if required_indices is None else required_indices

    for label_idx in label_indices:
        if label_idx not in labels_by_id:
            issues[label_idx] = "missing"
            continue

        try:
            label = _clean_llm_centroid_label(labels_by_id[label_idx])
        except ValueError as error:
            issues[label_idx] = str(error)
            continue

        quality_error = _llm_centroid_label_quality_error(label)
        if quality_error is not None:
            issues[label_idx] = f"{quality_error}: {label!r}"
            continue

        if label in seen_labels:
            issues[label_idx] = f"duplicates label {seen_labels[label]}: {label!r}"
            continue

        accepted[label_idx] = label
        seen_labels[label] = label_idx

    return accepted, issues


def _accepted_label_index(labels: Sequence[Optional[str]]) -> Dict[str, int]:
    return {label: idx for idx, label in enumerate(labels) if label is not None}


def _label_response_params(
    model: str,
    instructions: str,
    prompt: str,
    max_output_tokens: int,
) -> Dict[str, Any]:
    response_params: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }
    reasoning_effort = _reasoning_effort_for_label_model(model)
    if reasoning_effort is not None:
        response_params["reasoning"] = {"effort": reasoning_effort}
    return response_params


def _finalize_label_repairs(
    accepted_labels: Sequence[Optional[str]],
    fallback_labels: Sequence[str],
    errors: List[str],
) -> Tuple[List[str], int, List[str]]:
    final_labels = []
    fallback_count = 0
    for label_idx, label in enumerate(accepted_labels):
        if label is None:
            final_labels.append(fallback_labels[label_idx])
            fallback_count += 1
        else:
            final_labels.append(label)
    return final_labels, fallback_count, errors


async def _repair_llm_centroid_labels(
    client: Any,
    model: str,
    clusters: Sequence[Dict[str, Any]],
    candidate_labels: Dict[int, str],
    accepted_labels: Sequence[Optional[str]],
    issues: Dict[int, str],
    fallback_labels: Sequence[str],
    semaphore: asyncio.Semaphore,
    max_output_tokens: int,
) -> Tuple[List[Optional[str]], List[str]]:
    if not issues:
        return list(accepted_labels), []

    repair_indices = sorted(issues)
    repair_clusters = []
    for cluster_idx in repair_indices:
        repair_clusters.append(
            {
                "id": cluster_idx,
                "words": clusters[cluster_idx]["words"],
                "previous_label": candidate_labels.get(cluster_idx),
                "problem": issues[cluster_idx],
            }
        )

    instructions = (
        "You repair semantic labels for a compact word-tokenizer. "
        "Return only valid JSON in this exact shape: {\"labels\":{\"0\":\"label\"}}. "
        "Return labels only for the requested cluster ids. Labels must be clean semantic atoms: "
        "one common lowercase word is best, two words is the maximum. "
        "Do not use generic labels such as vocabulary, old english, misc, word, token, or cluster. "
        "Do not duplicate any existing label."
    )
    prompt = (
        "Repair the invalid semantic labels from their sampled representative words.\n\n"
        f"{json.dumps({'existing_labels': [label for label in accepted_labels if label], 'clusters': repair_clusters}, ensure_ascii=False)}"
    )

    try:
        response_params = _label_response_params(
            model=model,
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
        )
        async with semaphore:
            response = await client.responses.create(**response_params)
        repair_candidates = _parse_level_centroid_label_candidates(response.output_text)
    except Exception as error:
        return list(accepted_labels), [_format_llm_error(error)]

    repaired_labels, repair_issues = _apply_llm_centroid_label_candidates(
        labels_by_id=repair_candidates,
        n_clusters=len(fallback_labels),
        taken_labels=_accepted_label_index(accepted_labels),
        required_indices=repair_indices,
    )
    updated_labels = list(accepted_labels)
    for label_idx, label in enumerate(repaired_labels):
        if label is not None:
            updated_labels[label_idx] = label

    errors = [
        f"label {label_idx}: {issue}"
        for label_idx, issue in repair_issues.items()
    ]
    return updated_labels, errors


async def _label_one_level_with_llm(
    client: Any,
    model: str,
    level_idx: int,
    clusters: Sequence[Dict[str, Any]],
    fallback_labels: Sequence[str],
    semaphore: asyncio.Semaphore,
    max_output_tokens: int,
) -> Tuple[int, List[str], int, int, List[str]]:
    instructions = (
        "You name semantic centroids for a compact word-tokenizer. "
        "You will receive all clusters for one tokenizer coordinate at once. "
        "For each cluster, infer the shared semantic idea from the representative words. "
        "Return only valid JSON in this exact shape: {\"labels\":{\"0\":\"label\",\"1\":\"label\"}}. "
        "Every cluster id must appear exactly once. Labels should be clean semantic atoms: "
        "prefer one common lowercase word, use at most two words, avoid generic labels like "
        "\"vocabulary\", \"old english\", \"misc\", \"word\", or \"cluster\", and avoid copying rare examples "
        "unless the rare word truly names the group. Labels should be distinct within the level; "
        "duplicate, generic, or longer labels will be rejected."
    )
    prompt = (
        "Name each semantic cluster from its sampled representative words.\n\n"
        f"{json.dumps({'clusters': clusters}, ensure_ascii=False)}"
    )
    primary_error: Optional[str] = None
    try:
        response_params = _label_response_params(
            model=model,
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
        )
        async with semaphore:
            response = await client.responses.create(**response_params)
        candidate_labels = _parse_level_centroid_label_candidates(response.output_text)
    except Exception as error:
        accepted_labels = [None for _ in fallback_labels]
        candidate_labels = {}
        issues = {idx: "primary LLM response failed" for idx in range(len(fallback_labels))}
        primary_error = _format_llm_error(error)
    else:
        accepted_labels, issues = _apply_llm_centroid_label_candidates(
            labels_by_id=candidate_labels,
            n_clusters=len(fallback_labels),
        )

    initial_issue_count = len(issues)
    repair_errors: List[str] = []
    if issues:
        accepted_labels, repair_errors = await _repair_llm_centroid_labels(
            client=client,
            model=model,
            clusters=clusters,
            candidate_labels=candidate_labels,
            accepted_labels=accepted_labels,
            issues=issues,
            fallback_labels=fallback_labels,
            semaphore=semaphore,
            max_output_tokens=max_output_tokens,
        )

    errors = []
    labels, fallback_count, errors = _finalize_label_repairs(
        accepted_labels=accepted_labels,
        fallback_labels=fallback_labels,
        errors=errors,
    )
    if fallback_count > 0 and primary_error is not None:
        errors.append(primary_error)
    if fallback_count > 0:
        errors.extend(repair_errors)
    repaired_count = max(0, initial_issue_count - fallback_count)
    return level_idx, labels, fallback_count, repaired_count, errors[:5]


async def _label_centroids_with_llm_async(
    tokens: Sequence[str],
    levels: Sequence[Dict[str, Any]],
    model: str,
    concurrency: int,
    examples_per_cluster: int,
    max_output_tokens: int,
    random_state: int,
) -> Tuple[int, int, List[str]]:
    """
    Label residual centroids with concurrent OpenAI Responses API calls.
    """
    if concurrency < 1:
        raise ValueError("centroid_label_concurrency must be >= 1")
    if examples_per_cluster < 1:
        raise ValueError("centroid_label_examples must be >= 1")
    if max_output_tokens < 1:
        raise ValueError("centroid_label_max_output_tokens must be >= 1")

    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    load_dotenv()
    semaphore = asyncio.Semaphore(concurrency)
    fallback_by_level = []
    clusters_by_level = []
    for level in levels:
        level_idx = int(level["level"])
        nearest_metric = str(level.get("nearest_metric", "euclidean"))
        fallback_labels = _centroid_token_labels(
            tokens=tokens,
            vectors=level["vectors"],
            centers=level["centers"],
            labels=level["labels"],
            metric=nearest_metric,
        )
        sampled_indices = _sample_token_indices_by_cluster(
            labels=level["labels"],
            n_clusters=len(level["centers"]),
            limit=examples_per_cluster,
            random_state=random_state + level_idx,
        )
        clusters = []
        for center_idx, token_indices in enumerate(sampled_indices):
            examples = _format_centroid_example_tokens(tokens, token_indices)
            if len(examples) == 0:
                examples = [fallback_labels[center_idx]]
            clusters.append({"id": center_idx, "words": examples})
        fallback_by_level.append(fallback_labels)
        clusters_by_level.append(clusters)

    async with AsyncOpenAI() as client:
        tasks = []
        for level_pos, level in enumerate(levels):
            level_idx = int(level["level"])
            fallback_labels = fallback_by_level[level_pos]
            tasks.append(
                _label_one_level_with_llm(
                    client=client,
                    model=model,
                    level_idx=level_idx,
                    clusters=clusters_by_level[level_pos],
                    fallback_labels=fallback_labels,
                    semaphore=semaphore,
                    max_output_tokens=max_output_tokens,
                )
            )

        fallback_count = 0
        repaired_count = 0
        errors: List[str] = []
        labels_by_level = [
            [fallback_label for fallback_label in fallback_labels]
            for fallback_labels in fallback_by_level
        ]
        level_positions = {int(level["level"]): level_pos for level_pos, level in enumerate(levels)}
        print(
            "Labeling centroids with "
            f"{model} ({len(tasks)} level calls, concurrency={concurrency}, "
            f"examples={examples_per_cluster}, max_output_tokens={max_output_tokens})"
        )
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Labeling centroids"):
            level_idx, labels, level_fallback_count, level_repaired_count, level_errors = await task
            labels_by_level[level_positions[level_idx]] = labels
            fallback_count += level_fallback_count
            repaired_count += level_repaired_count
            for error in level_errors:
                if len(errors) < 5 and error not in errors:
                    errors.append(error)

    for level_pos, level in enumerate(levels):
        level["centroid_labels"] = labels_by_level[level_pos]
    return fallback_count, repaired_count, errors


def label_centroids(
    tokens: Sequence[str],
    levels: Sequence[Dict[str, Any]],
    method: CentroidLabelMethod = "closest-token",
    model: str = DEFAULT_CENTROID_LABEL_MODEL,
    concurrency: int = 8,
    examples_per_cluster: int = 20,
    max_output_tokens: int = DEFAULT_CENTROID_LABEL_MAX_OUTPUT_TOKENS,
    random_state: int = 42,
) -> None:
    """
    Add centroid_labels to each residual level using the requested strategy.
    """
    method = _normalize_centroid_label_method(method)
    if method == "closest-token":
        for level in levels:
            nearest_metric = str(level.get("nearest_metric", "euclidean"))
            level["centroid_labels"] = _centroid_token_labels(
                tokens=tokens,
                vectors=level["vectors"],
                centers=level["centers"],
                labels=level["labels"],
                metric=nearest_metric,
            )
        return

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        fallback_count, repaired_count, errors = asyncio.run(
            _label_centroids_with_llm_async(
                tokens=tokens,
                levels=levels,
                model=model,
                concurrency=concurrency,
                examples_per_cluster=examples_per_cluster,
                max_output_tokens=max_output_tokens,
                random_state=random_state,
            )
        )
    else:
        raise RuntimeError("LLM centroid labeling cannot run inside an existing asyncio event loop.")

    total_labels = sum(len(level["centers"]) for level in levels)
    if fallback_count == total_labels:
        details = f" First error: {errors[0]}" if errors else ""
        raise RuntimeError(
            f"LLM centroid labeling failed for all {total_labels} centroid labels. "
            "No semantic spelling file was written with fallback labels."
            f"{details}"
        )
    if repaired_count > 0:
        print(f"LLM centroid labeling repaired {repaired_count} labels without fallback.")
    if fallback_count > 0:
        print(f"LLM centroid labeling used closest-token fallbacks for {fallback_count} labels.")
        if errors:
            print("First LLM centroid labeling errors:")
            for error in errors:
                print(f"- {error}")


def _sample_plot_indices(n_items: int, sample_size: Optional[int], random_state: int) -> np.ndarray:
    if sample_size is None or sample_size <= 0 or sample_size >= n_items:
        return np.arange(n_items, dtype=np.int64)

    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_items, size=sample_size, replace=False)).astype(np.int64)


def plot_pca_centroids_3d(
    tokens: Sequence[str],
    reduced_vectors: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
    title: str = "PCA 3D Projection With Centroids",
    centroid_label: str = "Centroids",
    colorbar_label: str = "Cluster id",
    color_clusters: bool = False,
    label_centroids: bool = True,
    sample_size: Optional[int] = 20000,
    random_state: int = 42,
    nearest_metric: str = "euclidean",
    centroid_labels: Optional[Sequence[str]] = None,
) -> None:
    """
    Plot PCA dimensions 1-3 with centroids and nearest-token labels.
    """
    if reduced_vectors.shape[1] < 3:
        raise ValueError("PCA centroid plot requires at least 3 reduced dimensions.")

    import matplotlib.pyplot as plt

    sample_indices = _sample_plot_indices(len(reduced_vectors), sample_size, random_state)
    sample_vectors = reduced_vectors[sample_indices, :3]
    sample_labels = labels[sample_indices]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    if color_clusters:
        points = ax.scatter(
            sample_vectors[:, 0],
            sample_vectors[:, 1],
            sample_vectors[:, 2],
            c=sample_labels,
            cmap="tab20",
            s=5,
            alpha=0.35,
            linewidths=0,
        )
        fig.colorbar(points, ax=ax, pad=0.08, shrink=0.7, label=colorbar_label)
    else:
        ax.scatter(
            sample_vectors[:, 0],
            sample_vectors[:, 1],
            sample_vectors[:, 2],
            color="#8a8a8a",
            s=5,
            alpha=0.25,
            linewidths=0,
        )

    center_vectors = centers[:, :3]
    ax.scatter(
        center_vectors[:, 0],
        center_vectors[:, 1],
        center_vectors[:, 2],
        color="black",
        marker="x",
        s=80,
        linewidths=2,
        label=centroid_label,
    )

    if label_centroids:
        if centroid_labels is None:
            centroid_labels = _centroid_token_labels(
                tokens=tokens,
                vectors=reduced_vectors,
                centers=centers,
                labels=labels,
                metric=nearest_metric,
            )
        for center_idx, centroid_label_text in enumerate(centroid_labels):
            label = f"{center_idx}: {_format_token_preview(centroid_label_text, max_width=18)}"
            ax.text(
                center_vectors[center_idx, 0],
                center_vectors[center_idx, 1],
                center_vectors[center_idx, 2],
                label,
                fontsize=7,
            )

    plotted_count = len(sample_indices)
    total_count = len(reduced_vectors)
    title_suffix = f"{plotted_count}/{total_count} tokens"
    if plotted_count == total_count:
        title_suffix = f"{total_count} tokens"
    ax.set_title(f"{title} ({title_suffix})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_residual_levels_3d(
    tokens: Sequence[str],
    levels: Sequence[Dict[str, Any]],
    color_clusters: bool = False,
    label_centroids: bool = True,
    sample_size: Optional[int] = 20000,
    random_state: int = 42,
) -> None:
    """
    Plot one 3D figure for each residual clustering level.
    """
    for level in levels:
        level_idx = int(level["level"])
        plot_pca_centroids_3d(
            tokens=tokens,
            reduced_vectors=level["vectors"],
            centers=level["centers"],
            labels=level["labels"],
            title=f"Residual Level {level_idx}: E_{level_idx}(w)",
            centroid_label=f"Level {level_idx} centroids",
            colorbar_label=f"Level {level_idx} cluster id",
            color_clusters=color_clusters,
            label_centroids=label_centroids,
            sample_size=sample_size,
            random_state=random_state,
            nearest_metric=str(level.get("nearest_metric", "euclidean")),
            centroid_labels=level.get("centroid_labels"),
        )


def build_semantic_spelling_mapping(
    tokens: Sequence[str],
    levels: Sequence[Dict[str, Any]],
    token_order: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """
    Map each token to the nearest-token labels of its assigned residual centroids.
    """
    spelling_parts: List[List[str]] = [[] for _ in tokens]
    for level in levels:
        centroid_words = level.get("centroid_labels")
        if centroid_words is None:
            nearest_metric = str(level.get("nearest_metric", "euclidean"))
            centroid_words = _centroid_token_labels(
                tokens=tokens,
                vectors=level["vectors"],
                centers=level["centers"],
                labels=level["labels"],
                metric=nearest_metric,
            )
        for token_idx, label in enumerate(level["labels"].tolist()):
            spelling_parts[token_idx].append(centroid_words[int(label)])

    token_to_spelling = {
        token: ":".join(spelling_parts[token_idx])
        for token_idx, token in enumerate(tokens)
    }
    ordered_tokens = token_order if token_order is not None else tokens
    return {token: token_to_spelling[token] for token in ordered_tokens}


def save_semantic_spelling_mapping(
    tokens: Sequence[str],
    levels: Sequence[Dict[str, Any]],
    filename: str,
    token_order: Optional[Sequence[str]] = None,
) -> None:
    """
    Save the human-readable centroid-word spelling table.
    """
    save_dictionary(
        build_semantic_spelling_mapping(
            tokens=tokens,
            levels=levels,
            token_order=token_order,
        ),
        force_json_extension(filename),
    )


def _codes_to_ids(codes: np.ndarray, base: int = 64) -> np.ndarray:
    multipliers = np.asarray(
        [base ** power for power in range(codes.shape[1] - 1, -1, -1)],
        dtype=np.int64,
    )
    return (codes.astype(np.int64) * multipliers).sum(axis=1)


def _code_to_id(code: Sequence[int], base: int = 64) -> int:
    code_id = 0
    for value in code:
        code_id = code_id * base + int(value)
    return code_id


def _code_string_to_id(code: str, base: int = 64) -> int:
    return _code_to_id([int(part) for part in code.split(":")], base=base)


def _candidate_codes(
    code: Sequence[int],
    token_rankings: Sequence[Sequence[int]],
    depth: int,
) -> Generator[Tuple[int, ...], None, None]:
    """
    Yield deterministic repair candidates, preserving longer prefixes first.
    """
    yielded = set()
    for suffix_start in range(depth - 1, -1, -1):
        prefix = tuple(int(value) for value in code[:suffix_start])
        suffix_rankings = [token_rankings[level] for level in range(suffix_start, depth)]
        for suffix in itertools.product(*suffix_rankings):
            candidate = prefix + tuple(int(value) for value in suffix)
            if candidate not in yielded:
                yielded.add(candidate)
                yield candidate


def _format_code(code: Sequence[int]) -> str:
    return ":".join(str(int(part)) for part in code)


def _format_token_preview(token: Optional[str], max_width: int = 28) -> str:
    if token is None:
        return ""
    preview = repr(token)
    if len(preview) <= max_width:
        return preview
    return preview[:max_width - 3] + "..."


def _shared_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_value, right_value in zip(left, right):
        if int(left_value) != int(right_value):
            break
        count += 1
    return count


def _print_collision_report_header() -> None:
    print("\nResidual code collision repairs:")
    print("+--------+-----------+------------------------------+-------------+-------------+-------------+")
    print("| repair | token_idx | token                        | preferred   | repaired    | kept_prefix |")
    print("+--------+-----------+------------------------------+-------------+-------------+-------------+")


def _print_collision_report_row(
    repair_number: int,
    token_idx: int,
    token: Optional[str],
    preferred_code: Sequence[int],
    repaired_code: Sequence[int],
) -> None:
    kept_prefix = _shared_prefix_length(preferred_code, repaired_code)
    print(
        "| "
        f"{repair_number:<6} | "
        f"{token_idx:<9} | "
        f"{_format_token_preview(token):<28} | "
        f"{_format_code(preferred_code):<11} | "
        f"{_format_code(repaired_code):<11} | "
        f"{kept_prefix}/{len(preferred_code):<9} |"
    )


def _print_collision_report_footer(collision_count: int, reported_count: int) -> None:
    print("+--------+-----------+------------------------------+-------------+-------------+-------------+")
    hidden_count = collision_count - reported_count
    if hidden_count > 0:
        print(f"Residual collision repair summary: {collision_count} repaired, {hidden_count} not shown.")
    else:
        print(f"Residual collision repair summary: {collision_count} repaired.")


def _repair_code_collisions(
    codes: np.ndarray,
    rankings_by_level: Sequence[np.ndarray],
    tokens: Optional[Sequence[str]] = None,
    report: bool = False,
    report_limit: Optional[int] = None,
    base: int = 64,
) -> Tuple[np.ndarray, int]:
    """
    Make residual codes one-to-one for the tokenizer reverse mapping.
    """
    depth = codes.shape[1]
    if len(codes) > base ** depth:
        raise ValueError(
            f"{len(codes)} tokens exceed the code capacity {base}^{depth}={base ** depth}. "
            "Increase --depth or reduce the vocabulary."
        )

    repaired = codes.copy()
    code_ids = _codes_to_ids(repaired, base=base)
    used_ids = set()
    collision_count = 0
    reported_count = 0
    report_started = False
    report_limit_reached = False

    for token_idx, code_id in enumerate(code_ids.tolist()):
        if code_id not in used_ids:
            used_ids.add(code_id)
            continue

        collision_count += 1
        preferred_code = repaired[token_idx].copy()
        token_rankings = [rankings[token_idx].tolist() for rankings in rankings_by_level]
        for candidate in _candidate_codes(repaired[token_idx], token_rankings, depth):
            candidate_id = _code_to_id(candidate, base=base)
            if candidate_id not in used_ids:
                repaired[token_idx] = np.asarray(candidate, dtype=np.int64)
                used_ids.add(candidate_id)
                if report and (report_limit is None or reported_count < report_limit):
                    if not report_started:
                        _print_collision_report_header()
                        report_started = True
                    token = tokens[token_idx] if tokens is not None else None
                    _print_collision_report_row(
                        repair_number=collision_count,
                        token_idx=token_idx,
                        token=token,
                        preferred_code=preferred_code,
                        repaired_code=repaired[token_idx],
                    )
                    reported_count += 1
                elif report and report_limit is not None and not report_limit_reached:
                    print(f"... collision report limit reached at {report_limit} rows; counting remaining repairs.")
                    report_limit_reached = True
                break
        else:
            raise RuntimeError("Unable to repair tokenizer code collision; no free code remains.")

    if report:
        if collision_count == 0:
            print("Residual code collision repairs: none.")
        elif report_started:
            _print_collision_report_footer(collision_count, reported_count)

    return repaired, collision_count


def _build_residual_codes(
    reduced_vectors: np.ndarray,
    method: ClusterMethod,
    depth: int,
    random_state: int,
    tokens: Optional[Sequence[str]] = None,
    report_collisions: bool = False,
    collision_report_limit: Optional[int] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    base: int = 64,
) -> Tuple[np.ndarray, int]:
    """
    Encode each vector as a residual path in the first centroid basis.
    """
    method = _normalize_cluster_method(method)
    residual_vectors = reduced_vectors.astype(np.float32).copy()
    codes = np.zeros((len(reduced_vectors), depth), dtype=np.int64)
    rankings_by_level = []
    nearest_metric = "cosine" if method == "spherical-kmeans" else "euclidean"

    first_centers, first_labels = _fit_cluster_level(
        vectors=residual_vectors,
        method=method,
        random_state=random_state,
    )
    reference_centers = first_centers
    if diagnostics is not None:
        diagnostics["levels"] = []

    for level in range(depth):
        if level == 0:
            centers = first_centers
            labels = first_labels
            code_values = labels
            rankings = _rankings_for_cluster_method(
                residual_vectors,
                reference_centers,
                method=method,
                base=base,
            )
        else:
            centers, labels = _fit_cluster_level(
                vectors=residual_vectors,
                method=method,
                random_state=random_state + level,
            )
            center_indices = _nearest_reference_indices(centers, reference_centers)
            code_values = center_indices[labels]
            rankings = _cosine_rankings(residual_vectors, reference_centers, base=base)

        if diagnostics is not None:
            diagnostics["levels"].append(
                {
                    "level": level,
                    "vectors": residual_vectors.copy(),
                    "centers": centers.copy(),
                    "labels": labels.copy(),
                    "nearest_metric": nearest_metric,
                }
            )

        codes[:, level] = code_values
        rankings_by_level.append(rankings)
        residual_vectors = residual_vectors - centers[labels]

    return _repair_code_collisions(
        codes,
        rankings_by_level,
        tokens=tokens,
        report=report_collisions,
        report_limit=collision_report_limit,
        base=base,
    )


def build_mapping_from_vectors(
    tokens: Sequence[str],
    reduced_vectors: np.ndarray,
    method: ClusterMethod = "spherical-kmeans",
    depth: int = 3,
    random_state: int = 42,
    report_collisions: bool = False,
    collision_report_limit: Optional[int] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build direct/reverse tokenizer mappings with residual centroid codes.
    """
    method = _normalize_cluster_method(method)
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if len(tokens) > 64 ** depth:
        raise ValueError(
            f"{len(tokens)} tokens exceed the code capacity 64^{depth}={64 ** depth}. "
            "Increase --depth or reduce the vocabulary."
        )

    codes, collision_repairs = _build_residual_codes(
        reduced_vectors=reduced_vectors,
        method=method,
        depth=depth,
        random_state=random_state,
        tokens=tokens,
        report_collisions=report_collisions,
        collision_report_limit=collision_report_limit,
        diagnostics=diagnostics,
    )
    unsorted_direct_mapping = {
        token: ":".join(str(part) for part in codes[token_idx].tolist())
        for token_idx, token in enumerate(tokens)
    }

    ordered_items = sorted(
        unsorted_direct_mapping.items(),
        key=lambda item: (_code_string_to_id(item[1]), item[0]),
    )
    direct_mapping = dict(ordered_items)

    reverse_mapping: Dict[str, str] = {}
    for token, code in direct_mapping.items():
        if code in reverse_mapping:
            raise RuntimeError(f"Duplicate tokenizer code generated: {code}")
        reverse_mapping[code] = token

    return {
        "direct": direct_mapping,
        "reverse": reverse_mapping,
        "metadata": {
            "residual_collision_repairs": collision_repairs,
        },
    }


def create_embedding_cluster_tokenizer(
    source_file: str,
    target_file: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    embedding_batch_size: int = 512,
    reduced_dim: int = 10,
    cluster_method: ClusterMethod = "spherical-kmeans",
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
    plot_clusters: bool = False,
    plot_centroid_labels: bool = True,
    plot_sample_size: Optional[int] = 20000,
    semantic_spelling_file: Optional[str] = None,
    centroid_label_method: CentroidLabelMethod = "closest-token",
    centroid_label_model: str = DEFAULT_CENTROID_LABEL_MODEL,
    centroid_label_concurrency: int = 8,
    centroid_label_examples: int = 20,
    centroid_label_max_output_tokens: int = DEFAULT_CENTROID_LABEL_MAX_OUTPUT_TOKENS,
) -> Tuple[Callable[[str], List[Optional[str]]], Callable[[List[str]], List[Optional[str]]]]:
    """
    Create an embedding-space hierarchical tokenizer from a source file.
    """
    cluster_method = _normalize_cluster_method(cluster_method)
    tokens = collect_unique_tokens(source_file)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]

    if len(tokens) == 0:
        raise ValueError("No tokens found in source_file.")

    embeddings = load_or_create_embeddings(
        tokens=tokens,
        cache_file=cache_file,
        model=embedding_model,
        batch_size=embedding_batch_size,
        strip_embedding_text=strip_embedding_text,
        refresh_cache=refresh_cache,
    )
    reduced_vectors = reduce_embeddings(
        embeddings=embeddings,
        reduced_dim=reduced_dim,
        normalize=normalize,
        random_state=random_state,
    )

    diagnostics: Optional[Dict[str, Any]] = {} if plot_pca_3d or semantic_spelling_file is not None else None
    token_mapping = build_mapping_from_vectors(
        tokens=tokens,
        reduced_vectors=reduced_vectors,
        method=cluster_method,
        depth=depth,
        random_state=random_state,
        report_collisions=report_collisions,
        collision_report_limit=collision_report_limit,
        diagnostics=diagnostics,
    )
    if plot_pca_3d:
        if diagnostics is None or "levels" not in diagnostics:
            raise RuntimeError("Missing residual level diagnostics for PCA plot.")
    if diagnostics is not None and "levels" in diagnostics:
        label_centroids(
            tokens=tokens,
            levels=diagnostics["levels"],
            method=centroid_label_method,
            model=centroid_label_model,
            concurrency=centroid_label_concurrency,
            examples_per_cluster=centroid_label_examples,
            max_output_tokens=centroid_label_max_output_tokens,
            random_state=random_state,
        )
    if semantic_spelling_file is not None:
        if diagnostics is None or "levels" not in diagnostics:
            raise RuntimeError("Missing residual level diagnostics for semantic spelling export.")
        save_semantic_spelling_mapping(
            tokens=tokens,
            levels=diagnostics["levels"],
            filename=semantic_spelling_file,
            token_order=token_mapping["direct"].keys(),
        )
        print(f"Semantic spelling saved to {force_json_extension(semantic_spelling_file)}")
    if plot_pca_3d:
        plot_residual_levels_3d(
            tokens=tokens,
            levels=diagnostics["levels"],
            color_clusters=plot_clusters,
            label_centroids=plot_centroid_labels,
            sample_size=plot_sample_size,
            random_state=random_state,
        )

    residual_metadata = token_mapping.get("metadata", {})
    token_mapping["metadata"] = {
        **residual_metadata,
        "method": "embedding-cluster",
        "coding": "residual-centroid",
        "cluster_method": cluster_method,
        "embedding_model": embedding_model,
        "embedding_batch_size": embedding_batch_size,
        "reduced_dim": reduced_dim,
        "actual_reduced_dim": int(reduced_vectors.shape[1]),
        "depth": depth,
        "base": 64,
        "normalize": normalize,
        "strip_embedding_text": strip_embedding_text,
        "token_count": len(tokens),
        "collision_report_enabled": report_collisions,
        "collision_report_limit": collision_report_limit,
        "plot_pca_3d": plot_pca_3d,
        "plot_clusters": plot_clusters,
        "plot_centroid_labels": plot_centroid_labels,
        "plot_sample_size": plot_sample_size,
        "semantic_spelling_file": (
            force_json_extension(semantic_spelling_file)
            if semantic_spelling_file is not None
            else None
        ),
        "centroid_label_method": _normalize_centroid_label_method(centroid_label_method),
        "centroid_label_model": centroid_label_model,
        "centroid_label_concurrency": centroid_label_concurrency,
        "centroid_label_examples": centroid_label_examples,
        "centroid_label_max_output_tokens": centroid_label_max_output_tokens,
    }

    if target_file is not None:
        save_dictionary(token_mapping, force_json_extension(target_file))

    return generate_tokenizer(token_mapping["direct"]), generate_detokenizer(token_mapping["reverse"])
