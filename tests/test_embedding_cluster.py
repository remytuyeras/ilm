import asyncio
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np

sys.path.insert(1, "./")

from ilm.tokenizer import embedding_cluster
from ilm.tokenizer.embedding_cluster import (
    _clean_llm_centroid_label,
    _fit_spherical_kmeans,
    _label_one_level_with_llm,
    _nearest_token_indices_to_centers,
    _normalize_centroid_label_method,
    _normalize_cluster_method,
    _parse_level_centroid_labels,
    _repair_code_collisions,
    _reasoning_effort_for_label_model,
    _sample_plot_indices,
    build_semantic_spelling_mapping,
    build_mapping_from_vectors,
)


class TestEmbeddingClusterTokenizer(unittest.TestCase):
    def test_residual_codes_are_invertible(self):
        rng = np.random.default_rng(42)
        tokens = [f"token_{idx}" for idx in range(256)]
        vectors = rng.normal(size=(256, 12)).astype(np.float32)

        mapping = build_mapping_from_vectors(
            tokens=tokens,
            reduced_vectors=vectors,
            method="spherical-kmeans",
            depth=3,
            random_state=42,
        )

        self.assertEqual(len(mapping["direct"]), len(tokens))
        self.assertEqual(len(mapping["reverse"]), len(tokens))
        self.assertIn("residual_collision_repairs", mapping["metadata"])

        for code in mapping["direct"].values():
            self.assertEqual(len(code.split(":")), 3)

    def test_spherical_kmeans_clusters_by_direction(self):
        vectors = np.asarray(
            [
                [100.0, 0.0],
                [1.0, 0.0],
                [0.0, 2.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        centers, labels = _fit_spherical_kmeans(
            vectors=vectors,
            n_clusters=2,
            random_state=42,
        )

        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])
        self.assertEqual(centers.shape, (2, 2))

    def test_cosine_kmeans_alias_uses_spherical_method(self):
        self.assertEqual(_normalize_cluster_method("cosine-kmeans"), "spherical-kmeans")
        self.assertEqual(_normalize_cluster_method("cosine"), "spherical-kmeans")

    def test_centroid_label_method_aliases(self):
        self.assertEqual(_normalize_centroid_label_method("closest-word"), "closest-token")
        self.assertEqual(_normalize_centroid_label_method("nearest-token"), "closest-token")
        self.assertEqual(_normalize_centroid_label_method("gpt"), "llm")

    def test_direct_mapping_is_ordered_by_code(self):
        tokens = ["high", "low", "middle"]
        vectors = np.asarray(
            [
                [10.0, 0.0],
                [0.0, 0.0],
                [5.0, 0.0],
            ],
            dtype=np.float32,
        )

        with patch.object(
            embedding_cluster,
            "_build_residual_codes",
            return_value=(
                np.asarray(
                    [
                        [2, 0, 0],
                        [0, 0, 0],
                        [1, 0, 0],
                    ],
                    dtype=np.int64,
                ),
                0,
            ),
        ):
            mapping = build_mapping_from_vectors(
                tokens=tokens,
                reduced_vectors=vectors,
                method="kmeans",
                depth=3,
                random_state=42,
            )

        self.assertEqual(list(mapping["direct"].keys()), ["low", "middle", "high"])
        self.assertEqual(list(mapping["reverse"].keys()), ["0:0:0", "1:0:0", "2:0:0"])

    def test_residual_collision_repair_keeps_codes_unique(self):
        codes = np.asarray(
            [
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 4],
            ],
            dtype=np.int64,
        )
        rankings = np.tile(np.arange(64, dtype=np.int16), (3, 1))
        rankings_by_level = [rankings, rankings, rankings]

        repaired, collision_repairs = _repair_code_collisions(codes, rankings_by_level)
        repaired_codes = [tuple(code.tolist()) for code in repaired]

        self.assertEqual(collision_repairs, 1)
        self.assertEqual(len(set(repaired_codes)), len(repaired_codes))

    def test_collision_report_prints_repair_table(self):
        codes = np.asarray(
            [
                [1, 2, 3],
                [1, 2, 3],
            ],
            dtype=np.int64,
        )
        rankings = np.tile(np.arange(64, dtype=np.int16), (2, 1))
        rankings_by_level = [rankings, rankings, rankings]
        output = StringIO()

        with redirect_stdout(output):
            _repair_code_collisions(
                codes,
                rankings_by_level,
                tokens=["king", "queen"],
                report=True,
            )

        report = output.getvalue()
        self.assertIn("Residual code collision repairs:", report)
        self.assertIn("'queen'", report)
        self.assertIn("1:2:3", report)
        self.assertIn("1:2:0", report)
        self.assertIn("Residual collision repair summary: 1 repaired.", report)

    def test_residual_codes_subtract_centroids_and_rename_against_first_basis(self):
        vectors = np.asarray(
            [
                [3.0, 0.0],
                [0.0, 3.0],
            ],
            dtype=np.float32,
        )
        first_centers = np.asarray(
            [
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=np.float32,
        )
        first_labels = np.asarray([0, 1], dtype=np.int64)
        residual_centers = np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )
        residual_labels = np.asarray([1, 0], dtype=np.int64)
        seen_vectors = []

        def fake_fit_cluster_level(vectors, method, random_state):
            seen_vectors.append(vectors.copy())
            if len(seen_vectors) == 1:
                return first_centers, first_labels
            return residual_centers, residual_labels

        diagnostics = {}
        with patch.object(embedding_cluster, "_fit_cluster_level", side_effect=fake_fit_cluster_level):
            codes, collision_repairs = embedding_cluster._build_residual_codes(
                reduced_vectors=vectors,
                method="kmeans",
                depth=2,
                random_state=42,
                diagnostics=diagnostics,
            )

        expected_residuals = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(seen_vectors[1], expected_residuals)
        np.testing.assert_array_equal(codes, np.asarray([[0, 0], [1, 1]], dtype=np.int64))
        self.assertEqual(len(diagnostics["levels"]), 2)
        np.testing.assert_allclose(diagnostics["levels"][0]["vectors"], vectors)
        np.testing.assert_allclose(diagnostics["levels"][1]["vectors"], expected_residuals)
        self.assertEqual(collision_repairs, 0)

    def test_embedding_cluster_rejects_unknown_cluster_methods(self):
        rng = np.random.default_rng(42)
        tokens = [f"token_{idx}" for idx in range(16)]
        vectors = rng.normal(size=(16, 4)).astype(np.float32)

        with self.assertRaisesRegex(ValueError, "Unknown cluster_method"):
            build_mapping_from_vectors(
                tokens=tokens,
                reduced_vectors=vectors,
                method="balanced-kmeans",
                depth=2,
                random_state=42,
            )

    def test_nearest_token_indices_to_centers(self):
        vectors = np.asarray(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [10.0, 10.0],
                [10.1, 10.0],
            ],
            dtype=np.float32,
        )
        centers = np.asarray(
            [
                [0.1, 0.0],
                [10.05, 10.0],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

        nearest = _nearest_token_indices_to_centers(vectors, centers, labels)

        self.assertEqual(nearest, [0, 2])

    def test_sample_plot_indices_is_deterministic(self):
        first = _sample_plot_indices(n_items=100, sample_size=10, random_state=42)
        second = _sample_plot_indices(n_items=100, sample_size=10, random_state=42)
        all_indices = _sample_plot_indices(n_items=5, sample_size=0, random_state=42)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(all_indices, np.arange(5, dtype=np.int64))
        self.assertEqual(len(first), 10)

    def test_semantic_spelling_uses_nearest_centroid_words(self):
        tokens = [" apple", " pear", " flame", " ember"]
        levels = [
            {
                "level": 0,
                "vectors": np.asarray(
                    [
                        [1.0, 0.0],
                        [1.2, 0.0],
                        [0.0, 1.0],
                        [0.0, 1.2],
                    ],
                    dtype=np.float32,
                ),
                "centers": np.asarray(
                    [
                        [1.1, 0.0],
                        [0.0, 1.1],
                    ],
                    dtype=np.float32,
                ),
                "labels": np.asarray([0, 0, 1, 1], dtype=np.int64),
                "nearest_metric": "euclidean",
            },
            {
                "level": 1,
                "vectors": np.asarray(
                    [
                        [0.0, 1.0],
                        [0.0, 1.2],
                        [1.0, 0.0],
                        [1.2, 0.0],
                    ],
                    dtype=np.float32,
                ),
                "centers": np.asarray(
                    [
                        [0.0, 1.1],
                        [1.1, 0.0],
                    ],
                    dtype=np.float32,
                ),
                "labels": np.asarray([0, 0, 1, 1], dtype=np.int64),
                "nearest_metric": "euclidean",
            },
        ]

        spelling = build_semantic_spelling_mapping(
            tokens=tokens,
            levels=levels,
            token_order=[" flame", " apple"],
        )

        self.assertEqual(list(spelling.keys()), [" flame", " apple"])
        self.assertEqual(spelling[" apple"], " apple: apple")
        self.assertEqual(spelling[" flame"], " flame: flame")

    def test_semantic_spelling_prefers_precomputed_centroid_labels(self):
        tokens = [" apple", " pear", " flame", " ember"]
        levels = [
            {
                "level": 0,
                "vectors": np.asarray(
                    [
                        [1.0, 0.0],
                        [1.2, 0.0],
                        [0.0, 1.0],
                        [0.0, 1.2],
                    ],
                    dtype=np.float32,
                ),
                "centers": np.asarray(
                    [
                        [1.1, 0.0],
                        [0.0, 1.1],
                    ],
                    dtype=np.float32,
                ),
                "labels": np.asarray([0, 0, 1, 1], dtype=np.int64),
                "centroid_labels": ["fruit", "fire"],
            },
            {
                "level": 1,
                "vectors": np.asarray(
                    [
                        [0.0, 1.0],
                        [0.0, 1.2],
                        [1.0, 0.0],
                        [1.2, 0.0],
                    ],
                    dtype=np.float32,
                ),
                "centers": np.asarray(
                    [
                        [0.0, 1.1],
                        [1.1, 0.0],
                    ],
                    dtype=np.float32,
                ),
                "labels": np.asarray([0, 0, 1, 1], dtype=np.int64),
                "centroid_labels": ["sweet", "heat"],
            },
        ]

        spelling = build_semantic_spelling_mapping(tokens=tokens, levels=levels)

        self.assertEqual(spelling[" apple"], "fruit:sweet")
        self.assertEqual(spelling[" flame"], "fire:heat")

    def test_llm_centroid_label_cleanup_removes_separator_punctuation(self):
        self.assertEqual(_clean_llm_centroid_label("Label: red fruit."), "red-fruit")
        with self.assertRaisesRegex(ValueError, "empty LLM centroid label"):
            _clean_llm_centroid_label("")

    def test_llm_centroid_label_validation_rejects_bad_semantic_atoms(self):
        with self.assertRaisesRegex(ValueError, "too generic"):
            _parse_level_centroid_labels('{"labels":{"0":"old English vocabulary"}}', 1)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _parse_level_centroid_labels('{"labels":{"0":"fruit","1":"fruit"}}', 2)
        with self.assertRaisesRegex(ValueError, "too long"):
            _parse_level_centroid_labels('{"labels":{"0":"red sweet fruit"}}', 1)

    def test_llm_label_model_reasoning_defaults(self):
        self.assertEqual(_reasoning_effort_for_label_model("gpt-5.6-terra"), "low")
        self.assertEqual(_reasoning_effort_for_label_model("gpt-5-mini"), "minimal")
        self.assertIsNone(_reasoning_effort_for_label_model("gpt-4.1-mini"))

    def test_llm_level_label_request_omits_temperature_and_bookkeeping(self):
        class FakeResponse:
            output_text = '{"labels":{"0":"Fruit","1":"Heat"}}'

        class FakeResponses:
            def __init__(self):
                self.kwargs = None

            async def create(self, **kwargs):
                self.kwargs = kwargs
                return FakeResponse()

        class FakeClient:
            def __init__(self):
                self.responses = FakeResponses()

        client = FakeClient()
        async def run_label():
            return await _label_one_level_with_llm(
                client=client,
                model="gpt-5.6-terra",
                level_idx=0,
                clusters=[
                    {"id": 0, "words": ["apple", "pear"]},
                    {"id": 1, "words": ["flame", "ember"]},
                ],
                fallback_labels=["apple", "flame"],
                semaphore=asyncio.Semaphore(1),
                max_output_tokens=4096,
            )

        result = asyncio.run(run_label())

        self.assertEqual(result, (0, ["fruit", "heat"], 0, 0, []))
        self.assertNotIn("temperature", client.responses.kwargs)
        self.assertEqual(client.responses.kwargs["reasoning"], {"effort": "low"})
        self.assertEqual(client.responses.kwargs["max_output_tokens"], 4096)
        self.assertNotIn("Residual level", client.responses.kwargs["input"])
        self.assertNotIn("Centroid id", client.responses.kwargs["input"])
        self.assertIn("apple", client.responses.kwargs["input"])

    def test_llm_level_label_repairs_bad_labels_before_fallback(self):
        class FakeResponse:
            def __init__(self, output_text):
                self.output_text = output_text

        class FakeResponses:
            def __init__(self):
                self.outputs = [
                    '{"labels":{"0":"Fruit","1":"Fruit"}}',
                    '{"labels":{"1":"Fire"}}',
                ]

            async def create(self, **kwargs):
                return FakeResponse(self.outputs.pop(0))

        class FakeClient:
            def __init__(self):
                self.responses = FakeResponses()

        async def run_label():
            return await _label_one_level_with_llm(
                client=FakeClient(),
                model="gpt-5.6-terra",
                level_idx=0,
                clusters=[
                    {"id": 0, "words": ["apple", "pear"]},
                    {"id": 1, "words": ["flame", "ember"]},
                ],
                fallback_labels=["apple", "flame"],
                semaphore=asyncio.Semaphore(1),
                max_output_tokens=4096,
            )

        result = asyncio.run(run_label())

        self.assertEqual(result, (0, ["fruit", "fire"], 0, 1, []))

    def test_empty_llm_level_label_falls_back_only_after_repair_fails(self):
        class FakeResponse:
            output_text = ""

        class FakeResponses:
            async def create(self, **kwargs):
                return FakeResponse()

        class FakeClient:
            def __init__(self):
                self.responses = FakeResponses()

        async def run_label():
            return await _label_one_level_with_llm(
                client=FakeClient(),
                model="gpt-4.1-mini",
                level_idx=0,
                clusters=[{"id": 0, "words": ["apple", "pear"]}],
                fallback_labels=["apple"],
                semaphore=asyncio.Semaphore(1),
                max_output_tokens=4096,
            )

        result = asyncio.run(run_label())

        self.assertEqual(result[:4], (0, ["apple"], 1, 0))
        self.assertIn("LLM response did not contain a JSON object", result[4][0])


if __name__ == "__main__":
    unittest.main()
