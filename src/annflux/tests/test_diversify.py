from __future__ import annotations

import unittest

import numpy as np

from annflux.algorithms.most_needed import diversify


class TestDiversify(unittest.TestCase):
    def test_returns_correct_length(self):
        """Result should contain 1 (seed) + diversify_from items."""
        np.random.seed(42)
        features = np.random.rand(100, 16).astype(np.float32)
        result = diversify(features, diversify_from=10, fraction_others=0.5)
        self.assertEqual(len(result), 11)  # 1 seed + 10

    def test_all_indices_valid(self):
        """All returned indices should be valid indices into the features array."""
        np.random.seed(42)
        features = np.random.rand(50, 8).astype(np.float32)
        result = diversify(features, diversify_from=10, fraction_others=0.5)
        for idx in result:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(features))

    def test_no_duplicate_indices(self):
        """Each selected index should appear only once."""
        np.random.seed(42)
        features = np.random.rand(50, 8).astype(np.float32)
        result = diversify(features, diversify_from=10, fraction_others=0.5)
        self.assertEqual(len(result), len(set(result)))

    def test_starts_with_zero(self):
        """First selected index is always 0."""
        np.random.seed(42)
        features = np.random.rand(20, 4).astype(np.float32)
        result = diversify(features, diversify_from=5, fraction_others=0.5)
        self.assertEqual(result[0], 0)

    def test_selects_spread_out_points(self):
        """Given well-separated clusters, diversify should pick from different clusters."""
        np.random.seed(42)
        # 3 tight clusters far apart
        cluster_a = np.zeros((10, 2), dtype=np.float32)
        cluster_b = np.full((10, 2), 100.0, dtype=np.float32)
        cluster_c = np.full((10, 2), -100.0, dtype=np.float32)
        # Add small noise
        cluster_a += np.random.randn(10, 2).astype(np.float32) * 0.01
        cluster_b += np.random.randn(10, 2).astype(np.float32) * 0.01
        cluster_c += np.random.randn(10, 2).astype(np.float32) * 0.01
        features = np.vstack([cluster_a, cluster_b, cluster_c])

        result = diversify(features, diversify_from=2, fraction_others=1.0)
        # Should pick from at least 2 different clusters in 3 selections
        clusters_hit = set()
        for idx in result:
            if idx < 10:
                clusters_hit.add("a")
            elif idx < 20:
                clusters_hit.add("b")
            else:
                clusters_hit.add("c")
        self.assertGreaterEqual(len(clusters_hit), 2)

    def test_diversify_from_none_defaults_to_len(self):
        """When diversify_from is None, it defaults to len(features). Stops early if exhausted."""
        np.random.seed(42)
        features = np.random.rand(10, 4).astype(np.float32)
        result = diversify(features, diversify_from=None, fraction_others=1.0)
        self.assertEqual(len(result), 10)  # all 10 items selected, stops when exhausted

    def test_small_input(self):
        """Should work with a small number of features."""
        np.random.seed(42)
        features = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = diversify(features, diversify_from=2, fraction_others=1.0)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(set(result)), 3)  # all unique
        self.assertEqual(result[0], 0)


if __name__ == "__main__":
    unittest.main()
