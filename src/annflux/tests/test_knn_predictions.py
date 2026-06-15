"""
Tests that make_predictions and make_predictions_alt produce identical outputs,
focusing on multi-label cases.
"""
import numpy as np
import pandas

from annflux.training.annflux.quick import make_predictions, make_predictions_alt, make_predictions_fast


def _make_data(n=10):
    cols = [
        "label_predicted", "score_predicted", "scores_predicted",
        "label_possible", "score_possible", "num_labeled_nn", "min_distance",
    ]
    df = pandas.DataFrame({c: [None] * n for c in cols})
    df["uid"] = [f"uid_{i}" for i in range(n)]
    return df


def _run_both(indices, distances, train_labels, skip_first=False, knn_rank_exponent=6):
    N = len(indices)
    data_indices = list(range(N))

    data_alt = _make_data(N)
    make_predictions_alt(
        data_alt, indices, distances, train_labels, data_indices,
        skip_first=skip_first, knn_rank_exponent=knn_rank_exponent,
    )

    data_new = _make_data(N)
    make_predictions_fast(
        data_new, indices, distances, train_labels, data_indices,
        skip_first=skip_first, knn_rank_exponent=knn_rank_exponent,
    )

    return data_alt, data_new


def _empty(v):
    return v is None or v == ""


def _label_prob_dict(labels_str, probs_str):
    """Parse paired label/probability strings into a {label: float} dict."""
    if _empty(labels_str) or _empty(probs_str):
        return {}
    labels = str(labels_str).split(",")
    probs = [float(p) for p in str(probs_str).split(",")]
    assert len(labels) == len(probs), f"label/prob length mismatch: {labels_str!r} {probs_str!r}"
    return dict(zip(labels, probs))


def _assert_label_probs_equal(labels_a, probs_a, labels_b, probs_b, row, context, tol=0.01):
    """Assert two label+probability pairs are numerically equivalent (order-independent)."""
    dict_a = _label_prob_dict(labels_a, probs_a)
    dict_b = _label_prob_dict(labels_b, probs_b)
    assert set(dict_a.keys()) == set(dict_b.keys()), \
        f"row {row} {context}: label sets differ alt={set(dict_a.keys())} new={set(dict_b.keys())}"
    for lbl in dict_a:
        assert abs(dict_a[lbl] - dict_b[lbl]) < tol, \
            f"row {row} {context} label '{lbl}': alt={dict_a[lbl]:.4f} new={dict_b[lbl]:.4f}"


def _assert_cols_equal(data_alt, data_new, cols=None):
    if cols is None:
        cols = ["label_predicted", "scores_predicted", "label_possible"]
    for i in range(len(data_alt)):
        row_a, row_b = data_alt.iloc[i], data_new.iloc[i]
        for col in cols:
            a, b = row_a[col], row_b[col]
            if col == "scores_predicted":
                _assert_label_probs_equal(
                    row_a["label_predicted"], a,
                    row_b["label_predicted"], b,
                    i, "scores_predicted",
                )
            elif col == "score_possible":
                _assert_label_probs_equal(
                    row_a["label_possible"], a,
                    row_b["label_possible"], b,
                    i, "score_possible",
                )
            elif col in ("label_predicted", "label_possible"):
                norm_a = None if _empty(a) else ",".join(sorted(str(a).split(",")))
                norm_b = None if _empty(b) else ",".join(sorted(str(b).split(",")))
                assert norm_a == norm_b, f"row {i} col '{col}': alt={a!r} new={b!r}"
            else:
                assert a == b, f"row {i} col '{col}': alt={a!r} new={b!r}"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _indices_distances(N, k, M, rng):
    """Random (N, k) neighbor indices into [0, M) with distances in (0, 1]."""
    indices = rng.integers(0, M, size=(N, k)).astype(np.int64)
    distances = rng.uniform(0.01, 1.0, size=(N, k)).astype(np.float32)
    return indices, distances


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMakePredictionsMultiLabel:

    def test_single_label_only(self):
        """All neighbors have exactly one label — basic sanity check."""
        rng = np.random.default_rng(0)
        M = 20
        train_labels = np.array(
            [["cat"] if i % 2 == 0 else ["dog"] for i in range(M)], dtype=object
        )
        indices, distances = _indices_distances(10, 5, M, rng)
        data_alt, data_new = _run_both(indices, distances, train_labels)
        _assert_cols_equal(data_alt, data_new,
                           ["label_predicted", "scores_predicted", "label_possible"])

    def test_multi_label_neighbors(self):
        """Some neighbors carry two labels; verify multi-label prediction matches."""
        rng = np.random.default_rng(1)
        M = 30
        # half neighbors have two labels
        train_labels = np.array(
            [["A", "B"] if i % 3 == 0 else ["A"] if i % 3 == 1 else ["B"]
             for i in range(M)],
            dtype=object,
        )
        indices, distances = _indices_distances(10, 8, M, rng)
        data_alt, data_new = _run_both(indices, distances, train_labels)
        _assert_cols_equal(data_alt, data_new,
                           ["label_predicted", "scores_predicted",
                            "label_possible", "score_possible"])

    def test_mixed_labeled_unlabeled(self):
        """Some neighbors are unlabeled (None)."""
        rng = np.random.default_rng(2)
        M = 25
        train_labels = np.array(
            [None if i % 4 == 0 else ["X", "Y"] if i % 4 == 1 else ["X"]
             for i in range(M)],
            dtype=object,
        )
        indices, distances = _indices_distances(10, 6, M, rng)
        data_alt, data_new = _run_both(indices, distances, train_labels)
        _assert_cols_equal(data_alt, data_new,
                           ["label_predicted", "scores_predicted", "label_possible"])

    def test_skip_first(self):
        """skip_first=True (used when predicting on labeled samples)."""
        rng = np.random.default_rng(3)
        M = 20
        train_labels = np.array(
            [["P"] if i % 2 == 0 else ["Q", "R"] for i in range(M)], dtype=object
        )
        indices, distances = _indices_distances(8, 5, M, rng)
        data_alt, data_new = _run_both(indices, distances, train_labels, skip_first=True)
        _assert_cols_equal(data_alt, data_new,
                           ["label_predicted", "scores_predicted", "label_possible"])

    def test_score_predicted_close(self):
        """score_predicted should match to 2 decimal places for all rows."""
        rng = np.random.default_rng(4)
        M = 40
        train_labels = np.array(
            [["A", "B"] if i % 5 == 0 else ["A"] if i % 5 < 3 else ["B"]
             for i in range(M)],
            dtype=object,
        )
        indices, distances = _indices_distances(15, 10, M, rng)
        data_alt, data_new = _run_both(indices, distances, train_labels)
        for i, (a, b) in enumerate(zip(data_alt["score_predicted"], data_new["score_predicted"])):
            if a is None:
                assert b is None or float(b) == 0.0, f"row {i}: alt=None new={b}"
            else:
                assert abs(float(a) - float(b)) < 0.01, \
                    f"row {i} score_predicted: alt={a:.4f} new={b:.4f}"

    def test_num_labeled_nn(self):
        """num_labeled_nn counts must match exactly."""
        rng = np.random.default_rng(5)
        M = 20
        train_labels = np.array(
            [None if i % 3 == 0 else ["C"] for i in range(M)], dtype=object
        )
        indices, distances = _indices_distances(10, 7, M, rng)
        data_alt, data_new = _run_both(indices, distances, train_labels)
        for i, (a, b) in enumerate(zip(data_alt["num_labeled_nn"], data_new["num_labeled_nn"])):
            assert int(a) == int(b), f"row {i} num_labeled_nn: alt={a} new={b}"
