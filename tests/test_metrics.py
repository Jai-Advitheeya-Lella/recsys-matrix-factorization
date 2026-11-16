import numpy as np
import pytest
import scipy.sparse as sp

from recsys.metrics import (evaluate_ranking, gini, ranking_metrics_from_topk, rmse,
                            topk_from_scores)


def test_topk_sorted_and_correct():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(5, 50))
    top = topk_from_scores(s, 7)
    ref = np.argsort(-s, axis=1)[:, :7]
    np.testing.assert_array_equal(top, ref)


def test_ndcg_hand_computed():
    # ranked list [3, 1, 4, 0]; relevant {1, 0}: hits at ranks 2 and 4
    m = ranking_metrics_from_topk(np.array([[3, 1, 4, 0]]), [np.array([0, 1])], ks=(4,))
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    idcg = 1 / np.log2(2) + 1 / np.log2(3)
    assert m["ndcg@4"][0] == pytest.approx(dcg / idcg)
    assert m["recall@4"][0] == pytest.approx(1.0)
    assert m["hr@4"][0] == 1.0


def test_perfect_and_worst_ranking():
    m = ranking_metrics_from_topk(np.array([[0, 1, 2]]), [np.array([0, 1, 2, 3, 4])], ks=(3,))
    assert m["ndcg@3"][0] == pytest.approx(1.0)       # IDCG truncated at K
    assert m["recall@3"][0] == pytest.approx(3 / 5)
    m = ranking_metrics_from_topk(np.array([[5, 6, 7]]), [np.array([0])], ks=(3,))
    assert m["ndcg@3"][0] == 0 and m["hr@3"][0] == 0


def test_evaluate_ranking_masks_training_items():
    # the oracle scorer puts training items first; masking must remove them
    train = sp.csr_matrix(np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=np.float32))
    test = sp.csr_matrix(np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 1]], dtype=np.float32))

    def score(users):
        s = np.zeros((len(users), 5))
        s[0, :2] = 10.0            # training items of user 0 – must be ignored
        s[1, 2] = 10.0             # training item of user 1 – must be ignored
        s[0, 2] = 5.0
        s[1, 4] = 5.0
        return s

    r = evaluate_ranking(score, train, test, ks=(1,))
    assert r["hr@1"] == 1.0 and r["ndcg@1"] == 1.0


def test_gini_bounds():
    assert gini(np.ones(10)) == pytest.approx(0.0)
    x = np.zeros(1000)
    x[0] = 1
    assert gini(x) == pytest.approx(1.0, abs=1e-2)


def test_rmse():
    assert rmse(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == pytest.approx(np.sqrt(2))
