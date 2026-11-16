"""Evaluation metrics.

All ranking metrics are computed with **full ranking**: every item the user has
not interacted with in training is a candidate (no sampled negatives), as
recommended by Krichene & Rendle (2020). A sampled-metric variant is provided
only to demonstrate how much it inflates scores.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def topk_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` highest scores per row, sorted descending."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    row = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[row, part], axis=1, kind="stable")
    return part[row, order]


def ranking_metrics_from_topk(
    topk: np.ndarray, test_sets: list[np.ndarray], ks: tuple[int, ...]
) -> dict[str, np.ndarray]:
    """Per-user Recall@K, NDCG@K and HitRate@K given ranked lists and held-out sets.

    * Recall@K  = |top-K ∩ test| / |test|
    * HR@K      = 1[|top-K ∩ test| > 0]
    * NDCG@K    = DCG@K / IDCG@K with binary relevance and ``1/log2(rank+1)`` discount,
      where IDCG places ``min(K, |test|)`` relevant items at the top.
    """
    kmax = max(ks)
    discounts = 1.0 / np.log2(np.arange(2, kmax + 2))
    out = {f"{m}@{k}": np.zeros(len(test_sets)) for k in ks for m in ("recall", "ndcg", "hr")}
    for u, (ranked, test) in enumerate(zip(topk, test_sets)):
        hits = np.isin(ranked[:kmax], test)
        n_rel = len(test)
        for k in ks:
            h = hits[:k]
            nh = h.sum()
            out[f"recall@{k}"][u] = nh / n_rel
            out[f"hr@{k}"][u] = float(nh > 0)
            idcg = discounts[: min(k, n_rel)].sum()
            out[f"ndcg@{k}"][u] = (discounts[:k] * h).sum() / idcg
    return out


def evaluate_ranking(
    score_fn: Callable[[np.ndarray], np.ndarray],
    train: sp.csr_matrix,
    test: sp.csr_matrix,
    ks: tuple[int, ...] = (10, 20),
    batch_size: int = 1024,
    return_topk: bool = False,
    item_popularity: np.ndarray | None = None,
) -> dict:
    """Full-ranking evaluation of a scoring function.

    Args:
        score_fn: maps an array of user indices to a dense ``(len(users), n_items)``
            score matrix.
        train: user x item CSR of training interactions; these items are excluded
            from each user's candidate set.
        test: user x item CSR of held-out interactions. Users with no test items
            are skipped.
        ks: cut-offs.
        item_popularity: training interaction counts per item; enables the
            popularity-bias diagnostics (coverage, average recommendation popularity,
            Gini of exposure, long-tail share).

    Returns:
        dict with mean metrics, plus ``per_user`` arrays and optionally ``topk``.
    """
    users = np.flatnonzero(np.diff(test.indptr) > 0)
    kmax = max(ks)
    per_user: dict[str, list[np.ndarray]] = {}
    topk_all = []
    for start in range(0, len(users), batch_size):
        ub = users[start : start + batch_size]
        scores = np.asarray(score_fn(ub), dtype=np.float64)
        # mask training items
        sub = train[ub]
        rows = np.repeat(np.arange(len(ub)), np.diff(sub.indptr))
        scores[rows, sub.indices] = -np.inf
        topk = topk_from_scores(scores, kmax)
        test_sets = [test.indices[test.indptr[u] : test.indptr[u + 1]] for u in ub]
        m = ranking_metrics_from_topk(topk, test_sets, ks)
        for key, v in m.items():
            per_user.setdefault(key, []).append(v)
        topk_all.append(topk)
    per_user_arr = {k: np.concatenate(v) for k, v in per_user.items()}
    result: dict = {k: float(v.mean()) for k, v in per_user_arr.items()}
    result["n_users"] = int(len(users))
    result["per_user"] = per_user_arr
    topk_all = np.concatenate(topk_all)
    if item_popularity is not None:
        result.update(popularity_diagnostics(topk_all, item_popularity, ks))
    if return_topk:
        result["topk"] = topk_all
        result["users"] = users
    return result


def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector (0 = perfectly equal exposure)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def popularity_diagnostics(
    topk: np.ndarray, item_popularity: np.ndarray, ks: tuple[int, ...], head_frac: float = 0.2
) -> dict[str, float]:
    """Beyond-accuracy metrics of the recommendation lists.

    * ``coverage@K``  – fraction of the catalogue recommended to at least one user.
    * ``arp@K``       – average recommendation popularity (mean training count of
      recommended items; Abdollahpouri et al. 2019).
    * ``gini@K``      – Gini coefficient of how often each item is recommended.
    * ``longtail@K``  – share of recommended slots that go to items outside the
      ``head_frac`` most popular items (the "short head").
    """
    n_items = len(item_popularity)
    order = np.argsort(-item_popularity, kind="stable")
    head = np.zeros(n_items, dtype=bool)
    head[order[: int(round(head_frac * n_items))]] = True
    out = {}
    for k in ks:
        rec = topk[:, :k].ravel()
        counts = np.bincount(rec, minlength=n_items)
        out[f"coverage@{k}"] = float((counts > 0).mean())
        out[f"arp@{k}"] = float(item_popularity[rec].mean())
        out[f"gini@{k}"] = gini(counts)
        out[f"longtail@{k}"] = float((~head[rec]).mean())
    return out


def sampled_metrics(
    score_fn: Callable[[np.ndarray], np.ndarray],
    train: sp.csr_matrix,
    test_items: np.ndarray,
    users: np.ndarray,
    n_negatives: int = 100,
    k: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """HR@K / NDCG@K when the held-out item is ranked against ``n_negatives`` random
    unobserved items (the protocol of He et al. 2017). Provided for comparison only.
    """
    rng = np.random.default_rng(seed)
    n_items = train.shape[1]
    hr, ndcg = [], []
    scores = np.asarray(score_fn(users), dtype=np.float64)
    for row, (u, t) in enumerate(zip(users, test_items)):
        seen = set(train.indices[train.indptr[u] : train.indptr[u + 1]].tolist()) | {int(t)}
        negs = []
        while len(negs) < n_negatives:
            cand = rng.integers(0, n_items, size=2 * n_negatives)
            negs.extend(c for c in cand.tolist() if c not in seen)
        negs = np.array(negs[:n_negatives])
        s_pos = scores[row, t]
        rank = int((scores[row, negs] > s_pos).sum())  # 0-based rank of the positive
        hr.append(rank < k)
        ndcg.append(1.0 / np.log2(rank + 2) if rank < k else 0.0)
    return {f"sampled_hr@{k}": float(np.mean(hr)), f"sampled_ndcg@{k}": float(np.mean(ndcg))}
