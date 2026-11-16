"""Non-personalised and bias-only baselines."""

from __future__ import annotations

import numpy as np

from ..data import Interactions


class GlobalMean:
    """Predict the global mean rating for every (user, item) pair."""

    def fit(self, train: Interactions) -> "GlobalMean":
        self.mu = float(train.ratings.mean())
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return np.full(len(users), self.mu)


class BiasBaseline:
    r"""Baseline predictor $\hat r_{ui} = \mu + b_u + b_i$ (Koren 2010, Sec. 2.1).

    The biases minimise $\sum (r_{ui}-\mu-b_u-b_i)^2 + \lambda_u \sum b_u^2 + \lambda_i \sum b_i^2$
    and are fitted by exact alternating minimisation (each half-step is a
    closed-form ridge solution).
    """

    def __init__(self, reg_user: float = 15.0, reg_item: float = 10.0, n_iters: int = 10):
        self.reg_user, self.reg_item, self.n_iters = reg_user, reg_item, n_iters

    def fit(self, train: Interactions) -> "BiasBaseline":
        u, i, r = train.users, train.items, train.ratings.astype(np.float64)
        self.mu = float(r.mean())
        self.bu = np.zeros(train.n_users)
        self.bi = np.zeros(train.n_items)
        n_u = np.bincount(u, minlength=train.n_users)
        n_i = np.bincount(i, minlength=train.n_items)
        for _ in range(self.n_iters):
            self.bi = np.bincount(i, r - self.mu - self.bu[u], train.n_items) / (self.reg_item + n_i)
            self.bu = np.bincount(u, r - self.mu - self.bi[i], train.n_users) / (self.reg_user + n_u)
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return np.clip(self.mu + self.bu[users] + self.bi[items], 1.0, 5.0)


class Popularity:
    """Rank items by their number of training interactions (same list for everyone)."""

    def fit(self, train: Interactions) -> "Popularity":
        self.counts = np.bincount(train.items, minlength=train.n_items).astype(np.float64)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.counts, (len(users), len(self.counts))).copy()
