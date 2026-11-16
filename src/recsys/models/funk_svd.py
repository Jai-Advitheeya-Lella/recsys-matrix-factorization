r"""Funk-SVD / biased matrix factorisation trained with per-sample SGD.

Model (Koren, Bell & Volinsky 2009, eq. 3):

.. math::  \hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i

Per observed rating the SGD step descends the sample loss

.. math::  \ell_{ui} = \tfrac12 (r_{ui}-\hat r_{ui})^2
           + \tfrac{\lambda}{2}(\lVert p_u\rVert^2 + \lVert q_i\rVert^2)
           + \tfrac{\lambda_b}{2}(b_u^2 + b_i^2).

With ``use_bias=False`` the model is the original Funk (2006) factorisation
$\hat r_{ui} = \mu + p_u^\top q_i$. The inner loop is JIT-compiled with numba.
"""

from __future__ import annotations

import time

import numba as nb
import numpy as np

from ..data import Interactions
from ..metrics import rmse


@nb.njit(cache=True, fastmath=True)
def _sgd_epoch(users, items, ratings, perm, P, Q, bu, bi, mu, lr, reg, reg_bias, use_bias):
    k = P.shape[1]
    sq_err = 0.0
    for idx in perm:
        u = users[idx]
        i = items[idx]
        pred = mu + bu[u] + bi[i]
        for f in range(k):
            pred += P[u, f] * Q[i, f]
        e = ratings[idx] - pred
        sq_err += e * e
        if use_bias:
            bu[u] += lr * (e - reg_bias * bu[u])
            bi[i] += lr * (e - reg_bias * bi[i])
        for f in range(k):
            puf = P[u, f]
            qif = Q[i, f]
            P[u, f] += lr * (e * qif - reg * puf)
            Q[i, f] += lr * (e * puf - reg * qif)
    return np.sqrt(sq_err / len(perm))


def sample_loss_and_grad(r, mu, bu, bi, pu, qi, reg, reg_bias):
    """Loss :math:`\\ell_{ui}` and its gradient for one rating (used by the tests).

    Returns ``(loss, (d_bu, d_bi, d_pu, d_qi))``.
    """
    e = r - (mu + bu + bi + pu @ qi)
    loss = 0.5 * e**2 + 0.5 * reg * (pu @ pu + qi @ qi) + 0.5 * reg_bias * (bu**2 + bi**2)
    return loss, (-e + reg_bias * bu, -e + reg_bias * bi, -e * qi + reg * pu, -e * pu + reg * qi)


class BiasedMF:
    """Biased MF (``use_bias=True``) or Funk-SVD (``use_bias=False``) with SGD."""

    def __init__(
        self,
        n_factors: int = 100,
        lr: float = 0.01,
        reg: float = 0.05,
        reg_bias: float = 0.01,
        n_epochs: int = 40,
        lr_decay: float = 0.95,
        init_std: float = 0.1,
        use_bias: bool = True,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.n_factors, self.lr, self.reg, self.reg_bias = n_factors, lr, reg, reg_bias
        self.n_epochs, self.lr_decay, self.init_std = n_epochs, lr_decay, init_std
        self.use_bias, self.seed, self.verbose = use_bias, seed, verbose
        self.history: list[dict] = []

    def fit(self, train: Interactions, val: Interactions | None = None) -> "BiasedMF":
        rng = np.random.default_rng(self.seed)
        self.mu = float(train.ratings.mean())
        self.P = rng.normal(0, self.init_std, (train.n_users, self.n_factors))
        self.Q = rng.normal(0, self.init_std, (train.n_items, self.n_factors))
        self.bu = np.zeros(train.n_users)
        self.bi = np.zeros(train.n_items)
        users = train.users.astype(np.int64)
        items = train.items.astype(np.int64)
        ratings = train.ratings.astype(np.float64)
        lr = self.lr
        self.history = []
        for epoch in range(self.n_epochs):
            t0 = time.perf_counter()
            perm = rng.permutation(len(train))
            tr = _sgd_epoch(users, items, ratings, perm, self.P, self.Q, self.bu, self.bi,
                            self.mu, lr, self.reg, self.reg_bias, self.use_bias)
            rec = {"epoch": epoch + 1, "train_rmse_running": float(tr), "time": time.perf_counter() - t0}
            if val is not None:
                rec["val_rmse"] = rmse(val.ratings, self.predict(val.users, val.items))
            self.history.append(rec)
            if self.verbose:
                print(rec)
            lr *= self.lr_decay
        return self

    def predict(self, users: np.ndarray, items: np.ndarray, clip: bool = True) -> np.ndarray:
        pred = self.mu + self.bu[users] + self.bi[items] + np.einsum("ij,ij->i", self.P[users], self.Q[items])
        return np.clip(pred, 1.0, 5.0) if clip else pred

    def score(self, users: np.ndarray) -> np.ndarray:
        """Predicted rating for every item (used to rank with an explicit model)."""
        return self.mu + self.bu[users, None] + self.bi[None, :] + self.P[users] @ self.Q.T

    @property
    def item_factors(self) -> np.ndarray:
        return self.Q
