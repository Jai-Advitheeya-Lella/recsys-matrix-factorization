r"""Bayesian Personalised Ranking with a matrix-factorisation scorer (Rendle et al. 2009).

For a user $u$, an observed item $i$ and an unobserved item $j$,
$\hat x_{uij} = \hat x_{ui} - \hat x_{uj}$ with $\hat x_{ui} = b_i + w_u^\top h_i$.
BPR-Opt maximises

.. math::  \sum_{(u,i,j)} \ln \sigma(\hat x_{uij}) - \lambda \lVert\Theta\rVert^2

with LearnBPR: stochastic gradient ascent on triples drawn by bootstrap sampling
($(u,i)$ uniformly from the observed pairs, $j$ uniformly from the items $u$ has
not interacted with). The sampling + update loop is JIT-compiled with numba.
"""

from __future__ import annotations

import time

import numba as nb
import numpy as np
import scipy.sparse as sp

from ..data import Interactions


@nb.njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@nb.njit(cache=True, fastmath=True)
def _is_observed(indptr, indices, u, j):
    lo, hi = indptr[u], indptr[u + 1]
    # binary search in the sorted row
    while lo < hi:
        mid = (lo + hi) // 2
        v = indices[mid]
        if v == j:
            return True
        if v < j:
            lo = mid + 1
        else:
            hi = mid
    return False


@nb.njit(cache=True, fastmath=True)
def _bpr_epoch(indptr, indices, pos_u, pos_i, n_samples, W, H, b, lr, reg, reg_bias):
    n_items, k = H.shape
    loss = 0.0
    for _ in range(n_samples):
        s = np.random.randint(len(pos_u))
        u = pos_u[s]
        i = pos_i[s]
        j = np.random.randint(n_items)
        while _is_observed(indptr, indices, u, j):
            j = np.random.randint(n_items)
        x = b[i] - b[j]
        for f in range(k):
            x += W[u, f] * (H[i, f] - H[j, f])
        # d/dx ln sigma(x) = sigma(-x)
        if x > 0:
            ex = np.exp(-x)
            g = ex / (1.0 + ex)
            loss += np.log1p(ex)
        else:
            ex = np.exp(x)
            g = 1.0 / (1.0 + ex)
            loss += -x + np.log1p(ex)
        for f in range(k):
            wu = W[u, f]
            hi = H[i, f]
            hj = H[j, f]
            W[u, f] += lr * (g * (hi - hj) - reg * wu)
            H[i, f] += lr * (g * wu - reg * hi)
            H[j, f] += lr * (-g * wu - reg * hj)
        b[i] += lr * (g - reg_bias * b[i])
        b[j] += lr * (-g - reg_bias * b[j])
    return loss / n_samples


def triple_loss_and_grad(wu, hi, hj, bi, bj, reg, reg_bias):
    """Negative BPR objective for one triple and its gradient (used by the tests).

    ``-ln σ(x_uij) + reg/2 (|w_u|² + |h_i|² + |h_j|²) + reg_bias/2 (b_i² + b_j²)``.
    """
    x = bi - bj + wu @ (hi - hj)
    loss = np.log1p(np.exp(-x)) + 0.5 * reg * (wu @ wu + hi @ hi + hj @ hj) + 0.5 * reg_bias * (bi**2 + bj**2)
    g = 1.0 / (1.0 + np.exp(x))  # sigma(-x)
    grads = (-g * (hi - hj) + reg * wu, -g * wu + reg * hi, g * wu + reg * hj, -g + reg_bias * bi, g + reg_bias * bj)
    return loss, grads


class BPRMF:
    """BPR-optimised matrix factorisation with an item bias."""

    def __init__(
        self,
        n_factors: int = 64,
        lr: float = 0.05,
        reg: float = 0.002,
        reg_bias: float = 0.0,
        n_epochs: int = 60,
        init_std: float = 0.1,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.n_factors, self.lr, self.reg, self.reg_bias = n_factors, lr, reg, reg_bias
        self.n_epochs, self.init_std, self.seed, self.verbose = n_epochs, init_std, seed, verbose
        self.history: list[dict] = []

    def fit(self, train: Interactions | sp.csr_matrix, callback=None, eval_every: int = 5) -> "BPRMF":
        """Run ``n_epochs`` epochs of ``nnz`` sampled triples each.

        ``callback(model) -> dict`` is invoked every ``eval_every`` epochs; if it
        returns a ``"score"`` key the best-scoring parameters are kept (early
        stopping on a validation metric).
        """
        R = train.to_csr(binary=True) if isinstance(train, Interactions) else sp.csr_matrix(train)
        R.sort_indices()
        coo = R.tocoo()
        rng = np.random.default_rng(self.seed)
        _seed(self.seed)
        self.W = rng.normal(0, self.init_std, (R.shape[0], self.n_factors))
        self.H = rng.normal(0, self.init_std, (R.shape[1], self.n_factors))
        self.b = np.zeros(R.shape[1])
        indptr, indices = R.indptr.astype(np.int64), R.indices.astype(np.int64)
        pos_u, pos_i = coo.row.astype(np.int64), coo.col.astype(np.int64)
        best, best_state = -np.inf, None
        self.history = []
        for epoch in range(self.n_epochs):
            t0 = time.perf_counter()
            loss = _bpr_epoch(indptr, indices, pos_u, pos_i, R.nnz, self.W, self.H, self.b,
                              self.lr, self.reg, self.reg_bias)
            rec = {"epoch": epoch + 1, "bpr_loss": float(loss), "time": time.perf_counter() - t0}
            if callback is not None and ((epoch + 1) % eval_every == 0 or epoch + 1 == self.n_epochs):
                rec.update(callback(self))
                if "score" in rec and rec["score"] > best:
                    best = rec["score"]
                    best_state = (self.W.copy(), self.H.copy(), self.b.copy(), epoch + 1)
            self.history.append(rec)
            if self.verbose:
                print(rec)
        if best_state is not None:
            self.W, self.H, self.b, self.best_epoch = best_state
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.W[users] @ self.H.T + self.b[None, :]

    @property
    def item_factors(self) -> np.ndarray:
        return self.H
