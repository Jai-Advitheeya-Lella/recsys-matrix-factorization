r"""Weighted ALS for implicit feedback (Hu, Koren & Volinsky 2008).

Every (user, item) cell carries a binary preference $p_{ui} = 1[r_{ui} > 0]$ and a
confidence $c_{ui} = 1 + \alpha r_{ui}$. The objective over **all** cells is

.. math::
    \min_{X, Y} \sum_{u,i} c_{ui} (p_{ui} - x_u^\top y_i)^2
        + \lambda \Big(\sum_u \lVert x_u\rVert^2 + \sum_i \lVert y_i\rVert^2\Big).

Fixing $Y$, each user has the closed-form ridge solution

.. math::
    x_u = (Y^\top C^u Y + \lambda I)^{-1} Y^\top C^u p(u),
    \qquad Y^\top C^u Y = Y^\top Y + Y^\top (C^u - I) Y,

where $Y^\top Y$ is shared by all users and $C^u - I$ is non-zero only on the
$n_u$ observed items, so a user costs $O(f^2 n_u + f^3)$ instead of $O(f^2 n)$.

Two solvers are provided:

* ``"cholesky"`` – the exact solve above.
* ``"cg"`` – a few conjugate-gradient iterations warm-started at the previous
  factors (Takács, Pilászy & Tikk 2011). The matrix is never formed: a
  matrix–vector product costs $O(f^2 + f n_u)$, giving $O(f n_u)$-ish per user.
"""

from __future__ import annotations

import time

import numba as nb
import numpy as np
import scipy.sparse as sp

from ..data import Interactions


@nb.njit(parallel=True, cache=True, fastmath=True)
def _solve_cholesky(indptr, indices, conf, Y, YtY, reg, X):
    n_users, f = X.shape
    eye = np.eye(f)
    for u in nb.prange(n_users):
        start, end = indptr[u], indptr[u + 1]
        A = YtY + reg * eye
        b = np.zeros(f)
        if end > start:
            ids = indices[start:end]
            c = conf[start:end]
            Yu = Y[ids]                       # (n_u, f)
            W = np.empty_like(Yu)
            for r in range(Yu.shape[0]):
                W[r] = Yu[r] * (c[r] - 1.0)
            A += W.T @ Yu
            b = Yu.T @ c                      # p_ui = 1 on observed cells
        X[u] = np.linalg.solve(A, b)


@nb.njit(parallel=True, cache=True, fastmath=True)
def _solve_cg(indptr, indices, conf, Y, YtY, reg, X, cg_steps):
    n_users, f = X.shape
    for u in nb.prange(n_users):
        start, end = indptr[u], indptr[u + 1]
        x = X[u].copy()
        # residual r = b - A x, with A x = YtY x + sum_i (c_i - 1)(y_i.x) y_i + reg x
        r = -(YtY @ x) - reg * x
        for idx in range(start, end):
            i = indices[idx]
            c = conf[idx]
            yx = 0.0
            for k in range(f):
                yx += Y[i, k] * x[k]
            coef = c - (c - 1.0) * yx        # c * p_ui - (c-1) * y.x
            for k in range(f):
                r[k] += coef * Y[i, k]
        p = r.copy()
        rs_old = r @ r
        if rs_old < 1e-20:
            continue
        for _ in range(cg_steps):
            Ap = YtY @ p + reg * p
            for idx in range(start, end):
                i = indices[idx]
                c = conf[idx]
                yp = 0.0
                for k in range(f):
                    yp += Y[i, k] * p[k]
                coef = (c - 1.0) * yp
                for k in range(f):
                    Ap[k] += coef * Y[i, k]
            alpha = rs_old / (p @ Ap)
            x += alpha * p
            r -= alpha * Ap
            rs_new = r @ r
            if rs_new < 1e-20:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        X[u] = x


def als_half_step(
    R: sp.csr_matrix, Y: np.ndarray, X: np.ndarray, alpha: float, reg: float,
    solver: str = "cholesky", cg_steps: int = 3,
) -> None:
    """Update ``X`` in place given fixed ``Y``. ``R`` holds raw strengths r_ui (rows = X side)."""
    conf = 1.0 + alpha * R.data.astype(np.float64)
    YtY = Y.T @ Y
    indptr = R.indptr.astype(np.int64)
    indices = R.indices.astype(np.int64)
    if solver == "cholesky":
        _solve_cholesky(indptr, indices, conf, Y, YtY, reg, X)
    elif solver == "cg":
        _solve_cg(indptr, indices, conf, Y, YtY, reg, X, cg_steps)
    else:
        raise ValueError(f"unknown solver {solver!r}")


def implicit_loss(R: sp.csr_matrix, X: np.ndarray, Y: np.ndarray, alpha: float, reg: float) -> float:
    r"""Full weighted objective, computed in $O(f^2(m+n) + f\,\mathrm{nnz})$.

    Uses $\sum_{u,i} (x_u^\top y_i)^2 = \langle X^\top X, Y^\top Y\rangle_F$ for the
    unobserved cells and corrects the observed ones explicitly.
    """
    coo = R.tocoo()
    s = np.einsum("ij,ij->i", X[coo.row], Y[coo.col])
    c = 1.0 + alpha * coo.data
    all_sq = float(np.sum((X.T @ X) * (Y.T @ Y)))
    observed = float(np.sum(c * (1.0 - s) ** 2 - s**2))
    return all_sq + observed + reg * (float((X**2).sum()) + float((Y**2).sum()))


class ImplicitALS:
    """Hu et al. (2008) weighted matrix factorisation for implicit feedback."""

    def __init__(
        self,
        n_factors: int = 64,
        reg: float = 1.0,
        alpha: float = 10.0,
        n_iters: int = 15,
        solver: str = "cholesky",
        cg_steps: int = 3,
        init_std: float = 0.01,
        seed: int = 0,
        track_loss: bool = False,
        verbose: bool = False,
    ):
        self.n_factors, self.reg, self.alpha, self.n_iters = n_factors, reg, alpha, n_iters
        self.solver, self.cg_steps, self.init_std, self.seed = solver, cg_steps, init_std, seed
        self.track_loss, self.verbose = track_loss, verbose
        self.history: list[dict] = []

    def fit(self, train: Interactions | sp.csr_matrix, callback=None) -> "ImplicitALS":
        R = train.to_csr(binary=True) if isinstance(train, Interactions) else sp.csr_matrix(train)
        R = R.astype(np.float64)
        Rt = R.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.X = rng.normal(0, self.init_std, (R.shape[0], self.n_factors))
        self.Y = rng.normal(0, self.init_std, (R.shape[1], self.n_factors))
        self.history = []
        for it in range(self.n_iters):
            t0 = time.perf_counter()
            als_half_step(R, self.Y, self.X, self.alpha, self.reg, self.solver, self.cg_steps)
            als_half_step(Rt, self.X, self.Y, self.alpha, self.reg, self.solver, self.cg_steps)
            rec = {"iter": it + 1, "time": time.perf_counter() - t0}
            if self.track_loss:
                rec["loss"] = implicit_loss(R, self.X, self.Y, self.alpha, self.reg)
            if callback is not None:
                rec.update(callback(self))
            self.history.append(rec)
            if self.verbose:
                print(rec)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.X[users] @ self.Y.T

    @property
    def item_factors(self) -> np.ndarray:
        return self.Y
