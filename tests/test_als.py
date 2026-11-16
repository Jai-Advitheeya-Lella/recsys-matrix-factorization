"""Implicit ALS: closed-form solve, CG solver, loss, monotonicity, reference agreement."""

import numpy as np
import pytest
import scipy.sparse as sp

from recsys.models.als import ImplicitALS, als_half_step, implicit_loss


def _random_problem(n_users=30, n_items=25, f=6, density=0.2, seed=0):
    rng = np.random.default_rng(seed)
    R = sp.random(n_users, n_items, density=density, format="csr", random_state=seed,
                  data_rvs=lambda n: rng.integers(1, 4, n).astype(float))
    R.sort_indices()
    Y = rng.normal(size=(n_items, f))
    X = rng.normal(size=(n_users, f))
    return R, X, Y


def _dense_solution(R, Y, alpha, reg):
    D = R.toarray()
    P = (D > 0).astype(float)
    C = 1 + alpha * D
    f = Y.shape[1]
    X = np.zeros((D.shape[0], f))
    for u in range(D.shape[0]):
        Cu = np.diag(C[u])
        X[u] = np.linalg.solve(Y.T @ Cu @ Y + reg * np.eye(f), Y.T @ Cu @ P[u])
    return X


def test_cholesky_step_matches_dense_normal_equations():
    R, X, Y = _random_problem()
    als_half_step(R, Y, X, alpha=5.0, reg=0.3, solver="cholesky")
    np.testing.assert_allclose(X, _dense_solution(R, Y, 5.0, 0.3), rtol=1e-8, atol=1e-10)


def test_cg_converges_to_exact_solution():
    R, X, Y = _random_problem(seed=1)
    exact = X.copy()
    als_half_step(R, Y, exact, alpha=5.0, reg=0.3, solver="cholesky")
    approx = X.copy()
    als_half_step(R, Y, approx, alpha=5.0, reg=0.3, solver="cg", cg_steps=Y.shape[1] + 2)
    np.testing.assert_allclose(approx, exact, rtol=1e-6, atol=1e-8)


def test_cg_few_steps_reduces_residual():
    R, X, Y = _random_problem(seed=2, f=12)
    exact = X.copy()
    als_half_step(R, Y, exact, 5.0, 0.3, "cholesky")
    errs = []
    for steps in (1, 2, 4, 8):
        approx = X.copy()
        als_half_step(R, Y, approx, 5.0, 0.3, "cg", cg_steps=steps)
        errs.append(np.linalg.norm(approx - exact))
    assert all(a >= b for a, b in zip(errs, errs[1:])) and errs[-1] < 0.1 * errs[0]


def test_loss_matches_dense_objective():
    R, X, Y = _random_problem(seed=3)
    D = R.toarray()
    P, C = (D > 0).astype(float), 1 + 2.0 * D
    dense = (C * (P - X @ Y.T) ** 2).sum() + 0.5 * ((X**2).sum() + (Y**2).sum())
    assert implicit_loss(R, X, Y, alpha=2.0, reg=0.5) == pytest.approx(dense, rel=1e-10)


@pytest.mark.parametrize("solver", ["cholesky", "cg"])
def test_als_loss_decreases_monotonically(clustered, solver):
    data, _, _ = clustered
    m = ImplicitALS(n_factors=8, reg=0.1, alpha=10, n_iters=8, solver=solver, cg_steps=3,
                    track_loss=True).fit(data)
    losses = [h["loss"] for h in m.history]
    assert all(a >= b - 1e-6 * abs(a) for a, b in zip(losses, losses[1:]))


def test_als_ranks_in_cluster_items_first(clustered):
    data, uc, ic = clustered
    m = ImplicitALS(n_factors=8, reg=0.1, alpha=10, n_iters=10).fit(data)
    scores = m.score(np.arange(data.n_users))
    top = np.argsort(-scores, axis=1)[:, :10]
    precision = (ic[top] == uc[:, None]).mean()
    assert precision > 0.9


def test_matches_reference_implicit_library():
    """One half-step equals ``implicit``'s solver when its confidence is set to 1 + alpha*r."""
    als_ref = pytest.importorskip("implicit.cpu._als")
    R, X, Y = _random_problem(seed=4, f=8)
    alpha, reg = 4.0, 0.2
    ours = X.copy()
    als_half_step(R, Y, ours, alpha, reg, "cholesky")
    Cui = R.copy()
    Cui.data = 1.0 + alpha * Cui.data
    theirs = X.copy()
    als_ref.least_squares(Cui.astype(np.float64), theirs, Y.copy(), reg, num_threads=1)
    np.testing.assert_allclose(ours, theirs, rtol=1e-6, atol=1e-8)
