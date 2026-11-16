"""Biased MF / Funk-SVD: analytic gradients, SGD step, and low-rank recovery."""

import numpy as np
import pytest

from recsys.data import random_split
from recsys.metrics import rmse
from recsys.models import BiasBaseline, BiasedMF, GlobalMean
from recsys.models.funk_svd import _sgd_epoch, sample_loss_and_grad


def _num_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for k in range(x.size):
        e = np.zeros_like(x)
        e.flat[k] = eps
        g.flat[k] = (f(x + e) - f(x - e)) / (2 * eps)
    return g


def test_sample_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    pu, qi = rng.normal(size=5), rng.normal(size=5)
    args = dict(r=4.0, mu=3.5, reg=0.07, reg_bias=0.03)
    bu, bi = 0.2, -0.4
    _, (g_bu, g_bi, g_pu, g_qi) = sample_loss_and_grad(bu=bu, bi=bi, pu=pu, qi=qi, **args)
    loss = lambda **kw: sample_loss_and_grad(**{**dict(bu=bu, bi=bi, pu=pu, qi=qi), **args, **kw})[0]
    np.testing.assert_allclose(g_pu, _num_grad(lambda x: loss(pu=x), pu), rtol=1e-6)
    np.testing.assert_allclose(g_qi, _num_grad(lambda x: loss(qi=x), qi), rtol=1e-6)
    assert g_bu == pytest.approx(_num_grad(lambda x: loss(bu=x[0]), np.array([bu]))[0], rel=1e-6)
    assert g_bi == pytest.approx(_num_grad(lambda x: loss(bi=x[0]), np.array([bi]))[0], rel=1e-6)


def test_jit_sgd_step_is_negative_gradient_step():
    rng = np.random.default_rng(1)
    P, Q = rng.normal(size=(2, 4)), rng.normal(size=(3, 4))
    bu, bi = rng.normal(size=2), rng.normal(size=3)
    mu, lr, reg, reg_b = 3.6, 0.01, 0.05, 0.02
    u, i, r = 1, 2, 5.0
    _, (g_bu, g_bi, g_pu, g_qi) = sample_loss_and_grad(r, mu, bu[u], bi[i], P[u], Q[i], reg, reg_b)
    expect = (bu[u] - lr * g_bu, bi[i] - lr * g_bi, P[u] - lr * g_pu, Q[i] - lr * g_qi)
    _sgd_epoch(np.array([u]), np.array([i]), np.array([r]), np.array([0]), P, Q, bu, bi,
               mu, lr, reg, reg_b, True)
    np.testing.assert_allclose(bu[u], expect[0])
    np.testing.assert_allclose(bi[i], expect[1])
    np.testing.assert_allclose(P[u], expect[2])
    np.testing.assert_allclose(Q[i], expect[3])


def test_mf_recovers_low_rank_ratings(lowrank):
    tr, te = random_split(lowrank, 0.15, seed=0)
    baseline = rmse(te.ratings, GlobalMean().fit(tr).predict(te.users, te.items))
    biases = rmse(te.ratings, BiasBaseline(reg_user=1, reg_item=1).fit(tr).predict(te.users, te.items))
    mf = BiasedMF(n_factors=3, lr=0.02, reg=0.02, n_epochs=200, lr_decay=1.0, seed=0).fit(tr)
    err = rmse(te.ratings, mf.predict(te.users, te.items, clip=False))
    assert err < 0.6 * biases < baseline
    # training error should be close to the injected noise level
    assert rmse(tr.ratings, mf.predict(tr.users, tr.items, clip=False)) < 0.2


def test_funk_svd_without_bias_keeps_biases_zero(lowrank):
    mf = BiasedMF(n_factors=2, n_epochs=3, use_bias=False).fit(lowrank)
    assert np.all(mf.bu == 0) and np.all(mf.bi == 0)
