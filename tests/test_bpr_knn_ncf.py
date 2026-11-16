"""BPR gradients and sampling, item-kNN similarity, and NCF sanity checks."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from recsys.metrics import evaluate_ranking
from recsys.models import BPRMF, ItemKNN, Popularity
from recsys.models.bpr import _bpr_epoch, _is_observed, _seed, triple_loss_and_grad
from recsys.models.ncf import NCFRecommender, NeuMF, GMF, MLP, sample_negatives


def _num_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for k in range(x.size):
        e = np.zeros_like(x)
        e.flat[k] = eps
        g.flat[k] = (f(x + e) - f(x - e)) / (2 * eps)
    return g


def test_bpr_triple_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    wu, hi, hj = rng.normal(size=(3, 6))
    bi, bj, reg, rb = 0.3, -0.2, 0.05, 0.01
    base = dict(wu=wu, hi=hi, hj=hj, bi=bi, bj=bj, reg=reg, reg_bias=rb)
    _, grads = triple_loss_and_grad(**base)
    f = lambda **kw: triple_loss_and_grad(**{**base, **kw})[0]
    np.testing.assert_allclose(grads[0], _num_grad(lambda x: f(wu=x), wu), rtol=1e-5)
    np.testing.assert_allclose(grads[1], _num_grad(lambda x: f(hi=x), hi), rtol=1e-5)
    np.testing.assert_allclose(grads[2], _num_grad(lambda x: f(hj=x), hj), rtol=1e-5)
    assert grads[3] == pytest.approx(_num_grad(lambda x: f(bi=x[0]), np.array([bi]))[0], rel=1e-5)
    assert grads[4] == pytest.approx(_num_grad(lambda x: f(bj=x[0]), np.array([bj]))[0], rel=1e-5)


def test_bpr_binary_search_membership():
    R = sp.random(20, 50, density=0.3, format="csr", random_state=0)
    R.sort_indices()
    D = R.toarray() > 0
    for u in range(20):
        for j in range(50):
            assert _is_observed(R.indptr.astype(np.int64), R.indices.astype(np.int64), u, j) == D[u, j]


def test_bpr_single_triple_update_is_gradient_ascent():
    # with one user who has seen every item but one, the negative item is forced
    R = sp.csr_matrix(np.array([[1, 1, 0]], dtype=np.float32))
    rng = np.random.default_rng(0)
    W, H, b = rng.normal(size=(1, 4)), rng.normal(size=(3, 4)), rng.normal(size=3)
    _seed(0)
    # pick positive item index 0 deterministically by passing a single positive pair
    W0, H0, b0 = W.copy(), H.copy(), b.copy()
    lr, reg, rb = 0.1, 0.01, 0.0
    _bpr_epoch(R.indptr.astype(np.int64), R.indices.astype(np.int64), np.array([0]), np.array([0]),
               1, W, H, b, lr, reg, rb)
    _, (gw, gi, gj, gbi, gbj) = triple_loss_and_grad(W0[0], H0[0], H0[2], b0[0], b0[2], reg, rb)
    np.testing.assert_allclose(W[0], W0[0] - lr * gw)
    np.testing.assert_allclose(H[0], H0[0] - lr * gi)
    np.testing.assert_allclose(H[2], H0[2] - lr * gj)
    np.testing.assert_allclose(H[1], H0[1])
    np.testing.assert_allclose(b[[0, 2]], [b0[0] - lr * gbi, b0[2] - lr * gbj])


def test_bpr_beats_popularity_on_clustered_data(clustered):
    data, uc, ic = clustered
    from recsys.data import temporal_user_split
    tr, te = temporal_user_split(data, 0.2)
    Rtr, Rte = tr.to_csr(), te.to_csr()
    bpr = BPRMF(n_factors=8, lr=0.05, reg=0.01, n_epochs=40, seed=0).fit(tr)
    pop = Popularity().fit(tr)
    s_bpr = evaluate_ranking(bpr.score, Rtr, Rte, ks=(10,))["ndcg@10"]
    s_pop = evaluate_ranking(pop.score, Rtr, Rte, ks=(10,))["ndcg@10"]
    assert s_bpr > 1.5 * s_pop


def test_item_knn_cosine_matches_sklearn(clustered):
    from sklearn.metrics.pairwise import cosine_similarity
    data, _, _ = clustered
    knn = ItemKNN(k=data.n_items).fit(data)
    X = data.to_csr().toarray()
    ref = cosine_similarity(X.T)
    np.fill_diagonal(ref, 0)
    np.testing.assert_allclose(knn.S, ref, atol=1e-5)


def test_item_knn_keeps_k_neighbours(clustered):
    data, _, _ = clustered
    knn = ItemKNN(k=5).fit(data)
    assert np.all((knn.S > 0).sum(axis=0) <= 5)


def test_negative_sampler_never_returns_observed():
    g = torch.Generator().manual_seed(0)
    obs = torch.rand(10, 30, generator=g) < 0.7
    pos_u = torch.arange(10).repeat(20)
    j = sample_negatives(pos_u, obs, 30, 3, g)
    assert not obs[pos_u.repeat_interleave(3), j].any()


def test_neumf_pretrained_init_reproduces_weighted_branches():
    torch.manual_seed(0)
    gmf, mlp = GMF(7, 9, 4), MLP(7, 9, (8, 4))
    net = NeuMF(7, 9, 4, (8, 4))
    net.load_pretrained(gmf, mlp, alpha=0.5)
    u, i = torch.arange(7), torch.arange(7)
    expected = 0.5 * gmf(u, i) + 0.5 * mlp(u, i)
    torch.testing.assert_close(net(u, i), expected)


@pytest.mark.parametrize("arch", ["gmf", "mlp", "neumf"])
def test_ncf_learns_clusters(clustered, arch):
    data, uc, ic = clustered
    torch.set_num_threads(2)
    m = NCFRecommender(arch=arch, gmf_dim=8, layers=(16, 8), n_epochs=40, batch_size=256,
                       lr=1e-2, device="cpu").fit(data)
    scores = m.score(np.arange(data.n_users))
    top = np.argsort(-scores, axis=1)[:, :10]
    assert (ic[top] == uc[:, None]).mean() > 0.6
    assert m.history[-1]["loss"] < m.history[0]["loss"]
