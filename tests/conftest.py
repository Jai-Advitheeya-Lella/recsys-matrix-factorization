import os

import numpy as np
import pytest

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from recsys.data import Interactions  # noqa: E402


def make_lowrank_interactions(n_users=60, n_items=40, rank=3, density=0.3, seed=0, noise=0.1):
    """Synthetic explicit ratings from a rank-``rank`` model (for recovery tests)."""
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(n_users, rank))
    Q = rng.normal(size=(n_items, rank))
    full = 3.0 + P @ Q.T / np.sqrt(rank) + noise * rng.normal(size=(n_users, n_items))
    mask = rng.random((n_users, n_items)) < density
    # make sure every user and item has at least two ratings
    mask[np.arange(n_users), rng.integers(0, n_items, n_users)] = True
    mask[rng.integers(0, n_users, n_items), np.arange(n_items)] = True
    u, i = np.nonzero(mask)
    return Interactions(
        users=u.astype(np.int32), items=i.astype(np.int32),
        ratings=full[u, i].astype(np.float32), timestamps=rng.integers(0, 10_000, len(u)),
        n_users=n_users, n_items=n_items,
    )


def make_clustered_implicit(n_users=200, n_items=100, n_clusters=4, p_in=0.4, p_out=0.02, seed=0):
    """Users and items in ``n_clusters`` blocks; users mostly consume their own block."""
    rng = np.random.default_rng(seed)
    uc = rng.integers(0, n_clusters, n_users)
    ic = np.arange(n_items) % n_clusters
    prob = np.where(uc[:, None] == ic[None, :], p_in, p_out)
    mask = rng.random((n_users, n_items)) < prob
    mask[np.arange(n_users), rng.integers(0, n_items, n_users)] = True
    u, i = np.nonzero(mask)
    return Interactions(
        users=u.astype(np.int32), items=i.astype(np.int32),
        ratings=np.ones(len(u), dtype=np.float32), timestamps=rng.integers(0, 10_000, len(u)),
        n_users=n_users, n_items=n_items,
    ), uc, ic


@pytest.fixture
def lowrank():
    return make_lowrank_interactions()


@pytest.fixture
def clustered():
    return make_clustered_implicit()
