import numpy as np

from recsys.data import leave_last_out_split, random_split, temporal_user_split

from conftest import make_lowrank_interactions


def test_temporal_split_is_chronological_per_user():
    data = make_lowrank_interactions(n_users=50, n_items=60, density=0.5)
    train, test = temporal_user_split(data, test_frac=0.2)
    assert len(train) + len(test) == len(data)
    for u in range(data.n_users):
        tr, te = train.timestamps[train.users == u], test.timestamps[test.users == u]
        assert len(tr) >= 1 and len(te) >= 1
        assert tr.max() <= te.min()
        n = len(tr) + len(te)
        assert abs(len(te) - 0.2 * n) <= 1


def test_leave_last_out_holds_out_latest():
    data = make_lowrank_interactions(n_users=30, n_items=40, density=0.5)
    train, test = leave_last_out_split(data)
    assert np.all(np.bincount(test.users, minlength=data.n_users) == 1)
    for u in range(data.n_users):
        assert train.timestamps[train.users == u].max() <= test.timestamps[test.users == u][0]


def test_random_split_partition():
    data = make_lowrank_interactions()
    tr, te = random_split(data, 0.1, seed=1)
    assert len(tr) + len(te) == len(data)
    pairs = set(zip(tr.users.tolist(), tr.items.tolist()))
    assert not pairs & set(zip(te.users.tolist(), te.items.tolist()))
