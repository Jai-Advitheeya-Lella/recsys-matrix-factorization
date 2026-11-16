"""MovieLens-1M loading and train/validation/test splitting.

The raw dataset is downloaded from grouplens.org on first use and cached under
``data/``. User and movie ids are remapped to contiguous ``0..n-1`` indices so
that every model can use dense embedding tables.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
GENRES = (
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)


@dataclass
class Interactions:
    """A set of (user, item, rating, timestamp) events with fixed index spaces."""

    users: np.ndarray
    items: np.ndarray
    ratings: np.ndarray
    timestamps: np.ndarray
    n_users: int
    n_items: int

    def __len__(self) -> int:
        return len(self.users)

    def subset(self, mask: np.ndarray) -> "Interactions":
        """Return the events selected by a boolean or integer index ``mask``."""
        return Interactions(
            self.users[mask], self.items[mask], self.ratings[mask],
            self.timestamps[mask], self.n_users, self.n_items,
        )

    def concat(self, other: "Interactions") -> "Interactions":
        """Concatenate two interaction sets that share the same index spaces."""
        assert (self.n_users, self.n_items) == (other.n_users, other.n_items)
        return Interactions(
            np.concatenate([self.users, other.users]),
            np.concatenate([self.items, other.items]),
            np.concatenate([self.ratings, other.ratings]),
            np.concatenate([self.timestamps, other.timestamps]),
            self.n_users, self.n_items,
        )

    def to_csr(self, binary: bool = True) -> sp.csr_matrix:
        """User x item CSR matrix (binary indicators, or the explicit ratings)."""
        vals = np.ones(len(self), dtype=np.float32) if binary else self.ratings.astype(np.float32)
        m = sp.csr_matrix(
            (vals, (self.users, self.items)), shape=(self.n_users, self.n_items), dtype=np.float32
        )
        m.sum_duplicates()
        if binary:
            m.data[:] = 1.0
        m.sort_indices()
        return m


@dataclass
class MovieLens:
    """MovieLens-1M with contiguous ids plus movie metadata."""

    interactions: Interactions
    titles: np.ndarray                    # (n_items,) movie titles
    genres: list[list[str]]               # per item list of genres
    raw_item_ids: np.ndarray              # contiguous index -> original MovieID
    raw_user_ids: np.ndarray              # contiguous index -> original UserID
    genre_matrix: np.ndarray = field(repr=False, default=None)  # (n_items, 18) multi-hot

    @property
    def n_users(self) -> int:
        return self.interactions.n_users

    @property
    def n_items(self) -> int:
        return self.interactions.n_items


def _download(data_dir: Path) -> Path:
    target = data_dir / "ml-1m"
    if (target / "ratings.dat").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ML1M_URL} ...")
    with urllib.request.urlopen(ML1M_URL, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(data_dir)
    return target


def load_movielens_1m(data_dir: str | Path = "data") -> MovieLens:
    """Download (if needed) and parse MovieLens-1M.

    Returns all 1,000,209 ratings of 6,040 users on 3,706 movies, with ids
    remapped to contiguous indices ordered by original id.
    """
    root = _download(Path(data_dir))
    ratings = pd.read_csv(
        root / "ratings.dat", sep="::", engine="python", header=None,
        names=["user", "item", "rating", "ts"], encoding="latin-1",
    )
    movies = pd.read_csv(
        root / "movies.dat", sep="::", engine="python", header=None,
        names=["item", "title", "genres"], encoding="latin-1",
    ).set_index("item")

    raw_users = np.sort(ratings["user"].unique())
    raw_items = np.sort(ratings["item"].unique())  # only movies that were rated
    u_map = pd.Series(np.arange(len(raw_users)), index=raw_users)
    i_map = pd.Series(np.arange(len(raw_items)), index=raw_items)

    inter = Interactions(
        users=u_map.loc[ratings["user"].values].values.astype(np.int32),
        items=i_map.loc[ratings["item"].values].values.astype(np.int32),
        ratings=ratings["rating"].values.astype(np.float32),
        timestamps=ratings["ts"].values.astype(np.int64),
        n_users=len(raw_users),
        n_items=len(raw_items),
    )
    titles = movies.loc[raw_items, "title"].values
    genres = [g.split("|") for g in movies.loc[raw_items, "genres"].values]
    gm = np.zeros((len(raw_items), len(GENRES)), dtype=np.float32)
    for j, gs in enumerate(genres):
        for g in gs:
            gm[j, GENRES.index(g)] = 1.0
    return MovieLens(inter, titles, genres, raw_items, raw_users, gm)


# ----------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------

def random_split(
    data: Interactions, test_frac: float = 0.1, seed: int = 0
) -> tuple[Interactions, Interactions]:
    """Uniformly random split of rating events (the classic explicit-feedback protocol)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data))
    n_test = int(round(test_frac * len(data)))
    return data.subset(np.sort(perm[n_test:])), data.subset(np.sort(perm[:n_test]))


def temporal_user_split(
    data: Interactions, test_frac: float = 0.2
) -> tuple[Interactions, Interactions]:
    """Per-user chronological split: the last ``test_frac`` of each user's events go to test.

    Every user keeps at least one training event and (if they have >= 2 events)
    at least one test event. Ties in timestamp are broken by item index so the
    split is deterministic.
    """
    order = np.lexsort((data.items, data.timestamps, data.users))
    users_sorted = data.users[order]
    starts = np.r_[0, np.flatnonzero(np.diff(users_sorted)) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    rank = np.arange(len(order)) - np.repeat(starts, counts)        # position within user
    n_test = np.clip(np.round(test_frac * counts).astype(int), 1, None)
    n_test = np.minimum(n_test, counts - 1)
    is_test = rank >= np.repeat(counts - n_test, counts)
    test_idx, train_idx = order[is_test], order[~is_test]
    return data.subset(np.sort(train_idx)), data.subset(np.sort(test_idx))


def leave_last_out_split(data: Interactions, n: int = 1) -> tuple[Interactions, Interactions]:
    """Hold out each user's ``n`` most recent events (the NCF protocol of He et al. 2017)."""
    order = np.lexsort((data.items, data.timestamps, data.users))
    users_sorted = data.users[order]
    ends = np.r_[np.flatnonzero(np.diff(users_sorted)) + 1, len(order)]
    starts = np.r_[0, ends[:-1]]
    counts = ends - starts
    rank = np.arange(len(order)) - np.repeat(starts, counts)
    is_test = rank >= np.repeat(counts - n, counts)
    return data.subset(np.sort(order[~is_test])), data.subset(np.sort(order[is_test]))
