r"""Item-based k-nearest-neighbour recommender for implicit feedback.

Item similarity is the (optionally shrunk) cosine between binary item columns,

.. math::  s_{ij} = \frac{|U_i \cap U_j|}{\sqrt{|U_i|}\sqrt{|U_j|} + \beta},

pruned to the ``k`` most similar neighbours of every item (Sarwar et al. 2001;
Deshpande & Karypis 2004). A user's score for item $j$ is the sum of the
similarities between $j$ and the items in their history,
$\hat x_{uj} = \sum_{i \in I_u} s_{ij}$.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..data import Interactions


class ItemKNN:
    def __init__(self, k: int = 100, shrink: float = 0.0):
        self.k, self.shrink = k, shrink

    def fit(self, train: Interactions | sp.csr_matrix) -> "ItemKNN":
        X = train.to_csr(binary=True) if isinstance(train, Interactions) else sp.csr_matrix(train)
        self.X = X.astype(np.float32).tocsr()
        co = (self.X.T @ self.X).toarray()                      # co-occurrence counts
        norms = np.sqrt(np.diag(co))
        S = co / (np.outer(norms, norms) + self.shrink + 1e-12)
        np.fill_diagonal(S, 0.0)
        if self.k < S.shape[0]:
            # keep the k largest similarities in every column (neighbours of item j)
            drop = np.argpartition(-S, self.k, axis=0)[self.k :, :]
            np.put_along_axis(S, drop, 0.0, axis=0)
        self.S = S.astype(np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.asarray(self.X[users] @ self.S)
