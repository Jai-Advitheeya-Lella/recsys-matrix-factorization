r"""Neural Collaborative Filtering (He et al. 2017) in PyTorch.

Three scorers share one training loop (pointwise log-loss with sampled negatives):

* **GMF** – $\hat y_{ui} = \sigma(h^\top (p_u \odot q_i))$, a generalised MF.
* **MLP** – $\hat y_{ui} = \sigma(h^\top \phi_L(\dots\phi_1([p_u; q_i])))$ with a ReLU tower.
* **NeuMF** – concatenates the last GMF and MLP layers before the output, with
  separate embeddings for each branch; optionally initialised from pre-trained
  GMF and MLP models, as in Sec. 3.4.1 of the paper.

Each epoch pairs every observed (u, i) with ``n_neg`` items drawn uniformly
from the items u has not interacted with.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import Interactions


class GMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 32):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        self.out = nn.Linear(dim, 1)
        nn.init.normal_(self.user.weight, std=0.01)
        nn.init.normal_(self.item.weight, std=0.01)

    def features(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return self.user(u) * self.item(i)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return self.out(self.features(u, i)).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, n_users: int, n_items: int, layers: tuple[int, ...] = (128, 64, 32), dropout: float = 0.0):
        super().__init__()
        self.user = nn.Embedding(n_users, layers[0] // 2)
        self.item = nn.Embedding(n_items, layers[0] // 2)
        mods: list[nn.Module] = []
        for a, b in zip(layers[:-1], layers[1:]):
            mods += [nn.Linear(a, b), nn.ReLU()]
            if dropout > 0:
                mods.append(nn.Dropout(dropout))
        self.tower = nn.Sequential(*mods)
        self.out = nn.Linear(layers[-1], 1)
        nn.init.normal_(self.user.weight, std=0.01)
        nn.init.normal_(self.item.weight, std=0.01)

    def features(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return self.tower(torch.cat([self.user(u), self.item(i)], dim=-1))

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return self.out(self.features(u, i)).squeeze(-1)


class NeuMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, gmf_dim: int = 32,
                 layers: tuple[int, ...] = (128, 64, 32), dropout: float = 0.0):
        super().__init__()
        self.gmf = GMF(n_users, n_items, gmf_dim)
        self.mlp = MLP(n_users, n_items, layers, dropout)
        self.out = nn.Linear(gmf_dim + layers[-1], 1)

    def load_pretrained(self, gmf: GMF, mlp: MLP, alpha: float = 0.5) -> None:
        """Initialise from pre-trained branches; output layer = [α h_gmf ; (1-α) h_mlp]."""
        self.gmf.load_state_dict(gmf.state_dict())
        self.mlp.load_state_dict(mlp.state_dict())
        with torch.no_grad():
            self.out.weight.copy_(torch.cat([alpha * gmf.out.weight, (1 - alpha) * mlp.out.weight], dim=1))
            self.out.bias.copy_(alpha * gmf.out.bias + (1 - alpha) * mlp.out.bias)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.gmf.features(u, i), self.mlp.features(u, i)], dim=-1)
        return self.out(h).squeeze(-1)


def sample_negatives(
    pos_u: torch.Tensor, observed: torch.Tensor, n_items: int, n_neg: int, gen: torch.Generator
) -> torch.Tensor:
    """For every positive user index draw ``n_neg`` items the user has not seen.

    ``observed`` is a dense (n_users, n_items) bool tensor; collisions are re-drawn
    until none remain (fast because ML-1M is ~95% sparse).
    """
    u = pos_u.repeat_interleave(n_neg)
    j = torch.randint(0, n_items, (len(u),), device=u.device, generator=gen)
    bad = observed[u, j]
    while bad.any():
        j[bad] = torch.randint(0, n_items, (int(bad.sum()),), device=u.device, generator=gen)
        bad = observed[u, j]
    return j


class NCFRecommender:
    """Training / scoring wrapper around :class:`GMF`, :class:`MLP` or :class:`NeuMF`."""

    def __init__(
        self,
        arch: str = "neumf",
        gmf_dim: int = 32,
        layers: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.0,
        n_neg: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 2048,
        n_epochs: int = 20,
        optimizer: str = "adam",
        seed: int = 0,
        device: str | None = None,
        pretrained: tuple["NCFRecommender", "NCFRecommender"] | None = None,
        verbose: bool = False,
    ):
        self.arch, self.gmf_dim, self.layers, self.dropout = arch, gmf_dim, tuple(layers), dropout
        self.n_neg, self.lr, self.weight_decay = n_neg, lr, weight_decay
        self.batch_size, self.n_epochs, self.optimizer = batch_size, n_epochs, optimizer
        self.seed, self.pretrained, self.verbose = seed, pretrained, verbose
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.history: list[dict] = []

    def _build(self, n_users: int, n_items: int) -> nn.Module:
        if self.arch == "gmf":
            return GMF(n_users, n_items, self.gmf_dim)
        if self.arch == "mlp":
            return MLP(n_users, n_items, self.layers, self.dropout)
        if self.arch == "neumf":
            m = NeuMF(n_users, n_items, self.gmf_dim, self.layers, self.dropout)
            if self.pretrained is not None:
                g, p = self.pretrained
                m.load_pretrained(g.model.cpu(), p.model.cpu())
            return m
        raise ValueError(self.arch)

    def fit(
        self, train: Interactions | sp.csr_matrix,
        callback: Callable[["NCFRecommender"], dict] | None = None,
    ) -> "NCFRecommender":
        """Train; if ``callback`` returns a ``"score"`` the best epoch's weights are kept."""
        torch.manual_seed(self.seed)
        R = train.to_csr(binary=True) if isinstance(train, Interactions) else sp.csr_matrix(train)
        n_users, n_items = R.shape
        self.n_items = n_items
        self.model = self._build(n_users, n_items).to(self.device)
        coo = R.tocoo()
        pos_u = torch.as_tensor(coo.row, dtype=torch.long, device=self.device)
        pos_i = torch.as_tensor(coo.col, dtype=torch.long, device=self.device)
        observed = torch.as_tensor(R.toarray() > 0, device=self.device)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed)
        if self.optimizer == "adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state = -np.inf, None
        self.history = []
        for epoch in range(self.n_epochs):
            t0 = time.perf_counter()
            self.model.train()
            neg_i = sample_negatives(pos_u, observed, n_items, self.n_neg, gen)
            u_all = torch.cat([pos_u, pos_u.repeat_interleave(self.n_neg)])
            i_all = torch.cat([pos_i, neg_i])
            y_all = torch.cat([torch.ones_like(pos_u), torch.zeros_like(neg_i)]).float()
            perm = torch.randperm(len(u_all), device=self.device, generator=gen)
            total, n_batches = 0.0, 0
            for s in range(0, len(perm), self.batch_size):
                idx = perm[s : s + self.batch_size]
                logits = self.model(u_all[idx], i_all[idx])
                loss = F.binary_cross_entropy_with_logits(logits, y_all[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach())
                n_batches += 1
            rec = {"epoch": epoch + 1, "loss": total / n_batches, "time": time.perf_counter() - t0}
            if callback is not None:
                rec.update(callback(self))
                if "score" in rec and rec["score"] > best:
                    best = rec["score"]
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    self.best_epoch = epoch + 1
            self.history.append(rec)
            if self.verbose:
                print(rec)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def score(self, users: np.ndarray, chunk: int = 128) -> np.ndarray:
        """Logits for every item, for each user in ``users`` (full ranking)."""
        self.model.eval()
        items = torch.arange(self.n_items, device=self.device)
        out = []
        for s in range(0, len(users), chunk):
            ub = torch.as_tensor(np.asarray(users[s : s + chunk]), dtype=torch.long, device=self.device)
            uu = ub.repeat_interleave(self.n_items)
            ii = items.repeat(len(ub))
            out.append(self.model(uu, ii).view(len(ub), self.n_items).float().cpu().numpy())
        return np.concatenate(out)

    @property
    def item_factors(self) -> np.ndarray:
        m = self.model.gmf if self.arch == "neumf" else self.model
        return m.item.weight.detach().cpu().numpy()
