"""Wall-clock comparison of the exact (Cholesky) and conjugate-gradient ALS solvers.

Fits 10 ALS iterations on the full MovieLens-1M interaction matrix for several factor counts
(4 threads) and records time and final training objective.

    python scripts/bench_als.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.utils import limit_threads  # noqa: E402

limit_threads(4)

from recsys.data import load_movielens_1m  # noqa: E402
from recsys.models.als import ImplicitALS, implicit_loss  # noqa: E402
from recsys.utils import save_json  # noqa: E402


def main() -> None:
    ml = load_movielens_1m("data")
    R = ml.interactions.to_csr().astype("float64")
    ImplicitALS(n_factors=8, n_iters=1).fit(ml.interactions)          # JIT warm-up
    rows = []
    for f in (32, 64, 128, 256):
        for solver in ("cholesky", "cg"):
            t0 = time.perf_counter()
            m = ImplicitALS(n_factors=f, reg=25.0, alpha=1.0, n_iters=10, solver=solver, cg_steps=3, seed=0).fit(ml.interactions)
            dt = time.perf_counter() - t0
            loss = implicit_loss(R, m.X, m.Y, 1.0, 25.0)
            rows.append({"n_factors": f, "solver": solver, "seconds": dt, "objective": loss})
            print(f"f={f:3d} {solver:8s} {dt:7.2f}s objective={loss:.1f}", flush=True)
    save_json({"iterations": 10, "threads": 4, "rows": rows}, "results/als_timing.json")


if __name__ == "__main__":
    main()
