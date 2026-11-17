"""Explicit-feedback benchmark: RMSE on a random 90/10 split of MovieLens-1M.

Protocol
--------
1. Hold out 10% of ratings uniformly at random as the test set (seed 0).
2. Hold out another 10% of the remaining ratings as validation; grid-search each
   model on validation RMSE (the number of SGD epochs is chosen per config).
3. Refit the best config on the full 90% training set and report test RMSE,
   mean ± std over three initialisation seeds.

Usage::

    python scripts/run_explicit.py --out results/explicit.json
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.utils import limit_threads  # noqa: E402

limit_threads(4)

import numpy as np  # noqa: E402

from recsys.data import load_movielens_1m, random_split  # noqa: E402
from recsys.metrics import mae, rmse  # noqa: E402
from recsys.models import BiasBaseline, BiasedMF, GlobalMean  # noqa: E402
from recsys.utils import hardware_info, mean_std, save_json  # noqa: E402


def grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*axes.values())]


GRIDS = {
    "BiasBaseline": grid(reg_user=[5.0, 10.0, 15.0, 25.0], reg_item=[5.0, 10.0, 25.0]),
    "FunkSVD": grid(n_factors=[50, 100, 200], lr=[0.005, 0.01], reg=[0.03, 0.05, 0.08]),
    "BiasedMF": grid(n_factors=[50, 100, 200], lr=[0.005, 0.01], reg=[0.03, 0.05, 0.08]),
}
MAX_EPOCHS = 80


def build(name: str, params: dict, seed: int = 0, n_epochs: int = MAX_EPOCHS):
    if name == "BiasBaseline":
        return BiasBaseline(**params)
    # biases share the factor regularisation; constant learning rate
    return BiasedMF(**params, reg_bias=params["reg"], n_epochs=n_epochs, lr_decay=1.0,
                    use_bias=(name == "BiasedMF"), seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/explicit.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--quick", action="store_true", help="tiny grid for a smoke test")
    args = ap.parse_args()

    ml = load_movielens_1m(args.data)
    train_full, test = random_split(ml.interactions, 0.1, seed=0)
    train, val = random_split(train_full, 0.1, seed=1)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    grids = {k: (v[:2] if args.quick else v) for k, v in GRIDS.items()}
    results: dict = {"protocol": "random 90/10 split (seed 0); validation = 10% of train",
                     "hardware": hardware_info(), "tuning": {}, "final": {}}

    # ---------------- tuning on validation ----------------
    best_cfg = {}
    for name, cands in grids.items():
        rows = []
        for params in cands:
            t0 = time.perf_counter()
            model = build(name, params)
            if name == "BiasBaseline":
                model.fit(train)
                score, epochs = rmse(val.ratings, model.predict(val.users, val.items)), None
            else:
                model.fit(train, val)
                curve = [h["val_rmse"] for h in model.history]
                epochs = int(np.argmin(curve)) + 1
                score = float(min(curve))
            row = {"params": params, "val_rmse": score, "best_epoch": epochs, "time": time.perf_counter() - t0}
            if name != "BiasBaseline":
                row["val_curve"] = curve
                row["train_curve"] = [h["train_rmse_running"] for h in model.history]
            rows.append(row)
            print(f"[tune] {name} {params} val_rmse={score:.4f} epochs={epochs}", flush=True)
        results["tuning"][name] = rows
        best = min(rows, key=lambda r: r["val_rmse"])
        best_cfg[name] = best
        print(f"[best] {name}: {best}")

    # ---------------- refit on train+val, evaluate on test ----------------
    mu_model = GlobalMean().fit(train_full)
    pred = mu_model.predict(test.users, test.items)
    results["final"]["GlobalMean"] = {"mean": {"rmse": rmse(test.ratings, pred), "mae": mae(test.ratings, pred)}}

    for name, best in best_cfg.items():
        seeds = [0] if name == "BiasBaseline" else args.seeds
        recs = []
        for seed in seeds:
            t0 = time.perf_counter()
            model = build(name, best["params"], seed=seed, n_epochs=best["best_epoch"] or 0)
            model.fit(train_full)
            p = model.predict(test.users, test.items)
            recs.append({"rmse": rmse(test.ratings, p), "mae": mae(test.ratings, p),
                         "fit_time": time.perf_counter() - t0})
            print(f"[final] {name} seed={seed} {recs[-1]}", flush=True)
        mean, std = mean_std(recs)
        results["final"][name] = {"params": best["params"], "epochs": best["best_epoch"],
                                  "seeds": recs, "mean": mean, "std": std}

    # ---------------- reference: scikit-surprise SVD ----------------
    try:
        from surprise import SVD, Dataset, Reader
        import pandas as pd

        reader = Reader(rating_scale=(1, 5))
        df = pd.DataFrame({"u": train_full.users, "i": train_full.items, "r": train_full.ratings})
        trainset = Dataset.load_from_df(df, reader).build_full_trainset()
        for label, kw in [("surprise SVD (defaults)", {}),
                          ("surprise SVD (our best config)", {
                              "n_factors": best_cfg["BiasedMF"]["params"]["n_factors"],
                              "lr_all": best_cfg["BiasedMF"]["params"]["lr"],
                              "reg_all": best_cfg["BiasedMF"]["params"]["reg"],
                              "n_epochs": best_cfg["BiasedMF"]["best_epoch"]})]:
            t0 = time.perf_counter()
            algo = SVD(random_state=0, **kw)
            algo.fit(trainset)
            p = np.array([algo.predict(int(u), int(i)).est for u, i in zip(test.users, test.items)])
            results["final"][label] = {"params": kw, "mean": {"rmse": rmse(test.ratings, p),
                                       "mae": mae(test.ratings, p), "fit_time": time.perf_counter() - t0}}
            print(f"[ref] {label}: {results['final'][label]['mean']}")
    except ImportError:
        print("scikit-surprise not installed; skipping reference")

    save_json(results, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
