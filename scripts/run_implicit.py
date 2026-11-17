"""Implicit-feedback top-N benchmark on MovieLens-1M with full-ranking metrics.

Every rating is treated as an implicit interaction. Protocol:

* ``temporal`` (main): per-user chronological split; the last 20% of each user's
  interactions are the test set. The last 12.5% of the remaining interactions
  (10% overall) are the validation set used for hyper-parameter search and for
  choosing the number of epochs. Final models are refit on train+validation
  and scored on test; stochastic models are averaged over three seeds.
* ``loo``: leave-last-one-out (He et al. 2017) with the hyper-parameters found
  on the temporal protocol; reports full-ranking and sampled (100 negatives)
  HR@10 / NDCG@10 side by side.

Stages (each well under 20 minutes)::

    python scripts/run_implicit.py --stage tune
    python scripts/run_implicit.py --stage final
    python scripts/run_implicit.py --stage loo
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
import torch  # noqa: E402

from recsys.data import leave_last_out_split, load_movielens_1m, temporal_user_split  # noqa: E402
from recsys.metrics import evaluate_ranking, sampled_metrics  # noqa: E402
from recsys.models import BiasedMF, BPRMF, ImplicitALS, ItemKNN, Popularity  # noqa: E402
from recsys.models.ncf import NCFRecommender  # noqa: E402
from recsys.utils import hardware_info, load_json, mean_std, save_json, seed_everything  # noqa: E402

KS = (10, 20)
SEL = "ndcg@10"          # model-selection metric on validation


def grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*axes.values())]


GRIDS = {
    "Popularity": [{}],
    "ItemKNN": grid(k=[20, 50, 100, 200, 500, 1000], shrink=[0.0, 10.0, 100.0]),
    "ALS": grid(n_factors=[64, 128], reg=[10.0, 25.0, 50.0, 100.0, 200.0], alpha=[0.3, 1.0, 3.0, 10.0, 30.0]),
    "BPR": grid(n_factors=[128], lr=[0.05, 0.1], reg=[0.01, 0.03, 0.1]),
    "MF-explicit": grid(n_factors=[100], lr=[0.01], reg=[0.05]),
    "GMF": grid(gmf_dim=[64], lr=[1e-3, 3e-4]),
    "MLP": grid(layers=[(128, 64, 32)], lr=[1e-3, 3e-4]),
    "NeuMF": grid(pretrained=[False, True]),
}
EPOCHS = {"BPR": 300, "MF-explicit": 40, "GMF": 60, "MLP": 30, "NeuMF": 30}


def make_model(name: str, p: dict, seed: int, n_epochs: int | None = None, pretrained=None):
    """Instantiate a model. ``n_epochs`` overrides the tuning budget (used for refits)."""
    ep = n_epochs or EPOCHS.get(name)
    if name == "Popularity":
        return Popularity()
    if name == "ItemKNN":
        return ItemKNN(**p)
    if name == "ALS":
        q = {k: v for k, v in p.items() if k != "solver"}
        return ImplicitALS(**q, n_iters=15, solver=p.get("solver", "cg"), seed=seed)
    if name == "BPR":
        return BPRMF(**p, n_epochs=ep, seed=seed)
    if name == "MF-explicit":
        return BiasedMF(**p, n_epochs=ep, seed=seed)
    if name == "GMF":
        return NCFRecommender(arch="gmf", **p, n_epochs=ep, seed=seed)
    if name == "MLP":
        return NCFRecommender(arch="mlp", **p, n_epochs=ep, seed=seed)
    if name == "NeuMF":
        if p.get("pretrained"):
            # He et al. 2017: initialise from GMF + MLP and fine-tune with plain SGD
            return NCFRecommender(arch="neumf", gmf_dim=64, layers=(128, 64, 32), optimizer="sgd",
                                  lr=0.05, n_epochs=ep, seed=seed, pretrained=pretrained)
        return NCFRecommender(arch="neumf", gmf_dim=64, layers=(128, 64, 32), lr=1e-3,
                              n_epochs=ep, seed=seed)
    raise ValueError(name)


def fit_model(model, name, train, val_csr=None, train_csr=None):
    """Fit; for epoch-based models optionally track validation NDCG for early stopping."""
    cb = None
    if val_csr is not None and name in ("BPR", "GMF", "MLP", "NeuMF"):
        def cb(m):
            r = evaluate_ranking(m.score, train_csr, val_csr, ks=(10,))
            return {"score": r[SEL], "val_ndcg@10": r[SEL]}
    if name == "MF-explicit":
        model.fit(train)
    elif name == "BPR":
        model.fit(train, callback=cb, eval_every=5)
    elif name in ("GMF", "MLP", "NeuMF"):
        model.fit(train, callback=cb)
    else:
        model.fit(train)
    return model


def temporal_splits(ml):
    train_full, test = temporal_user_split(ml.interactions, 0.2)
    train, val = temporal_user_split(train_full, 0.125)
    return train, val, train_full, test


# ---------------------------------------------------------------------------

def stage_tune(ml, args):
    train, val, _, _ = temporal_splits(ml)
    Rtr, Rva = train.to_csr(), val.to_csr()
    meta = {"protocol": "temporal 70/10/20 per user; selection on validation " + SEL,
            "hardware": hardware_info()}
    best_ncf = {}
    for name in args.models:
        rows = []
        for p in GRIDS[name]:
            seed_everything(0)
            t0 = time.perf_counter()
            pre = None
            if name == "NeuMF" and p.get("pretrained"):
                pre = (best_ncf["GMF"], best_ncf["MLP"])
            model = fit_model(make_model(name, p, seed=0, pretrained=pre), name, train, Rva, Rtr)
            fit_time = time.perf_counter() - t0
            r = evaluate_ranking(model.score, Rtr, Rva, ks=KS)
            row = {"params": p, "val": {k: v for k, v in r.items() if "@" in k}, "fit_time": fit_time,
                   "best_epoch": getattr(model, "best_epoch", None)}
            if name in ("BPR", "GMF", "MLP", "NeuMF", "ALS"):
                row["history"] = model.history
            rows.append(row)
            print(f"[tune] {name} {p} {SEL}={r[SEL]:.4f} recall@20={r['recall@20']:.4f} "
                  f"epoch={row['best_epoch']} t={fit_time:.1f}s", flush=True)
            if name in ("GMF", "MLP"):
                if name not in best_ncf or r[SEL] > best_ncf[name + "_score"]:
                    best_ncf[name], best_ncf[name + "_score"] = model, r[SEL]
        best = max(rows, key=lambda x: x["val"][SEL])
        print(f"[best] {name}: {best['params']} {SEL}={best['val'][SEL]:.4f}", flush=True)
        # one file per model so that tuning runs for different models can run in parallel
        save_json({**meta, "model": name, "best": best, "grid": rows},
                  Path(args.results) / "tune" / f"{name}.json")


def load_tuned(results_dir: str) -> dict:
    return {p.stem: load_json(p) for p in sorted((Path(results_dir) / "tune").glob("*.json"))}


def _refit_epochs(best: dict, name: str) -> int | None:
    if name in ("BPR", "GMF", "MLP", "NeuMF"):
        # train+val has 1/7 more data per epoch than train; keep the same epoch count
        return int(best["best_epoch"])
    return None


def _final_variants(tuned: dict) -> list[tuple[str, str, dict]]:
    """(display name, model family, params) for the final table."""
    v = []
    for name in ["Popularity", "ItemKNN", "MF-explicit", "ALS", "BPR", "GMF", "MLP", "NeuMF"]:
        if name not in tuned:
            continue
        p = dict(tuned[name]["best"]["params"])
        if name == "ALS":
            v.append(("ALS (CG)", "ALS", {**p, "solver": "cg"}))
            v.append(("ALS (Cholesky)", "ALS", {**p, "solver": "cholesky"}))
        else:
            v.append((name, name, p))
    return v


def _reference_models(tuned, train_full, Rtr, seed):
    """`implicit` library ALS/BPR with our tuned hyper-parameters (reference only)."""
    refs = {}
    try:
        from implicit.cpu.als import AlternatingLeastSquares
        from implicit.cpu.bpr import BayesianPersonalizedRanking
    except ImportError:
        return refs
    p = tuned["ALS"]["best"]["params"]
    t0 = time.perf_counter()
    # implicit uses c_ui = alpha * r_ui on observed cells; ours is 1 + alpha * r_ui
    als = AlternatingLeastSquares(factors=p["n_factors"], regularization=p["reg"], alpha=1.0 + p["alpha"],
                                  iterations=15, use_cg=True, random_state=seed, num_threads=4,
                                  calculate_training_loss=False)
    als.fit(Rtr, show_progress=False)
    refs["implicit ALS (ref)"] = (lambda u, a=als: a.user_factors[u] @ a.item_factors.T, time.perf_counter() - t0)
    pb = tuned["BPR"]["best"]
    t0 = time.perf_counter()
    bpr = BayesianPersonalizedRanking(factors=pb["params"]["n_factors"], learning_rate=pb["params"]["lr"],
                                      regularization=pb["params"]["reg"], iterations=int(pb["best_epoch"]),
                                      random_state=seed, num_threads=4)
    bpr.fit(Rtr, show_progress=False)
    refs["implicit BPR (ref)"] = (lambda u, b=bpr: b.user_factors[u] @ b.item_factors.T, time.perf_counter() - t0)
    return refs


def stage_final(ml, args):
    tuned = load_tuned(args.results)
    _, _, train_full, test = temporal_splits(ml)
    Rtr, Rte = train_full.to_csr(), test.to_csr()
    pop = np.asarray(Rtr.sum(0)).ravel()
    out = {"protocol": "temporal 80/20 per user, refit on train+val", "hardware": hardware_info(),
           "models": {}}
    deterministic = {"Popularity", "ItemKNN"}
    topk_store, factors_store = {}, {}
    for display, name, p in _final_variants(tuned):
        recs = []
        seeds = [0] if name in deterministic else args.seeds
        for seed in seeds:
            seed_everything(seed)
            pre = None
            if name == "NeuMF" and p.get("pretrained"):
                pre = tuple(fit_model(make_model(n, tuned[n]["best"]["params"], seed,
                                                 _refit_epochs(tuned[n]["best"], n)), n, train_full)
                            for n in ("GMF", "MLP"))
            t0 = time.perf_counter()
            model = fit_model(make_model(name, p, seed, _refit_epochs(tuned[name]["best"], name), pre),
                              name, train_full)
            fit_time = time.perf_counter() - t0
            r = evaluate_ranking(model.score, Rtr, Rte, ks=KS, item_popularity=pop, return_topk=(seed == 0))
            rec = {k: v for k, v in r.items() if "@" in k}
            rec["fit_time"] = fit_time
            recs.append(rec)
            print(f"[final] {display} seed={seed} ndcg@10={rec['ndcg@10']:.4f} recall@20={rec['recall@20']:.4f} "
                  f"cov@10={rec['coverage@10']:.3f} t={fit_time:.1f}s", flush=True)
            if seed == 0:
                topk_store[display] = r["topk"][:, :20]
                topk_store["users"] = r["users"]
                topk_store[display + "/ndcg@10"] = r["per_user"]["ndcg@10"]
                if hasattr(model, "item_factors"):
                    factors_store[display] = model.item_factors
        mean, std = mean_std(recs)
        out["models"][display] = {"params": p, "seeds": recs, "mean": mean, "std": std}
    for display, (score_fn, fit_time) in _reference_models(tuned, train_full, Rtr, 0).items():
        r = evaluate_ranking(score_fn, Rtr, Rte, ks=KS, item_popularity=pop)
        rec = {k: v for k, v in r.items() if "@" in k}
        rec["fit_time"] = fit_time
        out["models"][display] = {"seeds": [rec], "mean": rec, "std": {k: 0.0 for k in rec}}
        print(f"[ref] {display} ndcg@10={rec['ndcg@10']:.4f} recall@20={rec['recall@20']:.4f}", flush=True)
    save_json(out, Path(args.results) / "implicit_final.json")
    np.savez_compressed(Path(args.results) / "topk_seed0.npz", **topk_store)
    np.savez_compressed(Path(args.results) / "item_factors_seed0.npz", **factors_store)


def stage_loo(ml, args):
    tuned = load_tuned(args.results)
    train, test = leave_last_out_split(ml.interactions, 1)
    Rtr, Rte = train.to_csr(), test.to_csr()
    users = np.arange(ml.n_users)
    test_item = np.zeros(ml.n_users, dtype=np.int64)
    test_item[test.users] = test.items
    out = {"protocol": "leave-last-one-out; hyper-parameters from the temporal protocol",
           "hardware": hardware_info(), "models": {}}
    fitted = {}
    for display, name, p in _final_variants(tuned):
        if display == "ALS (Cholesky)":
            continue
        seed_everything(0)
        pre = (fitted["GMF"], fitted["MLP"]) if (name == "NeuMF" and p.get("pretrained")) else None
        model = fit_model(make_model(name, p, 0, _refit_epochs(tuned[name]["best"], name), pre), name, train)
        if name in ("GMF", "MLP"):
            fitted[name] = model
        full = evaluate_ranking(model.score, Rtr, Rte, ks=(10,))
        samp = sampled_metrics(model.score, Rtr, test_item, users, n_negatives=100, k=10, seed=0)
        out["models"][display] = {"hr@10": full["hr@10"], "ndcg@10": full["ndcg@10"], **samp}
        print(f"[loo] {display} full HR@10={full['hr@10']:.4f} NDCG@10={full['ndcg@10']:.4f} | "
              f"sampled HR@10={samp['sampled_hr@10']:.4f} NDCG@10={samp['sampled_ndcg@10']:.4f}", flush=True)
    save_json(out, Path(args.results) / "implicit_loo.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["tune", "final", "loo"], required=True)
    ap.add_argument("--models", nargs="+", default=list(GRIDS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    ml = load_movielens_1m(args.data)
    {"tune": stage_tune, "final": stage_final, "loo": stage_loo}[args.stage](ml, args)


if __name__ == "__main__":
    main()
