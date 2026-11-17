"""Print the Markdown result tables used in the README from ``results/*.json``.

    python scripts/make_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path("results")


def load(name):
    return json.loads((R / name).read_text())


def explicit() -> str:
    f = load("explicit.json")["final"]
    rows = ["| Model | test RMSE | test MAE |", "|---|---|---|"]
    for k in ["GlobalMean", "BiasBaseline", "FunkSVD", "BiasedMF", "surprise SVD (defaults)",
              "surprise SVD (our best config)"]:
        m, s = f[k]["mean"], f[k].get("std", {})
        pm = lambda key: f"{m[key]:.4f}" + (f" ± {s[key]:.4f}" if s.get(key, 0) > 0 else "")  # noqa: E731
        rows.append(f"| {k} | {pm('rmse')} | {pm('mae')} |")
    return "\n".join(rows)


def implicit() -> str:
    f = load("implicit_final.json")["models"]
    rows = ["| Model | NDCG@10 | Recall@20 | HR@10 | coverage@10 | Gini@10 | avg. popularity of top-10 |",
            "|---|---|---|---|---|---|---|"]
    for k, v in f.items():
        m, s = v["mean"], v["std"]
        nd = f"{m['ndcg@10']:.4f}" + (f" ± {s['ndcg@10']:.4f}" if s["ndcg@10"] > 0 else "")
        rc = f"{m['recall@20']:.4f}" + (f" ± {s['recall@20']:.4f}" if s["recall@20"] > 0 else "")
        rows.append(f"| {k} | {nd} | {rc} | {m['hr@10']:.3f} | {m['coverage@10'] * 100:.1f}% | {m['gini@10']:.3f} | "
                    f"{m['arp@10']:.0f} |")
    return "\n".join(rows)


def loo() -> str:
    f = load("implicit_loo.json")["models"]
    rows = ["| Model | full HR@10 | full NDCG@10 | sampled HR@10 | sampled NDCG@10 |", "|---|---|---|---|---|"]
    for k, v in f.items():
        rows.append(f"| {k} | {v['hr@10']:.4f} | {v['ndcg@10']:.4f} | {v['sampled_hr@10']:.4f} | {v['sampled_ndcg@10']:.4f} |")
    return "\n".join(rows)


def als_timing() -> str:
    rows = load("als_timing.json")["rows"]
    fs = sorted({r["n_factors"] for r in rows})
    out = ["| factors | Cholesky (s) | CG, 3 steps (s) | speed-up | objective (Cholesky) | objective (CG) |", "|---|---|---|---|---|---|"]
    for f in fs:
        a = next(r for r in rows if r["n_factors"] == f and r["solver"] == "cholesky")
        b = next(r for r in rows if r["n_factors"] == f and r["solver"] == "cg")
        out.append(f"| {f} | {a['seconds']:.2f} | {b['seconds']:.2f} | {a['seconds'] / b['seconds']:.1f}x | "
                   f"{a['objective']:.0f} | {b['objective']:.0f} |")
    return "\n".join(out)


def tuned() -> str:
    rows = ["| Model | selected on validation NDCG@10 | grid size | val. NDCG@10 |", "|---|---|---|---|"]
    for p in sorted((R / "tune").glob("*.json")):
        d = load(f"tune/{p.name}")
        b = d["best"]
        ep = f", {b['best_epoch']} epochs" if b.get("best_epoch") else ""
        rows.append(f"| {d['model']} | `{json.dumps(b['params'])}`{ep} | {len(d['grid'])} | {b['val']['ndcg@10']:.4f} |")
    return "\n".join(rows)


if __name__ == "__main__":
    for title, fn in [("EXPLICIT", explicit), ("IMPLICIT", implicit), ("LOO", loo), ("ALS TIMING", als_timing),
                      ("TUNED", tuned)]:
        print(f"\n### {title}\n{fn()}")
    print("\n### GENRE PURITY\n", Path("results/genre_purity.json").read_text())
