"""Build every figure in ``assets/`` and the example-recommendation table from ``results/``.

Run after ``run_explicit.py`` and all ``run_implicit.py`` stages::

    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.utils import limit_threads  # noqa: E402

limit_threads(4)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

from recsys.data import load_movielens_1m, temporal_user_split  # noqa: E402
from recsys.plotting import (BLUE, CATEGORICAL, FAMILY_COLORS, GRID, MUTED, ORANGE, TEXT, TEXT2,  # noqa: E402
                             apply_style, family)
from recsys.utils import load_json  # noqa: E402

GENRE_GROUPS = {
    "Children's / Animation": {"Children's", "Animation"},
    "Horror": {"Horror"},
    "Sci-Fi": {"Sci-Fi"},
    "Romance": {"Romance"},
    "War": {"War"},
    "Crime / Film-Noir": {"Crime", "Film-Noir"},
}
ANNOTATE = [
    "Toy Story (1995)", "Star Wars: Episode IV - A New Hope (1977)", "Shining, The (1980)",
    "Titanic (1997)", "Saving Private Ryan (1998)", "Pulp Fiction (1994)", "Matrix, The (1999)",
    "Lion King, The (1994)", "Scream (1996)", "Godfather, The (1972)", "Sleepless in Seattle (1993)",
    "Apocalypse Now (1979)", "Alien (1979)", "Notting Hill (1999)",
]
ORDER = ["Popularity", "ItemKNN", "MF-explicit", "ALS (CG)", "ALS (Cholesky)", "BPR", "GMF", "MLP", "NeuMF",
         "implicit ALS (ref)", "implicit BPR (ref)"]


def genre_group(genres: list[str]) -> str | None:
    hit = [g for g, s in GENRE_GROUPS.items() if s & set(genres)]
    return hit[0] if len(hit) == 1 else None


def short_title(t: str, n: int = 28) -> str:
    # "Shining, The (1980)" -> "The Shining"
    base = t.rsplit(" (", 1)[0]
    for art in (", The", ", A", ", An"):
        if base.endswith(art):
            base = art[2:] + " " + base[: -len(art)]
    return base if len(base) <= n else base[: n - 1] + "…"


# ---------------------------------------------------------------------------

def genre_purity(F: np.ndarray, groups: np.ndarray, k: int = 10) -> float:
    """Mean fraction of an item's k cosine-nearest neighbours that share its genre group."""
    lab = groups >= 0
    Fn = F[lab] / (np.linalg.norm(F[lab], axis=1, keepdims=True) + 1e-12)
    S = Fn @ Fn.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :k]
    g = groups[lab]
    return float((g[nn] == g[:, None]).mean())


def fig_embeddings(ml, factors, out: Path, model_key: str) -> dict:
    cnt = np.bincount(ml.interactions.items, minlength=ml.n_items)
    top = np.argsort(-cnt)[:1500]
    names = list(GENRE_GROUPS)
    groups = np.array([names.index(g) if (g := genre_group(ml.genres[j])) else -1 for j in top])

    purity = {}
    rng = np.random.default_rng(0)
    for key, F in factors.items():
        purity[key] = genre_purity(F[top], groups)
    perm_g = groups.copy()
    lab = perm_g >= 0
    perm_g[lab] = rng.permutation(perm_g[lab])
    purity["random (shuffled labels)"] = genre_purity(factors[model_key][top], perm_g)

    F = factors[model_key][top]
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    xy = TSNE(n_components=2, metric="cosine", perplexity=30, init="pca", random_state=0,
              max_iter=1500).fit_transform(F)

    fig, ax = plt.subplots(figsize=(11, 8.2))
    other = groups < 0
    ax.scatter(xy[other, 0], xy[other, 1], s=9, c="#d6d4cf", lw=0, label="other / multiple", zorder=1)
    for gi, gname in enumerate(names):
        m = groups == gi
        ax.scatter(xy[m, 0], xy[m, 1], s=22, c=CATEGORICAL[gi], lw=0.6, edgecolors="white",
                   label=f"{gname} ({m.sum()})", zorder=2)
    titles = list(ml.titles[top])
    for t in ANNOTATE:
        if t in titles:
            j = titles.index(t)
            ax.annotate(short_title(t), xy[j], xytext=(6, 5), textcoords="offset points", fontsize=8.5,
                        color=TEXT, zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
            ax.scatter(*xy[j], s=34, facecolors="none", edgecolors=TEXT, lw=1.0, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"t-SNE of {model_key} item embeddings: 1,500 most-rated MovieLens-1M films",
                 loc="left", fontsize=13, pad=26)
    ax.text(0.0, 1.0, "Genres were never shown to the model; colour = genre group (films matching exactly one group)",
            transform=ax.transAxes, fontsize=9.5, color=TEXT2, va="bottom")
    leg = ax.legend(loc="lower left", markerscale=1.4, ncol=2, fontsize=9, bbox_to_anchor=(0.0, -0.08))
    for txt in leg.get_texts():
        txt.set_color(TEXT2)
    fig.savefig(out)
    plt.close(fig)
    return purity


def fig_model_comparison(final: dict, explicit: dict, out: Path) -> None:
    names = [n for n in ORDER if n in final]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, metric, title in [(axes[0], "ndcg@10", "NDCG@10"), (axes[1], "recall@20", "Recall@20")]:
        vals = [final[n]["mean"][metric] for n in names]
        errs = [final[n]["std"].get(metric, 0.0) for n in names]
        y = np.arange(len(names))[::-1]
        cols = [FAMILY_COLORS[family(n)] for n in names]
        ax.barh(y, vals, xerr=errs, color=cols, height=0.68, error_kw=dict(ecolor=TEXT2, lw=1, capsize=2))
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.015, yi, f"{v:.4f}", va="center", fontsize=8.5, color=TEXT2)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlim(0, max(vals) * 1.18)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Top-N: {title} (full ranking, temporal split)", loc="left", fontsize=11)
    # explicit
    ex = explicit["final"]
    order = ["GlobalMean", "BiasBaseline", "FunkSVD", "BiasedMF", "surprise SVD (defaults)",
             "surprise SVD (our best config)"]
    order = [n for n in order if n in ex]
    vals = [ex[n]["mean"]["rmse"] for n in order]
    errs = [ex[n].get("std", {}).get("rmse", 0.0) for n in order]
    y = np.arange(len(order))[::-1]
    ax = axes[2]
    ax.barh(y, vals, xerr=errs, color=[FAMILY_COLORS[family(n)] for n in order], height=0.68,
            error_kw=dict(ecolor=TEXT2, lw=1, capsize=2))
    for yi, v in zip(y, vals):
        ax.text(v + 0.003, yi, f"{v:.4f}", va="center", fontsize=8.5, color=TEXT2)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("surprise SVD ", "surprise SVD\n") for n in order], fontsize=8.5)
    ax.set_xlim(0.8, max(vals) + 0.06)
    ax.grid(axis="y", visible=False)
    ax.set_title("Rating prediction: test RMSE (lower is better)", loc="left", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    fig.legend(handles, ["non-personalised", "neighbourhood", "matrix factorisation", "neural (NCF)",
                         "library reference"], loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def fig_popularity(final: dict, topk: dict, pop: np.ndarray, out: Path) -> None:
    names = [n for n in ORDER if n in final and n in topk and "(ref)" not in n and n != "ALS (Cholesky)"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [0.9, 1, 1.25]})
    # (a) long tail
    ax = axes[0]
    s = np.sort(pop)[::-1]
    share = np.cumsum(s) / s.sum()
    x = np.arange(1, len(s) + 1) / len(s) * 100
    ax.plot(x, share * 100, color=BLUE)
    h = int(round(0.2 * len(s)))
    ax.axvline(20, color=MUTED, lw=1, ls="--")
    ax.text(21.5, 10, f"top 20% of items\n= {share[h - 1] * 100:.0f}% of interactions", fontsize=9, color=TEXT2)
    ax.set_xlabel("% of catalogue (items sorted by popularity)")
    ax.set_ylabel("% of training interactions")
    ax.set_title("Long tail of MovieLens-1M", loc="left", fontsize=11)
    # (b) accuracy vs coverage
    ax = axes[1]
    for n in names:
        m = final[n]["mean"]
        ax.scatter(m["coverage@10"] * 100, m["ndcg@10"], s=70, color=FAMILY_COLORS[family(n)],
                   edgecolors="white", lw=1.2, zorder=3)
        ax.annotate(n, (m["coverage@10"] * 100, m["ndcg@10"]), xytext=(6, 4), textcoords="offset points",
                    fontsize=8.5, color=TEXT)
    ax.set_xlabel("catalogue coverage@10 (%)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Accuracy vs catalogue coverage", loc="left", fontsize=11)
    # (c) popularity percentile of recommended items
    ax = axes[2]
    pct = np.argsort(np.argsort(pop)) / (len(pop) - 1) * 100   # 100 = most popular item
    data = [pct[topk[n][:, :10].ravel()] for n in names]
    parts = ax.violinplot(data, positions=np.arange(len(names)), showmedians=True, widths=0.8)
    for body, n in zip(parts["bodies"], names):
        body.set_facecolor(FAMILY_COLORS[family(n)])
        body.set_alpha(0.75)
        body.set_edgecolor("none")
    for k in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[k].set_color(TEXT2)
        parts[k].set_linewidth(1)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("popularity percentile of recommended items")
    ax.axhline(80, color=MUTED, lw=1, ls="--")
    ax.text(len(names) - 0.5, 81, "short head", fontsize=8.5, color=TEXT2, ha="right", va="bottom")
    ax.set_title("Where do top-10 recommendations come from?", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_training_curves(explicit: dict, tuned: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    for name, col in [("FunkSVD", MUTED), ("BiasedMF", BLUE)]:
        rows = explicit["tuning"][name]
        best = min(rows, key=lambda r: r["val_rmse"])
        ep = np.arange(1, len(best["val_curve"]) + 1)
        ax.plot(ep, best["val_curve"], color=col, label=f"{name} validation")
        ax.plot(ep, best["train_curve"], color=col, ls=":", lw=1.5, label=f"{name} train (running)")
        ax.scatter([best["best_epoch"]], [best["val_rmse"]], color=col, s=40, zorder=3)
    ax.set_ylim(0.70, 1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("SGD matrix factorisation (best configs)", loc="left", fontsize=11)
    ax.legend()
    ax = axes[1]
    hist = tuned["BPR"]["best"]["history"]
    e = [h["epoch"] for h in hist if "val_ndcg@10" in h]
    ax.plot(e, [h["val_ndcg@10"] for h in hist if "val_ndcg@10" in h], color=BLUE, label="validation NDCG@10")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NDCG@10")
    ax.set_title("BPR-MF (LearnBPR, SGD)", loc="left", fontsize=11)
    ax = axes[2]
    for name, col in [("GMF", BLUE), ("MLP", ORANGE), ("NeuMF", CATEGORICAL[2])]:
        if name not in tuned:
            continue
        for row in tuned[name]["grid"]:
            hist = row["history"]
            lbl = f"{name} {row['params']}".replace("{", "").replace("}", "").replace("'", "")
            best = row is tuned[name]["best"] or row["params"] == tuned[name]["best"]["params"]
            ax.plot([h["epoch"] for h in hist], [h["val_ndcg@10"] for h in hist], color=col,
                    lw=2 if best else 1, alpha=1 if best else 0.45, label=lbl if best else None)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NDCG@10")
    ax.set_title("NCF (best config bold)", loc="left", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_als_grid(tuned: dict, out: Path) -> None:
    rows = tuned["ALS"]["grid"]
    fs = sorted({r["params"]["n_factors"] for r in rows})
    regs = sorted({r["params"]["reg"] for r in rows})
    alphas = sorted({r["params"]["alpha"] for r in rows})
    fig, axes = plt.subplots(1, len(fs), figsize=(5 * len(fs), 3.8), squeeze=False)
    vmax = max(r["val"]["ndcg@10"] for r in rows)
    vmin = min(r["val"]["ndcg@10"] for r in rows)
    for ax, f in zip(axes[0], fs):
        M = np.full((len(regs), len(alphas)), np.nan)
        for r in rows:
            if r["params"]["n_factors"] == f:
                M[regs.index(r["params"]["reg"]), alphas.index(r["params"]["alpha"])] = r["val"]["ndcg@10"]
        im = ax.imshow(M, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(len(regs)):
            for j in range(len(alphas)):
                ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center", fontsize=8.5,
                        color="white" if M[i, j] > (vmin + vmax) / 2 else TEXT)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:g}" for a in alphas])
        ax.set_yticks(range(len(regs)))
        ax.set_yticklabels([f"{r:g}" for r in regs])
        ax.set_xlabel(r"confidence weight $\alpha$")
        ax.set_ylabel(r"regularisation $\lambda$")
        ax.grid(False)
        ax.set_title(f"ALS, f = {f}: validation NDCG@10", loc="left", fontsize=11)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85)
    fig.savefig(out)
    plt.close(fig)


def fig_sampled(loo: dict, out: Path) -> None:
    names = [n for n in ORDER if n in loo["models"]]
    full = np.array([loo["models"][n]["hr@10"] for n in names])
    samp = np.array([loo["models"][n]["sampled_hr@10"] for n in names])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x - 0.2, full, 0.38, color=BLUE, label="full ranking (all ~3.7k items)")
    ax.bar(x + 0.2, samp, 0.38, color=ORANGE, label="sampled (1 positive vs 100 negatives)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("HR@10")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("Leave-one-out HR@10: sampled vs full ranking", loc="left", fontsize=11)
    ax = axes[1]
    rf = np.argsort(np.argsort(-full)) + 1
    rs = np.argsort(np.argsort(-samp)) + 1
    for n, a, b in zip(names, rf, rs):
        col = FAMILY_COLORS[family(n)]
        ax.plot([0, 1], [a, b], color=col, lw=2, marker="o", ms=7)
        ax.text(-0.04, a, n, ha="right", va="center", fontsize=8.5, color=TEXT)
        ax.text(1.04, b, n, ha="left", va="center", fontsize=8.5, color=TEXT)
    ax.set_xlim(-0.6, 1.6)
    ax.invert_yaxis()
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["rank by full HR@10", "rank by sampled HR@10"])
    ax.set_yticks([])
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title("Model ranking under each protocol", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def example_table(ml, topk: dict, out: Path, n_show: int = 5) -> str:
    _, test = temporal_user_split(ml.interactions, 0.2)
    train_full, _ = temporal_user_split(ml.interactions, 0.2)
    users = topk["users"]
    test_sets = {}
    for u, i in zip(test.users, test.items):
        test_sets.setdefault(int(u), set()).add(int(i))
    models = [n for n in ["Popularity", "ItemKNN", "ALS (CG)", "BPR", "NeuMF"] if n in topk]
    # pick users whose history is dominated by one genre, with a moderate history length
    picks = []
    for genre in ["Children's", "Horror", "Sci-Fi"]:
        best_u, best_frac = None, 0
        for row, u in enumerate(users):
            hist = train_full.items[train_full.users == u]
            if not 40 <= len(hist) <= 150:
                continue
            frac = np.mean([genre in ml.genres[i] for i in hist])
            if frac > best_frac:
                best_u, best_frac = (row, u), frac
        picks.append((genre, best_u, best_frac))
    lines = []
    for genre, (row, u), frac in picks:
        hist_mask = train_full.users == u
        hist = train_full.items[hist_mask][np.argsort(train_full.timestamps[hist_mask])][::-1][:n_show]
        lines.append(f"**User {ml.raw_user_ids[u]}** — {frac * 100:.0f}% of their {hist_mask.sum()} training "
                     f"ratings are *{genre}* films. Most recent: "
                     + "; ".join(short_title(ml.titles[i], 40) for i in hist) + ".\n")
        lines.append("| rank | " + " | ".join(models) + " |")
        lines.append("|---" * (len(models) + 1) + "|")
        for r in range(n_show):
            cells = []
            for m in models:
                i = int(topk[m][row, r])
                t = short_title(ml.titles[i], 30)
                cells.append(f"**{t}**" if i in test_sets.get(int(u), set()) else t)
            lines.append(f"| {r + 1} | " + " | ".join(cells) + " |")
        lines.append("")
    md = "\n".join(lines)
    out.write_text(md)
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--data", default="data")
    ap.add_argument("--embedding-model", default="ALS (CG)")
    args = ap.parse_args()
    apply_style()
    R, A = Path(args.results), Path(args.assets)
    A.mkdir(exist_ok=True)
    ml = load_movielens_1m(args.data)
    final = load_json(R / "implicit_final.json")["models"]
    explicit = load_json(R / "explicit.json")
    tuned = {p.stem: load_json(p) for p in (R / "tune").glob("*.json")}
    topk = dict(np.load(R / "topk_seed0.npz", allow_pickle=False))
    factors = dict(np.load(R / "item_factors_seed0.npz", allow_pickle=False))
    train_full, _ = temporal_user_split(ml.interactions, 0.2)
    pop = np.bincount(train_full.items, minlength=ml.n_items).astype(float)

    purity = fig_embeddings(ml, factors, A / "embeddings_tsne.png", args.embedding_model)
    print("genre purity@10:", {k: round(v, 3) for k, v in purity.items()})
    (R / "genre_purity.json").write_text(__import__("json").dumps(purity, indent=2))
    fig_model_comparison(final, explicit, A / "model_comparison.png")
    fig_popularity(final, topk, pop, A / "popularity_bias.png")
    fig_training_curves(explicit, tuned, A / "training_curves.png")
    fig_als_grid(tuned, A / "als_grid.png")
    if (R / "implicit_loo.json").exists():
        fig_sampled(load_json(R / "implicit_loo.json"), A / "sampled_vs_full.png")
    print(example_table(ml, topk, R / "examples.md"))


if __name__ == "__main__":
    main()
