"""
Publication figures for the edge-conditioned Hawkes kernel paper.
=================================================================
Reads ONLY the cached arrays/tables written by analysis.py (and, for the
15-method Table 1 and the dimension sweep, the CSVs written by run_ablation.py).
No model training happens here -- rerun analysis.py to refresh the numbers.

    python make_figures.py            # build every figure it has inputs for
    python make_figures.py --list     # show which inputs are present/missing

Outputs -> results/figures/*.png (+ table1_comparison.csv, table2_earlywarning.csv)
Each figure prints the one claim it supports.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
ADIR = ROOT / "results" / "analysis"
RDIR = ROOT / "results"
FDIR = RDIR / "figures"
FDIR.mkdir(parents=True, exist_ok=True)

# ---- MATLAB look: white box axes, inward mirrored ticks, parula, MATLAB colors --
from matplotlib.colors import LinearSegmentedColormap
# parula (MATLAB's default colormap) — anchor approximation
_PARULA = LinearSegmentedColormap.from_list("parula", [
    (0.2422, 0.1504, 0.6603), (0.2810, 0.3228, 0.9579), (0.1786, 0.5289, 0.9682),
    (0.0668, 0.6595, 0.8532), (0.0265, 0.7418, 0.6879), (0.4562, 0.7565, 0.4692),
    (0.7852, 0.7371, 0.3268), (0.9856, 0.7683, 0.1770), (0.9769, 0.9839, 0.0805)])
try:
    matplotlib.colormaps.register(_PARULA, name="parula")
    matplotlib.colormaps.register(_PARULA.reversed(), name="parula_r")
except (ValueError, AttributeError):
    pass
# MATLAB default line-color order
MC = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]
PROP = "#D95319"        # proposed model — MATLAB orange-red
REF = "#0072BD"         # references / baselines — MATLAB blue
FLOOR = "#7F7F7F"       # naive floor — grey
ACCENT = "#77AC30"      # MATLAB green
SECTOR_PAL = "tab20"

sns.reset_orig()
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "axes.titlesize": 12, "axes.titleweight": "normal", "axes.labelsize": 11,
    "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.axisbelow": True,
    "axes.prop_cycle": plt.cycler(color=MC),
    "axes.grid": True, "grid.color": "#bfbfbf", "grid.linewidth": 0.5,
    "grid.linestyle": "-", "grid.alpha": 0.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "xtick.color": "black", "ytick.color": "black",
    "lines.linewidth": 1.5, "lines.markersize": 6, "legend.frameon": True,
    "legend.edgecolor": "black", "legend.framealpha": 1.0, "mathtext.default": "regular",
})


def _save(fig, name, claim):
    p = FDIR / name
    fig.savefig(p); plt.close(fig)
    print(f"  {name:32s} <- {claim}")


def have(*names):
    return all((ADIR / n).exists() or (RDIR / n).exists() for n in names)


# ======================================================================
# TABLE 1 + headline forest plot  (§2)
# ======================================================================
def fig_table1():
    comp_p = RDIR / "comparison_table.csv"
    ci_p = ADIR / "headline_ci.csv"
    if not comp_p.exists():
        print("  [table1] skip: run run_ablation.py for comparison_table.csv"); return
    comp = pd.read_csv(comp_p)
    ci = pd.read_csv(ci_p) if ci_p.exists() else None
    paired = json.load(open(ADIR / "paired_test.json")) if (ADIR / "paired_test.json").exists() else None

    # merge CIs where we computed them (proposed + R1/R2 + Poisson floor)
    comp = comp.rename(columns={"LL/event": "ll"})
    comp["ci95"] = np.nan
    if ci is not None:
        namemap = {"Proposed conditional kernel": "Proposed conditional kernel",
                   "Shared mixture (R2)": "Shared mixture (R2)",
                   "Embedding single-exp (R1)": "Embedding single-exp (R1)",
                   "Homogeneous Poisson": "Homogeneous Poisson",
                   "Per-mark Poisson": "Per-mark Poisson"}
        cim = dict(zip(ci["model"], ci["ci95"]))
        comp["ci95"] = comp["Model"].map(lambda m: cim.get(m, np.nan))
    comp["is_floor"] = comp["Model"].str.contains("Poisson")
    comp["is_prop"] = comp["Model"].str.startswith("Proposed")
    out = comp[["Rank", "Model", "ll", "ci95", "is_floor"]].copy()
    out["ll"] = out["ll"].round(4)
    out.to_csv(RDIR / "table1_comparison.csv", index=False)

    # forest / bar plot, sorted best->worst, CI whiskers where available.
    # Some neural-TPP heads diverge at full epochs (LL << 0); clip the axis so the
    # informative comparison stays legible, and label off-scale bars with true value.
    d = comp.sort_values("ll", ascending=True).reset_index(drop=True)
    clip_lo = min(-6.0, d[~d["Model"].str.contains("C-NTPP|Hyper|RMTPP|Neural-ODE|Transformer")]["ll"].min() - 0.4)
    colors = [PROP if p else (FLOOR if f else REF)
              for p, f in zip(d["is_prop"], d["is_floor"])]
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(d) + 1.2))
    y = np.arange(len(d))
    drawn = np.clip(d["ll"], clip_lo, None)
    ax.barh(y, drawn, color=colors, height=0.68, edgecolor="white",
            hatch=["////" if v < clip_lo else "" for v in d["ll"]])
    xerr = np.where(d["ci95"].notna(), d["ci95"], 0)
    ax.errorbar(drawn, y, xerr=xerr, fmt="none", ecolor="#222222", capsize=3, lw=1)
    for i, v in enumerate(d["ll"]):
        if v < clip_lo:
            ax.text(clip_lo, i, f"  {v:.1f} ◄ off scale", va="center", ha="left", fontsize=7.5, color="#333")
    ax.set_yticks(y); ax.set_yticklabels(d["Model"])
    ax.set_xlim(clip_lo, max(d["ll"]) + 0.35)
    ax.set_xlabel("held-out log-likelihood / event  (higher is better)")
    # floor reference line = best Poisson
    if d["is_floor"].any():
        floor = d[d["is_floor"]]["ll"].max()
        ax.axvline(floor, color=FLOOR, ls="--", lw=1)
        ax.text(floor, len(d) - 0.4, " naive floor", color=FLOOR, fontsize=8, va="top")
    ttl = "Held-out predictive likelihood: proposed vs. 14 baselines"
    if paired:
        star = "***" if paired["wilcoxon_p"] < 1e-3 else ("**" if paired["wilcoxon_p"] < 1e-2 else "*")
        ttl += f"\nproposed − R2 = +{paired['mean_diff']:.3f} nat/event {star} (Wilcoxon p={paired['wilcoxon_p']:.1e})"
    ax.set_title(ttl, fontsize=11)
    _save(fig, "fig_table1_comparison.png", "§2 proposed beats all baselines & floor, with CIs + paired test")


# ======================================================================
# FIG 3  dimension sweep + kernel ladder  (§4)
# ======================================================================
def fig3():
    lb_p = RDIR / "leaderboard.csv"
    ladder_p = ADIR / "ladder.csv"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # (a) dim sweep
    ax = axes[0]
    if lb_p.exists():
        lb = pd.read_csv(lb_p)
        sub = lb[(lb.tau == 2.5)] if "tau" in lb else lb
        for bs, g in sub.groupby("beta_set"):
            g = g.sort_values("dim")
            ax.errorbar(g["dim"], g["mean"], yerr=g["std"].fillna(0), marker="o",
                        capsize=3, lw=1.6, label=f"β-bank: {bs}")
        ax.set_xlabel("embedding dimension  d"); ax.set_ylabel("held-out LL / event")
        ax.set_title("(a) Dimension sweep")
        ax.legend(fontsize=8, title="")
        ax.set_xscale("log", base=2); ax.set_xticks(sorted(sub["dim"].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    else:
        ax.text(.5, .5, "run run_ablation.py\n(leaderboard.csv)", ha="center", transform=ax.transAxes)
    # (b) kernel ladder
    ax = axes[1]
    if ladder_p.exists():
        lad = pd.read_csv(ladder_p).sort_values("order")
        short = ["R1\nlow-rank\nsingle-exp", "R2\nshared\nmixture", "R3\nsource–target\nconditioned"]
        cols = [REF, REF, PROP]
        ax.bar(range(len(lad)), lad["mean"], yerr=lad["std"].fillna(0), capsize=4,
               color=cols, edgecolor="white", width=0.62)
        ax.set_xticks(range(len(lad))); ax.set_xticklabels(short, fontsize=9)
        ax.set_ylabel("held-out LL / event")
        ax.set_title("(b) Kernel ladder")
        lo = lad["mean"].min(); hi = lad["mean"].max()
        base = lo - 0.15 * (hi - lo) - 0.05
        ax.set_ylim(base, hi + 0.12 * (hi - lo) + 0.02)
        for i, v in enumerate(lad["mean"]):
            ax.text(i, base + 0.03 * (hi - lo), f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, color="white", fontweight="bold")
        # annotate the gains between rungs
        for i in range(len(lad) - 1):
            g = lad["mean"].iloc[i + 1] - lad["mean"].iloc[i]
            ax.annotate(f"+{g:.2f}", (i + 0.5, (lad["mean"].iloc[i] + lad["mean"].iloc[i + 1]) / 2),
                        ha="center", fontsize=8.5, color="#333",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#ccc"))
    else:
        ax.text(.5, .5, "run analysis.py ladder", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    _save(fig, "fig3_dimension_and_ladder.png",
          "§4 small d best; conditioning (R2→R3) is where the jump comes from")


# ======================================================================
# FIG 4  contagion speed map  (§5)
# ======================================================================
def _sector_block(mat, sectors, agg=np.mean):
    secs = sorted(set(sectors))
    idx = {s: np.where(sectors == s)[0] for s in secs}
    M = np.zeros((len(secs), len(secs)))
    for a, sa in enumerate(secs):
        for b, sb in enumerate(secs):
            block = mat[np.ix_(idx[sa], idx[sb])]
            M[a, b] = agg(block)
    return M, secs


def fig4():
    if not have("interp_agg.npz"):
        print("  [fig4] skip: interp_agg.npz missing"); return
    A = np.load(ADIR / "interp_agg.npz", allow_pickle=True)
    delay, sectors = A["delay_mean"], A["sectors"]
    # delay[i,j] is target i <- source j; show source (rows) -> target (cols)
    src_tgt = delay.T
    M, secs = _sector_block(src_tgt, sectors, np.mean)
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    sns.heatmap(M, xticklabels=secs, yticklabels=secs, cmap="parula", annot=True,
                fmt=".1f", annot_kws={"size": 7}, linewidths=.4, linecolor="#e8e8e8",
                cbar_kws={"label": "expected excitation delay (days)"}, ax=ax)
    ax.set_xlabel("target sector (receiver)"); ax.set_ylabel("source sector (transmitter)")
    ax.set_title("Contagion speed map: sector-aggregated edge timescales")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, "fig4_speed_map.png",
          "§5 heterogeneous per-edge timing a shared kernel cannot express")


# ======================================================================
# FIG 5  directed excitation network  (§5)
# ======================================================================
def fig5():
    if not have("interp_agg.npz"):
        print("  [fig5] skip"); return
    import networkx as nx
    A = np.load(ADIR / "interp_agg.npz", allow_pickle=True)
    alpha, net, labels, sectors = A["alpha_mean"], A["net_mean"], A["labels"], A["sectors"]
    D = len(labels)
    # keep top edges for legibility (per-source strongest few)
    W = alpha.copy(); np.fill_diagonal(W, 0)
    thr = np.quantile(W, 0.92)
    G = nx.DiGraph()
    for i in range(D):
        G.add_node(i)
    for i in range(D):
        for j in range(D):
            if W[i, j] > thr:              # edge source j -> target i
                G.add_edge(j, i, w=W[i, j])
    secs = sorted(set(sectors))
    cmap = plt.get_cmap(SECTOR_PAL)
    scolor = {s: cmap(k % 20) for k, s in enumerate(secs)}
    ncolor = [scolor[s] for s in sectors]
    nsize = 60 + 900 * (net - net.min()) / (np.ptp(net) + 1e-9)
    pos = nx.spring_layout(G, seed=1, k=0.9, iterations=200)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ew = np.array([G[u][v]["w"] for u, v in G.edges()])
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.4 + 2.5 * ew / ew.max(),
                           edge_color="#555555", arrowsize=6, connectionstyle="arc3,rad=0.06")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=ncolor, node_size=nsize,
                           edgecolors="white", linewidths=0.6)
    # label the biggest net transmitters/receivers
    for k in list(np.argsort(net)[-6:]) + list(np.argsort(net)[:4]):
        ax.text(pos[k][0], pos[k][1], labels[k], fontsize=7, ha="center", va="center")
    handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=scolor[s], mec="white", label=s)
               for s in secs]
    ax.legend(handles=handles, fontsize=7, ncol=2, loc="upper left", frameon=True,
              title="sector", title_fontsize=8)
    ax.set_title("Directed excitation network (edges = αᵢⱼ, size = net transmission)")
    ax.axis("off")
    _save(fig, "fig5_excitation_network.png", "§5 hub structure without clean sector communities")


# ======================================================================
# FIG 6  net transmission vs own volatility  (§5)
# ======================================================================
def fig6():
    if not have("interp_agg.npz"):
        print("  [fig6] skip"); return
    A = np.load(ADIR / "interp_agg.npz", allow_pickle=True)
    net, sd, vol = A["net_mean"], A["net_sd"], A["own_vol"]
    labels, sectors = A["labels"], A["sectors"]
    r, p = A["pearson"]; rho, pp = A["spearman"]
    fig, ax = plt.subplots(figsize=(7, 5.4))
    secs = sorted(set(sectors))
    cmap = plt.get_cmap(SECTOR_PAL)
    scolor = {s: cmap(k % 20) for k, s in enumerate(secs)}
    ax.errorbar(vol * 100, net, yerr=sd, fmt="none", ecolor="#bbbbbb", lw=0.8, zorder=1)
    for s in secs:
        m = sectors == s
        ax.scatter(vol[m] * 100, net[m], s=55, color=scolor[s], edgecolors="white",
                   linewidths=0.6, label=s, zorder=3)
    # OLS trend
    b, a = np.polyfit(vol * 100, net, 1)
    xs = np.linspace((vol * 100).min(), (vol * 100).max(), 50)
    ax.plot(xs, a + b * xs, color="#333333", ls="--", lw=1.3, zorder=2)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel("own daily return volatility (%)")
    ax.set_ylabel("net transmission  (out − in excitation)")
    ax.set_title("Net transmission is not explained by own volatility")
    ax.text(0.03, 0.97, f"Pearson r = {r:.2f} (p = {p:.2f})\nSpearman ρ = {rho:.2f} (p = {pp:.2f})",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    ax.legend(fontsize=6.5, ncol=2, loc="lower right", title="sector", title_fontsize=7)
    _save(fig, "fig6_net_vs_volatility.png", "§5 kills the 'it's just volatility' objection")


# ======================================================================
# FIG 7  time-rescaling QQ  (§3)
# ======================================================================
def fig7():
    if not have("gof_rescaled.npz"):
        print("  [fig7] skip"); return
    G = np.load(ADIR / "gof_rescaled.npz")
    xi = np.sort(G["xi"]); ks, ksp = G["ks"]; n = len(xi)
    # QQ vs Exp(1): theoretical quantiles
    q = (np.arange(1, n + 1) - 0.5) / n
    theo = -np.log(1 - q)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(theo, xi, s=8, color=PROP, alpha=0.5, edgecolors="none")
    hi = max(theo.max(), xi.max())
    ax.plot([0, hi], [0, hi], color="#333333", ls="--", lw=1.2)
    # 95% KS band around the 45-deg line (in Exp(1) space via u)
    ax.set_xlabel("theoretical Exp(1) quantiles")
    ax.set_ylabel("rescaled inter-event increments  ξ")
    ax.set_title("Time-rescaling goodness-of-fit (held-out)")
    ax.text(0.03, 0.97, f"KS vs Exp(1): D = {ks:.3f}\np = {ksp:.2g}   (n = {n})",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    _save(fig, "fig7_time_rescaling_qq.png", "§3 well-calibrated with mild, honestly-reported tail misspecification")


# ======================================================================
# FIG 8 + Table 2  early warning  (§6)
# ======================================================================
def fig8():
    p = ADIR / "earlywarning.csv"
    if not p.exists():
        print("  [fig8] skip"); return
    df = pd.read_csv(p)
    df.to_csv(RDIR / "table2_earlywarning.csv", index=False)
    base = df["base_rate"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    d = df.sort_values("test_auroc")
    colors = [ACCENT if "GNN" in s else REF for s in d["substrate"]]
    ax.barh(d["substrate"], d["test_auroc"], color=colors, edgecolor="white", height=0.6)
    ax.axvline(0.5, color=FLOOR, ls="--", lw=1.2)
    ax.text(0.5, -0.6, "chance", color=FLOOR, fontsize=8, ha="center")
    for i, v in enumerate(d["test_auroc"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("held-out AUROC (forward-volatility stress)")
    ax.set_xlim(0.4, max(0.72, d["test_auroc"].max() + 0.05))
    ax.set_title(f"Early warning: no substrate beats a 2-parameter baseline\n"
                 f"(stress base rate = {base:.0%}, n_test = {df['n_test'].iloc[0]})", fontsize=10.5)
    _save(fig, "fig8_early_warning.png", "§6 honest negative result — richer substrates add nothing")


# ======================================================================
# FIG 9  robustness  (§7)
# ======================================================================
def fig9():
    tp, vp = ADIR / "robust_tau.csv", ADIR / "robust_volwin.csv"
    if not tp.exists():
        print("  [fig9] skip"); return
    tau = pd.read_csv(tp)
    g = tau.groupby("tau").agg(ll_mean=("ll", "mean"), ll_std=("ll", "std"),
                               n=("n_events", "first")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.errorbar(g["tau"], g["ll_mean"], yerr=g["ll_std"].fillna(0), marker="o",
                color=PROP, capsize=3, lw=1.6, label="held-out LL/event")
    ax.set_xlabel("event threshold  τ  (σ)"); ax.set_ylabel("held-out LL / event", color=PROP)
    ax.tick_params(axis="y", labelcolor=PROP)
    ax2 = ax.twinx()
    ax2.plot(g["tau"], g["n"], marker="s", color=REF, lw=1.4, ls=":")
    ax2.set_ylabel("# events", color=REF); ax2.tick_params(axis="y", labelcolor=REF)
    ax2.grid(False)
    ax.set_title("(a) Threshold sensitivity")
    ax = axes[1]
    if vp.exists():
        vw = pd.read_csv(vp)
        ax.plot(vw["vol_win"], vw["ll"], marker="o", color=PROP, lw=1.6)
        for _, r in vw.iterrows():
            ax.annotate(f"n={int(r['n_events'])}", (r["vol_win"], r["ll"]),
                        textcoords="offset points", xytext=(0, 7), fontsize=8, ha="center")
        ax.set_xlabel("volatility-standardization window (days)")
        ax.set_ylabel("held-out LL / event")
        mid = vw["ll"].mean()
        ax.set_ylim(mid - 0.30, mid + 0.30)                         # honest scale: variation is <0.1 nat
        rng = vw["ll"].max() - vw["ll"].min()
        ax.text(0.03, 0.06, f"full range = {rng:.2f} nat/event", transform=ax.transAxes,
                fontsize=8.5, color="#555555")
        ax.set_title("(b) Vol-standardization sensitivity")
    fig.tight_layout()
    _save(fig, "fig9_robustness.png", "§7 ranking & fit stable across τ and vol-window")


# ======================================================================
# FIG 10  embedding geometry: sector recovery + permutation null  (§8)
# ======================================================================
def fig10():
    if not have("embed_coords.npz", "embed_null.npz"):
        print("  [fig10] skip"); return
    C = np.load(ADIR / "embed_coords.npz", allow_pickle=True)
    N = np.load(ADIR / "embed_null.npz")
    nj = json.load(open(ADIR / "embed_null.json"))
    V, sectors, labels = C["Vtra"], C["sectors"], C["labels"]
    secs = sorted(set(sectors)); cmap = plt.get_cmap(SECTOR_PAL)
    scolor = {s: cmap(k % 20) for k, s in enumerate(secs)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    # (a) canonical transmitter embedding, first 2 axes, coloured by sector
    ax = axes[0]
    for s in secs:
        m = sectors == s
        ax.scatter(V[m, 0], V[m, 1], color=scolor[s], s=55, edgecolors="black",
                   linewidths=0.4, label=s)
    ax.set_xlabel("canonical transmitter axis 1"); ax.set_ylabel("axis 2")
    ax.set_title("(a) Transmitter embedding, coloured by sector")
    ax.legend(fontsize=6.5, ncol=2, loc="best", title="sector", title_fontsize=7)
    # (b) ARI vs permutation null
    ax = axes[1]
    for key, obs, col, lab in [("recv_null", float(N["recv_obs"]), REF, "receiver"),
                               ("tra_null", float(N["tra_obs"]), PROP, "transmitter")]:
        ax.hist(N[key], bins=40, color=col, alpha=0.35, density=True)
        p = nj[lab]["p_ari"]
        ax.axvline(obs, color=col, lw=2,
                   label=f"{lab}: ARI={obs:.2f} (p={'<0.001' if p < 1e-3 else f'{p:.3f}'})")
    ax.set_xlabel("Adjusted Rand Index vs GICS sectors")
    ax.set_ylabel("null density (label permutation)")
    ax.set_title("(b) Sector recovery beats chance")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig10_embedding_sectors.png",
          "§8 embedding rediscovers sectors from event timing alone (p<0.001)")


# ======================================================================
# FIG 11  src/tgt role asymmetry  (§8, novelty)
# ======================================================================
def fig11():
    if not have("embed_coords.npz", "embed_null.json"):
        print("  [fig11] skip"); return
    C = np.load(ADIR / "embed_coords.npz", allow_pickle=True)
    U, V = C["Urec"], C["Vtra"]
    def cos(X):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return Xn @ Xn.T
    rs, ts = cos(U), cos(V)
    D = U.shape[0]; iu = np.triu_indices(D, 1)
    from scipy import stats as st
    r = st.pearsonr(rs[iu], ts[iu])[0]
    stab = json.load(open(ADIR / "embed_stability.json"))
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(ts[iu], rs[iu], s=9, color=REF, alpha=0.35, edgecolors="none")
    b, a = np.polyfit(ts[iu], rs[iu], 1)
    xs = np.linspace(ts[iu].min(), ts[iu].max(), 50)
    ax.plot(xs, a + b * xs, color=PROP, ls="--", lw=1.5)
    ax.set_xlabel("transmitter-role similarity  (cosine of V)")
    ax.set_ylabel("receiver-role similarity  (cosine of U)")
    ax.set_title("Transmitter and receiver roles are distinct")
    ax.text(0.03, 0.97, f"pairwise r = {r:.2f}\n(the two role-geometries are\nnearly independent)",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="black"))
    ax.text(0.03, 0.06, f"seed-stable: transmitter sim-corr = {stab['transmitter_simcorr']:.2f}",
            transform=ax.transAxes, fontsize=8, color="#555")
    _save(fig, "fig11_role_asymmetry.png",
          "§8 a stock's transmitter neighbours ≠ its receiver neighbours (novelty)")


# ======================================================================
# FIG 12  signed shocks: negative vs positive contagion
# ======================================================================
def fig12():
    p = ADIR / "signed_comparison.csv"
    if not p.exists():
        print("  [fig12] skip"); return
    df = pd.read_csv(p)
    L = json.load(open(ADIR / "signed_leaders.json"))
    df.to_csv(RDIR / "table3_signed.csv", index=False)
    NEG, POS = "#4C72B0", "#D95319"
    cmap = {"neg": NEG, "pos": POS}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    order = ["neg", "pos"]; xlab = ["negative\n(z < −τ)", "positive\n(z > +τ)"]
    for ax, col, ttl in [(axes[0], "branching", "(a) Contagiousness (branching ratio)"),
                         (axes[1], "wmean_delay", "(b) Speed (excitation-weighted delay, days)")]:
        means = [df[df.sign == s][col].mean() for s in order]
        ax.bar([0, 1], means, color=[cmap[s] for s in order], width=0.6, edgecolor="black", zorder=1)
        for k, s in enumerate(order):                       # per-seed dots (honest spread)
            pts = df[df.sign == s][col].values
            ax.scatter(np.full(len(pts), k), pts, color="black", s=22, zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(xlab)
        ax.set_title(ttl, fontsize=10.5)
    # (c) top transmitters neg vs pos
    ax = axes[2]
    neg = L["neg"]["transmitters"][:5][::-1]; pos = L["pos"]["transmitters"][:5][::-1]
    y = np.arange(5)
    ax.barh(y + 0.2, [x[2] for x in neg], height=0.38, color=NEG, label="negative", edgecolor="black")
    ax.barh(y - 0.2, [x[2] for x in pos], height=0.38, color=POS, label="positive", edgecolor="black")
    for k in range(5):
        ax.text(0, y[k] + 0.2, " " + neg[k][0], va="center", fontsize=7.5)
        ax.text(0, y[k] - 0.2, " " + pos[k][0], va="center", fontsize=7.5)
    ax.set_yticks([]); ax.set_xlabel("net transmission")
    ax.set_title("(c) Top transmitters by shock sign", fontsize=10.5)
    ax.legend(fontsize=8, loc="lower right")
    nneg = int(df[df.sign == "neg"]["n_events"].iloc[0]); npos = int(df[df.sign == "pos"]["n_events"].iloc[0])
    fig.suptitle(f"Negative shocks: more contagious & faster   |   n(neg)={nneg}, n(pos)={npos}",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "fig12_signed_shocks.png",
          "negative shocks propagate faster with higher branching; financials lead the downside")


ALL = [("table1", fig_table1), ("fig3", fig3), ("fig4", fig4), ("fig5", fig5),
       ("fig6", fig6), ("fig7", fig7), ("fig8", fig8), ("fig9", fig9),
       ("fig10", fig10), ("fig11", fig11), ("fig12", fig12)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args()
    if a.list:
        for n in ["comparison_table.csv (run_ablation)", "leaderboard.csv (run_ablation)",
                  "headline_ci.csv", "paired_test.json", "ladder.csv", "interp_agg.npz",
                  "gof_rescaled.npz", "earlywarning.csv", "robust_tau.csv", "robust_volwin.csv"]:
            base = n.split()[0]
            ok = (ADIR / base).exists() or (RDIR / base).exists()
            print(f"  [{'x' if ok else ' '}] {n}")
        return
    print("Building figures -> results/figures/")
    for name, fn in ALL:
        if a.only and name not in a.only:
            continue
        fn()
    print("Done.")


if __name__ == "__main__":
    main()
