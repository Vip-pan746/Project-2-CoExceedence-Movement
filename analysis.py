"""
Results analysis for the edge-conditioned Hawkes kernel paper.
=============================================================
Everything the Results section needs BEYOND the leaderboard that run_ablation.py
already produces. Trains the canonical proposed model (R3) once per seed, then
derives — in pure numpy, no autograd — every quantity the figures need, and
caches each to results/analysis/ so make_figures.py never has to retrain.

Sections produced (map to the paper's Results arc):
  §2  headline CIs + paired significance   -> headline_ci.csv, paired_test.json
  §3  goodness of fit (time-rescaling)     -> gof_rescaled.npz
  §4  dimension sweep + kernel ladder       -> (uses run_ablation leaderboard) ladder.csv
  §5  interpretability                      -> interp_seed*.npz, interp_agg.npz
  §6  early warning (negative result)       -> earlywarning.csv
  §7  robustness (threshold + vol-window)   -> robust_tau.csv, robust_volwin.csv

USAGE
    python analysis.py            # run every stage (idempotent; skips cached)
    python analysis.py --quick    # fewer seeds / epochs, for a smoke test
    python analysis.py --stages interp gof   # run only named stages
Stages: headline gof interp earlywarning robust ladder
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

import run_ablation as R

ROOT = Path(__file__).resolve().parent
ADIR = ROOT / "results" / "analysis"
ADIR.mkdir(parents=True, exist_ok=True)

CANON = dict(tau=2.5, dim=8, betas=(0.35, 1.0, 3.0, 8.0), vol_win=60,
             split=0.70, epochs=400, lr=0.03, hid=16)


# ----------------------------------------------------------------------
# TRAIN + EXTRACT (torch -> numpy arrays; that is all downstream code needs)
# ----------------------------------------------------------------------
def train_r3(times, marks, T, D, tr, betas, dim, epochs, lr, hid, seed):
    torch.manual_seed(seed)
    prep = R.prep_mixture(times[tr], marks[tr], D, CANON["split"] * T, betas)
    m = R.train(R.ConditionalKernelHawkes(D, betas, dim, hid), prep, epochs, lr)
    with torch.no_grad():
        return dict(mu=m.mu().numpy(), alpha=m.alpha().numpy(),
                    pi=m.pi().numpy(), betas=np.asarray(betas, float),
                    src=m.emb.src.numpy(), tgt=m.emb.tgt.numpy(),
                    gain=float(m.emb.gain.numpy()),
                    ll=R.window_ll_conditional(m, times, marks, D, CANON["split"] * T, T),
                    branching=m.branching_ratio())


def load_canon_events():
    t, m, T, D, labels, sectors = R.get_events(CANON["tau"], CANON["vol_win"])
    return t, m.astype(int), float(T), int(D), list(labels), list(sectors)


# ----------------------------------------------------------------------
# NUMPY intensity / compensator primitives (shared by GOF, early-warning,
# paired test). Kernel of R3:  lambda_i(t) = mu_i + sum_{k:t_k<t} sum_c
#   alpha[i,mk] * pi[i,mk,c] * beta_c * exp(-beta_c (t - t_k)).
# Each component integrates to alpha*pi (beta*exp integrates to 1).
# ----------------------------------------------------------------------
def intensity_at(P, times, marks, query):
    """lambda_i(query) for R3, all marks. Returns (Q, D). Strictly causal
    (t_k < query): correct for off-event query times such as the daily grid used
    by the early-warning features (leakage-safe)."""
    mu, A, pi, betas = P["mu"], P["alpha"], P["pi"], P["betas"]
    q = np.asarray(query, float)[:, None]
    dt = q - times[None, :]                                           # (Q,N)
    causal = (dt > 0)
    out = np.tile(mu.astype(float), (len(q), 1))                      # (Q,D)
    for c, bc in enumerate(betas):
        k = bc * np.exp(-bc * np.clip(dt, 0, None)) * causal          # (Q,N)
        Wc = A[:, marks] * pi[:, marks, c]                            # (D,N)
        out += k @ Wc.T                                              # (Q,D)
    return out


def event_intensities_R3(P, times, marks):
    """lambda_{m_i}(t_i) at every event, using the model's exact tie-inclusive
    (array-index) convention -- identical to window_ll_conditional -- so per-event
    scores sum to the reported held-out LL. Same-day ties (k<i by index) contribute."""
    mu, A, pi, betas = P["mu"], P["alpha"], P["pi"], P["betas"]
    N = len(times); mi = marks
    dt = times[:, None] - times[None, :]
    mask = np.tril(np.ones((N, N)), -1)
    lam = mu[mi].astype(float)
    for c, bc in enumerate(betas):
        Bm = A * pi[:, :, c] * bc
        K = np.exp(-bc * np.clip(dt, 0, None)) * mask
        lam += (K * Bm[mi][:, mi]).sum(1)
    return lam


def event_intensities_R2(mu, A, w, betas, times, marks):
    """Same tie-inclusive per-event intensity for the shared-mixture reference."""
    N = len(times); mi = marks
    dt = times[:, None] - times[None, :]
    mask = np.tril(np.ones((N, N)), -1)
    lam = mu[mi].astype(float)
    for m, bm in enumerate(betas):
        K = np.exp(-bm * np.clip(dt, 0, None)) * mask
        lam += w[m] * (K * A[mi][:, mi]).sum(1)
    return lam


def total_compensator(P, times, marks, D, a, b):
    """int_a^b sum_d lambda_d(s) ds for R3, exact (windowed)."""
    mu, A, pi, betas = P["mu"], P["alpha"], P["pi"], P["betas"]
    base = mu.sum() * (b - a)
    # column mass per source j and component c:  sum_i alpha[i,j]*pi[i,j,c]
    excite = 0.0
    tt = times
    with np.errstate(over="ignore"):
        for c, bc in enumerate(betas):
            colmass = (A * pi[:, :, c]).sum(0)                # (D,) over source j
            relu_a = np.clip(a - tt, 0, None)
            term = np.where(tt < b, np.exp(-bc * relu_a) - np.exp(-bc * (b - tt)), 0.0)
            excite += (colmass[marks] * term).sum()
    return base + excite


def per_mark_compensator_R3(P, times, marks, d, query):
    """Lambda_d(query) = int_0^query lambda_d.  Vectorized over query times."""
    mu, A, pi, betas = P["mu"], P["alpha"], P["pi"], P["betas"]
    q = np.asarray(query, float)[:, None]
    tt = times[None, :]
    dt = q - tt
    causal = (dt > 0)
    acc = np.zeros(dt.shape)
    for c, bc in enumerate(betas):
        surv = (1 - np.exp(-bc * np.clip(dt, 0, None))) * causal
        acc += (A[d, marks] * pi[d, marks, c])[None, :] * surv
    return mu[d] * q[:, 0] + acc.sum(1)


# ======================================================================
# §5  INTERPRETABILITY
# ======================================================================
def stage_interp(ev, seeds):
    times, marks, T, D, labels, sectors = ev
    tr = times < CANON["split"] * T
    sectors = np.array(sectors)
    per_seed = []
    for s in seeds:
        f = ADIR / f"interp_seed{s}.npz"
        if f.exists():
            per_seed.append(dict(np.load(f, allow_pickle=True)))
            continue
        P = train_r3(times, marks, T, D, tr, CANON["betas"], CANON["dim"],
                     CANON["epochs"], CANON["lr"], CANON["hid"], s)
        A, pi, betas = P["alpha"], P["pi"], P["betas"]
        # effective edge speed j->i:  beta_bar = sum_c pi*beta ; delay = sum_c pi/beta
        beta_bar = (pi * betas[None, None, :]).sum(2)             # (D,D) [target,source]
        delay = (pi * (1.0 / betas)[None, None, :]).sum(2)        # (D,D) expected days
        out_str = A.sum(0)                                        # transmitter (col sum)
        in_str = A.sum(1)                                         # receiver (row sum)
        net = out_str - in_str
        rec = dict(alpha=A, pi=pi, betas=betas, beta_bar=beta_bar, delay=delay,
                   out_str=out_str, in_str=in_str, net=net, mu=P["mu"], ll=P["ll"],
                   src=P["src"], tgt=P["tgt"], gain=P["gain"],
                   branching=P["branching"], labels=np.array(labels), sectors=sectors)
        np.savez(f, **rec)
        per_seed.append(rec)
        print(f"  [interp] seed {s}: LL={P['ll']:.4f} branching={P['branching']:.3f}")

    # aggregate across seeds (uncertainty control per §5)
    net_stack = np.stack([p["net"] for p in per_seed])           # (S, D)
    beta_stack = np.stack([p["beta_bar"] for p in per_seed])
    delay_stack = np.stack([p["delay"] for p in per_seed])
    alpha_mean = np.mean([p["alpha"] for p in per_seed], 0)
    # own volatility per stock (std of daily log returns) for Fig 6
    ret = np.load(ROOT / "data" / "prices.npz", allow_pickle=True)["returns"]
    own_vol = ret.std(0)                                         # (D,)
    net_mean, net_sd = net_stack.mean(0), net_stack.std(0)
    r_pear, p_pear = stats.pearsonr(net_mean, own_vol)
    r_spear, p_spear = stats.spearmanr(net_mean, own_vol)
    np.savez(ADIR / "interp_agg.npz",
             net_mean=net_mean, net_sd=net_sd, out_mean=np.mean([p["out_str"] for p in per_seed], 0),
             in_mean=np.mean([p["in_str"] for p in per_seed], 0),
             beta_bar_mean=beta_stack.mean(0), delay_mean=delay_stack.mean(0),
             alpha_mean=alpha_mean, own_vol=own_vol,
             labels=np.array(labels), sectors=sectors,
             pearson=np.array([r_pear, p_pear]), spearman=np.array([r_spear, p_spear]))
    print(f"  [interp] net-transmission vs own-vol: Pearson r={r_pear:.3f} (p={p_pear:.3g}); "
          f"Spearman rho={r_spear:.3f} (p={p_spear:.3g})")


# ======================================================================
# §3  GOODNESS OF FIT  (time-rescaling theorem)
# ======================================================================
def stage_gof(ev):
    f = ADIR / "gof_rescaled.npz"
    if f.exists():
        print("  [gof] cached"); return
    times, marks, T, D, labels, sectors = ev
    tr = times < CANON["split"] * T
    # reuse seed-0 interp model if present, else train
    p0 = ADIR / "interp_seed0.npz"
    P = dict(np.load(p0, allow_pickle=True)) if p0.exists() else \
        train_r3(times, marks, T, D, tr, CANON["betas"], CANON["dim"],
                 CANON["epochs"], CANON["lr"], CANON["hid"], 0)
    t_cut = CANON["split"] * T
    xis = []
    for d in range(D):
        td = np.sort(times[marks == d])
        if len(td) < 3:
            continue
        Lam = per_mark_compensator_R3(P, times, marks, d, td)     # cumulative compensator
        inc = np.diff(Lam)                                        # ~Exp(1) if well specified
        later = td[1:]                                            # increment "ends" at later event
        xis.append(inc[later > t_cut])                            # held-out increments only
    xi = np.concatenate(xis)
    xi = xi[np.isfinite(xi) & (xi >= 0)]
    u = 1 - np.exp(-xi)                                           # -> Uniform(0,1) under H0
    ks_stat, ks_p = stats.kstest(u, "uniform")
    np.savez(f, xi=xi, u=u, ks=np.array([ks_stat, ks_p]), n=len(xi))
    print(f"  [gof] {len(xi)} held-out rescaled increments; KS(uniform) stat={ks_stat:.4f} p={ks_p:.3g}")


# ======================================================================
# §2  HEADLINE CIs + PAIRED SIGNIFICANCE
# ======================================================================
def _per_event_scores_R3(P, times, marks, D, a, b):
    """Per-event held-out score s_i = log lambda_{m_i}(t_i) - dComp_i, summing to
    the total windowed LL. dComp_i = compensator mass on (t_{i-1}, t_i]."""
    lamv = event_intensities_R3(P, times, marks)
    inw = np.where((times > a) & (times <= b))[0]
    ti = times[inw]
    loglam = np.log(np.clip(lamv[inw], 1e-12, None))
    edges = np.concatenate([[a], ti])
    dcomp = np.array([total_compensator(P, times, marks, D, edges[k], edges[k + 1])
                      for k in range(len(ti))])
    return loglam - dcomp


def _per_event_scores_R2(mu, A, w, betas, times, marks, D, a, b):
    """Same decomposition for the shared-mixture reference (R2)."""
    lamv = event_intensities_R2(mu, A, w, betas, times, marks)
    inw = np.where((times > a) & (times <= b))[0]
    ti = times[inw]
    loglam = np.log(np.clip(lamv[inw], 1e-12, None))
    colmass = A.sum(0)                                            # sum_i alpha[i,j]

    def comp(u, v):
        base = mu.sum() * (v - u)
        exc = 0.0
        with np.errstate(over="ignore"):
            for m, bm in enumerate(betas):
                relu = np.clip(u - times, 0, None)
                term = np.where(times < v, np.exp(-bm * relu) - np.exp(-bm * (v - times)), 0.0)
                exc += (w[m] / bm) * (colmass[marks] * term).sum()
        return base + exc
    edges = np.concatenate([[a], ti])
    dcomp = np.array([comp(edges[k], edges[k + 1]) for k in range(len(ti))])
    return loglam - dcomp


def stage_headline(ev, seeds):
    f = ADIR / "headline_ci.csv"
    times, marks, T, D, labels, sectors = ev
    tr = times < CANON["split"] * T
    t_cut = CANON["split"] * T

    if not f.exists():
        # multi-seed LL for proposed + structured references (CIs). Poisson floor
        # is closed-form (seed-invariant) -> point value.
        rows = []
        for s in seeds:
            ic = ADIR / f"interp_seed{s}.npz"
            ll3 = float(np.load(ic)["ll"]) if ic.exists() else \
                train_r3(times, marks, T, D, tr, CANON["betas"], CANON["dim"],
                         CANON["epochs"], CANON["lr"], CANON["hid"], s)["ll"]
            rows.append(("Proposed conditional kernel", s, ll3))
            torch.manual_seed(s)
            prep = R.prep_mixture(times[tr], marks[tr], D, t_cut, CANON["betas"])
            m2 = R.train(R.MixtureHawkes(D, CANON["dim"], CANON["betas"]), prep,
                         CANON["epochs"], CANON["lr"])
            with torch.no_grad():
                ll2 = R.window_ll_mixture(m2.mu(), m2.alpha(), m2.weights(), CANON["betas"],
                                          times, marks, D, t_cut, T)
            rows.append(("Shared mixture (R2)", s, ll2))
            torch.manual_seed(s)
            prep1 = R.prep_single(times[tr], marks[tr], D, t_cut, 0.35)
            m1 = R.train(R.SingleExpHawkes(D, CANON["dim"], 0.35), prep1, CANON["epochs"], CANON["lr"])
            with torch.no_grad():
                ll1 = R.window_ll_single(m1.mu(), m1.alpha(), 0.35, times, marks, D, t_cut, T)
            rows.append(("Embedding single-exp (R1)", s, ll1))
        # Poisson floor (closed form)
        for nm, pm in [("Homogeneous Poisson", False), ("Per-mark Poisson", True)]:
            mu_p = R.fit_poisson_rates(times, marks, D, t_cut, per_mark=pm)
            rows.append((nm, 0, R.window_ll_poisson(mu_p, times, marks, t_cut, T)))
        df = pd.DataFrame(rows, columns=["model", "seed", "ll"])
        agg = df.groupby("model")["ll"].agg(["mean", "std", "count"]).reset_index()
        agg["sem"] = agg["std"].fillna(0) / np.sqrt(agg["count"])
        agg["ci95"] = 1.96 * agg["sem"]
        agg["is_floor"] = agg["model"].str.contains("Poisson")
        agg.sort_values("mean", ascending=False).to_csv(f, index=False)
        print("  [headline]\n", agg.sort_values("mean", ascending=False).to_string(index=False))

    # paired significance: Proposed vs R2, per-held-out-event log-lik contributions
    pf = ADIR / "paired_test.json"
    if not pf.exists():
        P = dict(np.load(ADIR / "interp_seed0.npz", allow_pickle=True)) if (ADIR / "interp_seed0.npz").exists() \
            else train_r3(times, marks, T, D, tr, CANON["betas"], CANON["dim"],
                          CANON["epochs"], CANON["lr"], CANON["hid"], 0)
        torch.manual_seed(0)
        prep = R.prep_mixture(times[tr], marks[tr], D, t_cut, CANON["betas"])
        m2 = R.train(R.MixtureHawkes(D, CANON["dim"], CANON["betas"]), prep, CANON["epochs"], CANON["lr"])
        with torch.no_grad():
            mu2, A2, w2 = m2.mu().numpy(), m2.alpha().numpy(), m2.weights().numpy()
        s_prop = _per_event_scores_R3(P, times, marks, D, t_cut, T)
        s_r2 = _per_event_scores_R2(mu2, A2, w2, CANON["betas"], times, marks, D, t_cut, T)
        diff = s_prop - s_r2
        w_stat, w_p = stats.wilcoxon(diff)
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(2000)])
        res = dict(n_events=len(diff), mean_diff=float(diff.mean()),
                   ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
                   wilcoxon_stat=float(w_stat), wilcoxon_p=float(w_p),
                   mean_ll_prop=float(s_prop.mean()), mean_ll_r2=float(s_r2.mean()))
        json.dump(res, open(pf, "w"), indent=1)
        print(f"  [paired] Proposed-R2 mean per-event delta={res['mean_diff']:.4f} "
              f"[{res['ci_lo']:.4f},{res['ci_hi']:.4f}]  Wilcoxon p={w_p:.3g}")


# ======================================================================
# §4  KERNEL LADDER table (low-rank single-exp -> shared mixture -> conditional)
# ======================================================================
def stage_ladder(ev, seeds):
    f = ADIR / "ladder.csv"
    if f.exists():
        print("  [ladder] cached"); return
    times, marks, T, D, labels, sectors = ev
    tr = times < CANON["split"] * T
    t_cut = CANON["split"] * T
    rows = []
    for s in seeds:
        torch.manual_seed(s)
        p1 = R.prep_single(times[tr], marks[tr], D, t_cut, 0.35)
        m1 = R.train(R.SingleExpHawkes(D, CANON["dim"], 0.35), p1, CANON["epochs"], CANON["lr"])
        with torch.no_grad():
            rows.append(("R1 low-rank single-exp", s,
                         R.window_ll_single(m1.mu(), m1.alpha(), 0.35, times, marks, D, t_cut, T)))
        torch.manual_seed(s)
        p2 = R.prep_mixture(times[tr], marks[tr], D, t_cut, CANON["betas"])
        m2 = R.train(R.MixtureHawkes(D, CANON["dim"], CANON["betas"]), p2, CANON["epochs"], CANON["lr"])
        with torch.no_grad():
            rows.append(("R2 shared mixture", s,
                         R.window_ll_mixture(m2.mu(), m2.alpha(), m2.weights(), CANON["betas"],
                                             times, marks, D, t_cut, T)))
        ic = ADIR / f"interp_seed{s}.npz"
        ll3 = float(np.load(ic)["ll"]) if ic.exists() else \
            train_r3(times, marks, T, D, tr, CANON["betas"], CANON["dim"],
                     CANON["epochs"], CANON["lr"], CANON["hid"], s)["ll"]
        rows.append(("R3 source-target conditioned", s, ll3))
    df = pd.DataFrame(rows, columns=["rung", "seed", "ll"])
    agg = df.groupby("rung")["ll"].agg(["mean", "std", "count"]).reset_index()
    agg["order"] = agg["rung"].map({"R1 low-rank single-exp": 0, "R2 shared mixture": 1,
                                    "R3 source-target conditioned": 2})
    agg.sort_values("order").to_csv(f, index=False)
    print("  [ladder]\n", agg.sort_values("order").to_string(index=False))


# ======================================================================
# §6  EARLY WARNING (honest negative result)
# ======================================================================
def stage_earlywarning(ev):
    f = ADIR / "earlywarning.csv"
    if f.exists():
        print("  [earlywarning] cached"); return
    times, marks, T, D, labels, sectors = ev
    tr = times < CANON["split"] * T
    p0 = ADIR / "interp_seed0.npz"
    P = dict(np.load(p0, allow_pickle=True)) if p0.exists() else \
        train_r3(times, marks, T, D, tr, CANON["betas"], CANON["dim"],
                 CANON["epochs"], CANON["lr"], CANON["hid"], 0)
    A = P["alpha"]
    # forward volatility label, aligned to event-frame days (events drop first vol_win days)
    fwd_vol = np.load(ROOT / "data" / "prices.npz", allow_pickle=True)["fwd_vol"]
    fwd = fwd_vol[CANON["vol_win"]:CANON["vol_win"] + int(T)]         # (T,)
    days = np.arange(1, int(T))                                       # decision days (>=1 for causal features)
    thr = np.nanquantile(fwd, 0.80)
    y = (fwd[days] >= thr).astype(int)
    base_rate = y.mean()

    # per-mark model intensity on each decision day -> node feature matrix (days, D)
    node_lam = intensity_at(P, times, marks, days.astype(float))     # (days, D)
    lam_tot = node_lam.sum(1)                                         # total intensity
    # per-day, per-stock event-count matrix (vectorized), then leakage-safe trailing sums
    Tint = int(T)
    C = np.zeros((Tint, D))                                           # C[day, stock]
    np.add.at(C, (np.clip(times.astype(int), 0, Tint - 1), marks), 1.0)
    cum = np.vstack([np.zeros(D), np.cumsum(C, 0)])                   # cum[t] = events on days <t

    def trail_mat(win):                                              # (days, D) counts on [d-win, d)
        lo = np.clip(days - win, 0, Tint)
        return cum[days] - cum[lo]
    tm7, tm21 = trail_mat(7), trail_mat(21)
    c7, c21 = tm7.sum(1), tm21.sum(1)
    disp = tm21.std(1)                                               # cross-sectional dispersion
    # GNN substrate: graph diffusion of node intensities on the learned network
    An = A / (A.sum(1, keepdims=True) + 1e-9)                         # row-normalized
    h1 = node_lam @ An.T                                             # 1-hop
    h2 = h1 @ An.T                                                    # 2-hop
    gnn_feat = np.column_stack([node_lam.sum(1), h1.sum(1), h2.sum(1),
                                node_lam.max(1), h1.max(1)])

    substrates = {
        "Branching-ratio logistic (2 feat)": (np.column_stack([lam_tot, c7]), LogisticRegression(max_iter=1000)),
        "Low-dim summary (logistic)": (np.column_stack([c7, c21, disp, lam_tot]), LogisticRegression(max_iter=1000)),
        "Low-dim summary (RF)": (np.column_stack([c7, c21, disp, lam_tot]), RandomForestClassifier(n_estimators=200, random_state=0)),
        "Learned-network GNN substrate (RF)": (gnn_feat, RandomForestClassifier(n_estimators=200, random_state=0)),
    }
    ntr = int(0.70 * len(days))                                      # chronological, leakage-safe
    rows = []
    for name, (X, clf) in substrates.items():
        Xs = StandardScaler().fit(X[:ntr]).transform(X)
        clf.fit(Xs[:ntr], y[:ntr])
        proba = clf.predict_proba(Xs[ntr:])[:, 1]
        auc = roc_auc_score(y[ntr:], proba) if len(np.unique(y[ntr:])) > 1 else np.nan
        rows.append((name, X.shape[1], round(auc, 4)))
    df = pd.DataFrame(rows, columns=["substrate", "n_features", "test_auroc"])
    df["base_rate"] = round(base_rate, 4)
    df["n_test"] = len(days) - ntr
    df.to_csv(f, index=False)
    print(f"  [earlywarning] base rate={base_rate:.3f}, n_test={len(days)-ntr}\n",
          df.to_string(index=False))


# ======================================================================
# §7  ROBUSTNESS (threshold tau + volatility-window)
# ======================================================================
def stage_robust(seeds):
    ft, fv = ADIR / "robust_tau.csv", ADIR / "robust_volwin.csv"
    if not ft.exists():
        rows = []
        for tau in [2.0, 2.5, 3.0]:
            t, m, T, D, labels, sectors = R.get_events(tau, CANON["vol_win"])
            m = m.astype(int); tr = t < CANON["split"] * T
            for s in seeds:
                P = train_r3(t, m, T, D, tr, CANON["betas"], CANON["dim"],
                             300, CANON["lr"], CANON["hid"], s)
                rows.append((tau, len(t), s, P["ll"], P["branching"]))
                print(f"  [robust-tau] tau={tau} events={len(t)} seed={s} LL={P['ll']:.4f}")
        pd.DataFrame(rows, columns=["tau", "n_events", "seed", "ll", "branching"]).to_csv(ft, index=False)
    if not fv.exists():
        rows = []
        for vw in [40, 60, 90]:
            t, m, Tv = R.events_from_returns(
                np.load(ROOT / "data" / "prices.npz", allow_pickle=True)["returns"], CANON["tau"], vw)
            m = m.astype(int); D = 48; tr = t < CANON["split"] * Tv
            P = train_r3(t, m, Tv, D, tr, CANON["betas"], CANON["dim"], 300, CANON["lr"], CANON["hid"], 0)
            rows.append((vw, len(t), P["ll"], P["branching"]))
            print(f"  [robust-volwin] vol_win={vw} events={len(t)} LL={P['ll']:.4f}")
        pd.DataFrame(rows, columns=["vol_win", "n_events", "ll", "branching"]).to_csv(fv, index=False)


# ======================================================================
# §8  EMBEDDING GEOMETRY  (word2vec-style: gauge-fix -> ARI -> stability -> asymmetry)
# The raw src/tgt are identified only up to a GL(d) transform, so their cosine
# geometry is NOT reproducible. The identified object is the bilinear score
# Z = tgt @ src.T; its SVD gives canonical, rotation-fixed receiver (U) and
# transmitter (V) coordinates. All geometry below is computed on U, V.
# ======================================================================
def _gauge_fix(src, tgt, r=None):
    Z = tgt @ src.T                                   # (D,D) identified bilinear score
    U, sv, Vt = np.linalg.svd(Z)
    if r is None:
        r = max(2, int((sv > 1e-6 * sv[0]).sum()))
    sq = np.sqrt(sv[:r])
    return U[:, :r] * sq, Vt[:r].T * sq, sv           # receiver coords, transmitter coords, spectrum


def _cos(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def _knn_same_sector(sim, sec, k=3):
    D = len(sec); out = []
    for i in range(D):
        order = np.argsort(-sim[i]); order = order[order != i][:k]
        out.append(np.mean([sec[j] == sec[i] for j in order]))
    return float(np.mean(out))


def _knn_sets(sim, k=3):
    D = sim.shape[0]
    return [set(np.argsort(-sim[i])[np.argsort(-sim[i]) != i][:k]) for i in range(D)]


def stage_embed(ev, seeds, nperm=2000):
    f = ADIR / "embed_sector_recovery.csv"
    if f.exists():
        print("  [embed] cached"); return
    caches = [ADIR / f"interp_seed{s}.npz" for s in seeds]
    if not all(c.exists() and "src" in np.load(c) for c in caches):
        print("  [embed] need interp caches WITH src/tgt — delete results/analysis/interp_seed*.npz "
              "and rerun stage interp first."); return
    _, _, _, D, labels, sectors = ev
    sec = np.array(sectors); labels = np.array(labels)
    nsec = len(set(sectors))
    sec_code = pd.factorize(sec)[0]
    rng = np.random.default_rng(0)

    rows, rec_sims, tra_sims = [], [], []
    Ur0 = Vr0 = None
    for s in seeds:
        d = np.load(caches[seeds.index(s)], allow_pickle=True)
        Ur, Vr, sv = _gauge_fix(d["src"], d["tgt"])
        rs, ts = _cos(Ur), _cos(Vr)
        rec_sims.append(rs); tra_sims.append(ts)
        if s == seeds[0]:
            Ur0, Vr0, sv0 = Ur, Vr, sv
        for space, X, sim in [("receiver", Ur, rs), ("transmitter", Vr, ts)]:
            ari, nmi, _ = (lambda lab: (adjusted_rand_score(sec_code, lab),
                                        normalized_mutual_info_score(sec_code, lab), lab))(
                AgglomerativeClustering(n_clusters=nsec).fit_predict(X))
            rows.append(dict(seed=s, space=space, ari=round(ari, 4), nmi=round(nmi, 4),
                             knn3_same_sector=round(_knn_same_sector(sim, sec, 3), 4)))
    df = pd.DataFrame(rows)
    agg = df.groupby("space")[["ari", "nmi", "knn3_same_sector"]].agg(["mean", "std"])
    df.to_csv(f, index=False)

    # permutation null (fix seed0 clusters, shuffle sector labels)
    null, null_dist = {}, {}
    for space, X in [("receiver", Ur0), ("transmitter", Vr0)]:
        lab = AgglomerativeClustering(n_clusters=nsec).fit_predict(X)
        obs_ari = adjusted_rand_score(sec_code, lab)
        obs_knn = _knn_same_sector(_cos(X), sec, 3)
        nd_ari = np.array([adjusted_rand_score(rng.permutation(sec_code), lab) for _ in range(nperm)])
        sim = _cos(X); sets = _knn_sets(sim, 3)
        nd_knn = np.array([np.mean([np.mean([sec[p][j] == sec[p][i] for j in sets[i]])
                                    for i in range(len(sec))])
                           for p in (rng.permutation(len(sec)) for _ in range(nperm))])
        null_dist[space] = nd_ari
        null[space] = dict(obs_ari=float(obs_ari), p_ari=float((nd_ari >= obs_ari).mean()),
                           null_ari_mean=float(nd_ari.mean()),
                           obs_knn=float(obs_knn), p_knn=float((nd_knn >= obs_knn).mean()),
                           null_knn_mean=float(nd_knn.mean()))
    json.dump(null, open(ADIR / "embed_null.json", "w"), indent=1)
    np.savez(ADIR / "embed_null.npz",
             recv_null=null_dist["receiver"], recv_obs=null["receiver"]["obs_ari"],
             tra_null=null_dist["transmitter"], tra_obs=null["transmitter"]["obs_ari"])

    # seed stability: cosine-sim matrices are rotation-invariant -> compare directly
    iu = np.triu_indices(D, 1)
    def _pair_corr(sims):
        cs = [stats.pearsonr(sims[a][iu], sims[b][iu])[0]
              for a in range(len(sims)) for b in range(a + 1, len(sims))]
        return float(np.mean(cs)) if cs else 1.0
    def _pair_knn_overlap(sims, k=3):
        sets = [_knn_sets(s, k) for s in sims]
        ov = []
        for a in range(len(sims)):
            for b in range(a + 1, len(sims)):
                ov.append(np.mean([len(sets[a][i] & sets[b][i]) / k for i in range(D)]))
        return float(np.mean(ov)) if ov else 1.0
    stability = dict(
        receiver_simcorr=_pair_corr(rec_sims), transmitter_simcorr=_pair_corr(tra_sims),
        receiver_knn3_overlap=_pair_knn_overlap(rec_sims), transmitter_knn3_overlap=_pair_knn_overlap(tra_sims),
        n_seed_pairs=len(seeds) * (len(seeds) - 1) // 2)
    json.dump(stability, open(ADIR / "embed_stability.json", "w"), indent=1)

    # src/tgt asymmetry (seed 0): transmitter- vs receiver-neighbourhood per stock
    rs0, ts0 = _cos(Ur0), _cos(Vr0)
    global_asym = 1 - stats.pearsonr(rs0[iu], ts0[iu])[0]
    rec_sets, tra_sets = _knn_sets(rs0, 3), _knn_sets(ts0, 3)
    def _nn_names(sim, i, k=3):
        return ",".join(labels[j] for j in [x for x in np.argsort(-sim[i]) if x != i][:k])
    arows = []
    for i in range(D):
        ov = len(rec_sets[i] & tra_sets[i]) / 3
        arows.append(dict(stock=labels[i], sector=sec[i], role_overlap=round(ov, 3),
                          transmitter_neighbours=_nn_names(ts0, i),
                          receiver_neighbours=_nn_names(rs0, i)))
    adf = pd.DataFrame(arows).sort_values("role_overlap").reset_index(drop=True)
    adf.to_csv(ADIR / "embed_asymmetry.csv", index=False)
    np.savez(ADIR / "embed_coords.npz", Urec=Ur0, Vtra=Vr0, sv=sv0,
             labels=labels, sectors=sec,
             clab_recv=AgglomerativeClustering(n_clusters=nsec).fit_predict(Ur0))
    print("  [embed] sector recovery (mean over seeds):\n", agg.to_string())
    print(f"  [embed] null: receiver ARI={null['receiver']['obs_ari']:.3f} "
          f"(p={null['receiver']['p_ari']:.3g}); transmitter ARI={null['transmitter']['obs_ari']:.3f} "
          f"(p={null['transmitter']['p_ari']:.3g})")
    print(f"  [embed] seed stability sim-corr recv={stability['receiver_simcorr']:.3f} "
          f"tra={stability['transmitter_simcorr']:.3f}; kNN-overlap recv={stability['receiver_knn3_overlap']:.2f}")
    print(f"  [embed] global role-asymmetry (1-corr) = {global_asym:.3f}; "
          f"most asymmetric: {', '.join(adf['stock'].head(5))}")


# ======================================================================
# §9  SIGNED SHOCKS  (isolate negative vs positive; the model itself is unsigned)
# Refit the SAME R3 on negative-only (z<-tau) and positive-only (z>+tau) events
# and compare contagiousness (branching), predictability (LL), speed and leaders.
# ======================================================================
def signed_events(returns, tau, vol_win, sign):
    df = pd.DataFrame(returns)
    z = (df / df.rolling(vol_win).std()).iloc[vol_win:].reset_index(drop=True)
    ev_day, ev_mark = [], []
    for j in range(df.shape[1]):
        v = z[j].values
        hits = np.where(v > tau)[0] if sign == "pos" else np.where(v < -tau)[0]
        ev_day.extend(hits.tolist()); ev_mark.extend([j] * len(hits))
    order = np.argsort(ev_day, kind="mergesort")
    return (np.array(ev_day, float)[order], np.array(ev_mark, int)[order], float(len(z)))


def stage_signed(ev, seeds):
    f = ADIR / "signed_comparison.csv"
    if f.exists():
        print("  [signed] cached"); return
    returns = np.load(ROOT / "data" / "prices.npz", allow_pickle=True)["returns"]
    _, _, _, D, labels, sectors = ev
    labels = np.array(labels); sectors = np.array(sectors)
    betas = np.asarray(CANON["betas"], float)
    rows, leaders = [], {}
    for sign in ["neg", "pos"]:
        t, m, T = signed_events(returns, CANON["tau"], CANON["vol_win"], sign)
        tr = t < CANON["split"] * T
        nets = []
        for s in seeds:
            P = train_r3(t, m, T, D, tr, CANON["betas"], CANON["dim"],
                         CANON["epochs"], CANON["lr"], CANON["hid"], s)
            A, pi = P["alpha"], P["pi"]
            delay = (pi * (1.0 / betas)[None, None, :]).sum(2)              # (D,D) expected days
            wdelay = float((A * delay).sum() / A.sum())                    # excitation-weighted delay
            rows.append(dict(sign=sign, seed=s, n_events=len(t), ll=round(P["ll"], 4),
                             branching=round(P["branching"], 4),
                             alpha_mass=round(float(A.sum()), 3),
                             wmean_delay=round(wdelay, 3)))
            nets.append(A.sum(0) - A.sum(1))                               # net transmission
            print(f"  [signed-{sign}] seed={s} n={len(t)} LL={P['ll']:.4f} "
                  f"branching={P['branching']:.3f} wdelay={wdelay:.2f}")
        net = np.mean(nets, 0)
        top = np.argsort(-net)[:6]; bot = np.argsort(net)[:6]
        leaders[sign] = dict(
            transmitters=[(labels[i], sectors[i], round(float(net[i]), 3)) for i in top],
            receivers=[(labels[i], sectors[i], round(float(net[i]), 3)) for i in bot])
    pd.DataFrame(rows).to_csv(f, index=False)
    json.dump(leaders, open(ADIR / "signed_leaders.json", "w"), indent=1)
    g = pd.DataFrame(rows).groupby("sign").agg(
        n=("n_events", "first"), ll=("ll", "mean"), branching=("branching", "mean"),
        wmean_delay=("wmean_delay", "mean")).reset_index()
    print("  [signed] summary:\n", g.to_string(index=False))
    print(f"  [signed] top NEGATIVE-shock transmitters: "
          f"{', '.join(x[0] for x in leaders['neg']['transmitters'][:5])}")
    print(f"  [signed] top POSITIVE-shock transmitters: "
          f"{', '.join(x[0] for x in leaders['pos']['transmitters'][:5])}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stages", nargs="+",
                    default=["interp", "gof", "headline", "ladder", "earlywarning",
                             "robust", "embed", "signed"])
    a = ap.parse_args()
    if a.quick:
        CANON["epochs"] = 120
        seeds = [0]
    else:
        seeds = [0, 1, 2]
    t0 = time.time()
    ev = load_canon_events()
    print(f"canonical: tau={CANON['tau']} D={ev[3]} events={len(ev[0])} T={int(ev[2])} "
          f"train={(ev[0] < CANON['split']*ev[2]).sum()}")
    if "interp" in a.stages: stage_interp(ev, seeds)
    if "gof" in a.stages: stage_gof(ev)
    if "headline" in a.stages: stage_headline(ev, seeds)
    if "ladder" in a.stages: stage_ladder(ev, seeds)
    if "earlywarning" in a.stages: stage_earlywarning(ev)
    if "robust" in a.stages: stage_robust(seeds)
    if "embed" in a.stages: stage_embed(ev, seeds)
    if "signed" in a.stages: stage_signed(ev, seeds)
    print(f"\nDone in {time.time()-t0:.0f}s. Caches in {ADIR}")


if __name__ == "__main__":
    main()
