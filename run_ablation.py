"""
Nifty-50 Edge-Conditioned Temporal Kernel — Ablation Runner
=============================================================
ONE FILE. Run it top to bottom (locally or in Colab) and it will:

  1. load the cached returns panel (data/prices.npz), or download it via
     yfinance if that cache isn't present,
  2. build event data (marked point process) for every `tau` in the sweep,
     from the SAME returns panel -- no re-downloading per tau,
  3. train the proposed edge-conditioned kernel (R3) across the full grid of
     tau x embedding-dim x beta-bank x seed, plus the comparison set (run ONCE
     each, at the canonical config): 2 Poisson nulls, 2 Hawkes references,
     4 classical Hawkes (dense / L1-sparse / diagonal / correlation-network),
     1 neural-kernel Hawkes -- 10 methods total (including the proposed one),
     checkpointing every finished run to results/ablation_log.json so a
     Colab timeout / disconnect just resumes instead of restarting,
  4. compiles everything into a CSV leaderboard + summary plots, automatically,
     at the end of the same run.

USAGE
-----
Local / Colab shell:
    python run_ablation.py                # full grid (edit CONFIG below first)
    python run_ablation.py --quick        # tiny grid, few epochs -- smoke test
    python run_ablation.py --compile-only # just re-summarize an existing log
    python run_ablation.py --taus 2.0 2.5 3.0 --dims 4 8 16 --seeds 0 1 2

Google Colab (paste into a cell, or run as a shell cell with `!`):
    !unzip -o nifty_ablation.zip -d .
    %cd nifty_ablation
    !pip install -q torch numpy pandas scipy matplotlib yfinance
    !python run_ablation.py --quick        # sanity check first
    !python run_ablation.py                # then the real grid

Outputs (all under results/):
    ablation_log.json     raw per-run records (resumable checkpoint)
    ablation_results.csv  same thing, flat table
    leaderboard.csv       R3 runs averaged over seeds, sorted best-first
    heatmap_tau_dim.png   held-out LL as a function of (tau, dim)
    beta_dim_ablation.png held-out LL vs dim, one line per beta-bank/tau
    comparison_table.csv  every method at the canonical config, ranked (the paper table)
    comparison_ranking.png horizontal bar of that table
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PRICES_CACHE = DATA_DIR / "prices.npz"
LOG_PATH = RESULTS_DIR / "ablation_log.json"

# ----------------------------------------------------------------------
# CONFIG -- edit this block to change what gets ablated
# ----------------------------------------------------------------------
DEFAULT_CONFIG = dict(
    TAUS=[2.0, 2.5, 3.0],                        # event-definition threshold (std devs)
    DIMS=[4, 8, 16],                              # embedding dimension d
    BETA_SETS={                                   # named timescale banks to ablate
        "fast":    (0.35, 1.0),
        "default": (0.35, 1.0, 3.0, 8.0),
        "wide":    (0.1, 0.35, 1.0, 3.0, 8.0, 15.0),
    },
    SEEDS=[0, 1, 2],
    EPOCHS=400,
    LR=0.03,
    HID=16,          # hypernetwork hidden width
    SPLIT=0.70,       # temporal train/test split
    VOL_WIN=60,       # rolling-vol window used to standardize returns before thresholding
    CLASSIC_EPOCHS=500,       # classical dense/sparse/diagonal Hawkes need more steps to converge
    CLASSIC_LR=0.05,
    # Canonical config: the SINGLE (tau, dim, beta-bank) point at which every
    # comparison method is evaluated once, to build the comparison table.
    # The parameter search (the tau x dim x beta x seed grid) runs on the
    # PROPOSED model ONLY -- baselines are never swept.
    CANON_TAU=2.5,
    CANON_DIM=8,
    CANON_BETA="default",
)

# Nifty-50 constituents (ticker, sector) -- only used if data/prices.npz is missing
# and needs to be rebuilt from scratch via yfinance.
NIFTY = [
    ("HDFCBANK", "Fin"), ("ICICIBANK", "Fin"), ("SBIN", "Fin"), ("KOTAKBANK", "Fin"),
    ("AXISBANK", "Fin"), ("BAJFINANCE", "Fin"), ("BAJAJFINSV", "Fin"), ("INDUSINDBK", "Fin"),
    ("SBILIFE", "Fin"), ("HDFCLIFE", "Fin"), ("SHRIRAMFIN", "Fin"),
    ("TCS", "IT"), ("INFY", "IT"), ("WIPRO", "IT"), ("HCLTECH", "IT"), ("TECHM", "IT"), ("LTIM", "IT"),
    ("RELIANCE", "Energy"), ("ONGC", "Energy"), ("NTPC", "Energy"), ("POWERGRID", "Energy"),
    ("COALINDIA", "Energy"), ("BPCL", "Energy"),
    ("MARUTI", "Auto"), ("TATAMOTORS", "Auto"), ("EICHERMOT", "Auto"), ("HEROMOTOCO", "Auto"),
    ("BAJAJ-AUTO", "Auto"), ("M&M", "Auto"),
    ("HINDUNILVR", "FMCG"), ("ITC", "FMCG"), ("NESTLEIND", "FMCG"), ("BRITANNIA", "FMCG"),
    ("TATACONSUM", "FMCG"),
    ("SUNPHARMA", "Pharma"), ("DRREDDY", "Pharma"), ("CIPLA", "Pharma"), ("DIVISLAB", "Pharma"),
    ("APOLLOHOSP", "Pharma"),
    ("TATASTEEL", "Metal"), ("JSWSTEEL", "Metal"), ("HINDALCO", "Metal"),
    ("ULTRACEMCO", "Matl"), ("GRASIM", "Matl"), ("ASIANPAINT", "Matl"),
    ("LT", "Infra"), ("ADANIENT", "Infra"), ("ADANIPORTS", "Infra"),
    ("BHARTIARTL", "Telecom"), ("TITAN", "Cons"),
]


# ========================================================================
# 1. DATA LAYER
# ========================================================================
def get_returns_panel(rebuild=False):
    """Returns (returns[T,D] np.ndarray, dates, labels, sectors). Cached to
    data/prices.npz so tau-ablation never needs to re-download anything."""
    if PRICES_CACHE.exists() and not rebuild:
        d = np.load(PRICES_CACHE, allow_pickle=True)
        return d["returns"], list(d["dates"]), list(d["labels"]), list(d["sectors"])

    print(f"[data] {PRICES_CACHE} not found -> downloading via yfinance (needs internet)...")
    import yfinance as yf
    series, labels, sectors = {}, [], []
    for tk, sec in NIFTY:
        try:
            s = yf.download(tk + ".NS", start="2017-06-01", end="2024-12-31",
                             progress=False, auto_adjust=True)["Close"].dropna()
            if hasattr(s, "columns"):
                s = s.iloc[:, 0]
            if len(s) < 400:
                continue
            series[tk] = s
            labels.append(tk)
            sectors.append(sec)
        except Exception:
            continue
    px = pd.DataFrame(series).dropna(how="all")
    px = px.dropna(axis=0, thresh=int(0.9 * px.shape[1]))
    px = px.ffill().dropna()
    ret = np.log(px).diff().dropna()
    dates = [str(d.date()) for d in ret.index]
    labels = list(ret.columns)
    sec_map = dict(NIFTY)
    sectors = [sec_map[t] for t in labels]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PRICES_CACHE, returns=ret.values, dates=np.array(dates),
              labels=np.array(labels), sectors=np.array(sectors))
    print(f"[data] cached {PRICES_CACHE}: days={len(ret)} stocks={len(labels)}")
    return ret.values, dates, labels, sectors


def events_from_returns(returns, tau, vol_win):
    """Rolling-vol standardize each column, flag |z| > tau as an event.
    Same recipe for every tau -- only the threshold changes."""
    df = pd.DataFrame(returns)
    z = df / df.rolling(vol_win).std()
    z = z.iloc[vol_win:].reset_index(drop=True)
    ev_day, ev_mark = [], []
    for j in range(df.shape[1]):
        hits = np.where(np.abs(z[j].values) > tau)[0]
        ev_day.extend(hits.tolist())
        ev_mark.extend([j] * len(hits))
    order = np.argsort(ev_day, kind="mergesort")
    times = np.array(ev_day, dtype=float)[order]
    marks = np.array(ev_mark, dtype=int)[order]
    T = float(len(z))
    return times, marks, T


def get_events(tau, vol_win):
    """Cached per-tau event data so re-running the sweep doesn't recompute."""
    cache_path = DATA_DIR / f"events_tau_{tau}.npz"
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        return (d["times"], d["marks"].astype(int), float(d["T"]), int(d["D"]),
                list(d["labels"]), list(d["sectors"]))
    returns, dates, labels, sectors = get_returns_panel()
    times, marks, T = events_from_returns(returns, tau, vol_win)
    D = returns.shape[1]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, times=times, marks=marks, T=T, D=D,
              labels=np.array(labels), sectors=np.array(sectors))
    print(f"[data] tau={tau}: D={D} events={len(times)} T={int(T)}")
    return times, marks, T, D, labels, sectors


# ========================================================================
# 2. MODELS
# ========================================================================
class Emb(nn.Module):
    """Shared source/target embedding factorization of the excitation matrix.
    alpha_ij = softplus(<target_i, source_j> + gain). Every model below reuses
    this -- only the temporal kernel on top of alpha changes."""
    def __init__(self, D, dd):
        super().__init__()
        self.log_mu = nn.Parameter(torch.log(torch.rand(D) * 0.05 + 1e-3))
        self.src = nn.Parameter(torch.randn(D, dd) * 0.1)
        self.tgt = nn.Parameter(torch.randn(D, dd) * 0.1)
        self.gain = nn.Parameter(torch.tensor(0.0))

    def alpha(self):
        return F.softplus(self.tgt @ self.src.t() + self.gain)

    def mu(self):
        return torch.exp(self.log_mu)


class SingleExpHawkes(nn.Module):
    """R1 -- shared single-exponential kernel (fixed beta, no timing flexibility)."""
    def __init__(self, D, dd, beta):
        super().__init__()
        self.emb = Emb(D, dd)
        self.beta = beta

    def alpha(self): return self.emb.alpha()
    def mu(self): return self.emb.mu()

    def branching_ratio(self):
        with torch.no_grad():
            return float(torch.linalg.eigvals(self.alpha() / self.beta).abs().max())

    def neg_loglik(self, prep):
        K, marks, G, Tw = prep
        mu, A = self.mu(), self.alpha()
        comp = Tw * mu.sum() + (A * G.unsqueeze(0)).sum() / self.beta
        lam = mu[marks] + (K * A[marks][:, marks]).sum(1)
        return comp - torch.log(lam.clamp(min=1e-12)).sum()


class MixtureHawkes(nn.Module):
    """R2 -- shared mixture-of-exponentials (flexible but market-wide timing:
    the mixture weights are the same for every edge)."""
    def __init__(self, D, dd, betas):
        super().__init__()
        self.emb = Emb(D, dd)
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.log_w = nn.Parameter(torch.zeros(len(betas)))

    def weights(self): return F.softplus(self.log_w)
    def alpha(self): return self.emb.alpha()
    def mu(self): return self.emb.mu()

    def branching_ratio(self):
        with torch.no_grad():
            r = torch.linalg.eigvals(self.alpha()).abs().max()
            return float(r * (self.weights() / self.betas).sum())

    def neg_loglik(self, prep):
        Ks, Gs, marks, Tw = prep
        mu, A, w = self.mu(), self.alpha(), self.weights()
        comp = Tw * mu.sum()
        lam = mu[marks].clone()
        for m in range(len(self.betas)):
            comp = comp + (w[m] / self.betas[m]) * (A * Gs[m].unsqueeze(0)).sum()
            lam = lam + w[m] * (Ks[m] * A[marks][:, marks]).sum(1)
        return comp - torch.log(lam.clamp(min=1e-12)).sum()


class ConditionalKernelHawkes(nn.Module):
    """R3 -- THE PROPOSED MODEL: edge-conditioned temporal kernel.
    phi_ij(tau) = sum_m pi_ij^m * b_m e^{-b_m tau}, where
    pi_ij = softmax(hypernet([target_i, source_j])) over a fixed timescale
    bank {b_m}. Each edge gets its OWN mixture over timescales -> directional
    contagion *speed*, not just directional strength.
    Nesting: if the hypernet ignores its input, pi collapses to R2 (shared
    mixture); R2 collapses to R1 if the mixture collapses to one component.
    So this class alone, swept over betas/dim, reproduces R1/R2/R3 behavior --
    R1 and R2 below are kept as separate light reference models purely so the
    ablation results have a fixed, non-ablated baseline to compare against."""
    def __init__(self, D, betas, dd, hid=16):
        super().__init__()
        self.D, self.dd = D, dd
        self.emb = Emb(D, dd)
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.M = len(betas)
        self.hyper = nn.Sequential(nn.Linear(2 * dd, hid), nn.Tanh(), nn.Linear(hid, self.M))

    def alpha(self): return self.emb.alpha()
    def mu(self): return self.emb.mu()

    def pi(self):
        t, s = self.emb.tgt, self.emb.src
        T_ = t.unsqueeze(1).expand(self.D, self.D, self.dd)
        S_ = s.unsqueeze(0).expand(self.D, self.D, self.dd)
        return torch.softmax(self.hyper(torch.cat([T_, S_], dim=-1)), dim=-1)

    def branching_ratio(self):
        with torch.no_grad():
            return float(torch.linalg.eigvals(self.alpha()).abs().max())

    def neg_loglik(self, prep):
        Ks, Gs, marks, Tw = prep
        mu, A, pi = self.mu(), self.alpha(), self.pi()
        comp = Tw * mu.sum()
        lam = mu[marks].clone()
        for m in range(self.M):
            Bm = A * pi[:, :, m] * self.betas[m]
            lam = lam + (Ks[m] * Bm[marks][:, marks]).sum(1)
            comp = comp + ((A * pi[:, :, m]) * Gs[m].unsqueeze(0)).sum()
        return comp - torch.log(lam.clamp(min=1e-12)).sum()


# ========================================================================
# 2b-classical. CLASSICAL HAWKES BASELINES (the "estimator ladder" rungs).
# All use the SAME single-exponential kernel (beta fixed) and the SAME
# windowed held-out LL as R1/R2/R3, so they are directly comparable.
#   Dense Hawkes       -- full free DxD excitation matrix (MLE), light L2
#                         (Ozaki 1979)
#   L1-Sparse Hawkes   -- same, with an L1 penalty on the excitation matrix
#                         (Bacry, Bompaire, Gaiffas, Muzy, JMLR 2020)
#   Diagonal Hawkes    -- self-excitation only (alpha diagonal); each stock
#                         excites only itself (Hawkes 1971, per-node)
#   Correlation net    -- non-parametric: alpha = c * (thresholded co-event
#                         correlation matrix), c line-searched on train
#                         (Mantegna 1999, correlation-based financial network)
#   Neural-kernel      -- factorized embedding alpha + a FLEXIBLE shared
#                         triggering kernel (fine exponential mixture, learned
#                         weights): a tractable stand-in for the neural/flexible
#                         triggering kernel of Isik et al. (MLHC 2023). NOT a
#                         faithful reproduction of their sigmoid-gated encoder.
# ========================================================================
class DenseHawkes(nn.Module):
    """Full free DxD excitation matrix, single-exponential kernel. lam_l1>0
    turns this into the L1-Sparse Hawkes; wd (in train) adds the light L2 that
    keeps the plain Dense-MLE from exploding."""
    def __init__(self, D, beta=0.35, lam_l1=0.0):
        super().__init__()
        self.log_mu = nn.Parameter(torch.log(torch.rand(D) * 0.05 + 1e-3))
        self.log_a = nn.Parameter(torch.randn(D, D) * 0.1 - 4.0)  # cool init -> starts sub-critical
        self.beta = beta
        self.lam_l1 = lam_l1

    def mu(self): return torch.exp(self.log_mu)
    def alpha(self): return F.softplus(self.log_a)
    def penalty(self):
        return self.lam_l1 * self.alpha().abs().sum() if self.lam_l1 > 0 else 0.0 * self.log_mu.sum()

    def branching_ratio(self):
        with torch.no_grad():
            return float(torch.linalg.eigvals(self.alpha() / self.beta).abs().max())

    def neg_loglik(self, prep):
        K, marks, G, Tw = prep
        mu, A = self.mu(), self.alpha()
        comp = Tw * mu.sum() + (A * G.unsqueeze(0)).sum() / self.beta
        lam = mu[marks] + (K * A[marks][:, marks]).sum(1)
        return comp - torch.log(lam.clamp(min=1e-12)).sum()


class DiagonalHawkes(nn.Module):
    """Self-excitation only: alpha is diagonal, so a stock's events excite only
    its own future events -- no cross-stock contagion (Hawkes 1971, per node)."""
    def __init__(self, D, beta=0.35):
        super().__init__()
        self.log_mu = nn.Parameter(torch.log(torch.rand(D) * 0.05 + 1e-3))
        self.log_d = nn.Parameter(torch.randn(D) * 0.1 - 4.0)  # cool init -> starts sub-critical
        self.beta = beta

    def mu(self): return torch.exp(self.log_mu)
    def alpha(self): return torch.diag(F.softplus(self.log_d))

    def branching_ratio(self):
        with torch.no_grad():
            return float((F.softplus(self.log_d) / self.beta).max())

    def neg_loglik(self, prep):
        K, marks, G, Tw = prep
        mu, A = self.mu(), self.alpha()
        comp = Tw * mu.sum() + (A * G.unsqueeze(0)).sum() / self.beta
        lam = mu[marks] + (K * A[marks][:, marks]).sum(1)
        return comp - torch.log(lam.clamp(min=1e-12)).sum()


def cojump_correlation(times, marks, D, n_days):
    """Symmetric co-event correlation matrix: bin events into days, take the
    per-stock daily event-count series, Pearson-correlate all pairs, clip
    negatives to 0, zero the diagonal. This is the correlation-based financial
    network of Mantegna (1999) applied to the event panel."""
    counts = np.zeros((int(n_days) + 1, D))
    for t, m in zip(times, marks):
        counts[int(t), int(m)] += 1.0
    C = np.corrcoef(counts.T)
    C = np.nan_to_num(C)
    np.fill_diagonal(C, 0.0)
    return np.clip(C, 0.0, None)


# ========================================================================
# 2c. POISSON NULL BASELINES (no history dependence at all -- the honest
# floor every temporal-point-process model must beat). Two variants:
#   P0 homogeneous  -- one global rate, split equally across all D marks
#                      (mu_d = N_train / (D * T_train) for every d)
#   P1 per-mark     -- an independent constant rate per stock
#                      (mu_d = n_d,train / T_train), i.e. an inhomogeneous-
#                      in-mark, homogeneous-in-time Poisson process.
# These are closed-form MLE (no gradient descent) and use the SAME windowed
# LL/event metric as the Hawkes models, so they are directly comparable on
# the leaderboard.
# ========================================================================
def fit_poisson_rates(times, marks, D, t_cut, per_mark=True):
    """MLE constant rate(s) on the training window [0, t_cut]."""
    tr = times < t_cut
    if per_mark:
        counts = np.bincount(marks[tr].astype(int), minlength=D).astype(float)
        mu = counts / max(float(t_cut), 1e-9)
    else:
        mu = np.full(D, int(tr.sum()) / (D * max(float(t_cut), 1e-9)))
    return mu


def window_ll_poisson(mu, times_all, marks_all, a, b):
    """Windowed held-out LL/event for a constant-rate Poisson process:
    LL = sum_{i in (a,b]} log mu_{m_i}  -  (b-a) * sum_d mu_d ,   then / n."""
    tt = np.asarray(times_all, dtype=float)
    mm = np.asarray(marks_all).astype(int)
    inw = (tt > a) & (tt <= b)
    n = int(inw.sum())
    loglam = np.log(np.clip(mu[mm[inw]], 1e-12, None)).sum()
    comp = (b - a) * float(mu.sum())
    return float((loglam - comp) / max(n, 1))


# ========================================================================
# 3. TRAIN / EVAL (exact windowed held-out log-likelihood)
# ========================================================================
def prep_single(times, marks, D, Tw, beta):
    tt = torch.tensor(times, dtype=torch.float32)
    mm = torch.tensor(marks, dtype=torch.long)
    n = len(times)
    dt = tt.unsqueeze(1) - tt.unsqueeze(0)
    mask = torch.tril(torch.ones(n, n), -1)
    K = torch.exp(-beta * dt.clamp(min=0)) * mask
    surv = 1 - torch.exp(-beta * (Tw - tt))
    G = torch.zeros(D).index_add(0, mm, surv)
    return K, mm, G, float(Tw)


def prep_mixture(times, marks, D, Tw, betas):
    tt = torch.tensor(times, dtype=torch.float32)
    mm = torch.tensor(marks, dtype=torch.long)
    n = len(times)
    dt = tt.unsqueeze(1) - tt.unsqueeze(0)
    mask = torch.tril(torch.ones(n, n), -1)
    Ks, Gs = [], []
    for b in betas:
        Ks.append(torch.exp(-b * dt.clamp(min=0)) * mask)
        surv = 1 - torch.exp(-b * (Tw - tt))
        Gs.append(torch.zeros(D).index_add(0, mm, surv))
    return Ks, Gs, mm, float(Tw)


def train(model, prep, epochs, lr, wd=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        loss = model.neg_loglik(prep)
        if hasattr(model, "penalty"):
            loss = loss + model.penalty()
        loss.backward()
        opt.step()
    return model


def window_ll_single(mu, A, beta, times_all, marks_all, D, a, b):
    tt = torch.tensor(times_all, dtype=torch.float32)
    mm = torch.tensor(marks_all, dtype=torch.long)
    n = len(times_all)
    dt = tt.unsqueeze(1) - tt.unsqueeze(0)
    mask = torch.tril(torch.ones(n, n), -1)
    K = torch.exp(-beta * dt.clamp(min=0)) * mask
    lam = mu[mm] + (K * A[mm][:, mm]).sum(1)
    inw = (tt > a) & (tt <= b)
    relu_a = torch.clamp(a - tt, min=0.0)
    term = torch.where(tt < b, torch.exp(-beta * relu_a) - torch.exp(-beta * (b - tt)), torch.zeros_like(tt))
    colmass = torch.zeros(D).index_add(0, mm, term)
    comp = mu.sum() * (b - a) + (A * colmass.unsqueeze(0)).sum() / beta
    with torch.no_grad():
        return float((torch.log(lam[inw].clamp(min=1e-12)).sum() - comp).item() / int(inw.sum()))


def window_ll_mixture(mu, A, w, betas, times_all, marks_all, D, a, b):
    tt = torch.tensor(times_all, dtype=torch.float32)
    mm = torch.tensor(marks_all, dtype=torch.long)
    n = len(times_all)
    dt = tt.unsqueeze(1) - tt.unsqueeze(0)
    mask = torch.tril(torch.ones(n, n), -1)
    lam = mu[mm].clone()
    comp = mu.sum() * (b - a)
    for m, bm in enumerate(betas):
        K = torch.exp(-bm * dt.clamp(min=0)) * mask
        lam = lam + w[m] * (K * A[mm][:, mm]).sum(1)
        relu_a = torch.clamp(a - tt, min=0.0)
        term = torch.where(tt < b, torch.exp(-bm * relu_a) - torch.exp(-bm * (b - tt)), torch.zeros_like(tt))
        colmass = torch.zeros(D).index_add(0, mm, term)
        comp = comp + (w[m] / bm) * (A * colmass.unsqueeze(0)).sum()
    inw = (tt > a) & (tt <= b)
    with torch.no_grad():
        return float((torch.log(lam[inw].clamp(min=1e-12)).sum() - comp).item() / int(inw.sum()))


def window_ll_conditional(model, times_all, marks_all, D, a, b):
    with torch.no_grad():
        mu, A, pi = model.mu(), model.alpha(), model.pi()
        tt = torch.tensor(times_all, dtype=torch.float32)
        mm = torch.tensor(marks_all, dtype=torch.long)
        n = len(times_all)
        dt = tt.unsqueeze(1) - tt.unsqueeze(0)
        mask = torch.tril(torch.ones(n, n), -1)
        lam = mu[mm].clone()
        comp = mu.sum() * (b - a)
        for m in range(model.M):
            bm = model.betas[m]
            K = torch.exp(-bm * dt.clamp(min=0)) * mask
            Bm = A * pi[:, :, m] * bm
            lam = lam + (K * Bm[mm][:, mm]).sum(1)
            relu_a = torch.clamp(a - tt, min=0.0)
            term = torch.where(tt < b, torch.exp(-bm * relu_a) - torch.exp(-bm * (b - tt)), torch.zeros_like(tt))
            colmass = torch.zeros(D).index_add(0, mm, term)
            comp = comp + ((A * pi[:, :, m]) * colmass.unsqueeze(0)).sum()
        inw = (tt > a) & (tt <= b)
        return float((torch.log(lam[inw].clamp(min=1e-12)).sum() - comp).item() / int(inw.sum()))


# ========================================================================
# 4. SWEEP RUNNER (checkpointed / resumable)
# ========================================================================
def load_log():
    if LOG_PATH.exists():
        return json.load(open(LOG_PATH))
    return []


def save_log(log):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(log, open(LOG_PATH, "w"), indent=1)


def run_key(tag, tau, dim, beta_name, seed):
    return f"{tag}|tau={tau}|dim={dim}|beta={beta_name}|seed={seed}"


def run_comparison_baselines(cfg, log, done):
    """Every comparison method, ONCE, at the canonical (tau, dim, beta) point.
    None of these is parameter-searched -- that is reserved for the proposed
    model (R3) in run_sweep below."""
    tau = cfg["CANON_TAU"]; dim = cfg["CANON_DIM"]; beta_name = cfg["CANON_BETA"]
    betas = cfg["BETA_SETS"].get(beta_name, tuple(cfg["BETA_SETS"].values())[0])
    times, marks, T, D, labels, sectors = get_events(tau, cfg["VOL_WIN"])
    t_cut = cfg["SPLIT"] * T
    tr = times < t_cut
    print(f"\n=== comparison baselines @ canonical (tau={tau}, dim={dim}, beta={beta_name}) "
          f"| D={D} events={len(times)} train={int(tr.sum())} test={len(times)-int(tr.sum())} ===")

    def _log(rec):
        log.append(rec); save_log(log); print("  ", rec)

    # -- Poisson null baselines (closed-form, no training) --
    for name, per_mark in [("P0-homogeneous-poisson", False), ("P1-permark-poisson", True)]:
        key = run_key(name, tau, dim, beta_name, 0)
        if key in done: continue
        mu_p = fit_poisson_rates(times, marks, D, t_cut, per_mark=per_mark)
        ll = window_ll_poisson(mu_p, times, marks, t_cut, T)
        _log(dict(key=key, model=name, tau=tau, dim=dim, beta_set=beta_name, seed=0,
                  held_out_ll=round(ll, 4), branching=0.0, n_events=len(times)))

    # -- R1 single-exponential Hawkes (embedding, fixed beta) --
    key = run_key("R1-single-exp", tau, dim, beta_name, 0)
    if key not in done:
        torch.manual_seed(0); beta_ref = 0.35
        prep = prep_single(times[tr], marks[tr], D, t_cut, beta_ref)
        m = train(SingleExpHawkes(D, dim, beta_ref), prep, cfg["EPOCHS"], cfg["LR"])
        with torch.no_grad():
            ll = window_ll_single(m.mu(), m.alpha(), beta_ref, times, marks, D, t_cut, T)
        _log(dict(key=key, model="R1-single-exp", tau=tau, dim=dim, beta_set=beta_name, seed=0,
                  held_out_ll=round(ll, 4), branching=round(m.branching_ratio(), 3), n_events=len(times)))

    # -- R2 shared mixture-of-exponentials Hawkes --
    key = run_key("R2-shared-mixture", tau, dim, beta_name, 0)
    if key not in done:
        torch.manual_seed(0)
        prep = prep_mixture(times[tr], marks[tr], D, t_cut, betas)
        m = train(MixtureHawkes(D, dim, betas), prep, cfg["EPOCHS"], cfg["LR"])
        with torch.no_grad():
            ll = window_ll_mixture(m.mu(), m.alpha(), m.weights(), betas, times, marks, D, t_cut, T)
        _log(dict(key=key, model="R2-shared-mixture", tau=tau, dim=dim, beta_set=beta_name, seed=0,
                  held_out_ll=round(ll, 4), branching=round(m.branching_ratio(), 3), n_events=len(times)))

    # -- classical Hawkes ladder (single-exp kernel, beta=0.35) --
    beta_ref = 0.35
    c_ep, c_lr = cfg.get("CLASSIC_EPOCHS", 500), cfg.get("CLASSIC_LR", 0.05)
    # Dense Hawkes (full free DxD alpha, light L2)
    key = run_key("Dense-Hawkes", tau, dim, beta_name, 0)
    if key not in done:
        torch.manual_seed(0)
        prep = prep_single(times[tr], marks[tr], D, t_cut, beta_ref)
        m = train(DenseHawkes(D, beta_ref, lam_l1=0.0), prep, c_ep, c_lr, wd=1e-4)
        with torch.no_grad():
            ll = window_ll_single(m.mu(), m.alpha(), beta_ref, times, marks, D, t_cut, T)
        _log(dict(key=key, model="Dense-Hawkes", tau=tau, dim=dim, beta_set=beta_name, seed=0,
                  held_out_ll=round(ll, 4), branching=round(m.branching_ratio(), 3), n_events=len(times)))

    # L1-Sparse Hawkes (same, with L1 penalty on alpha)
    key = run_key("L1-Sparse-Hawkes", tau, dim, beta_name, 0)
    if key not in done:
        torch.manual_seed(0)
        prep = prep_single(times[tr], marks[tr], D, t_cut, beta_ref)
        m = train(DenseHawkes(D, beta_ref, lam_l1=0.02), prep, c_ep, c_lr)
        with torch.no_grad():
            ll = window_ll_single(m.mu(), m.alpha(), beta_ref, times, marks, D, t_cut, T)
        _log(dict(key=key, model="L1-Sparse-Hawkes", tau=tau, dim=dim, beta_set=beta_name, seed=0,
                  held_out_ll=round(ll, 4), branching=round(m.branching_ratio(), 3), n_events=len(times)))

    # Diagonal Hawkes (self-excitation only)
    key = run_key("Diagonal-Hawkes", tau, dim, beta_name, 0)
    if key not in done:
        torch.manual_seed(0)
        prep = prep_single(times[tr], marks[tr], D, t_cut, beta_ref)
        m = train(DiagonalHawkes(D, beta_ref), prep, c_ep, c_lr)
        with torch.no_grad():
            ll = window_ll_single(m.mu(), m.alpha(), beta_ref, times, marks, D, t_cut, T)
        _log(dict(key=key, model="Diagonal-Hawkes", tau=tau, dim=dim, beta_set=beta_name, seed=0,
                  held_out_ll=round(ll, 4), branching=round(m.branching_ratio(), 3), n_events=len(times)))

    # Correlation network (non-parametric: alpha = c * co-event correlation)
    key = run_key("Correlation-network", tau, dim, beta_name, 0)
    if key not in done:
        mu_p = torch.tensor(fit_poisson_rates(times, marks, D, t_cut, per_mark=True), dtype=torch.float32)
        C = torch.tensor(cojump_correlation(times, marks, D, T), dtype=torch.float32)
        best_c, best = 0.0, -1e18
        for c in np.linspace(0, 2, 41):
            ll_tr = window_ll_single(mu_p, c * C, beta_ref, times, marks, D, 0.0, t_cut)
            if ll_tr > best: best, best_c = ll_tr, c
        ll = window_ll_single(mu_p, best_c * C, beta_ref, times, marks, D, t_cut, T)
        br = float(torch.linalg.eigvals(best_c * C / beta_ref).abs().max())
        _log(dict(key=key, model="Correlation-network", tau=tau, dim=dim, beta_set=beta_name, seed=0,
                  held_out_ll=round(ll, 4), branching=round(br, 3), n_events=len(times)))

    # Neural-kernel Hawkes (embedding alpha + flexible shared kernel = fine exp mixture)
    key = run_key("Neural-kernel-Hawkes", tau, dim, beta_name, 0)
    if key not in done:
        fine = tuple(float(x) for x in np.geomspace(0.1, 15.0, 10))
        torch.manual_seed(0)
        prep = prep_mixture(times[tr], marks[tr], D, t_cut, fine)
        m = train(MixtureHawkes(D, dim, fine), prep, cfg["EPOCHS"], cfg["LR"])
        with torch.no_grad():
            ll = window_ll_mixture(m.mu(), m.alpha(), m.weights(), fine, times, marks, D, t_cut, T)
        _log(dict(key=key, model="Neural-kernel-Hawkes", tau=tau, dim=dim, beta_set=beta_name, seed=0,
                  held_out_ll=round(ll, 4), branching=round(m.branching_ratio(), 3), n_events=len(times)))


def run_sweep(cfg):
    log = load_log()
    done = {rec["key"] for rec in log}
    t0 = time.time()

    # ---- comparison methods: run ONCE each, at the canonical config ----
    run_comparison_baselines(cfg, log, done)
    done = {rec["key"] for rec in log}

    # ---- PROPOSED model (R3): the ONLY parameter search -- tau x dim x beta x seed ----
    for tau in cfg["TAUS"]:
        times, marks, T, D, labels, sectors = get_events(tau, cfg["VOL_WIN"])
        t_cut = cfg["SPLIT"] * T
        tr = times < t_cut
        n_grid = len(cfg["DIMS"]) * len(cfg["BETA_SETS"]) * len(cfg["SEEDS"])
        print(f"\n=== [proposed grid] tau={tau}  D={D}  events={len(times)}  "
              f"train={int(tr.sum())} test={len(times) - int(tr.sum())}  ({n_grid} R3 runs) ===")
        for dim in cfg["DIMS"]:
            for beta_name, betas in cfg["BETA_SETS"].items():
                for seed in cfg["SEEDS"]:
                    key = run_key("R3-conditional", tau, dim, beta_name, seed)
                    if key in done:
                        continue
                    torch.manual_seed(seed)
                    prep = prep_mixture(times[tr], marks[tr], D, t_cut, betas)
                    m = train(ConditionalKernelHawkes(D, betas, dim, cfg["HID"]), prep, cfg["EPOCHS"], cfg["LR"])
                    ll = window_ll_conditional(m, times, marks, D, t_cut, T)
                    rec = dict(key=key, model="R3-conditional", tau=tau, dim=dim, beta_set=beta_name,
                               seed=seed, held_out_ll=round(ll, 4), branching=round(m.branching_ratio(), 3),
                               n_events=len(times))
                    log.append(rec); save_log(log)
                    print(f"  [{time.time()-t0:6.0f}s] {rec}")

    return log


# ========================================================================
# 5. COMPILE RESULTS -- CSV + leaderboard + plots, run automatically at the end
# ========================================================================
def compile_results(cfg=None):
    if cfg is None:
        cfg = DEFAULT_CONFIG
    log = load_log()
    if not log:
        print("No results in results/ablation_log.json yet -- run the sweep first.")
        return
    df = pd.DataFrame(log)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)

    r3 = df[df.model == "R3-conditional"]
    if r3.empty:
        print(f"Saved {len(df)} rows to results/ablation_results.csv (no R3 runs yet).")
        return

    agg = (r3.groupby(["tau", "dim", "beta_set"])["held_out_ll"]
              .agg(["mean", "std", "count"]).reset_index()
              .sort_values("mean", ascending=False))
    agg.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)
    print("\n=== TOP configs, proposed model (held-out LL/event, avg over seeds) ===")
    print(agg.head(10).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_beta = agg.iloc[0]["beta_set"]

    # (a) heatmap: tau x dim, at the best beta-bank
    piv = agg[agg.beta_set == best_beta].pivot(index="dim", columns="tau", values="mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("tau (event threshold, std devs)"); ax.set_ylabel("embedding dim")
    ax.set_title(f"Held-out LL/event  (beta-bank = {best_beta})", fontweight="bold")
    fig.colorbar(im, ax=ax, label="LL/event")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "heatmap_tau_dim.png", dpi=130); plt.close(fig)

    # (b) beta-bank x dim ablation, one line per (beta_set, tau)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for beta_name in sorted(agg.beta_set.unique()):
        for tau in sorted(agg.tau.unique()):
            sub = agg[(agg.beta_set == beta_name) & (agg.tau == tau)].sort_values("dim")
            if len(sub):
                ax.errorbar(sub.dim, sub["mean"], yerr=sub["std"].fillna(0.0), marker="o",
                             label=f"{beta_name}  (tau={tau})", capsize=3, lw=1.4)
    ax.set_xlabel("embedding dim"); ax.set_ylabel("held-out LL/event")
    ax.set_title("Ablation: beta-bank x dim x tau", fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "beta_dim_ablation.png", dpi=130); plt.close(fig)

    # (c) CANONICAL COMPARISON TABLE -- every method at one (tau,dim,beta) point,
    #     one LL/event each, sorted best-first (this is the paper's comparison table).
    canon_tau = cfg["CANON_TAU"]; canon_dim = cfg["CANON_DIM"]; canon_beta = cfg["CANON_BETA"]
    disp = {
        "R3-conditional": "Proposed conditional kernel",
        "R2-shared-mixture": "Shared mixture (R2)",
        "R1-single-exp": "Embedding single-exp (R1)",
        "P0-homogeneous-poisson": "Homogeneous Poisson",
        "P1-permark-poisson": "Per-mark Poisson",
        "Dense-Hawkes": "Dense Hawkes", "L1-Sparse-Hawkes": "L1-Sparse Hawkes",
        "Diagonal-Hawkes": "Diagonal Hawkes", "Correlation-network": "Correlation network",
        "Neural-kernel-Hawkes": "Neural-kernel Hawkes",
    }
    # method -> (paper title, citation key) for the methods_and_papers table
    papers = {
        "R3-conditional": ("(proposed) edge-conditioned Hawkes kernel", "--"),
        "R2-shared-mixture": ("(reference) shared mixture-of-exponentials Hawkes", "--"),
        "R1-single-exp": ("(reference) embedding single-exponential Hawkes", "--"),
        "P0-homogeneous-poisson": ("An Introduction to the Theory of Point Processes", "r_dvj"),
        "P1-permark-poisson": ("An Introduction to the Theory of Point Processes", "r_dvj"),
        "Dense-Hawkes": ("Maximum likelihood estimation of Hawkes' self-exciting point processes", "r_ozaki"),
        "L1-Sparse-Hawkes": ("Sparse and low-rank multivariate Hawkes processes", "bacry2020lowrank"),
        "Diagonal-Hawkes": ("Spectra of some self-exciting and mutually exciting point processes", "hawkes1971"),
        "Correlation-network": ("Hierarchical structure in financial markets", "mantegna1999"),
        "Neural-kernel-Hawkes": ("Hawkes Process with Flexible Triggering Kernels", "isik2023"),
    }
    baseline_order = ["R1-single-exp", "R2-shared-mixture", "Dense-Hawkes", "L1-Sparse-Hawkes",
                      "Diagonal-Hawkes", "Correlation-network", "Neural-kernel-Hawkes",
                      "P0-homogeneous-poisson", "P1-permark-poisson"]
    rows = []
    # proposed model at canonical config (mean over seeds)
    r3c = agg[(agg.tau == canon_tau) & (agg.dim == canon_dim) & (agg.beta_set == canon_beta)]
    if len(r3c):
        rows.append(("Proposed conditional kernel", float(r3c.iloc[0]["mean"])))
    # every baseline at canonical config
    for mdl in baseline_order:
        sub = df[(df.model == mdl) & (df.tau == canon_tau) &
                 (df.dim == canon_dim) & (df.beta_set == canon_beta)]
        if len(sub):
            rows.append((disp.get(mdl, mdl), float(sub.iloc[0]["held_out_ll"])))
    comp = (pd.DataFrame(rows, columns=["Model", "LL/event"])
              .sort_values("LL/event", ascending=False).reset_index(drop=True))
    comp.insert(0, "Rank", comp.index + 1)
    comp.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    print(f"\n=== COMPARISON TABLE @ canonical (tau={canon_tau}, dim={canon_dim}, beta={canon_beta}) ===")
    print(comp.to_string(index=False))

    # methods_and_papers.csv -- method name -> implemented paper + citation key
    mp_rows = [("R3-conditional", "Proposed conditional kernel")] + [(m, disp[m]) for m in baseline_order]
    mp = pd.DataFrame(
        [(dn, papers[code][0], papers[code][1]) for code, dn in mp_rows],
        columns=["Method", "Paper implemented", "Citation key"])
    mp.to_csv(RESULTS_DIR / "methods_and_papers.csv", index=False)
    print("\n=== METHODS AND THE PAPER EACH IMPLEMENTS ===")
    print(mp.to_string(index=False))

    # horizontal ranking bar
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(comp) + 1))
    colors = ["#C44E52" if m.startswith("Proposed") else "#4C72B0" for m in comp["Model"]]
    ax.barh(comp["Model"][::-1], comp["LL/event"][::-1], color=colors[::-1])
    ax.set_xlabel("held-out LL/event  (higher = better)")
    ax.set_title(f"Method comparison @ tau={canon_tau}, dim={canon_dim}, beta={canon_beta}", fontweight="bold")
    for i, v in enumerate(comp["LL/event"][::-1]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "comparison_ranking.png", dpi=130); plt.close(fig)

    print(f"\nSaved to {RESULTS_DIR}/:")
    print("  ablation_results.csv, leaderboard.csv, comparison_table.csv, methods_and_papers.csv")
    print("  heatmap_tau_dim.png, beta_dim_ablation.png, comparison_ranking.png")


# ========================================================================
# 6. ENTRYPOINT
# ========================================================================
def main():
    ap = argparse.ArgumentParser(description="Ablation runner for the edge-conditioned Hawkes kernel.")
    ap.add_argument("--quick", action="store_true", help="tiny grid + few epochs, for a smoke test")
    ap.add_argument("--compile-only", action="store_true", help="skip the sweep, just re-summarize results/ablation_log.json")
    ap.add_argument("--taus", type=float, nargs="+", default=None)
    ap.add_argument("--dims", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args, _unknown = ap.parse_known_args()   # known_args so it also works pasted into a Colab cell

    cfg = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
           for k, v in DEFAULT_CONFIG.items()}

    if args.quick:
        cfg.update(TAUS=[2.5], DIMS=[4, 8],
                   BETA_SETS={"fast": (0.35, 1.0), "default": (0.35, 1.0, 3.0, 8.0)},
                   SEEDS=[0], EPOCHS=80, CLASSIC_EPOCHS=300,
                   CANON_TAU=2.5, CANON_DIM=8, CANON_BETA="default")
    if args.taus: cfg["TAUS"] = args.taus
    if args.dims: cfg["DIMS"] = args.dims
    if args.seeds: cfg["SEEDS"] = args.seeds
    if args.epochs: cfg["EPOCHS"] = args.epochs

    print("Config:", {k: v for k, v in cfg.items() if k not in ("BETA_SETS",)})
    print("Beta banks:", cfg["BETA_SETS"])
    if not args.compile_only:
        run_sweep(cfg)
    compile_results(cfg)


if __name__ == "__main__":
    main()
