"""Shared utilities for the NeuroCare CT re-analysis (all computations from per-scan files in the project folder)."""
import json, ast, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

# project root = parent of this folder (override with NEUROCARE_WORK)
W = Path(os.environ.get('NEUROCARE_WORK', Path(__file__).resolve().parents[1]))
D = W / 'data'
E = D / 'evidence'
T = D / 'tables'
OUT = W / 'out'
OUT.mkdir(exist_ok=True, parents=True)

Z = 1.959963984540054


def wilson(k, n, z=Z):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def icc_a1(p, y):
    """Two-way random/mixed, absolute agreement, single measure ICC(A,1) (McGraw & Wong)."""
    R = np.c_[np.asarray(p, float), np.asarray(y, float)]
    n, k = R.shape
    g = R.mean()
    msr = k * ((R.mean(1) - g) ** 2).sum() / (n - 1)
    msc = n * ((R.mean(0) - g) ** 2).sum() / (k - 1)
    e = R - R.mean(1)[:, None] - R.mean(0) + g
    mse = (e ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def icc_a1_multi(R):
    """ICC(A,1) for an n x k matrix (k raters)."""
    R = np.asarray(R, float)
    n, k = R.shape
    g = R.mean()
    msr = k * ((R.mean(1) - g) ** 2).sum() / (n - 1)
    msc = n * ((R.mean(0) - g) ** 2).sum() / (k - 1)
    e = R - R.mean(1)[:, None] - R.mean(0) + g
    mse = (e ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def agree(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    d = p - y
    rel = np.abs(d[y > 0]) / y[y > 0]
    sd = d.std(ddof=1)
    return dict(MAE=np.abs(d).mean(), medAE=np.median(np.abs(d)), RMSE=np.sqrt((d * d).mean()),
                rel_med_pct=100 * np.median(rel), ICC=icc_a1(p, y), bias=d.mean(),
                LoA_lo=d.mean() - 1.96 * sd, LoA_hi=d.mean() + 1.96 * sd)


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    ok = ~np.isnan(s)
    return roc_auc_score(y[ok], s[ok])


def delong_var(y, s):
    """Fast DeLong: returns AUC and variance (Sun & Xu 2014)."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    pos = s[y == 1]; neg = s[y == 0]
    m, n = len(pos), len(neg)
    allv = np.r_[pos, neg]
    r_all = stats.rankdata(allv)
    r_pos = stats.rankdata(pos)
    r_neg = stats.rankdata(neg)
    v10 = (r_all[:m] - r_pos) / n
    v01 = 1 - (r_all[m:] - r_neg) / m
    a = v10.mean()
    return a, v10.var(ddof=1) / m + v01.var(ddof=1) / n, v10, v01


def delong_ci(y, s, alpha=0.05):
    a, v, _, _ = delong_var(y, s)
    se = np.sqrt(v)
    # logit transform for better coverage near bounds
    z = stats.norm.ppf(1 - alpha / 2)
    return a, max(0, a - z * se), min(1, a + z * se)


def delong_paired(y, s1, s2):
    y = np.asarray(y).astype(int)
    a1, _, v10a, v01a = delong_var(y, s1)
    a2, _, v10b, v01b = delong_var(y, s2)
    m, n = (y == 1).sum(), (y == 0).sum()
    S10 = np.cov(np.vstack([v10a, v10b]))
    S01 = np.cov(np.vstack([v01a, v01b]))
    S = S10 / m + S01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    zst = (a1 - a2) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(zst))
    se = np.sqrt(var)
    return a1 - a2, (a1 - a2) - Z * se, (a1 - a2) + Z * se, p


def fleiss_kappa(M):
    """M: n_subjects x n_categories counts."""
    M = np.asarray(M, float)
    n, k = M.shape
    r = M.sum(1)[0]
    p = M.sum(0) / (n * r)
    P = (np.square(M).sum(1) - r) / (r * (r - 1))
    Pbar = P.mean(); Pe = np.square(p).sum()
    return (Pbar - Pe) / (1 - Pe) if Pe < 1 else np.nan


def cohen_kappa(a, b):
    a = np.asarray(a).astype(int); b = np.asarray(b).astype(int)
    po = (a == b).mean()
    pa = a.mean(); pb = b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def fmt(x, d=3):
    return f"{x:.{d}f}"


def ci_str(est, lo, hi, d=3):
    return f"{est:.{d}f} ({lo:.{d}f}–{hi:.{d}f})"


def load_cq500():
    p = pd.read_csv(E / 'cq500_external/cq500_per_scan.csv')
    v = pd.read_csv(E / 'v3_subtypes/cq500_v3b_per_scan.csv')
    r = pd.read_csv(E / 'cq500_external/reads.csv')
    v['dz'] = v.spacing.apply(lambda s: ast.literal_eval(s)[2])
    v['dxy'] = v.spacing.apply(lambda s: ast.literal_eval(s)[0])
    m = p.merge(v[['name', 'EDH_ml', 'IPH_ml', 'IVH_ml', 'SAH_ml', 'SDH_ml', 'ANY_ml', 'dz', 'dxy', 'sec']], on='name', how='left')
    m = m.merge(r, on='name', how='left')
    assert len(m) == 489 and m.ANY_ml.notna().all()
    return m


LAB = ['ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH', 'BleedLocation-Left', 'BleedLocation-Right', 'ChronicBleed',
       'Fracture', 'CalvarialFracture', 'OtherFracture', 'MassEffect', 'MidlineShift']
SUB = ['EDH', 'IPH', 'IVH', 'SAH', 'SDH']


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)
    Path(path).write_text(json.dumps(obj, indent=1, default=conv))
