"""Post hoc: robustness of volumetric agreement statistics. ICC(A,1) and Lin's concordance correlation coefficient on raw volumes
and on log-transformed volumes, log10(1 + V) with V in mL, and the range of ICC(A,1) over all leave-one-scan-out recomputations
(jackknife), with the scan whose removal changes it most. Seg-CQ500 (43 scans: v2 without post-mask, BLAST-CT) and the BHSD test
set (96 scans: v1, v2 prespecified). Bootstrap intervals: 10,000 resamples (seed 20261029).
Output: out/volume_robust.json, out/tables/volume_robust.csv"""
import numpy as np
import pandas as pd

from common import E, OUT, T, icc_a1, save_json

rng = np.random.default_rng(20261029)


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    return 2 * sxy / (x.var(ddof=1) + y.var(ddof=1) + (x.mean() - y.mean()) ** 2)


def boot(f, n, B=10000):
    v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        r = f(i)
        if np.isfinite(r):
            v.append(r)
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def summarise(name, ids, p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    n = len(y)
    lp, ly = np.log10(1 + p), np.log10(1 + y)
    jk = np.array([icc_a1(np.delete(p, i), np.delete(y, i)) for i in range(n)])
    jl = np.array([icc_a1(np.delete(lp, i), np.delete(ly, i)) for i in range(n)])
    full = icc_a1(p, y)
    k = int(np.argmax(np.abs(jk - full)))
    return dict(set=name, n=n, icc_raw=float(full), icc_raw_ci=boot(lambda i: icc_a1(p[i], y[i]), n),
                icc_log=float(icc_a1(lp, ly)), icc_log_ci=boot(lambda i: icc_a1(lp[i], ly[i]), n),
                ccc_raw=float(ccc(p, y)), ccc_log=float(ccc(lp, ly)),
                jackknife_icc_raw=[float(jk.min()), float(jk.max())], jackknife_icc_log=[float(jl.min()), float(jl.max())],
                most_influential=ids[k], icc_raw_without_it=float(jk[k]), icc_log_without_it=float(jl[k]))


res = []
# Seg-CQ500
segnm = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/segcq500_no_postmask.csv').set_index('case_id')
ks = sorted(segnm.index)
y = segnm.loc[ks, 'ref_ml'].values
seg4 = pd.read_csv(T / 'SegCQ500_four_method_comparison.csv')
g = seg4[seg4.method == 'blast_model1'].set_index('case_id').loc[ks]
assert np.allclose(g.reference_volume_ml.values, y, rtol=1e-6, atol=1e-6)
res.append(summarise('Seg-CQ500, v2 (no post-mask)', ks, segnm.loc[ks, 'pred_ml'].values, y))
res.append(summarise('Seg-CQ500, BLAST-CT', ks, g.prediction_volume_ml.fillna(g.pred_volume_ml).values, y))
# BHSD test set
v1 = pd.read_csv(E / 'neurocare_neuroimaging_advanced_20261021/per_scan_quantitative_metrics.csv').set_index('case_id')
v2 = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/v2_metrics.csv')
v2b = v2[v2.dataset == 'BHSD_test'].set_index('case_id')
kb = sorted(v1.index)
yb = v1.loc[kb, 'reference_volume_ml'].values
res.append(summarise('BHSD test, v1', kb, v1.loc[kb, 'prediction_volume_ml'].values, yb))
res.append(summarise('BHSD test, v2 prespecified', kb, v2b.loc[kb, 'pred_ml'].values, yb))
pd.DataFrame(res).to_csv(OUT / 'tables' / 'volume_robust.csv', index=False)
save_json(res, OUT / 'volume_robust.json')
for r in res:
    print(f"{r['set']}: n {r['n']}; ICC raw {r['icc_raw']:.3f}, log {r['icc_log']:.3f} ({r['icc_log_ci'][0]:.3f}-{r['icc_log_ci'][1]:.3f}); "
          f"CCC raw {r['ccc_raw']:.3f}, log {r['ccc_log']:.3f}; jackknife raw {r['jackknife_icc_raw'][0]:.3f}-{r['jackknife_icc_raw'][1]:.3f}, "
          f"log {r['jackknife_icc_log'][0]:.3f}-{r['jackknife_icc_log'][1]:.3f}; most influential {r['most_influential']} -> {r['icc_raw_without_it']:.3f}")
