"""Post hoc: bootstrap CI for v2 AUC on CQ500 excluding the 43 Seg-CQ500 scans (seed 20261022, 2000 resamples, as locked CQ500 CIs)."""
import json
import numpy as np, pandas as pd
from common import *
m = load_cq500()
seg = set(pd.read_csv(T / 'SegCQ500_four_method_comparison.csv').case_id.unique())
ex = m[~m.name.isin(seg)]
y = ex.ref_ICH.values; s = ex.pred_volume_ml.values
rng = np.random.default_rng(20261022); bs = []
for _ in range(2000):
    i = rng.integers(0, len(y), len(y))
    if y[i].min() != y[i].max():
        bs.append(auc(y[i], s[i]))
lo, hi = np.percentile(bs, [2.5, 97.5])
d = delong_ci(y, s)
R = dict(cq_v2_excl_seg=dict(n=int(len(y)), pos=int(y.sum()), auc=float(auc(y, s)), boot_ci=[float(lo), float(hi)], delong_ci=[float(d[1]), float(d[2])]))
json.dump(R, open(OUT / 'extra_checks2.json', 'w'), indent=1)
print(R)
