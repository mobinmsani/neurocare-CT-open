"""Post hoc: same-scan comparisons for the proxy (452 scans), MR-BHSD conventions, gated v3b class ICCs, class-fraction AUCs."""
import json
import numpy as np, pandas as pd
from common import *
rng = np.random.default_rng(20261033)
R = {}
# (1) same-scan volume AUCs for midline shift and mass effect on proxy-computable scans
m = load_cq500(); mp = m[m.csf_offset_maxabs_mm.notna()]
for lab in ['MidlineShift', 'MassEffect']:
    y = mp['ref_' + lab].values; s = mp.pred_volume_ml.values; px = mp.csf_offset_maxabs_mm.values
    bs = []; n = len(y)
    for _ in range(10000):
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n:
            bs.append(auc(y[i], s[i]))
    R[lab] = dict(n=int(n), pos=int(y.sum()), auc_vol_same=float(auc(y, s)), ci=[float(v) for v in np.percentile(bs, [2.5, 97.5])],
                  auc_proxy=float(auc(y, px)), delong_proxy_minus_vol=[float(v) for v in delong_paired(y, px, s)])
# (2) MR-BHSD empty-mask conventions
mr = pd.read_csv(T / 'MRBHSD_full43_compact_model_rater_metrics.csv')
rp = mr.pivot(index='case_id', columns='reference_rater', values='reference_positive').astype(bool)
npos = rp.sum(axis=1)
allneg = rp.index[npos == 0].tolist()
d = mr.groupby('case_id').dice.mean()
rr = pd.read_csv(T / 'TableS4_Reference_reference_agreement.csv')
tot = (rr.n_scan_pairs * rr.mean_dice).sum(); be = rr.both_empty_pairs.sum(); npairs = rr.n_scan_pairs.sum()
pair_ex = ((rr.n_scan_pairs * rr.mean_dice - rr.both_empty_pairs) / (rr.n_scan_pairs - rr.both_empty_pairs)).tolist()
R['mrbhsd'] = dict(scans_by_n_positive_raters={str(int(k_)): int(v) for k_, v in npos.value_counts().sort_index().items()}, empty_per_rater=(~rp).sum().astype(int).to_dict(),
                   all_empty_scans=allneg, v1_dice_all=float(d.mean()), v1_dice_excl_all_empty=float(d.drop(allneg).mean()),
                   rr_dice_all=float(tot / npairs), rr_dice_excl_joint_empty=float((tot - be) / (npairs - be)), rr_pairs_excl=[float(v) for v in pair_ex], joint_empty_pairs=int(be))
# (3) gated class ICCs for v3b on BHSD
v3b = pd.read_csv(E / 'v3_subtypes/v3b/v3b_bhsd_test_metrics.csv').set_index('case_id')
g = v3b[~v3b.qc_flag.astype(bool)]
R['v3b_class_icc_gated'] = {}
for c in ['EDH', 'IPH', 'IVH', 'SAH', 'SDH']:
    p = g[c + '_pred_ml'].values; yv = g[c + '_ref_ml'].values
    bs = []
    for _ in range(10000):
        i = rng.integers(0, len(p), len(p)); bs.append(icc_a1(p[i], yv[i]))
    R['v3b_class_icc_gated'][c] = [float(icc_a1(p, yv))] + [float(v) for v in np.nanpercentile(bs, [2.5, 97.5])]
json.dump(R, open(OUT / 'extra_checks3.json', 'w'), indent=1)
print(json.dumps(R, indent=1))
