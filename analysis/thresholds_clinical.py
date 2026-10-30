"""Post hoc: agreement of compartment-specific volumes with clinical decision thresholds (BHSD test, internal).
Writes out/thresholds_clinical.json. Uses locked per-scan outputs only."""
import json
import numpy as np, pandas as pd
from common import *

v3b = pd.read_csv(E / 'v3_subtypes/v3b/v3b_bhsd_test_metrics.csv').set_index('case_id')
nm = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/ablation_no_postmask.csv').set_index('case_id')
k = sorted(v3b.index)
qc = v3b.loc[k, 'qc_flag'].astype(bool).values
R = {}

def kappa2(a, b):
    a = np.asarray(a, int); b = np.asarray(b, int)
    cats = sorted(set(a) | set(b)); n = len(a)
    po = (a == b).mean()
    pe = sum(((a == c).mean() * (b == c).mean()) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan

def cls(ref, pred, thr):
    r = ref >= thr; p = pred >= thr
    return dict(ref_pos=int(r.sum()), n=int(len(r)), tp=int((r & p).sum()), fn=int((r & ~p).sum()), tn=int((~r & ~p).sum()), fp=int((~r & p).sum()),
                kappa=float(kappa2(r, p)), misclassified=int((r != p).sum()))

def bins(ref, pred, edges):
    rb = np.digitize(ref, edges); pb = np.digitize(pred, edges)
    return dict(agree=int((rb == pb).sum()), n=int(len(rb)), kappa=float(kappa2(rb, pb)),
                ref_counts=[int((rb == i).sum()) for i in range(len(edges) + 1)])

for gate in ['all', 'gated']:
    sel = np.ones(len(k), bool) if gate == 'all' else ~qc
    out = {}
    # total volume, v2 no post-mask
    ref_t = nm.loc[k, 'ref_ml'].values[sel]; pr_t = nm.loc[k, 'pred_ml'].values[sel]
    out['total_v2_30'] = cls(ref_t, pr_t, 30)
    out['total_v2_bins_30_80'] = bins(ref_t, pr_t, [30, 80])
    # IPH-specific, v3b class volume
    ri = v3b.loc[k, 'IPH_ref_ml'].values[sel]; pi = v3b.loc[k, 'IPH_pred_ml'].values[sel]
    a = agree(pi, ri)
    out['iph_v3b_icc'] = icc_a1(pi, ri)
    out['iph_v3b_bias'] = [float(a['bias']), float(a['loa_lo']), float(a['loa_hi'])] if 'loa_lo' in a else a
    out['iph_v3b_30'] = cls(ri, pi, 30)
    out['iph_v3b_bins_30_80'] = bins(ri, pi, [30, 80])
    # EDH-specific
    re_ = v3b.loc[k, 'EDH_ref_ml'].values[sel]; pe_ = v3b.loc[k, 'EDH_pred_ml'].values[sel]
    out['edh_v3b_30'] = cls(re_, pe_, 30)
    out['edh_ref_max'] = float(re_.max())
    out['edh_pred_ge30_list'] = [float(x) for x in pe_[pe_ >= 30]]
    out['edh_ref_pos'] = int((re_ > 0).sum())
    # SDH volume (no thickness reference): count only
    rs = v3b.loc[k, 'SDH_ref_ml'].values[sel]
    out['sdh_ref_pos'] = int((rs > 0).sum())
    R[gate] = out
json.dump(R, open(OUT / 'thresholds_clinical.json', 'w'), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
