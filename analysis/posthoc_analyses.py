"""Post hoc analyses of label ambiguity, multiplicity, predictive values and reference discordance (28 Oct 2026).

Every value is computed from the per-scan files of the original pipeline (no new model inference):
  1. cross-subtype call matrix for v3b (recreates out/cq500_call_matrix_colneg.json used by Figure 5 and Supplementary Table)
  2. label ambiguity versus hemorrhage extent: sensitivity of v2 at 1 mL by reader vote split, overall and restricted to
     single-subtype majority positives; model output in unanimous-negative versus one-vote negative scans
  3. Holm adjustment of the five within-ICH paired DeLong tests (v3b class volume versus v2 total volume)
  4. counterfactual PPV (B2 operating point at B1 prevalence) with bootstrap CI, and decomposition of the B1-B2 PPV gap
  5. integer-count examples for the attainable Wilson lower bound
  6. lower bounds on scan-level subtype discordance between BHX boxes and the CQ500 reader majority (from the numbers of positive scans)
  7. the largest BHSD test hemorrhage missed entirely by v2
Outputs: out/posthoc.json, out/tables/cq500_ambiguity_by_extent.csv, out/tables/bhx_cq500_subtype_discordance.csv, out/cq500_call_matrix_colneg.json
"""
import json
import numpy as np, pandas as pd
from scipy import stats
from common import *

OUTT = OUT / 'tables'; OUTT.mkdir(exist_ok=True, parents=True)
rng = np.random.default_rng(20261028)
B = 10000
R = {}

m = load_cq500()
SUBS = ['EDH', 'IPH', 'IVH', 'SAH', 'SDH']


def newcombe(k1, n1, k2, n2):
    """Newcombe hybrid score (method 10) 95% CI for p1 - p2."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


# ----------------------------------------------------------------------------- 1. call matrix
K = np.zeros((5, 5), int); Nn = np.zeros((5, 5), int)
for i, r in enumerate(SUBS):
    rp = (m[f'ref_{r}'] == 1).values
    for j, c in enumerate(SUBS):
        sel = rp if i == j else rp & (m[f'ref_{c}'] == 0).values
        Nn[i, j] = sel.sum(); K[i, j] = ((m[f'{c}_ml'] >= 1).values & sel).sum()
Mx = K / Nn
cm = dict(rows=SUBS, k=K.tolist(), n=Nn.tolist(), M=np.round(Mx, 6).tolist(),
          note='diagonal: sensitivity among scans with the reader-majority subtype; off-diagonal: among scans with the row subtype and without the column subtype')
json.dump(cm, open(OUT / 'cq500_call_matrix_colneg.json', 'w'), indent=1)
offd = [Mx[i, 3] for i in range(5) if i != 3]
R['sah_overcall_range'] = [float(min(offd)), float(max(offd))]

# ----------------------------------------------------------------------------- 2. ambiguity versus extent
votes = m[['R1:ICH', 'R2:ICH', 'R3:ICH']].sum(axis=1).astype(int).values
v = m.pred_volume_ml.values
call = v >= 1
nsub = m[[f'ref_{s}' for s in SUBS]].sum(axis=1).values
pos = m.ref_ICH.values == 1
rows = []
for k in (2, 3):
    for lab, sel_n in [('all', np.ones(len(m), bool)), ('no subtype label', nsub == 0), ('single subtype', nsub == 1), ('two or more subtypes', nsub >= 2)]:
        s = pos & (votes == k) & sel_n
        n = int(s.sum()); d = int(call[s].sum())
        if n == 0:
            rows.append(dict(votes=f'{k} of 3', stratum=lab, n=0, detected=0, sens=np.nan, lo=np.nan, hi=np.nan, median_volume_ml=np.nan)); continue
        lo, hi = wilson(d, n)
        rows.append(dict(votes=f'{k} of 3', stratum=lab, n=n, detected=d, sens=d / n, lo=lo, hi=hi, median_volume_ml=float(np.median(v[s]))))
amb = pd.DataFrame(rows); amb.to_csv(OUTT / 'cq500_ambiguity_by_extent.csv', index=False)
R['ambiguity_by_extent'] = rows
s2 = pos & (votes == 2) & (nsub == 1); s3 = pos & (votes == 3) & (nsub == 1)
R['single_subtype_sens_diff'] = list(newcombe(int(call[s3].sum()), int(s3.sum()), int(call[s2].sum()), int(s2.sum())))
R['single_subtype_fisher_p'] = float(stats.fisher_exact([[call[s3].sum(), (~call[s3]).sum()], [call[s2].sum(), (~call[s2]).sum()]])[1])
R['split_positives_max_subtypes'] = int(nsub[pos & (votes == 2)].max())
R['unanimous_positives_multi'] = int((nsub[pos & (votes == 3)] >= 2).sum())
# negatives: unanimous negative (0 votes) versus one dissenting reader (1 vote)
neg0 = votes == 0; neg1 = votes == 1
R['neg_call'] = dict(k0=[int(call[neg0].sum()), int(neg0.sum())], k1=[int(call[neg1].sum()), int(neg1.sum())],
                     k0_ci=list(wilson(int(call[neg0].sum()), int(neg0.sum()))), k1_ci=list(wilson(int(call[neg1].sum()), int(neg1.sum()))),
                     diff=list(newcombe(int(call[neg1].sum()), int(neg1.sum()), int(call[neg0].sum()), int(neg0.sum()))),
                     median=[float(np.median(v[neg0])), float(np.median(v[neg1]))],
                     mannwhitney_p=float(stats.mannwhitneyu(v[neg1], v[neg0]).pvalue))

# ----------------------------------------------------------------------------- 3. Holm adjustment (within-ICH DeLong family)
NA = json.load(open(OUT / 'new_analyses_cq500.json'))
dec = {r['sub']: r for r in NA['subtype_decomposition']}
praw = {s: float(dec[s]['delong_within_ich_v3b_minus_v2'][3]) for s in SUBS}
order = sorted(SUBS, key=lambda s: praw[s])
holm, running = {}, 0.0
for i, s in enumerate(order):
    running = max(running, min(1.0, (len(SUBS) - i) * praw[s]))
    holm[s] = running
R['within_ich_delong'] = {s: dict(p_raw=praw[s], p_holm=holm[s], diff=dec[s]['delong_within_ich_v3b_minus_v2'][:3]) for s in SUBS}

# ----------------------------------------------------------------------------- 4. counterfactual PPV
b1 = (m.Category == 'B1').values; b2 = (m.Category == 'B2').values
y = pos


def opstats(sel):
    yy, cc = y[sel], call[sel]
    se = (cc & yy).sum() / yy.sum(); sp = (~cc & ~yy).sum() / (~yy).sum(); prev = yy.mean()
    ppv = (cc & yy).sum() / cc.sum()
    return se, sp, prev, ppv


def ppv_at(se, sp, pi):
    return se * pi / (se * pi + (1 - sp) * (1 - pi))


se1, sp1, pi1, ppv1 = opstats(b1); se2, sp2, pi2, ppv2 = opstats(b2)
cf = ppv_at(se2, sp2, pi1)
i1 = np.where(b1)[0]; i2 = np.where(b2)[0]
cf_b, share_b = [], []
for _ in range(B):
    r1 = rng.choice(i1, len(i1)); r2 = rng.choice(i2, len(i2))
    s1 = np.zeros(len(m), bool)
    a1 = opstats_idx = None
    yy1, cc1 = y[r1], call[r1]; yy2, cc2 = y[r2], call[r2]
    if yy1.sum() == 0 or yy2.sum() == 0 or cc1.sum() == 0:
        continue
    se2b = (cc2 & yy2).sum() / yy2.sum(); sp2b = (~cc2 & ~yy2).sum() / (~yy2).sum(); pi1b = yy1.mean()
    ppv1b = (cc1 & yy1).sum() / cc1.sum(); ppv2b = (cc2 & yy2).sum() / cc2.sum()
    cfb = ppv_at(se2b, sp2b, pi1b)
    cf_b.append(cfb)
    tot = ppv2b - ppv1b
    if tot > 0:
        share_b.append((ppv2b - cfb) / tot)
R['ppv_counterfactual'] = dict(se_B1=se1, sp_B1=sp1, prev_B1=pi1, ppv_B1=ppv1, se_B2=se2, sp_B2=sp2, prev_B2=pi2, ppv_B2=ppv2,
                               ppv_cf=cf, ppv_cf_ci=list(np.percentile(cf_b, [2.5, 97.5])),
                               gap_total=ppv2 - ppv1, gap_prevalence=ppv2 - cf, gap_operating_point=cf - ppv1,
                               share_prevalence=(ppv2 - cf) / (ppv2 - ppv1), share_prevalence_ci=list(np.percentile(share_b, [2.5, 97.5])),
                               B=B, seed=20261028)

# ----------------------------------------------------------------------------- 5. integer examples for attainable bounds
ex = {}
for k_, n_ in [(35, 35), (34, 34), (132, 139), (133, 140), (114, 120), (152, 160), (13, 13), (28, 28), (19, 19)]:
    ex[f'{k_}/{n_}'] = float(wilson(k_, n_)[0])
# smallest n with an exact 95% observed sensitivity (k = 0.95 n, n multiple of 20) whose bound reaches 0.90
n95 = next(n for n in range(20, 1000, 20) if wilson(0.95 * n, n)[0] >= 0.90)
R['bound_examples'] = ex; R['n_exact95_for_090'] = n95

# ----------------------------------------------------------------------------- 6. BHX versus CQ500 majority (lower bounds)
CQ = json.load(open(E / 'cq500_external/cq500_results.json'))
bx = CQ['bhx_box_hit']
pairs = [('Subarachnoid', 'SAH'), ('Subdural', 'SDH'), ('Intraventricular', 'IVH'), ('Intraparenchymal', 'IPH'), ('Epidural', 'EDH'), ('Chronic', 'ChronicBleed')]
disc = []
for bl, cl in pairs:
    nb = int(bx[bl]['scans']); nc = int(m[f'ref_{cl}'].sum())
    disc.append(dict(bhx_label=bl, cq500_label=cl, bhx_scans_with_boxes=nb, cq500_majority_positive_scans=nc,
                     min_scans_with_bhx_box_without_majority_label=max(0, nb - nc)))
pd.DataFrame(disc).to_csv(OUTT / 'bhx_cq500_subtype_discordance.csv', index=False)
R['bhx_subtype_discordance'] = disc
R['bhx_scans_matched'] = int(CQ['bhx_scans_matched'])
R['bhx_ich_discordance_min'] = int(CQ['bhx_scans_matched']) - int(m.ref_ICH.sum())

# ----------------------------------------------------------------------------- 7. largest missed BHSD hemorrhage
nm = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/ablation_no_postmask.csv').set_index('case_id')
v3b = pd.read_csv(E / 'v3_subtypes/v3b/v3b_bhsd_test_metrics.csv').set_index('case_id')
zero = nm[nm.dice_nopostmask == 0]
big = zero.ref_ml.idxmax()
R['bhsd_largest_miss'] = dict(case=big, ref_ml=float(zero.loc[big, 'ref_ml']), v2_pred_ml=float(zero.loc[big, 'pred_ml']),
                              subtype='SDH' if v3b.loc[big, 'SDH_ref_ml'] > 0 else 'other', v3b_any_dice=float(v3b.loc[big, 'ANY_dice']),
                              n_zero_v2=int(len(zero)), n_zero_v2_lt10=int((zero.ref_ml < 10).sum()))

save_json(R, OUT / 'posthoc.json')
print(json.dumps({k: R[k] for k in ['sah_overcall_range', 'single_subtype_sens_diff', 'single_subtype_fisher_p', 'neg_call', 'within_ich_delong',
                                    'ppv_counterfactual', 'bound_examples', 'n_exact95_for_090', 'bhx_subtype_discordance', 'bhsd_largest_miss']}, default=float, indent=1))
print(amb.to_string())
