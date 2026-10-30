"""Replication with a strong, independently developed public model on CQ500 (post hoc; plan locked 28 Oct 2026, 22:48 UTC).

Model: ianpan/ct-head-hemorrhage-detection (MaxViT-Tiny slice encoder with a bidirectional GRU; trained on RSNA 2019 data;
CQ500 not used). Score: series-level fixed-blend probabilities ('any' and five subtypes), from per-scan outputs in
data/evidence/public_model_cq500/public_model_per_scan.csv (written by run/collect_public_model.py).
Endpoints E1-E8 follow data/evidence/public_model_cq500/PROTOCOL_LOCK_public_model_CQ500.json; definitions mirror
posthoc_analyses.py and new_analyses.py (reader majority, vote strata, subtype-count strata, column-negative call matrix).
Outputs: out/public_model_cq500.json, out/tables/public_model_*.csv
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from common import *

OUTT = OUT / 'tables'; OUTT.mkdir(exist_ok=True, parents=True)
PM = E / 'public_model_cq500'
rng = np.random.default_rng(20261029)
NB = int(os.environ.get("PM_NB", 10000))
SUBS = ['EDH', 'IPH', 'IVH', 'SAH', 'SDH']
CLS = dict(EDH='epidural', IPH='intraparenchymal', IVH='intraventricular', SAH='subarachnoid', SDH='subdural', ICH='any')
THR = 0.5

m = load_cq500()                                   # 489 analyzed scans with reader labels and v2 outputs
pm = pd.read_csv(PM / 'public_model_per_scan.csv')
m = m.merge(pm, left_on='name', right_on='scan', how='inner')
R = dict(n_scans=int(len(m)), n_expected=489, threshold=THR)
y = (m.ref_ICH == 1).values
p = m['p_any'].values
votes = m[['R1:ICH', 'R2:ICH', 'R3:ICH']].sum(axis=1).astype(int).values
nsub = m[[f'ref_{s}' for s in SUBS]].sum(axis=1).values
b1 = (m.Category == 'B1').values; b2 = (m.Category == 'B2').values
call = p >= THR


def newcombe(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def boot_auc(yy, ss, n=NB):
    yy = np.asarray(yy).astype(int); ss = np.asarray(ss, float)
    a = roc_auc_score(yy, ss); idx = np.arange(len(yy)); bs = []
    for _ in range(n):
        i = rng.choice(idx, len(idx))
        if yy[i].min() != yy[i].max():
            bs.append(roc_auc_score(yy[i], ss[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(auc=float(a), lo=float(lo), hi=float(hi), n=int(len(yy)), pos=int(yy.sum()))


def ops(sel, c=None):
    c = call if c is None else c
    yy, cc = y[sel], c[sel]
    tp = int((cc & yy).sum()); fn = int((~cc & yy).sum()); tn = int((~cc & ~yy).sum()); fp = int((cc & ~yy).sum())
    out = dict(tp=tp, fn=fn, tn=tn, fp=fp, n=int(sel.sum()), prev=float(yy.mean()))
    for k, (a, b) in dict(sens=(tp, tp + fn), spec=(tn, tn + fp), ppv=(tp, tp + fp), npv=(tn, tn + fn)).items():
        out[k] = a / b if b else np.nan; out[k + '_ci'] = list(wilson(a, b)) if b else [np.nan, np.nan]
    return out


# E1 discrimination
allsel = np.ones(len(m), bool)
R['auc_all'] = boot_auc(y, p)
R['auc_B1'] = boot_auc(y[b1], p[b1]); R['auc_B2'] = boot_auc(y[b2], p[b2])
i1, i2 = np.where(b1)[0], np.where(b2)[0]; d = []
for _ in range(NB):
    r1 = rng.choice(i1, len(i1)); r2 = rng.choice(i2, len(i2))
    if y[r1].min() != y[r1].max() and y[r2].min() != y[r2].max():
        d.append(roc_auc_score(y[r1], p[r1]) - roc_auc_score(y[r2], p[r2]))
R['auc_B1_minus_B2'] = dict(diff=R['auc_B1']['auc'] - R['auc_B2']['auc'], lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
un = (votes == 0) | (votes == 3)
R['auc_unanimous'] = boot_auc(votes[un] == 3, p[un])
# v2 on the same scans (paired DeLong)
v2 = m.pred_volume_ml.values
R['auc_v2_same_scans'] = float(roc_auc_score(y, v2))
R['delong_public_minus_v2'] = [float(x) for x in delong_paired(y, p, v2)]

# E2 operating characteristics at 0.5, by batch; counterfactual PPV
R['ops_all'] = ops(allsel); R['ops_B1'] = ops(b1); R['ops_B2'] = ops(b2)


def ppv_at(se, sp, pi):
    return se * pi / (se * pi + (1 - sp) * (1 - pi))


o1, o2 = R['ops_B1'], R['ops_B2']
cf = ppv_at(o2['sens'], o2['spec'], o1['prev'])
cfb, shb = [], []
for _ in range(NB):
    r1 = rng.choice(i1, len(i1)); r2 = rng.choice(i2, len(i2))
    yy1, cc1, yy2, cc2 = y[r1], call[r1], y[r2], call[r2]
    if yy1.sum() == 0 or yy2.sum() == 0 or cc1.sum() == 0 or (~yy2).sum() == 0:
        continue
    se2 = (cc2 & yy2).sum() / yy2.sum(); sp2 = (~cc2 & ~yy2).sum() / (~yy2).sum(); pi1 = yy1.mean()
    c_ = ppv_at(se2, sp2, pi1); cfb.append(c_)
    pp1 = (cc1 & yy1).sum() / cc1.sum(); pp2 = (cc2 & yy2).sum() / cc2.sum()
    if pp2 - pp1 > 0:
        shb.append((pp2 - c_) / (pp2 - pp1))
R['ppv_counterfactual'] = dict(ppv_B1=o1['ppv'], ppv_B2=o2['ppv'], ppv_cf=cf, ppv_cf_ci=list(np.percentile(cfb, [2.5, 97.5])),
                               share_prevalence=(o2['ppv'] - cf) / (o2['ppv'] - o1['ppv']) if o2['ppv'] != o1['ppv'] else np.nan,
                               share_prevalence_ci=list(np.percentile(shb, [2.5, 97.5])) if shb else [np.nan, np.nan])
# PPV difference between batches (B2 minus B1), Newcombe interval, for the public model and for v2 at 1.0 mL on the same scans (post hoc)
R['ppv_diff_B2_minus_B1'] = list(newcombe(o2['tp'], o2['tp'] + o2['fp'], o1['tp'], o1['tp'] + o1['fp']))
c2 = v2 >= 1.0
v1b = dict(tp=int((c2 & y & b1).sum()), fp=int((c2 & ~y & b1).sum())); v2b = dict(tp=int((c2 & y & b2).sum()), fp=int((c2 & ~y & b2).sum()))
R['v2_ppv_diff_B2_minus_B1'] = list(newcombe(v2b['tp'], v2b['tp'] + v2b['fp'], v1b['tp'], v1b['tp'] + v1b['fp']))
# secondary thresholds (descriptive): specificity 0.90 and the v2 specificity at 1 mL (169/284)
negp = np.sort(p[~y])
def thr_for_spec(sp):
    k = int(np.ceil(sp * len(negp)))   # at least k negatives below the threshold
    return float(negp[k - 1]) + 1e-9 if k >= 1 else 0.0
R['thresholds_secondary'] = {}
for lab, sp in [('spec090', 0.90), ('spec_v2', 169 / 284)]:
    t = thr_for_spec(sp)
    R['thresholds_secondary'][lab] = dict(threshold=t, ops_all=ops(allsel, p > t), ops_B1=ops(b1, p > t), ops_B2=ops(b2, p > t))

# E3 sensitivity by vote split and subtype-count stratum (at 0.5 and at the secondary thresholds)
def amb(c):
    rows = []
    for k in (2, 3):
        for lab, sel_n in [('all', np.ones(len(m), bool)), ('no subtype label', nsub == 0), ('single subtype', nsub == 1), ('two or more subtypes', nsub >= 2)]:
            s = y & (votes == k) & sel_n; n = int(s.sum()); dd = int(c[s].sum())
            lo, hi = wilson(dd, n) if n else (np.nan, np.nan)
            rows.append(dict(votes=f'{k} of 3', stratum=lab, n=n, detected=dd, sens=dd / n if n else np.nan, lo=lo, hi=hi,
                             median_p=float(np.median(p[s])) if n else np.nan))
    s2 = y & (votes == 2); s3 = y & (votes == 3)
    s2s = s2 & (nsub == 1); s3s = s3 & (nsub == 1)
    res = dict(rows=rows,
               diff_all=list(newcombe(int(c[s3].sum()), int(s3.sum()), int(c[s2].sum()), int(s2.sum()))),
               fisher_all=float(stats.fisher_exact([[c[s3].sum(), (~c[s3]).sum()], [c[s2].sum(), (~c[s2]).sum()]])[1]),
               diff_single=list(newcombe(int(c[s3s].sum()), int(s3s.sum()), int(c[s2s].sum()), int(s2s.sum()))),
               fisher_single=float(stats.fisher_exact([[c[s3s].sum(), (~c[s3s]).sum()], [c[s2s].sum(), (~c[s2s]).sum()]])[1]),
               missed_split=int((~c & s2).sum()), missed_total=int((~c & y).sum()))
    neg0, neg1 = votes == 0, votes == 1
    res['neg_call'] = dict(k0=[int(c[neg0].sum()), int(neg0.sum())], k1=[int(c[neg1].sum()), int(neg1.sum())],
                           diff=list(newcombe(int(c[neg1].sum()), int(neg1.sum()), int(c[neg0].sum()), int(neg0.sum()))),
                           median_p=[float(np.median(p[neg0])), float(np.median(p[neg1]))],
                           mannwhitney_p=float(stats.mannwhitneyu(p[neg1], p[neg0]).pvalue))
    return res


R['ambiguity'] = amb(call)
R['ambiguity_spec090'] = amb(p > R['thresholds_secondary']['spec090']['threshold'])
R['ambiguity_spec_v2'] = amb(p > R['thresholds_secondary']['spec_v2']['threshold'])
R['median_p_by_votes'] = {int(k): float(np.median(p[votes == k])) for k in range(4)}
R['n_by_votes'] = {int(k): int((votes == k).sum()) for k in range(4)}
pd.DataFrame(R['ambiguity']['rows']).to_csv(OUTT / 'public_model_ambiguity.csv', index=False)

# E5 kappa with readers
kap = {}
for r_ in (1, 2, 3):
    kap[f'model_R{r_}'] = float(cohen_kappa_score(call.astype(int), m[f'R{r_}:ICH'].astype(int)))
for a, b in [(1, 2), (1, 3), (2, 3)]:
    kap[f'R{a}_R{b}'] = float(cohen_kappa_score(m[f'R{a}:ICH'].astype(int), m[f'R{b}:ICH'].astype(int)))
R['kappa'] = kap

# E6 subtype AUCs: all scans and within ICH-positive scans; paired DeLong against the 'any' probability (type-free baseline)
rows = []; praw = {}
for s in SUBS:
    ys = (m[f'ref_{s}'] == 1).values; ps = m[f'p_{CLS[s]}'].values
    a_all = boot_auc(ys, ps)
    yi, pi_, pa = ys[y], ps[y], p[y]
    a_w = boot_auc(yi, pi_); a_any = boot_auc(yi, pa)
    dd = [float(x) for x in delong_paired(yi, pi_, pa)]
    rows.append(dict(sub=s, pos_all=int(ys.sum()), auc_all=a_all['auc'], auc_all_lo=a_all['lo'], auc_all_hi=a_all['hi'],
                     pos_within=int(yi.sum()), auc_within=a_w['auc'], auc_within_lo=a_w['lo'], auc_within_hi=a_w['hi'],
                     auc_within_any=a_any['auc'], auc_within_any_lo=a_any['lo'], auc_within_any_hi=a_any['hi'],
                     delong_diff=None if dd is None else dd[0], delong_lo=None if dd is None else dd[1], delong_hi=None if dd is None else dd[2],
                     delong_p=None if dd is None else dd[3]))
    if dd is not None:
        praw[s] = dd[3]
if praw:
    order = sorted(praw, key=praw.get); run = 0.0
    for i, s in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * praw[s]))
        [r for r in rows if r['sub'] == s][0]['delong_p_holm'] = run
R['subtypes'] = rows
pd.DataFrame(rows).to_csv(OUTT / 'public_model_subtypes.csv', index=False)

# E7 cross-subtype call matrix at 0.5 (column-negative convention)
Kc = np.zeros((5, 5), int); Nc = np.zeros((5, 5), int)
for i, r_ in enumerate(SUBS):
    rp = (m[f'ref_{r_}'] == 1).values
    for j, c_ in enumerate(SUBS):
        sel = rp if i == j else rp & (m[f'ref_{c_}'] == 0).values
        Nc[i, j] = sel.sum(); Kc[i, j] = ((m[f'p_{CLS[c_]}'] >= THR).values & sel).sum()
R['call_matrix'] = dict(rows=SUBS, k=Kc.tolist(), n=Nc.tolist(), M=np.round(Kc / np.maximum(Nc, 1), 4).tolist())
offd = [Kc[i, 3] / Nc[i, 3] for i in range(5) if i != 3 and Nc[i, 3]]
R['sah_overcall_range'] = [float(min(offd)), float(max(offd))] if offd else None
spec_sub = {}
for s in SUBS:
    ys = (m[f'ref_{s}'] == 1).values; cs = (m[f'p_{CLS[s]}'] >= THR).values
    spec_sub[s] = dict(sens=[int((cs & ys).sum()), int(ys.sum())], spec=[int((~cs & ~ys).sum()), int((~ys).sum())])
R['subtype_ops'] = spec_sub

# E8 discordant references
foc = {}
for s_ in ('CQ500-CT-108', 'CQ500-CT-187'):
    r_ = m[m.name == s_]
    if len(r_):
        foc[s_] = dict(p_any=float(r_.p_any.iloc[0]), p_subdural=float(r_.p_subdural.iloc[0]), p_intraparenchymal=float(r_.p_intraparenchymal.iloc[0]),
                       votes=int(r_[['R1:ICH', 'R2:ICH', 'R3:ICH']].sum(axis=1).iloc[0]), v2_ml=float(r_.pred_volume_ml.iloc[0]))
R['segcq500_discordant'] = foc
bhx_file = E / 'bhx_scan_labels' / 'bhx_scan_labels.csv'
if bhx_file.exists():
    bx = pd.read_csv(bhx_file)
    mm = m.merge(bx, left_on='name', right_on='scan', how='left')
    groups = []
    for s, lab in [('ICH', 'any'), ('SAH', 'Subarachnoid'), ('SDH', 'Subdural'), ('IVH', 'Intraventricular'), ('IPH', 'Intraparenchymal'), ('EDH', 'Epidural')]:
        bpos = (mm[f'bhx_{lab}'].fillna(0) > 0).values; mpos = (mm[f'ref_{s}'] == 1).values
        ps = mm['p_any'].values if s == 'ICH' else mm[f'p_{CLS[s]}'].values
        for g, sel in [('both', bpos & mpos), ('BHX only', bpos & ~mpos), ('majority only', ~bpos & mpos), ('neither', ~bpos & ~mpos)]:
            n = int(sel.sum())
            groups.append(dict(label=s, group=g, n=n, median_p=float(np.median(ps[sel])) if n else np.nan,
                               q1=float(np.percentile(ps[sel], 25)) if n else np.nan, q3=float(np.percentile(ps[sel], 75)) if n else np.nan,
                               k_ge05=int((ps[sel] >= THR).sum()), share_ge05=float((ps[sel] >= THR).mean()) if n else np.nan,
                               any_read=int(((mm[[f'R1:{s}', f'R2:{s}', f'R3:{s}']].sum(axis=1) >= 1).values & sel).sum()) if s != 'ICH' or True else None))
    R['bhx_groups'] = groups
    pd.DataFrame(groups).to_csv(OUTT / 'public_model_bhx_groups.csv', index=False)
save_json(R, OUT / 'public_model_cq500.json')
print(json.dumps({k: R[k] for k in ['n_scans', 'auc_all', 'auc_B1', 'auc_B2', 'auc_B1_minus_B2', 'auc_unanimous', 'ppv_counterfactual', 'kappa', 'sah_overcall_range', 'segcq500_discordant']}, indent=1, default=float))
print({k: (R[k]['sens'], R[k]['spec'], R[k]['ppv'], R[k]['npv']) for k in ['ops_all', 'ops_B1', 'ops_B2']})
print(pd.DataFrame(R['ambiguity']['rows']).to_string()); print(R['ambiguity']['diff_all'], R['ambiguity']['diff_single'], R['ambiguity']['fisher_single'], R['ambiguity']['neg_call'])
print(pd.DataFrame(rows)[['sub', 'pos_all', 'auc_all', 'pos_within', 'auc_within', 'auc_within_any', 'delong_p']].to_string())
