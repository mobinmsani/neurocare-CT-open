"""Post hoc: how much the choice of reference moves detection estimates for any intracranial hemorrhage on CQ500.
References: the reader majority (the reference of the locked evaluation), each reader alone, at least one positive read, the reader
majority with BHX added (majority positive or any BHX box on the scan), and the majority restricted to unanimous scans.
Models: v2 (total predicted volume, threshold 1.0 mL; locked evaluation outputs) and the public model (probability for any
hemorrhage, threshold 0.5). AUC with bootstrap percentile intervals (10,000 resamples of scans, seed 20261029, as in Methods); the
majority and unanimous-only rows repeat the intervals of the main analyses (locked v2 evaluation and new_analyses.py;
public_model_cq500.py). Sensitivity and specificity
with Wilson intervals.
Output: out/reference_choice.json, out/tables/reference_choice.csv"""
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import E, OUT, load_cq500, save_json, wilson

NB = int(os.environ.get('PM_NB', 10000))


def boot_auc(yy, ss, seed=20261029, n=NB):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(yy)); bs = []
    for _ in range(n):
        i = rng.choice(idx, len(idx))
        if yy[i].min() != yy[i].max():
            bs.append(roc_auc_score(yy[i], ss[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(roc_auc_score(yy, ss)), float(lo), float(hi)


_t = json.load(open(E / 'cq500_external' / 'cq500_results.json'))['table_detection']
_v2 = [r for r in _t if r['Finding'] == 'ICH'][0]['AUC (95% CI)']                    # locked evaluation, e.g. '0.806 (0.769–0.844)'
V2_MAJ = [float(x) for x in re.findall(r'[0-9.]+', _v2)][1:3]
PM_MAJ = json.load(open(OUT / 'public_model_cq500.json'))['auc_all']
PM_UN = json.load(open(OUT / 'public_model_cq500.json'))['auc_unanimous']
V2_UN = json.load(open(OUT / 'new_analyses_cq500.json'))['amb_auc_ICH']['unanimous'][1:3]   # post hoc v2 analysis

m = load_cq500()
bx = pd.read_csv(E / 'bhx_scan_labels' / 'bhx_scan_labels.csv')[['scan', 'bhx_any']].rename(columns={'scan': 'name'})
m = m.merge(bx, on='name', how='left')
pm = pd.read_csv(E / 'public_model_cq500' / 'public_model_per_scan.csv')[['scan', 'p_any']].rename(columns={'scan': 'name'})
m = m.merge(pm, on='name', how='left')
votes = m['R1:ICH'] + m['R2:ICH'] + m['R3:ICH']
REFS = [('Reader majority (reference of the locked evaluation)', m.ref_ICH),
        ('Reader 1', m['R1:ICH']), ('Reader 2', m['R2:ICH']), ('Reader 3', m['R3:ICH']),
        ('At least one positive read', (votes >= 1).astype(int)),
        ('Reader majority or a BHX box', ((m.ref_ICH == 1) | (m.bhx_any > 0)).astype(int)),
        ('Reader majority, unanimous scans only', m.ref_ICH.where(votes.isin([0, 3])))]
MODELS = [('v2', 'pred_volume_ml', 1.0), ('public model', 'p_any', 0.5)]
rows = []
for mname, col, thr in MODELS:
    for rname, y in REFS:
        ok = y.notna() & m[col].notna()
        yy, ss = y[ok].astype(int).values, m.loc[ok, col].values
        if rname.startswith('Reader majority (reference'):
            a = float(roc_auc_score(yy, ss))
            lo, hi = V2_MAJ if mname == 'v2' else (PM_MAJ['lo'], PM_MAJ['hi'])
        elif rname.startswith('Reader majority, unanimous'):
            a = float(roc_auc_score(yy, ss))
            lo, hi = V2_UN if mname == 'v2' else (PM_UN['lo'], PM_UN['hi'])
        else:
            a, lo, hi = boot_auc(yy, ss)
        det = ss >= thr
        tp, fn = int((det & (yy == 1)).sum()), int((~det & (yy == 1)).sum())
        tn, fp = int((~det & (yy == 0)).sum()), int((det & (yy == 0)).sum())
        rows.append(dict(model=mname, reference=rname, n=int(ok.sum()), positives=int(yy.sum()), auc=a, auc_lo=lo, auc_hi=hi,
                         sens=tp / (tp + fn), sens_lo=wilson(tp, tp + fn)[0], sens_hi=wilson(tp, tp + fn)[1], tp=tp, fn=fn,
                         spec=tn / (tn + fp), spec_lo=wilson(tn, tn + fp)[0], spec_hi=wilson(tn, tn + fp)[1], tn=tn, fp=fp))
df = pd.DataFrame(rows)
df.to_csv(OUT / 'tables' / 'reference_choice.csv', index=False)
summ = {}
for mname, _, _ in MODELS:
    d = df[df.model == mname]
    rd = d[d.reference.str.startswith('Reader ') & d.reference.str[-1].str.isdigit()]
    summ[mname] = dict(n=int(d.n.iloc[0]), auc_readers=[float(rd.auc.min()), float(rd.auc.max())], sens_readers=[float(rd.sens.min()), float(rd.sens.max())],
                       spec_readers=[float(rd.spec.min()), float(rd.spec.max())], auc_majority=float(d.auc.iloc[0]),
                       auc_composite=float(d[d.reference.str.contains('BHX')].auc.iloc[0]),
                       sens_composite=float(d[d.reference.str.contains('BHX')].sens.iloc[0]),
                       auc_any_read=float(d[d.reference.str.contains('one positive')].auc.iloc[0]))
# predictive values by batch against the reader majority and against the composite reference (majority positive or any BHX box)
ppv = {}
for mname, col, thr in MODELS:
    for key, (rname, y) in (('majority', REFS[0]), ('composite', [r for r in REFS if 'BHX' in r[0]][0])):
        for b in ('B1', 'B2'):
            ok = y.notna() & m[col].notna() & (m.Category == b)
            yy, det = y[ok].astype(int).values, m.loc[ok, col].values >= thr
            tp, fp = int((det & (yy == 1)).sum()), int((det & (yy == 0)).sum())
            ppv.setdefault(mname, {}).setdefault(key, {})[b] = dict(tp=tp, fp=fp, ppv=tp / (tp + fp), ci=list(wilson(tp, tp + fp)))
summ['ppv_by_batch'] = ppv
save_json(dict(rows=rows, summary=summ), OUT / 'reference_choice.json')
print(df[['model', 'reference', 'n', 'positives', 'auc', 'sens', 'spec']].round(3).to_string(index=False))
print(summ)
