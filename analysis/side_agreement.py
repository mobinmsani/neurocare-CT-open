"""Post hoc: reader agreement on hemorrhage side among the CQ500 scans on which side is evaluated.
The side labels (BleedLocation-Left, BleedLocation-Right) are released for every scan and are 0 whenever a reader called no
hemorrhage, so Fleiss' kappa over all 491 scans mixes agreement on detection with agreement on side. Here kappa is computed
among the scans that the reader majority called hemorrhage-positive (the population in which side is evaluated) and, as a
sensitivity analysis, among the scans that all three readers called positive. Bootstrap percentile intervals: 10,000 resamples
of scans, seed 20261029 (as in kappa491.py).
Output: out/side_agreement.json, out/tables/cq500_side_agreement.csv"""
import numpy as np
import pandas as pd

from common import E, OUT, cohen_kappa, fleiss_kappa, save_json

r = pd.read_csv(E / 'cq500_external/reads.csv')
assert len(r) == 491
votes = r[[f'R{i}:ICH' for i in (1, 2, 3)]].sum(axis=1)
POP = {'all scans': votes >= 0, 'majority hemorrhage-positive scans': votes >= 2, 'unanimously hemorrhage-positive scans': votes == 3}
rows = []
for pop, sel in POP.items():
    for lab in ('BleedLocation-Left', 'BleedLocation-Right'):
        rng = np.random.default_rng(20261029)
        X = r.loc[sel, [f'R{i}:{lab}' for i in (1, 2, 3)]].values
        cnt = np.c_[X.sum(1), 3 - X.sum(1)]
        n = len(X)
        fk = fleiss_kappa(cnt)
        bs = [fleiss_kappa(cnt[rng.integers(0, n, n)]) for _ in range(10000)] if pop != 'all scans' else [np.nan]
        lo, hi = np.nanpercentile(bs, [2.5, 97.5]) if pop != 'all scans' else (np.nan, np.nan)
        pk = [cohen_kappa(X[:, a], X[:, b]) for a, b in [(0, 1), (1, 2), (2, 0)]]
        k = X.sum(1)
        if pop == 'all scans':   # one interval per statistic: the 491-scan values and intervals are those of kappa491.py
            ref = pd.read_csv(OUT / 'tables' / 'cq500_reader_agreement_491.csv').set_index('label')
            assert abs(ref.loc[lab, 'fleiss'] - fk) < 1e-9
            lo, hi = float(ref.loc[lab, 'lo']), float(ref.loc[lab, 'hi'])
        rows.append(dict(population=pop, label=lab, n=n, fleiss=fk, lo=lo, hi=hi, cohen_min=min(pk), cohen_max=max(pk),
                         majority_pos=int((k >= 2).sum()), split_among_majority_pos_pct=100 * (k == 2).sum() / max((k >= 2).sum(), 1)))
df = pd.DataFrame(rows)
df.to_csv(OUT / 'tables' / 'cq500_side_agreement.csv', index=False)
out = {}
for pop in POP:
    d = df[df.population == pop].set_index('label')
    out[pop] = dict(n=int(d.n.iloc[0]), **{l: dict(k=float(d.loc[l, 'fleiss']), lo=float(d.loc[l, 'lo']), hi=float(d.loc[l, 'hi']),
                                                     majority_pos=int(d.loc[l, 'majority_pos'])) for l in d.index})
    out[pop]['min'] = float(d.fleiss.min())
save_json(out, OUT / 'side_agreement.json')
print(df.round(4).to_string(index=False))
