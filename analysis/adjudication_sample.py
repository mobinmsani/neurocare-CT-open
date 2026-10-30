"""Stratified case list for independent neuroradiologist adjudication of discordant open references (post hoc, 29 Oct 2026).
Strata (sizes fixed in advance; random draws with seed 20261029 inside each stratum):
  A  Seg-CQ500 masks on CQ500 majority-negative scans: all (CT-108, CT-187)
  B  BHX subarachnoid boxes without a majority SAH label: 15
  C  BHX subdural boxes without a majority SDH label: 10
  D  BHX intraventricular boxes without a majority IVH label: 5
  E  BHX boxes of any type on majority ICH-negative scans: 5
  F  majority ICH-positive scans decided 2-1: 10
  G  majority ICH-negative scans with one positive read: 10
  H  unanimous ICH-negative scans that the public model called positive (probability >= 0.5): all
  I  controls, unanimous ICH-positive scans with BHX boxes: 10
  J  controls, unanimous ICH-negative scans without BHX boxes: 10
A scan enters the first stratum it qualifies for. The coordinator keeps the stratum; readers receive scans in random order
without strata, labels, boxes, masks or model outputs.
Outputs: out/tables/adjudication_case_list.csv (coordinator), out/tables/adjudication_reading_order.csv (readers)"""
import numpy as np, pandas as pd
from common import *
rng = np.random.default_rng(20261029)
m = load_cq500()
bx = pd.read_csv(E / 'bhx_scan_labels' / 'bhx_scan_labels.csv')
m = m.merge(bx, left_on='name', right_on='scan', how='left').fillna({c: 0 for c in bx.columns if c != 'scan'})
pm = pd.read_csv(E / 'public_model_cq500' / 'public_model_per_scan.csv')[['scan', 'p_any']].rename(columns={'scan': 'name'})
m = m.merge(pm, on='name', how='left')
votes = m[['R1:ICH', 'R2:ICH', 'R3:ICH']].sum(axis=1).astype(int)
S = [('A', 'Seg-CQ500 mask on a majority-negative scan', m.name.isin(['CQ500-CT-108', 'CQ500-CT-187']), None),
     ('B', 'BHX SAH box without a majority SAH label', (m.bhx_Subarachnoid > 0) & (m.ref_SAH == 0), 15),
     ('C', 'BHX SDH box without a majority SDH label', (m.bhx_Subdural > 0) & (m.ref_SDH == 0), 10),
     ('D', 'BHX IVH box without a majority IVH label', (m.bhx_Intraventricular > 0) & (m.ref_IVH == 0), 5),
     ('E', 'BHX box on a majority ICH-negative scan', (m.bhx_any > 0) & (m.ref_ICH == 0), 5),
     ('F', 'Majority ICH-positive, decided 2-1', (votes == 2), 10),
     ('G', 'Majority ICH-negative, one positive read', (votes == 1), 10),
     ('H', 'Unanimous ICH-negative, public model positive (probability >= 0.5)', (votes == 0) & (m.p_any >= 0.5), None),
     ('I', 'Control: unanimous positive with BHX boxes', (votes == 3) & (m.bhx_any > 0), 10),
     ('J', 'Control: unanimous negative without BHX boxes, public model negative (probability < 0.5)', (votes == 0) & (m.bhx_any == 0) & (m.p_any < 0.5), 10)]
taken, rows = set(), []
for code, desc, sel, k in S:
    pool = sorted(set(m.name[sel]) - taken, key=lambda s: int(s.split('-')[-1]))
    pick = pool if k is None else list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    for s in sorted(pick, key=lambda s: int(s.split('-')[-1])):
        rows.append(dict(scan=s, stratum=code, description=desc, pool_size=len(pool))); taken.add(s)
cl = pd.DataFrame(rows)
cl.to_csv(OUT / 'tables' / 'adjudication_case_list.csv', index=False)
order = cl.sample(frac=1, random_state=20261029).reset_index(drop=True)
order.insert(0, 'reading_number', range(1, len(order) + 1))
order[['reading_number', 'scan']].to_csv(OUT / 'tables' / 'adjudication_reading_order.csv', index=False)
print(cl.groupby(['stratum', 'description']).agg(n=('scan', 'size'), pool=('pool_size', 'first')))
print(len(cl), 'scans')
