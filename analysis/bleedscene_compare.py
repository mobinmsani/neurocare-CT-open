"""Agreement between BleedScene3D scene-graph annotations and the CQ500 three-reader labels on the CQ500 scans they share
(28 Oct 2026; model-independent, post hoc).

BleedScene3D (Sanner, Kaggle sannera/bleedscene3d, version 2 of 23 Oct 2026, CC BY-NC-SA 4.0) provides, for 574 hemorrhage-
positive volumes from five public datasets, voxel masks of every bleeding instance with six subtype attributes, the ventricle
system and the midline, and three relations (bleeding causes shift of the midline; flows into the ventricles; causes
ventricle asymmetry). Volume CQ500_<nnn> corresponds to CQ500-CT-<n> (dataset preparation notebook). The dataset was released
after the probe analyses, so it enters only this comparison and the capacity rubric.

Usage: python bleedscene_compare.py <path to BleedScene3D raw/CQ500/psgg folder>
Outputs: out/bleedscene_cq500.json, out/tables/bleedscene_cq500.csv
"""
import json, os, sys
import numpy as np, pandas as pd
from common import *

PSGG = Path(sys.argv[1]) if len(sys.argv) > 1 else None
assert PSGG is not None and PSGG.is_dir(), 'pass the BleedScene3D raw/CQ500/psgg folder'
ATTR = {1: 'SAH (not basal)', 2: 'SAH (basal)', 3: 'IPH', 4: 'IVH', 5: 'EDH', 6: 'SDH'}
rows = []
for fn in sorted(os.listdir(PSGG)):
    if not fn.endswith('.json'):
        continue
    d = json.load(open(PSGG / fn))
    pid = int(fn.split('_')[1].split('.')[0])
    bleeds = [b for b in d['bounding_boxes'] if b['class_id'] == 3]

    def flag(i):
        return any(any(a['id'] == i and a['value'] for a in b['attributes']) for b in bleeds)

    rel = d.get('relations', [])
    rows.append(dict(name=f'CQ500-CT-{pid}', instances=len(bleeds), SAH=flag(1) or flag(2), IPH=flag(3), IVH=flag(4), EDH=flag(5), SDH=flag(6),
                     shift=any(r['id'] == 1 for r in rel), flows=any(r['id'] == 2 for r in rel), asym=any(r['id'] == 3 for r in rel)))
bs = pd.DataFrame(rows)
m = load_cq500().set_index('name')
assert bs.name.isin(m.index).all(), 'BleedScene3D CQ500 volume outside the analyzed scans'
bs = bs.set_index('name'); mm = m.loc[bs.index]


def kappa(a, b):
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe))


R = dict(n=int(len(bs)), majority_ich=int(mm.ref_ICH.sum()),
         ich_votes={int(k): int(v) for k, v in mm[['R1:ICH', 'R2:ICH', 'R3:ICH']].sum(axis=1).value_counts().sort_index().items()})
tab = []
for s, lab in [('IPH', 'IPH'), ('IVH', 'IVH'), ('SAH', 'SAH'), ('SDH', 'SDH'), ('EDH', 'EDH'), ('shift', 'MidlineShift')]:
    a = bs[s].values.astype(bool); b = (mm[f'ref_{lab}'] == 1).values
    rec = dict(label=s if s != 'shift' else 'midline shift (relation) vs MLS >5 mm', bleedscene_pos=int(a.sum()), cq500_majority_pos=int(b.sum()),
               both=int((a & b).sum()), bleedscene_only=int((a & ~b).sum()), cq500_only=int((~a & b).sum()), kappa=kappa(a, b))
    tab.append(rec)
R['agreement'] = tab
mv = mm[['R1:MidlineShift', 'R2:MidlineShift', 'R3:MidlineShift']].sum(axis=1).values
R['shift_by_mls_votes'] = {int(k): dict(n=int((mv == k).sum()), with_relation=int(bs['shift'].values[mv == k].sum())) for k in range(4)}
R['relations_total'] = dict(shift=int(bs['shift'].sum()), flows=int(bs.flows.sum()), asym=int(bs.asym.sum()))
pd.DataFrame(tab).to_csv(OUT / 'tables' / 'bleedscene_cq500.csv', index=False)
save_json(R, OUT / 'bleedscene_cq500.json')
print(json.dumps(R, indent=1))
