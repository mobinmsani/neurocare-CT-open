"""Exact scan-level discordance between BHX boxes and the CQ500 reader majority (post hoc, 29 Oct 2026).
BHX boxes (initial manual labeling on thick slices, BHX v1.1) are linked to CQ500 scans by DICOM StudyInstanceUID
(data/evidence/bhx_scan_labels/bhx_scan_labels.csv, from bhx_scan_labels.py). Counts replace the lower bounds of the first version.
Outputs: out/bhx_exact.json, out/tables/bhx_exact.csv"""
import json
import pandas as pd
from common import *
m = load_cq500()
bx = pd.read_csv(E / 'bhx_scan_labels' / 'bhx_scan_labels.csv')
m = m.merge(bx, left_on='name', right_on='scan', how='left')
missing = int(m['bhx_any'].isna().sum())
m = m.fillna({c: 0 for c in bx.columns if c != 'scan'})
MAP = dict(ICH='any', IPH='Intraparenchymal', IVH='Intraventricular', SAH='Subarachnoid', SDH='Subdural', EDH='Epidural', ChronicBleed='Chronic')
R, rows = dict(n_scans=int(len(m)), scans_without_uid=missing), []
for lab, b in MAP.items():
    col = f'bhx_{b}'
    if col not in m:
        continue
    bp = (m[col] > 0).values; mp = (m[f'ref_{lab}'] == 1).values
    anyread = (m[[f'R1:{lab}', f'R2:{lab}', f'R3:{lab}']].sum(axis=1) >= 1).values
    d = dict(label=lab, bhx_label=b, bhx_scans=int(bp.sum()), majority_pos=int(mp.sum()), both=int((bp & mp).sum()), bhx_only=int((bp & ~mp).sum()),
             majority_only=int((~bp & mp).sum()), neither=int((~bp & ~mp).sum()), bhx_only_no_read=int((bp & ~mp & ~anyread).sum()),
             bhx_only_one_read=int((bp & ~mp & anyread).sum()), bhx_only_ich_majority_negative=int((bp & ~mp & (m.ref_ICH == 0).values).sum()),
             lower_bound_v1=max(0, int(bp.sum()) - int(mp.sum())))
    rows.append(d); R[lab] = d
pd.DataFrame(rows).to_csv(OUT / 'tables' / 'bhx_exact.csv', index=False)
save_json(R, OUT / 'bhx_exact.json')
print(pd.DataFrame(rows).to_string())
