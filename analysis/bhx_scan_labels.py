"""Scan-level BHX labels: number of boxes of each type per CQ500 scan, linked by DICOM StudyInstanceUID (post hoc).
Inputs: the BHX v1.1 CSV files (PhysioNet, CC BY-SA 4.0) and data/evidence/cq500_uids/cq500_series_uids.csv (study and series UIDs
of all 489 CQ500 scans, read from the released DICOM headers by run/fetch_uids_range.py; checked against the UIDs read by the public-model run). Output: data/evidence/bhx_scan_labels/bhx_scan_labels.csv
Usage: python bhx_scan_labels.py <bhx_dir>"""
import sys
import pandas as pd
from common import *
bdir = sys.argv[1]
u = pd.read_csv(E / 'cq500_uids' / 'cq500_series_uids.csv')
assert u.scan.nunique() == 489, 'UIDs missing for some CQ500 scans'
study = u.groupby('study_uid').scan.agg(lambda s: sorted(set(s)))
assert study.apply(len).max() == 1, 'a study UID maps to more than one CQ500 scan'
s2scan = {k: v[0] for k, v in study.items()}
ser2scan = dict(zip(u.series_uid, u.scan))
out = pd.DataFrame({'scan': sorted(u.scan.unique())})
summ = {}
for fn, tag in [('1_Initial_Manual_Labeling.csv', 'bhx'), ('2_Extrapolation_to_All_Series.csv', 'bhxall')]:
    b = pd.read_csv(f'{bdir}/{fn}')
    b['scan'] = b.StudyInstanceUID.map(s2scan)
    unmatched = int(b.scan.isna().sum())
    c = b.dropna(subset=['scan']).groupby(['scan', 'labelName']).size().unstack(fill_value=0)
    c.columns = [f'{tag}_{x}' for x in c.columns]
    c[f'{tag}_any'] = c.sum(axis=1)
    out = out.merge(c, left_on='scan', right_index=True, how='left')
    print(fn, 'rows', len(b), 'unmatched rows', unmatched, 'scans with boxes', int((c[f"{tag}_any"] > 0).sum()), 'labels', sorted(b.labelName.unique()))
    summ[tag] = dict(file=fn, rows=int(len(b)), unmatched_rows=unmatched, scans_with_boxes=int((c[f'{tag}_any'] > 0).sum()))
out = out.fillna(0)
for c in out.columns[1:]:
    out[c] = out[c].astype(int)
(E / 'bhx_scan_labels').mkdir(exist_ok=True, parents=True)
out.to_csv(E / 'bhx_scan_labels' / 'bhx_scan_labels.csv', index=False)
import json
json.dump(summ, open(E / 'bhx_scan_labels' / 'bhx_linkage_summary.json', 'w'), indent=1)
print(out.describe().T[['count', 'mean', 'max']])
