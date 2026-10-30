"""Model-free extent of hemorrhage per CQ500 scan from the BHX boxes (post hoc): the volume covered by the union of the boxes on each
slice of the series that the locked evaluation analyzed, in mL (an upper bound of hemorrhage volume, not a measurement of it).
Boxes drawn by hand on that series (initial manual labeling) are used where they exist; otherwise the boxes extrapolated to that
series. Geometry (in-plane pixel spacing and slice interval) is that of the analyzed series.
Inputs: the BHX v1.1 CSV files (PhysioNet, CC BY-SA 4.0), data/evidence/cq500_uids/cq500_series_uids.csv and the per-scan geometry
of the locked series. Output: data/evidence/bhx_scan_labels/bhx_scan_extent.csv (CC BY-SA 4.0, derived from BHX).
Usage: python bhx_extent.py <bhx_dir>"""
import ast
import sys

import numpy as np
import pandas as pd

from common import E, load_cq500

bdir = sys.argv[1]
u = pd.read_csv(E / 'cq500_uids' / 'cq500_series_uids.csv')
locked = u[u.locked_soft_series == 1].set_index('scan').series_uid
assert locked.index.nunique() == 489
m = load_cq500().set_index('name')
rows = []
src = {}
for fn, tag in [('1_Initial_Manual_Labeling.csv', 'manual'), ('2_Extrapolation_to_All_Series.csv', 'extrapolated')]:
    b = pd.read_csv(f'{bdir}/{fn}')
    b = b[b.SeriesInstanceUID.isin(set(locked.values))].copy()
    b['scan'] = b.SeriesInstanceUID.map({v: k for k, v in locked.items()})
    src[tag] = b
for scan in sorted(locked.index):
    use = None
    for tag in ('manual', 'extrapolated'):
        bb = src[tag][src[tag].scan == scan]
        if len(bb):
            use = (tag, bb)
            break
    if use is None:
        rows.append(dict(scan=scan, bhx_extent_source='none', bhx_extent_slices=0, bhx_extent_ml=0.0, bhx_extent_boxes=0))
        continue
    tag, bb = use
    vox = 0
    for sop, g in bb.groupby('SOPInstanceUID'):
        grid = np.zeros((1024, 1024), bool)
        for d in g['data']:
            r = ast.literal_eval(d)
            x0, y0 = max(0, int(np.floor(r['x']))), max(0, int(np.floor(r['y'])))
            x1, y1 = int(np.ceil(r['x'] + r['width'])), int(np.ceil(r['y'] + r['height']))
            grid[y0:y1, x0:x1] = True
        vox += int(grid.sum())
    dxy, dz = float(m.loc[scan, 'dxy']), float(m.loc[scan, 'dz'])
    rows.append(dict(scan=scan, bhx_extent_source=tag, bhx_extent_slices=int(bb.SOPInstanceUID.nunique()),
                     bhx_extent_ml=round(vox * dxy * dxy * dz / 1000.0, 3), bhx_extent_boxes=int(len(bb))))
out = pd.DataFrame(rows)
(E / 'bhx_scan_labels').mkdir(exist_ok=True, parents=True)
out.to_csv(E / 'bhx_scan_labels' / 'bhx_scan_extent.csv', index=False)
print(out.bhx_extent_source.value_counts().to_dict())
print(out[out.bhx_extent_ml > 0].bhx_extent_ml.describe())
