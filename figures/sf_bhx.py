"""SF_bhx - Any-hemorrhage overlap of the binary v2 model with subtype-labeled BHX boxes.

a share of BHX boxes containing at least one v2-predicted voxel on the box slice, by BHX box label (outputs of the
  locked pipeline; Wilson intervals treat boxes as independent and ignore clustering within scans), with the scan counts of
  BHX boxes and of CQ500 reader-majority labels for the same subtype
The CQ500 examples of overlapped and non-overlapped boxes are the scans of Fig. 4a, b and d (not repeated here); their overlap
status, stated in the legend, is checked below from the image overlays.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nc_b import *
from ncdata import *
from matplotlib.path import Path as MPath

HFIG = 72
fig = plt.figure(figsize=(W2, HFIG * MM))
m = load_cq500().set_index('name')
bx = CQ['bhx_box_hit']
disc = {d['bhx_label']: d for d in R3['bhx_subtype_discordance']}
ROWS = [('Epidural', 'EDH'), ('Intraparenchymal', 'IPH'), ('Intraventricular', 'IVH'), ('Subarachnoid', 'SAH'), ('Subdural', 'SDH'), ('Chronic', 'Chronic')]

# ============================================================================ a: overlap by box label
axA = axes_mm(fig, 20, 15, 60, 38)
for i, (k_, short) in enumerate(ROWS):
    r = bx[k_]
    if i % 2 == 0:
        axA.axhspan(i - 0.5, i + 0.5, color=BAND, lw=0, zorder=0)
    lo, hi = r['ci']                       # Wilson interval stored by the locked pipeline (z = 1.96)
    assert max(abs(a - b) for a, b in zip(wilson_ci(int(r['hit']), int(r['boxes'])), (lo, hi))) < 1e-5
    ci_point(axA, r['rate'], i, lo, hi, M['v2'], marker=MK['v2'], filled=True, ms=4.6)
    d = disc[k_]
    assert d['bhx_scans_with_boxes'] == int(r['scans'])
    cells = [f"{pct(r['rate'])}", f"{r['hit']}/{r['boxes']}", f"{int(r['scans'])}", f"{d['cq500_majority_positive_scans']}"]
    for xx, c in zip([1.06, 1.33, 1.73, 2.13], cells):
        axA.text(xx, i, c, transform=axA.get_yaxis_transform(), fontsize=7, ha='left', va='center', color=INK)
for xx, h in zip([1.06, 1.33, 1.73, 2.13], ['Share', 'Boxes\noverlapped', 'Scans with\nBHX boxes', 'CQ500 majority-\npositive scans']):
    axA.text(xx, -0.62, h, transform=axA.get_yaxis_transform(), fontsize=7, ha='left', va='bottom', color=INK2, linespacing=1.1)
axA.set_yticks(range(len(ROWS)), [s for _, s in ROWS])
axA.tick_params(axis='y', length=0, labelcolor=INK)
axA.spines['left'].set_visible(False)
axA.set_ylim(len(ROWS) - 0.5, -0.5); axA.set_xlim(0, 1.0)
axA.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ['0', '25', '50', '75', '100'])
axA.set_xlabel('BHX boxes with ≥1 v2-predicted voxel (%, 95% CI)')
grid(axA, 'x')
note(fig, 4, 63.0, 'v2 is a binary model: overlap shows that some hemorrhage was predicted inside a box, not that the subtype was recognized or delineated.\n'
                  f"A box is overlapped if it contains ≥1 v2-predicted voxel on its slice. Intervals treat boxes as independent and ignore clustering within scans ({CQ['bhx_scans_matched']} matched scans).", color=INK2)
ptitle(fig, axA, '', 'Any-hemorrhage overlap with subtype-labeled BHX boxes (v2, locked)', dx_mm=16.0, dy_mm=8.0)


# ============================================================================ b: examples
def overlaps(rec):
    """For each box overlay: True if a predicted contour lies inside it or encloses one of its corners."""
    boxes = [np.asarray(o['subpaths'][0]) for o in rec['overlays'] if o['kind'] == 'box']
    preds = [np.asarray(sp) for o in rec['overlays'] if o['kind'] == 'pred' for sp in o['subpaths']]
    out = []
    for b in boxes:
        x0, x1, y0, y1 = b[:, 0].min(), b[:, 0].max(), b[:, 1].min(), b[:, 1].max()
        bp = MPath([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        hit = any(bp.contains_points(p).any() for p in preds) or any(MPath(p).contains_points([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]).any() for p in preds if len(p) > 2)
        out.append((hit, (x0, x1, y0, y1)))
    return out


EX = [('gal_01', 'EDH', 1), ('gal_02', 'SDH', 1), ('gal_04', 'SAH', 0)]   # Fig. 4a, b and d: boxes on the displayed slice overlapped
for key, sub, expect in EX:
    rec, ct = load_asset(key)
    row = m.loc[rec['case']]
    assert int(row[f'ref_{sub}']) == 1 and sum(int(row[f'ref_{s_}']) for s_ in SUB) == 1
    ov = overlaps(rec)
    assert len(ov) == 1 and sum(h_ for h_, _ in ov) == expect, (key, ov)
    print(rec['case'], sub, 'boxes on slice', len(ov), 'overlapped', sum(h_ for h_, _ in ov))

print('overflow', overflow_mm(fig))
save(fig, 'SF_bhx', 'supp')
