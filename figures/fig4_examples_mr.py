"""Figure 4 - CQ500 scans with their open references and model outputs.

a-f: the scan with the middle-ranked v2 volume (upper middle for even counts) among scans with each subtype alone (chronic bleed not
counted as a subtype), among the 28 IVH-positive scans (no scan had isolated IVH) and among the 19 scans with exactly three subtypes;
g: the majority-negative scan with the largest v2 volume; h: the only majority-positive scan with zero v2 volume.
Below each image: reader votes, v2 and v3b volumes and the any-hemorrhage probability of the public model (read from its per-scan outputs).
The selection rule is re-applied here and asserted against the stored image assets.
"""
from nc_s1 import *

LBL = '#000000'   # backing for labels on CT images

H = 156
fig = plt.figure(figsize=(W2, H * MM))
m = load_cq500().set_index('name')
PMS = pd.read_csv(D / 'evidence' / 'public_model_cq500' / 'public_model_per_scan.csv').set_index('scan')
SUBS = ['EDH', 'IPH', 'IVH', 'SAH', 'SDH']
ns = m[[f'ref_{s}' for s in SUBS]].sum(axis=1)


def mid(sel):
    sel = sel.sort_values('pred_volume_ml')
    return sel.index[len(sel) // 2], len(sel)


def pfmt(p):
    return '>0.99' if p >= 0.995 else ('<0.01' if p < 0.005 else f'{p:.2f}')


rule = {}
for s in ['EDH', 'SDH', 'IPH', 'SAH']:
    rule[s] = mid(m[(m[f'ref_{s}'] == 1) & (ns == 1)])
rule['IVH'] = mid(m[m.ref_IVH == 1])
rule['three'] = mid(m[ns == 3])
neg = m[m.ref_ICH == 0]
rule['fp'] = (neg.pred_volume_ml.idxmax(), len(neg))
pz = m[(m.ref_ICH == 1) & (m.pred_volume_ml == 0)]
assert len(pz) == 1
rule['fn'] = (pz.index[0], int((m.ref_ICH == 1).sum()))

cases = [('gal_01', 'EDH', 'EDH alone', f"middle of {rule['EDH'][1]} scans"),
         ('gal_02', 'SDH', 'SDH alone', f"middle of {rule['SDH'][1]} scans"),
         ('gal_03', 'IPH', 'IPH alone', f"middle of {rule['IPH'][1]} scans"),
         ('gal_04', 'SAH', 'SAH alone', f"middle of {rule['SAH'][1]} scans"),
         ('gal_05', 'IVH', 'IVH, co-occurring', f"middle of all {rule['IVH'][1]} IVH scans"),
         ('gal_06', 'three', 'Three subtypes', f"middle of {rule['three'][1]} scans"),
         ('gal_07', 'fp', 'Probe-positive', 'majority negative; largest v2'),
         ('gal_08', 'fn', 'Probe-negative', 'majority positive; zero v2')]
NM = {'EDH': 'EDH', 'IPH': 'IPH', 'IVH': 'IVH', 'SAH': 'SAH', 'SDH': 'SDH', 'ChronicBleed': 'chronic'}
TW, GX = 41.6, 2.6
XS = [1.6, 1.6 + TW + GX, 1.6 + 2 * (TW + GX) + 2.2, 1.6 + 3 * (TW + GX) + 2.2]
assert XS[-1] + TW <= 179.0
ROWY = [16.5, 85.0]          # top of image, rows 1 and 2
letters = 'abcdefgh'
for i, (key, rk, ttl, sub) in enumerate(cases):
    rec, ct = load_asset(key)
    c = rec['case']
    assert c == rule[rk][0], (c, rule[rk])
    r = m.loc[c]
    assert f"model volume {r.pred_volume_ml:.1f} mL" == rec['titles'][1]
    row, col = divmod(i, 4)
    x, ytop = XS[col], ROWY[row]
    ax = axes_mm(fig, x, ytop, TW, TW)
    crop = head_bbox(ct, thr=55, margin=4)
    draw_ct(ax, ct, crop)
    draw_overlays(ax, rec, kinds=('box', 'pred'), lw=1.0)
    corner_text(ax, c.replace('CQ500-', ''), 'tl', size=T7, weight='bold', bg=LBL)
    orient_marks(ax, crop, size=T7, color='#d0d0d0')
    fig_text(fig, x, ytop - 4.6, letters[i], H, color=INK, ha='left', fontweight='bold')
    fig_text(fig, x + 3.0, ytop - 4.6, ttl, H, color=INK, ha='left', fontweight='bold')
    fig_text(fig, x + 3.0, ytop - 1.4, sub, H, color=INK2, ha='left')
    v = {l: int(r[f'R1:{l}'] + r[f'R2:{l}'] + r[f'R3:{l}']) for l in ['ICH'] + list(NM)}
    maj = [f"{NM[l]} {v[l]}" for l in NM if v[l] >= 2]
    lab = ' · '.join([f"ICH {v['ICH']}"] + maj)
    p_any = float(PMS.loc[c, 'p_any'])
    fig_text(fig, x, ytop + TW + 3.6, lab, H, color=INK, ha='left')
    fig_text(fig, x, ytop + TW + 6.8, f"v2 {r.pred_volume_ml:.1f} mL · v3b {r.ANY_ml:.1f} mL", H, color=INK2, ha='left')
    fig_text(fig, x, ytop + TW + 10.0, f"public model p {pfmt(p_any)}", H, color='#6a3d9a', ha='left')
    print(letters[i], c, lab, f"v2 {r.pred_volume_ml:.1f}", f"v3b {r.ANY_ml:.1f}", f"public {p_any:.3f}", sub)
fig_text(fig, 1.6, 5.4, 'Fixed-rule examples (a–f): middle-ranked v2 volume among scans with the named pattern (upper middle for even counts)', H, color=INK, ha='left')
fig_text(fig, XS[2], ROWY[1] - 9.2, 'Disagreements between the probe and the reader majority (g, h)', H, color=INK, ha='left')
xsep = (XS[1] + TW + XS[2]) / 2
fig.add_artist(Line2D([xsep * MM / W2] * 2, [1 - (ROWY[1] - 10.5) * MM / (H * MM), 1 - (ROWY[1] + TW + 11.0) * MM / (H * MM)], color=MUTED, lw=1.0))
yl = ROWY[1] + TW + 15.0
fig.legend(handles=[Line2D([], [], color=IMG['ref'], lw=1.5, ls=(0, (3.2, 1.6)), label='BHX box on the displayed slice'),
                    Line2D([], [], color=IMG['pred'], lw=1.5, label='v2 prediction')],
           loc='upper left', bbox_to_anchor=(1.6 * MM / W2, 1 - yl * MM / (H * MM)), ncol=2, borderaxespad=0, handlelength=2.4, columnspacing=1.6)
fig_text(fig, 1.6, yl + 7.6, 'Below each image: readers (of 3) calling ICH and each reader-majority label; v2 total and v3b any-class predicted volume;', H, color=INK2, ha='left')
fig_text(fig, 1.6, yl + 10.9, 'any-hemorrhage probability of the public model. Patient right on image left.', H, color=INK2, ha='left')
print('overflow', overflow_mm(fig))
save(fig, 'Figure_4_Examples', 'main_mr')
