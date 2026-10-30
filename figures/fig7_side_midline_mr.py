"""Figure 7 - Hemorrhage side can be validated with open references; midline shift can be ranked but not measured.

a  PHE-SICH-CT-IDS: expected (ideal) midline, expert perihematomal edema reference and v2 hemorrhage prediction in three patients
b  Side accuracy (Wilson 95% CI) with coverage: PHE-SICH-CT-IDS documented side and CQ500 reader-majority side
c  Schematic of how the CSF-offset distance is computed (not a midline-shift measurement)
d  CQ500 examples with the estimated ideal midline and the v2 hemorrhage contour (illustrative slices)
e  CSF-offset distance by the number of readers calling midline shift >5 mm (every computable scan shown)
All numbers are read from the
analysis outputs (ncdata); values derived here are printed and checked.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nc_s2 import *                      # ncstyle, ncdata, key_row, strip_text, tag, nminus (fig_text with figure height)
import nc_c
from matplotlib.patches import Ellipse, Polygon, FancyArrowPatch
from matplotlib.patches import Rectangle as Rect

H = 218
fig = plt.figure(figsize=(W2, H * MM))


def ft(x_mm, y_mm, s, **kw):
    """Figure text, baseline-aligned, ink by default (mm from the top-left corner)."""
    return nc_c.fig_text(fig, x_mm, y_mm, s, **kw)


m = load_cq500()
mm_ = m.set_index('name')
PR = N['proxy']

# ======================================================================= a: PHE-SICH-CT-IDS examples (expected midline)
ph = pd.read_csv(E / 'neurocare_v2_20261021/phe_sich/phe_laterality_midline.csv', dtype={'case': str}).set_index('case')
mid = V2R['midline']
assert len(ph) == 120 and (ph.midline_status == 'estimated').sum() == 120 == mid['estimated']
LS_MID = (0, (5.0, 1.6, 1.2, 1.6))      # expected midline: dash-dot (magenta)
LS_PRED = (0, (2.4, 1.3))               # model prediction: dashed (azure); expert reference: solid (orange)
SIDE = {'L': 'left', 'R': 'right'}


def draw_phe(ax, rec, msk, lw=1.0):
    for o in rec['overlays']:
        if o['kind'] == 'midline':
            ln = midline_centerline(o['subpaths'])
            ax.plot(ln[:, 0], ln[:, 1], color=IMG['midline'], lw=1.2, ls=LS_MID, zorder=4, dash_capstyle='butt')
        elif o['kind'] == 'edema':
            for sp in o['subpaths']:
                sp = np.asarray(sp)
                ax.plot(sp[:, 0], sp[:, 1], color=IMG['edema'], lw=lw, ls='-', zorder=5, solid_joinstyle='round')
    ax.contour(msk.astype(float), [0.5], colors=[IMG['pred']], linewidths=lw, linestyles=[LS_PRED], zorder=6)


xw, gap, x0, ytop = 56.0, 4.0, 2.0, 22.5
axs = []
for i, key in enumerate(['phe_01', 'phe_02', 'phe_03']):
    rec, ct = load_asset(key)
    msk = load_npy(ASSETS / f'{key}_mask.npy')
    cid = rec['case'].replace('PHE-', '')
    ax = axes_mm(fig, x0 + i * (xw + gap), ytop, xw, xw)
    crop = head_bbox(ct, thr=55, margin=8)
    draw_ct(ax, ct, crop)
    draw_phe(ax, rec, msk)
    t0, t1 = rec['titles']
    doc = t0.split('documented ')[1].split(',')[0].strip(); auto = t0.split('edema auto ')[1].strip()
    assert doc == ph.loc[cid, 'documented_side'] and auto == ph.loc[cid, 'edema_side_auto']
    r = ph.loc[cid]
    atb_ = abs(r.midline_angle_deg) >= 20
    l3 = f'Mean symmetry Dice {r.mean_symmetry_dice:.3f}'
    l4 = f'Angle {r.midline_angle_deg:+.1f}°' + (' (search bound)' if atb_ else '') + f'; {int(r.n_reliable_slices)} reliable slices'
    strip_text(ax, f'PHE-SICH-CT-IDS {cid}', 'above', pad_mm=11.8, weight='bold')
    strip_text(ax, f'Documented {SIDE[doc]}; edema-mask side {SIDE[auto]}', 'above', pad_mm=8.2)
    strip_text(ax, nminus(l3), 'above', pad_mm=4.6, color=INK2)
    strip_text(ax, nminus(l4), 'above', pad_mm=1.0, color=INK2)
    orient_marks(ax, crop, size=7.0, color='#d9d9d9')
    axs.append(ax)
    if i == 0:          # lesion inset for the smallest example (no smoothing; same contours, magnified)
        ys, xs = np.nonzero(msk) if msk.any() else (np.array([msk.shape[0] // 2]), np.array([msk.shape[1] // 2]))
        cx, cy, half = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, 30
        zx0, zx1, zy0, zy1 = cx - half, cx + half, cy - half, cy + half
        ax.add_patch(Rect((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec=WHITE, lw=1.0, zorder=7))
        axi = axes_mm(fig, x0 + xw - 21.0, ytop + 1.0, 20.0, 20.0)
        draw_ct(axi, ct, (zx0, zx1, zy0, zy1))
        draw_phe(axi, rec, msk, lw=1.2)
        for s_ in axi.spines.values():
            s_.set_visible(True); s_.set_color(WHITE); s_.set_linewidth(1.0)
    print('a:', cid, 'documented', doc, 'edema side', auto, 'symmetry %.3f angle %.1f slices %d' % (r.mean_symmetry_dice, r.midline_angle_deg, r.n_reliable_slices))
ptitle(fig, axs[0], 'a', 'Expected midline and hemorrhage side in PHE-SICH-CT-IDS', dx_mm=0.0, dy_mm=16.6)
ky = ytop + xw + 4.0
key_row(fig, x0, ky, [('line', IMG['midline'], 'Expected (ideal) midline', dict(ls=LS_MID, lw=1.2)), ('line', IMG['edema'], 'Expert perihematomal edema reference', dict(ls='-', lw=1.2)),
                      ('line', IMG['pred'], 'v2 hemorrhage prediction', dict(ls=LS_PRED, lw=1.2))], H, sw=7.0)
fig_text(fig, x0, ky + 3.0, 'One slice per patient; side and midline use all reliable slices. Inset (left image): lesion area magnified, same contours. '
         'Patient right on image left.', H)

# ======================================================================= b: laterality with coverage
YB = 102.0
AX, AW = 25.0, 24.0
axA = axes_mm(fig, AX, YB, AW, 36)
xa = lambda x_mm: (x_mm - AX) / AW
lat = CQ['laterality']
phe = {o['Output']: o for o in V2R['lat']}
e_, v1_, v2_ = S['phe_lat_edema'], S['phe_lat_v1'], phe['v2-nomask hemorrhage + auto midline']
n_phe = int(S['phe_unilateral_geo'])
assert e_['evaluable'] + e_['abstained'] == v1_['evaluable'] + v1_['abstained'] == v2_['Evaluable patients'] + v2_['Abstained/no mask'] == n_phe
rows = [
    ('h', f'PHE-SICH-CT-IDS · documented side ({n_phe} patients)'),
    ('i', 'v2', v2_['Correct side'], v2_['Evaluable patients'], n_phe, 'v2'),
    ('i', 'v1', v1_['correct'], v1_['evaluable'], n_phe, 'v1'),
    ('i', 'Expert edema masks', e_['correct'], e_['evaluable'], n_phe, 'edema'),
    ('h', f"CQ500 · reader-majority side ({lat['unilateral_ich']} scans)"),
    ('i', 'v2', lat['correct'], lat['evaluable'], lat['unilateral_ich'], 'v2'),
]
C1, C2 = 51.5, 71.0
y = 0.0
for r in rows:
    if r[0] == 'h':
        if y > 0:
            y += 0.35
        axA.text(xa(2.0), y, r[1], transform=axA.get_yaxis_transform(), fontsize=7, fontweight='bold', va='center', ha='left', color=INK)
        y += 1
        continue
    _, lab, k_, n_, el, mk = r
    p = k_ / n_; lo, hi = wilson(k_, n_)
    ci_point(axA, p, y, lo, hi, M[mk], marker=MK[mk], ms=4.6)
    axA.text(xa(2.0), y, lab, transform=axA.get_yaxis_transform(), fontsize=7, va='center', ha='left', color=INK)
    axA.text(xa(C1), y, f'{100 * p:.1f}% ({k_}/{n_})', transform=axA.get_yaxis_transform(), fontsize=7, va='center', ha='left', color=INK)
    axA.text(xa(C2), y, f'{n_}/{el}', transform=axA.get_yaxis_transform(), fontsize=7, va='center', ha='left', color=INK2)
    print('b: %s %d/%d = %.3f (%.3f-%.3f); assigned %d/%d' % (lab, k_, n_, p, lo, hi, n_, el))
    y += 1
axA.text(xa(C1), -1.05, 'correct/assigned', transform=axA.get_yaxis_transform(), fontsize=7, va='center', ha='left', color=INK2)
axA.text(xa(C2), -1.05, 'assigned/\neligible', transform=axA.get_yaxis_transform(), fontsize=7, va='center', ha='left', color=INK2, linespacing=1.0)
axA.set_ylim(y - 0.45, -1.6)
axA.set_yticks([]); axA.spines['left'].set_visible(False)
axA.set_xlim(0.8, 1.0); axA.set_xticks([0.8, 0.9, 1.0], ['0.8', '0.9', '1'])
axA.set_xlabel('Correct side (95% CI)')
grid(axA, 'x')
ptitle(fig, axA, 'b', 'Hemorrhage side', dx_mm=AX - 2.0)

# ======================================================================= c: schematic of the distance
DXs, DYs, DWs = 98.0, YB, 30.0
axD = axes_mm(fig, DXs, DYs, DWs, 36)
axD.set_xlim(-1.0, 1.0); axD.set_ylim(-1.18, 1.07); axD.set_aspect('equal'); axD.axis('off')
axD.add_patch(Ellipse((0, 0), 1.62, 2.02, fc='#f4f2ee', ec=INK, lw=1.4, zorder=1))
axD.add_patch(Ellipse((-0.40, 0.22), 0.40, 0.58, angle=12, fc='#8c8c91', ec=INK, lw=1.0, zorder=3))
CSF = '#bcd7ee'
vent = [Polygon([[0.10, 0.50], [0.30, 0.46], [0.36, 0.20], [0.30, -0.05], [0.19, 0.04], [0.18, 0.28], [0.08, 0.38]], closed=True),
        Polygon([[0.38, 0.52], [0.56, 0.46], [0.60, 0.20], [0.52, -0.02], [0.43, 0.08], [0.44, 0.30], [0.37, 0.40]], closed=True)]
for pv in vent:
    pv.set_facecolor(CSF); pv.set_edgecolor(INK); pv.set_linewidth(1.0); pv.set_zorder(3)
    axD.add_patch(pv)
for (cx_, cy_, w_, h_) in [(0.60, -0.70, 0.14, 0.07), (0.54, 0.74, 0.12, 0.06), (-0.22, -0.72, 0.12, 0.06), (0.66, -0.22, 0.08, 0.12)]:
    axD.add_patch(Ellipse((cx_, cy_), w_, h_, fc=CSF, ec=INK, lw=1.0, zorder=3))
axD.plot([0, 0], [-1.01, 1.01], color=IMG['midline'], lw=1.5, ls=LS_MID, zorder=4)
cxx, cyy = 0.36, 0.18
axD.plot([cxx], [cyy], 'o', ms=4.4, mfc=INK, mec=WHITE, mew=0.6, zorder=6)
YD = -0.40
axD.plot([cxx, cxx], [cyy - 0.06, YD - 0.04], color=INK, lw=1.0, ls=(0, (1.2, 1.2)), zorder=5)
axD.add_patch(FancyArrowPatch((0.0, YD), (cxx, YD), arrowstyle='<|-|>', mutation_scale=6, lw=1.0, color=INK, zorder=7, shrinkA=0, shrinkB=0))
axD.text(cxx / 2 + 0.09, YD - 0.06, 'offset', fontsize=7, ha='center', va='top', color=INK, zorder=8)
axD.text(-0.40, 0.22 - 0.36, 'mass', fontsize=7, ha='center', va='top', color=INK, zorder=8)
axD.text(-0.98, 1.05, 'R', fontsize=7, fontweight='bold', color=INK2, ha='left', va='top')
axD.text(0.98, 1.05, 'L', fontsize=7, fontweight='bold', color=INK2, ha='right', va='top')
ptitle(fig, axD, 'c', 'How the CSF-offset distance is computed', dx_mm=0.0, dy_mm=3.2)
SX = DXs + DWs + 2.0
steps = [
    ('1', 'Ideal midline (pink, dash-dot):', 'mirror line maximizing symmetry'),
    ('2', 'Voxels of 0–15 HU (blue) in the', 'eroded intracranial region'),
    ('3', 'Offset of their centroid (dot)', 'from the midline, on each slice'),
    ('4', 'Distance: largest absolute', 'offset across reliable slices'),
    ('5', 'Septum pellucidum not located;', 'edema or infarct voxels count'),
]
yy = DYs + 2.0
for num, *ls_ in steps:
    ft(SX, yy, num, fontweight='bold')
    for j, l_ in enumerate(ls_):
        ft(SX + 3.0, yy + j * 3.0, l_, color=INK if j == 0 else INK2)
    yy += len(ls_) * 3.0 + 0.9

# ======================================================================= d: CQ500 examples (illustrative slices)
# CT-488, the scan with the largest distance overall (shift positive), is the IVH example of Fig. 4e and is not repeated here
ex = [('mid_02', 'Near the median of', 'shift-positive scans'), ('mid_04', 'Largest distance among', 'shift-negative scans')]
TS, TY, GAP = 38.0, 160.0, 4.0
X0E = 2.0
tiles = []
ok = np.isfinite(m.csf_offset_maxabs_mm.values)
mp = m[ok]
assert ok.sum() == PR['n']
kv = votes(mp, 'MidlineShift')
prox = mp.csf_offset_maxabs_mm.values
neg = kv <= 1
for i, (key, lab1, lab2) in enumerate(ex):
    rec, ct = load_asset(key)
    x0_ = X0E + i * (TS + GAP)
    ax = axes_mm(fig, x0_, TY, TS, TS)
    crop = head_bbox(ct, thr=55, margin=3)
    draw_ct(ax, ct, crop)
    for o in rec['overlays']:
        if o['kind'] == 'midline':
            ln = midline_centerline(o['subpaths'])
            ax.plot(ln[:, 0], ln[:, 1], color=IMG['midline'], lw=1.3, ls=LS_MID, zorder=4, dash_capstyle='butt')
        elif o['kind'] == 'pred':
            for sp in o['subpaths']:
                sp = np.asarray(sp)
                ax.plot(sp[:, 0], sp[:, 1], color=IMG['pred'], lw=1.2, ls=(0, (2.2, 1.2)), zorder=5)
    row = mm_.loc[rec['case']]
    km = int(sum(row[f'R{q}:MidlineShift'] for q in (1, 2, 3))); ki = int(sum(row[f'R{q}:ICH'] for q in (1, 2, 3)))
    nc_c.tile_label(ax, rec['case'].replace('CQ500-', ''))
    orient_marks(ax, crop, size=7.0, color='#d9d9d9')
    ft(x0_, TY - 4.6, lab1, fontweight='bold')
    ft(x0_, TY - 1.6, lab2, fontweight='bold')
    ft(x0_, TY + TS + 3.4, f"distance {row.csf_offset_maxabs_mm:.1f} mm", color=INK)
    ft(x0_, TY + TS + 6.6, f"v2 {row.pred_volume_ml:.1f} mL", color=INK)
    ft(x0_, TY + TS + 9.8, f"shift {km}/3, ICH {ki}/3", color=INK2)
    print('d:', rec['case'], 'distance %.2f shift %d/3 ICH %d/3 v2 %.2f' % (row.csf_offset_maxabs_mm, km, ki, row.pred_volume_ml))
    tiles.append(ax)
assert abs(mm_.loc['CQ500-CT-2', 'csf_offset_maxabs_mm'] - prox[neg].max()) < 1e-9 and abs(mm_.loc['CQ500-CT-488', 'csf_offset_maxabs_mm'] - prox.max()) < 1e-9
print('d: largest distance overall CT-488 %.1f mm (shown in Fig. 4e)' % mm_.loc['CQ500-CT-488', 'csf_offset_maxabs_mm'])
ptitle(fig, tiles[0], 'd', 'CQ500 examples (illustrative slices)', dx_mm=0.0, dy_mm=9.0)
yk = TY + TS + 14.0
lg = axes_mm(fig, X0E, yk - 1.6, 8, 3.2); lg.axis('off'); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
lg.plot([0, 1], [0.5, 0.5], color=IMG['midline'], lw=2.0, ls=LS_MID)
ft(X0E + 9.5, yk + 1.0, 'ideal midline')
lg = axes_mm(fig, X0E + 31.0, yk - 1.6, 8, 3.2); lg.axis('off'); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
lg.plot([0, 1], [0.5, 0.5], color=IMG['pred'], lw=2.0, ls=(0, (2.2, 1.2)))
ft(X0E + 40.5, yk + 1.0, 'v2 hemorrhage prediction')

# ======================================================================= e: CSF-offset distance by reader votes
EX, EW = 117.0, 58.0
axE = axes_mm(fig, EX, TY, EW, 34)
rng = np.random.default_rng(20261028)
axE.axvspan(-0.5, 1.5, color=BAND, lw=0, zorder=0)
for kk in range(4):
    v = prox[kv == kk]
    assert len(v) == [b['n'] for b in PR['by_k'] if b['k'] == kk][0]
    xj = kk + rng.uniform(-0.27, 0.27, len(v))
    axE.scatter(xj, v, s=6, color=M['rule'], alpha=0.45 if kk < 2 else 0.7, lw=0, zorder=2)
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    axE.add_patch(Rectangle((kk - 0.3, q1), 0.6, q3 - q1, fill=False, ec=INK, lw=1.0, zorder=4))
    axE.plot([kk - 0.3, kk + 0.3], [q2, q2], color=INK, lw=1.8, zorder=5, solid_capstyle='butt')
    axE.text(kk + 0.33, q2, f'{q2:.1f}', fontsize=7, va='center', ha='left', color=INK, path_effects=halo(2.0), zorder=6)
axE.axhline(5, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=3)
axE.text(3.47, 5.6, '5 mm', fontsize=7, color=INK, ha='right', va='bottom', path_effects=halo(2.0), zorder=6)
med_neg = float(np.median(prox[neg]))
assert abs(med_neg - PR['median_neg']) < 1e-9 and int((prox[neg] > 5).sum()) == PR['neg_over5'] and int(neg.sum()) == PR['neg_n']
_lk = re.match(r'^([\d.]+) \(([\d.]+)–([\d.]+)\)$', REG['cq_MidlineShift_max']['auc'])   # locked CQ500 analysis, as in the text
assert _lk and abs(float(_lk.group(1)) - PR['auc_all'][0]) < 5e-4
axE.text(-0.44, 45.3, f"Majority negative: median {med_neg:.1f} mm;\n{PR['neg_over5']} of {PR['neg_n']} above 5 mm", fontsize=7, ha='left', va='top', color=INK, linespacing=1.1)
axE.set_xlim(-0.5, 3.5); axE.set_ylim(0, 46)
axE.set_xticks(range(4), [f'{kk} of 3\n({int((kv == kk).sum())})' for kk in range(4)])
axE.set_xlabel('Readers calling midline shift >5 mm (scans)')
axE.set_ylabel('CSF-offset distance (mm)')
axE.set_yticks([0, 5, 10, 20, 30, 40])
grid(axE, 'y')
ptitle(fig, axE, 'e', 'CSF-offset distance is not midline shift', sub=f"CQ500, {PR['n']} scans; AUC {_lk.group(1)} ({_lk.group(2)}–{_lk.group(3)})", dx_mm=9.0)
print('e: median by k', [(b['k'], b['n'], round(b['median'], 2)) for b in PR['by_k']], 'neg median %.2f, >5 %d/%d, max %.1f' % (med_neg, PR['neg_over5'], PR['neg_n'], prox.max()))

print('overflow', overflow_mm(fig))
save(fig, 'Figure_7_Side_Midline', 'main_mr')
