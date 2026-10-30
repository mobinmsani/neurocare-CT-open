"""Supplementary figure - Segmentation agreement.

a  Dice forest by reference and model (BHSD test, Seg-CQ500, MR-BHSD with the rater-rater human agreement benchmark)
b  Per-scan v2 Dice against reference volume (BHSD test), bin means with bootstrap CIs, the 50.1 mL subdural miss
c  Seg-CQ500 examples (v1 prediction)
All numbers are read from the analysis outputs (ncdata); values derived here are printed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncstyle import *
from ncdata import *
from nc_c import *
from scipy import stats

HFIG = 115
fig = plt.figure(figsize=(W2, HFIG * MM))

k, v1, nm, v2b, v3b, v3 = bhsd()
seg = {r['model']: r for r in S['seg_agreement']}
bha = {(r['model'], r['cohort']): r for r in S['bhsd_agreement']}
L3 = {(r['model'], r['cohort']): r for r in VO['bhsd_table3']}
rr = rr_pairs()

# ======================================================================= a: Dice forest
AX, AW = 41.0, 40.0
axA = axes_mm(fig, AX, 10, AW, 46)
xf = lambda x_mm: (x_mm - AX) / AW
mr_est = float(np.mean(mr().groupby('case_id').dice.mean()))
assert abs(mr_est - float(REG['mr_v1_dice_mean'])) < 5e-4
rows = [
    ('h', 'BHSD test (96 scans) · internal'),
    ('i', 'v1', 'prespecified', L3[('v1', 'all 96')]['mean_dice'], L3[('v1', 'all 96')]['lo'], L3[('v1', 'all 96')]['hi'], 'v1', True),
    ('i', 'v2-PM', 'prespecified', L3[('v2_prespecified', 'all 96')]['mean_dice'], L3[('v2_prespecified', 'all 96')]['lo'], L3[('v2_prespecified', 'all 96')]['hi'], 'v2pm', True),
    ('i', 'v2', 'post hoc', L3[('v2_nomask_posthoc', 'all 96')]['mean_dice'], L3[('v2_nomask_posthoc', 'all 96')]['lo'], L3[('v2_nomask_posthoc', 'all 96')]['hi'], 'v2', False),
    ('i', 'v3b, any class', 'post hoc', VO['bhsd_v3']['v3b']['any_dice'], *VO['bhsd_v3']['v3b']['any_ci'], 'v3b', False),
    ('h', 'Seg-CQ500 (43 scans) · transfer, not blind (used in development)'),
    ('i', 'v1', 'prespecified', seg['v1']['dice'], seg['v1']['dice_lo'], seg['v1']['dice_hi'], 'v1', True),
    ('i', 'v2-PM', 'prespecified', seg['v2 prespecified']['dice'], seg['v2 prespecified']['dice_lo'], seg['v2 prespecified']['dice_hi'], 'v2pm', True),
    ('i', 'v2', 'post hoc (locked)', seg['v2 no post-mask']['dice'], seg['v2 no post-mask']['dice_lo'], seg['v2 no post-mask']['dice_hi'], 'v2', False),
    ('i', 'BLAST-CT', 'pretrained', seg['BLAST-CT']['dice'], seg['BLAST-CT']['dice_lo'], seg['BLAST-CT']['dice_hi'], 'blast', True),
    ('h', 'MR-BHSD (43 scans, 3 raters) · transfer, RSNA-derived'),
    ('i', 'v1 vs each rater', 'prespecified', mr_est, float(REG['mr_v1_dice_lo']), float(REG['mr_v1_dice_hi']), 'v1', True),
    ('r', 'Rater vs rater', '3 pairs; gray band', float(rr.mean_dice.mean()), float(rr.mean_dice.min()), float(rr.mean_dice.max())),
]
y = 0.0
ymr = []
for r in rows:
    if r[0] == 'h':
        if y > 0:
            y += 0.4
        axA.text(xf(2.0), y, r[1], transform=axA.get_yaxis_transform(), fontsize=7, fontweight='bold', color=INK, va='center', ha='left')
        if r[1].startswith('MR-BHSD'):
            ymr.append(y)
        y += 1
        continue
    if r[0] == 'r':
        _, lab, tag, est, lo, hi = r
        ymr.append(y)
        axA.fill_betweenx([ymr[0] + 0.5, y + 0.5], lo, hi, color=CEIL, lw=0, zorder=0)
        ci_point(axA, est, y, lo, hi, INK, marker='D', ms=4.4)
    else:
        _, lab, tag, est, lo, hi, mk, filled = r
        ci_point(axA, est, y, lo, hi, M[mk], marker=MK[mk], filled=filled, ms=4.6)
    axA.text(xf(2.0), y, lab, transform=axA.get_yaxis_transform(), fontsize=7, color=INK, va='center', ha='left')
    axA.text(xf(22.5), y, tag, transform=axA.get_yaxis_transform(), fontsize=7, color=INK2, va='center', ha='left')
    axA.text(hi + 0.025, y, f'{est:.2f}', fontsize=7, color=INK, va='center', ha='left')
    y += 1
axA.set_ylim(y - 0.4, -0.6)
axA.set_yticks([]); axA.spines['left'].set_visible(False)
axA.set_xlim(0, 1.0); axA.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ['0', '0.25', '0.50', '0.75', '1'])
axA.set_xlabel('Mean Dice')
axA.text(1.0, 0, 'bars: 95% CI', fontsize=7, color=INK2, ha='right', va='center')
grid(axA, 'x')
ptitle(fig, axA, 'a', 'Voxel overlap by reference and model', dx_mm=AX - 2.0)
print('a: rater-rater mean %.3f range %.3f-%.3f' % (rr.mean_dice.mean(), rr.mean_dice.min(), rr.mean_dice.max()))

# ======================================================================= b: Dice vs reference volume (v2, BHSD test)
axB = axes_mm(fig, 100, 10, 77, 46)
x = nm.ref_ml.values; yv = nm.dice_nopostmask.values
for xv in (10, 30):
    axB.axvline(xv, color='#b9b5ad', lw=1.0, ls=(0, (2.5, 2)), zorder=1)
z = yv == 0
axB.scatter(x[~z], yv[~z], s=13, marker=MK['v2'], color=M['v2'], alpha=0.8, lw=0.6, edgecolor=WHITE, zorder=3)
axB.scatter(x[z], yv[z], s=17, marker='x', color=INK, lw=1.0, zorder=4)
bins_ = [('<10 mL', x < 10), ('10 to\n<30 mL', (x >= 10) & (x < 30)), ('≥30 mL', x >= 30)]
st = {r['stratum']: r for r in VO['strata_table5']}
YT = 1.47
for (lab, sel), key, xt in zip(bins_, ['<10', '10-<30', '>=30'], [0.9, 17.3, 88]):
    est, lo, hi = boot_mean_ci(yv[sel])
    assert abs(est - st[key]['v2']) < 1e-9 and sel.sum() == st[key]['n']
    xc = float(np.exp(np.mean(np.log(x[sel]))))
    ci_point(axB, xc, est, lo, hi, INK, marker='D', ms=5.0, orient='v', z=6)
    axB.text(xc * 1.13, est, f'{est:.2f}', fontsize=7, ha='left', va='center', color=INK, fontweight='bold', path_effects=halo(2.0), zorder=7)
    nz = int((yv[sel] == 0).sum())
    nl = lab.count('\n')
    axB.text(xt, YT, lab, fontsize=7, ha='center', va='top', color=INK, fontweight='bold', linespacing=1.0)
    axB.text(xt, YT - 0.105 * (nl + 1), f'n = {int(sel.sum())}', fontsize=7, ha='center', va='top', color=INK2)
    print('b: bin %s mean %.3f (%.3f-%.3f) n=%d zero=%d' % (lab.replace(chr(10), ' '), est, lo, hi, sel.sum(), nz))
big = R3['bhsd_largest_miss']
bx, by = big['ref_ml'], 0.0
assert abs(nm.loc[big['case'], 'ref_ml'] - bx) < 1e-6 and nm.loc[big['case'], 'dice_nopostmask'] == 0
v1big = float(v1.loc[big['case'], 'dice'])
assert v1big == 0.0
axB.text(bx * 1.16, 0.015, f"{bx:.1f} mL SDH\nmissed by\nv2 and v1", fontsize=7, ha='left', va='bottom', color=INK, linespacing=1.05, zorder=7)
axB.set_xscale('log'); axB.set_xlim(0.08, 260); axB.set_ylim(-0.05, 1.50)
plain_log(axB, 'x', [0.1, 1, 10, 30, 100], ['0.1', '1', '10', '30', '100'])
axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ['0', '0.25', '0.50', '0.75', '1'])
axB.set_xlabel('Reference hemorrhage volume (mL, log scale)'); axB.set_ylabel('Dice per scan (v2)')
rho = S['size_spearman_v2 no post-mask']
axB.legend(handles=legend_handles([('scan', dict(color=M['v2'], marker=MK['v2'], ms=4.4)),
                                   (f'Dice 0 (n = {int(z.sum())})', dict(color=INK, marker='x', ms=4.6, filled=False)),
                                   ('bin mean (95% CI)', dict(color=INK, marker='D', ms=4.6))]),
           loc='upper left', bbox_to_anchor=(0.085, 1.02), bbox_transform=axB.transData, fontsize=7, handletextpad=0.3, labelspacing=0.3, borderaxespad=0.0)
axB.text(0.088, 0.60, minus(f'Spearman ρ = {rho:.2f}'), fontsize=7, color=INK, va='top', ha='left')
axB.axhline(1.13, color=GRID, lw=1.0, zorder=0)
grid(axB, 'y')
ptitle(fig, axB, 'b', 'Dice by hemorrhage volume (BHSD test, v2)', dx_mm=10.0)
print('b: zero-overlap scans', int(z.sum()), 'below 10 mL', int((z & (x < 10)).sum()), '; largest miss', big)

# ======================================================================= c: Seg-CQ500 examples (v1 prediction)
segm = pd.read_csv(T / 'SegCQ500_four_method_comparison.csv')
sv1 = segm[segm.method == 'lightunet25d'].set_index('case_id')
nmS = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/segcq500_no_postmask.csv').set_index('case_id')
cases = [('seg_1', 'CQ500-CT-55'), ('seg_2', 'CQ500-CT-243'), ('seg_3', 'CQ500-CT-208')]
TS, TY, GAP = 31.0, 76.0, 3.0
xs = [2.0 + i * (TS + GAP) for i in range(4)]
tiles = []
zoom_box = None
for i, (key, cid) in enumerate(cases):
    ax = axes_mm(fig, xs[i], TY, TS, TS)
    rec = json.load(open(ASSETS / f'{key}.json'))
    assert rec['case'] == cid
    ct, crop = ct_tile(fig, ax, key, crop_thr=60, margin=5)
    e = load_npy(ASSETS / f'{key}_errors.npy')
    refm = (e == 1) | (e == 3); prm = (e == 1) | (e == 2)
    if refm.any():
        ax.contour(refm.astype(float), [0.5], colors=[IMG['ref']], linewidths=1.0, zorder=5)
    if prm.any():
        ax.contour(prm.astype(float), [0.5], colors=[IMG['pred']], linewidths=1.0, linestyles=[(0, (2.2, 1.2))], zorder=6)
    tile_label(ax, cid.replace('CQ500-', ''))
    orient_marks(ax, crop, size=7.0, color='#d9d9d9')
    d1 = sv1.loc[cid, 'dice']; d2 = nmS.loc[cid, 'dice_nopostmask']
    rv = sv1.loc[cid, 'reference_volume_ml']; pv = sv1.loc[cid, 'pred_volume_ml']
    fig_text(fig, xs[i], TY + TS + 3.4, f'Dice v1 {d1:.2f}, v2 {d2:.2f}', color=INK)
    fig_text(fig, xs[i], TY + TS + 6.6, f'reference {rv:.1f}, v1 {pv:.1f} mL', color=INK2)
    print('e:', cid, 'v1 dice %.3f v2 dice %.3f ref %.2f v1 %.2f' % (d1, d2, rv, pv))
    tiles.append(ax)
    if key == 'seg_3':
        ys_, xs_ = np.nonzero(refm | prm) if (refm | prm).any() else (np.array([e.shape[0] // 2]), np.array([e.shape[1] // 2]))   # blank if not redistributed
        cx, cy = (xs_.min() + xs_.max()) / 2, (ys_.min() + ys_.max()) / 2
        half = max(xs_.max() - xs_.min(), ys_.max() - ys_.min()) / 2 + 9
        zoom_box = (cx - half, cx + half, cy - half, cy + half)
        ax.add_patch(Rectangle((zoom_box[0], zoom_box[2]), 2 * half, 2 * half, fill=False, ec=WHITE, lw=1.0, zorder=8))
        zct, zcrop, zref, zpr, zfull = ct, zoom_box, refm, prm, crop
# zoom tile for CT-208
axZ = axes_mm(fig, xs[3], TY, TS, TS)
draw_ct(axZ, zct, zcrop, interp='nearest')
axZ.contour(zref.astype(float), [0.5], colors=[IMG['ref']], linewidths=1.3, zorder=5)
axZ.contour(zpr.astype(float), [0.5], colors=[IMG['pred']], linewidths=1.3, linestyles=[(0, (2.2, 1.2))], zorder=6)
for s_ in axZ.spines.values():
    s_.set_visible(True); s_.set_edgecolor(INK); s_.set_linewidth(1.0)
zf = (zfull[1] - zfull[0]) / (zcrop[1] - zcrop[0])
tile_label(axZ, f'CT-208 ×{zf:.0f}')
fig_text(fig, xs[3], TY + TS + 3.4, 'zoom of the framed region', color=INK)
fig_text(fig, xs[3], TY + TS + 6.6, 'no overlap', color=INK2)
ptitle(fig, tiles[0], 'c', 'Seg-CQ500 examples: v1 prediction on the slice with the largest reference area', dx_mm=0.0)
# key
KX = xs[3] + TS + 4.0
lg = axes_mm(fig, KX, TY, 180 - KX - 0.5, TS + 7)
lg.axis('off'); lg.set_xlim(0, 1); lg.set_ylim(1, 0)
lg.plot([0.0, 0.14], [0.05, 0.05], color=IMG['ref'], lw=2.2, solid_capstyle='butt')
lg.text(0.19, 0.05, 'Seg-CQ500 reference (solid)', fontsize=7, va='center')
lg.plot([0.0, 0.14], [0.155, 0.155], color=IMG['pred'], lw=2.2, ls=(0, (2.2, 1.2)), dash_capstyle='butt')
lg.text(0.19, 0.155, 'v1 prediction (dashed)', fontsize=7, va='center')
lg.text(0.0, 0.25, 'Dice is per scan.\nR and L: patient right and left.',
        fontsize=7, color=INK2, va='top', linespacing=1.18)

print('overflow', overflow_mm(fig))
save(fig, 'SF_segmentation', 'supp_mr')
