"""SF_segcq500_methods - Seg-CQ500: Dice across methods (pretrained BLAST-CT, two post-processing variants, v1, v2-PM, v2)
and the precision-recall balance (the voxel-error examples are in SF_segcq500_errors)."""
from nc_s2 import *

H = 116
fig = plt.figure(figsize=(W2, H * MM))
seg4 = pd.read_csv(T / 'SegCQ500_four_method_comparison.csv')
segnm = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/segcq500_no_postmask.csv').set_index('case_id')
v2m = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/v2_metrics.csv')
v2s = v2m[v2m.dataset == 'Seg-CQ500'].set_index('case_id')
ks = sorted(segnm.index)
assert len(ks) == 43 and set(ks) == set(v2s.index)
R = segnm.loc[ks, 'ref_ml'].values
TEAL = dict(base=M['blast'], light='#6dbdb0', dark='#1d6e64')
meth = []   # key, label, colour, marker, dice, pred
for mth, lab, col, mk in [('blast_model1', 'BLAST-CT', TEAL['base'], 'D'), ('blast_min100mm3', 'BLAST-CT + size filter', TEAL['light'], 'v'),
                          ('blast_hu45_r2', 'BLAST-CT + HU growth', TEAL['dark'], 'P'), ('lightunet25d', 'v1', M['v1'], MK['v1'])]:
    g = seg4[seg4.method == mth].set_index('case_id').loc[ks]
    assert np.allclose(g.reference_volume_ml.values, R, rtol=1e-6, atol=1e-6)
    meth.append((mth, lab, col, mk, g.dice.values, g.prediction_volume_ml.fillna(g.pred_volume_ml).values))
meth.append(('v2pm', 'v2-PM', M['v2pm'], MK['v2pm'], v2s.loc[ks, 'dice'].values, v2s.loc[ks, 'pred_ml'].values))
meth.append(('v2', 'v2', M['v2'], MK['v2'], segnm.loc[ks, 'dice_nopostmask'].values, segnm.loc[ks, 'pred_ml'].values))
SA = {r['model']: r for r in S['seg_agreement']}
skey = {'blast_model1': 'BLAST-CT', 'blast_min100mm3': 'BLAST-CT + size filter', 'blast_hu45_r2': 'BLAST-CT + HU growth', 'lightunet25d': 'v1',
        'v2pm': 'v2 prespecified', 'v2': 'v2 no post-mask'}
for key, lab, col, mk, d, p in meth:
    assert abs(d.mean() - SA[skey[key]]['dice']) < 1e-9, key
assert f"{SA['BLAST-CT']['dice']:.3f} ({SA['BLAST-CT']['dice_lo']:.3f}–{SA['BLAST-CT']['dice_hi']:.3f})" == '0.531 (0.433–0.626)'
dv = S['seg_v2nm_minus_blast']
assert f"{dv['p']:.2f}" == '0.26'
pr4 = {p_['cmp']: p_ for p_ in VO['pairs_table4']}['v2_nomask-v1 SegCQ500']
assert f"{pr4['p']:.2f}" == '0.22'

# ======================================================================= a: per-scan Dice by method
axA = axes_mm(fig, 15, 17, 98, 50)
xpos = [0, 1, 2, 3.45, 4.45, 5.45]
rng = np.random.default_rng(7)
for (key, lab, col, mk, d, p), x in zip(meth, xpos):
    xj = x - 0.14 + rng.uniform(-0.15, 0.15, len(d))          # horizontal jitter only; Dice values unchanged
    axA.scatter(xj, d, s=9, color=col, marker=mk, alpha=0.75, lw=0, zorder=3)
    r = SA[skey[key]]
    ci_point(axA, x + 0.27, r['dice'], r['dice_lo'], r['dice_hi'], col, marker=mk, orient='v', ms=5.0, lw=1.4)
    axA.text(x + 0.27, r['dice_hi'] + 0.03, f"{r['dice']:.3f}", fontsize=F7, ha='center', va='bottom', color=INK)
axA.set_xlim(-0.55, 6.0); axA.set_ylim(-0.04, 1.12)
axA.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
axA.set_xticks(xpos, ['BLAST-CT', '+ size\nfilter', '+ HU\ngrowth', 'v1', 'v2-PM', 'v2'])
axA.set_ylabel('Dice per scan (43 scans)')
grid(axA, 'y')
axA.spines['left'].set_bounds(0, 1.0)
for x0_, x1_, s_ in [(-0.35, 2.4, 'Pretrained comparator and variants'), (3.1, 5.8, 'Trained on 92 BHSD training scans')]:
    axA.plot([x0_, x1_], [-0.25, -0.25], transform=axA.get_xaxis_transform(), color=MUTED, lw=1.0, clip_on=False)
    axA.text((x0_ + x1_) / 2, -0.28, s_, transform=axA.get_xaxis_transform(), fontsize=F7, color=INK2, ha='center', va='top')
ptitle(fig, axA, 'a', 'Dice across methods', sub='Points: scans; markers and bars: mean and bootstrap 95% CI', dx_mm=12.5, dy_mm=3.2)
res = [nminus(f"v2 − v1: {pr4['mean']:+.3f} (95% CI {pr4['lo']:+.3f} to {pr4['hi']:+.3f}); Wilcoxon {fmt_p(pr4['p'])}"),
       nminus(f"v2 − BLAST-CT: {dv['mean']:+.3f} (95% CI {dv['ci'][0]:+.3f} to {dv['ci'][1]:+.3f}); Wilcoxon {fmt_p(dv['p'])}"),
       'Paired differences; no equivalence margin was specified, so overlap does not establish equivalence.']
for i_, t in enumerate(res):
    fig_text(fig, 3.0, 87.0 + i_ * 3.4, t, H, color=INK if i_ < 2 else INK2)

# ======================================================================= b: precision vs recall (equal ranges, equal aspect)
axB = axes_mm(fig, 130, 17, 46, 46)
LO_, HI_ = 0.25, 0.95
axB.plot([LO_, HI_], [LO_, HI_], color='#cfcbc3', lw=1.0, ls=(0, (3, 2)), zorder=1)
PRV = {}
for key, lab, col, mk, d, p in meth:
    tp = d * (p + R) / 2
    prec = np.where(p > 0, tp / np.where(p > 0, p, 1), np.nan)    # undefined when a method predicts nothing in a scan
    rec = tp / R
    mp, plo, phi = boot_mean_ci(prec)
    mr_, rlo, rhi = boot_mean_ci(rec, seed=20261027)
    npc = int(np.isfinite(prec).sum())
    PRV[lab] = (mp, plo, phi, mr_, rlo, rhi, npc)
    axB.plot([rlo, rhi], [mp, mp], color=col, lw=1.0, zorder=3)
    axB.plot([mr_, mr_], [plo, phi], color=col, lw=1.0, zorder=3)
    axB.plot([mr_], [mp], marker=mk, ms=5.2, mfc=col, mec=WHITE, mew=0.6, ls='', zorder=4)
axB.set_xlim(LO_, HI_); axB.set_ylim(LO_, HI_); axB.set_aspect('equal')
tk = [0.3, 0.5, 0.7, 0.9]
axB.set_xticks(tk); axB.set_yticks(tk)
axB.set_xlabel('Voxel recall (mean)'); axB.set_ylabel('Voxel precision (mean)')
axB.text(0.27, 0.93, 'precision > recall', fontsize=F7, color=MUTED, ha='left', va='top')
axB.text(0.93, 0.27, 'recall > precision', fontsize=F7, color=MUTED, ha='right', va='bottom')
grid(axB, 'both')
ptitle(fig, axB, 'b', 'Precision–recall balance', sub='Marginal 95% CIs (bootstrap)', dx_mm=12.0, dy_mm=3.2)
# method key with the number of scans entering the precision mean
kax = overlay_axes(fig, H)
for i_, (key, lab, col, mk, d, p) in enumerate(meth):
    npc = PRV[lab][6]
    lab2 = lab.replace('BLAST-CT + ', '+ ') + (f' (n = {npc})' if npc < 43 else '')
    cx, cy = (119.0 if i_ < 3 else 152.0), 74.5 + (i_ % 3) * 3.6
    key_row(fig, cx, cy, [('marker', col, lab2, dict(marker=mk, ms=5.0))], H, ax=kax)
fig_text(fig, 3.0, 99.5, 'Precision is undefined for a scan with no predicted voxels, so its mean is taken over scans with any prediction (n shown when below 43); '
         'recall uses all 43 scans.\nSeg-CQ500 had been used during pipeline development (comparisons not blind); post-processing variants are reported for completeness. '
         'Volumes assume\nmillimeter voxel spacing.', H, linespacing=1.25)
print({k: tuple(round(x, 3) for x in v[:6]) + (v[6],) for k, v in PRV.items()})
print('overflow', overflow_mm(fig))
save(fig, 'SF_segcq500_methods', 'supp')
check_qa('SF_segcq500_methods')
