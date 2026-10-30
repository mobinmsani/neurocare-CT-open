"""SF_v3b_bhsd - Subtype segmentation with v3b on the BHSD test set (96 scans; RSNA-derived; post hoc models).

a any-class Dice of v3b versus v2 per scan; b class Dice in scans containing each class (v3b mean and 95% CI; v3 mean);
c predicted versus reference class volume (log scale, plotted at volume + 0.1 mL) with the 1 mL decision line;
d QC-gated ICC(A,1) with 95% CI and false calls (>=1 mL predicted for a class absent from the reference).
Values from verify_original.json (bhsd_v3), new_analyses_seg.json (v3b_subtype_volume), extra_checks3.json (gated class ICC)
and thresholds_clinical.json; per-scan data from the v3/v3b BHSD metrics.
"""
from nc_s1 import *
from matplotlib.ticker import NullLocator, FixedLocator
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis'))
from common import icc_a1

H = 204
fig = plt.figure(figsize=(W2, H * MM))
k, v1, nm, v2b, v3b, v3 = bhsd()
isq = np.array([c == QCID for c in k])
assert isq.sum() == 1
V = VO['bhsd_v3']['v3b']
SV = {r['sub']: r for r in S['v3b_subtype_volume']}
GI = X3['v3b_class_icc_gated']

# ======================================================================= a: any-class Dice, v3b vs v2
YA = 16.0
axA = axes_mm(fig, 16, YA, 44, 44)
xa, yb = nm.dice_nopostmask.values, v3b.ANY_dice.values
d = yb - xa
assert abs(d.mean() - V['delta']) < 1e-9
axA.plot([0, 1], [0, 1], color=MUTED, lw=1.0, zorder=1)
axA.scatter(xa, yb, s=12, color=M['v3b'], alpha=0.75, lw=0, zorder=3)
axA.set_xlim(-0.03, 1.0); axA.set_ylim(-0.03, 1.0); axA.set_aspect('equal')
axA.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ['0', '0.25', '0.50', '0.75', '1']); axA.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ['0', '0.25', '0.50', '0.75', '1'])
axA.set_xlabel('v2 Dice'); axA.set_ylabel('v3b any-class Dice')
grid(axA, 'both')
nh, nl, ne = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
fig_text(fig, 64, YA + 4.0, 'Paired difference, v3b − v2', H, color=INK, ha='left', fontweight='bold')
fig_text(fig, 64, YA + 7.6, nminus(f"Mean {V['delta']:.3f}"), H, color=INK, ha='left')
fig_text(fig, 64, YA + 11.0, nminus(f"(95% CI {V['delta_ci'][0]:.3f} to {V['delta_ci'][1]:.3f})"), H, color=INK, ha='left')
fig_text(fig, 64, YA + 14.4, nminus(f"Median {np.median(d):.3f}"), H, color=INK, ha='left')
fig_text(fig, 64, YA + 17.8, f"Wilcoxon signed-rank {fmt_p(V['p'])}", H, color=INK, ha='left')
fig_text(fig, 64, YA + 21.2, f"{nh} higher, {nl} lower, {ne} equal", H, color=INK2, ha='left')
fig_text(fig, 64, YA + 25.8, 'The interval includes 0; the', H, color=INK2, ha='left')
fig_text(fig, 64, YA + 29.2, 'signed-rank test reflects a', H, color=INK2, ha='left')
fig_text(fig, 64, YA + 32.6, 'small, consistent paired shift.', H, color=INK2, ha='left')
ptitle(fig, axA, 'a', 'Any-class overlap of v3b versus v2', sub='BHSD test, 96 scans; both models post hoc', dx_mm=14.0, dy_mm=3.0)

# ======================================================================= b: class Dice where the class is present
axB = axes_mm(fig, 120, YA, 58, 44)
assert True
rng = np.random.default_rng(3)
for i, s in enumerate(SUB):
    pres = v3b[s + '_ref_ml'].values > 0
    dv = v3b[s + '_dice'].values[pres]
    n = int(pres.sum())
    assert n == SV[s]['n_present'] and abs(dv.mean() - SV[s]['dice_present']) < 1e-9
    axB.scatter(i - 0.13 + rng.uniform(-0.15, 0.15, n), dv, s=6, color=M['v3b'], alpha=0.45, lw=0, zorder=3)
    lo, hi = SV[s]['dice_present_ci']
    ci_mark(axB, i + 0.2, dv.mean(), lo, hi, 'v3b', orient='v', ms=4.6, z=5)
    m3 = float(v3[s + '_dice'].values[pres].mean())
    assert abs(m3 - SV[s]['v3_dice_present']) < 1e-9
    axB.plot([i + 0.38], [m3], marker='D', ms=4.2, mfc=M['v3'], mec=V3_EDGE, mew=1.0, zorder=6)
    axB.text(i + 0.2, 1.03, f'{dv.mean():.2f}', fontsize=T7, ha='center', va='bottom', color=INK)
axB.set_xticks(range(5), [f'{s}{DG if s in SURG else ""}\n{SV[s]["n_present"]}' for s in SUB], linespacing=1.05)
axB.set_xlim(-0.55, 4.7); axB.set_ylim(-0.03, 1.12)
axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ['0', '0.25', '0.50', '0.75', '1'])
axB.set_ylabel('Class Dice (scans with the class)')
grid(axB, 'y')
fig_text(fig, 120, YA + 44 + 9.2, 'Class and scans with the class', H, color=INK, ha='left')
axB.legend(handles=[Line2D([], [], marker='o', ms=3.0, ls='', mfc=M['v3b'], mec=M['v3b'], alpha=0.6, label='v3b, scan'),
                    Line2D([], [], marker='s', ms=4.6, ls='-', lw=1.2, color=M['v3b'], mfc=M['v3b'], mec=M['v3b'], label='v3b mean (95% CI)'),
                    Line2D([], [], marker='D', ms=4.2, ls='', mfc=M['v3'], mec=V3_EDGE, mew=1.0, label='v3 mean')],
           loc='upper left', bbox_to_anchor=(-0.75, -0.24), ncol=3, borderaxespad=0, columnspacing=1.0, handletextpad=0.3)
ptitle(fig, axB, 'b', 'Class overlap is highest for IPH', sub='v3 predicted no EDH, SAH or SDH (mean 0)', dx_mm=0.0, dy_mm=3.0)

# ======================================================================= c: class volume agreement (3 + 2 layout) and d: table
OFF = 0.1
LO, HI = 0.07, 700
TV = [0, 1, 10, 100]                 # original volumes at the tick positions (plotted at v + 0.1 mL)
YC1, YC2, CW = 88.0, 143.0, 38.0
pos_c = [(16, YC1), (78, YC1), (140, YC1), (16, YC2), (78, YC2)]
tab = []
axc0 = None
for i, s in enumerate(SUB):
    x0, y0 = pos_c[i]
    ax = axes_mm(fig, x0, y0, CW, CW)
    p = v3b[s + '_pred_ml'].values; y = v3b[s + '_ref_ml'].values
    ia, ig = icc_a1(p, y), icc_a1(p[~isq], y[~isq])
    assert abs(ia - SV[s]['ICC']) < 1e-9 and abs(ig - GI[s][0]) < 1e-9
    if s == 'IPH':
        assert abs(ig - TC['gated']['iph_v3b_icc']) < 1e-9
    ax.axvspan(LO, 0.13, color=BAND2, lw=0, zorder=0)
    ax.plot([LO, HI], [LO, HI], color=MUTED, lw=1.0, zorder=1)
    ax.axhline(1.0 + OFF, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=1)
    pres = y > 0
    ax.scatter(y[~isq & pres] + OFF, p[~isq & pres] + OFF, s=10, color=M['v3b'], alpha=0.8, lw=0, zorder=3)
    ax.scatter(y[~isq & ~pres] + OFF, p[~isq & ~pres] + OFF, s=10, facecolor=WHITE, edgecolor=M['v3b'], linewidths=1.0, zorder=3)
    ax.scatter(y[isq] + OFF, p[isq] + OFF, s=40, marker='*', color=CRIT, lw=0, zorder=5)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(LO, HI); ax.set_ylim(LO, HI); ax.set_aspect('equal')
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_locator(FixedLocator([v + OFF for v in TV])); a_.set_ticklabels([f'{v:g}' for v in TV]); a_.set_minor_locator(NullLocator())
    grid(ax, 'both')
    ax.set_xlabel('Reference (mL)', labelpad=1.0)
    ax.set_ylabel('v3b predicted (mL)', labelpad=1.0)
    ax.text(0.5, 1.035, s + (DG if s in SURG else ''), transform=ax.transAxes, fontsize=T7, fontweight='bold', ha='center', va='bottom', color=INK)
    ax.text(0.98, 0.03, nminus(f'ICC {ig:.2f}'), transform=ax.transAxes, fontsize=T7, ha='right', va='bottom', color=INK, path_effects=halo(2.0))
    na = int((~pres & ~isq).sum()); fa = int(((p >= 1) & ~pres & ~isq).sum())
    assert int(((p >= 1) & ~pres).sum()) == SV[s]['pred_ge1_absent'] and int((~pres).sum()) == SV[s]['n_absent']
    tab.append((s, int((pres & ~isq).sum()), ig, GI[s][1], GI[s][2], ia, fa, na))
    if i == 0:
        axc0 = ax
ptitle(fig, axc0, 'c', 'Class volume agreement', sub='Log scales; volumes plotted at value + 0.1 mL so that 0 (shaded column: class absent from the reference) can be shown',
       dx_mm=14.0, dy_mm=7.0)
# ---- d: table in the sixth slot
axD = axes_mm(fig, 126, YC2, 52, CW)
axD.axis('off')
xs_ = [0.0, 0.19, 0.40, 1.0]
hdr = ['Class', 'Present', 'ICC (95% CI)', 'False calls']
for xx, h_ in zip(xs_, hdr):
    axD.text(xx, 0.98, h_, transform=axD.transAxes, fontsize=T7, fontweight='bold', ha='left' if xx < 0.9 else 'right', va='top', color=INK)
axD.plot([0, 1], [0.86, 0.86], transform=axD.transAxes, color=INK2, lw=1.0, clip_on=False)
for r_, (s, npres, ig, lo_, hi_, ia, fa, na) in enumerate(tab):
    yy = 0.78 - r_ * 0.135
    axD.text(0.0, yy, s + (DG if s in SURG else ''), transform=axD.transAxes, fontsize=T7, ha='left', va='center', color=INK)
    axD.text(0.19, yy, f'{npres}', transform=axD.transAxes, fontsize=T7, ha='left', va='center', color=INK)
    axD.text(0.40, yy, nminus(f'{ig:.2f} ({lo_:.2f}–{hi_:.2f})'), transform=axD.transAxes, fontsize=T7, ha='left', va='center', color=INK)
    axD.text(1.0, yy, f'{fa}/{na}', transform=axD.transAxes, fontsize=T7, ha='right', va='center', color=INK)
chg = [f"{t_[0]} {t_[5]:.2f}" for t_ in tab if abs(t_[5] - t_[2]) >= 0.005]
axD.text(0.0, -0.06, 'QC-gated 95 scans. False calls: scans\nwithout the class with ≥1 mL predicted.\nAll 96 scans: ICC ' + ', '.join(chg) + '.', transform=axD.transAxes,
         fontsize=T7, ha='left', va='bottom', color=INK2, linespacing=1.05)
letter(fig, 126 - 2.6, YC2 - 4.0, 'd')
fig.text((126 + 0.4) * MM / W2, 1 - (YC2 - 4.0) * MM / (H * MM), 'Agreement summary', fontsize=T7, fontweight='bold', color=INK, ha='left', va='baseline')
fig.legend(handles=[Line2D([], [], marker='o', ms=4.0, ls='', mfc=M['v3b'], mec=M['v3b'], label='Class present in reference'),
                    Line2D([], [], marker='o', ms=4.0, ls='', mfc=WHITE, mec=M['v3b'], mew=1.0, label='Class absent'),
                    Line2D([], [], marker='*', ms=7.0, ls='', mfc=CRIT, mec=CRIT, label='Non-HU-encoded scan (excluded by the QC gate)'),
                    Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 2)), label='1 mL predicted (decision rule)'),
                    Line2D([], [], color=MUTED, lw=1.0, label='Identity')],
           loc='upper left', bbox_to_anchor=(2.0 * MM / W2, 1 - (YC2 + CW + 9.5) * MM / (H * MM)), ncol=3, borderaxespad=0, columnspacing=1.2, handletextpad=0.4)
print('overflow', overflow_mm(fig))
for t_ in tab:
    print(t_)
save(fig, 'SF_v3b_bhsd', 'supp')
