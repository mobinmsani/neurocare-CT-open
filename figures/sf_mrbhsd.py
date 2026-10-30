"""SF_mrbhsd - MR-BHSD (43 scans, three independent raters): inter-rater variability, empty masks, v1 agreement with each
rater, failures versus rater disagreement, and model-rater versus rater-rater agreement."""
from nc_s2 import *
from scipy import stats

H = 210
fig = plt.figure(figsize=(W2, H * MM))
m_ = mr()
W = m_.pivot(index='case_id', columns='reference_rater', values='reference_volume_ml')
Dd = m_.pivot(index='case_id', columns='reference_rater', values='dice').loc[W.index]
P = m_.groupby('case_id').pred_volume_ml.first().loc[W.index]
RAT = ['annot_1', 'annot_3', 'annot_5']
RN = {'annot_1': 'R1', 'annot_3': 'R2', 'annot_5': 'R3'}          # display names; original annotator IDs in the legend
RMK = {'annot_1': 'o', 'annot_3': 's', 'annot_5': '^'}
assert list(W.columns) == RAT and len(W) == 43
rr = rr_pairs()
assert f"{Dd.mean(axis=1).mean():.3f}" == '0.399' and REG['mr_v1_dice'] == '0.399 (0.311–0.489)'
assert f"{rr.mean_dice.mean():.3f}" == '0.755' and f"{rr.mean_dice.min():.3f}–{rr.mean_dice.max():.3f}" == '0.734–0.789'
assert f"{S['mr_rater_icc']:.3f}" == '0.863' and f"{S['mr_model_vs_mean_icc']:.3f}" == '0.759'
cv = W.std(axis=1, ddof=1) / W.mean(axis=1).replace(0, np.nan)
assert cv.notna().sum() == 42 and f"{100 * cv.median():.0f}" == REG['mr_cv_median']
empty = (W == 0)
assert empty.sum().to_dict() == {'annot_1': 3, 'annot_3': 2, 'annot_5': 1} and int(empty.any(axis=1).sum()) == 3
rngD = Dd.max(axis=1) - Dd.min(axis=1)
assert int((rngD > 0.1).sum()) == REG['mr_range_gt01'] == 5 and f"{rngD.median():.3f}" == REG['mr_dice_range_med']
order = W.mean(axis=1).sort_values(kind='mergesort').index
xs = np.arange(1, 44)
XL, XW = 17.0, 106.0
LGRAY = '#b9b5ad'

# ======================================================================= a: rater volumes per scan (log axis) + separate zero strip
axA = axes_mm(fig, XL, 14, XW, 34)
axZ = axes_mm(fig, XL, 50.5, XW, 4.5)
for x, c in zip(xs, order):
    v = W.loc[c].values.astype(float)
    pos = v > 0
    if pos.sum() >= 2:
        axA.plot([x, x], [v[pos].min(), v[pos].max()], color=LGRAY, lw=1.0, zorder=1, solid_capstyle='butt')
    for r, vv in zip(RAT, v):
        if vv > 0:
            axA.plot(x, vv, RMK[r], ms=3.6, mfc=INK, mec=WHITE, mew=0.4, ls='', zorder=3)
        else:   # categorical zero strip: small horizontal offsets keep coincident empty masks visible (values are all 0 mL)
            axZ.plot(x + {'annot_1': -0.3, 'annot_3': 0.0, 'annot_5': 0.3}[r], 0, RMK[r], ms=3.6, mfc=WHITE, mec=INK, mew=1.0, ls='', zorder=3)
    p_ = float(P.loc[c])
    if p_ > 0:
        axA.plot(x, p_, '_', ms=6.0, mew=1.8, color=M['v1'], ls='', zorder=4)
    else:
        axZ.plot(x, 0, '_', ms=6.0, mew=1.8, color=M['v1'], ls='', zorder=4)
cmin = P[P > 0].idxmin()
xm = xs[list(order).index(cmin)]
axA.annotate(f'v1 {P.loc[cmin]:.2f} mL', (xm, P.loc[cmin]), xytext=(xm + 1.8, 0.012), fontsize=F7, ha='left', va='center', color=INK2,
             arrowprops=dict(arrowstyle='-', lw=1.0, color=INK2, shrinkA=1, shrinkB=3))
axA.set_yscale('log'); axA.set_ylim(0.008, 300); axA.set_xlim(0.2, 43.8)
plain_log(axA, 'y', [0.01, 0.1, 1, 10, 100], ['0.01', '0.1', '1', '10', '100'])
axA.set_xticks([1, 10, 20, 30, 43]); axA.tick_params(axis='x', labelbottom=False)
axA.set_ylabel('Volume (mL)')
grid(axA, 'y')
axZ.set_xlim(0.2, 43.8); axZ.set_ylim(-1, 1); axZ.set_yticks([0], ['0']); axZ.set_xticks([1, 10, 20, 30, 43])
axZ.tick_params(axis='x', labelbottom=False)
axZ.spines['top'].set_visible(False)
for yy_ in (1.0,):       # axis-break marks between the log axis and the zero strip
    pass
axA.plot([-0.012, 0.012], [-0.02, 0.02], transform=axA.transAxes, color=AXIS, lw=1.0, clip_on=False)
axZ.plot([-0.012, 0.012], [0.85, 1.15], transform=axZ.transAxes, color=AXIS, lw=1.0, clip_on=False)
ptitle(fig, axA, 'a', 'Rater volumes and v1 prediction per scan', sub='Empty masks plotted in a separate 0 mL strip below the axis break', dx_mm=14.5, dy_mm=3.2)
key_row(fig, XL, 59.5, [('marker', INK, 'R1', dict(marker='o', ms=3.6)), ('marker', INK, 'R2', dict(marker='s', ms=3.6)),
                               ('marker', INK, 'R3', dict(marker='^', ms=3.6)), ('marker', INK, 'empty mask (open)', dict(marker='s', ms=3.6, filled=False)),
                               ('line', M['v1'], 'v1 prediction', dict(lw=1.8))], H, gap_mm=3.2, sw=3.0)

# ======================================================================= b: v1 Dice against each rater, same scan order
axB = axes_mm(fig, XL, 70, XW, 36)
lo_c, hi_c = rr.mean_dice.min(), rr.mean_dice.max()
axB.axhspan(lo_c, hi_c, color=CEIL, lw=0, zorder=0)
mean_v1 = Dd.mean(axis=1).mean()
axB.axhline(mean_v1, color=M['v1'], lw=1.0, ls=(0, (3, 2)), zorder=1)
big = set(rngD[rngD > 0.1].index)
for x, c in zip(xs, order):
    d = Dd.loc[c].values
    axB.plot([x, x], [d.min(), d.max()], color=LGRAY, lw=1.0, zorder=1, solid_capstyle='butt')
    for r, dv in zip(RAT, d):
        axB.plot(x, dv, RMK[r], ms=3.6, mfc=M['v1'], mec=WHITE, mew=0.4, ls='', zorder=3)
    if c in big:
        axB.add_patch(FancyBboxPatch((x - 0.42, d.min() - 0.035), 0.84, d.max() - d.min() + 0.07, boxstyle='round,pad=0,rounding_size=0.2',
                                     fc='none', ec=INK, lw=1.0, zorder=4))
axB.set_xlim(0.2, 43.8); axB.set_ylim(-0.04, 1.04)
axB.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
axB.set_xticks([1, 10, 20, 30, 43])
axB.set_xlabel('Scans ordered by mean rater volume (smallest to largest)')
axB.set_ylabel('v1 Dice against each rater')
grid(axB, 'y')
ce = W.index[(W == 0).all(axis=1)][0]
xe = xs[list(order).index(ce)]
axB.annotate('no rater marked hemorrhage;\nDice 1 by convention', (xe + 0.4, 1.0), xytext=(xe + 2.6, 0.86), fontsize=F7, color=INK2, ha='left', va='center',
             arrowprops=dict(arrowstyle='-', lw=1.0, color=INK2, shrinkA=1, shrinkB=2), linespacing=1.1)
ptitle(fig, axB, 'b', 'v1 Dice against each rater',
       sub=f"Median within-scan range {rngD.median():.3f}; {int((rngD > 0.1).sum())} of 43 scans exceed 0.1 (boxed)", dx_mm=14.5, dy_mm=3.2)
key_row(fig, XL, 115.5, [('band', CEIL, f'range of rater-pair mean Dice ({lo_c:.3f}–{hi_c:.3f}), human agreement benchmark'),
                         ('line', M['v1'], f'v1 mean Dice {mean_v1:.3f}', dict(ls=(0, (3, 2)), lw=1.0)),
                         ('marker', M['v1'], 'v1 vs R1, R2, R3 (shapes as in a)', dict(marker='o', ms=3.6))], H, sw=4.0, gap_mm=3.0)

# ======================================================================= c: relative inter-rater variability vs size
ok = cv.notna()
mv = W.mean(axis=1)
rho_cv = stats.spearmanr(mv[ok], cv[ok])
axC = axes_mm(fig, 141, 14, 37, 37)
axC.scatter(mv[ok], 100 * cv[ok], s=12, color=INK2, marker='D', lw=0, alpha=0.85, zorder=3)
axC.set_xscale('log'); axC.set_xlim(0.15, 120)
plain_log(axC, 'x', [0.3, 1, 3, 10, 30, 100], ['0.3', '1', '3', '10', '30', '100'])
axC.set_ylim(-5, 185); axC.set_yticks([0, 50, 100, 150])
axC.set_xlabel('Mean rater volume (mL)'); axC.set_ylabel('Inter-rater volume CV (%)')
grid(axC, 'y')
axC.text(0.97, 0.97, nminus(f'Spearman ρ {rho_cv.correlation:.2f}\n{fmt_p(rho_cv.pvalue)}; 42 scans'), transform=axC.transAxes, fontsize=F7, ha='right', va='top',
         color=INK, linespacing=1.15)
ptitle(fig, axC, 'c', 'Variability by size', dx_mm=13.5, dy_mm=3.2)

# ======================================================================= d: v1 Dice vs inter-rater variability (from v3 Figure 2e)
da = Dd.mean(axis=1)
a = S['mr_ambiguity']
assert abs(stats.spearmanr(cv[ok], da[ok]).correlation - a['spearman_cv_vs_dice']) < 1e-9
axD = axes_mm(fig, 141, 70, 37, 36)
axD.scatter(100 * cv[ok], da[ok], s=12, color=M['v1'], marker=MK['v1'], lw=0, alpha=0.85, zorder=3)
axD.set_xlim(-6, 185); axD.set_ylim(-0.04, 1.04)
axD.set_xticks([0, 50, 100, 150]); axD.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
axD.set_xlabel('Inter-rater volume CV (%)'); axD.set_ylabel('v1 Dice, mean of 3 raters')
grid(axD, 'y')
axD.text(0.97, 0.97, nminus(f"Spearman ρ {a['spearman_cv_vs_dice']:.2f}\n{fmt_p(a['p'])}\npartial ρ {S['mr_partial_spearman_cv_dice_given_size']:.2f} (volume)"),
         transform=axD.transAxes, fontsize=F7, ha='right', va='top', color=INK, linespacing=1.15)
ptitle(fig, axD, 'd', 'Failures and rater disagreement', dx_mm=13.5, dy_mm=3.2)

# ======================================================================= e: model-rater and rater-rater agreement
pairs = [('annot_1', 'annot_3'), ('annot_1', 'annot_5'), ('annot_3', 'annot_5')]
rrD = [float(rr[(rr.annotator_a == a_) & (rr.annotator_b == b_)].mean_dice.iloc[0]) for a_, b_ in pairs]
v1D = [float(Dd[r].mean()) for r in RAT]
assert {r: f'{float(Dd[r].mean()):.3f}' for r in RAT} == REG['mr_per_rater']
rrI = list(S['mr_rater_pair_icc'].values()); v1I = list(S['mr_model_vs_rater_icc'].values())
rrM = list(S['mr_rater_pair_mad'].values()); v1M = [float(np.abs(P - W[r]).mean()) for r in RAT]
blocks = [('Dice (mean over scans)', [('Rater vs rater', 'range', rrD, None), ('v1 vs each rater', 'range', v1D, None)], (0, 1), [0, 0.25, 0.5, 0.75, 1.0], '{:.3f}'),
          ('Volume ICC(A,1)', [('Rater pairs', 'range', rrI, None), ('All three raters', 'ci', None, (S['mr_rater_icc'], *S['mr_rater_icc_ci'])),
                               ('v1 vs each rater', 'range', v1I, None), ('v1 vs rater mean', 'ci', None, (S['mr_model_vs_mean_icc'], *S['mr_model_vs_mean_icc_ci']))],
           (0.4, 1.0), [0.4, 0.6, 0.8, 1.0], '{:.3f}'),
          ('Mean absolute volume difference (mL)', [('Rater pairs', 'range', rrM, None), ('v1 vs each rater', 'range', v1M, None)], (0, 9), [0, 3, 6, 9], '{:.1f}')]
yb_ = [130.0, 149.5, 178.0]
axE0 = None
for (ttl, rows_, xl, xt, fmt), y_ in zip(blocks, yb_):
    nr = len(rows_)
    hh = 4.6 * nr
    ax = axes_mm(fig, 46.0, y_, 80.0, hh)
    axE0 = axE0 or ax
    fig_text(fig, 3.0, y_ - 4.4, ttl, H, color=INK, fontweight='bold')
    for i_, (lab, kind, vals, ci) in enumerate(rows_):
        col = INK if lab.startswith(('Rater', 'All')) else M['v1']
        mk = 'D' if lab.startswith(('Rater', 'All')) else 'o'
        if kind == 'range':
            vv = np.array(vals)
            ax.plot([vv.min(), vv.max()], [i_, i_], color=col, lw=1.0, zorder=2)
            for e_ in (vv.min(), vv.max()):
                ax.plot([e_, e_], [i_ - 0.22, i_ + 0.22], color=col, lw=1.0, zorder=2)
            ax.plot(vv, np.full(len(vv), i_), ls='', marker=mk, ms=3.8, mfc=WHITE, mec=col, mew=1.0, zorder=3)
            txt = (fmt + '–' + fmt).format(vv.min(), vv.max()) + ' (range of 3)'
        else:
            est, lo_, hi_ = ci
            ax.errorbar([est], [i_], xerr=[[est - lo_], [hi_ - est]], fmt=mk, ms=4.4, mfc=col, mec=col, ecolor=col, elinewidth=1.6, capsize=2.4, capthick=1.6, zorder=3)
            txt = (fmt + ' (95% CI ' + fmt + '–' + fmt + ')').format(est, lo_, hi_)
        ax.text(-0.03, i_, lab, transform=ax.get_yaxis_transform(), fontsize=F7, ha='right', va='center', color=INK)
        ax.text(1.04, i_, txt, transform=ax.get_yaxis_transform(), fontsize=F7, ha='left', va='center', color=INK)
    ax.set_ylim(nr - 0.5, -0.5); ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlim(*xl); ax.set_xticks(xt, [('{:g}'.format(t)) for t in xt])
    grid(ax, 'x')
ptitle(fig, axE0, 'e', 'Model–rater and rater–rater agreement', dx_mm=43.0, dy_mm=9.6)
key_row(fig, 3.0, 194.5, [('marker', INK, 'open markers between end caps: the three observed pairs or raters', dict(marker='D', ms=3.8, filled=False)),
                          ('marker', INK, 'filled marker with caps: estimate and bootstrap 95% CI', dict(marker='D', ms=4.2))], H, gap_mm=6.0)
fig_text(fig, 3.0, 199.0, 'Gray: v1 (compact U-Net); black: raters. Rater–rater agreement is a human agreement benchmark, not an upper bound on model–rater agreement. '
         'Jointly empty\nmasks score Dice 1 (one scan; four rater-pair comparisons). Rater analyses were added post hoc. Six of the 43 scans were used during '
         'pipeline development.', H, linespacing=1.25)
print('rho CV vs volume', rho_cv)
print('overflow', overflow_mm(fig))
save(fig, 'SF_mrbhsd', 'supp')
check_qa('SF_mrbhsd')
