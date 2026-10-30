"""SF_paired_dice - Paired per-scan Dice comparisons (prespecified, post hoc, locked transfer) and per-scan Dice on the BHSD
test set."""
from nc_s2 import *

H = 146
fig = plt.figure(figsize=(W2, H * MM))
k, v1, nm, v2b, v3b, v3 = bhsd()
segm = pd.read_csv(T / 'SegCQ500_four_method_comparison.csv')
s1 = segm[segm.method == 'lightunet25d'].set_index('case_id').dice
s3 = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/segcq500_no_postmask.csv').set_index('case_id').dice_nopostmask
ks = sorted(s1.index)
assert len(ks) == 43 and set(ks) == set(s3.index)
PR = {p['cmp']: p for p in VO['pairs_table4']}
assert f"{PR['v2_nomask_posthoc-v1 BHSD']['mean']:+.3f}" == '+0.026' and PR['v2_nomask_posthoc-v1 BHSD']['p'] < 0.001
assert f"{PR['v2_nomask-v1 SegCQ500']['mean']:+.3f}" == '+0.028' and f"{PR['v2_nomask-v1 SegCQ500']['p']:.2f}" == '0.22'
assert f"{PR['v2_prespecified-v1 BHSD']['mean']:+.3f}" == '-0.045' and f"{PR['v2_prespecified-v1 BHSD']['p']:.2f}" == '0.55'

panels = [
    ('a', 'Prespecified: v2-PM vs v1', 'BHSD test, 96 scans', v1.dice.values, v2b.dice.values, 'v2-PM Dice', PR['v2_prespecified-v1 BHSD'], 'v2pm'),
    ('b', 'Post hoc: v2 vs v1', 'BHSD test, 96 scans', v1.dice.values, nm.dice_nopostmask.values, 'v2 Dice', PR['v2_nomask_posthoc-v1 BHSD'], 'v2'),
    ('c', 'Locked transfer comparison: v2 vs v1', 'Seg-CQ500, 43 scans; not blind', s1.loc[ks].values, s3.loc[ks].values, 'v2 Dice',
     PR['v2_nomask-v1 SegCQ500'], 'v2'),
]
S_ = 44.0
xs = [17.0, 77.0, 135.0]
y0 = 19.0
for (L, ttl, sub, xa, yb, ylab, pr, mk), x0 in zip(panels, xs):
    col = M[mk]
    ax = axes_mm(fig, x0, y0, S_, S_)
    d = yb - xa
    up, dn, tie = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    both0 = (xa == 0) & (yb == 0)
    assert up == pr['improved'] and int(both0.sum()) == tie        # every tie is a pair with no overlap for either model
    ax.fill_between([0, 1], [0, 1], [1, 1], color=col, alpha=0.07, lw=0, zorder=0)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.0, zorder=1)
    nz = ~both0
    ax.scatter(xa[nz], yb[nz], s=13, color=col, marker=MK[mk], alpha=0.85, lw=0, zorder=3)
    if both0.any():         # coincident (0, 0) pairs: one count marker, area proportional to the number of pairs
        nb0 = int(both0.sum())
        ax.plot([0], [0], marker=MK[mk], ms=4.5 + 2.2 * np.sqrt(nb0), mfc=WHITE, mec=col, mew=1.2, ls='', zorder=4, clip_on=False)
        if nb0 > 1:
            ax.annotate(f'{nb0} pairs at 0, 0\n(no overlap, either model)', (0.0, 0.0), xytext=(0.36, 0.12), textcoords='data',
                        fontsize=F7, ha='left', va='center', color=INK, linespacing=1.1,
                        arrowprops=dict(arrowstyle='-', lw=1.0, color=INK2, shrinkA=1, shrinkB=5))
    ax.set_xlim(-0.04, 1.0); ax.set_ylim(-0.04, 1.0); ax.set_aspect('equal')
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
    ax.set_xlabel('v1 Dice'); ax.set_ylabel(ylab)
    grid(ax, 'both')
    ax.text(0.03, 0.97, f'higher than v1: {up}', transform=ax.transAxes, fontsize=F7, va='top', ha='left', color=INK2)
    ax.text(0.97, 0.03 if not both0.any() else 0.30, f'lower: {dn}', transform=ax.transAxes, fontsize=F7, va='bottom', ha='right', color=INK2)
    ptitle(fig, ax, L, ttl, sub=sub, dx_mm=13.5)
    # aligned results strip (below the plot, outside the data area)
    lo, hi = pr['lo'], pr['hi']
    med = float(np.median(d))
    meds = '0.000' if abs(med) < 5e-4 else f'{med:+.3f}'
    lines = [nminus(f"Mean difference {pr['mean']:+.3f}"), nminus(f"(95% CI {lo:+.3f} to {hi:+.3f})"),
             nminus(f"Median difference {meds}"), f"Wilcoxon signed-rank {fmt_p(pr['p'])}", f"Higher {up}, lower {dn}, equal {tie} (all at 0, 0)"]
    for i_, t in enumerate(lines):
        fig_text(fig, x0 - 13.5, y0 + S_ + 9.6 + i_ * 3.3, t, H, color=INK if i_ < 4 else INK2)
fig_text(fig, 3.5, y0 + S_ + 27.6, 'Differences are second model minus v1. The 95% CI (bootstrap, 2,000 resamples) describes the mean paired difference; the Wilcoxon '
         'signed-rank test\ncompares the distribution of paired differences, so the two can point in different directions when many differences are small.', H,
         linespacing=1.25)

# ======================================================================= d: per-scan Dice distributions on BHSD test
axD = axes_mm(fig, 36, 110, 104, 26)
rows = [('v1', 'prespecified', v1.dice.values, 'v1'), ('v2-PM', 'prespecified', v2b.dice.values, 'v2pm'),
        ('v2', 'post hoc', nm.dice_nopostmask.values, 'v2'), ('v3b, any class', 'post hoc', v3b.ANY_dice.values, 'v3b')]
means = {r['model']: r for r in VO['bhsd_table3'] if r['cohort'] == 'all 96'}
mkey = {'v1': 'v1', 'v2-PM': 'v2_prespecified', 'v2': 'v2_nomask_posthoc'}
rng = np.random.default_rng(5)
for i, (lab, tg, v, mk) in enumerate(rows):
    col = M[mk]
    yj = i + rng.uniform(-0.22, 0.22, len(v))      # vertical jitter only; Dice values are not altered
    axD.scatter(v, yj, s=8, color=col, marker=MK[mk], alpha=0.7, lw=0, zorder=3)
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    axD.add_patch(Rectangle((q1, i - 0.32), q3 - q1, 0.64, fill=False, ec='#9c978e', lw=1.0, zorder=4))
    axD.plot([q2, q2], [i - 0.32, i + 0.32], color=INK, lw=1.8, zorder=5, solid_capstyle='butt')
    mu = float(np.mean(v))
    if lab in mkey:
        assert abs(mu - means[mkey[lab]]['mean_dice']) < 1e-9
    else:
        assert abs(mu - VO['bhsd_v3']['v3b']['any_dice']) < 1e-9
    axD.plot([mu], [i], marker='D', ms=4.6, mfc=WHITE, mec=col, mew=1.3, zorder=6)
    axD.text(-0.03, i, f'{lab} ({tg})', transform=axD.get_yaxis_transform(), fontsize=F7, ha='right', va='center', color=INK)
    axD.text(1.04, i, f'{mu:.3f}', transform=axD.get_yaxis_transform(), fontsize=F7, ha='left', va='center', color=INK)
    axD.text(1.17, i, f'{q2:.3f}', transform=axD.get_yaxis_transform(), fontsize=F7, ha='left', va='center', color=INK)
    axD.text(1.36, i, f'{int((v == 0).sum())}', transform=axD.get_yaxis_transform(), fontsize=F7, ha='right', va='center', color=INK)
for xx_, t, ha in [(1.04, 'Mean', 'left'), (1.17, 'Median', 'left'), (1.36, 'Dice 0', 'right')]:
    axD.text(xx_, -0.85, t, transform=axD.get_yaxis_transform(), fontsize=F7, ha=ha, va='center', color=INK2, fontweight='bold')
axD.set_ylim(3.5, -0.6); axD.set_yticks([]); axD.spines['left'].set_visible(False)
axD.set_xlim(-0.02, 1.0); axD.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
axD.set_xlabel('Dice per scan, BHSD test (96 scans)')
grid(axD, 'x')
ptitle(fig, axD, 'd', 'BHSD per-scan Dice', dx_mm=33.0, dy_mm=5.4)
key_row(fig, 36, 107.4, [('fill', WHITE, 'interquartile range (box)', dict(ec='#9c978e')), ('line', INK, 'median', dict(lw=1.8)),
                         ('marker', INK2, 'mean', dict(marker='D', filled=False))], H, sw=4.0)
print({p[0]: (round(float(np.median(p[4] - p[3])), 4)) for p in panels})
print('overflow', overflow_mm(fig))
save(fig, 'SF_paired_dice', 'supp')
check_qa('SF_paired_dice')
