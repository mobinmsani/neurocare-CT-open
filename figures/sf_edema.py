"""SF_edema - Perihematomal edema segmentation in 37 held-out PHE-SICH-CT-IDS patients: volume agreement, Bland-Altman with
bootstrap intervals for the bias and limits, and Dice by expert volume."""
from nc_s2 import *
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis'))
from common import icc_a1
from scipy import stats

H = 86
fig = plt.figure(figsize=(W2, H * MM))
ed = pd.read_csv(E / 'neurocare_v2_20261021/edema/edema_test_metrics.csv', dtype={'case': str})
eb = S['edema_bias_by_size']; ve = VO['edema']
assert len(ed) == 37 == ve['n']
r, pr = ed.ref_ml.values, ed.pred_ml.values
icc = icc_a1(pr, r)
assert abs(icc - ve['ICC']) < 1e-12 and f'{icc:.3f}' == '0.258'
d = pr - r
bias, sd = d.mean(), d.std(ddof=1)
assert f'{bias:.2f}' == '1.97' and f'{bias - 1.96 * sd:.1f}' == '-6.8' and f'{bias + 1.96 * sd:.1f}' == '10.8'
assert f"{ve['dice']:.3f}" == '0.336' and f"{ve['lo']:.3f}" == '0.280' and f"{ve['hi']:.3f}" == '0.388'
# bootstrap 95% CIs for the bias and the two limits of agreement (patient as resampling unit)
rng = np.random.default_rng(20261028)
bs = []
for _ in range(10000):
    i = rng.integers(0, len(d), len(d))
    di = d[i]; m_, s_ = di.mean(), di.std(ddof=1)
    bs.append((m_, m_ - 1.96 * s_, m_ + 1.96 * s_))
bs = np.array(bs)
ci_b, ci_lo, ci_hi = [np.percentile(bs[:, j], [2.5, 97.5]) for j in range(3)]
infl = int(np.argmin(d))
cid = ed.case.iloc[infl]
assert cid == '0021'
COL = M['edema']
S_ = 44.0
fig_text(fig, 3.0, 3.0, 'PHE-SICH-CT-IDS, 37 held-out patients (one hospital; spontaneous basal-ganglia hemorrhage); one expert edema mask per patient.\n'
         'Open ring: patient 0021, the largest negative difference, marked in every panel.', H, color=INK2, linespacing=1.25)


def ring(ax, x, y):
    ax.plot([x], [y], marker='o', ms=8.0, mfc='none', mec=INK, mew=1.2, ls='', zorder=6)


# ======================================================================= a: predicted vs expert volume
axA = axes_mm(fig, 15, 22, S_, S_)
lim = 23
axA.plot([0, lim], [0, lim], color=MUTED, lw=1.0, zorder=1)
axA.scatter(r, pr, s=13, color=COL, lw=0, alpha=0.9, zorder=3)
ring(axA, r[infl], pr[infl])
axA.annotate(cid, (r[infl], pr[infl]), xytext=(-2, 7), textcoords='offset points', fontsize=F7, ha='right', va='bottom', color=INK)
axA.set_xlim(-0.5, lim); axA.set_ylim(-0.5, lim); axA.set_aspect('equal')
axA.set_xticks([0, 5, 10, 15, 20]); axA.set_yticks([0, 5, 10, 15, 20])
axA.set_xlabel('Expert edema volume (mL)'); axA.set_ylabel('Predicted edema volume (mL)')
grid(axA, 'both')
lo_i, hi_i = eb['icc_ci']
axA.text(0.03, 0.97, nminus(f"ICC(A,1) {icc:.3f}\n(95% CI {lo_i:.3f} to {hi_i:.3f})"), transform=axA.transAxes, fontsize=F7, ha='left', va='top', color=INK,
         linespacing=1.15)
ptitle(fig, axA, 'a', 'Volume agreement', dx_mm=12.5, dy_mm=3.2)

# ======================================================================= b: Bland-Altman with bootstrap CIs
axB = axes_mm(fig, 77, 22, 44, S_)
mn = (r + pr) / 2
axB.axhline(0, color=MUTED, lw=1.0, zorder=1)
XCI = 14.4      # bootstrap 95% CIs drawn as capped vertical bars at the right edge, away from the data
for (lo_, hi_) in (ci_b, ci_lo, ci_hi):
    axB.errorbar([XCI], [(lo_ + hi_) / 2], yerr=[[(hi_ - lo_) / 2], [(hi_ - lo_) / 2]], fmt='none', ecolor=INK2, elinewidth=1.2, capsize=2.0, capthick=1.2, zorder=4)
axB.axvline(13.8, color=GRID, lw=1.0, zorder=0)
axB.axhline(bias, color=INK, lw=1.4, zorder=2)
for v_ in (bias - 1.96 * sd, bias + 1.96 * sd):
    axB.axhline(v_, color=INK, lw=1.0, ls=(0, (3.5, 2)), zorder=2)
axB.scatter(mn, d, s=13, color=COL, lw=0, alpha=0.9, zorder=3)
ring(axB, mn[infl], d[infl])
axB.set_xlim(0, 15.0); axB.set_ylim(-20, 15)
axB.set_xticks([0, 2.5, 5, 7.5, 10, 12.5], ['0', '2.5', '5', '7.5', '10', '12.5'])
axB.set_xlabel('Mean of expert and predicted (mL)'); axB.set_ylabel('Predicted − expert (mL)')
for v_, t in [(bias, f'bias {bias:+.2f}'), (bias + 1.96 * sd, f'{bias + 1.96 * sd:+.1f}'), (bias - 1.96 * sd, f'{bias - 1.96 * sd:+.1f}')]:
    axB.text(1.02, v_, nminus(t), transform=axB.get_yaxis_transform(), fontsize=F7, ha='left', va='center', color=INK)
pb = eb['prop_bias']
axB.text(0.03, 0.03, nminus(f"proportional bias:\nslope {pb['slope']:.2f}, {fmt_p(pb['pvalue'])}"), transform=axB.transAxes, fontsize=F7, ha='left',
         va='bottom', color=INK2, linespacing=1.15)
grid(axB, 'y')
ptitle(fig, axB, 'b', 'Bland–Altman (mL)', dx_mm=12.5, dy_mm=3.2)

# ======================================================================= c: Dice vs expert volume
axC = axes_mm(fig, 145, 22, 33, S_)
axC.scatter(r, ed.dice.values, s=13, color=COL, lw=0, alpha=0.9, zorder=3)
ring(axC, r[infl], ed.dice.values[infl])
axC.axhline(ve['dice'], color=INK, lw=1.0, ls=(0, (3.5, 2)), zorder=2)
axC.set_xscale('log'); axC.set_xlim(0.08, 30); axC.set_ylim(-0.025, 1)
axC.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
plain_log(axC, 'x', [0.1, 1, 10], ['0.1', '1', '10'])
axC.set_xlabel('Expert edema volume (mL)'); axC.set_ylabel('Dice per patient')
q1, q3 = eb['dice_iqr']
rho, prho = stats.spearmanr(r, ed.dice.values)
axC.text(0.03, 0.97, f"Spearman ρ {rho:.2f}\n{fmt_p(prho)}", transform=axC.transAxes, fontsize=F7, ha='left', va='top', color=INK)
grid(axC, 'y')
ptitle(fig, axC, 'c', 'Overlap by expert volume', dx_mm=12.5, dy_mm=3.2)
key_row(fig, 3.0, 76.5, [('line', INK, 'bias (b)', dict(lw=1.4)), ('line', INK, '95% limits of agreement (b); mean Dice (c)', dict(ls=(0, (3.5, 2)), lw=1.0)),
                         ('text', INK2, 'capped bars at the right edge of b: bootstrap 95% CI of the bias and of each limit')], H, sw=5.0, gap_mm=5.0)
fig_text(fig, 3.0, 79.6, nminus(f"Dice: mean {ve['dice']:.3f} (95% CI {ve['lo']:.3f}–{ve['hi']:.3f}); median {ve['median']:.3f} (IQR {q1:.3f}–{q3:.3f}). "
                                f"Bias CI {ci_b[0]:.2f} to {ci_b[1]:.2f} mL; limits CI {ci_lo[0]:.1f} to {ci_lo[1]:.1f} and {ci_hi[0]:.1f} to {ci_hi[1]:.1f} mL."), H)
print('bias CI', np.round(ci_b, 2), 'LoA lo CI', np.round(ci_lo, 2), 'LoA hi CI', np.round(ci_hi, 2), 'rho', round(rho, 3), prho)
print('overflow', overflow_mm(fig))
save(fig, 'SF_edema', 'supp')
check_qa('SF_edema')
