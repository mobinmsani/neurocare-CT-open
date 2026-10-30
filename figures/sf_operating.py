"""SF_operating - Operating characteristics and error profile on CQ500 (v2 total volume, locked; v3b any-class volume, post hoc).

a sensitivity, specificity, and (v2) positive and negative predictive values by batch across volume thresholds (Wilson bands);
b v2 predicted volume by outcome against the reader majority at the prespecified 1 mL rule;
c any-ICH detection sensitivity at 1 mL by subtype present;
d sensitivity at fixed specificity (thresholds chosen on these data).
Checks every displayed value against new_analyses_cq500.json (thresholds, fp/fn profiles, sens_at_spec, batches) and cq500_results.json.
"""
from nc_s1 import *
from matplotlib.ticker import NullLocator

H = 134
fig = plt.figure(figsize=(W2, H * MM))
m = load_cq500()
y = m.ref_ICH.values.astype(int)
s2 = m.pred_volume_ml.values
s3 = m.ANY_ml.values
bt = m.Category.values
npos, nneg = int((y == 1).sum()), int((y == 0).sum())
assert (npos, nneg) == (205, 284)


def counts(s, t, sel=None):
    sel = np.ones(len(s), bool) if sel is None else sel
    tp = int(((s >= t) & (y == 1) & sel).sum()); tn = int(((s < t) & (y == 0) & sel).sum())
    fp = int(((s >= t) & (y == 0) & sel).sum()); fn = int(((s < t) & (y == 1) & sel).sum())
    return tp, tn, fp, fn


for r in N['thresholds']:
    s = s2 if r['model'] == 'v2' else s3
    tp, tn, fp, fn = counts(s, r['threshold_ml'])
    assert abs(tp / npos - r['sens']) < 1e-12 and abs(tn / nneg - r['spec']) < 1e-12
    assert np.allclose(wilson(tp, npos), r['sens_ci']) and np.allclose(wilson(tn, nneg), r['spec_ci'])
tp1, tn1, fp1, fn1 = counts(s2, 1.0)
assert (tp1, tn1, fp1, fn1) == (168, 169, 115, 37)
tp1b, tn1b, _, _ = counts(s3, 1.0)
bat = {(r['model'], r['batch']): r for r in N['batches']}
for b in ('B1', 'B2'):
    tp, tn, fp, fn = counts(s2, 1.0, bt == b)
    assert (tp, tn, fp, fn) == (bat[('v2', b)]['tp'], bat[('v2', b)]['tn'], bat[('v2', b)]['fp'], bat[('v2', b)]['fn'])


def thr_at_spec(s, target):
    for t in np.sort(np.unique(np.r_[s, s + 1e-9])):
        tp, tn, _, _ = counts(s, t)
        if tn / nneg >= target:
            return t, tp
    return None


# best balanced operating point (criterion stated in the panel)
def best_balanced(s):
    best = (0, None, None, None)
    for t in np.unique(s):
        tp, tn, _, _ = counts(s, t)
        if min(tp / npos, tn / nneg) > best[0]:
            best = (min(tp / npos, tn / nneg), t, tp / npos, tn / nneg)
    return best


bb2, bb3 = best_balanced(s2), best_balanced(s3)
assert bb2[0] < 0.90 and bb3[0] < 0.90

# ======================================================================= a: threshold sweep
thr = np.unique(np.r_[np.geomspace(0.1, 50, 260), [1.0]])
YA, HA = 15.0, 38.0
xs = [13.0, 58.0, 103.0, 145.0]
AWID = 33.0
axs = [axes_mm(fig, x, YA, AWID, HA) for x in xs]
MODELS = [('v2', s2), ('v3b', s3)]
for ax, which in zip(axs[:2], ['sens', 'spec']):
    for key, s in MODELS:
        est, lo, hi = [], [], []
        for t in thr:
            tp, tn, _, _ = counts(s, t)
            k, n = (tp, npos) if which == 'sens' else (tn, nneg)
            est.append(k / n); a_, b_ = wilson(k, n); lo.append(a_); hi.append(b_)
        ax.fill_between(thr, lo, hi, color=M[key], alpha=0.16, lw=0, zorder=2)
        ax.plot(thr, est, color=M[key], lw=1.5, ls=LS[key], zorder=3)
    for key, s in MODELS:
        tp, tn, _, _ = counts(s, 1.0)
        v, (lo_, hi_) = ((tp / npos, wilson(tp, npos)) if which == 'sens' else (tn / nneg, wilson(tn, nneg)))
        ci_mark(ax, 1.0, v, lo_, hi_, key, filled=False, ms=4.4, orient='v', z=5)
# PPV and NPV of v2 by batch (prevalence 16% in B1, 62% in B2)
BST = {'B1': dict(ls='-', label='B1 consecutive'), 'B2': dict(ls=(0, (1.2, 1.4)), label='B2 enriched')}
for ax, which in zip(axs[2:], ['ppv', 'npv']):
    for b in ('B1', 'B2'):
        sel = bt == b
        est, lo, hi, tt = [], [], [], []
        for t in thr:
            tp, tn, fp, fn = counts(s2, t, sel)
            k, n = (tp, tp + fp) if which == 'ppv' else (tn, tn + fn)
            if n == 0:
                continue
            est.append(k / n); a_, b_ = wilson(k, n); lo.append(a_); hi.append(b_); tt.append(t)
        ax.fill_between(tt, lo, hi, color=M['v2'], alpha=0.13, lw=0, zorder=2)
        ax.plot(tt, est, color=M['v2'], lw=1.5, ls=BST[b]['ls'], zorder=3)
        r = bat[('v2', b)]
        v = r[which]; ci = r[which + '_ci']
        ax.plot([1.0, 1.0], ci, color=M['v2'], lw=1.2, zorder=5)
        ax.plot([1.0], [v], marker='o', ms=4.4, mfc=WHITE, mec=M['v2'], mew=1.0, zorder=6)
for ax, lab in zip(axs, ['Sensitivity', 'Specificity', 'PPV, v2', 'NPV, v2']):
    ax.axvline(1.0, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.set_xscale('log'); ax.set_xlim(0.1, 50); ax.set_ylim(0, 1.0)
    plain_log(ax, 'x', [0.1, 1, 10], ['0.1', '1', '10']); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ['0', '0.25', '0.50', '0.75', '1'])
    ax.set_ylabel(lab + ' (95% CI)', labelpad=1.5)
    ax.set_xlabel('Threshold (mL)', labelpad=1.5)
    grid(ax, 'both')
fig.legend(handles=[Line2D([], [], color=M['v2'], lw=1.5, ls=LS['v2'], marker='o', ms=4.0, mfc=WHITE, mec=M['v2'], mew=1.0, label='v2, total volume'),
                    Line2D([], [], color=M['v3b'], lw=1.5, ls=LS['v3b'], marker='s', ms=4.0, mfc=WHITE, mec=M['v3b'], mew=1.0, label='v3b, any-class volume'),
                    Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 2)), label='Prespecified 1 mL rule'),
                    Line2D([], [], color=M['v2'], lw=1.5, ls=BST['B1']['ls'], label='v2 in B1 (consecutive)'),
                    Line2D([], [], color=M['v2'], lw=1.5, ls=BST['B2']['ls'], label='v2 in B2 (enriched)')],
           loc='upper left', bbox_to_anchor=(11.0 * MM / W2, 1 - (YA + HA + 11.5) * MM / (H * MM)), ncol=5, borderaxespad=0,
           handlelength=2.4, columnspacing=1.2, handletextpad=0.4)
ptitle(fig, axs[0], 'a', 'Sensitivity and specificity across thresholds, and predictive values by batch',
       sub=nminus(f'No threshold gives sensitivity and specificity ≥0.90 (best balanced, chosen on these data: v2 {bb2[0]:.2f} at {bb2[1]:.1f} mL; v3b {bb3[0]:.2f} at {bb3[1]:.1f} mL)'),
       dx_mm=11.0, dy_mm=3.0)

# ======================================================================= b: volume by outcome (v2, 1 mL)
YB, HB = 84.0, 36.0
axB = axes_mm(fig, 40, YB, 40, HB)
cls = [('Majority positive, ≥1 mL', (y == 1) & (s2 >= 1)), ('Majority positive, <1 mL', (y == 1) & (s2 < 1)),
       ('Majority negative, ≥1 mL', (y == 0) & (s2 >= 1)), ('Majority negative, <1 mL', (y == 0) & (s2 < 1))]
fp = N['fp_profile']; fn = N['fn_profile']
rng = np.random.default_rng(8)
axB.axhspan(1.5, 3.5, color=BAND, lw=0, zorder=0)
stats_ = {}
for i, (c, mask) in enumerate(cls):
    v = s2[mask]
    vv = np.clip(v, 0.01, None)
    axB.scatter(vv, i + rng.uniform(-0.25, 0.25, len(v)), s=5, color=M['v2'], alpha=0.45, lw=0, zorder=2)
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    stats_[c] = (len(v), q1, q2, q3)
    axB.add_patch(Rectangle((max(q1, 0.01), i - 0.32), q3 - max(q1, 0.01), 0.64, fill=False, ec=INK, lw=1.0, zorder=4))
    axB.plot([q2, q2], [i - 0.32, i + 0.32], color=INK, lw=1.8, zorder=5, solid_capstyle='butt')
    axB.text(-0.03, i - 0.17, f'{c} ({len(v)})', transform=axB.get_yaxis_transform(), ha='right', va='center', fontsize=T7, color=INK)
    axB.text(-0.03, i + 0.25, f'median {q2:.2f} mL' if q2 < 1 else f'median {q2:.1f} mL', transform=axB.get_yaxis_transform(), ha='right', va='center',
             fontsize=T7, color=INK2)
assert stats_[cls[2][0]][0] == fp['n_fp'] == 115 and stats_[cls[1][0]][0] == fn['n_fn'] == 37
assert abs(stats_[cls[2][0]][2] - fp['fp_vol_median']) < 1e-9 and abs(stats_[cls[1][0]][2] - fn['fn_vol_median']) < 1e-9
assert np.allclose([stats_[cls[2][0]][1], stats_[cls[2][0]][3]], fp['fp_vol_q'])
n10 = int((s2[(y == 0) & (s2 >= 1)] > 10).sum())
assert n10 == fp['fp_over10'] == 15 and int((s2[(y == 1)] == 0).sum()) == fn['fn_zero_volume'] == 1
XMAX = 300.0
assert s2.max() < XMAX
axB.axvline(1.0, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=1)
# bracket over the false-positive volumes above 10 mL
fpmax = float(s2[(y == 0)].max())
axB.plot([10, 10, fpmax, fpmax], [2.42, 2.52, 2.52, 2.42], color=INK, lw=1.0, zorder=6, clip_on=False)
axB.text(np.sqrt(10 * fpmax), 2.58, f'{n10} of 115 >10 mL', fontsize=T7, color=INK, ha='center', va='top')
axB.set_xscale('log'); axB.set_xlim(0.006, XMAX)
plain_log(axB, 'x', [0.01, 0.1, 1, 10, 100], ['≤0.01', '0.1', '1', '10', '100']); axB.xaxis.set_minor_locator(NullLocator())
axB.set_ylim(3.6, -0.6); axB.set_yticks([])
axB.spines['left'].set_visible(False)
axB.set_xlabel('v2 predicted volume (mL)')
grid(axB, 'x')
ptitle(fig, axB, 'b', 'Predicted volume by outcome at the 1 mL rule', sub='v2 against the reader majority; boxes: median and IQR',
       dx_mm=38.0, dy_mm=3.0)

# ======================================================================= c: sensitivity by subtype present
axC = axes_mm(fig, 103, YB, 28, HB)
subs = [('ICH', 'Any ICH')] + [(s, s + (DG if s in SURG else '')) for s in SUB] + [('ChronicBleed', 'Chronic bleed')]
sb = CQ['sens_by_subtype']
fnsub = fn['by_subtype']
for i, (lab, name) in enumerate(subs):
    pos = m[f'ref_{lab}'].values == 1
    n_ = int(pos.sum())
    k2 = int((s2[pos] >= 1).sum()); k3 = int((s3[pos] >= 1).sum())
    if lab != 'ICH':
        assert (k2, n_) == (sb[lab]['k'], sb[lab]['n']) and n_ - k2 == fnsub[lab]
    else:
        assert (k2, n_) == (168, 205)
    if i % 2 == 0:
        axC.axhspan(i - 0.5, i + 0.5, color=BAND, lw=0, zorder=0)
    for off, k_, key in [(-0.18, k2, 'v2'), (0.18, k3, 'v3b')]:
        lo, hi = wilson(k_, n_)
        ci_mark(axC, k_ / n_, i + off, lo, hi, key, filled=True, ms=4.0)
    axC.text(-0.04, i, name, transform=axC.get_yaxis_transform(), ha='right', va='center', fontsize=T7, color=INK)
    axC.text(1.04, i, f'{n_ - k2}/{n_}', transform=axC.get_yaxis_transform(), ha='left', va='center', fontsize=T7, color=INK)
axC.text(1.04, -0.65, 'v2 missed', transform=axC.get_yaxis_transform(), ha='left', va='bottom', fontsize=T7, color=INK2)
axC.set_ylim(len(subs) - 0.5, -0.5); axC.set_yticks([]); axC.spines['left'].set_visible(False)
axC.set_xlim(0.3, 1.02); axC.set_xticks([0.4, 0.6, 0.8, 1.0], ['0.4', '0.6', '0.8', '1'])
axC.set_xlabel('Sensitivity at 1 mL (95% CI)')
grid(axC, 'x')
ptitle(fig, axC, 'c', 'Any-ICH detection by subtype present', sub='Groups overlap', dx_mm=20.0, dy_mm=3.0)

# ======================================================================= d: sensitivity at fixed specificity
axD = axes_mm(fig, 159, YB, 18, HB)
targets = [0.80, 0.90, 0.95]
rowsD = []
for j, tgt in enumerate(targets):
    for off, key, s, store in [(-0.17, 'v2', s2, N['sens_at_spec_v2']), (0.17, 'v3b', s3, N['sens_at_spec_v3b'])]:
        t, tp = thr_at_spec(s, tgt)
        assert abs(tp / npos - store[str(tgt) if str(tgt) in store else f'{tgt}']) < 1e-12
        lo, hi = wilson(tp, npos)
        ci_mark(axD, tp / npos, j + off, lo, hi, key, filled=True, ms=4.0)
        rowsD.append((tgt, key, round(t, 2), tp, round(tp / npos, 3)))
    if j % 2 == 0:
        axD.axhspan(j - 0.5, j + 0.5, color=BAND, lw=0, zorder=0)
    axD.text(-0.08, j, f'{tgt:.2f}', transform=axD.get_yaxis_transform(), ha='right', va='center', fontsize=T7, color=INK)
axD.text(-0.08, -0.62, 'Specificity', transform=axD.get_yaxis_transform(), ha='right', va='bottom', fontsize=T7, color=INK2)
axD.set_ylim(len(targets) - 0.5, -0.5); axD.set_yticks([]); axD.spines['left'].set_visible(False)
axD.set_xlim(0.25, 0.85); axD.set_xticks([0.3, 0.5, 0.7], ['0.3', '0.5', '0.7'])
axD.set_xlabel('Sensitivity (95% CI)\nexploratory; cut-offs\nfrom these data', linespacing=1.05)
grid(axD, 'x')
ptitle(fig, axD, 'd', 'Fixed specificity', dx_mm=14.0, dy_mm=5.5)
axC.legend(handles=[Line2D([], [], marker='o', ms=4.0, ls='', mfc=M['v2'], mec=M['v2'], label='v2'),
                    Line2D([], [], marker='s', ms=4.0, ls='', mfc=M['v3b'], mec=M['v3b'], label='v3b')],
           loc='upper left', bbox_to_anchor=(-0.05, -0.2), ncol=2, borderaxespad=0, columnspacing=1.0, handletextpad=0.3)

print('overflow', overflow_mm(fig))
print('best balanced v2', bb2, 'v3b', bb3)
print('v3b 1 mL', tp1b, npos, tn1b, nneg)
print('stats', {k: tuple(round(x, 3) for x in v) for k, v in stats_.items()})
print('fixed spec', rowsD)
for b in ('B1', 'B2'):
    print(b, {k: round(bat[('v2', b)][k], 3) for k in ('ppv', 'npv', 'sens', 'spec')})
save(fig, 'SF_operating', 'supp')
