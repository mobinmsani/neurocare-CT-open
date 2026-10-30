"""SF_subtype_roc - Subtype ROC and precision-recall curves on CQ500 (all 489 scans against the reader majority).

v3b class volume (post hoc six-class model) and v2 total predicted volume (binary model; descriptive, no type information).
AUC 95% CIs from the stored analyses (verify_original.json for v3b; cq500_results.json table_detection for v2);
AP 95% CIs computed here by scan-level bootstrap (10,000 resamples, seed 20261028), printed and written to out/derived/sf_subtype_roc_ap_ci.csv.
Reader points: each reader against the concordant reads of the other two (different subsets; descriptive).
"""
from nc_s1 import *
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import re as _re

H = 136
fig = plt.figure(figsize=(W2, H * MM))
m = load_cq500()
loo = pd.DataFrame(N['loo'])
td = {r['Finding']: r for r in CQ['table_detection']}
n_ = len(m)
rng = np.random.default_rng(20261028)
B = 10000
IDX = rng.integers(0, n_, (B, n_))


def ap_fast(y, s):
    o = np.argsort(-s, kind='mergesort'); s_ = s[o]; y_ = y[o]
    d = np.where(np.diff(s_))[0]
    ti = np.r_[d, y_.size - 1]
    tps = np.cumsum(y_)[ti]; fps = 1 + ti - tps
    if tps[-1] == 0:
        return np.nan
    prec = tps / (tps + fps); rec = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def parse_ci(s):
    a, b, c = [float(x) for x in _re.findall(r'[\d.]+', s)[:3]]
    return a, b, c


X0, AW, GAP = 13.0, 30.0, 3.6
YR, YP = 16.0, 75.0
derived = []
for j, sub in enumerate(SUB):
    x = X0 + j * (AW + GAP)
    axR = axes_mm(fig, x, YR, AW, AW)
    axP = axes_mm(fig, x, YP, AW, AW)
    y = m['ref_' + sub].values.astype(int)
    s3 = m[sub + '_ml'].values; s2 = m.pred_volume_ml.values
    pos = int(y.sum())
    v = VO['cq_v3b']['all'][sub]
    a3, a2 = roc_auc_score(y, s3), roc_auc_score(y, s2)
    ap3, ap2 = average_precision_score(y, s3), average_precision_score(y, s2)
    assert abs(a3 - v['auc']) < 1e-12 and abs(ap3 - v['ap']) < 1e-12 and pos == v['pos']
    assert abs(ap_fast(y, s3) - ap3) < 1e-12 and abs(ap_fast(y, s2) - ap2) < 1e-12
    a2s = parse_ci(td[sub]['AUC (95% CI)'])
    assert f'{a2:.3f}' == f'{a2s[0]:.3f}'
    a3ci = v['auc_ci']
    apci = {}
    for key, s in (('v3b', s3), ('v2', s2)):
        bs = np.array([ap_fast(y[i], s[i]) for i in IDX])
        apci[key] = np.nanpercentile(bs, [2.5, 97.5])
    derived += [dict(subtype=sub, score='v3b class volume', AP=ap3, AP_lo=apci['v3b'][0], AP_hi=apci['v3b'][1], AUC=a3, AUC_lo=a3ci[0], AUC_hi=a3ci[1]),
                dict(subtype=sub, score='v2 total volume', AP=ap2, AP_lo=apci['v2'][0], AP_hi=apci['v2'][1], AUC=a2, AUC_lo=a2s[1], AUC_hi=a2s[2])]
    # ---------------- ROC
    axR.plot([0, 1], [0, 1], color='#d6d3cc', lw=1.0, ls=(0, (3, 2)), zorder=1)
    for s, key in ((s2, 'v2'), (s3, 'v3b')):
        f, t, _ = roc_curve(y, s)
        axR.plot(f, t, color=M[key], lw=1.5, ls=LS[key], zorder=3 if key == 'v3b' else 2, solid_joinstyle='round')
    se3, sp3 = v['sens1'] / pos, v['spec1'] / v['neg']
    se2 = ((s2 >= 1) & (y == 1)).sum() / pos; sp2 = ((s2 < 1) & (y == 0)).sum() / (n_ - pos)
    axR.plot([1 - sp2], [se2], marker=MK['v2'], ms=4.2, mfc=WHITE, mec=M['v2'], mew=1.0, ls='', zorder=5)
    axR.plot([1 - sp3], [se3], marker=MK['v3b'], ms=4.2, mfc=WHITE, mec=M['v3b'], mew=1.0, ls='', zorder=5)
    for q, r in loo[(loo.label == sub) & (loo.score == sub)].set_index('reader').iterrows():
        axR.plot([1 - r.reader_spec], [r.reader_sens], marker='D', ms=3.6, ls='', zorder=6, **RD[q])
    axR.set_xlim(-0.01, 1.01); axR.set_ylim(-0.01, 1.01); axR.set_aspect('equal')
    axR.set_xticks([0, 0.5, 1], ['0', '0.5', '1']); axR.set_yticks([0, 0.5, 1], ['0', '0.5', '1'] if j == 0 else [])
    axR.set_xlabel('1 − specificity', labelpad=1.0)
    grid(axR, 'both')
    fig_text(fig, x + AW / 2, YR - 5.2, sub + (DG if sub in SURG else ''), H, color=INK, ha='center', fontweight='bold')
    fig_text(fig, x + AW / 2, YR - 1.8, f'{pos} of {n_} ({100 * pos / n_:.0f}%)', H, color=INK2, ha='center')
    fig_text(fig, x, YR + AW + 9.6, nminus(f'v3b {a3:.2f} ({a3ci[0]:.2f}–{a3ci[1]:.2f})'), H, color=INK, ha='left')
    fig_text(fig, x, YR + AW + 12.8, nminus(f'v2 {a2:.2f} ({a2s[1]:.2f}–{a2s[2]:.2f})'), H, color=INK2, ha='left')
    # ---------------- PR
    prev = pos / n_
    axP.axhline(prev, color=INK2, lw=1.0, ls=(0, (1.2, 1.6)), zorder=1)
    for s, key in ((s2, 'v2'), (s3, 'v3b')):
        pr, rc, _ = precision_recall_curve(y, s)
        axP.plot(rc, pr, color=M[key], lw=1.5, ls=LS[key], zorder=3 if key == 'v3b' else 2, drawstyle='steps-post', solid_joinstyle='round')
    ppv3 = v['sens1'] / (v['sens1'] + (v['neg'] - v['spec1']))
    ppv2 = ((s2 >= 1) & (y == 1)).sum() / max((s2 >= 1).sum(), 1)
    axP.plot([se2], [ppv2], marker=MK['v2'], ms=4.2, mfc=WHITE, mec=M['v2'], mew=1.0, ls='', zorder=5)
    axP.plot([se3], [ppv3], marker=MK['v3b'], ms=4.2, mfc=WHITE, mec=M['v3b'], mew=1.0, ls='', zorder=5)
    axP.set_xlim(-0.01, 1.01); axP.set_ylim(-0.01, 1.01); axP.set_aspect('equal')
    axP.set_xticks([0, 0.5, 1], ['0', '0.5', '1']); axP.set_yticks([0, 0.5, 1], ['0', '0.5', '1'] if j == 0 else [])
    axP.set_xlabel('Recall', labelpad=1.0)
    grid(axP, 'both')
    fig_text(fig, x, YP + AW + 9.6, nminus(f"v3b {ap3:.2f} ({apci['v3b'][0]:.2f}–{apci['v3b'][1]:.2f})"), H, color=INK, ha='left')
    fig_text(fig, x, YP + AW + 12.8, nminus(f"v2 {ap2:.2f} ({apci['v2'][0]:.2f}–{apci['v2'][1]:.2f})"), H, color=INK2, ha='left')
    if j == 0:
        axR.set_ylabel('Sensitivity', labelpad=1.5)
        axP.set_ylabel('Precision (PPV)', labelpad=1.5)
        ptitle(fig, axR, 'a', 'ROC curves against all other scans', dx_mm=11.0, dy_mm=9.6)
        ptitle(fig, axP, 'b', 'Precision–recall at observed prevalences', dx_mm=11.0, dy_mm=3.2)
        fig_text(fig, x - 11.0, YR + AW + 9.6, 'AUC', H, color=INK, ha='left', fontweight='bold')
        fig_text(fig, x - 11.0, YP + AW + 9.6, 'AP', H, color=INK, ha='left', fontweight='bold')
    print(sub, pos, f'AUC v3b {a3:.3f} {a3ci} v2 {a2:.3f} {a2s}  AP v3b {ap3:.3f} {apci["v3b"]} v2 {ap2:.3f} {apci["v2"]}  '
          f'v3b@1 se {se3:.3f} sp {sp3:.3f} ppv {ppv3:.3f}  v2@1 se {se2:.3f} sp {sp2:.3f} ppv {ppv2:.3f}')
(ROOT / 'out' / 'derived').mkdir(exist_ok=True, parents=True)
pd.DataFrame(derived).to_csv(ROOT / 'out' / 'derived' / 'sf_subtype_roc_ap_ci.csv', index=False)

YL = YP + AW + 17.0
fig.legend(handles=[Line2D([], [], color=M['v3b'], lw=1.5, ls=LS['v3b'], label='v3b, class volume'),
                    Line2D([], [], color=M['v2'], lw=1.5, ls=LS['v2'], label='v2, total predicted volume (binary model; descriptive)'),
                    Line2D([], [], color=INK2, lw=1.0, ls=(0, (1.2, 1.6)), label='Prevalence (precision of a non-informative score)')],
           loc='upper left', bbox_to_anchor=(2.0 * MM / W2, 1 - YL * MM / (H * MM)), ncol=3, borderaxespad=0, handlelength=2.4, columnspacing=1.2, handletextpad=0.4)
fig.legend(handles=[Line2D([], [], marker='s', ms=4.2, ls='', mfc=WHITE, mec=M['v3b'], mew=1.0, label='v3b at 1 mL'),
                    Line2D([], [], marker='o', ms=4.2, ls='', mfc=WHITE, mec=M['v2'], mew=1.0, label='v2 at 1 mL')] +
                   [Line2D([], [], marker='D', ms=3.6, ls='', label=f'Reader {q}', **RD[q]) for q in ('R1', 'R2', 'R3')],
           loc='upper left', bbox_to_anchor=(2.0 * MM / W2, 1 - (YL + 4.4) * MM / (H * MM)), ncol=5, borderaxespad=0, handlelength=1.2, columnspacing=1.2, handletextpad=0.4)
fig_text(fig, 2.0, YL + 12.0, 'Readers: each against the concordant reads of the other two, on the scans on which those two agreed (different subsets; descriptive). '
                              f'{DG} Surgical criterion.', H, color=INK2, ha='left')
print('overflow', overflow_mm(fig))
save(fig, 'SF_subtype_roc', 'supp')
