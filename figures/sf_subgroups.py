"""SF_subgroups - Discrimination and 1 mL operating characteristics by batch and selected-series spacing on CQ500 (v2, v3b).

Descriptive, unadjusted strata. Overall AUCs as in the main tables (original-pipeline bootstrap); stratum AUCs from
new_analyses_cq500.json (batches, acq; 10,000 resamples); sensitivity and specificity with Wilson 95% CIs.
"""
from nc_s1 import *

H = 93
fig = plt.figure(figsize=(W2, H * MM))

bt = {(r['model'], r['batch']): r for r in N['batches']}
aq = {(r['model'], r['stratum']): r for r in N['acq']}
t9 = [r for r in VO['cq_det_table9'] if r[0] == 'ICH'][0]
v3a = VO['cq_v3b']['all']['ANY']
overall = {'v2': dict(auc=t9[1], lo=t9[2], hi=t9[3], tp=168, fn=37, tn=169, fp=115),
           'v3b': dict(auc=v3a['auc'], lo=v3a['auc_ci'][0], hi=v3a['auc_ci'][1], tp=v3a['sens1'], fn=205 - v3a['sens1'], tn=v3a['spec1'], fp=284 - v3a['spec1'])}
assert abs(t9[1] - 0.806389) < 1e-5 and (round(t9[2], 3), round(t9[3], 3)) == (0.769, 0.844)
assert (round(v3a['auc'], 3), round(v3a['auc_ci'][0], 3), round(v3a['auc_ci'][1], 3)) == (0.818, 0.778, 0.856)
assert (round(bt[('v2', 'B1')]['auc'], 3), round(bt[('v2', 'B1')]['lo'], 3), round(bt[('v2', 'B1')]['hi'], 3)) == (0.829, 0.736, 0.910)
assert (round(bt[('v2', 'B2')]['auc'], 3), round(bt[('v2', 'B2')]['lo'], 3), round(bt[('v2', 'B2')]['hi'], 3)) == (0.821, 0.769, 0.869)
assert (aq[('v2', '<=1.25 mm')]['n'], aq[('v2', '<=1.25 mm')]['pos'], aq[('v2', '~5 mm')]['n']) == (25, 6, 411)


def obs(stratum):
    dz = aq[('v2', stratum)]['dz_range']
    f_ = lambda v: f'{v:.1f}' if abs(round(v, 1) - v) < 1e-9 else f'{v:.2f}'
    return f'{f_(dz[0])}–{f_(dz[1])} mm'


rows = [('hdr', 'All analyzed scans', None),
        ('row', 'All', ('all', None)),
        ('hdr', 'Batch', None),
        ('row', 'B1, consecutive', ('batch', 'B1')),
        ('row', 'B2, enriched', ('batch', 'B2')),
        ('hdr', 'Observed slice spacing of the selected series', None),
        ('row', obs('<=1.25 mm'), ('acq', '<=1.25 mm')),
        ('row', obs('2.5-3.75 mm'), ('acq', '2.5-3.75 mm')),
        ('row', obs('~5 mm'), ('acq', '~5 mm'))]
assert [r[1] for r in rows if r[2] and r[2][0] == 'acq'] == ['0.5–1.0 mm', '3.0–3.75 mm', '5.0–7.5 mm']


def rec(model, key):
    kind, k = key
    if kind == 'all':
        return overall[model]
    return bt[(model, k)] if kind == 'batch' else aq[(model, k)]


ypos, y = [], 0.0
for kind, lab, key in rows:
    ypos.append(y)
    y += 0.8 if kind == 'hdr' else 1.0
YMAX = y - 0.5
PY, PH = 9.0, 60.0
axA = axes_mm(fig, 52, PY, 38, PH)
axB = axes_mm(fig, 123, PY, 24, PH)
axC = axes_mm(fig, 154, PY, 24, PH)
LX = 3.0
for ax in (axA, axB, axC):
    ax.set_ylim(YMAX, -0.6); ax.set_yticks([]); ax.spines['left'].set_visible(False)
band = 0
for (kind, lab, key), yy in zip(rows, ypos):
    yfig = PY + PH * (yy + 0.6) / (YMAX + 0.6)
    if kind == 'hdr':
        fig_text(fig, LX, yfig + 1.2, lab, H, color=MUTED, ha='left', fontweight='bold')
        continue
    r2, r3 = rec('v2', key), rec('v3b', key)
    n_ = r2['tp'] + r2['fn'] + r2['tn'] + r2['fp']; pos = r2['tp'] + r2['fn']
    assert n_ == r3['tp'] + r3['fn'] + r3['tn'] + r3['fp'] and pos == r3['tp'] + r3['fn']
    if band % 2 == 0:
        for ax in (axA, axB, axC):
            ax.axhspan(yy - 0.5, yy + 0.5, color=BAND, lw=0, zorder=0)
    band += 1
    fig_text(fig, LX, yfig - 0.2, lab, H, color=INK, ha='left', fontweight='bold' if key[0] == 'all' else 'normal')
    fig_text(fig, LX, yfig + 2.8, f'{n_} scans; {pos} ICH ({100 * pos / n_:.0f}%)', H, color=INK2, ha='left')
    for off, r, key_m in [(-0.18, r2, 'v2'), (0.18, r3, 'v3b')]:
        ci_mark(axA, r['auc'], yy + off, r['lo'], min(r['hi'], 1.0), key_m, ms=4.0)
        axA.text(1.04, yy + off, nminus(f"{key_m} {r['auc']:.2f} ({r['lo']:.2f}–{min(r['hi'], 1.0):.2f})"), transform=axA.get_yaxis_transform(), fontsize=T7,
                 color=INK, ha='left', va='center')
        se_lo, se_hi = wilson(r['tp'], r['tp'] + r['fn']); sp_lo, sp_hi = wilson(r['tn'], r['tn'] + r['fp'])
        ci_mark(axB, r['tp'] / (r['tp'] + r['fn']), yy + off, se_lo, se_hi, key_m, ms=4.0)
        ci_mark(axC, r['tn'] / (r['tn'] + r['fp']), yy + off, sp_lo, sp_hi, key_m, ms=4.0)
        if key[0] == 'batch':
            assert np.allclose([se_lo, se_hi], r['sens_ci']) and np.allclose([sp_lo, sp_hi], r['spec_ci'])
for ax, ref in [(axA, overall['v2']['auc']), (axB, 168 / 205), (axC, 169 / 284)]:
    ax.axvline(ref, color=M['v2'], lw=1.0, ls=(0, (1.2, 1.6)), zorder=1)
axA.set_xlim(0.5, 1.0); axA.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ['0.5', '0.6', '0.7', '0.8', '0.9', '1'])
axA.set_xlabel('AUC for any ICH (95% CI)')
grid(axA, 'x')
for ax, lab in [(axB, 'Sensitivity (95% CI)'), (axC, 'Specificity (95% CI)')]:
    ax.set_xlim(0, 1.04); ax.set_xticks([0, 0.5, 1.0], ['0', '0.5', '1'])
    ax.set_xlabel(lab)
    grid(ax, 'x')
ptitle(fig, axA, 'a', 'Discrimination by batch and selected-series spacing', dx_mm=52 - LX, dy_mm=3.0)
ptitle(fig, axB, 'b', 'Sensitivity at 1 mL', dx_mm=0.0, dy_mm=3.0)
ptitle(fig, axC, 'c', 'Specificity at 1 mL', dx_mm=0.0, dy_mm=3.0)
fig.legend(handles=[Line2D([], [], marker='o', ms=4.0, ls='-', lw=1.2, color=M['v2'], mfc=M['v2'], mec=M['v2'], label='v2, total volume'),
                    Line2D([], [], marker='s', ms=4.0, ls='-', lw=1.2, color=M['v3b'], mfc=M['v3b'], mec=M['v3b'], label='v3b, any-class volume'),
                    Line2D([], [], color=M['v2'], lw=1.0, ls=(0, (1.2, 1.6)), label='Overall v2 estimate')],
           loc='upper left', bbox_to_anchor=(LX * MM / W2, 1 - (PY + PH + 9.5) * MM / (H * MM)), ncol=3, borderaxespad=0, columnspacing=1.4, handletextpad=0.4)
d2, d3 = N['batch_auc_diff_v2'], N['batch_auc_diff_v3b']
assert (round(d2['diff'], 3), round(d2['ci'][0], 3), round(d2['ci'][1], 3)) == (-0.008, -0.105, 0.098)
th2, th3 = aq[('v2', '<=1.25 mm')], aq[('v3b', '<=1.25 mm')]
fig_text(fig, LX, PY + PH + 18.0, nminus(f"Descriptive, unadjusted strata that differ in case mix. AUC difference, B2 − B1: v2 {d2['diff']:.3f} ({d2['ci'][0]:.3f} to {d2['ci'][1]:.3f}); "
                                            f"v3b {d3['diff']:.3f} ({d3['ci'][0]:.3f} to {d3['ci'][1]:.3f})."), H, color=INK2, ha='left')
fig_text(fig, LX, PY + PH + 21.4, f"0.5–1.0 mm stratum ({th2['pos']} positives): v2 detected {th2['tp']} of {th2['tp'] + th2['fn']} and cleared {th2['tn']} of {th2['tn'] + th2['fp']} negatives; "
                                  f"v3b {th3['tp']} of {th3['tp'] + th3['fn']} and {th3['tn']} of {th3['tn'] + th3['fp']}. Protocol categories: ≤1.25, 2.5–3.75 and about 5 mm.", H, color=INK2, ha='left')
print('overflow', overflow_mm(fig))
for kind, lab, key in rows:
    if kind == 'row':
        for mdl in ('v2', 'v3b'):
            r = rec(mdl, key)
            print(lab, mdl, round(r['auc'], 3), round(r['lo'], 3), round(min(r['hi'], 1), 3), 'sens %d/%d' % (r['tp'], r['tp'] + r['fn']), 'spec %d/%d' % (r['tn'], r['tn'] + r['fp']))
save(fig, 'SF_subgroups', 'supp')
