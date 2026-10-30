"""SF_bhsd_volume - Volumetric agreement on the BHSD test set: predicted against reference volume, ratio (log-scale Bland-Altman)
and a table of additive agreement in milliliters and ICC, with and without the QC gate. The additive Bland-Altman plot of v2 is in Fig. 6c."""
from nc_s2 import *

H = 182
fig = plt.figure(figsize=(W2, H * MM))
k, v1, nm, v2b, v3b, v3 = bhsd()
bha = {(r['model'], r['cohort']): r for r in S['bhsd_agreement']}
isq = np.array([c == QCID for c in k])
y = v1.reference_volume_ml.values
assert np.allclose(y, nm.ref_ml.values, atol=1e-6) and np.allclose(y, v3b.ANY_ref_ml.values, atol=1e-6)
models = [('v1', 'v1', v1.prediction_volume_ml.values, 'v1', 'prespecified'),
          ('v2', 'v2 no post-mask', nm.pred_ml.values, 'v2', 'post hoc'),
          ('v3b, any class', 'v3b (any)', v3b.ANY_pred_ml.values, 'v3b', 'post hoc')]
OFF = 0.05
xs = [16.0, 77.0, 138.0]
SZ = 40.0
LIM = 8.0


def ratio_stats(p, r):
    ok = (p > 0) & (r > 0)
    lr = np.log2(p[ok] / r[ok])
    mu, sd = lr.mean(), lr.std(ddof=1)
    return ok, lr, mu, sd, 2 ** mu, 2 ** (mu - 1.96 * sd), 2 ** (mu + 1.96 * sd)


RS = {}
for j, ((lab, key, p, mk, tag_), x0) in enumerate(zip(models, xs)):
    col = M[mk]
    a_all, a_q = bha[(key, 'all 96')], bha[(key, 'QC-accepted 95')]
    # ---------------- row a: predicted vs reference
    ax = axes_mm(fig, x0, 12, SZ, SZ)
    lo_, hi_ = 0.04, 3000
    ax.plot([lo_, hi_], [lo_, hi_], color=MUTED, lw=1.0, zorder=1)
    ax.scatter(y[~isq] + OFF, p[~isq] + OFF, s=11, color=col, marker=MK[mk], alpha=0.85, lw=0, zorder=3)
    ax.plot(y[isq] + OFF, p[isq] + OFF, marker='*', ms=8.5, mfc=CRIT, mec=CRIT, ls='', zorder=5)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_); ax.set_aspect('equal')
    plain_log(ax, 'x', [0.1, 1, 10, 100, 1000], ['0.1', '1', '10', '100', '1,000'])
    plain_log(ax, 'y', [0.1, 1, 10, 100, 1000], ['0.1', '1', '10', '100', '1,000'])
    ax.set_xlabel('Reference + 0.05 (mL)')
    ax.set_ylabel('Predicted + 0.05 (mL)')
    grid(ax, 'both')
    hi_pred = p[isq][0] > 200
    ax.annotate(f'flagged scan:\n{p[isq][0]:,.1f} mL', (y[isq][0] + OFF, p[isq][0] + OFF), xytext=(5, -1) if hi_pred else (-4, 7), textcoords='offset points',
                fontsize=F7, color=CRIT, ha='left', va='top' if hi_pred else 'bottom', linespacing=1.1)
    ptitle(fig, ax, 'a' if j == 0 else '', f'{lab} ({tag_})', dx_mm=12.5 if j == 0 else 0.0, dy_mm=3.2)
    # ---------------- row b: ratio vs geometric mean (log2)
    axb = axes_mm(fig, x0, 74, SZ, 38)
    ok, lr, mu, sd, g, l_, h_ = ratio_stats(p, y)
    okq, lrq, muq, sdq, gq, lq, hq = ratio_stats(p[~isq], y[~isq])
    assert ok.sum() == a_all['n_logratio'] and okq.sum() == a_q['n_logratio']
    assert abs(g - a_all['gm_ratio']) < 1e-9 and abs(l_ - a_all['ratio_loa'][0]) < 1e-9 and abs(h_ - a_all['ratio_loa'][1]) < 1e-9
    assert abs(gq - a_q['gm_ratio']) < 1e-9 and abs(lq - a_q['ratio_loa'][0]) < 1e-9 and abs(hq - a_q['ratio_loa'][1]) < 1e-9
    RS[lab] = (int(ok.sum()), int((p == 0).sum()), g, l_, h_, int(okq.sum()), gq, lq, hq)
    gmv = np.sqrt(p[ok] * y[ok])
    flag_ok = isq[ok]
    axb.axhline(0, color=MUTED, lw=1.0, zorder=1)
    axb.axhline(mu, color=INK, lw=1.4, zorder=2)
    for v_ in (mu - 1.96 * sd, mu + 1.96 * sd):
        axb.axhline(v_, color=INK, lw=1.0, ls=(0, (3.5, 2)), zorder=2)
    axb.axhline(muq, color=INK2, lw=1.0, ls=(0, (6, 1.5, 1.2, 1.5)), zorder=2)
    for v_ in (muq - 1.96 * sdq, muq + 1.96 * sdq):
        axb.axhline(v_, color=INK2, lw=1.0, ls=(0, (1.2, 1.4)), zorder=2)
    assert ((lr[~isq[ok]] > -LIM) & (lr[~isq[ok]] < LIM)).all()
    axb.scatter(gmv[~flag_ok], lr[~flag_ok], s=11, color=col, marker=MK[mk], alpha=0.85, lw=0, zorder=3)
    if flag_ok.any():
        fx, fy = gmv[flag_ok][0], lr[flag_ok][0]
        axb.plot([fx], [LIM - 0.55], marker='^', ms=6, mfc=CRIT, mec=CRIT, ls='', zorder=5)
        axb.text(fx * 1.35, LIM - 0.55, f'{2 ** fy:,.0f}', fontsize=F7, color=CRIT, ha='left', va='center')
    axb.set_xscale('log'); axb.set_xlim(0.08, 300); axb.set_ylim(-LIM, LIM)
    plain_log(axb, 'x', [0.1, 1, 10, 100], ['0.1', '1', '10', '100'])
    axb.set_yticks([-6, -4, -2, 0, 2, 4, 6], ['1/64', '1/16', '1/4', '1', '4', '16', '64'])
    axb.set_xlabel('Geometric mean volume (mL)'); axb.set_ylabel('Predicted / reference')
    grid(axb, 'y')
    nz = int((p == 0).sum())
    ptitle(fig, axb, 'b' if j == 0 else '', f'{lab}: {int(ok.sum())} of 96 scans', sub=(f'{nz} zero {"prediction" if nz == 1 else "predictions"} excluded' if nz else 'no zero predictions'),
           dx_mm=12.5 if j == 0 else 0.0, dy_mm=3.2)
key_row(fig, 3.0, 123.5, [('marker', CRIT, 'flagged scan, off scale (its ratio printed)', dict(marker='^', ms=5.0, mec=CRIT)), ('line', MUTED, 'ratio 1', dict(lw=1.0)),
                          ('line', INK, 'geometric mean ratio, all scans', dict(lw=1.4))], H, sw=5.0, gap_mm=4.0)
key_row(fig, 3.0, 127.3, [('line', INK, '95% limits of agreement, all scans', dict(ls=(0, (3.5, 2)), lw=1.0)),
                          ('line', INK2, 'geometric mean ratio, QC-gated', dict(ls=(0, (6, 1.5, 1.2, 1.5)), lw=1.0)),
                          ('line', INK2, '95% limits, QC-gated', dict(ls=(0, (1.2, 1.4)), lw=1.0))], H, sw=5.0, gap_mm=4.0)

# ======================================================================= c: additive agreement and ICC for all three models (table)
p2 = nm.pred_ml.values
a2 = bha[('v2 no post-mask', 'all 96')]
assert abs((p2 - y).mean() - a2['bias']) < 1e-9
axt = axes_mm(fig, 16, 142, 160, 26); axt.axis('off')
ptitle(fig, axt, 'c', 'Agreement in milliliters and ICC(A,1), all scans and after the QC gate', dx_mm=12.5, dy_mm=3.2)
tx = [16.0, 62.0, 86.0, 124.0, 172.0]
heads = ['Model', 'Scans', 'Bias (mL)', '95% limits of agreement (mL)', 'ICC(A,1) (95% CI)']
ty0 = 143.0
for xx, hd, ha in zip(tx, heads, ['left', 'right', 'right', 'right', 'right']):
    fig_text(fig, xx, ty0, hd, H, color=INK, fontweight='bold', ha=ha)
ri = 0
for lab, key, p, mk, tag_ in models:
    for coh, nlab in [('all 96', 'all scans'), ('QC-accepted 95', 'QC-gated')]:
        r = bha[(key, coh)]
        yy = ty0 + 4.6 + ri * 3.7
        vals = [f"{lab.split(',')[0]}, {nlab}", f"{r['n']}", nminus(f"{r['bias']:+.1f}"), nminus(f"{r['LoA_lo']:.1f} to {r['LoA_hi']:.1f}"),
                nminus(f"{r['ICC']:.3f} ({r['ICC_ci'][0]:.3f} to {r['ICC_ci'][1]:.3f})")]
        for xx, v_, ha in zip(tx, vals, ['left', 'right', 'right', 'right', 'right']):
            fig_text(fig, xx, yy, v_, H, color=INK if nlab == 'all scans' else INK2, ha=ha)
        ri += 1
fig_text(fig, 16.0, ty0 + 4.6 + 6 * 3.7 + 1.0, 'v3b: any-class volume. Limits of agreement: bias ± 1.96 s.d. of the differences; ICC intervals: bootstrap, 10,000 resamples.', H,
         color=INK2)
fig_text(fig, 3.0, 178.0, 'Red star or triangle: the non-HU-encoded scan flagged by the QC gate (reference 0.11 mL). Marker shape identifies the model. '
         'v2 and v3b are post hoc models.', H)
for kk, v in RS.items():
    print(kk, 'n', v[0], 'zero preds', v[1], 'all', np.round(v[2:5], 3), 'gated n', v[5], np.round(v[6:9], 3))
print('overflow', overflow_mm(fig))
save(fig, 'SF_bhsd_volume', 'supp')
check_qa('SF_bhsd_volume')
