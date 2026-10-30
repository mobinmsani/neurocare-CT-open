"""Figure 6 - Volume measurement: agreement statistics are fragile and clinically coarse.

a  Single-scan influence on volume ICC(A,1): with and without the one scan (BHSD v1; Seg-CQ500 v2, BLAST-CT, CQ500 DICOM pipeline)
b  Leave-one-scan-out ICC(A,1) on raw and log-transformed volumes, log10(1 + V) (Seg-CQ500 v2; BHSD test v1)
c  Bland-Altman differences in milliliters, v2 on the BHSD test set (96 scans), against the 50 mL width of the ENRICH window
d  Intraparenchymal (IPH) volume at the 30 mL and 30-80 mL clinical thresholds (post hoc v3b vs reference IPH, QC-gated)
All numbers are read from the analysis outputs (ncdata); values derived here are printed and checked against them.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncstyle import *
from ncdata import *
from nc_c import *
sys.path.insert(0, str(Path(os.path.dirname(os.path.abspath(__file__))).parent / 'analysis'))
from common import icc_a1   # noqa: E402

HFIG = 124
fig = plt.figure(figsize=(W2, HFIG * MM))

k, v1, nm, v2b, v3b, v3 = bhsd()
seg = {r['model']: r for r in S['seg_agreement']}
bha = {(r['model'], r['cohort']): r for r in S['bhsd_agreement']}
m = load_cq500()
mq = m.set_index('name')
VR = {r['set']: r for r in json.load(open(OUT / 'volume_robust.json'))}
ROW1, ROW2 = 12.0, 74.0

# ======================================================================= a: single-scan influence on ICC
CX, CW = 61.0, 34.0
axA = axes_mm(fig, CX, ROW1, CW, 38)
cxf = lambda x_mm: (x_mm - CX) / CW
segx = pd.read_csv(OUT / 'tables/segcq500_excl108_sensitivity.csv')
segicc = lambda model, s_: float(segx[(segx.model == model) & (segx.set == s_)].ICC.iloc[0])
b_all, b_q = bha[('v1', 'all 96')], bha[('v1', 'QC-accepted 95')]
qpred = float(v1.loc[QCID, 'prediction_volume_ml'])
ct108_dicom = float(mq.loc['CQ500-CT-108', 'pred_volume_ml'])
ct108 = S['seg_ct108']
v108 = int(sum(mq.loc['CQ500-CT-108', f'R{q}:ICH'] for q in (1, 2, 3)))
cqp = S['cq500_pipeline_vs_segref']
items = [
    ('BHSD test · v1 (96 → 95)', f'1 scan not HU-encoded (v1 {comma(qpred)} mL)',
     (b_all['ICC'], *b_all['ICC_ci']), (b_q['ICC'], *b_q['ICC_ci']), 'v1'),
    ('Seg-CQ500 · v2 (43 → 42)', f"CT-108: mask {ct108['ref']:.1f} mL; {v108} of 3 readers positive",
     (seg['v2 no post-mask']['ICC'], *seg['v2 no post-mask']['ICC_ci']), (segicc('v2 no post-mask', 'excl108'), *X['seg_v2nm_excl108_icc_ci']), 'v2'),
    ('Seg-CQ500 · BLAST-CT (43 → 42)', 'same scan',
     (seg['BLAST-CT']['ICC'], *seg['BLAST-CT']['ICC_ci']), (segicc('BLAST-CT', 'excl108'), *X['seg_blast_excl108_icc_ci']), 'blast'),
    ('CQ500 DICOM · v2 (43 → 42)', f"CT-108: v2 {ct108_dicom:.1f} mL (DICOM), {ct108['pred']:.1f} mL (file)",
     (cqp['ICC'], *cqp['ICC_ci']), (segicc('v2 no post-mask, CQ500 DICOM pipeline', 'excl108'), *X['seg_dicom_excl108_icc_ci']), 'v2'),
]
for i, (grp, why, a, b, mk) in enumerate(items):
    y0 = i * 2.25
    col = M[mk]
    axA.plot([a[1], a[2]], [y0, y0], color=col, lw=1.0, alpha=0.6, zorder=2, solid_capstyle='round')
    axA.plot([b[1], b[2]], [y0 + 0.8, y0 + 0.8], color=col, lw=1.2, zorder=2, solid_capstyle='round')
    axA.plot([a[0]], [y0], marker=MK[mk], ms=4.6, mfc=WHITE, mec=col, mew=1.1, zorder=4, ls='')
    axA.plot([b[0]], [y0 + 0.8], marker=MK[mk], ms=4.8, mfc=col, mec=WHITE, mew=0.6, zorder=4, ls='')
    axA.annotate('', xy=(b[0], y0 + 0.66), xytext=(a[0], y0 + 0.14),
                 arrowprops=dict(arrowstyle='-|>', lw=1.0, color=INK2, mutation_scale=6, shrinkA=2.6, shrinkB=3.6), zorder=3)
    axA.text(a[0], y0 - 0.33, f'{a[0]:.2f}', fontsize=7, ha='center', va='bottom', color=INK2)
    axA.text(b[0], y0 + 1.12, f'{b[0]:.2f}', fontsize=7, ha='center', va='top', color=INK, fontweight='bold')
    axA.text(cxf(2.0), y0 + 0.05, grp, transform=axA.get_yaxis_transform(), fontsize=7, fontweight='bold', ha='left', va='center', color=INK)
    axA.text(cxf(2.0), y0 + 0.88, why, transform=axA.get_yaxis_transform(), fontsize=7, ha='left', va='center', color=INK2)
    print('a:', grp, 'all %.3f (%.3f-%.3f) -> without %.3f (%.3f-%.3f)' % (a[0], a[1], a[2], b[0], b[1], b[2]))
axA.set_ylim(3 * 2.25 + 1.75, -0.95)
axA.set_yticks([]); axA.spines['left'].set_visible(False)
axA.set_xlim(-0.1, 1.0); axA.set_xticks([0, 0.5, 1.0], ['0', '0.50', '1'])
axA.axvline(0, color='#cfcbc3', lw=1.0, zorder=0)
axA.set_xlabel('Volume ICC(A,1)')
grid(axA, 'x')
ptitle(fig, axA, 'a', 'One scan can change a volume ICC', sub='Open marker: all scans; filled: without the one scan', dx_mm=CX - 2.0)

# ======================================================================= b: leave-one-scan-out ICC, raw and log scale
segnm = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/segcq500_no_postmask.csv').set_index('case_id')
ks = sorted(segnm.index)
sets = [('Seg-CQ500 · v2', ks, segnm.loc[ks, 'pred_ml'].values, segnm.loc[ks, 'ref_ml'].values, 'Seg-CQ500, v2 (no post-mask)', 'v2', 'CT-108'),
        ('BHSD test · v1', list(v1.index), v1.prediction_volume_ml.values, v1.reference_volume_ml.values, 'BHSD test, v1', 'v1', 'non-HU scan')]
BX, BW, BH = 124.0, 52.0, 38.0
axB = axes_mm(fig, BX, ROW1, BW, BH)
bxf = lambda x_mm: (x_mm - BX) / BW
rng = np.random.default_rng(20261029)
yrow = 0.0
for lab, ids, p, y, vkey, mk, flag in sets:
    p = np.asarray(p, float); y = np.asarray(y, float)
    lp, ly = np.log10(1 + p), np.log10(1 + y)
    jr = np.array([icc_a1(np.delete(p, i), np.delete(y, i)) for i in range(len(y))])
    jl = np.array([icc_a1(np.delete(lp, i), np.delete(ly, i)) for i in range(len(y))])
    fr, fl = icc_a1(p, y), icc_a1(lp, ly)
    r_ = VR[vkey]
    assert abs(fr - r_['icc_raw']) < 1e-9 and abs(fl - r_['icc_log']) < 1e-9
    assert abs(jr.min() - r_['jackknife_icc_raw'][0]) < 1e-9 and abs(jr.max() - r_['jackknife_icc_raw'][1]) < 1e-9
    assert abs(jl.min() - r_['jackknife_icc_log'][0]) < 1e-9 and abs(jl.max() - r_['jackknife_icc_log'][1]) < 1e-9
    col = M[mk]
    for j, (vals, full, scale, filled) in enumerate([(jr, fr, 'raw', True), (jl, fl, 'log', False)]):
        yy = yrow + j * 0.9
        kmax = int(np.argmax(np.abs(vals - full)))
        jit = rng.uniform(-0.18, 0.18, len(vals))
        rest = np.ones(len(vals), bool); rest[kmax] = False
        axB.scatter(vals[rest], yy + jit[rest], s=7, marker=MK[mk], facecolor=col if filled else WHITE, edgecolor=col, lw=0.6, alpha=0.8, zorder=3)
        axB.plot([vals[kmax]], [yy], marker=MK[mk], ms=5.4, mfc=col if filled else WHITE, mec=INK, mew=1.0, ls='', zorder=5)
        axB.plot([full, full], [yy - 0.3, yy + 0.3], color=INK, lw=1.4, zorder=4, solid_capstyle='butt')
        axB.text(-2.0 / BW, yy, scale, transform=axB.get_yaxis_transform(), fontsize=7, ha='right', va='center', color=INK2)
        if scale == 'raw':
            axB.annotate(f'without {flag}', xy=(vals[kmax], yy), xytext=(vals[kmax] - 0.02, yy + 0.47), fontsize=7, ha='right', va='center', color=INK,
                         arrowprops=dict(arrowstyle='-', lw=0.8, color=INK2, shrinkA=0.5, shrinkB=2.5))
        print('b:', lab, scale, 'full %.3f, LOO range %.3f-%.3f, most influential %s -> %.3f' % (full, vals.min(), vals.max(), ids[kmax], vals[kmax]))
    axB.text(-2.0 / BW, yrow - 0.62, lab, transform=axB.get_yaxis_transform(), fontsize=7, ha='right', va='center', color=INK, fontweight='bold')
    yrow += 2.4
axB.set_ylim(yrow - 0.85, -1.0)
axB.set_yticks([]); axB.spines['left'].set_visible(False)
axB.set_xlim(-0.05, 1.0); axB.set_xticks([0, 0.5, 1.0], ['0', '0.50', '1'])
axB.set_xlabel('ICC(A,1) without one scan')
grid(axB, 'x')
ptitle(fig, axB, 'b', 'Single-scan influence is specific to the raw scale', sub='Points: each scan left out; bar: all scans', dx_mm=22.0)

# ======================================================================= c: Bland-Altman in mL, v2 on BHSD (96 scans)
isq = np.array([c == QCID for c in k])
y = v1.reference_volume_ml.values
p2 = nm.pred_ml.values
d2 = p2 - y; m2 = (p2 + y) / 2
a2 = bha[('v2 no post-mask', 'all 96')]
assert abs(d2.mean() - a2['bias']) < 1e-9 and len(d2) == a2['n'] == 96
assert abs(a2['LoA_lo'] - (d2.mean() - 1.96 * d2.std(ddof=1))) < 1e-6 and abs(a2['LoA_hi'] - (d2.mean() + 1.96 * d2.std(ddof=1))) < 1e-6
CX2, CW2 = 16.0, 54.0
axC = axes_mm(fig, CX2, ROW2, CW2, 36)
axC.axhline(0, color=MUTED, lw=1.0, zorder=1)
axC.axhline(a2['bias'], color=INK, lw=1.4, zorder=2)
for v_ in (a2['LoA_lo'], a2['LoA_hi']):
    axC.axhline(v_, color=INK, lw=1.0, ls=(0, (3.5, 2)), zorder=2)
axC.scatter(m2[~isq], d2[~isq], s=11, color=M['v2'], marker=MK['v2'], alpha=0.85, lw=0, zorder=3)
axC.plot(m2[isq], d2[isq], marker='*', ms=8.0, mfc=CRIT, mec=CRIT, ls='', zorder=5)
axC.set_xlim(0, 160); axC.set_ylim(-95, 65)
axC.set_xlabel('Mean of predicted and reference volume (mL)'); axC.set_ylabel('Predicted − reference (mL)')
grid(axC, 'y')
for v_, t in [(a2['LoA_hi'], f"{a2['LoA_hi']:+.1f}"), (a2['bias'], f"{a2['bias']:+.1f}"), (a2['LoA_lo'], f"{a2['LoA_lo']:+.1f}")]:
    axC.text(1.02, v_, t.replace('-', '−'), transform=axC.get_yaxis_transform(), fontsize=7, ha='left', va='center', color=INK)
span = round(a2['LoA_hi'], 1) - round(a2['LoA_lo'], 1)      # width of the limits as displayed (one decimal)
axC.text(0.98, 0.035, f"95% limits span {span:.1f} mL;\nENRICH window (30–80 mL) spans 50 mL", transform=axC.transAxes, fontsize=7, ha='right', va='bottom',
         color=INK, linespacing=1.2, path_effects=halo(2.0), zorder=6)
ptitle(fig, axC, 'c', 'v2 volume differences in milliliters', sub='BHSD test set, 96 scans; star: scan not HU-encoded', dx_mm=13.5)
print('c: bias %.2f, LoA %.2f to %.2f (span %.1f)' % (a2['bias'], a2['LoA_lo'], a2['LoA_hi'], span))

# ======================================================================= d: IPH volume at clinical thresholds (post hoc v3b, QC-gated)
DX, DW = 112.0, 34.0
axD = axes_mm(fig, DX, ROW2, DW, DW)
g = v3b.loc[[c for c in k if c != QCID]]
ri = g.IPH_ref_ml.values; pi = g.IPH_pred_ml.values
tg = TC['gated']
c30 = tg['iph_v3b_30']; cb = tg['iph_v3b_bins_30_80']
mis = (ri >= 30) != (pi >= 30)
assert len(g) == c30['n'] == 95 and int(mis.sum()) == c30['misclassified']
STRIP = 0.045
tr = lambda v: np.where(v < 0.1, STRIP, v)
axD.axvspan(30, 80, color=BAND, lw=0, zorder=0)
axD.axhspan(30, 80, color=BAND, lw=0, zorder=0)
for t_ in (30, 80):
    axD.axvline(t_, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=2)
    axD.axhline(t_, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=2)
axD.plot([0.1, 200], [0.1, 200], color=MUTED, lw=1.0, zorder=1)
both0 = (ri < 0.1) & (pi < 0.1)
sel = ~both0
axD.scatter(tr(ri[sel & ~mis]), tr(pi[sel & ~mis]), s=12, marker=MK['v3b'], color=M['v3b'], lw=0.6, edgecolor=WHITE, alpha=0.9, zorder=3)
axD.scatter(tr(ri[sel & mis]), tr(pi[sel & mis]), s=16, marker=MK['v3b'], facecolor=WHITE, edgecolor=M['v3b'], lw=1.1, zorder=4)
axD.scatter([STRIP], [STRIP], s=22, marker=MK['v3b'], color=M['v3b'], lw=0.6, edgecolor=WHITE, zorder=4)
print('d: scans below 0.1 mL on both axes', int(both0.sum()))
edh = g.EDH_ref_ml.values
je = int(np.argmax(edh))
assert ri[je] == 0 and edh[je] > 100 and mis[je]
axD.text(STRIP * 1.55, pi[je], f'EDH {edh[je]:.0f} mL', fontsize=7, color=INK, ha='left', va='center', zorder=6)
axD.set_xscale('log'); axD.set_yscale('log')
axD.set_xlim(0.03, 200); axD.set_ylim(0.03, 200)
ticks = [STRIP, 1, 10, 30, 80]
axD.set_xticks(ticks, ['<0.1', '1', '10', '30', '80']); axD.set_yticks(ticks, ['<0.1', '1', '10', '30', '80'])
axD.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); axD.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axD.set_aspect('equal')
axD.set_xlabel('Reference IPH volume (mL)'); axD.set_ylabel('v3b IPH volume (mL)')
ptitle(fig, axD, 'd', 'IPH volume at clinical thresholds', sub='BHSD test, 95 QC-gated scans; post hoc v3b', dx_mm=10.0)
icc_ci = X3['v3b_class_icc_gated']['IPH'][1:3]
tot = tg['total_v2_30']
TX = DX + DW + 2.5
lines = [
    ('ICC ' + ci_txt(tg['iph_v3b_icc'], *icc_ci), INK, 'bold'),
    ('30 mL (ICH score):', INK, 'normal'),
    (f"κ {c30['kappa']:.2f}; {c30['misclassified']} of {c30['n']}", INK, 'normal'),
    ('misclassified (open)', INK, 'normal'),
    ('30–80 mL (ENRICH):', INK, 'normal'),
    (f"κ {cb['kappa']:.2f}, 3 categories", INK, 'normal'),
    ('', INK, 'normal'),
    ('Total-volume surrogate', INK, 'bold'),
    (f"(v2): κ {tot['kappa']:.2f} at 30 mL", INK, 'normal'),
]
for j, (s_, c_, w_) in enumerate(lines):
    fig_text(fig, TX, ROW2 + 2.4 + j * 3.1, s_, color=c_, fontweight=w_)
print('d: ICC %.3f (%.3f-%.3f); kappa30 %.3f mis %d; bins kappa %.3f; total surrogate kappa %.3f mis %d' % (
    tg['iph_v3b_icc'], *icc_ci, c30['kappa'], c30['misclassified'], cb['kappa'], tot['kappa'], tot['misclassified']))

print('overflow', overflow_mm(fig))
save(fig, 'Figure_6_Volume', 'main_mr')
