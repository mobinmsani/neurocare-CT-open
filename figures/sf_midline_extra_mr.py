"""Supplementary figure - Discrimination of the CSF-offset distance and internal consistency of the expected midline.

a  ROC for reader-majority midline shift >5 mm in the 452 CQ500 scans in which the distance was computable: CSF-offset distance vs v2
   predicted volume, with each reader against the concordant reads of the other two
b  Mean symmetry Dice of the expected midline per PHE-SICH-CT-IDS patient (120 patients)
c  In-plane midline angle per patient, with the patients at the +/-20 degree search bound
The examples, side accuracy and distance by reader votes are in the main text (Fig. 7). Values are read from the analysis outputs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nc_s2 import *
import nc_c
from sklearn.metrics import roc_curve, roc_auc_score

H = 118
fig = plt.figure(figsize=(W2, H * MM))


def ft(x_mm, y_mm, s, **kw):
    return nc_c.fig_text(fig, x_mm, y_mm, s, **kw)


m = load_cq500()
PR = N['proxy']
LS_PROXY = (0, (6.5, 2.0))

# ======================================================================= a: ROC, same 452 scans
ok = np.isfinite(m.csf_offset_maxabs_mm.values)
mp = m[ok]
assert ok.sum() == PR['n']
prox = mp.csf_offset_maxabs_mm.values
CXp, CYp, CWp = 16.0, 12.0, 36.0
axC = axes_mm(fig, CXp, CYp, CWp, CWp)
yref = mp.ref_MidlineShift.values
assert int(yref.sum()) == PR['pos']
axC.plot([0, 1], [0, 1], color='#d6d3cc', lw=1.0, ls=(0, (3, 2)), zorder=1)
f_, t_, _ = roc_curve(yref, prox)
axC.plot(f_, t_, color=M['rule'], lw=1.6, ls=LS_PROXY, zorder=4)
f_, t_, _ = roc_curve(yref, mp.pred_volume_ml.values)
axC.plot(f_, t_, color=M['v2'], lw=1.6, ls=LS['v2'], zorder=3)
loo = pd.DataFrame(N['loo'])
lr = loo[loo.label == 'MidlineShift']
for _, r in lr.iterrows():
    axC.plot(1 - r.reader_spec, r.reader_sens, MK['reader'], ms=4.4, mfc=INK, mec=WHITE, mew=0.6, zorder=5)
a_p = roc_auc_score(yref, prox); a_v = roc_auc_score(yref, mp.pred_volume_ml.values)
xm = X3['MidlineShift']
assert abs(a_p - PR['auc_all'][0]) < 1e-9 and abs(a_v - xm['auc_vol_same']) < 1e-9 and xm['n'] == PR['n']
axC.set_aspect('equal'); axC.set_xlim(-0.01, 1.01); axC.set_ylim(-0.01, 1.01)
axC.set_xticks([0, 0.5, 1.0], ['0', '0.5', '1']); axC.set_yticks([0, 0.5, 1.0], ['0', '0.5', '1'])
axC.set_xlabel('1 − specificity'); axC.set_ylabel('Sensitivity')
grid(axC, 'both')
ptitle(fig, axC, 'a', 'Discrimination of reader-labeled midline shift >5 mm', dx_mm=12.0)
dl = xm['delong_proxy_minus_vol']
TX = CXp + CWp + 4.0
yy = CYp + 2.2


def tl(s_, c_=INK, w_='normal', dy=3.1):
    global yy
    ft(TX, yy, s_, color=c_, fontweight=w_)
    yy += dy


tl(f"AUC (95% CI), {PR['n']} scans, {PR['pos']} with shift (majority)", INK, 'bold', dy=4.0)
sw = axes_mm(fig, TX, yy - 2.2, 6, 2.6); sw.axis('off'); sw.set_xlim(0, 1); sw.set_ylim(0, 1)
sw.plot([0, 1], [0.5, 0.5], color=M['rule'], lw=1.6, ls=LS_PROXY)
_lk = re.match(r'^([\d.]+) \(([\d.]+)–([\d.]+)\)$', REG['cq_MidlineShift_max']['auc'])
assert _lk and abs(float(_lk.group(1)) - PR['auc_all'][0]) < 5e-4
ft(TX + 7.5, yy, 'CSF-offset distance  ' + nc_c.ci_txt(float(_lk.group(1)), float(_lk.group(2)), float(_lk.group(3)), 3)); yy += 4.0
sw = axes_mm(fig, TX, yy - 2.2, 6, 2.6); sw.axis('off'); sw.set_xlim(0, 1); sw.set_ylim(0, 1)
sw.plot([0, 1], [0.5, 0.5], color=M['v2'], lw=1.6, ls=LS['v2'])
ft(TX + 7.5, yy, 'v2 predicted volume  ' + nc_c.ci_txt(xm['auc_vol_same'], xm['ci'][0], xm['ci'][1], 3)); yy += 4.0
tl('Difference (95% CI): ' + minus(f"{dl[0]:.3f} ({dl[1]:.3f} to {dl[2]:.3f}); DeLong {fmt_p(dl[3])}"), dy=4.0)
tl(f"Within ICH-positive scans ({PR['n_ich_pos'][0]}; {PR['n_ich_pos'][1]} with shift): {PR['auc_ich_pos'][0]:.3f} vs {PR['auc_vol_ich_pos'][0]:.3f}", dy=4.6)
sw = axes_mm(fig, TX, yy - 2.4, 6, 3.0); sw.axis('off'); sw.set_xlim(0, 1); sw.set_ylim(0, 1)
sw.plot([0.5], [0.5], MK['reader'], ms=4.4, mfc=INK, mec=WHITE, mew=0.6)
ft(TX + 7.5, yy, f"Each reader against the concordant reads of the other two ({int(lr.n.min())}–{int(lr.n.max())} scans)")
print('a: AUC distance %.3f, volume %.3f, diff %.3f (%.3f to %.3f) P=%.3f; within ICH %.3f vs %.3f' % (
    a_p, a_v, dl[0], dl[1], dl[2], dl[3], PR['auc_ich_pos'][0], PR['auc_vol_ich_pos'][0]))

# ======================================================================= b-c: expected midline in all 120 PHE-SICH-CT-IDS patients
ph = pd.read_csv(E / 'neurocare_v2_20261021/phe_sich/phe_laterality_midline.csv', dtype={'case': str}).set_index('case')
mid = V2R['midline']
assert len(ph) == 120 and (ph.midline_status == 'estimated').sum() == 120 == mid['estimated']
assert abs(ph.mean_symmetry_dice.median() - mid['symmetry_median']) < 1e-12 and f"{mid['symmetry_median']:.3f}" == REG['phe']['sym'] == '0.952'
nb = int((ph.midline_angle_deg.abs() >= 20).sum())
assert nb == mid['at_search_bound'] == REG['phe']['bound'] == 17
yb = 74
axB = axes_mm(fig, 16, yb, 70, 30)
sd = ph.mean_symmetry_dice.values
bins = np.arange(0.85, 0.9851, 0.005)
axB.hist(sd, bins=bins, color='#c9c4ba', lw=1.0, edgecolor=WHITE, zorder=3)
med = float(np.median(sd))
axB.axvline(med, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=4)
axB.set_ylim(0, 21)
axB.text(med - 0.0012, 20.6, f'median {med:.3f}', fontsize=F7, ha='right', va='top', color=INK)
v68 = float(ph.loc['0068', 'mean_symmetry_dice'])
axB.annotate('0068', (v68, 1.2), xytext=(v68 + 0.004, 8.0), fontsize=F7, ha='left', va='center', color=INK,
             arrowprops=dict(arrowstyle='-', lw=1.0, color=INK2, shrinkA=1, shrinkB=1))
axB.set_xlim(0.85, 0.985)
axB.set_xticks([0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98])
axB.yaxis.set_major_locator(plt.MultipleLocator(5))
axB.set_xlabel('Mean symmetry Dice per patient'); axB.set_ylabel('Patients')
grid(axB, 'y')
ptitle(fig, axB, 'b', 'Midline symmetry per patient',
       sub=f"Expected midline returned in {int((ph.midline_status == 'estimated').sum())} of {len(ph)}; no manual midline reference",
       dx_mm=13.0, dy_mm=3.2)

axC2 = axes_mm(fig, 108, yb, 70, 30)
ang = ph.midline_angle_deg.values
bins = np.arange(-21.25, 21.3, 2.5)
atb = np.abs(ang) >= 20
axC2.hist([ang[~atb], ang[atb]], bins=bins, stacked=True, color=['#c9c4ba', INK2], lw=1.0, edgecolor=WHITE, zorder=3)
for b_ in (-20, 20):
    axC2.axvline(b_, color=INK2, lw=1.0, ls=(0, (1.2, 1.6)), zorder=2)
q1, q3 = np.percentile(ang, [25, 75])
axC2.axvline(np.median(ang), color=INK, lw=1.0, ls=(0, (3, 2)), zorder=4)
nlo, nhi = int((ang <= -20).sum()), int((ang >= 20).sum())
axC2.set_ylim(0, 21)
axC2.text(np.median(ang) - 0.8, 20.6, nminus(f'median {np.median(ang):.2f}°, IQR {q1:.0f}° to {q3:.0f}°'), fontsize=F7, ha='right', va='top', color=INK)
axC2.text(-19.3, 16.0, f'{nlo} at −20°', fontsize=F7, ha='left', va='center', color=INK2)
axC2.text(19.3, 16.0, f'{nhi} at +20°', fontsize=F7, ha='right', va='center', color=INK2)
axC2.set_xlim(-22.5, 22.5)
axC2.set_xticks([-20, -10, 0, 10, 20], [nminus(str(t)) + '°' for t in [-20, -10, 0, 10, 20]])
axC2.yaxis.set_major_locator(plt.MultipleLocator(5))
axC2.set_xlabel('In-plane midline angle'); axC2.set_ylabel('Patients')
grid(axC2, 'y')
ptitle(fig, axC2, 'c', 'Midline angle per patient', sub=f'{nb} of {len(ang)} at the ±20° search bound (dark bins, dotted lines)', dx_mm=13.0, dy_mm=3.2)
print('b-c: 0068 symmetry', v68, 'bound low/high', nlo, nhi)
print('overflow', overflow_mm(fig))
save(fig, 'SF_midline_extra', 'supp_mr')
