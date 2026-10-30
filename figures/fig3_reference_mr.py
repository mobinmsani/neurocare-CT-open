"""Figure 3 - Reliability of the CQ500 reference, discordances between the open references, and the
precision attainable with the available positives. None of the panels depends on a model. BHX-CQ500 discordances are exact
counts after linkage by DICOM study UID (bhx_exact.py); BleedScene3D-CQ500 discordances are included.

a  Fleiss kappa (95% CI) for every CQ500 label (491 released scans) and the reader vote pattern.
c  Discordance between the open references for the same CQ500 scans (ledger.json, discordance rows); quality-control exclusions and
   conventions are listed in the supplementary ledger.
b  Best attainable Wilson lower 95% bound for sensitivity versus the number of positives (positives among the 489 analyzed scans).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nc_a import *
from ncdata import REG, LED, R3, OUT
import json
import pandas as pd

Z = 1.959963984540054
H = 146
fig = plt.figure(figsize=(W2, H * MM))
TOP = 9.0

# ============================================================================== a: reader agreement
kap = pd.read_csv(OUT / 'tables/cq500_reader_agreement_491.csv').sort_values('fleiss', ascending=False).reset_index(drop=True)
NAMES = {'ICH': 'Any hemorrhage', 'IPH': 'Intraparenchymal', 'IVH': 'Intraventricular', 'SDH': 'Subdural', 'EDH': 'Epidural',
         'SAH': 'Subarachnoid', 'ChronicBleed': 'Chronic bleed', 'BleedLocation-Left': 'Side: left', 'BleedLocation-Right': 'Side: right',
         'Fracture': 'Skull fracture (any)', 'CalvarialFracture': 'Calvarial fracture', 'OtherFracture': 'Other fracture',
         'MassEffect': 'Mass effect', 'MidlineShift': 'Midline shift >5 mm'}
SURG = {'SDH', 'EDH', 'MidlineShift'}
KMIN = 0.61
nL = len(kap)
n_rel = int(kap.n.iloc[0])
assert n_rel == REG['n_cq_released']
AH = 71.0
axA = axes_mm(fig, 28.5, TOP, 35.0, AH)
axV = axes_mm(fig, 68.0, TOP, 33.0, AH)
YT = -1.75                                   # key row above the first label
for i, r in kap.iterrows():
    ci_point(axA, r.fleiss, i, r.lo, r.hi, INK, marker='D', ms=4.0, lw=1.2)
    lab = NAMES[r.label] + (' †' if r.label in SURG else '')
    axA.text(-0.035, i, lab, transform=axA.get_yaxis_transform(), fontsize=7, ha='right', va='center', color=INK)
# hemorrhage side: kappa among the scans that the reader majority called hemorrhage-positive (basis of the side rating)
SIDEK = json.load(open(OUT / 'side_agreement.json'))['majority hemorrhage-positive scans']
for i, r in kap.iterrows():
    if r.label in ('BleedLocation-Left', 'BleedLocation-Right'):
        sk = SIDEK[r.label]
        ci_point(axA, sk['k'], i + 0.3, sk['lo'], sk['hi'], INK, marker='D', filled=False, ms=3.6, lw=1.0)
axA.axvline(KMIN, color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=1)
axA.text(KMIN - 0.025, YT, f'κ {KMIN:.2f} (rubric R2)', fontsize=7, color=INK2, ha='right', va='center')
axA.set_ylim(nL - 0.5, YT - 0.6); axA.set_yticks([]); axA.spines['left'].set_visible(False)
axA.set_xlim(0, 1); axA.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
axA.set_xlabel("Fleiss’ κ, three readers (95% CI)")
grid(axA, 'x')
# vote pattern: majority positive (3 of 3, 2 of 3) | one reader positive (majority negative)
cU, cS, cO = INK, '#8c8780', '#d9d5cd'
XM = 262
for i, r in kap.iterrows():
    axV.barh(i, r.n3, color=cU, height=0.62, zorder=3, lw=0)
    axV.barh(i, r.n2, left=r.n3, color=cS, height=0.62, zorder=3, lw=0)
    axV.barh(i, r.n1, left=r.n3 + r.n2 + 5, color=cO, height=0.62, zorder=3, lw=0)
    axV.text(1.03, i, f"{r.split_among_majority_pos_pct:.0f}%", transform=axV.get_yaxis_transform(), fontsize=7, va='center',
             ha='left', color=INK)
assert (kap.n1 + kap.n2 + kap.n3 + 5).max() < XM
axV.set_ylim(nL - 0.5, YT - 0.6); axV.set_yticks([]); axV.spines['left'].set_visible(False)
axV.set_xlim(0, XM); axV.set_xticks([0, 100, 200])
axV.set_xlabel('Scans with ≥1 positive read')
axV.text(1.03, YT, '2–1 share', transform=axV.get_yaxis_transform(), fontsize=7, ha='left', va='center', color=INK2)
grid(axV, 'x')
for cc, tt, x0 in [(cU, '3 of 3', 0), (cS, '2 of 3', 84), (cO, '1 of 3', 168)]:
    axV.add_patch(Rectangle((x0, YT - 0.3), 14, 0.6, color=cc, clip_on=False, zorder=5, lw=0))
    axV.text(x0 + 19, YT, tt, fontsize=7, va='center', ha='left', color=INK2)
ptitle(fig, axA, 'a', f'Reader agreement on the CQ500 labels ({n_rel} released scans)', dx_mm=27.0)

# ============================================================================== c: attainable precision
axC = axes_mm(fig, 126.5, TOP, 50.0, 62.0)
n = np.geomspace(5, 400, 400)


def wlb(p, k):
    return (p + Z * Z / (2 * k) - Z * np.sqrt(p * (1 - p) / k + Z * Z / (4 * k * k))) / (1 + Z * Z / k)


ex = R3['bound_examples']
for obs, ls, lab in [(1.0, '-', f"100% (35 of 35: {ex['35/35']:.3f})"), (0.95, (0, (5, 2)), f"95% (133 of 140: {ex['133/140']:.3f})"),
                     (0.90, (0, (1.2, 1.6)), '90%')]:
    axC.plot(n, wlb(obs, n), color=INK, lw=1.5, ls=ls, label=lab)
axC.axhline(0.90, color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=1)
axC.text(5.6, 0.905, 'lower bound 0.90', fontsize=7, color=INK2, ha='left', va='bottom')
P = REG['cq_prev']
PTS = [('EDH', P['EDH']), ('chronic', P['ChronicBleed']), ('IVH', P['IVH']), ('SDH', P['SDH']), ('SAH', P['SAH']),
       ('MLS', P['MidlineShift']), ('IPH', P['IPH']), ('any', P['ICH'])]
for lab, k_ in PTS:
    lb = k_ / (k_ + Z * Z)
    axC.plot(k_, lb, 'o', ms=4.6, mfc=INK if lb >= 0.90 else WHITE, mec=INK, mew=1.2, zorder=5)
for lab, k_, off, ha, va in [('EDH', P['EDH'], (-5, 1), 'right', 'center'), ('chronic', P['ChronicBleed'], (-5, 1), 'right', 'center'),
                             ('IVH', P['IVH'], (-5, 2), 'right', 'center'), ('IPH', P['IPH'], (0, -6), 'center', 'top'),
                             ('any', P['ICH'], (0, 6), 'center', 'bottom')]:
    axC.annotate(f'{lab} {k_}', (k_, k_ / (k_ + Z * Z)), xytext=off, textcoords='offset points', fontsize=7, ha=ha, va=va,
                 color=INK, path_effects=halo(2.2), zorder=6)
xg = P['SAH']
axC.annotate(f"SDH {P['SDH']}, SAH {P['SAH']}, MLS {P['MidlineShift']}", (xg, xg / (xg + Z * Z)), xytext=(5.6, 0.985),
             textcoords='data', fontsize=7, ha='left', va='center', color=INK, path_effects=halo(2.2), zorder=6,
             arrowprops=dict(arrowstyle='-', lw=1.0, color=INK2, shrinkA=1, shrinkB=3))
# integer examples (exact Wilson bounds) marked on the curves, explained in a text block
ex = R3['bound_examples']
for key in ('35/35', '133/140'):
    k_, n_ = [int(v) for v in key.split('/')]
    axC.plot([n_, n_], [0.64, ex[key]], color=INK2, lw=1.0, ls=(0, (1, 1.5)), zorder=2)
    axC.text(n_, 0.633, f'{n_}', fontsize=7, color=INK, ha='center', va='top', path_effects=halo(2.2), zorder=6)
axC.set_xscale('log'); axC.set_xlim(5, 420); axC.set_ylim(0.45, 1.0)
plain_log(axC, 'x', [5, 10, 20, 50, 100, 200, 400])
axC.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axC.set_xlabel('Positive scans (log scale)'); axC.set_ylabel('Best attainable lower 95% bound')
leg = axC.legend(loc='lower right', bbox_to_anchor=(1.0, 0.02), fontsize=7, title='Observed sensitivity', title_fontsize=7, handlelength=2.6,
                 labelspacing=0.3, frameon=True, facecolor=WHITE, edgecolor='none', framealpha=1.0, borderpad=0.3)
leg._legend_box.align = 'left'
grid(axC, 'y')
ptitle(fig, axC, 'b', 'Precision attainable with the available positives', dx_mm=10.5)

# ============================================================================== c: discordance between open references (model-free)
BX0, BW = 70.0, 75.0
XDES = 1.5
BT = 96.0
rows = list(LED['rows'])
assert [r['kind'] for r in rows] == ['fault'] * 6 + ['discordance'] * 10 + ['convention'] * 3
assert not any('lower bound' in r['item'] for r in rows)
DESC = {  # short description (display only; counts come from the ledger)
    6: 'Seg-CQ500 mask on a majority-negative scan',
    7: 'BHX boxes on a majority-negative scan',
    8: 'subarachnoid boxes without a majority SAH label',
    9: 'subdural boxes without a majority SDH label',
    10: 'intraventricular boxes without a majority IVH label',
    11: 'chronic boxes without a majority chronic-bleed label',
    12: 'curated as positive, majority negative',
    13: 'IVH flag without a majority IVH label',
    14: 'SAH flag without a majority SAH label',
    15: 'SDH flag without a majority SDH label',
}
groups = [('Seg-CQ500 masks against the CQ500 reader majority', [6]), ('BHX boxes against the CQ500 reader majority', [7, 8, 9, 10, 11]),
          ('BleedScene3D annotations against the CQ500 reader majority', [12, 13, 14, 15])]
ROWMM = 2.98
nrow = sum(1 + len(ix) for _, ix in groups)
BHt = nrow * ROWMM
axB = axes_mm(fig, BX0, BT, BW, BHt)
yy = 0.0
ypos, gpos = {}, []
for gname, idx in groups:
    gpos.append((gname, yy)); yy += 1.0
    for i in idx:
        ypos[i] = yy; yy += 1.0
ny = yy
T = axB.get_yaxis_transform()
for i, y in ypos.items():
    r = rows[i]
    assert r['kind'] == 'discordance' and r['n'] is not None and r['denominator']
    k_, n_ = int(r['n']), int(r['denominator'])
    p = 100 * k_ / n_
    lo, hi = wilson(k_, n_)
    axB.plot([100 * lo, 100 * hi], [y, y], color=INK, lw=1.2, zorder=3, solid_capstyle='round')
    axB.plot([p], [y], marker='D', ms=4.2, mfc=WHITE, mec=INK, mew=1.2, ls='', zorder=4)
    axB.text(1.03, y, f"{k_} of {n_}", transform=T, fontsize=7, ha='left', va='center', color=INK)
    axB.text((XDES - BX0) / BW, y, DESC[i], transform=T, fontsize=7, ha='left', va='center', color=INK)
for gname, yg in gpos:
    axB.text((XDES - BX0) / BW, yg, gname, transform=T, fontsize=7, ha='left', va='center', color=INK2, fontweight='bold')
axB.set_ylim(ny - 0.5, -0.5); axB.set_yticks([]); axB.spines['left'].set_visible(False)
axB.set_xlim(0, 80); axB.set_xticks([0, 20, 40, 60, 80])
axB.set_xlabel('Scans affected (%; Wilson 95% CI)')
grid(axB, 'x')
ptitle(fig, axB, 'c', 'Discordance between open references for the same CQ500 scans (model-free)', dx_mm=BX0 - 1.5)

print('overflow', overflow_mm(fig))
save(fig, 'Figure_3_Reference', 'main_mr')
