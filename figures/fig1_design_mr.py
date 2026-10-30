"""Figure 1 - Study design.

Six open resources (grouped by image source) -> reference standards -> models (one row per version, the in-house probes,
the released BLAST-CT and the independently developed public model) -> evaluation settings; the model-free rating of
validation capacity (Fig. 2) as a band beneath. The provenance timeline moved to the supplement (sf_provenance_mr.py).
Counts from ledger.json / registry.json / new_analyses_cq500.json / bhx_exact.json; model specifications from the model table
and the protocol lock of the public-model replication.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nc_a import *
from ncdata import REG, N, LED, S, OUT, E
import json

H = 119
fig = plt.figure(figsize=(W2, H * MM))
ax = canvas_axes(fig)

# ----------------------------------------------------------------------------- numbers (from outputs)
F = {r['resource'] + ('_hu' if 'HU' in r['item'] else ''): (r['n'], r['denominator']) for r in LED['rows'] if r['kind'] == 'fault'}
n_bhsd_ex, n_bhsd = F['BHSD']; n_mr_ex, n_mr = F['MR-BHSD']; n_seg_ex, n_seg = F['Seg-CQ500']
n_phe_ex, n_phe = F['PHE-SICH-CT-IDS']; n_cq_ex, n_cq = F['CQ500']
assert (n_bhsd, n_mr, n_seg, n_phe, n_cq) == (192, 48, 51, 120, 491)
assert n_cq - n_cq_ex == REG['n_cq'] == 489
nB1, nB2 = N['batch_n']['B1'], N['batch_n']['B2']
n_bhx = REG['bhx_scans']
n_train, n_test = REG['qc']['train_n'], REG['qc']['test_n']
phe_uni = S['phe_unilateral_geo']
LS_ = LED['summary']
PMC = '#6a3d9a'
CAPBG = '#eef3f9'
BX_ = json.load(open(OUT / 'bhx_exact.json'))
n_bhx_box = BX_['ICH']['bhx_scans']
n_end = json.load(open(OUT / 'capacity_rubric_v2.json'))['summary']['n']
print('pairs analyzed %d of %d (unusable %d)' % (LS_['released_pairs'] - LS_['unusable_pairs'], LS_['released_pairs'], LS_['unusable_pairs']))

# ----------------------------------------------------------------------------- panel a

XL, XC, CW = 1.5, 9.0, 66.0          # lane left, card left, card width
XP, PW = 83.5, 59.0                  # probe column
XS, SW = 146.0, 33.5                 # settings column
HY = 3.4
txt(ax, XC, HY, 'Open resource: released → analyzed', bold=True, color=INK2)
txt(ax, XP, HY, 'Model', bold=True, color=INK2)
txt(ax, XP + PW, HY, 'Evaluated in', bold=True, color=INK2, ha='right')
txt(ax, XS, HY, 'Evaluation setting', bold=True, color=INK2)

CH, GAP = 11.6, 1.6
cards = [  # name, count, glyphs, line1, line2, setting, family
    ('BHSD', f'{n_bhsd} → {n_bhsd - n_bhsd_ex}', ['voxel'], 'Five-subtype voxel masks; radiologist-reviewed',
     f'{n_train} training and {n_test} test scans analyzed', 'I', 'rsna'),
    ('MR-BHSD', f'{n_mr} → {n_mr - n_mr_ex}', ['voxel3'], 'Voxel masks from three raters',
     'MBH-Seg 2025 validation release', 'T', 'rsna'),
    ('CQ500', f'{n_cq} → {n_cq - n_cq_ex}', ['label'], '14 scan-level labels from three radiologists',
     f'B1 consecutive {nB1}, B2 enriched {nB2} (analyzed)', 'E', 'cq'),
    ('BHX', f'{n_bhx_box} scans with boxes', ['box'], 'Subtype boxes from three neuroradiologists',
     'thick slices; linked to CQ500 by study identifier', 'E', 'cq'),
    ('Seg-CQ500', f'{n_seg} → {n_seg - n_seg_ex}', ['voxel'], 'Binary voxel masks; resident and neuroradiologist',
     'one mask per scan', 'T', 'cq'),
    ('PHE-SICH-CT-IDS', f'{n_phe} → {n_phe - n_phe_ex}', ['edema', 'side'], 'Edema masks; documented hemorrhage side',
     f'basal-ganglia hemorrhage; {phe_uni} unilateral', 'P', 'phe'),
]
y0 = 6.2
lanes = {'rsna': (y0, 2), 'cq': (y0 + 2 * (CH + GAP) + 2.0, 3), 'phe': (y0 + 5 * (CH + GAP) + 6.5, 1)}
lane_lab = {'rsna': ('RSNA 2019', 'training source'), 'cq': ('CQ500 images', 'New Delhi'), 'phe': ('PHE', 'one site')}
card_y = {}
for fam, (ly, n) in lanes.items():
    lh = n * CH + (n - 1) * GAP + 2.0
    rbox(ax, XL, ly - 1.0, XC + CW - XL + 1.2, lh, fc=FAM[fam], ec='none', r=1.4, z=1)
    l1, l2 = lane_lab[fam]
    txt(ax, XL + 2.15, ly - 1.0 + lh / 2, l1, rotation=90, ha='center', bold=True, color=INK2)
    txt(ax, XL + 4.75, ly - 1.0 + lh / 2, l2, rotation=90, ha='center', color=INK2)
fam_i = {'rsna': 0, 'cq': 0, 'phe': 0}
for name, cnt, glyphs, l1, l2, st, fam in cards:
    ly, n = lanes[fam]
    y = ly + fam_i[fam] * (CH + GAP); fam_i[fam] += 1
    card_y[name] = y
    rbox(ax, XC, y, CW, CH, fc=WHITE, ec=FAMEDGE[fam], lw=LW, r=1.0, z=3)
    txt(ax, XC + 2.0, y + 2.7, name, bold=True)
    txt(ax, XC + CW - 6.8, y + 2.7, cnt, bold=True, ha='right')
    setting_dot(ax, XC + CW - 3.2, y + 2.7, st, True)
    gx = XC + 3.6
    for g in glyphs:
        glyph(ax, gx, y + 6.15, g, c=INK, s=0.9)
        gx += 4.3
    txt(ax, gx - 1.7 + 0.6, y + 6.15, l1, color=INK)
    txt(ax, XC + 2.0, y + 9.35, l2, color=INK2)

# probes, one row per version
PR = [  # key, name, status, description, settings
    ('v1', 'v1', 'prespecified', 'Compact 2.5-dimensional U-Net', 'ITP'),
    ('v2pm', 'v2-PM', 'prespecified', 'Multi-window U-Net with intracranial post-mask', 'IT'),
    ('v2', 'v2', 'post hoc ablation', 'v2-PM weights without the post-mask', 'ITEP'),
    ('v3', 'v3', 'prespecified; failed internally', 'Six-class output (five subtypes)', 'I'),
    ('v3b', 'v3b', 'post hoc', 'v3 with rare classes oversampled', 'IE'),
    ('blast', 'BLAST-CT', 'pretrained comparator', 'Released model; no subarachnoid class', 'T'),
    ('pm', 'Public model', 'independent', 'RSNA 2019-trained; analysis plan locked', 'E'),
    ('rule', 'Rule modules', 'fixed before CQ500', 'QC gate; ideal midline; CSF-offset distance', 'ITEP'),
    ('edema', 'Edema model', 'prespecified', f'v2 architecture; 75 training, 37 held-out patients', 'P'),
]
PH, PG = 7.7, 1.15
py0 = y0 - 1.0
PY = {k: py0 + i * (PH + PG) for i, k in enumerate(['v1', 'v2pm', 'v2', 'v3', 'v3b'])}
PY['blast'] = PY['v3b'] + PH + 2.6
PY['pm'] = PY['blast'] + PH + PG
PY['rule'] = PY['pm'] + PH + PG
PY['edema'] = card_y['PHE-SICH-CT-IDS'] + 2.7 - 2.45
prow = {}
for i, (k, name, status, desc, sets) in enumerate(PR):
    y = PY[k]
    prow[k] = y
    rbox(ax, XP, y, PW, PH, fc=WHITE, ec='#cfcbc3', lw=LW, r=0.9, z=3)
    ax.add_patch(Rectangle((XP + 0.5, y + 0.5), 1.6, PH - 1.0, fc=(PMC if k == 'pm' else M[k]), ec='none', zorder=4))
    t = txt(ax, XP + 3.4, y + 2.45, name, bold=True)
    wn = width_mm(fig, name, bold=True)
    txt(ax, XP + 3.4 + wn + 1.3, y + 2.45, status, color=INK2)
    txt(ax, XP + 3.4, y + 5.5, desc, color=INK)
    for j, L in enumerate('ITEP'):
        setting_dot(ax, XP + PW - 13.3 + j * 3.75, y + 2.45, L, L in sets)

# training links: BHSD -> five hemorrhage probes (v1 ... v3b); PHE -> edema model
bx = XC + CW + 1.2
yb = card_y['BHSD'] + 2.7
xs_ = bx + 4.4
ax.plot([bx, xs_], [yb, yb], color=INK2, lw=LW, zorder=5, solid_capstyle='butt')
ax.plot([xs_, xs_], [yb, prow['v3b'] + 2.45], color=INK2, lw=LW, zorder=5, solid_capstyle='butt')
for k in ('v1', 'v2pm', 'v2', 'v3', 'v3b'):
    ax.annotate('', xy=(XP - 0.15, prow[k] + 2.45), xytext=(xs_, prow[k] + 2.45),
                arrowprops=dict(arrowstyle='-|>', lw=LW, color=INK2, mutation_scale=5.5, shrinkA=0, shrinkB=0), zorder=5)
txt(ax, bx + 2.0, (prow['v2pm'] + prow['v3b']) / 2 + 3.6, f'trained on {n_train} BHSD scans', rotation=90, ha='center', color=INK2)
ype = card_y['PHE-SICH-CT-IDS'] + 2.7
ax.plot([bx, xs_], [ype, ype], color=INK2, lw=LW, zorder=5, solid_capstyle='butt')
ax.plot([xs_, xs_], [ype, prow['edema'] + 2.45], color=INK2, lw=LW, zorder=5, solid_capstyle='butt')
ax.annotate('', xy=(XP - 0.15, prow['edema'] + 2.45), xytext=(xs_, prow['edema'] + 2.45),
            arrowprops=dict(arrowstyle='-|>', lw=LW, color=INK2, mutation_scale=5.5, shrinkA=0, shrinkB=0), zorder=5)

# evaluation settings
SETS = [('I', 'Internal test', [f'BHSD test, {n_test} scans', 'voxel masks', 'unit: scan']),
        ('T', 'Transfer (not blind)', [f'MR-BHSD, {n_mr - n_mr_ex} scans', f'Seg-CQ500, {n_seg - n_seg_ex} scans', 'partly used in development']),
        ('E', 'External, frozen', [f'CQ500, {n_cq - n_cq_ex} scans', f'BHX boxes, {n_bhx} scans', 'locked before labels joined']),
        ('P', 'Patient cohort', [f'side, {phe_uni} patients', 'edema, 37 held out', 'one site; unit: patient'])]
SH = 16.6
ys0, ys1 = y0 - 1.0, card_y['PHE-SICH-CT-IDS'] + CH - SH
for i, (L, name, lines) in enumerate(SETS):
    y = ys0 + i * (ys1 - ys0) / 3
    rbox(ax, XS, y, SW, SH, fc=WHITE, ec=INK, lw=LW, r=1.0, z=3)
    setting_dot(ax, XS + 2.9, y + 2.9, L, True)
    txt(ax, XS + 5.4, y + 2.9, name, bold=True)
    for j, ln in enumerate(lines):
        txt(ax, XS + 1.9, y + 6.9 + j * 3.15, ln, color=INK if j < 2 else INK2)
# model-free rating band (Fig. 2)
yr = card_y['PHE-SICH-CT-IDS'] + CH + 2.4
RH = 16.9
rbox(ax, XL, yr, 179.5 - XL, RH, fc=CAPBG, ec=CAP['adequate'], lw=LW, r=1.2, z=2)
txt(ax, XL + 2.0, yr + 3.0, 'Model-free rating of validation capacity (Fig. 2)', bold=True, color=CAP['adequate'])
txt(ax, XL + 2.0, yr + 7.0, f"For each of {n_end} endpoints, the best open reference is rated on construct (R1), reliability (R2), spectrum (R3) and precision (R4),", color=INK)
txt(ax, XL + 2.0, yr + 10.3, "with thresholds set by endpoint type; applicability is then judged by the training source of the model to be validated.", color=INK)
txt(ax, XL + 2.0, yr + 13.6, "BleedScene3D, released after the probe analyses, and the MBH-Seg 2025 training release, which contains the probes' training scans, enter only this rating.", color=INK)
ynote = yr + RH + 3.4
txt(ax, XL, ynote, f"{LS_['released_pairs'] - LS_['unusable_pairs']} of {LS_['released_pairs']} released image–reference pairs analyzed. "
    "Filled letters: the setting in which a resource was the reference or a model was evaluated.", color=INK2)
ya_end = ynote
print('ynote', round(ynote, 1), 'H', H)

print('overflow', overflow_mm(fig))
save(fig, 'Figure_1_Design', 'main_mr')
