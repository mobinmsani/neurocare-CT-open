"""Figure 2 - Endpoint-specific validation capacity of the open references, applicability to models
by training source, and the evidence obtained in this study, for 17 endpoints of quantitative head CT.

Rows, criteria states, capacity and applicability come from out/capacity_rubric_v2.json (analysis/capacity_rubric_v2.py).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nc_a import *
from ncdata import OUT

C2 = json.load(open(OUT / 'capacity_rubric_v2.json'))
rows = C2['rows']
assert len(rows) == 17
SUMM = C2['summary']
H = 176
fig = plt.figure(figsize=(W2, H * MM))
ax = canvas_axes(fig)

SHOW = {
    'Any intracranial hemorrhage (scan level)': ('Any hemorrhage (scan level)', False),
    'Hemorrhage subtype presence (5 subtypes)': ('Subtype presence (five subtypes)', False),
    'Hemorrhage localization (box level)': ('Hemorrhage localization (boxes)', False),
    'Hemorrhage side': ('Hemorrhage side', False),
    'Voxel segmentation of hemorrhage': ('Voxel segmentation', False),
    'Total hemorrhage volume': ('Total hemorrhage volume', False),
    'Compartment volume (intraparenchymal, epidural)': ('Compartment volume (IPH, EDH)', True),
    'Volume change on serial or postoperative scans': ('Volume change on serial scans', True),
    'Perihematomal edema volume': ('Perihematomal edema volume', False),
    'Midline shift >5 mm (presence)': ('Midline shift >5 mm (presence)', True),
    'Mass effect (presence)': ('Mass effect (presence)', False),
    'Midline shift magnitude (mm)': ('Midline shift magnitude (mm)', True),
    'Subdural hematoma thickness (mm)': ('Subdural thickness (mm)', True),
    'Basal cisterns, ventricles, herniation': ('Cisterns, ventricles, herniation', False),
    'Skull fracture (scan level)': ('Skull fracture (scan level)', False),
    'Fracture localization': ('Fracture localization', False),
    'Outcome or added prognostic value': ('Outcome, added prognostic value', False),
}
assert set(SHOW) == {r['endpoint'] for r in rows}
TYPE = {'binary': 'B', 'binary_subtype': 'B', 'side': 'B', 'segmentation': 'S', 'continuous': 'M', 'continuous_threshold': 'M'}
TYPE_OVERRIDE = {'Outcome or added prognostic value': 'B', 'Fracture localization': 'B', 'Basal cisterns, ventricles, herniation': 'B'}

COLS = [('BHSD', 'rsna'), ('MBH-Seg train', 'rsna'), ('MR-BHSD', 'rsna'), ('Seg-CQ500', 'cq'), ('CQ500', 'cq'), ('BHX', 'cq'), ('BleedScene3D', 'cq'),
        ('PHE-SICH-CT-IDS', 'phe')]
GL = {
    'Any intracranial hemorrhage (scan level)': ['voxel+', 'voxel3+', 'voxel3', 'voxel+', 'label', '', 'voxel+', ''],
    'Hemorrhage subtype presence (5 subtypes)': ['voxel', 'voxel3', '', '', 'label', 'box', 'voxel', ''],
    'Hemorrhage localization (box level)': ['', '', '', '', '', 'box', '', ''],
    'Hemorrhage side': ['', '', '', '', 'label', '', '', 'side'],
    'Voxel segmentation of hemorrhage': ['voxel', 'voxel3', 'voxel3', 'voxel', '', '', 'voxel', ''],
    'Total hemorrhage volume': ['voxel', 'voxel3', 'voxel3', 'voxel', '', '', 'voxel', ''],
    'Compartment volume (intraparenchymal, epidural)': ['voxel', 'voxel3', '', '', '', '', 'voxel', ''],
    'Perihematomal edema volume': ['', '', '', '', '', '', '', 'edema'],
    'Midline shift >5 mm (presence)': ['', '', '', '', 'label', '', 'relation', ''],
    'Mass effect (presence)': ['', '', '', '', 'label', '', '', ''],
    'Midline shift magnitude (mm)': ['', '', '', '', 'request', '', '', ''],
    'Basal cisterns, ventricles, herniation': ['', '', '', '', '', '', 'relation', ''],
    'Skull fracture (scan level)': ['', '', '', '', 'label', '', '', ''],
}


def glyph2(ax, x, y, kind, s=0.82):
    if kind == 'relation':           # categorical relation from one annotation: tag with one dot
        ax.add_patch(FancyBboxPatch((x - 1.7 * s, y - 1.0 * s), 3.4 * s, 2.0 * s, boxstyle=f'round,pad=0,rounding_size={0.55 * s}',
                                    fc=WHITE, ec=INK, lw=LW, zorder=6))
        ax.add_patch(plt.Circle((x, y), 0.3 * s, fc=INK, ec='none', zorder=7))
    else:
        glyph(ax, x, y, kind, c=INK, s=s)


def states(r):
    if r['capacity'] == 'none':
        return ['not met', 'na', 'na', 'na']
    u = r['unmet']
    if r['type'] == 'binary_subtype':
        some = bool(r['subtypes_adequate'])
        r2 = ('partial' if some else 'not met') if 'R2' in u else 'met'
        r4 = ('partial' if some else 'not met') if 'R4' in u else 'met'
        return ['met', r2, 'met', r4]
    r1 = 'partial' if 'R1 construct partial' in u else 'met'
    if 'R2 reader agreement not reported' in u:
        r2 = 'unknown'
    elif 'R2' in u:
        r2 = 'not met'
    else:
        r2 = 'met'
    if r['type'] == 'continuous':
        r3 = 'na'
    elif 'R3 coverage of decision thresholds unknown' in u:
        r3 = 'unknown'
    elif 'R3' in u:
        r3 = 'not met'
    else:
        r3 = 'met'
    r4 = 'not met' if 'R4' in u else 'met'
    if r['type'] == 'side' and 'R3/R4' in u:
        r3 = r4 = 'not met'
    return [r1, r2, r3, r4]


# applicability: capacity when references from a training source are removed
def scen(label):
    s = [a for a in SUMM['applicability'] if a['scenario'].lower() == label.lower()][0]
    out = {r['endpoint']: r['capacity'] for r in rows}
    if s['rating_changes'] != '–':
        for ch in s['rating_changes'].split('; '):
            ep, rest = ch.rsplit(': ', 1)
            out[ep] = rest.split(' -> ')[1]
    return out


APP = [('RSNA 2019', scen('model trained on RSNA 2019 data')), ('CQ500', scen('model trained or tuned on CQ500'))]

# ----------------------------------------------------------------------------- geometry (mm, y down)
XE = 1.5
XT = 45.0                              # endpoint type letter
XR0, RW = 50.0, 5.6                    # resource columns
XC0, CWc = XR0 + len(COLS) * RW + 1.6, 5.0
XK, KW = XC0 + 4 * CWc + 1.4, 14.6     # capacity badge
XA0, AW = XK + KW + 2.4, 6.6           # applicability chips (two columns)
XV, VW = XA0 + 2 * AW + 2.8, 25.4      # evidence badge
XRt = XV + VW + 0.6
RH, GH = 5.5, 4.2
HT = 29.0

corner = [(f"Validation capacity, {len(rows)} endpoints:", True),
          (f"{SUMM['capacity']['adequate']} adequate, {SUMM['capacity']['limited']} limited, {SUMM['capacity']['none']} none", False),
          ("For models trained on CQ500:", True),
          ("{} adequate, {} limited, {} none".format(*[[a for a in SUMM['applicability'] if a['scenario'].lower() == 'model trained or tuned on cq500'][0][k]
                                                       for k in ('adequate', 'limited', 'none')]), False)]
for i, (t_, b_) in enumerate(corner):
    txt(ax, XE, 5.6 + i * 3.0 + (0.7 if i >= 2 else 0), t_, color=INK2, bold=b_)

fy = 2.4
fams = {}
for j, (c, f) in enumerate(COLS):
    fams.setdefault(f, []).append(j)
FAMLAB = {'rsna': 'RSNA', 'cq': 'CQ500 scans', 'phe': 'PHE'}
for f, js in fams.items():
    x0 = XR0 + js[0] * RW + 0.25; x1 = XR0 + (js[-1] + 1) * RW - 0.25
    rbox(ax, x0, fy - 1.8, x1 - x0, 3.6, fc=FAM[f], ec=FAMEDGE[f], lw=LW, r=0.8, z=2)
    txt(ax, (x0 + x1) / 2, fy, FAMLAB[f], ha='center', size=7.0, z=3)
hb = HT - 2.6
for j, (c, f) in enumerate(COLS):
    txt(ax, XR0 + j * RW + RW / 2, hb, c, rotation=90, ha='center', va='bottom')
CRN = ['R1 construct', 'R2 reliability', 'R3 spectrum', 'R4 precision']
for j, c in enumerate(CRN):
    txt(ax, XC0 + j * CWc + CWc / 2, hb, c, rotation=90, ha='center', va='bottom')
txt(ax, XC0 + 2 * CWc, fy, 'Criteria', ha='center')
txt(ax, XT + 1.6, hb, 'Type', rotation=90, ha='center', va='bottom')
for j, (lab, _) in enumerate(APP):
    txt(ax, XA0 + j * AW + AW / 2, hb, lab, rotation=90, ha='center', va='bottom')
txt(ax, XA0 + AW, fy - 0.2, 'Trained on', ha='center')
txt(ax, XE, hb - 0.2, 'Endpoint', bold=True, va='bottom')
txt(ax, XK + KW / 2, hb - 0.2, 'Validation\ncapacity', bold=True, ha='center', va='bottom', linespacing=1.05)
txt(ax, XV + VW / 2, hb - 0.2, 'Evidence in\nthis study', bold=True, ha='center', va='bottom', linespacing=1.05)
ax.plot([XE - 0.5, XRt], [HT - 1.2, HT - 1.2], color=INK, lw=LW, zorder=3, solid_capstyle='butt')

CH = dict(adequate='A', limited='L', none='N')
y = HT
dom_prev = None
ri = 0
for r in rows:
    if r['domain'] != dom_prev:
        dname = r['domain'][0] + r['domain'][1:].lower()
        txt(ax, XE, y + GH / 2 + 0.3, dname, bold=True, color=INK2)
        y += GH
        dom_prev = r['domain']
    name, dag = SHOW[r['endpoint']]
    yc = y + RH / 2
    if ri % 2 == 1:
        ax.add_patch(Rectangle((XE - 0.5, y), XRt - XE + 0.5, RH, fc='#f6f5f2', ec='none', zorder=0.5))
    ri += 1
    txt(ax, XE, yc, name + (' †' if dag else ''))
    tl = TYPE_OVERRIDE.get(r['endpoint'], TYPE[r['type']])
    txt(ax, XT + 1.6, yc, tl, ha='center', color=INK2, bold=True)
    gl = GL.get(r['endpoint'], [''] * len(COLS))
    for j, g in enumerate(gl):
        cx = XR0 + j * RW + RW / 2
        if g:
            pos_only = g.endswith('+')
            gk = g.rstrip('+')
            glyph2(ax, cx - (0.95 if pos_only else 0), yc, gk)
            if pos_only:
                txt(ax, cx + 2.0, yc - 0.6, '+', ha='center', bold=True)
        else:
            ax.add_patch(plt.Circle((cx, yc), 0.32, fc='#bdb9b1', ec='none', zorder=2))
    for j, stt in enumerate(states(r)):
        crit_symbol(ax, XC0 + j * CWc + CWc / 2, yc, stt, r=1.1)
    c = r['capacity']
    badge(ax, XK, y + 0.7, KW, RH - 1.4, CAPLAB[c], CAP[c], CAPEDGE[c], CAPTXT[c], r=0.9)
    for j, (_, mp) in enumerate(APP):
        cc = mp[r['endpoint']]
        badge(ax, XA0 + j * AW + 0.45, y + 0.8, AW - 0.9, RH - 1.6, CH[cc], CAP[cc], CAPEDGE[cc], CAPTXT[cc], r=0.7)
    e = {'external': 'ext', 'internal or transfer': 'int', 'exploratory': 'exp', 'not evaluated': 'ne'}[r['evidence_here']]
    lab = {'ext': 'External', 'int': 'Internal/transfer', 'exp': 'Exploratory', 'ne': 'Not evaluated'}[e]
    badge(ax, XV, y + 0.7, VW, RH - 1.4, lab, EV[e], EV[e] if e != 'ne' else '#9a978f', EVTXT[e], r=0.9, bold=False)
    y += RH
ax.plot([XE - 0.5, XRt], [y + 0.5, y + 0.5], color=INK, lw=LW, zorder=3, solid_capstyle='butt')
y_end = y + 0.5

# ----------------------------------------------------------------------------- keys
ky = y_end + 3.8
items = [('voxel', 'voxel masks'), ('voxel3', 'masks, several raters'), ('label', 'labels, three readers'), ('box', 'subtype boxes'),
         ('relation', 'categorical relation'), ('edema', 'edema masks'), ('side', 'documented side'), ('request', 'millimeters on request'),
         ('+', 'positive scans only')]
lx = XE
txt(ax, lx, ky, 'Reference:', color=INK2); lx += width_mm(fig, 'Reference:') + 2.2
wrap_at = {'edema'}
for kind, desc in items:
    if kind in wrap_at:
        ky += 3.9; lx = XE + width_mm(fig, 'Reference:') + 2.2
    if kind == '+':
        txt(ax, lx + 0.9, ky - 0.5, '+', ha='center', bold=True)
        lx += 2.6
    else:
        glyph2(ax, lx + 1.8, ky, kind)
        lx += 4.4
    txt(ax, lx, ky, desc, color=INK2)
    lx += width_mm(fig, desc) + 3.0
ky2 = ky + 3.9
lx = XE
txt(ax, lx, ky2, 'Criteria:', color=INK2); lx += width_mm(fig, 'Criteria:') + 2.6
for stt, desc in [('met', 'met'), ('partial', 'partly met'), ('not met', 'not met'), ('na', 'not applicable'), ('unknown', 'not reported')]:
    crit_symbol(ax, lx + 1.1, ky2, stt, r=1.1)
    lx += 3.3
    txt(ax, lx, ky2, desc, color=INK2)
    lx += width_mm(fig, desc) + 3.0
ky3 = ky2 + 3.9
txt(ax, XE, ky3, 'Type: B, binary (presence or side); S, segmentation and total volume; M, measurement (compartment volume, millimeters, edema volume).',
    color=INK2)
txt(ax, XE, ky3 + 3.0, '† tied to a surgical or trial threshold (Table 2). Adequate requires every criterion met by one reference.', color=INK2)
r = SUMM['rules']
txt(ax, XE, ky3 + 6.0, f"R2: κ ≥{r['kappa_min']:.2f} from ≥2 readers (for side, among hemorrhage-positive scans), or masks from ≥2 raters.", color=INK2)
txt(ax, XE, ky3 + 9.0, f"R3: ≥{r['binary_min']} negatives (B, S), or ≥{r['threshold_min']} cases on each side of each decision threshold (M). "
    f"R4: ≥{r['binary_min']} positives (B) or ≥{r['continuous_min']} measured cases (S, M).", color=INK2)
txt(ax, XE, ky3 + 12.0, 'Trained on: capacity for a model trained on that source, using only references with independent images (A, adequate; L, limited; N, none).',
    color=INK2)
print('rows end %.1f, keys end %.1f of %d' % (y_end, ky3 + 13.5, H))
print('overflow', overflow_mm(fig))
save(fig, 'Figure_2_Capacity', 'main_mr')
