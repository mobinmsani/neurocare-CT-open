"""Supplementary figure - Provenance: what was fixed before each evaluation.

Rows are calendar days (or periods) in order; heavy rules mark the two protocol locks (the CQ500 evaluation of the probes,
and the replication with the public model). Markers: filled, fixed before the evaluation; open, defined after results were
known (post hoc); cross, deviation. Event texts come from the provenance and model tables; numbers from registry.json.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nc_a import *
from ncdata import REG, LED, OUT, E

PMC = '#6a3d9a'
F = {r['resource']: (r['n'], r['denominator']) for r in LED['rows'] if r['kind'] == 'fault'}
n_seg_ex, n_seg = F['Seg-CQ500']
LOCK = json.load(open(E / 'public_model_cq500' / 'PROTOCOL_LOCK_public_model_CQ500.json'))
import datetime as _dt
_t = _dt.datetime.fromisoformat(LOCK['locked_at_utc'])
lock_t = f"{_t.day} {_t.strftime('%b %Y')}, {_t.strftime('%H:%M')} UTC"
fr = REG['fracture']
_bc = json.load(open(E / 'public_model_cq500' / 'public_model_batch_check.json'))
assert f"{max(_bc['max_abs_diff_series'], _bc['max_abs_diff_slice']):.7f}" == '0.0000004', 'update the batch-size deviation text'

# rows: (label, [events]); event = (colour, marker, filled, text)
ROWS = [
    ('20 Oct 2026', [(M['v1'], 'o', True, 'v1 locked, then trained')]),
    ('21 Oct 2026', [(M['rule'], 's', False, 'QC gate added after a v1 test failure\n(post hoc for v1)'),
                     (M['v2pm'], 'o', True, 'v2-PM locked before training'),
                     (M['v2'], 'o', False, 'v2 defined after BHSD test results;\nlocked before Seg-CQ500 inference'),
                     (M['edema'], 'o', True, 'Edema model locked; held-out set\n40 → 37 before testing')]),
    ('22 Oct 2026,\nbefore the lock', [(M['v3'], 'o', True, 'v3 locked; later failed internally\n(no EDH, SAH or SDH predicted)')]),
    ('LOCK1', 'CQ500 protocol locked: reader-majority reference, frozen v2, predicted volume ≥1.0 mL;\n'
              '9 scans processed before the lock, no reader label joined to model output'),
    ('22 Oct 2026,\nafter the lock', [(M['v3b'], 'o', False, 'v3b defined; frozen before\nits CQ500 inference'),
                                      (INK, 'X', True, f"Fracture pilot deviated from the lock;\nstopped after one fold (AUC {float(fr['auc']):.2f}, {fr['n']} scans)")]),
    ('23–28 Oct 2026', [(M['v3b'], 'o', False, 'v3b evaluated on CQ500'),
                        (INK, 'D', False, f'CQ500 without the {n_seg - n_seg_ex} Seg-CQ500 scans'),
                        (INK, 'D', False, 'Reference-standard analyses'),
                        (INK, 'D', False, 'Validation-capacity rubric, first version')]),
    ('LOCK2', f'Public-model replication locked ({lock_t}): released weights and fixed blends unchanged, threshold 0.5,\n'
              'endpoints E1–E8; one scan (CT-108) processed beforehand to time the run (its output was seen)'),
    ('28–29 Oct 2026,\nafter the lock', [(PMC, 'h', True, 'Public model run on the 489 CQ500 scans\nwith the locked series rule'),
                                          (INK, 'X', True, 'Batches of one slice instead of two\nfor 479 scans (memory limit); largest\nchange in a stored output 0.0000004'),
                                          (INK, 'D', False, 'Endpoint-specific rubric (revised)'),
                                          (INK, 'D', False, 'Model-free check of Seg-CQ500 masks;\nBHX linked to CQ500 by study identifier')]),
]

LH = 2.9          # line height (mm)
XD, XE1, XE2 = 1.5, 33.0, 106.0
X1 = 179.5
# layout pass: row heights
def ev_lines(s):
    return s.count('\n') + 1
heights = []
for lab, evs in ROWS:
    if lab.startswith('LOCK'):
        heights.append(10.6)
        continue
    col = [0, 0]
    for i, e in enumerate(evs):
        col[i % 2] += ev_lines(e[3]) * LH + 1.6
    heights.append(max(max(col) + 2.2, ev_lines(lab) * LH + 3.0))
H = 6.0 + sum(heights) + 9.0
fig = plt.figure(figsize=(W2, H * MM))
ax = canvas_axes(fig)
txt(ax, XD, 3.0, 'Date', bold=True, color=INK2)
txt(ax, XE1, 3.0, 'Fixed or defined', bold=True, color=INK2)
y = 6.0
band = 0
for (lab, evs), h in zip(ROWS, heights):
    if lab.startswith('LOCK'):
        yc = y + h / 2
        ax.plot([XD, X1], [yc, yc], color=INK, lw=2.4, zorder=3, solid_capstyle='butt')
        ax.plot([XD + 0.8], [yc], marker='D', ms=6.5, mfc=INK if lab == 'LOCK1' else PMC, mec=WHITE, mew=0.8, zorder=7)
        rbox(ax, XE1 - 2.0, y + 0.6, X1 - XE1 + 1.5, h - 1.2, fc=WHITE, ec=INK if lab == 'LOCK1' else PMC, lw=LW, r=0.9, z=5)
        txt(ax, XE1 - 0.6, yc, evs, va='center', linespacing=1.12, z=7)
        y += h
        continue
    if band % 2 == 0:
        ax.add_patch(Rectangle((XD - 0.5, y), X1 - XD + 0.5, h, fc=BAND, ec='none', zorder=0))
    band += 1
    txt(ax, XD + 0.8, y + 2.6, lab, bold=True, va='top', linespacing=1.12, color=INK)
    cy = [y + 2.6, y + 2.6]
    for i, (c, mk, filled, s) in enumerate(evs):
        k = i % 2
        x = (XE1 if k == 0 else XE2) + 1.0
        yy = cy[k] + 0.95
        if mk == 'X':
            ax.plot([x], [yy], marker='X', ms=5.2, mfc=c, mec=WHITE, mew=0.6, ls='', zorder=6)
        else:
            ax.plot([x], [yy], marker=mk, ms=4.8 if mk not in ('s',) else 4.4, mfc=c if filled else WHITE, mec=c, mew=1.3, ls='', zorder=6)
        txt(ax, x + 2.4, cy[k], s, va='top', linespacing=1.12)
        cy[k] += ev_lines(s) * LH + 1.6
    y += h
# key
yk = H - 3.2
ax.plot([XD + 1.6], [yk], marker='o', ms=4.4, mfc=INK2, mec=INK2, ls='', zorder=6)
txt(ax, XD + 3.6, yk, 'fixed before the evaluation (prespecified)', color=INK2)
ax.plot([XD + 58.6], [yk], marker='o', ms=4.4, mfc=WHITE, mec=INK2, mew=1.3, ls='', zorder=6)
txt(ax, XD + 60.6, yk, 'defined after results were known (post hoc)', color=INK2)
ax.plot([XD + 119.0], [yk], marker='X', ms=5.0, mfc=INK, mec=WHITE, mew=0.6, ls='', zorder=6)
txt(ax, XD + 121.0, yk, 'deviation; rows are days, not to scale', color=INK2)
print('H', round(H, 1))
print('overflow', overflow_mm(fig))
save(fig, 'SF_provenance', 'supp_mr')
