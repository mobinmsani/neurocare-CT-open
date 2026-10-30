"""Helpers for the supplementary figures on CQ500 detection and reader agreement.

Only presentation helpers live here; every number is read through ncdata / ncstyle from the analysis outputs.
"""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncstyle import *
from ncdata import *

T7 = 7.0                         # the one type size used in every figure
DG = DAGGER                      # surgical- or trial-threshold label marker (ink, never colour alone)

# one row order for every CQ500 label panel: any hemorrhage, the five subtypes in the package order (ncstyle.SUB),
# chronic bleed, side, mass effect and midline, fracture
ORDER14 = ['ICH', 'EDH', 'IPH', 'IVH', 'SAH', 'SDH', 'ChronicBleed', 'BleedLocation-Left', 'BleedLocation-Right',
           'MassEffect', 'MidlineShift', 'Fracture', 'CalvarialFracture', 'OtherFracture']
NAME = {'ICH': 'Any ICH', 'EDH': 'Epidural (EDH)', 'IPH': 'Intraparenchymal (IPH)', 'IVH': 'Intraventricular (IVH)',
        'SAH': 'Subarachnoid (SAH)', 'SDH': 'Subdural (SDH)', 'ChronicBleed': 'Chronic bleed', 'BleedLocation-Left': 'Side: left',
        'BleedLocation-Right': 'Side: right', 'MassEffect': 'Mass effect', 'MidlineShift': 'Midline shift >5 mm',
        'Fracture': 'Skull fracture', 'CalvarialFracture': 'Calvarial fracture', 'OtherFracture': 'Other fracture'}
SURG = {'SDH', 'EDH', 'MidlineShift'}   # findings with surgical criteria (Table 2); the calvarial-fracture label does not record depression


def lname(lab, short=False):
    """Label name with a dagger for labels tied to a surgical criterion."""
    s = lab if (short and lab in SUB) else NAME.get(lab, lab)
    return s + (DG if lab in SURG else '')


def nminus(s):
    """Typographic minus for numbers only (keeps hyphens in words and case identifiers such as CT-108)."""
    return re.sub(r'(?<![A-Za-z0-9])-(?=\.?\d)', '−', s)


# model identity: colour + line style + marker, always together
STY = {k: dict(color=M[k], ls=LS[k], marker=MK[k]) for k in ('v2', 'v2pm', 'v3b', 'v3', 'v1', 'blast')}
V3_EDGE = '#9c4a2a'              # dark outline that keeps the light v3 colour visible on white

# readers: ink diamonds (R1 filled ink, R2 filled warm grey, R3 open ink)
RD = {'R1': dict(mfc=INK, mec=INK, mew=1.0), 'R2': dict(mfc='#8c8780', mec='#8c8780', mew=1.0), 'R3': dict(mfc=WHITE, mec=INK, mew=1.0)}


def reader_handles(ms=4.0):
    return [Line2D([], [], marker='D', ms=ms, ls='', label=q, **RD[q]) for q in ('R1', 'R2', 'R3')]


def model_handle(key, label, filled=True, ms=4.4, line=False, lw=1.5):
    st = STY[key]
    return Line2D([], [], color=st['color'], ls=st['ls'] if line else '', lw=lw, marker=st['marker'] if not line or filled is not None else '',
                  ms=ms, mfc=st['color'] if filled else WHITE, mec=st['color'], mew=1.0, label=label)


def ci_mark(ax, x, y, lo, hi, key, filled=True, ms=4.4, lw=1.2, orient='h', z=3, color=None, marker=None, mec=None):
    """Point estimate with a 95% CI in the model's colour and marker (open = white fill, 1 pt outline)."""
    st = STY.get(key, dict(color=color or INK, marker=marker or 'o'))
    c = color or st['color']
    mk = marker or st['marker']
    if orient == 'h':
        ax.plot([lo, hi], [y, y], color=c, lw=lw, zorder=z, solid_capstyle='butt')
    else:
        ax.plot([x, x], [lo, hi], color=c, lw=lw, zorder=z, solid_capstyle='butt')
    ax.plot([x], [y], marker=mk, ms=ms, mfc=c if filled else WHITE, mec=mec or c, mew=1.0, ls='', zorder=z + 1)


def band_rows(axes, n, start=0, color=None):
    for i in range(start, n, 2):
        for ax in axes:
            ax.axhspan(i - 0.5, i + 0.5, color=color or BAND, lw=0, zorder=0)


def fig_text(fig, x_mm, y_mm, s, H, **kw):
    """Text in figure mm coordinates (origin top-left)."""
    kw.setdefault('fontsize', T7)
    kw.setdefault('color', INK2)
    kw.setdefault('va', 'baseline')
    return fig.text(x_mm * MM / W2, 1 - y_mm * MM / (H * MM), s, **kw)


def fmt2(x):
    return nminus(f'{x:.2f}')


def fmt3(x):
    return nminus(f'{x:.3f}')
