"""Helpers for Figures 4 and 5 and SF_bhx. Imports the shared style unchanged."""
import numpy as np
from matplotlib.colors import to_rgb
from ncstyle import *


def rel_lum(c):
    """WCAG relative luminance of a colour."""
    r = np.array(to_rgb(c))
    r = np.where(r <= 0.04045, r / 12.92, ((r + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * r[0] + 0.7152 * r[1] + 0.0722 * r[2])


def text_on(c):
    """Ink or white text, whichever has the higher WCAG contrast against background colour c."""
    L = rel_lum(c)
    c_white = 1.05 / (L + 0.05)
    c_ink = (L + 0.05) / (rel_lum(INK) + 0.05)
    return WHITE if c_white > c_ink else INK


def note(fig, x_mm, y_mm, s, color=None, ha='left', va='top', **kw):
    """7 pt explanatory text placed in figure millimetres (origin top-left)."""
    W, H = fig.get_size_inches()
    return fig.text(x_mm * MM / W, 1 - y_mm * MM / H, s, fontsize=7.0, color=color or INK2, ha=ha, va=va,
                    linespacing=1.2, **kw)


def wilson_ci(k, n):
    lo, hi = wilson(k, n)
    return float(lo), float(hi)


def pct(x, d=0):
    return f'{100 * x:.{d}f}%'


def fmt_ci(v, lo, hi, d=2, sep='–'):
    """0.81 (0.76–0.87); uses a true minus sign and 'to' when an endpoint is negative."""
    def f(x):
        return minus(f'{x:.{d}f}')
    if lo < 0 or hi < 0:
        return f'{f(v)} ({f(lo)} to {f(hi)})'
    return f'{f(v)} ({f(lo)}{sep}{f(hi)})'


def draw_overlays_halo(ax, rec, kinds=('box', 'pred'), lw=1.2, halo_w=1.4):
    """Like ncstyle.draw_overlays, with a thin black halo so contours stay visible in greyscale print."""
    pe = [patheffects.withStroke(linewidth=lw + halo_w, foreground='black')]
    for o in rec['overlays'] if isinstance(rec['overlays'], list) else []:
        k = o['kind']
        if k not in kinds or k == 'midline':
            continue
        col = {'box': IMG['ref'], 'pred': IMG['pred'], 'edema': IMG['edema'], 'refcontour': IMG['ref']}[k]
        ls = (0, (3.2, 1.6)) if k == 'box' else '-'
        for sp in o['subpaths']:
            sp = np.asarray(sp)
            ax.plot(sp[:, 0], sp[:, 1], color=col, lw=lw, ls=ls, zorder=4, solid_joinstyle='round', path_effects=pe if k == 'pred' else None)
