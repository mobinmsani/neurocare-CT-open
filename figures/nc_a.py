"""Helpers for Figures 1-3 (schematic drawing in mm on a canvas axes; y grows downward).

Shared by fig1_design.py, fig2_capacity.py and fig3_reference.py. Uses ncstyle conventions only
(7 pt type, lines >= 1 pt); nothing here changes ncstyle or ncdata.
"""
from ncstyle import *

FS7 = 7.0
LW = 1.0          # thinnest line (pt)


def txt(ax, x, y, s, ha='left', va='center', color=INK, bold=False, size=FS7, italic=False, z=6, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight='bold' if bold else 'normal',
                   style='italic' if italic else 'normal', zorder=z, **kw)


def width_mm(fig, s, size=FS7, bold=False):
    """Rendered width of a string in mm (for layout)."""
    t = fig.text(0, 0, s, fontsize=size, fontweight='bold' if bold else 'normal')
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi * 25.4


def glyph(ax, x, y, kind, c=INK, s=1.0, z=6):
    """Reference-type glyphs, centred at (x, y) in mm; base width about 3 mm. Lines are 1 pt."""
    if kind == 'voxel':                      # voxel masks: 2 x 2 filled squares
        for i in range(2):
            for j in range(2):
                ax.add_patch(Rectangle((x - 1.25 * s + i * 1.35 * s, y - 1.25 * s + j * 1.35 * s), 1.15 * s, 1.15 * s,
                                       fc=c, ec='none', zorder=z))
    elif kind == 'voxel3':                   # voxel masks from three independent raters: three stacked outlines
        for k in range(3):
            ax.add_patch(Rectangle((x - 1.75 * s + k * 0.75 * s, y - 1.45 * s + (2 - k) * 0.5 * s), 2.0 * s, 1.55 * s,
                                   fc=WHITE, ec=c, lw=LW, zorder=z + k))
    elif kind == 'label':                    # scan-level labels from three readers: tag with three dots
        ax.add_patch(FancyBboxPatch((x - 1.7 * s, y - 1.0 * s), 3.4 * s, 2.0 * s, boxstyle=f'round,pad=0,rounding_size={0.55 * s}',
                                    fc=WHITE, ec=c, lw=LW, zorder=z))
        for k in range(3):
            ax.add_patch(plt.Circle((x - 0.85 * s + k * 0.85 * s, y), 0.3 * s, fc=c, ec='none', zorder=z + 1))
    elif kind == 'box':                      # bounding boxes: dashed square
        ax.add_patch(Rectangle((x - 1.35 * s, y - 1.2 * s), 2.7 * s, 2.4 * s, fc='none', ec=c, lw=LW, ls=(0, (2.0, 1.2)), zorder=z))
    elif kind == 'side':                     # documented hemorrhage side: double arrow
        ax.annotate('', xy=(x + 1.7 * s, y), xytext=(x - 1.7 * s, y),
                    arrowprops=dict(arrowstyle='<|-|>', lw=LW, color=c, mutation_scale=4.5, shrinkA=0, shrinkB=0), zorder=z)
    elif kind == 'edema':                    # edema masks: ring around a filled core
        ax.add_patch(plt.Circle((x, y), 1.35 * s, fc='none', ec=c, lw=LW, zorder=z))
        ax.add_patch(plt.Circle((x, y), 0.6 * s, fc=c, ec='none', zorder=z))
    elif kind == 'request':                  # measurements exist but only on request: dashed open tag with a ruler tick
        ax.add_patch(FancyBboxPatch((x - 1.7 * s, y - 1.0 * s), 3.4 * s, 2.0 * s, boxstyle=f'round,pad=0,rounding_size={0.55 * s}',
                                    fc=WHITE, ec=c, lw=LW, ls=(0, (1.6, 1.0)), zorder=z))
        ax.plot([x - 0.9 * s, x + 0.9 * s], [y, y], color=c, lw=LW, zorder=z + 1, solid_capstyle='butt')
        for xx in (-0.9, 0.9):
            ax.plot([x + xx * s, x + xx * s], [y - 0.45 * s, y + 0.45 * s], color=c, lw=LW, zorder=z + 1, solid_capstyle='butt')
    else:
        raise ValueError(kind)


def setting_dot(ax, x, y, letter_, on, r=1.55, z=6, c_on=INK, c_off='#9a978f'):
    """Evaluation-setting marker: filled ink disc with white letter (evaluated) or open grey ring with grey letter."""
    ax.add_patch(plt.Circle((x, y), r, fc=c_on if on else WHITE, ec=c_on if on else c_off, lw=LW, zorder=z))
    ax.text(x, y + 0.08, letter_, fontsize=FS7, color=WHITE if on else c_off, ha='center', va='center', fontweight='bold', zorder=z + 1)


def badge(ax, x, y, w, h, s, fc, ec, tc, r=1.0, bold=True, z=4):
    rbox(ax, x, y, w, h, fc=fc, ec=ec, lw=LW, r=r, z=z)
    ax.text(x + w / 2, y + h / 2 + 0.05, s, fontsize=FS7, color=tc, ha='center', va='center', fontweight='bold' if bold else 'normal', zorder=z + 1)


def crit_symbol(ax, x, y, state, r=1.2, c=INK, z=6):
    """Rubric criterion state: met = filled disc; partial = half-filled; not met = open ring; n/a = dash; unknown = '?'."""
    if state == 'met':
        ax.add_patch(plt.Circle((x, y), r, fc=c, ec=c, lw=LW, zorder=z))
    elif state == 'partial':
        from matplotlib.patches import Wedge
        ax.add_patch(plt.Circle((x, y), r, fc=WHITE, ec=c, lw=LW, zorder=z))
        ax.add_patch(Wedge((x, y), r, 90, 270, fc=c, ec='none', zorder=z + 1))   # left half filled (y axis points down: still left)
    elif state == 'not met':
        ax.add_patch(plt.Circle((x, y), r, fc=WHITE, ec=c, lw=LW, zorder=z))
    elif state == 'na':
        ax.plot([x - 0.9, x + 0.9], [y, y], color='#9a978f', lw=LW, zorder=z, solid_capstyle='butt')
    elif state == 'unknown':
        ax.text(x, y + 0.1, '?', fontsize=FS7, color=INK2, ha='center', va='center', fontweight='bold', zorder=z)
    else:
        raise ValueError(state)


def endash(s):
    """Numeric ranges with an en dash, 'kappa' as the Greek letter."""
    import re
    s = re.sub(r'(?<=\d)-(?=\d)', '–', s)
    return s.replace('kappa', 'κ')
