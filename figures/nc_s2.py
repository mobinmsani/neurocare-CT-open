"""Helpers for the supplementary figures on segmentation, volume, edema and midline. Read-only use of ncstyle/ncdata; 7 pt text, >=1 pt lines."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncstyle import *
from ncdata import *
from matplotlib.colors import to_rgb

F7 = 7.0
LBL = '#000000'


def nminus(s):
    """Typographic minus for numbers only (keeps hyphens in words and identifiers such as CT-108)."""
    return re.sub(r'(?<![A-Za-z0-9])-(?=\.?\d)', '−', s)


def boot_mean_ci(x, n=10000, seed=20261026):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    mb = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(mb, 2.5)), float(np.percentile(mb, 97.5))


def overlay_axes(fig, H_mm):
    """Full-figure axes in mm (origin top-left, y down), behind everything, for keys and notes."""
    W_mm = fig.get_size_inches()[0] / MM
    ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    ax.set_xlim(0, W_mm); ax.set_ylim(H_mm, 0); ax.axis('off')
    return ax


def fig_text(fig, x_mm, y_mm, s, H_mm, **kw):
    W = fig.get_size_inches()[0]
    kw.setdefault('fontsize', F7); kw.setdefault('color', INK2); kw.setdefault('va', 'top'); kw.setdefault('ha', 'left')
    return fig.text(x_mm * MM / W, 1 - y_mm * MM / (H_mm * MM), s, **kw)


def key_row(fig, x_mm, y_mm, items, H_mm, gap_mm=4.5, sw=5.0, size=F7, ax=None):
    """Legend row in figure mm. items: (kind, colour, label, style dict). kinds: line, fill, marker, band, ring."""
    ax = ax or overlay_axes(fig, H_mm)
    rend = fig.canvas.get_renderer()
    x = x_mm
    for it in items:
        kind, col, lab = it[:3]
        st = it[3] if len(it) > 3 else {}
        if kind == 'line':
            ax.plot([x, x + sw], [y_mm, y_mm], color=col, lw=st.get('lw', 1.5), ls=st.get('ls', '-'), solid_capstyle='butt', dash_capstyle='butt')
            xt = x + sw + 1.3
        elif kind == 'fill':
            ax.add_patch(Rectangle((x, y_mm - 1.25), 3.0, 2.5, fc=col, ec=st.get('ec', 'none'), lw=1.0, hatch=st.get('hatch')))
            xt = x + 3.0 + 1.3
        elif kind == 'band':
            ax.add_patch(Rectangle((x, y_mm - 1.0), sw, 2.0, fc=col, ec='none', lw=0))
            xt = x + sw + 1.3
        elif kind == 'marker':
            filled = st.get('filled', True)
            ax.plot([x + 1.4], [y_mm], marker=st.get('marker', 'o'), ms=st.get('ms', 4.6), ls='', mfc=col if filled else WHITE,
                    mec=st.get('mec', WHITE if filled else col), mew=st.get('mew', 0.6 if filled else 1.0))
            xt = x + 3.4
        elif kind == 'text':
            xt = x
        t = ax.text(xt, y_mm, lab, fontsize=size, va='center', ha='left', color=st.get('tc', INK2), fontweight=st.get('weight', 'normal'))
        bb = t.get_window_extent(renderer=rend)
        x = xt + bb.width / fig.dpi * 25.4 + gap_mm
    return x


def strip_text(ax, s, where='above', pad_mm=1.0, size=F7, color=INK, weight='normal', ha='left', x=0.0, linespacing=1.15):
    """Metadata text outside an image axes (above or below), so it never covers anatomy."""
    fig = ax.figure
    H = fig.get_size_inches()[1] / MM
    pos = ax.get_position()
    hmm = pos.height * H
    off = pad_mm / hmm
    y = 1 + off if where == 'above' else -off
    return ax.text(x, y, s, transform=ax.transAxes, fontsize=size, color=color, fontweight=weight, ha=ha,
                   va='bottom' if where == 'above' else 'top', linespacing=linespacing, clip_on=False)


def tag(ax, x, y, s, fc=LBL, tc=WHITE, size=F7, weight='bold', ha='left', va='top', transform=None, z=12, pad=0.25):
    t = ax.text(x, y, s, transform=transform or ax.transAxes, color=tc, fontsize=size, fontweight=weight, ha=ha, va=va, zorder=z)
    t.set_bbox(dict(boxstyle=f'round,pad={pad},rounding_size=0.3', fc=fc, ec='none'))
    return t


def err_overlay(err, alpha=0.75, keyline=False):
    """RGBA overlay for an error map: 1 TP white, 2 FP azure, 3 FN yellow (semi-transparent so anatomy stays visible)."""
    rgba = np.zeros(err.shape + (4,))
    for code, key in ((1, 'tp'), (2, 'fp'), (3, 'fn')):
        m = err == code
        if m.any():
            rgba[m, :3] = to_rgb(IMG[key]); rgba[m, 3] = alpha
    return rgba


def check_qa(name, sub='supp'):
    q = json.load(open(OUTDIR / 'qa' / f'{sub}__{name}.json'))
    print(name, 'min font', q['min_font_pt'], 'text<7:', q['n_text_below_7pt'], q['text_below_7pt'][:6], '| min line', q['min_line_pt'],
          'lines<1:', q['n_lines_below_1pt'], q['lines_below_1pt'], '| overflow', q['overflow_mm'], '| size', q['size_mm'])
    return q
