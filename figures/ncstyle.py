"""NeuroCare CT figure system: one visual language for every main and supplementary figure.

Design rules (common journal artwork requirements)
- Final size: 180 mm (A4 page width) or 88 mm (single column). Arial-metric sans serif throughout.
- Type at final size: one size, 7 pt Arial, for every text element (ticks, labels, keys, annotations, panel titles in
  bold, lower-case bold panel letters a, b, c in the same size); nothing below 7 pt.
  Sentence case; a single space between number and unit (30 mL); thousands with commas (1,000).
- Lines: the thinnest line in a final figure is 1 pt (axes, ticks, grids, box outlines, hatching); data lines 1.5 pt.
- No bars summarizing a distribution (mean/median): proportions are dots with Wilson intervals, distributions are strip or
  box plots showing every scan; bars only for counts.
- Colour encodes identity of a model/module (plots) or role of an overlay (images), and is always backed by a second
  channel (line style or marker shape) so that figures survive greyscale printing and colour-vision deficiency.
- Readers/raters are ink diamonds; human-agreement benchmarks are warm grey; thresholds are dashed ink lines.
- Labels tied to surgical criteria carry a dagger (†) rather than relying on colour.
"""
import os, re, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path

ROOT = Path(os.environ.get('NEUROCARE_FIGS', Path(__file__).resolve().parent))
OUTDIR = ROOT / 'out'
ASSETS = ROOT / 'assets'
_SHAPES = ASSETS / 'tile_shapes.json'
TILE_SHAPES = json.load(open(_SHAPES)) if _SHAPES.exists() else {}
TILES_ABSENT = []


def load_npy(path):
    """Load a stored image tile (CT display slice or mask). The CT tiles and masks are not redistributed with the code, so when a
    file is absent a blank array of the stored shape is returned and the image panel is drawn with a note; plots are unaffected."""
    path = Path(path)
    if path.exists():
        return np.load(path)
    if path.name not in TILE_SHAPES:
        raise FileNotFoundError(path)
    TILES_ABSENT.append(path.name)
    shape, dtype = TILE_SHAPES[path.name]
    return np.zeros(shape, dtype=dtype)


def tile_name(key):
    return f'{key}.npy' if ((ASSETS / f'{key}.npy').exists() or f'{key}.npy' in TILE_SHAPES) else f'{key}_ct.npy'
MM = 1 / 25.4
W2 = 180 * MM
W1 = 88 * MM

# type scale (pt) at final size
FS = dict(tick=7.0, note=7.0, key=7.0, label=7.0, title=7.0, letter=7.0, small=7.0, cell=7.0)
LW_MIN = 1.0   # pt, thinnest permitted line

# ------------------------------------------------------------------ palette
INK = '#1d1d1f'          # primary text, readers
INK2 = '#4b4b50'         # secondary text
MUTED = '#8c8c91'        # tertiary text, minor marks
AXIS = '#3c3c40'
GRID = '#ecebe7'
BAND = '#f4f2ee'         # grouping bands
BAND2 = '#ebe8e2'
CEIL = '#dedad2'         # reference (rater/reader) ceiling band
WHITE = '#ffffff'

M = dict(                # model / module identity (plots)
    v2='#2c6fb7',        # v2 multi-window U-Net (primary probe)
    v2pm='#86b3e4',      # v2 prespecified (with post-mask)
    v3b='#e0663a',       # v3b six-class subtype model
    v3='#f2b49a',
    v1='#8a8a8a',        # v1 compact U-Net (baseline)
    blast='#2a9d8f',     # BLAST-CT pretrained comparator
    rule='#c2185b',      # rule-based modules (ideal midline, CSF-offset proxy)
    edema='#b07d12',     # perihematomal-edema model / expert edema
    reader=INK,
)
LS = dict(v2='-', v2pm=(0, (4, 1.6)), v3b=(0, (5, 1.8)), v3=(0, (1.2, 1.4)), v1=(0, (1.2, 1.4)), blast=(0, (6, 1.6, 1.2, 1.6)), rule='-', edema='-')
MK = dict(v2='o', v2pm='o', v3b='s', v3='s', v1='^', blast='D', rule='o', edema='o', reader='D')
DAGGER = '\u2020'
CRIT = '#a3263a'         # data faults, not-evaluable, surgical-criterion labels
CRIT_NOTE = 'crimson is reserved for data faults and for endpoints without an open reference; surgical labels carry a dagger'
EV = dict(ext='#3d2b74', int='#7559b0', exp='#bba8e6', ne='#ffffff')   # evidence obtained in this study: ordinal violets; not evaluated = open
EVTXT = dict(ext=WHITE, int=WHITE, exp=INK, ne=INK2)
EVLAB = dict(ext='External', int='Internal or transfer', exp='Exploratory', ne='Not evaluated')
# validation capacity of the best open reference (model-independent rubric): ordinal fill of one hue + open crimson ring for none
CAP = dict(adequate='#0f4c81', limited='#8fb4d9', none='#ffffff')
CAPEDGE = dict(adequate='#0f4c81', limited='#0f4c81', none=CRIT)
CAPTXT = dict(adequate=WHITE, limited=INK, none=CRIT)
CAPLAB = dict(adequate='Adequate', limited='Limited', none='None')

# image overlays (bright, CVD-safe on greyscale CT): reference = yellow, prediction = azure
IMG = dict(ref='#ffd43b', pred='#4dabf7', midline='#ff4fa3', edema='#ffa94d', tp='#ffffff', fp='#4dabf7', fn='#ffd43b')

# source families (subtle lane tints in schematics)
FAM = dict(rsna='#e9eff7', cq='#f8efe7', phe='#eaf3ee')
FAMEDGE = dict(rsna='#9fb7d6', cq='#e0b99a', phe='#9cc8b0')

SUB = ['EDH', 'IPH', 'IVH', 'SAH', 'SDH']


def setup():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 7.0,
        'axes.titlesize': 7.0,
        'axes.labelsize': 7.0,
        'xtick.labelsize': 7.0,
        'ytick.labelsize': 7.0,
        'legend.fontsize': 7.0,
        'legend.title_fontsize': 7.0,
        'axes.linewidth': 1.0,
        'axes.edgecolor': AXIS,
        'axes.labelcolor': INK,
        'axes.labelpad': 2.5,
        'axes.titlepad': 5,
        'axes.titleweight': 'bold',
        'axes.titlelocation': 'left',
        'axes.titlecolor': INK,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.axisbelow': True,
        'xtick.color': AXIS, 'ytick.color': AXIS,
        'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
        'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
        'xtick.minor.width': 1.0, 'ytick.minor.width': 1.0,
        'xtick.major.size': 2.8, 'ytick.major.size': 2.8,
        'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6,
        'xtick.major.pad': 1.8, 'ytick.major.pad': 1.8,
        'text.color': INK,
        'lines.linewidth': 1.5,
        'lines.solid_capstyle': 'round',
        'legend.frameon': False,
        'legend.handlelength': 1.6,
        'legend.handletextpad': 0.5,
        'legend.borderaxespad': 0.3,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        # exponents (exact P values) are typeset with the text font, not a separate math font
        'mathtext.fontset': 'custom', 'mathtext.rm': 'sans', 'mathtext.it': 'sans:italic', 'mathtext.bf': 'sans:bold',
        'mathtext.sf': 'sans', 'mathtext.cal': 'sans', 'mathtext.tt': 'sans', 'mathtext.default': 'regular',
        'figure.dpi': 150, 'savefig.dpi': 600,
        'image.interpolation': 'bicubic',
        'hatch.linewidth': 1.0,
        'patch.linewidth': 1.0,
        'grid.linewidth': 1.0,
    })


setup()


# ------------------------------------------------------------------ helpers
def mm2fig(fig, x_mm, y_mm):
    w, h = fig.get_size_inches()
    return x_mm * MM / w, 1 - y_mm * MM / h


def axes_mm(fig, x_mm, y_mm, w_mm, h_mm, **kw):
    """Axes positioned in mm from the top-left corner of the figure."""
    W, H = fig.get_size_inches()
    return fig.add_axes([x_mm * MM / W, 1 - (y_mm + h_mm) * MM / H, w_mm * MM / W, h_mm * MM / H], **kw)


def letter(fig, x_mm, y_mm, s, size=7.0):
    W, H = fig.get_size_inches()
    fig.text(x_mm * MM / W, 1 - y_mm * MM / H, s, fontsize=size, fontweight='bold', color=INK, ha='left', va='baseline')


def title(ax, s, sub=None, pad=5.5):
    ax.set_title(s, loc='left', fontsize=7.0, fontweight='bold', color=INK, pad=pad + (7 if sub else 0))
    if sub:
        ax.text(0, 1.0, sub, transform=ax.transAxes, fontsize=7.0, color=INK2, ha='left', va='bottom',
                transform_rotates_text=False).set_y(1.0 + 0.0)


def ptitle(fig, ax, L, s, sub=None, dx_mm=None, dy_mm=3.2, lsize=7.0):
    """Panel letter + bold title, placed above the axes frame. dx_mm: letter offset left of the frame."""
    W, H = fig.get_size_inches()
    pos = ax.get_position()
    x0 = pos.x0 * W / MM
    y_top = (1 - pos.y1) * H / MM
    lx = x0 - (dx_mm if dx_mm is not None else 0)
    yb = y_top - dy_mm - (2.9 if sub else 0)
    fig.text(lx * MM / W, 1 - yb * MM / H, L.lower(), fontsize=lsize, fontweight='bold', color=INK, ha='left', va='baseline')
    tx = lx + 3.0 if dx_mm is not None else x0 + 3.0
    fig.text(tx * MM / W, 1 - yb * MM / H, s, fontsize=7.0, fontweight='bold', color=INK, ha='left', va='baseline')
    if sub:
        fig.text(tx * MM / W, 1 - (yb + 2.9) * MM / H, sub, fontsize=7.0, color=INK2, ha='left', va='baseline')


def grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, lw=1.0, zorder=0)
    ax.set_axisbelow(True)


def nospine(ax, which=('left',)):
    for s in which:
        ax.spines[s].set_visible(False)


def minus(s):
    return s.replace('-', '−')


def fmt_p(p):
    """Exact P value: two significant figures, scientific notation below 0.001."""
    if p < 0.001:
        m, e = f'{p:.1e}'.split('e')
        return f'P = {m} × 10$^{{\u2212{abs(int(e))}}}$'
    if p < 0.1:
        return f'P = {p:#.2g}'
    return f'P = {p:.2f}'


def ci_point(ax, x, y, lo, hi, color, marker='o', filled=True, ms=4.6, lw=1.2, orient='h', z=3, mew=None, ecolor=None):
    mfc = color if filled else WHITE
    mec = WHITE if filled else color
    mew = (0.6 if filled else 1.0) if mew is None else mew
    if orient == 'h':
        ax.plot([lo, hi], [y, y], color=ecolor or color, lw=lw, zorder=z, solid_capstyle='round')
    else:
        ax.plot([x, x], [lo, hi], color=ecolor or color, lw=lw, zorder=z, solid_capstyle='round')
    ax.plot([x], [y], marker=marker, ms=ms, mfc=mfc, mec=mec, mew=mew, ls='', zorder=z + 1)


def legend_handles(items):
    """items: list of (label, dict(marker=..., color=..., filled=..., ls=..., lw=...))"""
    hs = []
    for lab, st in items:
        if st.get('patch'):
            hs.append(Rectangle((0, 0), 1, 1, fc=st['color'], ec=st.get('ec', 'none'), lw=1.0, hatch=st.get('hatch'), label=lab))
            continue
        filled = st.get('filled', True)
        hs.append(Line2D([], [], marker=st.get('marker', 'o'), ms=st.get('ms', 4.6), ls=st.get('ls', ''), lw=st.get('lw', 1.4),
                         color=st['color'], mfc=st['color'] if filled else WHITE, mec=WHITE if (filled and st.get('marker', 'o') not in ('x', '+', '|', '_')) else st['color'],
                         mew=0.6 if filled else 1.0, label=lab))
    return hs


def halo(lw=2.2, fg='white'):
    return [patheffects.withStroke(linewidth=lw, foreground=fg)]


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def plain_log(ax, axis='x', ticks=None, labels=None):
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    a = ax.xaxis if axis == 'x' else ax.yaxis
    if ticks is not None:
        a.set_major_locator(FixedLocator(ticks))
        a.set_ticklabels(labels if labels is not None else [f'{t:g}' for t in ticks])
    a.set_minor_formatter(NullFormatter())


# ------------------------------------------------------------------ CT image helpers
def load_asset(key):
    rec = json.load(open(ASSETS / f'{key}.json'))
    return rec, load_npy(ASSETS / tile_name(key))


def head_bbox(ct, thr=40, margin=6, square=True):
    """Bounding box of the head (largest bright component), excluding the scanner table where possible."""
    from scipy import ndimage as ndi
    m = ct > thr
    lab, n = ndi.label(m)
    if n == 0:
        return 0, ct.shape[1], 0, ct.shape[0]
    sizes = ndi.sum(m, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(lab == k)
    x0, x1, y0, y1 = xs.min() - margin, xs.max() + margin, ys.min() - margin, ys.max() + margin
    if square:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        s = max(x1 - x0, y1 - y0)
        x0, x1, y0, y1 = cx - s / 2, cx + s / 2, cy - s / 2, cy + s / 2
    return x0, x1, y0, y1


def midline_centerline(subpaths):
    """Collapse the thin midline band (contour of a dilated line mask) to its centre line."""
    pts = np.vstack([np.asarray(s) for s in subpaths])
    c = pts.mean(0)
    u, s_, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]
    t = (pts - c) @ d
    return np.vstack([c + t.min() * d, c + t.max() * d])


def draw_ct(ax, ct, crop=None, vmin=0, vmax=255, interp='bicubic'):
    ax.imshow(ct, cmap='gray', vmin=vmin, vmax=vmax, interpolation=interp, zorder=0)
    if TILES_ABSENT and not np.asarray(ct).any():   # blank stand-in for a tile that is not redistributed
        ax.text(0.5, 0.5, 'CT image not\nredistributed', transform=ax.transAxes, ha='center', va='center', color='white',
                fontsize=FS['note'], zorder=10)
    if crop is not None:
        x0, x1, y0, y1 = crop
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor('black')


def draw_overlays(ax, rec, kinds=('box', 'pred', 'midline', 'edema', 'refcontour'), lw=1.0):
    for o in rec['overlays'] if isinstance(rec['overlays'], list) else []:
        k = o['kind']
        if k not in kinds:
            continue
        if k == 'midline':
            ln = midline_centerline(o['subpaths'])
            ax.plot(ln[:, 0], ln[:, 1], color=IMG['midline'], lw=lw * 1.05, ls=(0, (4.5, 2.2)), zorder=4, dash_capstyle='round')
            continue
        col = {'box': IMG['ref'], 'pred': IMG['pred'], 'edema': IMG['edema'], 'refcontour': IMG['ref']}[k]
        ls = (0, (3.2, 1.6)) if k == 'box' else '-'
        for sp in o['subpaths']:
            sp = np.asarray(sp)
            ax.plot(sp[:, 0], sp[:, 1], color=col, lw=lw, ls=ls, zorder=4, solid_joinstyle='round')


def orient_marks(ax, crop, color='#e6e6e6', size=7.0):
    """Radiological convention: patient right on image left."""
    x0, x1, y0, y1 = crop
    pe = [patheffects.withStroke(linewidth=1.6, foreground='black')]
    ax.text(x0 + 0.035 * (x1 - x0), y0 + 0.5 * (y1 - y0), 'R', color=color, fontsize=size, ha='left', va='center', fontweight='bold', path_effects=pe)
    ax.text(x1 - 0.035 * (x1 - x0), y0 + 0.5 * (y1 - y0), 'L', color=color, fontsize=size, ha='right', va='center', fontweight='bold', path_effects=pe)


def corner_text(ax, s, loc='tl', color='white', size=7.0, weight='normal', pad=0.03, bg=None):
    x = pad if loc[1] == 'l' else 1 - pad
    y = 1 - pad if loc[0] == 't' else pad
    t = ax.text(x, y, s, transform=ax.transAxes, color=color, fontsize=size, fontweight=weight,
                ha='left' if loc[1] == 'l' else 'right', va='top' if loc[0] == 't' else 'bottom', zorder=10, linespacing=1.15)
    if bg:
        t.set_bbox(dict(boxstyle='round,pad=0.25,rounding_size=0.3', fc=bg, ec='none', alpha=0.85))
    return t


# ------------------------------------------------------------------ saving
def _fix_svg_fonts(path):
    s = path.read_text()
    s = re.sub(r"font-family: '?Liberation Sans'?", "font-family: Arial, Helvetica, 'Liberation Sans', sans-serif", s)
    s = re.sub(r'font-family:Liberation Sans', "font-family:Arial, Helvetica, 'Liberation Sans', sans-serif", s)
    path.write_text(s)


def save(fig, name, sub='main', formats=('png', 'pdf', 'svg', 'tiff'), dpi=600):
    d = OUTDIR / sub
    d.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        p = d / f'{name}.{ext}'
        if ext == 'tiff':
            fig.savefig(p, dpi=dpi, facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            im = Image.open(p)
            if im.mode != 'RGB':
                bg = Image.new('RGB', im.size, 'white')
                bg.paste(im, mask=im.split()[-1] if im.mode == 'RGBA' else None)
                bg.save(p, compression='tiff_lzw', dpi=(dpi, dpi))
        else:
            fig.savefig(p, dpi=dpi, facecolor='white')
        if ext == 'svg':
            _fix_svg_fonts(p)
    # quick-look preview for checking, plus greyscale and colour-vision-deficiency simulations for QA
    pv = d / f'{name}_preview.png'
    fig.savefig(pv, dpi=200, facecolor='white')
    try:
        qa_write(fig, name, sub, pv)
    finally:
        plt.close(fig)


def overflow_mm(fig):
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    return tuple(round(v * 25.4, 1) for v in (-bb.x0, bb.x1 - w, -bb.y0, bb.y1 - h))


def canvas_axes(fig):
    """A full-figure axes in mm units (origin top-left, y downward) for schematic drawing."""
    W, H = fig.get_size_inches()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W / MM); ax.set_ylim(H / MM, 0)
    ax.axis('off')
    return ax


def rbox(ax, x, y, w, h, fc=WHITE, ec='#cfcbc3', lw=1.0, r=1.2, z=2, ls='-', alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=z, ls=ls, alpha=alpha)
    ax.add_patch(p)
    return p


# ------------------------------------------------------------------ QA (run automatically by save)
_DEUT = np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]])   # Machado 2009, severity 1
_PROT = np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]])


def _lin(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _gam(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055) * 255


def qa_write(fig, name, sub, preview_png):
    """Record the smallest text size and thinnest visible line, overflow, and write greyscale/CVD previews."""
    import matplotlib.text as mtext
    from matplotlib.lines import Line2D as _L
    from matplotlib.patches import Patch as _P
    from PIL import Image
    fig.canvas.draw()
    sizes, small = [], []
    for t in fig.findobj(mtext.Text):
        if t.get_visible() and t.get_text().strip():
            sizes.append(t.get_fontsize())
            if t.get_fontsize() < 6.99:
                small.append((round(t.get_fontsize(), 2), t.get_text()[:40]))
    widths, thin = [], []
    for a in fig.findobj(lambda o: isinstance(o, (_L, _P))):
        if not a.get_visible():
            continue
        lw = a.get_linewidth()
        if isinstance(a, _L):
            if a.get_linestyle() in ('None', '', ' ') or len(a.get_xdata()) < 2:
                continue
        else:
            ec = a.get_edgecolor()
            if lw == 0 or (ec is not None and len(ec) == 4 and ec[3] == 0):
                continue
        widths.append(lw)
        if lw < LW_MIN - 1e-6:
            thin.append(round(float(lw), 2))
    for ax in fig.axes:   # axis spines and ticks
        for sp in ax.spines.values():
            if sp.get_visible():
                widths.append(sp.get_linewidth())
                if sp.get_linewidth() < LW_MIN - 1e-6:
                    thin.append(round(float(sp.get_linewidth()), 2))
    rep = dict(name=name, size_mm=[round(v * 25.4, 1) for v in fig.get_size_inches()], min_font_pt=min(sizes) if sizes else None,
               n_text=len(sizes), n_text_below_7pt=len(small), text_below_7pt=small[:25], min_line_pt=min(widths) if widths else None,
               n_lines_below_1pt=len(thin), lines_below_1pt=sorted(set(thin))[:10], overflow_mm=overflow_mm(fig))
    qd = OUTDIR / 'qa'
    qd.mkdir(parents=True, exist_ok=True)
    (qd / f'{sub}__{name}.json').write_text(json.dumps(rep, indent=1, default=float))
    im = np.asarray(Image.open(preview_png).convert('RGB')).astype(float)
    lin = _lin(im)
    Y = lin @ np.array([0.2126, 0.7152, 0.0722])
    Image.fromarray(_gam(np.repeat(Y[..., None], 3, -1)).astype(np.uint8)).save(qd / f'{sub}__{name}_grey.png')
    Image.fromarray(_gam(lin @ _DEUT.T).astype(np.uint8)).save(qd / f'{sub}__{name}_deutan.png')
    return rep
