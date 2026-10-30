"""Helpers for Figures 6 and 7. Uses ncstyle without modifying it."""
import numpy as np
from ncstyle import *


def fig_text(fig, x_mm, y_mm, s, **kw):
    """Figure-level text at (x, y) in mm from the top-left corner; baseline alignment by default."""
    W, H = fig.get_size_inches()
    kw.setdefault('fontsize', 7.0)
    kw.setdefault('va', 'baseline')
    kw.setdefault('ha', 'left')
    kw.setdefault('color', INK)
    return fig.text(x_mm * MM / W, 1 - y_mm * MM / H, s, **kw)


def boot_mean_ci(x, B=10000, seed=20261028):
    """Bootstrap percentile 95% CI of a mean (scan as resampling unit)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (B, len(x)))
    mb = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(mb, 2.5)), float(np.percentile(mb, 97.5))


def fmt2(v):
    return minus(f'{v:.2f}')


def ci_txt(est, lo, hi, nd=2):
    return minus(f'{est:.{nd}f} ({lo:.{nd}f}–{hi:.{nd}f})')


def comma(v, nd=1):
    return f'{v:,.{nd}f}'


def ct_tile(fig, ax, key, crop_thr=60, margin=5, crop=None):
    """Draw a stored CT slice (uint8) with the shared CT template; returns crop."""
    import json
    nm = f'{key}_ct.npy' if ((ASSETS / f'{key}_ct.npy').exists() or f'{key}_ct.npy' in TILE_SHAPES) else f'{key}.npy'
    ct = load_npy(ASSETS / nm)
    if crop is None:
        crop = head_bbox(ct, thr=crop_thr, margin=margin)
    draw_ct(ax, ct, crop)
    return ct, crop


def tile_label(ax, s, loc='tl'):
    """Case identifier on a black chip in the image corner (7 pt bold, white)."""
    return corner_text(ax, s, loc, size=7.0, weight='bold', color='white', bg='#000000')
