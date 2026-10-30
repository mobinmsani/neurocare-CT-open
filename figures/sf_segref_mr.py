"""Supplementary figure - The two Seg-CQ500 reference masks on scans that the CQ500 reader majority
labeled negative (CT-108 and CT-187), shown on the released Seg-CQ500 images, and the attenuation inside every analyzed mask.

a  CT-108: the axial slice with the largest mask area, without and with the reference mask contour
b  CT-187: the same, with an enlarged view of the mask region
c  Attenuation (HU) distributions inside the masks: CT-108, CT-187 and the other 41 analyzed masks (pooled and per mask), with the
   brain-mask voxels outside the hemorrhage mask of CT-108 for comparison
Model-free and descriptive (segcq500_mask_hist.py, segcq500_mask_check.py); the figure does not adjudicate the masks.
Display: level 40 / width 80 HU, linear; no other image processing.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncstyle import *
from ncdata import *
from nc_c import *

if not all((ASSETS / f'segref_{k}_{p}.npy').exists() for k in (108, 187) for p in ('ct', 'mask')):
    print('sf_segref_mr.py: the CT-108 and CT-187 slices are not redistributed with the code. Run analysis/segcq500_mask_hist.py on the '
          'Seg-CQ500 archive (bash run_all.sh with SEGCQ500_ZIP set) to create them in figures/assets/. Figure skipped.')
    sys.exit(0)

HFIG = 108
fig = plt.figure(figsize=(W2, HFIG * MM))
H = np.load(D / 'derived' / 'segcq500_mask_hu_hist.npz')
edges, hin, hout, scans = H['edges'], H['inside'], H['brain_outside'], list(H['scans'])
MJ = json.load(open(D / 'derived' / 'segcq500_mask_check.json'))
mq = load_cq500().set_index('name')
assert len(scans) == 43
i108, i187 = scans.index('CQ500-CT-108'), scans.index('CQ500-CT-187')
oth = [i for i in range(len(scans)) if i not in (i108, i187)]
votes_ = {s: int(sum(mq.loc[s, f'R{q}:ICH'] for q in (1, 2, 3))) for s in ('CQ500-CT-108', 'CQ500-CT-187')}

TS, TY, GAP = 42.0, 9.0, 2.0
xs = [2.0 + i * (TS + GAP) for i in range(4)]


def tile(x0, key, contour, zoom=None, label=None):
    rec = json.load(open(ASSETS / f'{key}.json'))
    ct = np.load(ASSETS / f'{key}_ct.npy'); mk = np.load(ASSETS / f'{key}_mask.npy')
    ax = axes_mm(fig, x0, TY, TS, TS)
    crop = head_bbox(ct, thr=40, margin=4) if zoom is None else zoom
    draw_ct(ax, ct, crop, interp='bicubic' if zoom is None else 'nearest')
    if contour and mk.any():
        ax.contour(mk.astype(float), [0.5], colors=[IMG['ref']], linewidths=1.0 if zoom is None else 1.3, zorder=5)
    tile_label(ax, label or rec['case'].replace('CQ500-', ''))
    if zoom is None:
        orient_marks(ax, crop, size=7.0, color='#d9d9d9')
    else:
        for s_ in ax.spines.values():
            s_.set_visible(True); s_.set_edgecolor(INK); s_.set_linewidth(1.0)
    return ax, rec, ct, mk, crop


a1, r108, _, _, _ = tile(xs[0], 'segref_108', False)
a2, _, _, _, _ = tile(xs[1], 'segref_108', True)
a3, r187, ct187, mk187, crop187 = tile(xs[2], 'segref_187', True)
ys_, xs_ = np.nonzero(mk187)
cx, cy = (xs_.min() + xs_.max()) / 2, (ys_.min() + ys_.max()) / 2
half = max(xs_.max() - xs_.min(), ys_.max() - ys_.min()) / 2 + 22
zb = (cx - half, cx + half, cy - half, cy + half)
a3.add_patch(Rectangle((zb[0], zb[2]), 2 * half, 2 * half, fill=False, ec=WHITE, lw=1.0, zorder=8))
zf = (crop187[1] - crop187[0]) / (2 * half)
a4, _, _, _, _ = tile(xs[3], 'segref_187', True, zoom=zb, label=f'CT-187 ×{zf:.0f}')
f108 = MJ['focus']['CQ500-CT-108']; f187 = MJ['focus']['CQ500-CT-187']
cap = [(xs[0], 'no overlay', f"slice {r108['slice_index'] + 1} of {r108['n_slices']}"),
       (xs[1], f"mask {f108['mask_ml']:.1f} mL; on this slice", f"{r108['mask_area_cm2']:.1f} cm²; {votes_['CQ500-CT-108']} of 3 readers: ICH"),
       (xs[2], f"mask {f187['mask_ml']:.1f} mL; {votes_['CQ500-CT-187']} of 3 readers: ICH", f"slice {r187['slice_index'] + 1} of {r187['n_slices']}"),
       (xs[3], 'enlarged framed region', f"median {f187['hu_median']:.0f} HU in the mask")]
for x0, l1, l2 in cap:
    fig_text(fig, x0, TY + TS + 3.4, l1, color=INK)
    fig_text(fig, x0, TY + TS + 6.6, l2, color=INK2)
ptitle(fig, a1, 'a', 'CT-108: largest-area slice, without and with the reference mask', dx_mm=0.0)
ptitle(fig, a3, 'b', 'CT-187: reference mask and enlarged view', dx_mm=0.0)

# ======================================================================= c: attenuation distributions
CY = TY + TS + 19.0
axC = axes_mm(fig, 14.0, CY, 92.0, 28.0)
ctr = edges[:-1] + 0.5
def dens(h):
    h = np.asarray(h, float)
    return h / h.sum()
# per-mask curves (other masks), thin
for i in oth:
    axC.plot(ctr, 100 * dens(hin[i]), color='#c9c7c2', lw=1.0, zorder=1)
pooled = dens(hin[oth].sum(0))
axC.plot(ctr, 100 * pooled, color=INK2, lw=1.6, zorder=3, ls=(0, (5, 1.6)))
axC.plot(ctr, 100 * dens(hout[i108]), color=M['blast'], lw=1.4, zorder=2, ls=(0, (1.2, 1.4)))
axC.plot(ctr, 100 * dens(hin[i187]), color=INK, lw=1.3, zorder=4, ls=(0, (3, 1.5)))
axC.plot(ctr, 100 * dens(hin[i108]), color=INK, lw=1.8, zorder=5)
axC.set_xlim(-20, 120); axC.set_ylim(0, None)
axC.set_xticks([-20, 0, 20, 40, 60, 80, 100, 120])
axC.set_xlabel('Attenuation (HU)'); axC.set_ylabel('Voxels per 1 HU (%)')
grid(axC, 'y')
ptitle(fig, axC, 'c', 'Attenuation inside the reference masks', sub='Seg-CQ500 images, 43 analyzed masks', dx_mm=12.0)
# key
KX = 112.0
items = [(dict(color=INK, lw=1.8), f"CT-108 mask (median {f108['hu_median']:.0f} HU)"),
         (dict(color=INK, lw=1.3, ls=(0, (3, 1.5))), f"CT-187 mask (median {f187['hu_median']:.0f} HU)"),
         (dict(color=INK2, lw=1.6, ls=(0, (5, 1.6))), f"other 41 masks, pooled (median of"),
         (None, f"per-mask medians {MJ['reference_distribution']['hu_median']['median']:.0f} HU)"),
         (dict(color='#c9c7c2', lw=1.0), 'each of the other 41 masks'),
         (dict(color=M['blast'], lw=1.4, ls=(0, (1.2, 1.4))), 'CT-108 brain mask outside the'),
         (None, f"hemorrhage mask (median {f108['brain_parenchyma_hu_median']:.0f} HU)")]
yk = CY + 1.5
for st, lab in items:
    if st is not None:
        sw = axes_mm(fig, KX, yk - 1.3, 7.0, 2.6); sw.axis('off'); sw.set_xlim(0, 1); sw.set_ylim(0, 1)
        sw.plot([0, 1], [0.5, 0.5], **st, solid_capstyle='butt', dash_capstyle='butt')
    fig_text(fig, KX + 8.5, yk + 1.0, lab, color=INK if st is not None else INK)
    yk += 3.4
fig_text(fig, KX, yk + 2.2, 'Descriptive only: no adjudication.', color=INK2)
print('CT-108 slice', r108['slice_index'], 'area', r108['mask_area_cm2'], '; CT-187 slice', r187['slice_index'], 'zoom', round(zf, 1))
print('pooled other-mask median HU from histogram', float(ctr[np.searchsorted(np.cumsum(pooled), 0.5)]))
print('overflow', overflow_mm(fig))
save(fig, 'SF_segref', 'supp_mr')
