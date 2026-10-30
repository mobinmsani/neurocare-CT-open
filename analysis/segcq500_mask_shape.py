"""Model-free spatial description of the Seg-CQ500 reference masks (post hoc): craniocaudal extent, side, share of the
ipsilateral half of the released brain mask, and share of mask voxels within 5 mm of the brain-mask surface (a crude marker of a
peripheral, possibly extra-axial, location versus a deep one). These measurements describe the masks; they do not adjudicate them.
Side uses world coordinates from the NIfTI affine (RAS+, so smaller x is patient left) relative to the brain-mask centroid.
Distances are computed on the brain-mask bounding box at half the in-plane resolution (memory); the 5-mm criterion is coarse.
Usage: python segcq500_mask_shape.py Seg-CQ500.zip analyzed_ids.txt out_dir"""
import gc
import json
import os
import sys
import tempfile
import zipfile

import nibabel as nib
import numpy as np
from scipy import ndimage

zp, idfile, outd = sys.argv[1], sys.argv[2], sys.argv[3]
analyzed = sorted(set(open(idfile).read().split()), key=lambda s: int(s.split('-')[-1]))
z = zipfile.ZipFile(zp)


def load_bool(name, fn):
    data = z.read(f'Seg-CQ500/data/volumes/{name}/{fn}')
    with tempfile.NamedTemporaryFile(suffix='_' + fn) as t:
        t.write(data); t.flush()
        im = nib.load(t.name)
        arr = np.asarray(im.dataobj) > 0.5
        return arr, im.affine, [float(x) for x in im.header.get_zooms()[:3]]


rows = []
for s in analyzed:
    M, am, zm = load_bool(s, 'ICH_mask.nii.gz')
    B, _, _ = load_bool(s, 'brain.nii.gz')
    if not M.any():
        continue
    bb = np.argwhere(B.any(axis=2))            # in-plane bounding box of the brain mask
    (i0, j0), (i1, j1) = bb.min(0), bb.max(0) + 1
    ks = np.where(B.any(axis=(0, 1)))[0]
    k0, k1 = ks.min(), ks.max() + 1
    Bc, Mc = B[i0:i1, j0:j1, k0:k1], M[i0:i1, j0:j1, k0:k1]
    del B, M
    gc.collect()
    n_mask = int(Mc.sum())
    # side: world x (RAS+) of mask voxels against the brain-mask centroid (affine is linear: centroid maps to centroid)
    off = np.array([i0, j0, k0])
    cb = np.argwhere(Bc[::2, ::2, :]).mean(0) * np.array([2, 2, 1]) + off
    cx = nib.affines.apply_affine(am, cb)[0]
    mi = np.argwhere(Mc) + off
    mx = nib.affines.apply_affine(am, mi)[:, 0]
    left = float((mx < cx).mean())
    side = 'left' if left >= 0.5 else 'right'
    del mi, mx
    # half-brain voxel count on the mask's side, on a half-resolution in-plane grid
    Bd = Bc[::2, ::2, :]
    bi = np.argwhere(Bd) * np.array([2, 2, 1]) + off
    bx = nib.affines.apply_affine(am, bi)[:, 0]
    half = int(((bx < cx) if side == 'left' else (bx >= cx)).sum()) * 4
    del bi, bx
    d = ndimage.distance_transform_edt(Bd, sampling=[2 * zm[0], 2 * zm[1], zm[2]]).astype(np.float32)
    dm = d[Mc[::2, ::2, :]]
    zz = np.where(Mc.any(axis=(0, 1)))[0]
    vv = abs(float(np.linalg.det(am[:3, :3]))) / 1000
    rows.append(dict(scan=s, mask_ml=round(n_mask * vv, 2), side=side, share_on_side=round(max(left, 1 - left), 3),
                     share_of_half_brain=round(n_mask / max(half, 1), 3),
                     share_within_5mm_of_brain_surface=round(float((dm <= 5.0).mean()), 3) if dm.size else None,
                     median_depth_mm=round(float(np.median(dm)), 1) if dm.size else None,
                     craniocaudal_extent_mm=round(float((zz.max() - zz.min() + 1) * zm[2]), 1)))
    del d, dm, Bd, Bc, Mc
    gc.collect()
    print(rows[-1], flush=True)
os.makedirs(outd, exist_ok=True)
ref = [r for r in rows if r['scan'] not in ('CQ500-CT-108', 'CQ500-CT-187') and r['median_depth_mm'] is not None]
summ = {k: dict(median=float(np.median([r[k] for r in ref])), q1=float(np.percentile([r[k] for r in ref], 25)), q3=float(np.percentile([r[k] for r in ref], 75)),
                min=float(min(r[k] for r in ref)), max=float(max(r[k] for r in ref)))
        for k in ('share_of_half_brain', 'share_within_5mm_of_brain_surface', 'median_depth_mm', 'craniocaudal_extent_mm')}
out = dict(rows=rows, focus={r['scan']: r for r in rows if r['scan'] in ('CQ500-CT-108', 'CQ500-CT-187')}, reference_distribution=summ, n_reference=len(ref),
           note='Other analyzed Seg-CQ500 masks (excluding CT-108 and CT-187). Descriptive only; no adjudication.')
json.dump(out, open(os.path.join(outd, 'segcq500_mask_shape.json'), 'w'), indent=1)
print(json.dumps(out['focus'], indent=1)); print(json.dumps(summ, indent=1))
