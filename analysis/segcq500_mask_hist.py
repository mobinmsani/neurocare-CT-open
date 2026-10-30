"""Attenuation histograms of the Seg-CQ500 reference masks and illustrative slices of CT-108 and CT-187 (post hoc, 29 Oct 2026).
Model-free and descriptive, like segcq500_mask_check.py: for every analyzed scan, voxel counts per 1-HU bin (-100 to 200 HU) inside
the hemorrhage mask and inside the brain mask outside the hemorrhage mask; for CT-108 and CT-187, the axial slice with the largest
mask area in radiological orientation (patient right on image left), windowed at level 40 / width 80 HU, with the mask on that slice.
These describe the masks; they do not adjudicate whether hemorrhage is present.
Usage: python segcq500_mask_hist.py Seg-CQ500.zip analyzed_ids.txt out_dir assets_dir"""
import sys, os, json, zipfile, tempfile
import numpy as np, nibabel as nib
zp, idfile, outd, assets = sys.argv[1:5]
analyzed = sorted(set(open(idfile).read().split()), key=lambda s: int(s.split('-')[-1]))
z = zipfile.ZipFile(zp)
EDGES = np.arange(-100, 201)          # 1-HU bins: [e, e+1)
FOCUS = ('CQ500-CT-108', 'CQ500-CT-187')


def load(name, fn):
    data = z.read(f'Seg-CQ500/data/volumes/{name}/{fn}')
    with tempfile.NamedTemporaryFile(suffix='_' + fn) as t:
        t.write(data); t.flush()
        im = nib.as_closest_canonical(nib.load(t.name))
        return np.asanyarray(im.dataobj).astype(np.float32), im.header.get_zooms()[:3], im.affine


hin, hout, names = [], [], []
for s in analyzed:
    ct, zc, ac = load(s, 'CT.nii'); m, _, am = load(s, 'ICH_mask.nii.gz'); b, _, ab = load(s, 'brain.nii.gz')
    assert ct.shape == m.shape == b.shape and np.allclose(ac, am, atol=1e-3)
    M = m > 0.5; B = b > 0.5
    hin.append(np.histogram(ct[M], bins=EDGES)[0]); hout.append(np.histogram(ct[B & ~M], bins=EDGES)[0]); names.append(s)
    if s in FOCUS:
        k = int(np.argmax(M.sum(axis=(0, 1))))
        sl = ct[:, :, k].T[::-1, ::-1]; ms = M[:, :, k].T[::-1, ::-1]          # rows anterior to posterior, columns patient right to left
        w = np.clip((sl - 0.0) / 80.0 * 255.0, 0, 255).astype(np.uint8)         # level 40, width 80 HU
        key = 'segref_' + s.split('-')[-1]
        np.save(os.path.join(assets, key + '_ct.npy'), w); np.save(os.path.join(assets, key + '_mask.npy'), ms)
        json.dump(dict(key=key, case=s, slice_index=k, n_slices=int(ct.shape[2]), pixel_mm=[float(zc[0]), float(zc[1])], slice_mm=float(zc[2]),
                       mask_area_cm2=round(float(ms.sum()) * float(zc[0]) * float(zc[1]) / 100.0, 2), window='level 40, width 80 HU',
                       source='Seg-CQ500 released image and reference mask (CC BY-NC-SA 4.0 images from CQ500)'),
                  open(os.path.join(assets, key + '.json'), 'w'), indent=1)
    print(s, int(M.sum()), flush=True)
    del ct, m, b, M, B
os.makedirs(outd, exist_ok=True)
np.savez_compressed(os.path.join(outd, 'segcq500_mask_hu_hist.npz'), edges=EDGES, inside=np.array(hin), brain_outside=np.array(hout), scans=np.array(names))
print('done', len(names))
