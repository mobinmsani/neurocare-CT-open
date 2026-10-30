"""Descriptive, model-free characterization of the Seg-CQ500 reference masks (post hoc, 29 Oct 2026).
For every released scan: mask volume, Hounsfield-unit (HU) distribution inside the mask, overlap with the released brain mask,
and connected components. Purpose: describe the two masks on CQ500 majority-negative scans (CT-108, CT-187) against the other
masks. These measurements describe the masks; they do not adjudicate whether hemorrhage is present.
Usage: python segcq500_mask_check.py Seg-CQ500.zip analyzed_ids.txt out_dir"""
import sys, json, zipfile, io, os, tempfile, csv
import numpy as np, nibabel as nib
from scipy import ndimage
zp, idfile, outd = sys.argv[1], sys.argv[2], sys.argv[3]
analyzed = set(open(idfile).read().split())
z = zipfile.ZipFile(zp)
scans = sorted({n.split('/')[3] for n in z.namelist() if n.startswith('Seg-CQ500/data/volumes/CQ500-CT-') and n.count('/') >= 4},
               key=lambda s: int(s.split('-')[-1]))
def load(name, fn):
    data = z.read(f'Seg-CQ500/data/volumes/{name}/{fn}')
    with tempfile.NamedTemporaryFile(suffix='_' + fn) as t:
        t.write(data); t.flush()
        im = nib.load(t.name); arr = np.asanyarray(im.dataobj).astype(np.float32)
        return arr, im.header.get_zooms()[:3], im.affine, im.shape
rows = []
for s in scans:
    try:
        ct, zc, ac, shc = load(s, 'CT.nii'); m, zm, am, shm = load(s, 'ICH_mask.nii.gz'); b, zb, ab, shb = load(s, 'brain.nii.gz')
    except Exception as e:
        rows.append(dict(scan=s, error=repr(e)[:200])); continue
    geom_ok = (shc == shm == shb) and np.allclose(ac, am, atol=1e-3)
    vv = abs(float(np.linalg.det(ac[:3, :3]))) / 1000.0   # |det(affine)|: correct voxel volume also for gantry-tilted (sheared) series
    M = m > 0.5; B = b > 0.5
    r = dict(scan=s, analyzed=s in analyzed, shape='x'.join(map(str, shc)), spacing_mm=','.join(f'{x:.3f}' for x in zc), voxel_ml=round(vv * 1000, 4), geometry_ok=bool(geom_ok),
             mask_ml=round(float(M.sum()) * vv, 2), brain_ml=round(float(B.sum()) * vv, 1))
    if geom_ok and M.any():
        h = ct[M]
        r.update(hu_median=round(float(np.median(h)), 1), hu_q1=round(float(np.percentile(h, 25)), 1), hu_q3=round(float(np.percentile(h, 75)), 1),
                 frac_40_100=round(float(((h >= 40) & (h <= 100)).mean()), 3), frac_0_30=round(float(((h >= 0) & (h < 30)).mean()), 3),
                 frac_lt0=round(float((h < 0).mean()), 3), frac_gt100=round(float((h > 100).mean()), 3),
                 frac_in_brain=round(float((M & B).sum() / M.sum()), 3), mask_over_brain=round(float(M.sum() / max(B.sum(), 1)), 3))
        lab, nc = ndimage.label(M, structure=ndimage.generate_binary_structure(3, 1)); r['components'] = int(nc)
        cs = np.bincount(lab.ravel())[1:]; r['largest_ml'] = round(float(cs.max()) * vv, 2)
        zz = np.where(M.any(axis=(0, 1)))[0]; r['z_extent_slices'] = int(zz.max() - zz.min() + 1)
        pb = ct[B & ~M]
        r['brain_parenchyma_hu_median'] = round(float(np.median(pb)), 1) if pb.size else None
    rows.append(r)
    print(s, r.get('mask_ml'), r.get('hu_median'), r.get('frac_40_100'), r.get('frac_in_brain'), r.get('mask_over_brain'), flush=True)
os.makedirs(outd, exist_ok=True)
keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0].keys()).index(k) if k in rows[0] else 99)
with open(os.path.join(outd, 'segcq500_mask_check.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
ref = [r for r in rows if r.get('analyzed') and r['scan'] not in ('CQ500-CT-108', 'CQ500-CT-187') and 'hu_median' in r]
def summ(k):
    v = np.array([r[k] for r in ref], float); return dict(median=float(np.median(v)), q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)), min=float(v.min()), max=float(v.max()))
out = dict(n_released=len(scans), n_reference=len(ref), focus={r['scan']: r for r in rows if r['scan'] in ('CQ500-CT-108', 'CQ500-CT-187')},
           reference_distribution={k: summ(k) for k in ['mask_ml', 'hu_median', 'frac_40_100', 'frac_0_30', 'frac_in_brain', 'mask_over_brain', 'components']},
           note='Reference distribution: the other analyzed Seg-CQ500 scans (excluding CT-108 and CT-187). Descriptive only; no adjudication.')
for s in ('CQ500-CT-108', 'CQ500-CT-187'):
    f = out['focus'].get(s, {})
    out['focus'][s]['rank_mask_ml_desc'] = int(1 + sum(r['mask_ml'] > f.get('mask_ml', 0) for r in ref)) if f else None
    out['focus'][s]['n_ref_with_lower_hu_median'] = int(sum(r['hu_median'] < f.get('hu_median', 0) for r in ref)) if 'hu_median' in f else None
json.dump(out, open(os.path.join(outd, 'segcq500_mask_check.json'), 'w'), indent=1)
print(json.dumps(out['focus'], indent=1)); print(json.dumps(out['reference_distribution'], indent=1))
