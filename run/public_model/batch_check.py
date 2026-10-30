"""Batch-size check for the public-model replication. The first 10 scans were run with batches of two slices, as the locked plan
specified; the remaining 479 with batches of one, after the run had been stopped for lack of memory. This script re-runs scans
from the first group with batches of one and compares every stored output (series-level and slice-level probabilities).
Usage: python batch_check.py <json_dir> <result.json> scan [scan ...]"""
import sys, os, json, zipfile, shutil, subprocess
sys.path.insert(0, os.environ.get('PM_HOME', '/home/claude/work/ianpan')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(2)
from inference import HemorrhageSliceModel, read_dicom_series, CLASS_NAMES
from cq500_select import choose
S3 = 'https://s3.ap-south-1.amazonaws.com/qure.headct.study/'
jd, dst, names = sys.argv[1], sys.argv[2], sys.argv[3:]
TMP = os.environ.get('PM_TMP', '/home/claude/data/cq500_tmp') + '/batch_check'
os.makedirs(TMP, exist_ok=True)
model = HemorrhageSliceModel(local_dir=os.environ.get('PM_WEIGHTS', '/home/claude/work/ianpan/weights'), device='cpu')
res = []
for name in names:
    ref = json.load(open(os.path.join(jd, name + '.json')))
    zp = os.path.join(TMP, name + '.zip')
    subprocess.run(['curl', '-s', '-f', '--retry', '5', '-o', zp, S3 + 'CQ500-CT-' + name.split('-')[-1] + '.zip'], check=True)
    d = os.path.join(TMP, name + '_dcm'); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    z = zipfile.ZipFile(zp); sk, bk, allser, ok = choose(z)
    assert sk.rsplit('/', 1)[-1] == ref['soft_series']
    for f in ok[sk]:
        with open(os.path.join(d, f.rsplit('/', 1)[-1]), 'wb') as fh:
            fh.write(z.read(f))
    z.close()
    s = read_dicom_series(d, max_workers=2)
    v = np.ascontiguousarray(s['image'], dtype=np.float32)
    o = model.predict_hu_volume(v, batch_size=1, return_segmentation=False)
    diffs = {}
    for key in ['series_classification', 'base_series_classification', 'contextual_series_classification']:
        diffs[key] = max(abs(float(p) - ref[key][c]) for c, p in zip(CLASS_NAMES, o[key]))
    sl = o['slice_classification']
    diffs['slice_max'] = max(abs(float(p) - ref['slice_max'][c]) for c, p in zip(CLASS_NAMES, sl.max(0)))
    diffs['slice_any'] = float(np.max(np.abs(np.round(sl[:, 5], 5) - np.array(ref['slice_any']))))
    res.append(dict(scan=name, n_slices=int(v.shape[0]), any_stored=ref['series_classification']['any'], any_batch1=float(o['series_classification'][5]), **diffs))
    print(res[-1], flush=True)
    shutil.rmtree(d, ignore_errors=True); os.remove(zp)
out = dict(scans=res, max_abs_diff_series=max(max(r['series_classification'], r['base_series_classification'], r['contextual_series_classification']) for r in res),
           max_abs_diff_slice=max(max(r['slice_max'], r['slice_any']) for r in res),
           note='stored outputs computed with batches of two slices; re-run with batches of one; slice_any stored rounded to 5 decimals')
json.dump(out, open(dst, 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'scans'}))
