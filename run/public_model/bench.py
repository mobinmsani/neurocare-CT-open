import sys, os, time, zipfile, shutil, json
sys.path.insert(0, '/home/claude/work/ianpan'); sys.path.insert(0, '/home/claude/work/ianpan/run')
import numpy as np, torch
torch.set_num_threads(2)
from inference import HemorrhageSliceModel, read_dicom_series
from cq500_select import choose
t0 = time.time()
model = HemorrhageSliceModel(local_dir='/home/claude/work/ianpan/weights', device='cpu')
print('load', round(time.time()-t0, 1), flush=True)
zp = sys.argv[1]; tmp = '/home/claude/data/cq500_tmp/bench'
shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
z = zipfile.ZipFile(zp); sk, bk, allser, ok = choose(z)
print('series', sk, len(ok[sk]), allser, flush=True)
for f in ok[sk]:
    with open(os.path.join(tmp, f.rsplit('/', 1)[-1]), 'wb') as fh: fh.write(z.read(f))
s = read_dicom_series(tmp)
v = s['image']; print('vol', v.shape, v.dtype, float(np.percentile(v, 1)), float(v.max()), s['spacing'], s['orientation_code'], flush=True)
t1 = time.time()
out = model.predict_hu_volume(v, batch_size=2, return_segmentation=False)
dt = time.time()-t1
print('infer', round(dt, 1), 's', round(dt/len(v), 2), 's/slice', flush=True)
print('series', np.round(out['series_classification'], 4).tolist())
print('base', np.round(out['base_series_classification'], 4).tolist())
print('slicemax', np.round(out['slice_classification'].max(0), 4).tolist())
