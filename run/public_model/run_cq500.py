"""Run the public RSNA-trained model on the 489 CQ500 scans of the locked evaluation (same series).
Resumable; stops after a time budget. Raw DICOM is deleted after each scan; only probabilities are kept."""
import sys, os, time, zipfile, shutil, json, csv, subprocess, threading, random, traceback
sys.path.insert(0, '/home/claude/work/ianpan'); sys.path.insert(0, '/home/claude/work/ianpan/run')
import numpy as np, torch
torch.set_num_threads(2)
from inference import HemorrhageSliceModel, read_dicom_series, CLASS_NAMES
from cq500_select import choose
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 6300.0
OUT = '/home/claude/work/ianpan/out'; TMP = '/home/claude/data/cq500_tmp'; os.makedirs(TMP, exist_ok=True)
S3 = 'https://s3.ap-south-1.amazonaws.com/qure.headct.study/'
rows = list(csv.DictReader(open('/mnt/user-data/uploads/NeuroCare_CT/evidence/cq500_external/cq500_per_scan.csv')))
expected = {r['name']: r['soft_series'] for r in rows}
names = sorted(expected); random.Random(20261029).shuffle(names)
todo = [n for n in names if not os.path.exists(f'{OUT}/{n}.json')]
print('todo', len(todo), flush=True)
def fetch(name):
    zp = f'{TMP}/{name}.zip'
    if os.path.exists(zp + '.ok'): return zp
    num = name.split('-')[-1]
    subprocess.run(['curl', '-s', '-f', '--retry', '5', '-o', zp, S3 + 'CQ500-CT-' + num + '.zip'], check=True)
    open(zp + '.ok', 'w').close(); return zp
pref = {}
def prefetch(name):
    try: pref[name] = fetch(name)
    except Exception as e: pref[name] = e
t_start = time.time()
model = HemorrhageSliceModel(local_dir='/home/claude/work/ianpan/weights', device='cpu')
th = None
for i, name in enumerate(todo):
    if time.time() - t_start > BUDGET: print('budget reached', flush=True); break
    if th is not None: th.join()
    nxt = todo[i + 1] if i + 1 < len(todo) else None
    if name not in pref: prefetch(name)
    th = threading.Thread(target=prefetch, args=(nxt,)) if nxt else None
    if th: th.start()
    t0 = time.time(); rec = {'name': name}
    d = f'{TMP}/{name}_dcm'
    try:
        zp = pref.pop(name)
        if isinstance(zp, Exception): raise zp
        z = zipfile.ZipFile(zp); sk, bk, allser, ok = choose(z)
        rec['soft_series'] = sk.rsplit('/', 1)[-1] if sk else None
        rec['series_matches_locked'] = rec['soft_series'] == expected[name]
        rec['n_files'] = len(ok[sk])
        import pydicom, io as _io
        uids = {}
        for k, v in allser.items():
            files = [n for n in z.namelist() if n.startswith(k + '/') and n.endswith('.dcm')]
            if files:
                h = pydicom.dcmread(_io.BytesIO(z.read(files[0])), stop_before_pixels=True)
                uids[k.rsplit('/', 1)[-1]] = dict(series_uid=str(getattr(h, 'SeriesInstanceUID', '')), study_uid=str(getattr(h, 'StudyInstanceUID', '')), n=v)
        rec['series_uids'] = uids
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        for f in ok[sk]:
            with open(os.path.join(d, f.rsplit('/', 1)[-1]), 'wb') as fh: fh.write(z.read(f))
        z.close()
        s = read_dicom_series(d, max_workers=2)
        v = np.ascontiguousarray(s['image'], dtype=np.float32)
        rec.update(n_slices=int(v.shape[0]), shape=list(v.shape), spacing=[float(x) for x in s['spacing']],
                   orientation=str(s['orientation_code']), p01=float(np.percentile(v, 1)))
        o = model.predict_hu_volume(v, batch_size=1, return_segmentation=False)
        for key in ['series_classification', 'base_series_classification', 'contextual_series_classification']:
            rec[key] = {c: float(p) for c, p in zip(CLASS_NAMES, o[key])}
        sl = o['slice_classification']
        rec['slice_max'] = {c: float(p) for c, p in zip(CLASS_NAMES, sl.max(0))}
        rec['slice_any'] = [round(float(p), 5) for p in sl[:, 5]]
        rec['sec'] = round(time.time() - t0, 1)
        json.dump(rec, open(f'{OUT}/{name}.json', 'w'))
        print(name, 'ok', rec['n_slices'], rec['sec'], round(rec['series_classification']['any'], 4), flush=True)
    except Exception as e:
        rec['error'] = repr(e)[:500]
        json.dump(rec, open(f'{OUT}/{name}.error.json', 'w'))
        print(name, 'ERR', repr(e)[:200], flush=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
        for ext in ['.zip', '.zip.ok']:
            p = f'{TMP}/{name}{ext}'
            if os.path.exists(p): os.remove(p)
if th is not None: th.join()
done = len([f for f in os.listdir(OUT) if f.endswith('.json') and not f.endswith('.error.json')])
print('DONE_COUNT', done, 'elapsed', round(time.time() - t_start), flush=True)
