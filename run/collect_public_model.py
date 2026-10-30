"""Collect per-scan outputs of the public model (one JSON per scan) into one CSV, and the DICOM study/series UIDs.
Usage: python collect_public_model.py <json_dir> <out_dir>"""
import sys, json, glob, os, csv
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
rows, uids, errs = [], [], []
for f in sorted(glob.glob(os.path.join(src, 'CQ500-CT-*.json'))):
    d = json.load(open(f))
    if f.endswith('.error.json') or 'error' in d:
        errs.append(d); continue
    r = dict(scan=d['name'], soft_series=d['soft_series'], series_matches_locked=d['series_matches_locked'], n_slices=d['n_slices'],
             slice_mm=round(d['spacing'][0], 3), pixel_mm=round(d['spacing'][1], 4), p01_HU=d['p01'], sec=d['sec'])
    for k, pre in [('series_classification', 'p_'), ('base_series_classification', 'base_'), ('contextual_series_classification', 'ctx_'), ('slice_max', 'slicemax_')]:
        for c, v in d[k].items():
            r[pre + c] = round(v, 6)
    rows.append(r)
    su, how = d.get('series_uids'), 'model run'
    if not su and os.path.exists(os.path.join(src, 'uids', d['name'] + '.json')):   # 14 early scans: UIDs read later by fetch_uids_range.py
        su, how = json.load(open(os.path.join(src, 'uids', d['name'] + '.json')))['series_uids'], 'range read'
    for ser, u in (su or {}).items():
        uids.append(dict(scan=d['name'], series_description=ser, series_uid=u['series_uid'], study_uid=u['study_uid'], n_files=u['n'],
                         uid_source=how))
with open(os.path.join(dst, 'public_model_per_scan.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if uids:
    with open(os.path.join(dst, 'cq500_series_uids.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(uids[0])); w.writeheader(); w.writerows(uids)
json.dump(errs, open(os.path.join(dst, 'public_model_errors.json'), 'w'), indent=1)
print(len(rows), 'scans;', len({u['scan'] for u in uids}), 'with UIDs;', len(errs), 'errors')
