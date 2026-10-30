"""Collect the DICOM study and series UIDs of every CQ500 scan (read by fetch_uids_range.py) into one table, and check them
against the UIDs that the public-model run read from the fully downloaded archives.
Usage: python collect_uids.py <uids_dir> <public_model_uid_csv> <out_csv>"""
import sys, os, json, glob, csv
src, pmcsv, dst = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for f in sorted(glob.glob(os.path.join(src, 'CQ500-CT-*.json'))):
    d = json.load(open(f))
    for ser, u in d['series_uids'].items():
        rows.append(dict(scan=d['name'], series_description=ser, series_uid=u['series_uid'], study_uid=u['study_uid'], n_files=u['n'],
                         locked_soft_series=int(ser == d['soft_series'])))
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
key = lambda r: (r['scan'], r['series_description'])
a = {key(r): (r['series_uid'], r['study_uid'], int(r['n_files'])) for r in rows}
b = {key(r): (r['series_uid'], r['study_uid'], int(r['n_files'])) for r in csv.DictReader(open(pmcsv))
     if r.get('uid_source', 'model run') == 'model run'}        # UIDs read by the model run from the full archive (not the 14 range-filled scans)
common = set(a) & set(b)
mism = [k for k in common if a[k] != b[k]]
scans = {r['scan'] for r in rows}
studies = {}
for r in rows:
    studies.setdefault(r['study_uid'], set()).add(r['scan'])
print(len(scans), 'scans;', len(rows), 'series;', len(studies), 'study UIDs;', sum(len(v) > 1 for v in studies.values()), 'study UIDs shared by >1 scan;',
      len(common), 'series also read by the model run;', len(mism), 'mismatches')
assert not mism
json.dump(dict(scans=len(scans), series=len(rows), study_uids=len(studies), study_uids_shared=sum(len(v) > 1 for v in studies.values()),
               crosschecked_series=len(common), crosschecked_scans=len({k[0] for k in common}), mismatches=len(mism)), open(os.path.join(os.path.dirname(dst), 'cq500_uids_summary.json'), 'w'), indent=1)
