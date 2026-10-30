"""Counts of BleedScene3D volumes with bleeding instances of each subtype, by source dataset (28 Oct 2026).

Used for the capacity rubric (voxel segmentation, total and compartment volume). Model-independent; BleedScene3D was released
after the probe analyses. Usage: python bleedscene_counts.py <BleedScene3D raw folder containing <dataset>/psgg/*.json>
Output: out/bleedscene_counts.json
"""
import json, sys
from pathlib import Path
from common import OUT, save_json

RAW = Path(sys.argv[1])
ATTR = {1: 'SAH', 2: 'SAH', 3: 'IPH', 4: 'IVH', 5: 'EDH', 6: 'SDH'}
INDEPENDENT = {'CQ500', 'INSTANCE2022', 'PhysioNet'}   # not derived from RSNA 2019 (BHSD and HemSeg200 are)
R = {}
for ds in sorted(p.name for p in RAW.iterdir() if (p / 'psgg').is_dir()):
    files = sorted((RAW / ds / 'psgg').glob('*.json'))
    cnt = dict(volumes=len(files), any_instance=0, IPH=0, IVH=0, EDH=0, SDH=0, SAH=0)
    for f in files:
        d = json.load(open(f))
        bleeds = [b for b in d['bounding_boxes'] if b['class_id'] == 3]
        if bleeds:
            cnt['any_instance'] += 1
        present = set()
        for b in bleeds:
            for a in b['attributes']:
                if a['value'] and a['id'] in ATTR:
                    present.add(ATTR[a['id']])
        for s in present:
            cnt[s] += 1
    cnt['independent_of_RSNA2019'] = ds in INDEPENDENT
    R[ds] = cnt
ind = {k: sum(R[d][k] for d in R if R[d]['independent_of_RSNA2019']) for k in ['volumes', 'any_instance', 'IPH', 'IVH', 'EDH', 'SDH', 'SAH']}
R['independent_total'] = ind
save_json(R, OUT / 'bleedscene_counts.json')
print(json.dumps(R, indent=1))
