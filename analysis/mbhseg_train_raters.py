"""Post hoc, model-independent: the MBH-Seg 2025 training release (voxel-level part) as a candidate reference for the capacity
rubric (29 Oct 2026).

The release (Hugging Face mbhseg/mbhseg25, main branch, MBH_Train_2025_voxel-label.zip; downloaded without an account on
29 Oct 2026, SHA-256 a51012a7...c6b8) holds 192 RSNA 2019-derived volumes, each with one image and masks from two to four of
four annotators (label_annot_1 ... label_annot_4). This script reads the archive without extracting it, computes the class
volumes of every rater mask (labels 1-5 = EDH, IPH, IVH, SAH, SDH, as in BHSD, checked against the BHSD test-set reference
volumes), and summarizes what the rubric needs: hemorrhage-negative volumes, measured positive volumes, coverage of the
compartment-volume decision thresholds, and between-rater agreement (pairwise Dice of the hemorrhage masks and pairwise ICC(A,1)
of total volume).

Usage: python mbhseg_train_raters.py <MBH_Train_2025_voxel-label.zip>
Outputs: out/mbhseg_train_raters.json, out/tables/mbhseg_train_raters.csv (class volumes per volume and rater)
"""
import gzip
import io
import itertools
import json
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, icc_a1, save_json  # noqa: E402

ZIP = Path(sys.argv[1])
CLS = {1: 'EDH', 2: 'IPH', 3: 'IVH', 4: 'SAH', 5: 'SDH'}
z = zipfile.ZipFile(ZIP)
vols = {}
for n in z.namelist():
    if not n.endswith('/'):
        d, b = n.rsplit('/', 1)
        vols.setdefault(d, []).append(b)
assert len(vols) == 192


def load(name):
    with z.open(name) as f:
        raw = gzip.decompress(f.read())
    return nib.Nifti1Image.from_bytes(raw)


def header(name):
    with z.open(name) as f, gzip.GzipFile(fileobj=f) as g:
        return nib.Nifti1Header.from_fileobj(io.BytesIO(g.read(352)))


rows, dice, geo_bad = [], [], []
for case in sorted(vols):
    raters = sorted(b for b in vols[case] if b.startswith('label_annot_'))
    hi = header(f'{case}/image.nii.gz')
    masks = {}
    for b in raters:
        img = load(f'{case}/{b}')
        if img.shape != tuple(hi.get_data_shape()) or not np.allclose(img.affine, hi.get_best_affine(), atol=1e-3):
            geo_bad.append((case, b))
        a = np.asarray(img.dataobj).astype(np.int16)
        vox = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
        r = b.replace('label_annot_', 'annot_').replace('.nii.gz', '')
        rec = dict(case_id=case, rater=r, n_raters=len(raters))
        for k, s in CLS.items():
            rec[f'{s}_ml'] = float((a == k).sum()) * vox
        rec['other_labels'] = int(((a > 0) & ~np.isin(a, list(CLS))).sum())
        rec['total_ml'] = float((a > 0).sum()) * vox
        rows.append(rec)
        masks[r] = a > 0
    for r1, r2 in itertools.combinations(sorted(masks), 2):
        s = masks[r1].sum() + masks[r2].sum()
        dice.append(dict(case_id=case, pair=f'{r1}-{r2}', dice=(2 * (masks[r1] & masks[r2]).sum() / s) if s else np.nan,
                         both_empty=bool(s == 0)))
df = pd.DataFrame(rows)
df.to_csv(OUT / 'tables' / 'mbhseg_train_raters.csv', index=False)
dd = pd.DataFrame(dice)

# label mapping check against the BHSD test-set reference volumes (same RSNA-derived scans)
ROOT = Path(__file__).resolve().parents[1]
ref = pd.read_csv(ROOT / 'data/evidence/neurocare_neuroimaging_advanced_20261021/reference_subtype_volumes.csv')
refw = ref.pivot(index='case_id', columns='subtype', values='reference_only_volume_ml')
common_ids = sorted(set(refw.index) & set(df.case_id))
match = {}
for r in sorted(df.rater.unique()):
    sub = df[df.rater == r].set_index('case_id')
    ids = [i for i in common_ids if i in sub.index]
    err = np.mean([max(abs(sub.loc[i, f'{s}_ml'] - refw.loc[i, s]) for s in CLS.values()) for i in ids]) if ids else np.nan
    match[r] = dict(n=len(ids), mean_max_abs_diff_ml=float(err))
best = min(match, key=lambda r: match[r]['mean_max_abs_diff_ml'])
_b = df[df.rater == best].set_index('case_id').loc[common_ids]
class_r = {s: float(np.corrcoef(refw.loc[common_ids, s], _b[f'{s}_ml'])[0, 1]) for s in CLS.values()}   # diagonal: label mapping
offdiag_max = max(float(np.corrcoef(refw.loc[common_ids, s], _b[f'{t}_ml'])[0, 1]) for s in CLS.values() for t in CLS.values() if s != t)

# per volume: hemorrhage in how many rater masks
g = df.groupby('case_id')
anyh = g.total_ml.apply(lambda v: int((v > 0).sum()))
nr = g.n_raters.first()
neg_all = int((anyh == 0).sum())
pos_any = int((anyh > 0).sum())
pos_all = int((anyh == nr).sum())
mean_cls = g[[f'{s}_ml' for s in CLS.values()] + ['total_ml']].mean()
pres = g[[f'{s}_ml' for s in CLS.values()]].apply(lambda x: (x > 0).sum() >= (len(x) / 2))   # present in at least half of the masks
cls_present = {s: int(pres[f'{s}_ml'].sum()) for s in CLS.values()}
iph = mean_cls.loc[pres['IPH_ml'], 'IPH_ml']
edh = mean_cls.loc[pres['EDH_ml'], 'EDH_ml']
thr = [('IPH >=30 mL (ICH score, MISTIE III)', int((iph >= 30).sum()), int((iph < 30).sum())),
       ('IPH 30-80 mL (ENRICH)', int(((iph >= 30) & (iph <= 80)).sum()), int(((iph < 30) | (iph > 80)).sum())),
       ('EDH >30 mL (evacuation)', int((edh > 30).sum()), int((edh <= 30).sum()))]

# between-rater agreement of total volume, rater pairs with at least 30 shared volumes
W = df.pivot(index='case_id', columns='rater', values='total_ml')
WI = df.pivot(index='case_id', columns='rater', values='IPH_ml')
pair_icc = {}
for a, b in itertools.combinations(W.columns, 2):
    ok = W[a].notna() & W[b].notna()
    if ok.sum() >= 30:
        d = W.loc[ok, a] - W.loc[ok, b]
        lo, hi = float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))
        c1, c2 = (WI.loc[ok, a] >= 30).values, (WI.loc[ok, b] >= 30).values   # IPH volume at the 30 mL threshold
        pair_icc[f'{a}-{b}'] = dict(n=int(ok.sum()), icc=float(icc_a1(W.loc[ok, a], W.loc[ok, b])), bias=float(d.mean()), loa=[lo, hi],
                                    loa_width=hi - lo, iph30_both=int((c1 & c2).sum()), iph30_discordant=int((c1 != c2).sum()),
                                    iph_present=int(((WI.loc[ok, a] > 0) | (WI.loc[ok, b] > 0)).sum()))
vd = dd[~dd.both_empty]
case_dice = vd.groupby('case_id').dice.mean()
S = dict(release='MBH-Seg 2025 training release, voxel-level part', volumes=len(vols), masks=len(df),
         raters_per_volume={int(k): int(v) for k, v in nr.value_counts().sort_index().items()},
         masks_per_rater={k: int(v) for k, v in df.rater.value_counts().sort_index().items()},
         geometry_mismatches=len(geo_bad), other_label_voxels=int(df.other_labels.sum()),
         label_check=dict(bhsd_test_ids_shared=len(common_ids), per_rater=match, closest_rater=best, class_r=class_r, offdiag_r_max=offdiag_max),
         negative_all_raters=neg_all, positive_any_rater=pos_any, positive_all_raters=pos_all,
         class_present_majority=cls_present, thresholds=thr,
         dice_pairs=int(len(vd)), dice_median=float(vd.dice.median()), dice_iqr=[float(vd.dice.quantile(.25)), float(vd.dice.quantile(.75))],
         case_mean_dice_median=float(case_dice.median()), pair_icc=pair_icc,
         loa_width_range=[min(v['loa_width'] for v in pair_icc.values()), max(v['loa_width'] for v in pair_icc.values())],
         icc_range=[min(v['icc'] for v in pair_icc.values()), max(v['icc'] for v in pair_icc.values())],
         iph30_discordant_max=max(v['iph30_discordant'] for v in pair_icc.values()),
         bhsd_scans_shared=int(len(set(df.case_id) & set(pd.read_csv(ROOT / 'data/tables/QC_features_all_188.csv').case_id))))
save_json(S, OUT / 'mbhseg_train_raters.json')
print(json.dumps(S, indent=1))
