"""Integrity and discordance ledger (post hoc, 28 Oct 2026).

Collects, from the stored analysis outputs, every fault in a released image-reference pair, every disagreement between two
open references for the same scan, and every scoring convention that changes a summary statistic. Each entry records
whether it depends on the probe models (none do: all are properties of the released data).
Kinds:
  fault        the image and reference cannot be used as released (geometry or affine discordance, unreadable archive,
               missing series, intensities not HU-encoded)
  discordance  two open references disagree about the same scan (BHX: exact counts after linkage by DICOM study UID,
               bhx_exact.py; lower bounds from the numbers of positive scans in the first version)
  convention   a scoring rule that determines how a case enters a statistic
Outputs: out/ledger.json, out/tables/integrity_discordance_ledger.csv
"""
import json
import pandas as pd
from common import *

REG = json.load(open(OUT / 'registry.json'))
R3 = json.load(open(OUT / 'posthoc.json'))
X3 = json.load(open(OUT / 'extra_checks3.json'))
S = json.load(open(OUT / 'new_analyses_seg.json'))
disc = {d['cq500_label']: d for d in R3['bhx_subtype_discordance']}
mr = X3['mrbhsd']

L = []


def add(kind, resource, item, n, den, handling, effect):
    L.append(dict(kind=kind, resource=resource, item=item, n=n, denominator=den, handling=handling, effect=effect,
                  model_dependent='no'))


# ---------------------------------------------------------------- faults (released pairs not usable as released)
add('fault', 'BHSD', 'image-mask geometry discordant (training split)', 4, 192, 'excluded', 'training set 92 of 96 scans')
add('fault', 'MR-BHSD', 'no geometry-concordant mask from all three raters', 5, 48, 'excluded', '43 scans analyzed')
add('fault', 'Seg-CQ500', 'image-mask geometry or integrity check failed', 8, 51, 'excluded', '43 scans analyzed')
add('fault', 'PHE-SICH-CT-IDS', 'mask affine reverses an in-plane axis relative to the image', 8, 120, 'excluded without repair', '112 patients analyzed')
add('fault', 'CQ500', 'unreadable archive (1) or no non-contrast soft-tissue series (1)', 2, 491, 'excluded', '489 scans analyzed')
add('fault', 'BHSD', 'stored intensities not HU-encoded (test split)', int(REG['qc']['test_flag']), int(REG['qc']['test_n']), 'analyzed; QC gate abstains',
    'v1 volume ICC 0.054 with the scan, 0.910 without it')
# ---------------------------------------------------------------- discordances (two references, one scan)
add('discordance', 'Seg-CQ500 vs CQ500', 'hemorrhage mask on a scan that the CQ500 reader majority labeled negative', 2, 43,
    'analyzed; reported with and without CT-108',
    f"CT-108: reference {S['seg_ct108']['ref']:.1f} mL, one of three readers positive, v2 {S['seg_ct108']['pred']:.1f} mL; "
    "v2 volume ICC 0.104 with CT-108, 0.677 without; BLAST-CT 0.028 and 0.302")
BXE = json.load(open(OUT / 'bhx_exact.json')) if (OUT / 'bhx_exact.json').exists() else None   # exact linkage by study UID (29 Oct 2026)
if BXE:
    a = BXE['ICH']
    add('discordance', 'BHX vs CQ500', 'scans with BHX boxes that the CQ500 reader majority labeled negative', a['bhx_only'], a['bhx_scans'],
        'reported; not adjudicated', f"{a['bhx_only_no_read']} of them without any positive read; {a['majority_only']} majority-positive scans without boxes; "
        'box-level localization measured against boxes that the CQ500 majority does not support')
    for lab, name, lab2 in [('SAH', 'subarachnoid', 'SAH'), ('SDH', 'subdural', 'SDH'), ('IVH', 'intraventricular', 'IVH'), ('ChronicBleed', 'chronic', 'chronic bleed')]:
        d = BXE[lab]
        add('discordance', 'BHX vs CQ500', f"scans with {name} boxes without the CQ500 majority {lab2} label", d['bhx_only'], d['bhx_scans'],
            'reported; not adjudicated', f"{d['bhx_scans']} scans with boxes vs {d['majority_pos']} majority-positive scans; {d['bhx_only_no_read']} of the "
            f"{d['bhx_only']} without any positive {lab2} read; {d['majority_only']} majority-positive scans without boxes")
else:
    add('discordance', 'BHX vs CQ500', 'scans with BHX boxes beyond the number of CQ500 majority-positive scans (lower bound)',
        R3['bhx_ich_discordance_min'], R3['bhx_scans_matched'], 'reported; not adjudicated', 'box-level localization measured against boxes that the CQ500 majority does not support')
    for lab, name, lab2 in [('SAH', 'subarachnoid', 'SAH'), ('SDH', 'subdural', 'SDH'), ('IVH', 'intraventricular', 'IVH'), ('ChronicBleed', 'chronic subdural hematoma', 'chronic bleed')]:
        d = disc[lab]
        add('discordance', 'BHX vs CQ500', f"scans with {name} boxes minus scans with the CQ500 majority {lab2} label (lower bound)",
            d['min_scans_with_bhx_box_without_majority_label'], d['bhx_scans_with_boxes'], 'reported; not adjudicated',
            f"{d['bhx_scans_with_boxes']} scans with boxes vs {d['cq500_majority_positive_scans']} majority-positive scans")
BSC = json.load(open(OUT / 'bleedscene_cq500.json'))   # post hoc: BleedScene3D (released 23 Oct 2026) versus the CQ500 reads
add('discordance', 'BleedScene3D vs CQ500', 'volumes curated as hemorrhage-positive that the CQ500 reader majority labeled negative', BSC['n'] - BSC['majority_ich'], BSC['n'],
    'reported; not adjudicated (post hoc)', 'BleedScene3D entered only the capacity rubric and this comparison')
for row in BSC['agreement']:
    if row['label'] in ('SAH', 'SDH', 'IVH'):
        # denominator: BleedScene3D-positive scans for the subtype, as the BHX rows use scans with boxes of the subtype
        add('discordance', 'BleedScene3D vs CQ500', f"scans with a BleedScene3D {row['label']} flag without the CQ500 majority {row['label']} label", row['bleedscene_only'], row['bleedscene_pos'],
            'reported; not adjudicated (post hoc)', f"{row['bleedscene_pos']} BleedScene3D-positive vs {row['cq500_majority_pos']} majority-positive scans among the {BSC['n']} shared; Cohen kappa {row['kappa']:.2f}")
# ---------------------------------------------------------------- conventions
add('convention', 'MR-BHSD', 'no rater marked hemorrhage (1 scan); one or two raters left the mask empty (2 scans; in one, the three raters marked 0, 5.4 and 52.4 mL)', 3, 43,
    'Dice set to 1 when both masks are empty, as in the original analysis; results without these comparisons also given',
    f"v1 Dice {mr['v1_dice_all']:.3f} with, {mr['v1_dice_excl_all_empty']:.3f} without the all-empty scan; rater-rater {mr['rr_dice_all']:.3f} and {mr['rr_dice_excl_joint_empty']:.3f} without {mr['joint_empty_pairs']} jointly empty pairs")
add('convention', 'CQ500', 'follow-up scans of the same patient may be included (no patient identifier released)', None, 491,
    'not excluded; scan is the unit of analysis', 'confidence intervals may be too narrow if scans cluster within patients')
add('convention', 'BHX', 'boxes drawn on thick slices and extrapolated to thin series; a box counts as localized if it contains one predicted voxel', None, None,
    'lenient criterion; intervals ignore clustering of boxes within scans', 'hit rates are upper bounds of localization accuracy')

df = pd.DataFrame(L)
df.to_csv(OUT / 'tables' / 'integrity_discordance_ledger.csv', index=False)
summ = dict(faults=int((df.kind == 'fault').sum()), discordances=int((df.kind == 'discordance').sum()), conventions=int((df.kind == 'convention').sum()),
            unusable_pairs=int(REG['integrity_all']), released_pairs=int(REG['integrity_all_den']), geometry_faults=int(REG['integrity_geom']),
            geometry_den=int(REG['integrity_geom_den']))
save_json(dict(summary=summ, rows=L), OUT / 'ledger.json')
print(json.dumps(summ, indent=1))
print(df[['kind', 'resource', 'n', 'denominator', 'item']].to_string())
