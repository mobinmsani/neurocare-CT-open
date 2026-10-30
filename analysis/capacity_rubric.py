"""Validation-capacity rubric (post hoc, 28 Oct 2026).

Rates, for each endpoint, what the best OPENLY DOWNLOADABLE reference can support, independently of any model:
  C1 construct     the reference measures the endpoint at the required granularity (yes / partial / no)
  C2 independence  the reference images are institutionally independent of RSNA 2019, the source of BHSD and MR-BHSD
  C3 spectrum      reference-negative scans are present (detection, classification, voxel segmentation and total volume;
                   n/a for compartment and edema volume, which are measured in hemorrhage-positive patients)
  C4 positives     at least 35 reference-positive scans (or measured cases): the minimum for which a Wilson lower 95%
                   bound of 0.90 for sensitivity is attainable even with perfect detection (35/35 -> 0.901)
  C5 reliability   reader agreement reported and at least substantial (Fleiss kappa >= 0.61), or multiple independent raters
Capacity: 'none' if C1 = no; 'adequate' if C1 = yes and C2-C5 are met; otherwise 'limited'.
The evidence obtained in this study is rated separately (external / internal or transfer / exploratory / not evaluated).
Counts and kappa values are read from the analysis outputs; judgments are written out in the 'basis' fields.
Outputs: out/capacity_rubric.json, out/tables/capacity_rubric.csv
"""
import json
import pandas as pd
from common import *

REG = json.load(open(OUT / 'registry.json'))
K = {k: float(v['k4']) for k, v in REG['kappa'].items()}
P = REG['cq_prev']
BX = REG['bhx']
R3 = json.load(open(OUT / 'posthoc.json'))
BSC = json.load(open(OUT / 'bleedscene_cq500.json'))      # BleedScene3D vs CQ500 (154 CQ500 volumes, whose images are open)
BS3 = json.load(open(OUT / 'bleedscene_counts.json'))     # BleedScene3D subtype counts by source dataset
KAPPA_MIN, POS_MIN = 0.61, 35
assert abs(wilson(35, 35)[0] - 0.901) < 0.001 and wilson(34, 34)[0] < 0.90


def rel(lab, kmin=None):
    kmin = KAPPA_MIN if kmin is None else kmin
    labs = lab if isinstance(lab, (list, tuple)) else [lab]
    return 'yes' if min(K[l] for l in labs) >= kmin else 'no'


E = []  # (domain, endpoint, resources, C1, C2, C3, C4 (n or None), C5, basis, evidence_here, estimate)
E.append(('Detection, classification and localization', 'Any intracranial hemorrhage (scan level)', 'CQ500 labels; BHSD, MR-BHSD and Seg-CQ500 masks (positives only)',
          'yes', 'yes', 'yes', P['ICH'], 'ICH', f"CQ500: {P['ICH']} positives, 284 negatives, Fleiss kappa {K['ICH']:.3f}",
          'external', 'AUC 0.81 (v2); 0.82 (v3b)'))
sub_ok = [s for s in ['IPH', 'IVH', 'SAH', 'SDH', 'EDH'] if P[s] >= POS_MIN and K[s] >= KAPPA_MIN]
E.append(('Detection, classification and localization', 'Hemorrhage subtype presence (5 subtypes)', 'CQ500 labels; BHSD class masks (RSNA-derived); BHX boxes',
          'yes', 'yes', 'yes', min(P[s] for s in ['IPH', 'IVH', 'SAH', 'SDH', 'EDH']), 'no',
          'positives IPH %d, SAH %d, SDH %d, IVH %d, EDH %d; kappa IPH %.3f, SAH %.3f, SDH %.3f, IVH %.3f, EDH %.3f' % (
              P['IPH'], P['SAH'], P['SDH'], P['IVH'], P['EDH'], K['IPH'], K['SAH'], K['SDH'], K['IVH'], K['EDH']),
          'external', 'within-ICH AUC 0.72–0.91 (post hoc v3b)'))
E.append(('Detection, classification and localization', 'Hemorrhage localization (box level)', 'BHX boxes on CQ500',
          'partial', 'yes', 'n/a', min(int(v['scans']) for v in BX.values()), 'unknown',
          'boxes drawn on thick slices and extrapolated to thin series; box agreement not reported; boxes in >=%d CQ500 majority-negative scans and SAH boxes in >=%d scans without a majority SAH label; EDH boxes in %d scans' % (
              R3['bhx_ich_discordance_min'], [d for d in R3['bhx_subtype_discordance'] if d['cq500_label'] == 'SAH'][0]['min_scans_with_bhx_box_without_majority_label'], int(BX['Epidural']['scans'])),
          'external', 'any v2 voxel in 17–77% of boxes (binary model)'))
E.append(('Detection, classification and localization', 'Hemorrhage side', 'CQ500 side labels; PHE-SICH-CT-IDS documented side',
          'yes', 'yes', 'n/a', 134, ('BleedLocation-Left', 'BleedLocation-Right'), f"CQ500: 134 unilateral scans, kappa left {K['BleedLocation-Left']:.3f}, right {K['BleedLocation-Right']:.3f}; PHE-SICH-CT-IDS: 111 documented unilateral",
          'external', 'correct side 0.89 (CQ500); 0.97 (PHE-SICH-CT-IDS)'))
E.append(('Segmentation and volume', 'Voxel segmentation of hemorrhage', 'Seg-CQ500 (independent; positives only); BHSD and MR-BHSD (RSNA-derived)',
          'yes', 'yes', 'no', 43, 'unknown', 'Seg-CQ500: 43 usable masks, hemorrhage-positive by design (2 on majority-negative scans), one consensus mask per scan; multi-rater masks only in RSNA-derived MR-BHSD',
          'internal or transfer', 'Dice 0.46 (BHSD), 0.49 (Seg-CQ500)'))
E.append(('Segmentation and volume', 'Total hemorrhage volume', 'As voxel segmentation',
          'yes', 'yes', 'no', 43, 'unknown', 'as voxel segmentation; 36 BHSD test scans >=30 mL',
          'internal or transfer', 'ICC 0.86 (BHSD); 0.10 or 0.68 (Seg-CQ500)'))
bq = BS3['CQ500']; bnd = {k: BS3['INSTANCE2022'][k] + BS3['PhysioNet'][k] for k in ('IPH', 'EDH')}
E.append(('Segmentation and volume', 'Compartment volume (intraparenchymal, epidural)', 'BleedScene3D instance masks with subtype attributes (CQ500 volumes); BHSD class masks (RSNA-derived)',
          'yes', 'yes', 'n/a', min(bq['IPH'], bq['EDH']), 'unknown',
          f"BleedScene3D: IPH in {bq['IPH']} and EDH in {bq['EDH']} of the {bq['volumes']} CQ500 volumes whose images are open; annotation reliability not reported; released after the probe analyses; "
          f"its INSTANCE 2022 and CT-ICH volumes (IPH {bnd['IPH']}, EDH {bnd['EDH']}) have images that need an agreement. BHSD class masks are RSNA-derived: IPH in 63 test scans (14 in the 30-80 mL band), EDH in 13 (one >30 mL)",
          'internal or transfer', 'IPH ICC 0.87; kappa 0.62 at 30 mL (post hoc v3b)'))
E.append(('Segmentation and volume', 'Volume change on serial or postoperative scans', 'None',
          'no', 'n/a', 'n/a', None, 'n/a', 'no open serial or postoperative scans with voxel references (one BHSD patient contributed two training scans)',
          'not evaluated', 'no open reference identified'))
E.append(('Segmentation and volume', 'Perihematomal edema volume', 'PHE-SICH-CT-IDS edema masks (one hospital)',
          'yes', 'yes', 'n/a', 112, 'unknown', 'one expert mask per patient; agreement not reported; spontaneous basal-ganglia ICH only',
          'internal or transfer', 'Dice 0.34 (37 held-out patients)'))
E.append(('Mass effect and midline', 'Midline shift >5 mm (presence)', 'CQ500 labels; BleedScene3D shift relation (hemorrhage-positive volumes only)',
          'yes', 'yes', 'yes', P['MidlineShift'], 'MidlineShift', f"{P['MidlineShift']} positives; kappa {K['MidlineShift']:.3f}; 38% of positives decided 2–1",
          'exploratory', 'CSF-offset distance AUC 0.89 (ranking only)'))
E.append(('Mass effect and midline', 'Mass effect (presence)', 'CQ500 labels',
          'yes', 'yes', 'yes', P['MassEffect'], 'MassEffect', f"{P['MassEffect']} positives; kappa {K['MassEffect']:.3f}",
          'exploratory', 'AUC 0.81–0.85 (CSF-offset distance, volume)'))
E.append(('Mass effect and midline', 'Midline shift magnitude (mm)', 'None openly downloadable',
          'no', 'n/a', 'n/a', None, 'n/a', 'millimeter measurements for 509 RSNA scans and 92 CQ500 scans with shift (one neurosurgeon) are available only on request; BleedScene3D records midline shift only as a categorical relation, with no millimeter value',
          'not evaluated', 'no open reference identified'))
E.append(('Mass effect and midline', 'Subdural hematoma thickness (mm)', 'None',
          'no', 'n/a', 'n/a', None, 'n/a', 'SDH masks and labels exist, but no thickness measurements',
          'not evaluated', 'no open reference identified'))
E.append(('Mass effect and midline', 'Basal cisterns, ventricles, herniation', 'BleedScene3D ventricle-system masks and ventricle-asymmetry relation (CQ500 volumes)',
          'partial', 'yes', 'yes', BSC['relations_total']['asym'], 'unknown', f"ventricle-system masks and a binary hemorrhage-induced ventricle-asymmetry relation (present in {BSC['relations_total']['asym']}) in the {BSC['n']} CQ500 volumes whose images are open; the other BleedScene3D volumes are RSNA-derived or have images that need an account or agreement; no cistern or herniation grading; annotation reliability not reported; released 23 Oct 2026, after the probe analyses",
          'not evaluated', 'ventricle masks and asymmetry relation only (BleedScene3D)'))
E.append(('Fracture', 'Skull fracture (scan level)', 'CQ500 labels',
          'yes', 'yes', 'yes', P['Fracture'], 'Fracture', f"{P['Fracture']} positives (calvarial {P['CalvarialFracture']}); kappa {K['Fracture']:.3f} (calvarial {K['CalvarialFracture']:.3f})",
          'exploratory', 'pilot stopped (AUC 0.51)'))
E.append(('Fracture', 'Fracture localization', 'None openly downloadable',
          'no', 'n/a', 'n/a', None, 'n/a', 'CQ500 records fracture per scan only; slice-level fracture labels (PhysioNet CT-ICH, 22 fracture-positive patients) require a signed data use agreement; earlier CC BY 4.0 releases, no longer available from PhysioNet, persist in third-party mirrors and were not counted (counting them would rate this endpoint limited: slice level only, 22 positives)',
          'not evaluated', 'no open reference identified'))
E.append(('Outcome', 'Outcome or added prognostic value', 'None',
          'no', 'n/a', 'n/a', None, 'n/a', 'no open head CT resource linked to functional outcome identified',
          'not evaluated', 'no open reference identified'))

# Other candidate references, rated with the same criteria. 'open': openly downloadable but not probed; 'account': needs a
# free account plus click-through terms (not open by our definition; scenario); 'mirror-only': earlier releases no longer
# available from their original host that persist in third-party mirrors (not counted; scenario); 'agreement' and
# 'announced': listed for completeness, never counted. Fields: endpoint, resource, tier, C1-C5, note.
CANDS = [
    ('Voxel segmentation of hemorrhage', 'BleedScene3D instance masks (CQ500 volumes)', 'open', 'yes', 'yes', 'no', BS3['CQ500']['volumes'], 'unknown', 'hemorrhage-positive volumes only; one annotation per volume; released after the probe analyses'),
    ('Total hemorrhage volume', 'BleedScene3D instance masks (CQ500 volumes)', 'open', 'yes', 'yes', 'no', BS3['CQ500']['volumes'], 'unknown', 'hemorrhage-positive volumes only; released after the probe analyses'),
    ('Compartment volume (intraparenchymal, epidural)', 'BHSD class masks', 'open', 'yes', 'no', 'n/a', 13, 'unknown', 'RSNA 2019-derived; IPH in 63 and EDH in 13 test scans'),
    ('Voxel segmentation of hemorrhage', 'Río Hortega aneurysmal SAH masks', 'open', 'partial', 'yes', 'no', 100, 'unknown', 'SAH only; hemorrhage-positive scans only; one mask per scan'),
    ('Total hemorrhage volume', 'Río Hortega aneurysmal SAH masks', 'open', 'partial', 'yes', 'no', 100, 'unknown', 'SAH only; hemorrhage-positive scans only'),
    ('Voxel segmentation of hemorrhage', 'TEKNOFEST-2021 two-dimensional masks', 'open', 'partial', 'yes', 'unknown', None, 'unknown', 'two-dimensional slices, not volumes'),
    ('Skull fracture (scan level)', 'NeuroVision-CT image-level classes', 'open', 'partial', 'yes', 'yes', None, 'unknown', 'two-dimensional images, one physician'),
    ('Any intracranial hemorrhage (scan level)', 'RSNA 2019 labels', 'account', 'yes', 'no', 'yes', 1000, 'unknown', 'source of BHSD and MR-BHSD images'),
    ('Hemorrhage subtype presence (5 subtypes)', 'RSNA 2019 labels', 'account', 'yes', 'no', 'yes', 1000, 'unknown', 'source of BHSD and MR-BHSD images'),
    ('Voxel segmentation of hemorrhage', 'INSTANCE 2022 masks', 'agreement', 'yes', 'yes', 'no', 100, 'unknown', 'hemorrhage-positive volumes only; one consensus mask per volume'),
    ('Total hemorrhage volume', 'INSTANCE 2022 masks', 'agreement', 'yes', 'yes', 'no', 100, 'unknown', 'hemorrhage-positive volumes only'),
    ('Voxel segmentation of hemorrhage', 'RSNA ICH/IVH/PHE masks', 'announced', 'yes', 'no', 'no', 359, 'unknown', 'RSNA 2019-derived'),
    ('Compartment volume (intraparenchymal, epidural)', 'RSNA ICH/IVH/PHE masks', 'announced', 'partial', 'no', 'n/a', 359, 'unknown', 'RSNA 2019-derived; no epidural class'),
    ('Perihematomal edema volume', 'RSNA ICH/IVH/PHE masks', 'announced', 'yes', 'no', 'n/a', 359, 'unknown', 'RSNA 2019-derived'),
    ('Voxel segmentation of hemorrhage', 'HemSeg-200/500 masks', 'account', 'partial', 'no', 'no', 200, 'unknown', 'RSNA 2019 images; IPH and IVH only'),
    ('Fracture localization', 'CT-ICH earlier releases (versions 1.0.0, 1.2.0 and 1.3.0)', 'mirror-only', 'partial', 'yes', 'yes', 22, 'unknown', 'slice-level fracture labels in 22 patients'),
]
RANK = {'none': 0, 'limited': 1, 'adequate': 2}


def rate(c1, c2, c3, c4, c5v, pmin):
    """Capacity of one reference under the rubric (c5v already resolved to yes/no/unknown)."""
    if c1 == 'no':
        return 'none', ['no open reference']
    fails = []
    if c1 == 'partial': fails.append('construct (partial)')
    if c2 != 'yes': fails.append('not independent of RSNA 2019')
    if c3 in ('no', 'unknown'): fails.append('no negatives' if c3 == 'no' else 'negatives unknown')
    if c4 is None: fails.append('positives unknown')
    elif c4 < pmin: fails.append(f'<{pmin} positives')
    if c5v != 'yes': fails.append('reliability ' + ('unknown' if c5v == 'unknown' else 'below threshold'))
    return ('adequate' if not fails else 'limited'), fails


LABELS = set(K)


def c5_of(c5, kmin):
    if isinstance(c5, tuple) or c5 in LABELS:
        return rel(c5, kmin)
    return c5


def classify(kmin=KAPPA_MIN, pmin=POS_MIN, tiers=('open',)):
    """Apply the rubric with reliability threshold kmin and minimum positives pmin; candidate references of the given access
    tiers are rated with the same criteria and an endpoint takes the rating of its best reference."""
    ok = [s for s in ['IPH', 'IVH', 'SAH', 'SDH', 'EDH'] if P[s] >= pmin and K[s] >= kmin]
    out = []
    for dom, ep, res, c1, c2, c3, c4, c5, basis, ev, est in E:
        c5v = c5_of(c5, kmin)
        if ep.startswith('Hemorrhage subtype'):
            c5v = 'yes' if ok else 'no'
        if c1 == 'no':
            cap = 'none'; lim = 'no open reference'
        else:
            fails = []
            if c1 == 'partial': fails.append('construct (partial)')
            if c2 != 'yes': fails.append('not independent of RSNA 2019')
            if c3 == 'no': fails.append('no negatives')
            if c4 is not None and c4 < pmin: fails.append(f'<{pmin} positives')
            if c5v != 'yes': fails.append('reliability ' + ('unknown' if c5v == 'unknown' else f'kappa below {kmin:.2f}'))
            if ep.startswith('Hemorrhage subtype'):
                bad = [s for s in ['IPH', 'IVH', 'SAH', 'SDH', 'EDH'] if s not in ok]
                why = []
                for s in bad:
                    w = []
                    if P[s] < pmin: w.append(f'{P[s]} positives')
                    if K[s] < kmin: w.append(f'kappa {K[s]:.3f}')
                    why.append(f'{s} ' + ', '.join(w))
                fails = [f'adequate only for {" and ".join(ok) if ok else "none"}; ' + '; '.join(why)] if bad else []
            cap = 'adequate' if not fails else 'limited'; lim = '; '.join(fails) if fails else '\u2013'
        others = []
        for (cep, cres, tier, d1, d2, d3, d4, d5, note) in CANDS:
            if cep != ep or tier not in tiers:
                continue
            ccap, cf = rate(d1, d2, d3, d4, d5, pmin)
            others.append(f"{cres} ({tier}): {ccap}" + (f" ({'; '.join(cf)})" if cf else '') + f"; {note}")
            if RANK[ccap] > RANK[cap]:
                cap, lim, res = ccap, '; '.join(cf) if cf else '\u2013', f'{cres} ({tier})'
        out.append(dict(domain=dom, endpoint=ep, best_open_references=res, C1_construct=c1, C2_independent=c2, C3_negatives=c3,
                        C4_positives='' if c4 is None else c4, C5_reliability=c5v, capacity=cap, limiting=lim, basis=basis,
                        evidence_here=ev, estimate=est, other_references='; '.join(others)))
    return out, ok


rows, sub_ok = classify()
df = pd.DataFrame(rows)
df.to_csv(OUT / 'tables' / 'capacity_rubric.csv', index=False)
summary = dict(n=len(df), capacity=df.capacity.value_counts().to_dict(), evidence=df.evidence_here.value_counts().to_dict(),
               adequate=df[df.capacity == 'adequate'].endpoint.tolist(), none=df[df.capacity == 'none'].endpoint.tolist(),
               rules=dict(kappa_min=KAPPA_MIN, positives_min=POS_MIN), subtypes_adequate=sub_ok)
base = {r['endpoint']: r['capacity'] for r in rows}
sens = []
for kmin in (0.55, 0.60, 0.61, 0.65, 0.70, 0.75):
    for pmin in (25, 35, 50):
        rr, ok = classify(kmin, pmin)
        cnt = pd.Series([r['capacity'] for r in rr]).value_counts().to_dict()
        changed = [f"{r['endpoint']}: {base[r['endpoint']]} -> {r['capacity']}" for r in rr if r['capacity'] != base[r['endpoint']]]
        sens.append(dict(kappa_min=kmin, positives_min=pmin, adequate=cnt.get('adequate', 0), limited=cnt.get('limited', 0), none=cnt.get('none', 0),
                         subtypes_adequate=ok, changed=changed))
# access scenarios: resources that need only a free account and click-through terms, and releases withdrawn by their creators
for tiers, label in [(('open', 'account'), 'account-only resources counted'), (('open', 'mirror-only'), 'mirror-only releases counted')]:
    rr, ok = classify(KAPPA_MIN, POS_MIN, tiers)
    cnt = pd.Series([r['capacity'] for r in rr]).value_counts().to_dict()
    changed = [f"{r['endpoint']}: {base[r['endpoint']]} -> {r['capacity']} ({r['best_open_references']}: {r['limiting']})" for r in rr if r['capacity'] != base[r['endpoint']]]
    sens.append(dict(kappa_min=KAPPA_MIN, positives_min=POS_MIN, adequate=cnt.get('adequate', 0), limited=cnt.get('limited', 0), none=cnt.get('none', 0),
                     subtypes_adequate=ok, changed=changed, scenario=label))
pd.DataFrame([{**s, 'subtypes_adequate': ', '.join(s['subtypes_adequate']), 'changed': '; '.join(s['changed'])} for s in sens]).to_csv(
    OUT / 'tables' / 'capacity_rubric_sensitivity.csv', index=False)
summary['sensitivity'] = sens
save_json(dict(summary=summary, rows=rows), OUT / 'capacity_rubric.json')
print(json.dumps(summary, indent=1))
print(df[['endpoint', 'capacity', 'limiting', 'evidence_here']].to_string())
for s in sens:
    print(s['kappa_min'], s['positives_min'], s['adequate'], s['limited'], s['none'], s['subtypes_adequate'], s['changed'])
