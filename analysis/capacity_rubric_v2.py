"""Endpoint-specific validation-capacity rubric (revised, post hoc, 29 Oct 2026).

The rubric separates two questions that the first version combined.

1. Reference capacity: what the best openly downloadable reference can support for an endpoint, judged on properties of the
   reference alone and with criteria that depend on the endpoint type.
     R1 construct    the reference measures the endpoint at the required granularity and unit (yes / partial / no)
     R2 reliability  categorical labels: >=2 independent readers with reported agreement, Fleiss kappa >= 0.61 (Landis-Koch
                     'substantial'); for side, kappa among the scans that the reader majority called hemorrhage-positive, the
                     scans on which side is evaluated (side_agreement.py); voxel masks and continuous measurements: independent
                     masks or measurements from >=2 raters released, so that between-rater agreement can be estimated
     R3 spectrum     binary endpoints: >=35 reference-negative scans (precision of specificity); side: >=35 scans per side;
                     segmentation and total volume: >=35 reference-negative scans (false-positive volume); compartment volume
                     and millimeter measurements: >=35 reference cases on each side of every clinical decision threshold
                     (precision of classification at that threshold); edema volume: not applicable (no decision threshold)
     R4 precision    binary endpoints: >=35 reference-positive scans, the number for which the 95% confidence interval for an
                     anticipated sensitivity of 0.90 has a half-width of 0.10 (Buderer 1996) and a Wilson lower bound of 0.90
                     is attainable at 100% observed sensitivity; voxel segmentation, volumes and millimeter measurements:
                     >=100 measured reference-positive cases, for which each 95% Bland-Altman limit of agreement carries a
                     confidence interval of about +/-0.34 standard deviations of the differences (SE ~ sqrt(3 s^2 / n))
   Capacity: 'none' if R1 = no; 'adequate' if R1 = yes and R2-R4 are met; otherwise 'limited'. An endpoint takes the rating
   of its best reference; criteria met by different references are not combined.

2. Applicability to a model under test: whether a reference is external for that model depends on the model's development
   data, not on the reference. Every reference is tagged with its image source, and the rating is repeated with the
   references derived from a given training source removed (RSNA 2019; CQ500; both).

Outputs: out/capacity_rubric_v2.json, out/tables/capacity_rubric_v2.csv, out/tables/capacity_rubric_v2_sensitivity.csv,
out/tables/capacity_rubric_v2_applicability.csv, out/tables/capacity_rubric_v1_v2.csv
"""
import json
import re
import math
import pandas as pd
from common import *

REG = json.load(open(OUT / 'registry.json'))
K = {k: float(v['k4']) for k, v in REG['kappa'].items()}
_ka = pd.read_csv(OUT / 'tables' / 'cq500_reader_agreement_491.csv').set_index('label')
KLO = {k: float(_ka.loc[k, 'lo']) for k in K}   # lower 95% bootstrap bound of Fleiss' kappa (491 scans)
assert all(abs(float(_ka.loc[k, 'fleiss']) - K[k]) < 1e-3 for k in K)
SIDE = json.load(open(OUT / 'side_agreement.json'))['majority hemorrhage-positive scans']   # kappa among majority-positive scans
for _l in ('BleedLocation-Left', 'BleedLocation-Right'):
    K[_l + '|pos'], KLO[_l + '|pos'] = SIDE[_l]['k'], SIDE[_l]['lo']
MBH = json.load(open(OUT / 'mbhseg_train_raters.json'))   # MBH-Seg 2025 training release (masks from two to four raters per volume)
P = REG['cq_prev']
N_CQ = 489
NEG = {k: N_CQ - v for k, v in P.items()}
BX = REG['bhx']
BSC = json.load(open(OUT / 'bleedscene_cq500.json'))
BS3 = json.load(open(OUT / 'bleedscene_counts.json'))
V1 = json.load(open(OUT / 'capacity_rubric.json'))
PMJ = json.load(open(OUT / 'public_model_cq500.json')) if (OUT / 'public_model_cq500.json').exists() else None
KAPPA_MIN, BIN_MIN, CONT_MIN, THR_MIN = 0.61, 35, 100, 35
Z = 1.959963984540054
# the binary precision rule: smallest n with Wald half-width <= 0.10 at p = 0.90, and Wilson bound at 100%
assert math.ceil(Z ** 2 * 0.9 * 0.1 / 0.01) == 35 and abs(wilson(35, 35)[0] - 0.901) < 0.001 and wilson(34, 34)[0] < 0.90
# the continuous precision rule: half-width of the CI of a limit of agreement, in SD units, at n = 100
LOA_HW_100 = Z * math.sqrt(3 / 100)
assert abs(LOA_HW_100 - 0.339) < 0.001

SOURCES = {'CQ500': 'CQ500 images', 'RSNA2019': 'RSNA 2019 images', 'PHE': 'one hospital in China', 'CTICH': 'one hospital in Iraq',
           'INSTANCE': 'one hospital in Beijing', 'RIOHORTEGA': 'one hospital in Spain', 'TEKNOFEST': 'TEKNOFEST 2021, Türkiye',
           'NEUROVISION': 'one hospital in Bangladesh'}


def refname(r):
    """Reference name with its image source, without nested parentheses."""
    name, src = r['name'], SOURCES.get(r['source'], r['source'])
    m_ = re.match(r'^(.*) \(([^()]*)\)$', name)
    if m_ and r['source'] == 'CQ500' and 'CQ500 volumes' in m_.group(2):
        name = m_.group(1)                       # 'BleedScene3D ... (CQ500 volumes)': the source already says CQ500
    elif m_:
        name = f'{m_.group(1)}, {m_.group(2)}'   # 'MBH-Seg 2025 training masks, two to four raters'
    return f'{name} ({src})'

# ----------------------------------------------------------------------------- references
# Each reference: name, tier, source, R1 (construct), and the counts or flags the endpoint type needs.
#   kappa: label key(s) in K, or 'unknown'; raters: 'multi' | 'single' | 'unknown'; pos / neg: counts or None;
#   thresholds: list of (label, n_above, n_below) or 'unknown'; n_meas: measured reference-positive cases.
bq = BS3['CQ500']
EP = []  # (domain, endpoint, type, [references], evidence_here, estimate)


def ref(name, tier, source, r1, **kw):
    return dict(name=name, tier=tier, source=source, r1=r1, **kw)


EP.append(('Detection, classification and localization', 'Any intracranial hemorrhage (scan level)', 'binary', [
    ref('CQ500 reader labels', 'open', 'CQ500', 'yes', kappa='ICH', pos=P['ICH'], neg=NEG['ICH'],
        note=f"{P['ICH']} positives and {NEG['ICH']} negatives; predictive values estimable in the consecutive batch only (213 scans, 35 positives)"),
    ref('BHSD masks', 'open', 'RSNA2019', 'yes', kappa='unknown', pos=188, neg=0, note='hemorrhage-positive scans only'),
    ref('MR-BHSD masks (three raters)', 'open', 'RSNA2019', 'yes', kappa='unknown', pos=42, neg=1, note='one scan without hemorrhage in any rater mask'),
    ref('Seg-CQ500 masks', 'open', 'CQ500', 'yes', kappa='unknown', pos=43, neg=0, note='hemorrhage-positive by design'),
    ref('PHE-SICH-CT-IDS', 'open', 'PHE', 'yes', kappa='unknown', pos=112, neg=0, note='hemorrhage-positive patients only'),
    ref('RSNA 2019 labels', 'account', 'RSNA2019', 'yes', kappa='unknown', pos=1000, neg=1000, note='free account and click-through terms; source of BHSD and MR-BHSD images'),
], 'external', 'AUC 0.81 (v2), 0.82 (v3b)' + (f", {PMJ['auc_all']['auc']:.2f} (public model{'' if PMJ['n_scans'] == 489 else ', provisional'})" if PMJ else '')))

SUBTYPES = ['IPH', 'IVH', 'SAH', 'SDH', 'EDH']
EP.append(('Detection, classification and localization', 'Hemorrhage subtype presence (5 subtypes)', 'binary_subtype', [
    ref('CQ500 reader labels', 'open', 'CQ500', 'yes', kappa={s: s for s in SUBTYPES}, pos={s: P[s] for s in SUBTYPES}, neg={s: NEG[s] for s in SUBTYPES}),
    ref('BHSD class masks', 'open', 'RSNA2019', 'yes', kappa='unknown', pos={'IPH': 63, 'IVH': 54, 'SAH': 51, 'SDH': 35, 'EDH': 13},
        neg={'IPH': 33, 'IVH': 42, 'SAH': 45, 'SDH': 61, 'EDH': 83}, note='96 test scans counted; all scans hemorrhage-positive'),
    ref('MBH-Seg 2025 training class masks (two to four raters)', 'open', 'RSNA2019', 'yes', kappa='unknown',
        pos={s: MBH['class_present_majority'][s] for s in SUBTYPES}, neg={s: MBH['volumes'] - MBH['class_present_majority'][s] for s in SUBTYPES},
        note=f"{MBH['volumes']} hemorrhage-positive volumes; subtype present in at least half of the rater masks; agreement on subtype presence not reported"),
    ref('BleedScene3D subtype attributes (CQ500 volumes)', 'open', 'CQ500', r1='yes', host='kaggle', kappa='unknown',
        pos={s: bq[s] for s in SUBTYPES}, neg={s: bq['volumes'] - bq[s] for s in SUBTYPES}, note='154 hemorrhage-positive volumes; released after the probe analyses'),
], 'external', 'within-ICH AUC 0.72–0.91 (post hoc v3b)'))

EP.append(('Detection, classification and localization', 'Hemorrhage localization (box level)', 'binary', [
    ref('BHX boxes on CQ500', 'open', 'CQ500', 'partial', kappa='unknown', pos=REG['bhx_scans'], neg=N_CQ - REG['bhx_scans'],
        note='boxes drawn on thick slices and extrapolated to thin series; box agreement not reported'),
], 'exploratory', 'any v2 voxel in 17–77% of boxes (proxy; binary model)'))

EP.append(('Detection, classification and localization', 'Hemorrhage side', 'side', [
    ref('CQ500 reader labels', 'open', 'CQ500', 'yes', kappa=('BleedLocation-Left|pos', 'BleedLocation-Right|pos'), left=68, right=66,
        note=f"134 scans with unilateral hemorrhage by the reader majority; kappa among the {SIDE['n']} majority-positive scans "
             f"{SIDE['BleedLocation-Left']['k']:.3f} (left) and {SIDE['BleedLocation-Right']['k']:.3f} (right); "
             f"{K['BleedLocation-Left']:.3f} and {K['BleedLocation-Right']:.3f} across all 491 scans"),
    ref('PHE-SICH-CT-IDS documented side', 'open', 'PHE', 'yes', kappa='unknown', left=60, right=59,
        note='documented side in 119 patients with unilateral hemorrhage (60 left, 59 right); reliability of documentation not reported'),
], 'external', 'correct side 0.89 (CQ500); 0.97 (PHE-SICH-CT-IDS)'))

SEG_REFS = [
    ref('Seg-CQ500 masks', 'open', 'CQ500', 'yes', raters='single', pos=43, neg=0, note='43 of 51 usable; one consensus mask per scan'),
    ref('BleedScene3D instance masks (CQ500 volumes)', 'open', 'CQ500', r1='yes', host='kaggle', raters='single', pos=bq['volumes'], neg=0, note='hemorrhage-positive volumes; one annotation per volume'),
    ref('BHSD masks', 'open', 'RSNA2019', 'yes', raters='single', pos=188, neg=0, note='residents, reviewed by one radiologist; one mask per scan'),
    ref('MR-BHSD masks (three raters)', 'open', 'RSNA2019', 'yes', raters='multi', pos=42, neg=1, note='43 of 48 usable'),
    ref('MBH-Seg 2025 training masks (two to four raters)', 'open', 'RSNA2019', 'yes', raters='multi', pos=MBH['positive_any_rater'],
        neg=MBH['negative_all_raters'], note=f"{MBH['volumes']} volumes with {MBH['masks']} rater masks; hemorrhage-positive volumes only; "
                                              f"median pairwise Dice between raters {MBH['dice_median']:.2f}"),
    ref('Río Hortega aneurysmal SAH masks', 'open', 'RIOHORTEGA', 'partial', raters='single', pos=100, neg=0, note='SAH only'),
    ref('INSTANCE 2022 masks', 'agreement', 'INSTANCE', 'yes', raters='single', pos=100, neg=0, note='data use agreement'),
]
EP.append(('Segmentation and volume', 'Voxel segmentation of hemorrhage', 'segmentation', SEG_REFS, 'internal or transfer', 'Dice 0.46 (BHSD), 0.49 (Seg-CQ500)'))
EP.append(('Segmentation and volume', 'Total hemorrhage volume', 'segmentation', SEG_REFS, 'internal or transfer', 'ICC 0.86 (BHSD); 0.10 or 0.68 (Seg-CQ500)'))
EP.append(('Segmentation and volume', 'Compartment volume (intraparenchymal, epidural)', 'continuous_threshold', [
    ref('BleedScene3D instance masks with subtype attributes (CQ500 volumes)', 'open', 'CQ500', r1='yes', host='kaggle', raters='single', n_meas={'IPH': bq['IPH'], 'EDH': bq['EDH']},
        thresholds='unknown', note=f"IPH in {bq['IPH']}, EDH in {bq['EDH']} of {bq['volumes']} volumes; volumes by compartment not tallied"),
    ref('BHSD class masks', 'open', 'RSNA2019', 'yes', raters='single', n_meas={'IPH': 63, 'EDH': 13},
        thresholds=[('IPH >=30 mL (ICH score, MISTIE III)', 16, 47), ('IPH 30-80 mL (ENRICH)', 14, 49), ('EDH >30 mL (evacuation)', 1, 12)],
        note='test set: IPH in 63 scans, EDH in 13'),
    ref('MBH-Seg 2025 training class masks (two to four raters)', 'open', 'RSNA2019', 'yes', raters='multi',
        n_meas={'IPH': MBH['class_present_majority']['IPH'], 'EDH': MBH['class_present_majority']['EDH']},
        thresholds=[tuple(t) for t in MBH['thresholds']],
        note=f"IPH in {MBH['class_present_majority']['IPH']}, EDH in {MBH['class_present_majority']['EDH']} of {MBH['volumes']} volumes "
             '(present in at least half of the rater masks; volume = mean over raters)'),
], 'internal or transfer', 'IPH ICC 0.87; kappa 0.62 at 30 mL (post hoc v3b)'))
EP.append(('Segmentation and volume', 'Volume change on serial or postoperative scans', 'continuous_threshold', [], 'not evaluated', 'no open reference identified'))
EP.append(('Segmentation and volume', 'Perihematomal edema volume', 'continuous', [
    ref('PHE-SICH-CT-IDS edema masks', 'open', 'PHE', 'yes', raters='single', n_meas=112, note='one expert mask per patient; basal-ganglia ICH only'),
], 'internal or transfer', 'Dice 0.34 (37 held-out patients)'))
EP.append(('Mass effect and midline', 'Midline shift >5 mm (presence)', 'binary', [
    ref('CQ500 reader labels', 'open', 'CQ500', 'yes', kappa='MidlineShift', pos=P['MidlineShift'], neg=NEG['MidlineShift'],
        note=f"{P['MidlineShift']} positives; 38% of positives decided 2–1"),
    ref('BleedScene3D midline-shift relation (CQ500 volumes)', 'open', 'CQ500', r1='yes', host='kaggle', kappa='unknown', pos=BSC['relations_total']['shift'],
        neg=BSC['n'] - BSC['relations_total']['shift'], note='hemorrhage-positive volumes only'),
], 'exploratory', 'CSF-offset distance AUC 0.89 (ranking only)'))
EP.append(('Mass effect and midline', 'Mass effect (presence)', 'binary', [
    ref('CQ500 reader labels', 'open', 'CQ500', 'yes', kappa='MassEffect', pos=P['MassEffect'], neg=NEG['MassEffect']),
], 'exploratory', 'AUC 0.81–0.85 (CSF-offset distance, volume)'))
EP.append(('Mass effect and midline', 'Midline shift magnitude (mm)', 'continuous_threshold', [], 'not evaluated',
           'no open reference identified (millimeter values for 509 RSNA 2019 and 92 CQ500 scans available only on request)'))
EP.append(('Mass effect and midline', 'Subdural hematoma thickness (mm)', 'continuous_threshold', [], 'not evaluated', 'no open reference identified'))
EP.append(('Mass effect and midline', 'Basal cisterns, ventricles, herniation', 'binary', [
    ref('BleedScene3D ventricle masks and asymmetry relation (CQ500 volumes)', 'open', 'CQ500', r1='partial', host='kaggle', kappa='unknown', pos=BSC['relations_total']['asym'],
        neg=BSC['n'] - BSC['relations_total']['asym'], note='no cistern or herniation grading; released after the probe analyses'),
], 'not evaluated', 'ventricle masks and asymmetry relation only (BleedScene3D)'))
EP.append(('Fracture', 'Skull fracture (scan level)', 'binary', [
    ref('CQ500 reader labels', 'open', 'CQ500', 'yes', kappa='Fracture', pos=P['Fracture'], neg=NEG['Fracture'], note=f"calvarial {P['CalvarialFracture']}, kappa {K['CalvarialFracture']:.3f}"),
    ref('NeuroVision-CT image classes', 'open', 'NEUROVISION', 'partial', kappa='unknown', pos=None, neg=None, note='two-dimensional images, one physician'),
], 'exploratory', 'pilot stopped (AUC 0.51)'))
EP.append(('Fracture', 'Fracture localization', 'binary', [
    ref('CT-ICH earlier releases (third-party mirrors)', 'mirror-only', 'CTICH', 'partial', kappa='unknown', pos=22, neg=60, note='slice-level labels; current release requires a data use agreement'),
], 'not evaluated', 'no open reference identified'))
EP.append(('Outcome', 'Outcome or added prognostic value', 'binary', [], 'not evaluated', 'no open reference identified'))


# ----------------------------------------------------------------------------- rating
def rate(etype, r, kmin, bmin, cmin, tmin, sub=None, klow=False, pmin=None):
    """Rating of one reference for one endpoint type; returns (rating, list of unmet criteria).
    klow: compare the lower 95% bound of kappa, not the point estimate, with kmin; pmin: minimum positives (default bmin)."""
    pmin = bmin if pmin is None else pmin
    if r['r1'] == 'no':
        return 'none', ['construct']
    f = []
    if r['r1'] == 'partial':
        f.append('R1 construct partial')
    if etype in ('binary', 'binary_subtype', 'side'):
        kap = r['kappa'][sub] if isinstance(r['kappa'], dict) else r['kappa']
        if kap == 'unknown':
            f.append('R2 reader agreement not reported')
        else:
            KK = KLO if klow else K
            kv = min(KK[x] for x in kap) if isinstance(kap, tuple) else KK[kap]
            if kv < kmin:
                cond = ' among hemorrhage-positive scans' if (isinstance(kap, tuple) and all(x.endswith('|pos') for x in kap)) else ''
                f.append(f"R2 kappa{cond} {'lower bound ' if klow else ''}{kv:.3f} < {kmin:.2f}")
        if etype == 'side':
            if min(r['left'], r['right']) < max(bmin, pmin):
                f.append(f'R3/R4 <{max(bmin, pmin)} scans per side')
        else:
            pos = r['pos'][sub] if isinstance(r['pos'], dict) else r['pos']
            neg = r['neg'][sub] if isinstance(r['neg'], dict) else r['neg']
            if neg is None or neg < bmin:
                f.append('R3 no negatives' if neg == 0 else (f'R3 {neg} negatives' if neg is not None else 'R3 negatives unknown'))
            if pos is None or pos < pmin:
                f.append(f'R4 {pos} positives' if pos is not None else 'R4 positives unknown')
    elif etype == 'segmentation':
        if r['raters'] != 'multi':
            f.append('R2 one annotation per scan')
        if r['neg'] < bmin:
            f.append('R3 no negatives' if r['neg'] == 0 else f"R3 {r['neg']} negative")
        if r['pos'] < cmin:
            f.append(f"R4 {r['pos']} cases")
    elif etype in ('continuous', 'continuous_threshold'):
        if r['raters'] != 'multi':
            f.append('R2 one annotation per case')
        if etype == 'continuous_threshold':
            th = r.get('thresholds', 'unknown')
            if th == 'unknown':
                f.append('R3 coverage of decision thresholds unknown')
            else:
                bad = [t for t in th if min(t[1], t[2]) < tmin]
                if bad:
                    f.append('R3 ' + '; '.join(f'{t[0]}: {t[1]} cases' if t[1] <= t[2] else f'{t[0]}: {t[2]} cases outside' for t in bad))
        nm = r['n_meas'] if isinstance(r['n_meas'], dict) else {'': r['n_meas']}
        low = {k: v for k, v in nm.items() if v < cmin}
        if low:
            f.append('R4 ' + ', '.join(f'{k} {v}'.strip() for k, v in low.items()) + ' measured cases')
    return ('adequate' if not f else 'limited'), f


RANK = {'none': 0, 'limited': 1, 'adequate': 2}


def classify(kmin=KAPPA_MIN, bmin=BIN_MIN, cmin=CONT_MIN, tmin=THR_MIN, tiers=('open',), exclude=(), klow=False, pmin=None, exclude_hosts=()):
    out = []
    for dom, ep, etype, refs, ev, est in EP:
        cand = [r for r in refs if r['tier'] in tiers and r['source'] not in exclude and r.get('host') not in exclude_hosts]
        if etype == 'binary_subtype':
            per = {}
            for s in SUBTYPES:
                best = ('none', ['no reference'], None)
                for r in cand:
                    rt, fl = rate(etype, r, kmin, bmin, cmin, tmin, sub=s, klow=klow, pmin=pmin)
                    if RANK[rt] > RANK[best[0]] or best[2] is None:
                        best = (rt, fl, r['name'])
                per[s] = best
            ok = [s for s in SUBTYPES if per[s][0] == 'adequate']
            cap = 'adequate' if len(ok) == len(SUBTYPES) else ('limited' if cand else 'none')
            lim = '–' if cap == 'adequate' else (f"adequate only for {' and '.join(ok) if ok else 'no subtype'}; " +
                                                  '; '.join(f"{s}: {', '.join(per[s][1])}" for s in SUBTYPES if s not in ok))
            crl = [r for r in cand if r['name'] == 'CQ500 reader labels']
            bestname = refname(crl[0]) if crl else (refname(cand[0]) if cand else 'none')
            out.append(dict(domain=dom, endpoint=ep, type=etype, best_reference=bestname, capacity=cap, unmet=lim, subtypes_adequate=ok,
                            evidence_here=ev, estimate=est))
            continue
        best = None
        for r in cand:
            rt, fl = rate(etype, r, kmin, bmin, cmin, tmin, klow=klow, pmin=pmin)
            key = (RANK[rt], -len(fl))
            if best is None or key > best[0]:
                best = (key, rt, fl, r)
        if best is None:
            out.append(dict(domain=dom, endpoint=ep, type=etype, best_reference='none', capacity='none', unmet='R1 no reference measures the endpoint',
                            subtypes_adequate=[], evidence_here=ev, estimate=est))
        else:
            _, rt, fl, r = best
            out.append(dict(domain=dom, endpoint=ep, type=etype, best_reference=refname(r), capacity=rt,
                            unmet='–' if not fl else '; '.join(fl), subtypes_adequate=[], evidence_here=ev, estimate=est))
    return out


def counts(rows):
    c = pd.Series([r['capacity'] for r in rows]).value_counts().to_dict()
    return {k: int(c.get(k, 0)) for k in ('adequate', 'limited', 'none')}


base = classify()
df = pd.DataFrame(base)
df.to_csv(OUT / 'tables' / 'capacity_rubric_v2.csv', index=False)
B = {r['endpoint']: r for r in base}

# criteria sensitivity
sens = []
grid = ([dict(kmin=k) for k in (0.55, 0.60, 0.65, 0.70, 0.75)] + [dict(klow=True)] + [dict(bmin=b) for b in (25, 50)] + [dict(pmin=140)]
        + [dict(cmin=c) for c in (50, 200)] + [dict(tmin=t) for t in (25, 50)])
LAB = {'klow=True': 'Lower 95% bound of kappa >=0.61', 'pmin=140': 'Positives >=140 (Wilson lower bound 0.90 at an observed sensitivity of 0.95)'}
for g in grid:
    rr = classify(**g)
    ch = [f"{r['endpoint']}: {B[r['endpoint']]['capacity']} -> {r['capacity']}" for r in rr if r['capacity'] != B[r['endpoint']]['capacity']]
    sub = [r for r in rr if r['type'] == 'binary_subtype'][0]['subtypes_adequate']
    key = ', '.join(f'{k}={v}' for k, v in g.items())
    sens.append(dict(scenario=LAB.get(key, key), **counts(rr), subtypes_adequate=', '.join(sub), changed='; '.join(ch) or '–'))
for kw, lab in [(dict(tiers=('open', 'account')), 'Resources needing a free account counted'), (dict(tiers=('open', 'mirror-only')), 'Releases persisting only in mirrors counted'),
                (dict(exclude=('RSNA2019',)), 'Releases that re-host RSNA 2019 images not counted'), (dict(exclude_hosts=('kaggle',)), 'Releases hosted only on Kaggle not counted')]:
    rr = classify(**kw)
    ch = [f"{r['endpoint']}: {B[r['endpoint']]['capacity']} -> {r['capacity']} (" + (f"{r['best_reference']}; " if r['best_reference'] != 'none' else '') + f"{r['unmet']})"
          for r in rr if r['capacity'] != B[r['endpoint']]['capacity']]
    sub = [r for r in rr if r['type'] == 'binary_subtype'][0]['subtypes_adequate']
    sens.append(dict(scenario=lab, **counts(rr), subtypes_adequate=', '.join(sub), changed='; '.join(ch) or '–'))
pd.DataFrame(sens).to_csv(OUT / 'tables' / 'capacity_rubric_v2_sensitivity.csv', index=False)

# applicability by training source of the model under test
app = []
for exc, lab in [((), 'Model developed without these public sources'), (('RSNA2019',), 'Model trained on RSNA 2019 data'),
                 (('CQ500',), 'Model trained or tuned on CQ500'), (('RSNA2019', 'CQ500'), 'Model trained on RSNA 2019 and CQ500')]:
    rr = classify(exclude=exc)
    ch = [f"{r['endpoint']}: {B[r['endpoint']]['capacity']} -> {r['capacity']}" for r in rr if r['capacity'] != B[r['endpoint']]['capacity']]
    lost = [f"{r['endpoint']}: {r['best_reference']}" for r in rr if r['best_reference'] != B[r['endpoint']]['best_reference'] and r['capacity'] == B[r['endpoint']]['capacity']]
    sub = [r for r in rr if r['type'] == 'binary_subtype'][0]['subtypes_adequate']
    app.append(dict(scenario=lab, excluded_sources=', '.join(exc) or '–', **counts(rr), subtypes_adequate=', '.join(sub),
                    rating_changes='; '.join(ch) or '–', reference_changes='; '.join(lost) or '–',
                    adequate_endpoints=', '.join(r['endpoint'] for r in rr if r['capacity'] == 'adequate')))
pd.DataFrame(app).to_csv(OUT / 'tables' / 'capacity_rubric_v2_applicability.csv', index=False)

# comparison with the first version
v1 = {r['endpoint']: r for r in V1['rows']}
cmp = [dict(endpoint=r['endpoint'], v1_capacity=v1[r['endpoint']]['capacity'], v1_limiting=v1[r['endpoint']]['limiting'],
            v2_capacity=r['capacity'], v2_unmet=r['unmet'], changed=v1[r['endpoint']]['capacity'] != r['capacity']) for r in base]
pd.DataFrame(cmp).to_csv(OUT / 'tables' / 'capacity_rubric_v1_v2.csv', index=False)

summary = dict(n=len(base), capacity=counts(base), adequate=[r['endpoint'] for r in base if r['capacity'] == 'adequate'],
               none=[r['endpoint'] for r in base if r['capacity'] == 'none'], limited=[r['endpoint'] for r in base if r['capacity'] == 'limited'],
               subtypes_adequate=[r for r in base if r['type'] == 'binary_subtype'][0]['subtypes_adequate'],
               rules=dict(kappa_min=KAPPA_MIN, binary_min=BIN_MIN, continuous_min=CONT_MIN, threshold_min=THR_MIN, loa_ci_halfwidth_sd_at_100=round(LOA_HW_100, 3)),
               changed_from_v1=[c['endpoint'] for c in cmp if c['changed']], sensitivity=sens, applicability=app)
save_json(dict(summary=summary, rows=base, comparison=cmp), OUT / 'capacity_rubric_v2.json')
print(json.dumps({k: v for k, v in summary.items() if k not in ('sensitivity', 'applicability')}, indent=1))
print(df[['endpoint', 'type', 'capacity', 'unmet']].to_string())
for s in sens:
    print(s)
for a in app:
    print(a)
