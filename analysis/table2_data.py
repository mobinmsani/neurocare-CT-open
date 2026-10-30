"""Assemble Table 2 (segmentation and volumetric agreement) from per-scan files.
Locked Dice CIs (2000 resamples, original seeds) are used where they exist; all other CIs use 10,000 resamples (seed 20261032)."""
import json
import numpy as np, pandas as pd
from common import *

rng = np.random.default_rng(20261032)
VO = json.load(open(OUT / 'verify_original.json'))
REG = json.load(open(OUT / 'registry.json'))

def boot_mean(x, B=10000):
    x = np.asarray(x, float); n = len(x)
    bs = x[rng.integers(0, n, (B, n))].mean(1)
    return np.percentile(bs, [2.5, 97.5])

def boot_icc(p, y, B=10000):
    p = np.asarray(p, float); y = np.asarray(y, float); n = len(p)
    out = []
    for _ in range(B):
        i = rng.integers(0, n, n); out.append(icc_a1(p[i], y[i]))
    return np.nanpercentile(out, [2.5, 97.5])

def row(label, dice, pred, ref, locked_ci=None, note='', icc_ci=None, dice_ci=None):
    dice = np.asarray(dice, float); pred = np.asarray(pred, float); ref = np.asarray(ref, float)
    a = agree(pred, ref)
    ci = locked_ci if locked_ci is not None else (dice_ci if dice_ci is not None else boot_mean(dice))
    q1, q2, q3 = np.percentile(dice, [25, 50, 75])
    icc_ci = icc_ci if icc_ci is not None else boot_icc(pred, ref)
    return dict(label=label, n=int(len(dice)), dice=float(dice.mean()), dice_lo=float(ci[0]), dice_hi=float(ci[1]),
                med=float(q2), q1=float(q1), q3=float(q3), zero=int((dice == 0).sum()), mae=float(a['MAE']),
                icc=float(a['ICC']), icc_lo=float(icc_ci[0]), icc_hi=float(icc_ci[1]), bias=float(a['bias']),
                loa_lo=float(a['LoA_lo']), loa_hi=float(a['LoA_hi']), ci_source='locked' if locked_ci is not None else 're-analysis', note=note)

def parse_ci(s):
    # '0.436 (0.375–0.493)' -> (lo, hi)
    inner = s.split('(')[1].rstrip(')')
    lo, hi = inner.replace('−', '-').split('–')
    return float(lo), float(hi)

rows = []
import ast
BA = pd.read_csv(OUT / 'tables/bhsd_agreement_full.csv')
SA = pd.read_csv(OUT / 'tables/segcq500_agreement_full.csv')
X = json.load(open(OUT / 'extra_checks.json'))
def bci(model, cohort):
    r = BA[(BA.model == model) & (BA.cohort == cohort)].iloc[0]; return ast.literal_eval(r.ICC_ci)
def bdice(model, cohort):
    r = BA[(BA.model == model) & (BA.cohort == cohort)].iloc[0]; return (r.dice_lo, r.dice_hi)
def sci(model):
    r = SA[SA.model == model].iloc[0]; return ast.literal_eval(r.ICC_ci)
# ---------------- BHSD test
v1 = pd.read_csv(E / 'neurocare_neuroimaging_advanced_20261021/per_scan_quantitative_metrics.csv').set_index('case_id')
v2 = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/v2_metrics.csv'); v2b = v2[v2.dataset == 'BHSD_test'].set_index('case_id')
nm = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/ablation_no_postmask.csv').set_index('case_id')
v3b = pd.read_csv(E / 'v3_subtypes/v3b/v3b_bhsd_test_metrics.csv').set_index('case_id')
k = sorted(v1.index); qcid = 'ID_a429fd9a_ID_2eabc43f3d'; g = [c for c in k if c != qcid]
ref = v1.loc[k, 'reference_volume_ml']
rows.append(('BHSD', row('v1 (prespecified)', v1.loc[k, 'dice'], v1.loc[k, 'prediction_volume_ml'], ref, parse_ci(REG['bh_v1_dice']), icc_ci=bci('v1','all 96'))))
rows.append(('BHSD', row('v1, QC-gated', v1.loc[g, 'dice'], v1.loc[g, 'prediction_volume_ml'], v1.loc[g, 'reference_volume_ml'], parse_ci(REG['bh_v1q_dice']), icc_ci=bci('v1','QC-accepted 95'))))
rows.append(('BHSD', row('v2-PM (prespecified)', v2b.loc[k, 'dice'], v2b.loc[k, 'pred_ml'], v2b.loc[k, 'ref_ml'], parse_ci(REG['bh_v2p_dice']), icc_ci=bci('v2 prespecified','all 96'))))
rows.append(('BHSD', row('v2 (post hoc)', nm.loc[k, 'dice_nopostmask'], nm.loc[k, 'pred_ml'], nm.loc[k, 'ref_ml'], parse_ci(REG['bh_v2_dice']), icc_ci=bci('v2 no post-mask','all 96'))))
rows.append(('BHSD', row('v2, QC-gated', nm.loc[g, 'dice_nopostmask'], nm.loc[g, 'pred_ml'], nm.loc[g, 'ref_ml'], parse_ci(REG['bh_v2q_dice']), icc_ci=bci('v2 no post-mask','QC-accepted 95'))))
rows.append(('BHSD', row('v3b, any class (post hoc)', v3b.loc[k, 'ANY_dice'], v3b.loc[k, 'ANY_pred_ml'], v3b.loc[k, 'ANY_ref_ml'], parse_ci(REG['bh_v3b_any']), icc_ci=bci('v3b (any)','all 96'))))
rows.append(('BHSD', row('v3b, QC-gated', v3b.loc[g, 'ANY_dice'], v3b.loc[g, 'ANY_pred_ml'], v3b.loc[g, 'ANY_ref_ml'], icc_ci=bci('v3b (any)','QC-accepted 95'), dice_ci=bdice('v3b (any)','QC-accepted 95'))))
# ---------------- Seg-CQ500
seg = pd.read_csv(T / 'SegCQ500_four_method_comparison.csv')
segnm = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/segcq500_no_postmask.csv').set_index('case_id')
v2s = v2[v2.dataset == 'Seg-CQ500'].set_index('case_id')
def segm(method):
    d = seg[seg.method == method].set_index('case_id').sort_index()
    return d.dice, d.prediction_volume_ml.fillna(d.pred_volume_ml), d.reference_volume_ml
ks = sorted(seg.case_id.unique()); kx = [c for c in ks if c != 'CQ500-CT-108']
bl = segm('blast_model1'); l1 = segm('lightunet25d')
refs = bl[2]
# Seg-CQ500 and MR-BHSD Dice intervals come from the re-analysis (10,000 resamples)
rows.append(('Seg', row('BLAST-CT (pretrained)', bl[0], bl[1], bl[2], icc_ci=sci('BLAST-CT'), dice_ci=parse_ci(REG['seg_blast_dice']))))
rows.append(('Seg', row('v1', l1[0], l1[1], l1[2], icc_ci=sci('v1'), dice_ci=parse_ci(REG['seg_v1_dice']))))
rows.append(('Seg', row('v2-PM', v2s.loc[ks, 'dice'], v2s.loc[ks, 'pred_ml'], v2s.loc[ks, 'ref_ml'], icc_ci=sci('v2 prespecified'), dice_ci=parse_ci(REG['seg_v2p_dice']))))
print(segnm.columns.tolist())
pcol = [c for c in segnm.columns if 'pred' in c][0]; rcol = [c for c in segnm.columns if 'ref' in c][0]; dcol = [c for c in segnm.columns if 'dice' in c][0]
rows.append(('Seg', row('v2', segnm.loc[ks, dcol], segnm.loc[ks, pcol], segnm.loc[ks, rcol], icc_ci=sci('v2 no post-mask'), dice_ci=parse_ci(REG['seg_v2_dice']))))
rows.append(('Seg', row('BLAST-CT, without CT-108', bl[0].loc[kx], bl[1].loc[kx], bl[2].loc[kx], icc_ci=X['seg_blast_excl108_icc_ci'])))
rows.append(('Seg', row('v2, without CT-108', segnm.loc[kx, dcol], segnm.loc[kx, pcol], segnm.loc[kx, rcol], icc_ci=X['seg_v2nm_excl108_icc_ci'], dice_ci=X['seg_v2nm_excl108_dice'][1])))
# ---------------- MR-BHSD
mr = pd.read_csv(T / 'MRBHSD_full43_compact_model_rater_metrics.csv')
md = mr.groupby('case_id').dice.mean().sort_index()
rv = mr.pivot(index='case_id', columns='reference_rater', values='reference_volume_ml').sort_index()
pv = mr.groupby('case_id').pred_volume_ml.first().sort_index()
r_mr = row('v1 vs each rater (mean)', md, pv, rv.mean(axis=1), icc_ci=parse_ci(REG['mr_model_mean_icc']), dice_ci=parse_ci(REG['mr_v1_dice']))
rows.append(('MR', r_mr))
# rater vs rater: pairwise Dice per scan not in this file; use registry values
json.dump([dict(cohort=c, **r) for c, r in rows], open(OUT / 'table2_rows.json', 'w'), indent=1)
for c, r in rows:
    print(c, r['label'], r['n'], f"{r['dice']:.3f} ({r['dice_lo']:.3f}-{r['dice_hi']:.3f})", f"{r['med']:.3f} ({r['q1']:.3f}-{r['q3']:.3f})", r['zero'], f"MAE {r['mae']:.2f}", f"ICC {r['icc']:.3f} ({r['icc_lo']:.3f}-{r['icc_hi']:.3f})", f"bias {r['bias']:.2f} ({r['loa_lo']:.1f} to {r['loa_hi']:.1f})", r['ci_source'])
