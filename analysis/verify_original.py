"""Step 1: exact re-computation of every number reported by the original pipeline from the per-scan
files, using the original random seeds and call order where bootstrap intervals were used.
Writes out/verify_original.json and prints a comparison table."""
import json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from common import *

R = {}
chk = []  # (item, reported, recomputed)


def C(item, reported, recomputed):
    chk.append((item, str(reported), str(recomputed)))


# ------------------------------------------------------------------ BHSD (v2 report replica)
rng = np.random.default_rng(20261021)


def boot(x, f=np.mean, n=2000):
    x = np.asarray(x); b = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def bootpair(a, b, n=2000):
    a, b = np.asarray(a), np.asarray(b)
    i = [rng.integers(0, len(a), len(a)) for _ in range(n)]
    return np.percentile([(b[j] - a[j]).mean() for j in i], [2.5, 97.5])


v1 = pd.read_csv(E / 'neurocare_neuroimaging_advanced_20261021/per_scan_quantitative_metrics.csv').set_index('case_id')
v2 = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/v2_metrics.csv')
v2b = v2[v2.dataset == 'BHSD_test'].set_index('case_id')
v2s = v2[v2.dataset == 'Seg-CQ500'].set_index('case_id')
nm = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/ablation_no_postmask.csv').set_index('case_id')
qc = set(json.load(open(E / 'neurocare_qc_gate_20261021/QC_gate_results.json'))['flagged_ids'])
k = sorted(v1.index)
y = v1.loc[k, 'reference_volume_ml'].values
# reference volumes identical across files?
R['ref_vol_consistency_v2'] = float(np.abs(v2b.loc[k, 'ref_ml'].values - y).max())
R['ref_vol_consistency_nm'] = float(np.abs(nm.loc[k, 'ref_ml'].values - y).max())
M = {'v1': (v1.loc[k, 'dice'].values, v1.loc[k, 'prediction_volume_ml'].values),
     'v2_prespecified': (v2b.loc[k, 'dice'].values, v2b.loc[k, 'pred_ml'].values),
     'v2_nomask_posthoc': (nm.loc[k, 'dice_nopostmask'].values, nm.loc[k, 'pred_ml'].values)}
acc = np.array([c not in qc for c in k])
rows = []
for name, (d, p) in M.items():
    for sub, sel in [('all 96', np.ones(len(k), bool)), ('QC-accepted 95', acc)]:
        a = agree(p[sel], y[sel]); lo, hi = boot(d[sel])
        rows.append(dict(model=name, cohort=sub, n=int(sel.sum()), mean_dice=d[sel].mean(), lo=lo, hi=hi,
                         median=np.median(d[sel]), zero=int((d[sel] == 0).sum()), **a))
R['bhsd_table3'] = rows
pair = []
for name in ['v2_prespecified', 'v2_nomask_posthoc']:
    a, b = M['v1'][0], M[name][0]; lo, hi = bootpair(a, b); w = stats.wilcoxon(b, a)
    pair.append(dict(cmp=f'{name}-v1 BHSD', mean=(b - a).mean(), lo=lo, hi=hi, p=w.pvalue, improved=int((b > a).sum()), n=96))
seg4 = pd.read_csv(T / 'SegCQ500_four_method_comparison.csv')
s1 = seg4[seg4.method == 'lightunet25d'].set_index('case_id')['dice']
s3 = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/segcq500_no_postmask.csv').set_index('case_id')
ks = sorted(s1.index)
A = s1.loc[ks].values
for name, S in [('v2_prespecified', v2s['dice']), ('v2_nomask', s3['dice_nopostmask'])]:
    B = S.loc[ks].values; lo, hi = bootpair(A, B); w = stats.wilcoxon(B, A)
    pair.append(dict(cmp=f'{name}-v1 SegCQ500', mean=(B - A).mean(), lo=lo, hi=hi, p=w.pvalue, improved=int((B > A).sum()), n=43))
R['pairs_table4'] = pair
R['seg_means'] = dict(v1=A.mean(), v2=v2s['dice'].mean(), v2_nomask=s3['dice_nopostmask'].mean())
# edema (same rng stream continues: original script called boot(d) for edema after laterality; laterality used no rng)
ed = pd.read_csv(E / 'neurocare_v2_20261021/edema/edema_test_metrics.csv')
lo, hi = boot(ed.dice.values)
R['edema'] = dict(n=len(ed), dice=ed.dice.mean(), lo=lo, hi=hi, median=ed.dice.median(), **agree(ed.pred_ml, ed.ref_ml))

# size strata / compartments
sv = pd.read_csv(E / 'neurocare_neuroimaging_advanced_20261021/reference_subtype_volumes.csv')
pres = {s: set(sv[(sv.subtype == s) & (sv.reference_present)].case_id) for s in SUB}
strata = []
d1, d2 = M['v1'][0], M['v2_nomask_posthoc'][0]
for lab, sel in [('<10', y < 10), ('10-<30', (y >= 10) & (y < 30)), ('>=30', y >= 30)] + [(f'{s} present', np.array([c in pres[s] for c in k])) for s in SUB]:
    strata.append(dict(stratum=lab, n=int(sel.sum()), v1=d1[sel].mean(), v2=d2[sel].mean(), z1=int((d1[sel] == 0).sum()), z2=int((d2[sel] == 0).sum())))
R['strata_table5'] = strata

# v1 BHSD descriptive extras
R['v1_extra'] = dict(median_dice=np.median(d1), mean_iou=v1.loc[k, 'iou'].mean(), hd95=v1.loc[k, 'hd95_mm'].mean(),
                     asd=v1.loc[k, 'average_surface_distance_mm'].mean(), n_surface=int(v1.loc[k, 'hd95_mm'].notna().sum()),
                     empty_pred=int((v1.loc[k, 'prediction_volume_ml'] == 0).sum()),
                     outlier=v1.loc['ID_a429fd9a_ID_2eabc43f3d', ['prediction_volume_ml', 'reference_volume_ml']].tolist())

# ------------------------------------------------------------------ QC gate
qf = pd.read_csv(E / 'neurocare_qc_gate_20261021/QC_features_all_188.csv')
qs = pd.read_csv(E / 'neurocare_qc_gate_20261021/QC_features_SegCQ500.csv')
R['qc'] = dict(train_n=int((qf.partition == 'train').sum()), train_flag=int((qf[qf.partition == 'train'].p01 > -500).sum()),
               test_n=int((qf.partition == 'test').sum()), test_flag=int((qf[qf.partition == 'test'].p01 > -500).sum()),
               test_flag_ids=qf[(qf.partition == 'test') & (qf.p01 > -500)].case_id.tolist(),
               seg_n=len(qs), seg_flag=int((qs.p01 > -500).sum()))

# ------------------------------------------------------------------ Seg-CQ500 method table (Table 2)
s_all = pd.read_csv(T / 'SegCQ500_four_method_comparison.csv')
t2 = []
for mth, g in s_all.groupby('method'):
    t2.append(dict(method=mth, n=len(g), mean=g.dice.mean(), median=g.dice.median(), iou=g.iou.mean(),
                   prec=g.voxel_precision.mean(), rec=g.voxel_recall.mean()))
R['segcq500_table2'] = t2
mr = pd.read_csv(T / 'MRBHSD_full43_compact_model_rater_metrics.csv')
mr_avg = mr.groupby('case_id')[['dice', 'iou', 'voxel_precision', 'voxel_recall']].mean()
R['mrbhsd_v1'] = dict(n=len(mr_avg), mean=mr_avg.dice.mean(), median=mr_avg.dice.median(), iou=mr_avg.iou.mean(),
                      per_rater=mr.groupby('reference_rater').dice.mean().to_dict())
rr = pd.read_csv(T / 'TableS4_Reference_reference_agreement.csv')
R['mrbhsd_rater_rater_mean'] = rr.mean_dice.mean()

# ------------------------------------------------------------------ PHE laterality (edema + v1 recomputable)
lat = pd.read_csv(E / 'neurocare_v2_20261021/phe_sich/phe_laterality_midline.csv', dtype={'case': str})
R['phe_overview'] = dict(n=len(lat), sides=lat.documented_side.value_counts().to_dict(), status=lat.midline_status.value_counts().to_dict(),
                         angle_median=lat.midline_angle_deg.median(), angle_iqr=np.percentile(lat.midline_angle_deg, [25, 75]).tolist(),
                         at_bound=int((lat.midline_angle_deg.abs() >= 20).sum()), sym_median=lat.mean_symmetry_dice.median(),
                         p01_flag=int((lat.p01 > -500).sum()))
R['phe_cols_note'] = list(lat.columns)

# ------------------------------------------------------------------ CQ500 (replica of research_cq500_analyze.py)
m = load_cq500()
rng = np.random.default_rng(20261022)


def auc_ci(yv, s, n=2000):
    yv = np.asarray(yv); s = np.asarray(s, float); ok = ~np.isnan(s); yv, s = yv[ok], s[ok]
    a = roc_auc_score(yv, s); b = []
    for _ in range(n):
        i = rng.integers(0, len(yv), len(yv))
        if yv[i].min() != yv[i].max():
            b.append(roc_auc_score(yv[i], s[i]))
    return (a, *np.percentile(b, [2.5, 97.5]), len(yv), int(yv.sum()))


# majority labels recomputed from reads
for l in LAB:
    mj = (m[[f'R{i}:{l}' for i in (1, 2, 3)]].sum(1) >= 2).astype(int)
    assert (mj.values == m['ref_' + l].values).all(), l
vol = m.pred_volume_ml.values
det = []
for l in ['ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH', 'ChronicBleed']:
    det.append((l, *auc_ci(m['ref_' + l].values, vol)))
R['cq_det_table9'] = det
yI = m.ref_ICH.values; pI = vol >= 1.0
tp = int((pI & (yI == 1)).sum()); fn = int((~pI & (yI == 1)).sum()); tn = int((~pI & (yI == 0)).sum()); fp = int((pI & (yI == 0)).sum())
R['cq_rule'] = dict(tp=tp, fn=fn, tn=tn, fp=fp, sens=tp / (tp + fn), sens_ci=wilson(tp, tp + fn), spec=tn / (tn + fp), spec_ci=wilson(tn, tn + fp),
                    ppv=tp / (tp + fp), npv=tn / (tn + fn))
R['cq_sens_sub'] = {l: (int((pI & (m['ref_' + l] == 1)).sum()), int(m['ref_' + l].sum())) for l in ['IPH', 'IVH', 'SDH', 'EDH', 'SAH', 'ChronicBleed']}
sh = m.csf_offset_maxabs_mm.values
shm_dummy = sh.copy()  # placeholder for the median-offset variant, which came at this point in the original order; keeps the bootstrap stream aligned
M2 = []
for lab in ['MidlineShift', 'MassEffect']:
    for nmn, s in [('max', sh), ('median(dummy)', shm_dummy), ('vol', vol)]:
        M2.append((lab, nmn, *auc_ci(m['ref_' + lab].values, s)))
R['cq_mls_table11'] = M2
R['cq_prev'] = {l: int(m['ref_' + l].sum()) for l in LAB}
R['cq_misc'] = dict(midline_est=int((m.midline_status == 'estimated').sum()), proxy_n=int(np.isfinite(sh).sum()), proxy_max=np.nanmax(sh),
                    bone=int((m.bone_source == 'bone_kernel').sum()), qc_flag=int(m.qc_flag.sum()),
                    edh_with_fx=int(((m.ref_EDH == 1) & (m.ref_Fracture == 1)).sum()),
                    fx_with_ich=int(((m.ref_ICH == 1) & (m.ref_Fracture == 1)).sum()),
                    max_fp_volume=float(m.loc[m.ref_ICH == 0, 'pred_volume_ml'].max()),
                    uni=int(((m.ref_ICH == 1) & (m['ref_BleedLocation-Left'] != m['ref_BleedLocation-Right'])).sum()))
# exclude Seg-CQ500 scans
segids = set(s1.index)
ex = m[~m.name.isin(segids)]
yx = ex.ref_ICH.values; px = ex.pred_volume_ml.values >= 1
R['cq_excl_seg_v2'] = dict(n=len(ex), pos=int(yx.sum()), auc=roc_auc_score(yx, ex.pred_volume_ml),
                           sens=(int((px & (yx == 1)).sum()), int(yx.sum())), spec=(int((~px & (yx == 0)).sum()), int((yx == 0).sum())))
R['segcq500_in_cq500_all_found'] = bool(segids <= set(m.name))

# ------------------------------------------------------------------ v3/v3b (replica of research_v3b_analyze.py)
rng = np.random.default_rng(20261022)


def bci(yv, s, B=2000):
    v = []; n = len(yv)
    for _ in range(B):
        i = rng.integers(0, n, n)
        if 0 < yv[i].sum() < n:
            v.append(roc_auc_score(yv[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def mci(a, B=2000):
    return np.percentile([a[rng.integers(0, len(a), len(a))].mean() for _ in range(B)], [2.5, 97.5])


bh = {V: pd.read_csv(E / f'v3_subtypes/{V}/{V}_bhsd_test_metrics.csv').merge(nm.reset_index(), on='case_id') for V in ['v3', 'v3b']}
R['bhsd_v3'] = {}
for V, d in bh.items():
    a = d.ANY_dice.values; df = a - d.dice_nopostmask.values
    e = dict(n=len(d), any_dice=a.mean(), any_ci=mci(a).tolist(), zero=int((a == 0).sum()), delta=df.mean(), delta_ci=mci(df).tolist(),
             p=stats.wilcoxon(df).pvalue, classes={})
    for c in SUB:
        msk = d[c + '_ref_ml'] > 0; yy = msk.values.astype(int); s = d[c + '_pred_ml'].values
        e['classes'][c] = dict(n_present=int(yy.sum()), dice_present=d.loc[msk, c + '_dice'].mean(), auc=roc_auc_score(yy, s), auc_ci=bci(yy, s).tolist(),
                               mae=float(np.abs(s - d[c + '_ref_ml']).mean()), sens1=float(((s >= 1) & (yy == 1)).sum() / yy.sum()),
                               spec1=float(((s < 1) & (yy == 0)).sum() / (yy == 0).sum()),
                               pred_any_voxels=int((d[c + '_pred_ml'] > 0).sum()))
    R['bhsd_v3'][V] = e
R['cq_v3b'] = {}
for sub, dd in [('all', m), ('excl', ex)]:
    out = {}
    for c in ['ANY'] + SUB:
        yy = dd['ref_' + ('ICH' if c == 'ANY' else c)].values.astype(int); s = dd[c + '_ml'].values
        out[c] = dict(n=len(yy), pos=int(yy.sum()), auc=roc_auc_score(yy, s), auc_ci=bci(yy, s).tolist(), ap=average_precision_score(yy, s),
                      sens1=int(((s >= 1) & (yy == 1)).sum()), spec1=int(((s < 1) & (yy == 0)).sum()), neg=int((yy == 0).sum()))
    yy = dd.ref_ICH.values; a2 = dd.pred_volume_ml.values; a3 = dd.ANY_ml.values; dfl = []
    for _ in range(2000):
        i = rng.integers(0, len(yy), len(yy)); dfl.append(roc_auc_score(yy[i], a3[i]) - roc_auc_score(yy[i], a2[i]))
    out['delta'] = dict(delta=roc_auc_score(yy, a3) - roc_auc_score(yy, a2), ci=np.percentile(dfl, [2.5, 97.5]).tolist(), v2=roc_auc_score(yy, a2))
    R['cq_v3b'][sub] = out
# cross-subtype matrix
Mx = np.zeros((5, 5)); Nn = []
for i, c in enumerate(SUB):
    sb = m[m['ref_' + c] == 1]; Nn.append(len(sb))
    for j, kk in enumerate(SUB):
        Mx[i, j] = (sb[kk + '_ml'] >= 1).mean()
R['cq_confusion'] = dict(rows=SUB, n=Nn, M=Mx.round(3).tolist())

save_json(R, OUT / 'verify_original.json')

# ------------------------------------------------------------------ comparison with manuscript values
t3 = {(r['model'], r['cohort']): r for r in rows}
for (mm, cc), rep in {('v1', 'all 96'): '0.436 (0.375–0.493) med 0.434 zero 14 MAE 23.34 RMSE 141.82 ICC 0.054 bias 10.72 (-267.9 to 289.3)',
                      ('v1', 'QC-accepted 95'): '0.440 (0.381–0.503) 0.435 13 9.03 14.12 0.910 -3.72 (-30.6 to 23.1)',
                      ('v2_prespecified', 'all 96'): '0.390 (0.330–0.455) 0.394 20 13.86 23.30 0.738 -9.03 (-51.3 to 33.3)',
                      ('v2_prespecified', 'QC-accepted 95'): '0.395 (0.330–0.456) 0.421 19 13.59 23.06 0.745 -9.55 (-50.9 to 31.8)',
                      ('v2_nomask_posthoc', 'all 96'): '0.461 (0.404–0.519) 0.515 11 10.71 18.03 0.858 -1.29 (-36.7 to 34.1)',
                      ('v2_nomask_posthoc', 'QC-accepted 95'): '0.466 (0.411–0.524) 0.525 10 10.28 17.34 0.869 -1.84 (-35.8 to 32.1)'}.items():
    r = t3[(mm, cc)]
    C(f'Table3 {mm} {cc}', rep, f"{r['mean_dice']:.3f} ({r['lo']:.3f}–{r['hi']:.3f}) {r['median']:.3f} {r['zero']} {r['MAE']:.2f} {r['RMSE']:.2f} {r['ICC']:.3f} {r['bias']:.2f} ({r['LoA_lo']:.1f} to {r['LoA_hi']:.1f})")
for p_, rep in zip(pair, ['-0.045 (-0.084 to -0.010) p0.553 47/96', '+0.026 (+0.011 to +0.041) p0.0001 58/96', '+0.006 (-0.042 to +0.054) p0.876 19/43', '+0.028 (-0.017 to +0.072) p0.218 24/43']):
    C('Table4 ' + p_['cmp'], rep, f"{p_['mean']:+.3f} ({p_['lo']:+.3f} to {p_['hi']:+.3f}) p{p_['p']:.3g} {p_['improved']}/{p_['n']}")
C('Seg means v1/v2/v2nm', '0.465/0.471/0.493', '/'.join(f'{x:.3f}' for x in R['seg_means'].values()))
e = R['edema']
C('Edema', '0.336 (0.280–0.388) med 0.378 MAE 3.70 RMSE 4.84 ICC 0.258 bias 1.97 (-6.8 to 10.8)',
  f"{e['dice']:.3f} ({e['lo']:.3f}–{e['hi']:.3f}) med {e['median']:.3f} MAE {e['MAE']:.2f} RMSE {e['RMSE']:.2f} ICC {e['ICC']:.3f} bias {e['bias']:.2f} ({e['LoA_lo']:.1f} to {e['LoA_hi']:.1f})")
for s_, rep in zip(strata, ['37 0.212 0.239 13 10', '23 0.489 0.524 0 0', '36 0.631 0.649 1 1', '13 0.404 0.437 3 3', '63 0.529 0.554 3 2', '54 0.488 0.504 4 3', '51 0.415 0.439 4 2', '35 0.406 0.434 3 4']):
    C('Table5 ' + s_['stratum'], rep, f"{s_['n']} {s_['v1']:.3f} {s_['v2']:.3f} {s_['z1']} {s_['z2']}")
x = R['v1_extra']
C('v1 extras', 'median 0.434 IoU 0.328 HD95 41.95 ASD 18.28 n93; outlier 1382.8 vs 0.11', f"{x['median_dice']:.3f} {x['mean_iou']:.3f} {x['hd95']:.2f} {x['asd']:.2f} n{x['n_surface']} empty{x['empty_pred']} {x['outlier']}")
C('QC', '0/92, 1/96, 0/43', f"{R['qc']['train_flag']}/{R['qc']['train_n']}, {R['qc']['test_flag']}/{R['qc']['test_n']}, {R['qc']['seg_flag']}/{R['qc']['seg_n']}")
for t in t2:
    C('Table2 SegCQ500 ' + t['method'], '', f"n{t['n']} {t['mean']:.3f} {t['median']:.3f} IoU {t['iou']:.3f} P {t['prec']:.3f} R {t['rec']:.3f}")
C('MR-BHSD v1 reader-avg', '0.399 0.404 IoU 0.296', f"{R['mrbhsd_v1']['mean']:.3f} {R['mrbhsd_v1']['median']:.3f} {R['mrbhsd_v1']['iou']:.3f}")
C('MR-BHSD rater-rater', '0.755', f"{R['mrbhsd_rater_rater_mean']:.3f}")
po = R['phe_overview']
C('PHE midline', '120/120 sym 0.952 angle 1.25 IQR -7 to 8; 17 at bound; 60L/59R/1B',
  f"{po['status']} sym {po['sym_median']:.3f} angle {po['angle_median']} {po['angle_iqr']} bound {po['at_bound']} {po['sides']}")
for l, a, lo, hi, n, pos in det:
    C(f'CQ500 v2 AUC {l}', '', f'{pos}/{n} {a:.3f} ({lo:.3f}–{hi:.3f})')
r_ = R['cq_rule']
C('CQ500 rule', '168/205 0.820 (0.761–0.866); 169/284 0.595 (0.537–0.651); PPV 0.594 NPV 0.820',
  f"{r_['tp']}/{r_['tp']+r_['fn']} {r_['sens']:.3f} ({r_['sens_ci'][0]:.3f}–{r_['sens_ci'][1]:.3f}); {r_['tn']}/{r_['tn']+r_['fp']} {r_['spec']:.3f} ({r_['spec_ci'][0]:.3f}–{r_['spec_ci'][1]:.3f}); PPV {r_['ppv']:.3f} NPV {r_['npv']:.3f}")
C('CQ500 sens by subtype', 'IPH 112/134 IVH 28/28 SDH 44/53 EDH 13/13 SAH 54/60 CB 12/19', str(R['cq_sens_sub']))
for row in M2:
    if row[1] == 'median(dummy)':
        continue   # alignment placeholder, not a result
    C(f'CQ500 {row[0]} {row[1]}', '', f'{row[6]}/{row[5]} {row[2]:.3f} ({row[3]:.3f}–{row[4]:.3f})')
C('CQ500 prevalence', 'ICH205 IPH134 SDH53 SAH60 IVH28 EDH13 CB19 Fx39 ME127 MLS65', str(R['cq_prev']))
C('CQ500 misc', 'midline 487; proxy 452; max 37; bone 135; EDH+Fx 11; Fx+ICH 31; max FP 90.0; unilateral 134', str(R['cq_misc']))
ce = R['cq_excl_seg_v2']
C('CQ500 excl SegCQ500 (v2)', '446/164: AUC 0.783; 128/164 0.780; 167/282 0.592', f"{ce['n']}/{ce['pos']}: AUC {ce['auc']:.3f}; {ce['sens']}; {ce['spec']}")
for V in ['v3', 'v3b']:
    e = R['bhsd_v3'][V]
    C(f'BHSD {V} any', {'v3': '0.390 (0.334–0.450) Δ-0.072 p<0.001 zero 14', 'v3b': '0.446 (0.390–0.500) Δ-0.016 (-0.036–0.006) p0.020 zero 6'}[V],
      f"{e['any_dice']:.3f} ({e['any_ci'][0]:.3f}–{e['any_ci'][1]:.3f}) Δ{e['delta']:.3f} ({e['delta_ci'][0]:.3f}–{e['delta_ci'][1]:.3f}) p{e['p']:.3g} zero {e['zero']}")
    for c in SUB:
        cc = e['classes'][c]
        C(f'BHSD {V} {c}', '', f"n{cc['n_present']} Dice {cc['dice_present']:.3f} AUC {cc['auc']:.3f} ({cc['auc_ci'][0]:.3f}–{cc['auc_ci'][1]:.3f}) MAE {cc['mae']:.1f} predvox>0 in {cc['pred_any_voxels']}")
for sub in ['all', 'excl']:
    for c in ['ANY'] + SUB:
        o = R['cq_v3b'][sub][c]
        C(f'CQ500 v3b {sub} {c}', '', f"{o['pos']}/{o['n']} AUC {o['auc']:.3f} ({o['auc_ci'][0]:.3f}–{o['auc_ci'][1]:.3f}) AP {o['ap']:.3f} sens {o['sens1']}/{o['pos']} spec {o['spec1']}/{o['neg']}")
    dd = R['cq_v3b'][sub]['delta']
    C(f'CQ500 v3b-v2 {sub}', '0.011 (-0.017–0.039)' if sub == 'all' else '', f"{dd['delta']:.3f} ({dd['ci'][0]:.3f}–{dd['ci'][1]:.3f}) v2 {dd['v2']:.3f}")
C('CQ500 v3b confusion', 'SAH col high', str(R['cq_confusion']))

w = max(len(a) for a, _, _ in chk)
with open(OUT / 'verify_original_comparison.txt', 'w') as f:
    for a, b, c in chk:
        line = f'{a:<{w}} | reported: {b}\n{"":<{w}} | recomputed: {c}'
        print(line); f.write(line + '\n')
