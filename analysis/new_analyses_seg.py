"""Step 3: post hoc segmentation/volumetry analyses (BHSD, Seg-CQ500, MR-BHSD, PHE-SICH-CT-IDS)."""
import json
import numpy as np, pandas as pd
from scipy import stats
from common import *

OUTT = OUT / 'tables'; OUTT.mkdir(exist_ok=True)
rng = np.random.default_rng(20261027)
B = 10000
R = {}


def lin_ccc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    return 2 * sxy / (x.var(ddof=1) + y.var(ddof=1) + (x.mean() - y.mean()) ** 2)


def boot_ci(f, n, B=B):
    v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        try:
            r = f(i)
        except Exception:
            continue
        if np.isfinite(r):
            v.append(r)
    return np.percentile(v, [2.5, 97.5])


def agreement_full(p, y, label):
    p = np.asarray(p, float); y = np.asarray(y, float); n = len(y)
    a = agree(p, y)
    out = dict(label=label, n=n, **a)
    out['ICC_ci'] = boot_ci(lambda i: icc_a1(p[i], y[i]), n).tolist()
    out['MAE_ci'] = boot_ci(lambda i: np.abs(p[i] - y[i]).mean(), n).tolist()
    out['RMSE_ci'] = boot_ci(lambda i: np.sqrt(((p[i] - y[i]) ** 2).mean()), n).tolist()
    out['CCC'] = lin_ccc(p, y); out['CCC_ci'] = boot_ci(lambda i: lin_ccc(p[i], y[i]), n).tolist()
    out['spearman'] = stats.spearmanr(p, y).statistic
    # proportional bias: regress difference on mean
    d = p - y; mn = (p + y) / 2
    sl = stats.linregress(mn, d)
    out['prop_bias_slope'] = sl.slope; out['prop_bias_p'] = sl.pvalue
    # log-ratio limits (both > 0)
    ok = (p > 0) & (y > 0)
    lr = np.log(p[ok] / y[ok])
    out['n_logratio'] = int(ok.sum())
    out['gm_ratio'] = float(np.exp(lr.mean()))
    out['ratio_loa'] = [float(np.exp(lr.mean() - 1.96 * lr.std(ddof=1))), float(np.exp(lr.mean() + 1.96 * lr.std(ddof=1)))]
    out['within_20pct'] = float((np.abs(p - y) <= 0.2 * y).mean())
    out['within_5ml'] = float((np.abs(p - y) <= 5).mean())
    return out


def dice_ci(d):
    d = np.asarray(d, float); n = len(d)
    lo, hi = boot_ci(lambda i: d[i].mean(), n)
    return d.mean(), lo, hi


def threshold_agreement(p, y, t=30):
    p = np.asarray(p) >= t; y = np.asarray(y) >= t
    tp = int((p & y).sum()); fn = int((~p & y).sum()); tn = int((~p & ~y).sum()); fp = int((p & ~y).sum())
    return dict(t=t, tp=tp, fn=fn, tn=tn, fp=fp, sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1), kappa=cohen_kappa(p, y),
                agreement=(tp + tn) / len(y), misclassified=fn + fp)


# ------------------------------------------------------------------ BHSD
v1 = pd.read_csv(E / 'neurocare_neuroimaging_advanced_20261021/per_scan_quantitative_metrics.csv').set_index('case_id')
v2 = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/v2_metrics.csv')
v2b = v2[v2.dataset == 'BHSD_test'].set_index('case_id'); v2s = v2[v2.dataset == 'Seg-CQ500'].set_index('case_id')
nm = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/ablation_no_postmask.csv').set_index('case_id')
v3 = pd.read_csv(E / 'v3_subtypes/v3/v3_bhsd_test_metrics.csv').set_index('case_id')
v3b = pd.read_csv(E / 'v3_subtypes/v3b/v3b_bhsd_test_metrics.csv').set_index('case_id')
k = sorted(v1.index)
y = v1.loc[k, 'reference_volume_ml'].values
assert np.allclose(v3b.loc[k, 'ANY_ref_ml'].values, y, atol=1e-6)
qcids = set(json.load(open(E / 'neurocare_qc_gate_20261021/QC_gate_results.json'))['flagged_ids'])
acc = np.array([c not in qcids for c in k])
models = {'v1': (v1.loc[k, 'dice'].values, v1.loc[k, 'prediction_volume_ml'].values),
          'v2 prespecified': (v2b.loc[k, 'dice'].values, v2b.loc[k, 'pred_ml'].values),
          'v2 no post-mask': (nm.loc[k, 'dice_nopostmask'].values, nm.loc[k, 'pred_ml'].values),
          'v3 (any)': (v3.loc[k, 'ANY_dice'].values, v3.loc[k, 'ANY_pred_ml'].values),
          'v3b (any)': (v3b.loc[k, 'ANY_dice'].values, v3b.loc[k, 'ANY_pred_ml'].values)}
bh = []
thr = []
for name, (d, p) in models.items():
    for sub, sel in [('all 96', np.ones(96, bool)), ('QC-accepted 95', acc)]:
        dm, lo, hi = dice_ci(d[sel])
        ag = agreement_full(p[sel], y[sel], f'{name} | {sub}')
        bh.append(dict(model=name, cohort=sub, dice=dm, dice_lo=lo, dice_hi=hi, dice_median=float(np.median(d[sel])),
                       dice_iqr=np.percentile(d[sel], [25, 75]).tolist(), zero=int((d[sel] == 0).sum()), **{kk: vv for kk, vv in ag.items() if kk != 'label'}))
        thr.append(dict(model=name, cohort=sub, **threshold_agreement(p[sel], y[sel], 30)))
R['bhsd_agreement'] = bh
R['bhsd_threshold30'] = thr
pd.DataFrame(bh).to_csv(OUTT / 'bhsd_agreement_full.csv', index=False)
pd.DataFrame(thr).to_csv(OUTT / 'bhsd_threshold30.csv', index=False)
R['bhsd_ref_volume'] = dict(median=float(np.median(y)), iqr=np.percentile(y, [25, 75]).tolist(), min=float(y.min()), max=float(y.max()),
                            n_ge30=int((y >= 30).sum()), n_lt10=int((y < 10).sum()))
# size dependence
for name in ['v1', 'v2 no post-mask', 'v3b (any)']:
    d = models[name][0]
    R[f'size_spearman_{name}'] = stats.spearmanr(np.log(y), d).statistic
# paired v3b vs v2 no post-mask (already in locked results) and v3b vs v1
d1 = models['v1'][0]; d3b = models['v3b (any)'][0]; d2 = models['v2 no post-mask'][0]
R['v3b_minus_v1'] = dict(mean=(d3b - d1).mean(), ci=boot_ci(lambda i: (d3b[i] - d1[i]).mean(), 96).tolist(), p=stats.wilcoxon(d3b, d1).pvalue)

# subtype volume agreement for v3b (per class, all 96 scans)
sv = []
for c in SUB:
    p = v3b.loc[k, c + '_pred_ml'].values; yy = v3b.loc[k, c + '_ref_ml'].values
    pres = yy > 0
    sv.append(dict(sub=c, n_present=int(pres.sum()), ICC=icc_a1(p, yy), ICC_ci=boot_ci(lambda i: icc_a1(p[i], yy[i]), 96).tolist(),
                   MAE=float(np.abs(p - yy).mean()), bias=float((p - yy).mean()),
                   ref_median_present=float(np.median(yy[pres])), pred_median_present=float(np.median(p[pres])),
                   pred_ge1_absent=int(((p >= 1) & ~pres).sum()), n_absent=int((~pres).sum()),
                   dice_present=float(v3b.loc[k, c + '_dice'].values[pres].mean()),
                   dice_present_ci=boot_ci(lambda i: v3b.loc[k, c + '_dice'].values[pres][i].mean(), int(pres.sum())).tolist(),
                   v3_dice_present=float(np.nan_to_num(v3.loc[k, c + '_dice'].values[pres]).mean())))
R['v3b_subtype_volume'] = sv
pd.DataFrame(sv).to_csv(OUTT / 'bhsd_v3b_subtype_volume.csv', index=False)

# ------------------------------------------------------------------ Seg-CQ500
seg4 = pd.read_csv(T / 'SegCQ500_four_method_comparison.csv')
segnm = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/segcq500_no_postmask.csv').set_index('case_id')
ks = sorted(segnm.index)
yref = segnm.loc[ks, 'ref_ml'].values
seg_models = {}
for mth, lab in [('blast_model1', 'BLAST-CT'), ('blast_min100mm3', 'BLAST-CT + size filter'), ('blast_hu45_r2', 'BLAST-CT + HU growth'), ('lightunet25d', 'v1')]:
    g = seg4[seg4.method == mth].set_index('case_id').loc[ks]
    assert np.allclose(g.reference_volume_ml.values, yref, rtol=1e-6, atol=1e-6), mth
    seg_models[lab] = (g.dice.values, g.prediction_volume_ml.fillna(g.pred_volume_ml).values)
seg_models['v2 prespecified'] = (v2s.loc[ks, 'dice'].values, v2s.loc[ks, 'pred_ml'].values)
seg_models['v2 no post-mask'] = (segnm.loc[ks, 'dice_nopostmask'].values, segnm.loc[ks, 'pred_ml'].values)
sg = []
for name, (d, p) in seg_models.items():
    dm, lo, hi = dice_ci(d)
    ag = agreement_full(p, yref, name)
    sg.append(dict(model=name, dice=dm, dice_lo=lo, dice_hi=hi, dice_median=float(np.median(d)), zero=int((d == 0).sum()),
                   **{kk: vv for kk, vv in ag.items() if kk != 'label'}))
R['seg_agreement'] = sg
pd.DataFrame(sg).to_csv(OUTT / 'segcq500_agreement_full.csv', index=False)
R['seg_ref_volume'] = dict(median=float(np.median(yref)), iqr=np.percentile(yref, [25, 75]).tolist(), max=float(yref.max()),
                           top3=sorted(zip(yref.round(1), ks))[-3:])
# BLAST vs v2 nomask paired
dB = seg_models['BLAST-CT'][0]; dV = seg_models['v2 no post-mask'][0]
R['seg_v2nm_minus_blast'] = dict(mean=(dV - dB).mean(), ci=boot_ci(lambda i: (dV[i] - dB[i]).mean(), 43).tolist(), p=stats.wilcoxon(dV, dB).pvalue)

# cross-pipeline: CQ500 DICOM pipeline volume (v2 no post-mask) vs Seg-CQ500 NIfTI pipeline volume (same model)
cq = load_cq500().set_index('name')
cp = cq.loc[ks]
x_dicom = cp.pred_volume_ml.values; x_nifti = segnm.loc[ks, 'pred_ml'].values
R['cross_pipeline'] = dict(n=len(ks), ICC=icc_a1(x_dicom, x_nifti), ICC_ci=boot_ci(lambda i: icc_a1(x_dicom[i], x_nifti[i]), 43).tolist(),
                           spearman=stats.spearmanr(x_dicom, x_nifti).statistic, median_abs_diff=float(np.median(np.abs(x_dicom - x_nifti))),
                           bias=float((x_dicom - x_nifti).mean()), loa=[float((x_dicom - x_nifti).mean() - 1.96 * (x_dicom - x_nifti).std(ddof=1)),
                                                                         float((x_dicom - x_nifti).mean() + 1.96 * (x_dicom - x_nifti).std(ddof=1))],
                           within_1ml=int((np.abs(x_dicom - x_nifti) <= 1).sum()), within_10pct=int((np.abs(x_dicom - x_nifti) <= 0.1 * np.maximum(x_nifti, 1e-9)).sum()),
                           dz=cp.dz.round(2).value_counts().to_dict(), series=cp.soft_series.value_counts().to_dict())
R['cq500_pipeline_vs_segref'] = agreement_full(x_dicom, yref, 'CQ500 DICOM pipeline v2 vs Seg-CQ500 reference')
R['seg43_reader_labels'] = {l: int(cp['ref_' + l].sum()) for l in ['ICH'] + SUB + ['ChronicBleed', 'MidlineShift', 'MassEffect']}
R['seg43_ich_k'] = pd.Series(cp[['R1:ICH', 'R2:ICH', 'R3:ICH']].sum(1)).value_counts().to_dict()
R['seg43_detect_1ml_dicom'] = int((x_dicom >= 1).sum())
R['seg_ct108'] = dict(ref=float(segnm.loc['CQ500-CT-108', 'ref_ml']), pred=float(segnm.loc['CQ500-CT-108', 'pred_ml']),
                      labels={l: int(cq.loc['CQ500-CT-108', 'ref_' + l]) for l in ['ICH'] + SUB + ['ChronicBleed', 'MidlineShift', 'MassEffect']})

# ------------------------------------------------------------------ MR-BHSD multi-rater
mr = pd.read_csv(T / 'MRBHSD_full43_compact_model_rater_metrics.csv')
W = mr.pivot(index='case_id', columns='reference_rater', values='reference_volume_ml')
P = mr.groupby('case_id').pred_volume_ml.first().loc[W.index]
Dd = mr.pivot(index='case_id', columns='reference_rater', values='dice').loc[W.index]
R['mr_n'] = len(W)
R['mr_rater_icc'] = icc_a1_multi(W.values)
R['mr_rater_icc_ci'] = boot_ci(lambda i: icc_a1_multi(W.values[i]), len(W)).tolist()
pairs = [('annot_1', 'annot_3'), ('annot_1', 'annot_5'), ('annot_3', 'annot_5')]
R['mr_rater_pair_icc'] = {f'{a}-{b}': icc_a1(W[a], W[b]) for a, b in pairs}
R['mr_rater_pair_mad'] = {f'{a}-{b}': float(np.abs(W[a] - W[b]).mean()) for a, b in pairs}
R['mr_model_vs_rater_icc'] = {r: icc_a1(P, W[r]) for r in W.columns}
R['mr_model_vs_mean_icc'] = icc_a1(P, W.mean(1)); R['mr_model_vs_mean_icc_ci'] = boot_ci(lambda i: icc_a1(P.values[i], W.mean(1).values[i]), len(W)).tolist()
R['mr_model_vs_mean_mae'] = float(np.abs(P - W.mean(1)).mean())
R['mr_rater_vs_rater_mae_mean'] = float(np.mean(list(R['mr_rater_pair_mad'].values())))
cv = W.std(1, ddof=1) / W.mean(1).replace(0, np.nan)
davg = Dd.mean(1)
okk = cv.notna()
R['mr_ambiguity'] = dict(cv_median=float(cv.median()), cv_iqr=np.percentile(cv[okk], [25, 75]).tolist(),
                         spearman_cv_vs_dice=stats.spearmanr(cv[okk], davg[okk]).statistic, p=stats.spearmanr(cv[okk], davg[okk]).pvalue,
                         spearman_refvol_vs_dice=stats.spearmanr(W.mean(1), davg).statistic,
                         dice_range_median=float((Dd.max(1) - Dd.min(1)).median()), n_range_gt_0_1=int(((Dd.max(1) - Dd.min(1)) > 0.1).sum()),
                         empty_any_rater=int((W == 0).any(axis=1).sum()))
R['mr_partial_corr_note'] = 'cv vs dice adjusting for size computed below'
# partial Spearman (rank-based) of CV vs Dice controlling log mean volume
rk = lambda s: stats.rankdata(s)
from numpy.linalg import lstsq
Xr = np.c_[np.ones(okk.sum()), rk(np.log1p(W.mean(1)[okk]))]
res_cv = rk(cv[okk]) - Xr @ lstsq(Xr, rk(cv[okk]), rcond=None)[0]
res_d = rk(davg[okk]) - Xr @ lstsq(Xr, rk(davg[okk]), rcond=None)[0]
R['mr_partial_spearman_cv_dice_given_size'] = float(np.corrcoef(res_cv, res_d)[0, 1])
# per-rater model Dice distribution
R['mr_model_dice_by_rater'] = {r: float(Dd[r].mean()) for r in Dd.columns}

# ------------------------------------------------------------------ PHE laterality recomputation (edema & v1)
lat = pd.read_csv(E / 'neurocare_v2_20261021/phe_sich/phe_laterality_midline.csv', dtype={'case': str})
lat['case'] = lat.case.str.zfill(4)
split = json.load(open(E / 'neurocare_v2_20261021/edema/split.json'))
geo = set(split['train']) | set(split['test'])
R['phe_geo_n'] = len(geo)
ok = lat[lat.case.isin(geo) & lat.documented_side.isin(['L', 'R'])]
R['phe_unilateral_geo'] = len(ok)
for col, nm_ in [('edema_side_auto', 'edema'), ('blood_v1_side_auto', 'v1')]:
    v = ok[ok[col].isin(['L', 'R'])]
    c = int((v[col] == v.documented_side).sum())
    R[f'phe_lat_{nm_}'] = dict(evaluable=len(v), abstained=len(ok) - len(v), correct=c, acc=c / len(v), ci=wilson(c, len(v)))
R['phe_demographics'] = dict(age_median=float(lat.age.median()), age_iqr=np.percentile(lat.age, [25, 75]).tolist(), age_range=[int(lat.age.min()), int(lat.age.max())],
                             male=int((lat.sex == 'male').sum()), female=int((lat.sex == 'female').sum()))
R['phe_demographics_geo'] = dict(age_median=float(lat[lat.case.isin(geo)].age.median()), male=int((lat[lat.case.isin(geo)].sex == 'male').sum()), n=int(lat.case.isin(geo).sum()))
lg = lat[lat.case.isin(geo)]
R['phe_v1_blood_edema_spearman'] = stats.spearmanr(lg.blood_v1_ml_assumed_mm, lg.edema_ml_assumed_mm).statistic
# edema small-volume overestimation check
ed = pd.read_csv(E / 'neurocare_v2_20261021/edema/edema_test_metrics.csv')
med = ed.ref_ml.median()
R['edema_bias_by_size'] = dict(ref_median=float(med), bias_small=float((ed.pred_ml - ed.ref_ml)[ed.ref_ml < med].mean()),
                               bias_large=float((ed.pred_ml - ed.ref_ml)[ed.ref_ml >= med].mean()),
                               prop_bias=stats.linregress((ed.pred_ml + ed.ref_ml) / 2, ed.pred_ml - ed.ref_ml)._asdict(),
                               ccc=lin_ccc(ed.pred_ml, ed.ref_ml), icc_ci=boot_ci(lambda i: icc_a1(ed.pred_ml.values[i], ed.ref_ml.values[i]), len(ed)).tolist(),
                               dice_iqr=np.percentile(ed.dice, [25, 75]).tolist(), ref_iqr=np.percentile(ed.ref_ml, [25, 75]).tolist())

save_json(R, OUT / 'new_analyses_seg.json')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
print(pd.DataFrame(bh)[['model', 'cohort', 'dice', 'dice_lo', 'dice_hi', 'dice_median', 'zero', 'MAE', 'MAE_ci', 'RMSE', 'ICC', 'ICC_ci', 'CCC', 'CCC_ci', 'bias', 'LoA_lo', 'LoA_hi', 'gm_ratio', 'ratio_loa', 'within_20pct', 'within_5ml', 'prop_bias_slope', 'prop_bias_p']].round(3).to_string())
print(pd.DataFrame(thr).round(3).to_string())
print(pd.DataFrame(sv).round(3).to_string())
print(pd.DataFrame(sg)[['model', 'dice', 'dice_lo', 'dice_hi', 'dice_median', 'zero', 'MAE', 'RMSE', 'ICC', 'ICC_ci', 'CCC', 'bias', 'LoA_lo', 'LoA_hi', 'gm_ratio', 'ratio_loa']].round(3).to_string())
for kk in ['bhsd_ref_volume', 'seg_ref_volume', 'seg_v2nm_minus_blast', 'v3b_minus_v1', 'cross_pipeline', 'seg43_reader_labels', 'seg43_ich_k', 'seg43_detect_1ml_dicom', 'seg_ct108',
           'mr_n', 'mr_rater_icc', 'mr_rater_icc_ci', 'mr_rater_pair_icc', 'mr_rater_pair_mad', 'mr_model_vs_rater_icc', 'mr_model_vs_mean_icc', 'mr_model_vs_mean_icc_ci', 'mr_model_vs_mean_mae', 'mr_rater_vs_rater_mae_mean',
           'mr_ambiguity', 'mr_partial_spearman_cv_dice_given_size', 'mr_model_dice_by_rater', 'phe_geo_n', 'phe_unilateral_geo', 'phe_lat_edema', 'phe_lat_v1', 'phe_demographics', 'phe_demographics_geo', 'phe_v1_blood_edema_spearman', 'edema_bias_by_size']:
    print(kk, R[kk])
print({kk: R[kk] for kk in R if kk.startswith('size_spearman')})
r = R['cq500_pipeline_vs_segref']; print('CQ500 pipeline vs Seg ref:', {kk: r[kk] for kk in ['n', 'MAE', 'RMSE', 'ICC', 'ICC_ci', 'CCC', 'bias', 'LoA_lo', 'LoA_hi', 'gm_ratio', 'ratio_loa']})
