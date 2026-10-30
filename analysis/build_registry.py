"""Single source of truth for every number quoted in the manuscript, tables and figures.
Reads the locked-analysis replica (verify_original.json) and the post hoc outputs; writes out/registry.json."""
import json
import numpy as np, pandas as pd
from common import *

VO = json.load(open(OUT / 'verify_original.json'))
N = json.load(open(OUT / 'new_analyses_cq500.json'))
S = json.load(open(OUT / 'new_analyses_seg.json'))
X = json.load(open(OUT / 'extra_checks.json'))
ID = json.load(open(OUT / 'id_checks.json'))
CQ = json.load(open(E / 'cq500_external/cq500_results.json'))
V2R = json.load(open(E / 'neurocare_v2_20261021/v2_results.json'))
FR = json.load(open(E / 'cq500_external/fracture_pilot_result.json'))
KAP = pd.read_csv(OUT / 'tables/cq500_reader_agreement_491.csv').set_index('label')
MK = pd.read_csv(OUT / 'tables/cq500_model_reader_kappa.csv')
V = {}


def f3(x):
    return f'{x:.3f}'


def ci3(a, lo, hi, d=3):
    return f'{a:.{d}f} ({lo:.{d}f}–{hi:.{d}f})'


def pct(k, n, d=1):
    return f'{100 * k / n:.{d}f}%'


def wil(k, n, d=3):
    lo, hi = wilson(k, n)
    return f'{k}/{n} ({k / n:.{d}f}; {lo:.{d}f}–{hi:.{d}f})'


# ---------------- data assembly
V['n_cq_released'] = 491; V['n_cq'] = 489
V['cq_bone_kernel'] = CQ['bone_kernel_available']
V['bhx_scans'] = CQ['bhx_scans_matched']
V['integrity_geom'] = 4 + 8 + 5 + 8; V['integrity_geom_den'] = 192 + 51 + 48 + 120
V['integrity_geom_pct'] = f"{100 * V['integrity_geom'] / V['integrity_geom_den']:.1f}"
V['integrity_all'] = V['integrity_geom'] + 2; V['integrity_all_den'] = V['integrity_geom_den'] + 491
V['integrity_all_pct'] = f"{100 * V['integrity_all'] / V['integrity_all_den']:.1f}"
V['id'] = ID

# ---------------- BHSD (locked replica)
t3 = {(r['model'], r['cohort']): r for r in VO['bhsd_table3']}
for key, (m_, c_) in {'v1': ('v1', 'all 96'), 'v1q': ('v1', 'QC-accepted 95'), 'v2p': ('v2_prespecified', 'all 96'), 'v2pq': ('v2_prespecified', 'QC-accepted 95'),
                      'v2': ('v2_nomask_posthoc', 'all 96'), 'v2q': ('v2_nomask_posthoc', 'QC-accepted 95')}.items():
    r = t3[(m_, c_)]
    V[f'bh_{key}_dice'] = ci3(r['mean_dice'], r['lo'], r['hi'])
    V[f'bh_{key}_dice_mean'] = f3(r['mean_dice'])
    V[f'bh_{key}_med'] = f3(r['median']); V[f'bh_{key}_zero'] = r['zero']
    V[f'bh_{key}_mae'] = f"{r['MAE']:.2f}"; V[f'bh_{key}_rmse'] = f"{r['RMSE']:.2f}"; V[f'bh_{key}_icc'] = f3(r['ICC'])
    V[f'bh_{key}_bias'] = f"{r['bias']:.2f} ({r['LoA_lo']:.1f} to {r['LoA_hi']:.1f})"
    V[f'bh_{key}_medae'] = f"{r['medAE']:.2f}"
bhn = {(r['model'], r['cohort']): r for r in S['bhsd_agreement']}
for key, (m_, c_) in {'v1': ('v1', 'all 96'), 'v1q': ('v1', 'QC-accepted 95'), 'v2p': ('v2 prespecified', 'all 96'), 'v2': ('v2 no post-mask', 'all 96'),
                      'v2q': ('v2 no post-mask', 'QC-accepted 95'), 'v3b': ('v3b (any)', 'all 96'), 'v3bq': ('v3b (any)', 'QC-accepted 95'), 'v3': ('v3 (any)', 'all 96')}.items():
    r = bhn[(m_, c_)]
    V[f'bh_{key}_icc_ci'] = ci3(r['ICC'], *r['ICC_ci'])
    V[f'bh_{key}_ccc'] = ci3(r['CCC'], *r['CCC_ci'])
    V[f'bh_{key}_ratio'] = f"{r['gm_ratio']:.2f} ({r['ratio_loa'][0]:.2f}–{r['ratio_loa'][1]:.2f})"
    V[f'bh_{key}_ratio_lo'] = f"{r['ratio_loa'][0]:.2f}"; V[f'bh_{key}_ratio_hi'] = f"{r['ratio_loa'][1]:.1f}"
    V[f'bh_{key}_w20'] = f"{100 * r['within_20pct']:.0f}"; V[f'bh_{key}_w5'] = f"{100 * r['within_5ml']:.0f}"
    V[f'bh_{key}_mae_ci'] = f"{r['MAE']:.2f} ({r['MAE_ci'][0]:.2f}–{r['MAE_ci'][1]:.2f})"
    V[f'bh_{key}_propbias'] = (r['prop_bias_slope'], r['prop_bias_p'])
    V[f'bh_{key}_dice_med_iqr'] = f"{r['dice_median']:.3f} ({r['dice_iqr'][0]:.3f}–{r['dice_iqr'][1]:.3f})"
    V[f'bh_{key}_n_ratio'] = r['n_logratio']
for r in S['bhsd_threshold30']:
    key = {'v1|all 96': 'v1', 'v1|QC-accepted 95': 'v1q', 'v2 prespecified|all 96': 'v2p', 'v2 no post-mask|all 96': 'v2', 'v3b (any)|all 96': 'v3b', 'v3 (any)|all 96': 'v3'}.get(f"{r['model']}|{r['cohort']}")
    if key:
        V[f'bh_{key}_t30'] = dict(kappa=f"{r['kappa']:.2f}", sens=f"{r['tp']}/{r['tp'] + r['fn']}", spec=f"{r['tn']}/{r['tn'] + r['fp']}", mis=r['misclassified'])
V['bh_ref_median'] = f"{S['bhsd_ref_volume']['median']:.1f}"; V['bh_ref_iqr'] = f"{S['bhsd_ref_volume']['iqr'][0]:.1f}–{S['bhsd_ref_volume']['iqr'][1]:.1f}"
V['bh_ref_range'] = f"{S['bhsd_ref_volume']['min']:.2f}–{S['bhsd_ref_volume']['max']:.1f}"
V['bh_n_ge30'] = S['bhsd_ref_volume']['n_ge30']; V['bh_n_lt10'] = S['bhsd_ref_volume']['n_lt10']
V['size_rho_v2'] = f"{S['size_spearman_v2 no post-mask']:.2f}"; V['size_rho_v1'] = f"{S['size_spearman_v1']:.2f}"; V['size_rho_v3b'] = f"{S['size_spearman_v3b (any)']:.2f}"
st = {r['stratum']: r for r in VO['strata_table5']}
V['strata'] = {k: dict(n=r['n'], v1=f3(r['v1']), v2=f3(r['v2']), z1=r['z1'], z2=r['z2']) for k, r in st.items()}
V['outlier_pred'] = f"{VO['v1_extra']['outlier'][0]:.1f}"; V['outlier_ref'] = f"{VO['v1_extra']['outlier'][1]:.2f}"
V['v1_iou'] = f3(VO['v1_extra']['mean_iou']); V['v1_hd95'] = f"{VO['v1_extra']['hd95']:.2f}"; V['v1_asd'] = f"{VO['v1_extra']['asd']:.2f}"
V['v1_empty'] = VO['v1_extra']['empty_pred']
pairs = {p['cmp']: p for p in VO['pairs_table4']}
for k_, nm_ in [('p_v2p_bh', 'v2_prespecified-v1 BHSD'), ('p_v2_bh', 'v2_nomask_posthoc-v1 BHSD'), ('p_v2p_seg', 'v2_prespecified-v1 SegCQ500'), ('p_v2_seg', 'v2_nomask-v1 SegCQ500')]:
    p = pairs[nm_]
    V[k_] = dict(d=f"{p['mean']:+.3f}", ci=f"{p['lo']:+.3f} to {p['hi']:+.3f}", p=p['p'], imp=f"{p['improved']}/{p['n']}")
V['p_v2_bh']['p_str'] = 'P < .001'
for V_ in ['v3', 'v3b']:
    e = VO['bhsd_v3'][V_]
    V[f'bh_{V_}_any'] = ci3(e['any_dice'], *e['any_ci']); V[f'bh_{V_}_any_mean'] = f3(e['any_dice'])
    V[f'bh_{V_}_delta'] = f"{e['delta']:+.3f} ({e['delta_ci'][0]:+.3f} to {e['delta_ci'][1]:+.3f})"; V[f'bh_{V_}_p'] = e['p']; V[f'bh_{V_}_zero'] = e['zero']
    V[f'bh_{V_}_cls'] = {c: dict(n=e['classes'][c]['n_present'], dice=f3(e['classes'][c]['dice_present']), auc=ci3(e['classes'][c]['auc'], *e['classes'][c]['auc_ci']),
                                 auc_pt=f3(e['classes'][c]['auc']), mae=f"{e['classes'][c]['mae']:.1f}", vox=e['classes'][c]['pred_any_voxels']) for c in SUB}
V['v3b_minus_v1'] = f"{S['v3b_minus_v1']['mean']:+.3f} ({S['v3b_minus_v1']['ci'][0]:+.3f} to {S['v3b_minus_v1']['ci'][1]:+.3f})"
V['v3b_cls_icc'] = {r['sub']: ci3(r['ICC'], *r['ICC_ci']) for r in S['v3b_subtype_volume']}
V['v3b_cls_icc_pt'] = {r['sub']: f"{r['ICC']:.2f}" for r in S['v3b_subtype_volume']}
V['v3b_cls_fp_absent'] = {r['sub']: f"{r['pred_ge1_absent']}/{r['n_absent']}" for r in S['v3b_subtype_volume']}

# ---------------- QC gate
V['qc'] = VO['qc']

# ---------------- Seg-CQ500
sg = {r['model']: r for r in S['seg_agreement']}
for key, m_ in [('blast', 'BLAST-CT'), ('blast_sf', 'BLAST-CT + size filter'), ('blast_hu', 'BLAST-CT + HU growth'), ('v1', 'v1'), ('v2p', 'v2 prespecified'), ('v2', 'v2 no post-mask')]:
    r = sg[m_]
    V[f'seg_{key}_dice'] = ci3(r['dice'], r['dice_lo'], r['dice_hi']); V[f'seg_{key}_dice_mean'] = f3(r['dice'])
    V[f'seg_{key}_med'] = f3(r['dice_median']); V[f'seg_{key}_zero'] = r['zero']
    V[f'seg_{key}_mae'] = f"{r['MAE']:.2f}"; V[f'seg_{key}_icc_ci'] = ci3(r['ICC'], *r['ICC_ci']); V[f'seg_{key}_icc'] = f3(r['ICC'])
    V[f'seg_{key}_bias'] = f"{r['bias']:.2f} ({r['LoA_lo']:.1f} to {r['LoA_hi']:.1f})"
ex = pd.read_csv(OUT / 'tables/segcq500_excl108_sensitivity.csv')
def exr(model, s):
    return ex[(ex.model == model) & (ex.set == s)].iloc[0]
V['seg_v2_x_icc'] = ci3(exr('v2 no post-mask', 'excl108').ICC, *X['seg_v2nm_excl108_icc_ci'])
V['seg_blast_x_icc'] = ci3(exr('BLAST-CT', 'excl108').ICC, *X['seg_blast_excl108_icc_ci'])
V['seg_dicom_x_icc'] = ci3(exr('v2 no post-mask, CQ500 DICOM pipeline', 'excl108').ICC, *X['seg_dicom_excl108_icc_ci'])
V['seg_v2_x_mae'] = f"{exr('v2 no post-mask', 'excl108').MAE:.2f}"; V['seg_blast_x_mae'] = f"{exr('BLAST-CT', 'excl108').MAE:.2f}"
V['seg_v1_x_icc'] = f"{exr('v1', 'excl108').ICC:.3f}"; V['seg_v2p_x_icc'] = f"{exr('v2 prespecified', 'excl108').ICC:.3f}"
V['seg_v2_x_dice'] = ci3(X['seg_v2nm_excl108_dice'][0], *X['seg_v2nm_excl108_dice'][1])
V['seg_ct108'] = S['seg_ct108']; V['seg43_labels'] = S['seg43_reader_labels']; V['seg43_k'] = S['seg43_ich_k']
V['seg_ref_median'] = f"{S['seg_ref_volume']['median']:.1f}"; V['seg_ref_iqr'] = f"{S['seg_ref_volume']['iqr'][0]:.1f}–{S['seg_ref_volume']['iqr'][1]:.1f}"
V['seg_v2_minus_blast'] = f"{S['seg_v2nm_minus_blast']['mean']:+.3f} ({S['seg_v2nm_minus_blast']['ci'][0]:+.3f} to {S['seg_v2nm_minus_blast']['ci'][1]:+.3f}); P = {S['seg_v2nm_minus_blast']['p']:.2f}"
cp = S['cross_pipeline']
V['xpipe_icc'] = ci3(cp['ICC'], *cp['ICC_ci']); V['xpipe_bias'] = f"{cp['bias']:.1f} ({cp['loa'][0]:.1f} to {cp['loa'][1]:.1f})"; V['xpipe_rho'] = f"{cp['spearman']:.2f}"
V['xpipe_medabs'] = f"{cp['median_abs_diff']:.1f}"
V['seg_thin'] = 39; V['seg_thick'] = 4   # from voxel volumes (<=0.2 mm3 vs >0.5 mm3), see extra check output

# ---------------- MR-BHSD
mr = pd.read_csv(T / 'MRBHSD_full43_compact_model_rater_metrics.csv')
mra = mr.groupby('case_id').dice.mean().values
rng = np.random.default_rng(20261030)
bs = [mra[rng.integers(0, len(mra), len(mra))].mean() for _ in range(10000)]
V['mr_v1_dice'] = ci3(mra.mean(), *np.percentile(bs, [2.5, 97.5])); V['mr_v1_dice_mean'] = f3(mra.mean())
V['mr_v1_dice_lo'] = float(np.percentile(bs, 2.5)); V['mr_v1_dice_hi'] = float(np.percentile(bs, 97.5))
V['mr_v1_dice_med'] = f3(np.median(mra))
rr = pd.read_csv(T / 'TableS4_Reference_reference_agreement.csv')
V['mr_rr_dice'] = f"{rr.mean_dice.mean():.3f}"; V['mr_rr_range'] = f"{rr.mean_dice.min():.3f}–{rr.mean_dice.max():.3f}"
V['mr_rr_med_range'] = f"{rr.median_dice.min():.3f}–{rr.median_dice.max():.3f}"
V['mr_ratio_pct'] = f"{100 * mra.mean() / rr.mean_dice.mean():.0f}"
V['mr_per_rater'] = {k: f3(v) for k, v in S['mr_model_dice_by_rater'].items()}
V['mr_rater_icc'] = ci3(S['mr_rater_icc'], *S['mr_rater_icc_ci'])
V['mr_pair_icc_range'] = f"{min(S['mr_rater_pair_icc'].values()):.2f}–{max(S['mr_rater_pair_icc'].values()):.2f}"
V['mr_model_icc_range'] = f"{min(S['mr_model_vs_rater_icc'].values()):.2f}–{max(S['mr_model_vs_rater_icc'].values()):.2f}"
V['mr_model_mean_icc'] = ci3(S['mr_model_vs_mean_icc'], *S['mr_model_vs_mean_icc_ci'])
V['mr_model_mae'] = f"{S['mr_model_vs_mean_mae']:.1f}"; V['mr_rr_mae'] = f"{S['mr_rater_vs_rater_mae_mean']:.1f}"
a = S['mr_ambiguity']
V['mr_cv_median'] = f"{100 * a['cv_median']:.0f}"; V['mr_cv_iqr'] = f"{100 * a['cv_iqr'][0]:.0f}–{100 * a['cv_iqr'][1]:.0f}"
V['mr_rho'] = f"{a['spearman_cv_vs_dice']:.2f}"; V['mr_rho_p'] = a['p']; V['mr_prho'] = f"{S['mr_partial_spearman_cv_dice_given_size']:.2f}"
V['mr_dice_range_med'] = f"{a['dice_range_median']:.3f}"; V['mr_range_gt01'] = a['n_range_gt_0_1']; V['mr_empty_any'] = a['empty_any_rater']
V['mr_cv_n'] = 42

# ---------------- PHE
po = VO['phe_overview']
V['phe'] = dict(n=po['n'], L=po['sides']['L'], R=po['sides']['R'], B=po['sides']['bilateral'], sym=f"{po['sym_median']:.3f}", ang=f"{po['angle_median']:.2f}",
                ang_iqr=f"{po['angle_iqr'][0]:.0f}° to {po['angle_iqr'][1]:.0f}°", bound=po['at_bound'])
V['phe_lat_edema'] = wil(106, 108); V['phe_lat_v1'] = wil(102, 109); V['phe_lat_v2'] = wil(106, 109)
ed = VO['edema']
V['phe_edema'] = dict(dice=ci3(ed['dice'], ed['lo'], ed['hi']), med=f3(ed['median']), mae=f"{ed['MAE']:.2f}", rmse=f"{ed['RMSE']:.2f}", icc=f3(ed['ICC']),
                      icc_ci=ci3(ed['ICC'], *S['edema_bias_by_size']['icc_ci']), bias=f"{ed['bias']:.2f} ({ed['LoA_lo']:.1f} to {ed['LoA_hi']:.1f})",
                      ref_med=f"{S['edema_bias_by_size']['ref_median']:.2f}", bias_small=f"{S['edema_bias_by_size']['bias_small']:.1f}",
                      bias_large=f"{S['edema_bias_by_size']['bias_large']:.1f}", dice_iqr=f"{S['edema_bias_by_size']['dice_iqr'][0]:.3f}–{S['edema_bias_by_size']['dice_iqr'][1]:.3f}")
V['phe_demo'] = S['phe_demographics']
V['phe_blood_edema_v2'] = f"{V2R['blood_edema']['spearman']:.2f}"; V['phe_blood_edema_v1'] = f"{S['phe_v1_blood_edema_spearman']:.2f}"
V['phe_icv'] = dict(median=f"{V2R['icv']['median']:.0f}", iqr=f"{V2R['icv']['IQR'][0]:.0f}–{V2R['icv']['IQR'][1]:.0f}", burden=f"{V2R['icv']['normalized_burden_median_pct']:.2f}")

# ---------------- CQ500 external (locked replica)
det = {r[0]: r for r in VO['cq_det_table9']}
for l in ['ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH', 'ChronicBleed']:
    r = det[l]
    V[f'cq_v2_auc_{l}'] = ci3(r[1], r[2], r[3]); V[f'cq_v2_aucpt_{l}'] = f3(r[1])
r = VO['cq_rule']
V['cq_v2_sens'] = wil(r['tp'], r['tp'] + r['fn']); V['cq_v2_spec'] = wil(r['tn'], r['tn'] + r['fp'])
V['cq_v2_sens_pt'] = f3(r['sens']); V['cq_v2_spec_pt'] = f3(r['spec']); V['cq_v2_ppv'] = f3(r['ppv']); V['cq_v2_npv'] = f3(r['npv'])
V['cq_v2_tp'] = r['tp']; V['cq_v2_fp'] = r['fp']; V['cq_v2_fn'] = r['fn']; V['cq_v2_tn'] = r['tn']
V['cq_v2_sub_sens'] = {l: wil(k, n) for l, (k, n) in VO['cq_sens_sub'].items()}
V['cq_v2_sub_sens_short'] = {l: f"{k}/{n}" for l, (k, n) in VO['cq_sens_sub'].items()}
ce = VO['cq_excl_seg_v2']
V['cq_v2_excl'] = dict(n=ce['n'], pos=ce['pos'], auc=f3(ce['auc']), sens=f"{ce['sens'][0]}/{ce['sens'][1]}", spec=f"{ce['spec'][0]}/{ce['spec'][1]}")
for row in VO['cq_mls_table11']:
    lab, nm_, a_, lo_, hi_, n_, pos_ = row
    if nm_ == 'median(dummy)':
        continue
    V[f'cq_{lab}_{nm_}'] = dict(auc=ci3(a_, lo_, hi_), n=n_, pos=pos_)
V['cq_prev'] = VO['cq_prev']; V['cq_misc'] = VO['cq_misc']
V['cq_proxy_max'] = f"{VO['cq_misc']['proxy_max']:.1f}"
V['cq_maxfp'] = f"{VO['cq_misc']['max_fp_volume']:.1f}"
o = VO['cq_v3b']['all']; x_ = VO['cq_v3b']['excl']
for c in ['ANY'] + SUB:
    V[f'cq_v3b_{c}'] = dict(auc=ci3(o[c]['auc'], *o[c]['auc_ci']), aucpt=f3(o[c]['auc']), ap=f3(o[c]['ap']), sens=wil(o[c]['sens1'], o[c]['pos']), spec=wil(o[c]['spec1'], o[c]['neg']),
                            sens_s=f"{o[c]['sens1']}/{o[c]['pos']}", spec_s=f"{o[c]['spec1']}/{o[c]['neg']}", pos=o[c]['pos'],
                            excl=ci3(x_[c]['auc'], *x_[c]['auc_ci']), excl_pos=x_[c]['pos'])
V['cq_v3b_minus_v2'] = f"{o['delta']['delta']:+.3f} ({o['delta']['ci'][0]:+.3f} to {o['delta']['ci'][1]:+.3f})"
V['cq_v3b_minus_v2_excl'] = f"{x_['delta']['delta']:+.3f} ({x_['delta']['ci'][0]:+.3f} to {x_['delta']['ci'][1]:+.3f})"
dl = N['delong_v3b_vs_v2']; V['cq_delong_p'] = f"{dl[3]:.2f}"; dlx = N['delong_v3b_vs_v2_excl']; V['cq_delong_p_excl'] = f"{dlx[3]:.2f}"
V['cq_confusion'] = VO['cq_confusion']
M_ = np.array(VO['cq_confusion']['M'])
V['sah_offdiag_range'] = f"{M_[[0, 1, 2, 4], 3].min():.2f}–{M_[[0, 1, 2, 4], 3].max():.2f}"
lat = CQ['laterality']
V['cq_lat'] = wil(lat['correct'], lat['evaluable']); V['cq_lat_uni'] = lat['unilateral_ich']; V['cq_lat_eval'] = lat['evaluable']
V['bhx'] = {k_: dict(rate=f"{100 * v_['rate']:.1f}", ci=f"{100 * v_['ci'][0]:.1f}–{100 * v_['ci'][1]:.1f}", hit=v_['hit'], boxes=v_['boxes'], scans=v_['scans']) for k_, v_ in CQ['bhx_box_hit'].items()}
V['fracture'] = dict(auc=f"{FR['fold0_test_AUC']:.3f}", n=FR['fold0_test_n'])

# ---------------- CQ500 post hoc
V['kappa'] = {l: dict(k=f"{KAP.loc[l, 'fleiss']:.2f}", k4=f"{KAP.loc[l, 'fleiss']:.4f}", ci=f"{KAP.loc[l, 'lo']:.2f}–{KAP.loc[l, 'hi']:.2f}",
                      split=f"{KAP.loc[l, 'split_among_majority_pos_pct']:.0f}", pos=int(KAP.loc[l, 'majority_pos']),
                      pair=f"{min(KAP.loc[l, ['cohen_R1R2', 'cohen_R2R3', 'cohen_R3R1']]):.2f}–{max(KAP.loc[l, ['cohen_R1R2', 'cohen_R2R3', 'cohen_R3R1']]):.2f}")
              for l in KAP.index}
loo = pd.DataFrame(N['loo'])
def loo_range(lab, sc):
    d = loo[(loo.label == lab) & (loo.score == sc)]
    return dict(sens=f"{d.reader_sens.min():.2f}–{d.reader_sens.max():.2f}", spec=f"{d.reader_spec.min():.2f}–{d.reader_spec.max():.2f}",
                m_sens=f"{d.model_sens_at_reader_spec.min():.2f}–{d.model_sens_at_reader_spec.max():.2f}",
                m_spec=f"{d.model_spec_at_reader_sens.min():.2f}–{d.model_spec_at_reader_sens.max():.2f}",
                m_auc=f"{d.model_auc.min():.2f}–{d.model_auc.max():.2f}", n=f"{d.n.min()}–{d.n.max()}")
V['loo'] = {f'{lab}|{sc}': loo_range(lab, sc) for lab, sc in [('ICH', 'ICH'), ('ICH', 'ICH_v3b'), ('IPH', 'IPH'), ('IVH', 'IVH'), ('SDH', 'SDH'), ('EDH', 'EDH'), ('SAH', 'SAH'),
                                                               ('MidlineShift', 'MidlineShift'), ('MassEffect', 'MassEffect')]}
amb = {(r['score'], r['k']): r for r in N['amb']}
V['amb'] = {k: dict(n=amb[('ICH', k)]['n'], med=f"{amb[('ICH', k)]['median']:.2f}", iqr=f"{amb[('ICH', k)]['q1']:.2f}–{amb[('ICH', k)]['q3']:.2f}",
                    frac=f"{100 * amb[('ICH', k)]['frac1ml']:.0f}", pos=amb[('ICH', k)]['pos1ml']) for k in range(4)}
V['amb_tau'] = f"{N['amb_tau_ICH']['tau']:.2f}"
au = N['amb_auc_ICH']
V['amb_auc_un'] = ci3(*au['unanimous']); V['amb_n_un'] = au['n_un']; V['amb_pos_un'] = au['pos_un']
V['amb_sens_k2'] = f"{au['sens_k2'][0]}/{au['sens_k2'][1]}"; V['amb_sens_k3'] = f"{au['sens_k3'][0]}/{au['sens_k3'][1]}"
V['amb_spec_k0'] = f"{au['spec_k0'][0]}/{au['spec_k0'][1]}"; V['amb_spec_k1'] = f"{au['spec_k1'][0]}/{au['spec_k1'][1]}"
V['amb_sens_k2_pct'] = f"{100 * au['sens_k2'][0] / au['sens_k2'][1]:.0f}"; V['amb_sens_k3_pct'] = f"{100 * au['sens_k3'][0] / au['sens_k3'][1]:.0f}"
V['amb_trend_p'] = N['amb_trend_v2'][1]
au3 = N['amb_auc_ICH_v3b']; V['amb_auc_un_v3b'] = ci3(*au3['unanimous'])
V['sub_un'] = {r['sub']: dict(auc_un=ci3(*r['auc_un']), pos_un=r['pos_un']) for r in N['sub_unanimous']}
V['fn'] = N['fn_profile']; V['fp'] = N['fp_profile']
V['fn_k2_pct'] = f"{100 * N['fn_profile']['fn_k2'] / N['fn_profile']['n_fn']:.0f}"
bt = {(r['model'], r['batch']): r for r in N['batches']}
for mdl in ['v2', 'v3b']:
    for b_ in ['B1', 'B2', 'All']:
        r = bt[(mdl, b_)]
        V[f'bt_{mdl}_{b_}'] = dict(n=r['n'], pos=r['pos'], prev=f"{100 * r['prev']:.1f}", auc=ci3(r['auc'], r['lo'], r['hi']),
                                  sens=wil(r['tp'], r['tp'] + r['fn']), spec=wil(r['tn'], r['tn'] + r['fp']),
                                  ppv=wil(r['tp'], r['tp'] + r['fp']), npv=wil(r['tn'], r['tn'] + r['fn']),
                                  ppv_pt=f"{r['ppv']:.2f}", npv_pt=f"{r['npv']:.2f}", auc_pt=f"{r['auc']:.3f}")
bd = N['batch_auc_diff_v2']; V['bt_diff_v2'] = f"{bd['diff']:+.3f} ({bd['ci'][0]:+.3f} to {bd['ci'][1]:+.3f})"
bd3 = N['batch_auc_diff_v3b']; V['bt_diff_v3b'] = f"{bd3['diff']:+.3f} ({bd3['ci'][0]:+.3f} to {bd3['ci'][1]:+.3f})"
V['batch_prev'] = N['batch_prev']
acq = {(r['model'], r['stratum']): r for r in N['acq']}
V['acq'] = {f"{k[0]}|{k[1]}": dict(n=v['n'], pos=v['pos'], auc=ci3(v['auc'], v['lo'], v['hi'])) for k, v in acq.items()}
dec = {r['sub']: r for r in N['subtype_decomposition']}
V['dec'] = {c: dict(n_pure=dec[c]['n_pure'], pos=dec[c]['pos'], wi_v3b=ci3(*dec[c]['auc_within_ich_v3b']), wi_v2=ci3(*dec[c]['auc_within_ich_v2']),
                    wi_v3b_pt=f"{dec[c]['auc_within_ich_v3b'][0]:.3f}", wi_v2_pt=f"{dec[c]['auc_within_ich_v2'][0]:.3f}",
                    p=dec[c]['delong_within_ich_v3b_minus_v2'][3], pure=(ci3(*dec[c]['auc_pure_vs_neg_v3b']) if dec[c]['auc_pure_vs_neg_v3b'][0] is not None and np.isfinite(dec[c]['auc_pure_vs_neg_v3b'][0]) else 'n.e.'),
                    wi_n=f"{dec[c]['pos_within_ich']}/{dec[c]['neg_within_ich']}") for c in SUB}
V['ich_nsub'] = N['n_subtypes_per_ich_scan']; V['ich_multi'] = sum(v for k, v in N['n_subtypes_per_ich_scan'].items() if int(k) >= 2)
V['ich_nosub'] = N['ich_without_any_subtype']
V['chronic_all_ich'] = N['chronic_vs_ich']
pr = N['proxy']
V['proxy'] = dict(n=pr['n'], pos=pr['pos'], med_pos=f"{pr['median_pos']:.1f}", iqr_pos=f"{pr['iqr_pos'][0]:.1f}–{pr['iqr_pos'][1]:.1f}",
                  med_neg=f"{pr['median_neg']:.1f}", iqr_neg=f"{pr['iqr_neg'][0]:.1f}–{pr['iqr_neg'][1]:.1f}", neg_over5=pr['neg_over5'], neg_n=pr['neg_n'],
                  neg_over5_pct=f"{100 * pr['neg_over5'] / pr['neg_n']:.0f}", pos_over15=pr['pos_over15'], max=f"{pr['max']:.1f}",
                  auc_ichpos=ci3(*pr['auc_ich_pos']), n_ichpos=pr['n_ich_pos'], auc_vol_ichpos=ci3(*pr['auc_vol_ich_pos']),
                  cv_vol=f"{pr['cv_auc_volume_only']:.3f}", cv_both=f"{pr['cv_auc_volume_plus_proxy']:.3f}",
                  delong_d=f"{pr['delong_proxy_vs_volume'][0]:+.3f}", delong_ci=f"{pr['delong_proxy_vs_volume'][1]:+.3f} to {pr['delong_proxy_vs_volume'][2]:+.3f}",
                  delong_p=f"{pr['delong_proxy_vs_volume'][3]:.2f}", tau=f"{pr['tau_k']:.2f}", not_comp=pr['mls_not_computable'],
                  by_k={d['k']: f"{d['median']:.1f}" for d in pr['by_k']})
V['mls_without_ich'] = X['mls_pos_ich_neg']; V['me_without_ich'] = X['me_pos_ich_neg']
V['ich_bilateral'] = X['ich_bilateral']; V['ich_noside'] = X['ich_noside']
ev = pd.DataFrame(N['evaluability'])
V['eval'] = {f"{r.dataset}|{r.endpoint}": dict(pos=r.positives, lb=f"{r.max_lb_perfect_sens:.2f}", lb95=f"{r.lb_if_sens_095:.2f}", hw=f"{r.auc_halfwidth_at_090:.3f}") for r in ev.itertuples()}
V['n_needed'] = N['n_needed']
thr = pd.DataFrame(N['thresholds'])
V['thr'] = {f"{r.model}|{r.threshold_ml:g}": dict(sens=f"{r.sens:.2f}", spec=f"{r.spec:.2f}", ppv=f"{r.ppv:.2f}", b1ppv=f"{r.b1_ppv:.2f}") for r in thr.itertuples()}
V['sens_at_spec90_v2'] = f"{N['sens_at_spec_v2']['0.9']:.2f}"; V['sens_at_spec90_v3b'] = f"{N['sens_at_spec_v3b']['0.9']:.2f}"
V['mk'] = {f"{r.label}|{r.score}": dict(rr=f"{r.reader_reader_mean:.2f}", rr_rng=f"{r.rr_min:.2f}–{r.rr_max:.2f}", mr=f"{r.model_reader_mean:.2f}", mr_rng=f"{r.mr_min:.2f}–{r.mr_max:.2f}") for r in MK.itertuples()}
V['cq_slices'] = N['acq_counts']; V['cq_series_C'] = N['series_CT_C']

save_json(V, OUT / 'registry.json')
print('registry keys:', len(V))
