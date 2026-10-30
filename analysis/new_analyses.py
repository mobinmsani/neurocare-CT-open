"""Step 2: post hoc analyses (all computed from per-scan files in the project folder).
All outputs -> out/new_analyses.json and out/tables/*.csv
"""
import json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from common import *

OUTT = OUT / 'tables'; OUTT.mkdir(exist_ok=True)
B = 10000
rng = np.random.default_rng(20261026)
R = {}


def fast_auc(y, s):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    m = y.sum(); n = (~y).sum()
    if m == 0 or n == 0:
        return np.nan
    r = stats.rankdata(s)
    return (r[y].sum() - m * (m + 1) / 2) / (m * n)


def boot_auc(y, s, B=B):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float); n = len(y)
    out = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        yy = y[i]
        if 0 < yy.sum() < n:
            out.append(fast_auc(yy, s[i]))
    a = fast_auc(y, s)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return a, lo, hi


def boot_stat(f, n, B=B):
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        v = f(i)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5])


def sens_spec(y, pos):
    y = np.asarray(y).astype(bool); pos = np.asarray(pos).astype(bool)
    tp = int((pos & y).sum()); fn = int((~pos & y).sum()); tn = int((~pos & ~y).sum()); fp = int((pos & ~y).sum())
    return tp, fn, tn, fp


m = load_cq500()
N = len(m)
readers = {l: m[[f'R{i}:{l}' for i in (1, 2, 3)]].values.astype(int) for l in LAB}
kcount = {l: readers[l].sum(1) for l in LAB}

# ------------------------------------------------------------------ 1. Reader agreement (CQ500, 489 analysed scans)
rows = []
labs_k = ['ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH', 'ChronicBleed', 'BleedLocation-Left', 'BleedLocation-Right', 'Fracture',
          'CalvarialFracture', 'OtherFracture', 'MassEffect', 'MidlineShift']
for l in labs_k:
    X = readers[l]
    cnt = np.c_[X.sum(1), 3 - X.sum(1)]
    fk = fleiss_kappa(cnt)
    lo, hi = boot_stat(lambda i: fleiss_kappa(cnt[i]), N, B=4000)
    pk = [cohen_kappa(X[:, a], X[:, b]) for a, b in [(0, 1), (0, 2), (1, 2)]]
    k = X.sum(1)
    rows.append(dict(label=l, fleiss=fk, lo=lo, hi=hi, cohen_min=min(pk), cohen_max=max(pk),
                     n0=int((k == 0).sum()), n1=int((k == 1).sum()), n2=int((k == 2).sum()), n3=int((k == 3).sum()),
                     majority_pos=int((k >= 2).sum()), unanimous_pct=100 * float(((k == 0) | (k == 3)).mean()),
                     split_among_majority_pos_pct=100 * float((k == 2).sum() / max((k >= 2).sum(), 1))))
kap = pd.DataFrame(rows)
kap.to_csv(OUTT / 'cq500_reader_agreement.csv', index=False)
R['kappa'] = rows

# ------------------------------------------------------------------ 2. Scores per finding
score = {'ICH': m.pred_volume_ml.values, 'ICH_v3b': m.ANY_ml.values,
         'IPH': m.IPH_ml.values, 'IVH': m.IVH_ml.values, 'SDH': m.SDH_ml.values, 'EDH': m.EDH_ml.values, 'SAH': m.SAH_ml.values,
         'MidlineShift': m.csf_offset_maxabs_mm.values, 'MassEffect': m.csf_offset_maxabs_mm.values}


# ------------------------------------------------------------------ 3. Leave-one-reader-out operating points
def model_at_spec(y, s, spec_target):
    """sensitivity of the score at the lowest threshold achieving specificity >= target"""
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    thr = np.unique(s)
    best = None
    for t in thr:
        p = s >= t
        sp = (~p & ~y).sum() / (~y).sum()
        if sp >= spec_target:
            se = (p & y).sum() / y.sum()
            best = se
            break
    # thresholds ascending -> first t meeting spec is the most sensitive
    return best


def model_at_sens(y, s, sens_target):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    thr = np.unique(s)[::-1]
    for t in thr:
        p = s >= t
        se = (p & y).sum() / y.sum()
        if se >= sens_target:
            return (~p & ~y).sum() / (~y).sum()
    return 0.0


loo = []
for lab, sc_key in [('ICH', 'ICH'), ('ICH', 'ICH_v3b'), ('IPH', 'IPH'), ('IVH', 'IVH'), ('SDH', 'SDH'), ('EDH', 'EDH'), ('SAH', 'SAH'),
                    ('MidlineShift', 'MidlineShift'), ('MassEffect', 'MassEffect')]:
    X = readers[lab]; s_all = score[sc_key]
    for i in range(3):
        j, k = [a for a in range(3) if a != i]
        conc = X[:, j] == X[:, k]
        ok = conc & np.isfinite(s_all)
        yref = X[ok, j].astype(bool); rd = X[ok, i].astype(bool); s = s_all[ok]
        tp, fn, tn, fp = sens_spec(yref, rd)
        rs = tp / (tp + fn); rsp = tn / (tn + fp)
        a = fast_auc(yref, s)
        loo.append(dict(label=lab, score=sc_key, reader=f'R{i+1}', n=int(ok.sum()), pos=int(yref.sum()),
                        reader_sens=rs, reader_sens_ci=wilson(tp, tp + fn), reader_spec=rsp, reader_spec_ci=wilson(tn, tn + fp),
                        model_auc=a, model_sens_at_reader_spec=model_at_spec(yref, s, rsp), model_spec_at_reader_sens=model_at_sens(yref, s, rs),
                        model_sens_1ml=float(((s >= 1) & yref).sum() / yref.sum()) if sc_key not in ('MidlineShift', 'MassEffect') else np.nan,
                        model_spec_1ml=float(((s < 1) & ~yref).sum() / (~yref).sum()) if sc_key not in ('MidlineShift', 'MassEffect') else np.nan))
loo = pd.DataFrame(loo)
loo.to_csv(OUTT / 'cq500_leave_one_reader_out.csv', index=False)
R['loo'] = loo.to_dict('records')

# readers vs 2-of-3 majority (includes own read; optimistic)
vm = []
for lab in ['ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH', 'MidlineShift', 'MassEffect', 'Fracture']:
    y = m['ref_' + lab].values.astype(bool)
    for i in range(3):
        tp, fn, tn, fp = sens_spec(y, readers[lab][:, i])
        vm.append(dict(label=lab, reader=f'R{i+1}', sens=tp / (tp + fn), spec=tn / (tn + fp)))
R['reader_vs_majority'] = vm
pd.DataFrame(vm).to_csv(OUTT / 'cq500_reader_vs_majority.csv', index=False)

# ------------------------------------------------------------------ 4. Label ambiguity (soft labels)
amb = []
for lab, sc_key in [('ICH', 'ICH'), ('ICH', 'ICH_v3b')]:
    s = score[sc_key]; k = kcount[lab]
    for kk in range(4):
        sel = k == kk
        amb.append(dict(label=lab, score=sc_key, k=kk, n=int(sel.sum()), median=float(np.median(s[sel])),
                        q1=float(np.percentile(s[sel], 25)), q3=float(np.percentile(s[sel], 75)),
                        pos1ml=int((s[sel] >= 1).sum()), frac1ml=float((s[sel] >= 1).mean())))
    tau = stats.kendalltau(k, s)
    R[f'amb_tau_{sc_key}'] = dict(tau=tau.statistic, p=tau.pvalue)
    # unanimous-only AUC vs all
    un = (k == 0) | (k == 3)
    R[f'amb_auc_{sc_key}'] = dict(all=boot_auc(k >= 2, s), unanimous=boot_auc(k[un] == 3, s[un]), n_un=int(un.sum()), pos_un=int((k[un] == 3).sum()),
                                  # specificity among unanimous negatives vs dissent negatives
                                  spec_k0=(int((s[k == 0] < 1).sum()), int((k == 0).sum())),
                                  spec_k1=(int((s[k == 1] < 1).sum()), int((k == 1).sum())),
                                  sens_k2=(int((s[k == 2] >= 1).sum()), int((k == 2).sum())),
                                  sens_k3=(int((s[k == 3] >= 1).sum()), int((k == 3).sum())))
# Cochran-Armitage trend for proportion >=1 mL (v2)
def cochran_armitage(x, n, t):
    x, n, t = map(np.asarray, (x, n, t)); N_ = n.sum(); p = x.sum() / N_
    T_ = (t * (x - n * p)).sum()
    V = p * (1 - p) * ((n * t ** 2).sum() - (n * t).sum() ** 2 / N_)
    zst = T_ / np.sqrt(V)
    return zst, 2 * stats.norm.sf(abs(zst))
a2 = [r for r in amb if r['score'] == 'ICH']
R['amb_trend_v2'] = cochran_armitage([r['pos1ml'] for r in a2], [r['n'] for r in a2], [0, 1, 2, 3])
pd.DataFrame(amb).to_csv(OUTT / 'cq500_label_ambiguity.csv', index=False)
R['amb'] = amb
# subtype unanimity AUCs (v3b)
subun = []
for c in SUB:
    k = kcount[c]; s = score[c]; un = (k == 0) | (k == 3)
    a_all = boot_auc(k >= 2, s); a_un = boot_auc(k[un] == 3, s[un]) if (k[un] == 3).sum() > 1 else (np.nan,) * 3
    subun.append(dict(sub=c, pos=int((k >= 2).sum()), auc=a_all, n_un=int(un.sum()), pos_un=int((k[un] == 3).sum()), auc_un=a_un,
                      n_k1=int((k == 1).sum()), n_k2=int((k == 2).sum())))
R['sub_unanimous'] = subun

# ------------------------------------------------------------------ 5. Spectrum: CQ500 batches B1 vs B2
spec_rows = []
for sc_key, s_all in [('v2', m.pred_volume_ml.values), ('v3b', m.ANY_ml.values)]:
    for bt in ['B1', 'B2', 'All']:
        sel = np.ones(N, bool) if bt == 'All' else (m.Category.values == bt)
        y = m.ref_ICH.values[sel].astype(bool); s = s_all[sel]
        a, lo, hi = boot_auc(y, s)
        tp, fn, tn, fp = sens_spec(y, s >= 1)
        spec_rows.append(dict(model=sc_key, batch=bt, n=int(sel.sum()), pos=int(y.sum()), prev=y.mean(), auc=a, lo=lo, hi=hi,
                              sens=tp / (tp + fn), sens_ci=wilson(tp, tp + fn), spec=tn / (tn + fp), spec_ci=wilson(tn, tn + fp),
                              ppv=tp / (tp + fp), ppv_ci=wilson(tp, tp + fp), npv=tn / (tn + fn), npv_ci=wilson(tn, tn + fn), tp=tp, fn=fn, tn=tn, fp=fp))
    # AUC difference B2-B1 (independent bootstrap)
    s1_ = s_all[m.Category.values == 'B1']; y1 = m.ref_ICH.values[m.Category.values == 'B1'].astype(bool)
    s2_ = s_all[m.Category.values == 'B2']; y2 = m.ref_ICH.values[m.Category.values == 'B2'].astype(bool)
    d = []
    for _ in range(B):
        i1 = rng.integers(0, len(y1), len(y1)); i2 = rng.integers(0, len(y2), len(y2))
        if 0 < y1[i1].sum() < len(y1) and 0 < y2[i2].sum() < len(y2):
            d.append(fast_auc(y2[i2], s2_[i2]) - fast_auc(y1[i1], s1_[i1]))
    d = np.array(d)
    R[f'batch_auc_diff_{sc_key}'] = dict(diff=fast_auc(y2, s2_) - fast_auc(y1, s1_), ci=np.percentile(d, [2.5, 97.5]).tolist(),
                                         p_two_sided=float(2 * min((d <= 0).mean(), (d >= 0).mean())))
pd.DataFrame(spec_rows).to_csv(OUTT / 'cq500_batches.csv', index=False)
R['batches'] = spec_rows
# subtype prevalence by batch
R['batch_prev'] = {bt: {l: int(m.loc[m.Category == bt, 'ref_' + l].sum()) for l in ['ICH'] + SUB + ['ChronicBleed', 'Fracture', 'MassEffect', 'MidlineShift']} for bt in ['B1', 'B2']}
R['batch_n'] = m.Category.value_counts().to_dict()

# ------------------------------------------------------------------ 6. Acquisition strata (slice spacing)
m['dz_cat'] = pd.cut(m.dz, [0, 1.25, 4.0, 10], labels=['<=1.25 mm', '2.5-3.75 mm', '~5 mm'])
acq = []
for sc_key, s_all in [('v2', m.pred_volume_ml.values), ('v3b', m.ANY_ml.values)]:
    for cat in ['<=1.25 mm', '2.5-3.75 mm', '~5 mm']:
        sel = (m.dz_cat == cat).values
        y = m.ref_ICH.values[sel].astype(bool); s = s_all[sel]
        a, lo, hi = boot_auc(y, s)
        tp, fn, tn, fp = sens_spec(y, s >= 1)
        acq.append(dict(model=sc_key, stratum=cat, n=int(sel.sum()), pos=int(y.sum()), auc=a, lo=lo, hi=hi,
                        sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1), tp=tp, fn=fn, tn=tn, fp=fp,
                        median_vol_neg=float(np.median(s[~y])) if (~y).sum() else np.nan,
                        dz_range=(float(m.dz[sel].min()), float(m.dz[sel].max()))))
pd.DataFrame(acq).to_csv(OUTT / 'cq500_slice_strata.csv', index=False)
R['acq'] = acq
R['acq_counts'] = m.dz_cat.value_counts().to_dict()
R['series_CT_C'] = m.loc[m.soft_series == 'CT C', ['name', 'pred_volume_ml', 'ref_ICH', 'dz']].to_dict('records')
R['bone_kernel_counts'] = m.bone_source.value_counts().to_dict()
# predicted volume in reader-negative scans by slice stratum (Kruskal)
neg = m.ref_ICH == 0
R['neg_vol_by_dz_kruskal'] = stats.kruskal(*[m.loc[neg & (m.dz_cat == c), 'pred_volume_ml'] for c in ['<=1.25 mm', '2.5-3.75 mm', '~5 mm']]).pvalue

# ------------------------------------------------------------------ 7. Subtype decomposition (v3b vs v2 total volume)
dec = []
ich = m.ref_ICH.values == 1
anysub = m[['ref_' + c for c in SUB]].sum(1).values
for c in SUB:
    y = m['ref_' + c].values.astype(bool)
    pure = y & (anysub == 1)
    negs = m.ref_ICH.values == 0
    r = dict(sub=c, pos=int(y.sum()), n_pure=int(pure.sum()), pos_within_ich=int(y[ich].sum()), neg_within_ich=int((~y[ich]).sum()))
    r['auc_presence_v3b'] = boot_auc(y, m[c + '_ml'].values)
    r['auc_presence_v2'] = boot_auc(y, m.pred_volume_ml.values)
    r['auc_within_ich_v3b'] = boot_auc(y[ich], m[c + '_ml'].values[ich])
    r['auc_within_ich_v2'] = boot_auc(y[ich], m.pred_volume_ml.values[ich])
    # share of class volume relative to total (fraction score) within ICH
    frac = m[c + '_ml'].values / np.maximum(m.ANY_ml.values, 1e-6)
    r['auc_within_ich_v3b_fraction'] = boot_auc(y[ich], frac[ich])
    if pure.sum() >= 3:
        sel = pure | negs
        r['auc_pure_vs_neg_v3b'] = boot_auc(pure[sel], m[c + '_ml'].values[sel])
    else:
        r['auc_pure_vs_neg_v3b'] = (np.nan, np.nan, np.nan)
    # paired DeLong within ICH: v3b class volume vs v2 total volume
    dd = delong_paired(y[ich], m[c + '_ml'].values[ich], m.pred_volume_ml.values[ich])
    r['delong_within_ich_v3b_minus_v2'] = dd
    dec.append(r)
R['subtype_decomposition'] = dec
rows = []
for r in dec:
    rows.append({'Subtype': r['sub'], 'Positives': r['pos'], 'Pure': r['n_pure'],
                 'Presence AUC v3b': ci_str(*r['auc_presence_v3b']), 'Presence AUC v2 total': ci_str(*r['auc_presence_v2']),
                 'Within-ICH n (+/-)': f"{r['pos_within_ich']}/{r['neg_within_ich']}",
                 'Within-ICH AUC v3b': ci_str(*r['auc_within_ich_v3b']), 'Within-ICH AUC v2 total': ci_str(*r['auc_within_ich_v2']),
                 'Within-ICH AUC v3b fraction': ci_str(*r['auc_within_ich_v3b_fraction']),
                 'Pure vs negative AUC v3b': ci_str(*r['auc_pure_vs_neg_v3b']) if np.isfinite(r['auc_pure_vs_neg_v3b'][0]) else 'n.e.',
                 'DeLong p (within-ICH v3b vs v2)': f"{r['delong_within_ich_v3b_minus_v2'][3]:.3g}"})
pd.DataFrame(rows).to_csv(OUTT / 'cq500_subtype_decomposition.csv', index=False)
# co-occurrence
co = pd.DataFrame([[int(((m['ref_' + a] == 1) & (m['ref_' + b] == 1)).sum()) for b in SUB] for a in SUB], index=SUB, columns=SUB)
co.to_csv(OUTT / 'cq500_subtype_cooccurrence.csv')
R['cooccurrence'] = co.values.tolist()
R['n_subtypes_per_ich_scan'] = pd.Series(anysub[ich]).value_counts().sort_index().to_dict()
R['ich_without_any_subtype'] = int(((anysub == 0) & ich).sum())

# ------------------------------------------------------------------ 8. Statistical evaluability
def wilson_lb_all(n):
    return n / (n + Z * Z)


def n_needed(sens_obs, lb_target):
    # observed sensitivity treated as an exact proportion (x = p*n), matching the continuous curves in Figure 6B;
    # for a fixed proportion the Wilson lower bound increases monotonically with n
    for n in range(5, 5000):
        if wilson(sens_obs * n, n)[0] >= lb_target:
            return n
    return np.nan


def hanley_mcneil_halfwidth(a, npos, nneg):
    q1 = a / (2 - a); q2 = 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (npos - 1) * (q1 - a * a) + (nneg - 1) * (q2 - a * a)) / (npos * nneg))
    return Z * se


ev = []
cases = [('CQ500', 'Any ICH', 205, 284), ('CQ500', 'IPH', 134, 355), ('CQ500', 'SAH', 60, 429), ('CQ500', 'SDH', 53, 436),
         ('CQ500', 'IVH', 28, 461), ('CQ500', 'Chronic bleed', 19, 470), ('CQ500', 'EDH', 13, 476), ('CQ500', 'Midline shift (presence)', 65, 424),
         ('CQ500', 'Mass effect', 127, 362), ('CQ500', 'Skull fracture', 39, 450), ('CQ500', 'Calvarial fracture', 34, 455),
         ('BHSD test', 'EDH (voxel)', 13, 83), ('BHSD test', 'SDH (voxel)', 35, 61), ('BHSD test', 'SAH (voxel)', 51, 45),
         ('BHSD test', 'IVH (voxel)', 54, 42), ('BHSD test', 'IPH (voxel)', 63, 33)]
for ds, ep, pos, negn in cases:
    ev.append(dict(dataset=ds, endpoint=ep, positives=pos, negatives=negn, max_lb_perfect_sens=wilson_lb_all(pos),
                   lb_if_sens_095=wilson(0.95 * pos, pos)[0],
                   auc_halfwidth_at_090=hanley_mcneil_halfwidth(0.90, pos, negn)))
R['n_needed'] = dict(lb090_obs100=n_needed(1.0, 0.90), lb090_obs095=n_needed(0.95, 0.90), lb095_obs100=n_needed(1.0, 0.95),
                     lb080_obs090=n_needed(0.90, 0.80), lb090_obs098=n_needed(0.98, 0.90))
pd.DataFrame(ev).to_csv(OUTT / 'statistical_evaluability.csv', index=False)
R['evaluability'] = ev

# ------------------------------------------------------------------ 9. Threshold table (CQ500 any ICH)
thr_rows = []
for sc_key, s_all in [('v2', m.pred_volume_ml.values), ('v3b', m.ANY_ml.values)]:
    y = m.ref_ICH.values.astype(bool)
    b1 = (m.Category == 'B1').values
    for t in [0.1, 0.5, 1, 2, 5, 10, 20]:
        tp, fn, tn, fp = sens_spec(y, s_all >= t)
        tp1, fn1, tn1, fp1 = sens_spec(y[b1], s_all[b1] >= t)
        thr_rows.append(dict(model=sc_key, threshold_ml=t, sens=tp / (tp + fn), sens_ci=wilson(tp, tp + fn), spec=tn / (tn + fp), spec_ci=wilson(tn, tn + fp),
                             ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1), b1_ppv=tp1 / max(tp1 + fp1, 1), b1_npv=tn1 / max(tn1 + fn1, 1),
                             b1_sens=tp1 / (tp1 + fn1), b1_spec=tn1 / (tn1 + fp1)))
    # sensitivity at 90% and 95% specificity
    R[f'sens_at_spec_{sc_key}'] = {sp: model_at_spec(y, s_all, sp) for sp in [0.80, 0.90, 0.95]}
pd.DataFrame(thr_rows).to_csv(OUTT / 'cq500_threshold_table.csv', index=False)
R['thresholds'] = thr_rows

# ------------------------------------------------------------------ 10. DeLong
y = m.ref_ICH.values
R['delong_v2_ich'] = delong_ci(y, m.pred_volume_ml.values)
R['delong_v3b_ich'] = delong_ci(y, m.ANY_ml.values)
R['delong_v3b_vs_v2'] = delong_paired(y, m.ANY_ml.values, m.pred_volume_ml.values)
ex = ~m.name.isin(pd.read_csv(T / 'SegCQ500_four_method_comparison.csv').case_id.unique())
R['delong_v3b_vs_v2_excl'] = delong_paired(y[ex], m.ANY_ml.values[ex], m.pred_volume_ml.values[ex])

# ------------------------------------------------------------------ 11. FP / FN characterisation (v2, 1 mL)
pI = m.pred_volume_ml >= 1
fp_ = m[(m.ref_ICH == 0) & pI]; tn_ = m[(m.ref_ICH == 0) & ~pI]; fn_ = m[(m.ref_ICH == 1) & ~pI]
def cnt(df, col):
    return int(df[col].sum())
R['fp_profile'] = dict(n_fp=len(fp_), n_tn=len(tn_),
                       fp_k1=int((kcount['ICH'][fp_.index] == 1).sum()), tn_k1=int((kcount['ICH'][tn_.index] == 1).sum()),
                       fp_fracture=cnt(fp_, 'ref_Fracture'), tn_fracture=cnt(tn_, 'ref_Fracture'),
                       fp_chronic=cnt(fp_, 'ref_ChronicBleed'), tn_chronic=cnt(tn_, 'ref_ChronicBleed'),
                       fp_masseffect=cnt(fp_, 'ref_MassEffect'), tn_masseffect=cnt(tn_, 'ref_MassEffect'),
                       fp_B1=int((fp_.Category == 'B1').sum()), tn_B1=int((tn_.Category == 'B1').sum()),
                       fp_thin=int((fp_.dz <= 1.25).sum()), tn_thin=int((tn_.dz <= 1.25).sum()),
                       fp_vol_median=float(fp_.pred_volume_ml.median()), fp_vol_q=np.percentile(fp_.pred_volume_ml, [25, 75]).tolist(),
                       fp_over10=int((fp_.pred_volume_ml >= 10).sum()), fp_components_median=float(fp_.pred_components.median()),
                       tp_components_median=float(m[(m.ref_ICH == 1) & pI].pred_components.median()))
R['fn_profile'] = dict(n_fn=len(fn_), by_subtype={c: int(fn_['ref_' + c].sum()) for c in SUB + ['ChronicBleed']},
                       fn_k2=int((kcount['ICH'][fn_.index] == 2).sum()), fn_zero_volume=int((fn_.pred_volume_ml == 0).sum()),
                       fn_vol_median=float(fn_.pred_volume_ml.median()))
# chronic bleed relationship with ICH label
R['chronic_vs_ich'] = dict(chronic_total=int(m.ref_ChronicBleed.sum()), chronic_ich_pos=int(((m.ref_ChronicBleed == 1) & (m.ref_ICH == 1)).sum()))
# Fisher tests FP vs TN for features
fis = {}
for col in ['ref_Fracture', 'ref_ChronicBleed', 'ref_MassEffect']:
    tab = [[cnt(fp_, col), len(fp_) - cnt(fp_, col)], [cnt(tn_, col), len(tn_) - cnt(tn_, col)]]
    fis[col] = stats.fisher_exact(tab)
tab = [[int((kcount['ICH'][fp_.index] == 1).sum()), int((kcount['ICH'][fp_.index] == 0).sum())],
       [int((kcount['ICH'][tn_.index] == 1).sum()), int((kcount['ICH'][tn_.index] == 0).sum())]]
fis['one_reader_positive'] = stats.fisher_exact(tab)
R['fp_fisher'] = {k_: (v.statistic, v.pvalue) for k_, v in fis.items()}

# ------------------------------------------------------------------ 12. Midline-shift proxy
ok = np.isfinite(m.csf_offset_maxabs_mm.values)
mp = m[ok].copy()
R['proxy'] = dict(n=int(ok.sum()), pos=int(mp.ref_MidlineShift.sum()),
                  median_pos=float(mp.loc[mp.ref_MidlineShift == 1, 'csf_offset_maxabs_mm'].median()),
                  iqr_pos=np.percentile(mp.loc[mp.ref_MidlineShift == 1, 'csf_offset_maxabs_mm'], [25, 75]).tolist(),
                  median_neg=float(mp.loc[mp.ref_MidlineShift == 0, 'csf_offset_maxabs_mm'].median()),
                  iqr_neg=np.percentile(mp.loc[mp.ref_MidlineShift == 0, 'csf_offset_maxabs_mm'], [25, 75]).tolist(),
                  neg_over5=int((mp.loc[mp.ref_MidlineShift == 0, 'csf_offset_maxabs_mm'] > 5).sum()),
                  neg_n=int((mp.ref_MidlineShift == 0).sum()),
                  pos_over15=int((mp.loc[mp.ref_MidlineShift == 1, 'csf_offset_maxabs_mm'] > 15).sum()),
                  max=float(mp.csf_offset_maxabs_mm.max()),
                  mls_not_computable=int(((~ok) & (m.ref_MidlineShift == 1)).sum()))
R['proxy']['auc_all'] = boot_auc(mp.ref_MidlineShift.values, mp.csf_offset_maxabs_mm.values)
for nmx, sel in [('ich_pos', mp.ref_ICH == 1), ('ich_neg', mp.ref_ICH == 0)]:
    yy = mp.ref_MidlineShift.values[sel.values]; ss = mp.csf_offset_maxabs_mm.values[sel.values]
    R['proxy'][f'auc_{nmx}'] = boot_auc(yy, ss) if 0 < yy.sum() < len(yy) else (np.nan,) * 3
    R['proxy'][f'n_{nmx}'] = (int(len(yy)), int(yy.sum()))
    R['proxy'][f'auc_vol_{nmx}'] = boot_auc(yy, mp.pred_volume_ml.values[sel.values]) if 0 < yy.sum() < len(yy) else (np.nan,) * 3
k_mls = kcount['MidlineShift'][ok]
R['proxy']['by_k'] = [dict(k=kk, n=int((k_mls == kk).sum()), median=float(np.median(mp.csf_offset_maxabs_mm.values[k_mls == kk]))) for kk in range(4)]
R['proxy']['tau_k'] = stats.kendalltau(k_mls, mp.csf_offset_maxabs_mm.values).statistic
# incremental value (exploratory, repeated CV within CQ500)
Xv = np.log1p(mp[['pred_volume_ml']].values); Xb = np.c_[Xv, mp.csf_offset_maxabs_mm.values]
yv = mp.ref_MidlineShift.values
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=50, random_state=20261026)
av, ab = [], []
for tr, te in cv.split(Xv, yv):
    av.append(fast_auc(yv[te], LogisticRegression(max_iter=1000).fit(Xv[tr], yv[tr]).predict_proba(Xv[te])[:, 1]))
    ab.append(fast_auc(yv[te], LogisticRegression(max_iter=1000).fit(Xb[tr], yv[tr]).predict_proba(Xb[te])[:, 1]))
R['proxy']['cv_auc_volume_only'] = float(np.mean(av)); R['proxy']['cv_auc_volume_plus_proxy'] = float(np.mean(ab))
R['proxy']['cv_delta_q'] = np.percentile(np.array(ab) - np.array(av), [2.5, 50, 97.5]).tolist()
# paired DeLong proxy vs volume for MLS on the 452 scans
R['proxy']['delong_proxy_vs_volume'] = delong_paired(yv, mp.csf_offset_maxabs_mm.values, mp.pred_volume_ml.values)
yme = mp.ref_MassEffect.values
R['proxy']['delong_proxy_vs_volume_masseffect'] = delong_paired(yme, mp.csf_offset_maxabs_mm.values, mp.pred_volume_ml.values)

save_json(R, OUT / 'new_analyses_cq500.json')
print(json.dumps({k_: R[k_] for k_ in ['n_needed', 'batch_n', 'batch_prev', 'fp_profile', 'fn_profile', 'proxy', 'delong_v3b_vs_v2', 'delong_v3b_vs_v2_excl', 'n_subtypes_per_ich_scan', 'ich_without_any_subtype', 'chronic_vs_ich']}, default=str, indent=1))
print(kap.round(3).to_string())
print(loo.round(3).to_string())
print(pd.DataFrame(amb).round(3).to_string())
for k_ in ['amb_tau_ICH', 'amb_tau_ICH_v3b', 'amb_auc_ICH', 'amb_auc_ICH_v3b', 'amb_trend_v2']:
    print(k_, R[k_])
print(pd.DataFrame(subun).to_string())
print(pd.DataFrame(spec_rows).round(3).to_string())
print(R['batch_auc_diff_v2'], R['batch_auc_diff_v3b'])
print(pd.DataFrame(acq).round(3).to_string())
print(R['series_CT_C'], R['neg_vol_by_dz_kruskal'])
print(pd.read_csv(OUTT / 'cq500_subtype_decomposition.csv').to_string())
print(co)
print(pd.DataFrame(ev).round(3).to_string())
print(pd.DataFrame(thr_rows).round(3).to_string())
print({k_: R[k_] for k_ in ['sens_at_spec_v2', 'sens_at_spec_v3b', 'delong_v2_ich', 'delong_v3b_ich', 'fp_fisher']})
