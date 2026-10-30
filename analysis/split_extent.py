"""Post hoc: are positives decided 2-1 missed more often only because they are smaller? Model-free extent from the BHX boxes
(bhx_extent.py) is compared between split and unanimous majority positives, and detection by each model is compared between them
within tertiles of extent (Mantel-Haenszel odds ratio with the Robins-Breslow-Greenland interval; Cochran-Mantel-Haenszel test).
Models: v2 (total predicted volume >= 1.0 mL, locked evaluation) and the public model (probability for any hemorrhage >= 0.5).
Output: out/split_extent.json, out/tables/split_extent.csv"""
import numpy as np
import pandas as pd
from scipy import stats

from common import E, OUT, Z, load_cq500, save_json, wilson

m = load_cq500()
ext = pd.read_csv(E / 'bhx_scan_labels' / 'bhx_scan_extent.csv')
m = m.merge(ext, left_on='name', right_on='scan', how='left')
pmp = E / 'public_model_cq500' / 'public_model_per_scan.csv'
pm = pd.read_csv(pmp)[['scan', 'p_any']].rename(columns={'scan': 'name'})
m = m.merge(pm, on='name', how='left')
m['votes'] = m['R1:ICH'] + m['R2:ICH'] + m['R3:ICH']
m['n_sub'] = m[[f'ref_{s}' for s in ['EDH', 'IPH', 'IVH', 'SAH', 'SDH']]].sum(axis=1)
m['cc_mm'] = m.bhx_extent_slices * m.dz
pos = m[m.ref_ICH == 1].copy()
pos['split'] = (pos.votes == 2).astype(int)
withbox = pos[pos.bhx_extent_ml > 0].copy()


def mh(tabs):
    """tabs: list of (a, b, c, d) = (split detected, split missed, unanimous detected, unanimous missed)."""
    R = S = 0.0
    PR = PS_QR = QS = 0.0
    num = var = 0.0
    for a, b, c, d in tabs:
        n = a + b + c + d
        if n < 2 or (a + b) == 0 or (c + d) == 0:
            continue
        r, s = a * d / n, b * c / n
        P, Q = (a + d) / n, (b + c) / n
        R += r; S += s; PR += P * r; PS_QR += P * s + Q * r; QS += Q * s
        m1, n1, n0 = a + c, a + b, c + d
        num += a - n1 * m1 / n
        var += n1 * n0 * m1 * (n - m1) / (n * n * (n - 1))
    chi = (abs(num) - 0.5) ** 2 / var if var > 0 else float('nan')
    p_cmh = float(stats.chi2.sf(chi, 1)) if var > 0 else float('nan')
    if R == 0 or S == 0:
        return dict(or_mh=float('nan'), lo=float('nan'), hi=float('nan'), p_cmh=p_cmh, note='odds ratio not estimable (a zero margin in every stratum)')
    orr = R / S
    se = np.sqrt(PR / (2 * R * R) + PS_QR / (2 * R * S) + QS / (2 * S * S))
    return dict(or_mh=orr, lo=float(np.exp(np.log(orr) - Z * se)), hi=float(np.exp(np.log(orr) + Z * se)), p_cmh=p_cmh)


res = dict(n_pos=int(len(pos)), n_pos_with_boxes=int(len(withbox)), n_split_with_boxes=int(withbox.split.sum()),
           n_split=int(pos.split.sum()), source=withbox.bhx_extent_source.value_counts().to_dict())
rows = []
for grp_name, g in [('all majority positives', withbox), ('single subtype', withbox[withbox.n_sub == 1])]:
    sp, un = g[g.split == 1], g[g.split == 0]
    d = dict(group=grp_name, n_split=int(len(sp)), n_unan=int(len(un)),
             ext_ml_split=[float(x) for x in sp.bhx_extent_ml.quantile([.25, .5, .75])],
             ext_ml_unan=[float(x) for x in un.bhx_extent_ml.quantile([.25, .5, .75])],
             cc_mm_split=[float(x) for x in sp.cc_mm.quantile([.25, .5, .75])],
             cc_mm_unan=[float(x) for x in un.cc_mm.quantile([.25, .5, .75])],
             p_mw_ext=float(stats.mannwhitneyu(sp.bhx_extent_ml, un.bhx_extent_ml).pvalue) if len(sp) and len(un) else float('nan'))
    cuts = g.bhx_extent_ml.quantile([1 / 3, 2 / 3]).values
    g = g.assign(tert=np.digitize(g.bhx_extent_ml, cuts))
    for model, col, thr in [('v2', 'pred_volume_ml', 1.0), ('public model', 'p_any', 0.5)]:
        gg = g[g[col].notna()]
        if not len(gg):
            continue
        tabs = []
        for t in range(3):
            q = gg[gg.tert == t]
            det = q[col] >= thr
            a, b = int((det & (q.split == 1)).sum()), int((~det & (q.split == 1)).sum())
            c, dd = int((det & (q.split == 0)).sum()), int((~det & (q.split == 0)).sum())
            tabs.append((a, b, c, dd))
            rows.append(dict(group=grp_name, model=model, tertile=t + 1, extent_ml_range=f'{q.bhx_extent_ml.min():.1f}–{q.bhx_extent_ml.max():.1f}' if len(q) else '',
                             split_detected=a, split_n=a + b, unanimous_detected=c, unanimous_n=c + dd,
                             sens_split=a / (a + b) if a + b else float('nan'), sens_unan=c / (c + dd) if c + dd else float('nan')))
        r = mh(tabs)
        sa = sum(x[0] for x in tabs); sn = sum(x[0] + x[1] for x in tabs)
        ua = sum(x[2] for x in tabs); un_ = sum(x[2] + x[3] for x in tabs)
        sb, ub = sn - sa, un_ - ua
        crude = (sa * ub) / (sb * ua) if sb * ua > 0 else float('nan')
        cse = np.sqrt(1 / sa + 1 / sb + 1 / ua + 1 / ub) if min(sa, sb, ua, ub) > 0 else float('nan')
        d[model] = dict(n=int(len(gg)), split=f'{sa} of {sn}', unanimous=f'{ua} of {un_}', sens_split=sa / sn if sn else float('nan'),
                        sens_unan=ua / un_ if un_ else float('nan'), sens_split_ci=list(wilson(sa, sn)), sens_unan_ci=list(wilson(ua, un_)),
                        or_crude=crude, or_crude_ci=[float(np.exp(np.log(crude) - Z * cse)), float(np.exp(np.log(crude) + Z * cse))] if np.isfinite(cse) else [float('nan')] * 2, **r)
    res[grp_name] = d
pd.DataFrame(rows).to_csv(OUT / 'tables' / 'split_extent.csv', index=False)
save_json(res, OUT / 'split_extent.json')
for k in ('all majority positives', 'single subtype'):
    d = res[k]
    print(k, 'split', d['n_split'], 'unan', d['n_unan'], 'extent mL median split', round(d['ext_ml_split'][1], 1), 'unan', round(d['ext_ml_unan'][1], 1),
          'P', f"{d['p_mw_ext']:.2g}", '| cc mm', d['cc_mm_split'][1], d['cc_mm_unan'][1])
    for model in ('v2', 'public model'):
        if model in d:
            x = d[model]
            print('  ', model, x['split'], x['unanimous'], 'crude OR', round(x['or_crude'], 2), 'MH OR', round(x['or_mh'], 2), round(x['lo'], 2), round(x['hi'], 2), 'P', f"{x['p_cmh']:.2g}")
