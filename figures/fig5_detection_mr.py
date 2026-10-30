"""Figure 5 - Detection on CQ500 depends on reference certainty and sampling, for the locked v2 probe
and a strong, independently developed public model (RSNA 2019-trained; plan locked before its evaluation outputs, except a timing run on one scan).

a ROC of both models against the reader majority, with operating points and reader points (subset references)
b share of scans called positive by the number of readers calling ICH (both models)
c sensitivity by vote split, overall and by the number of reader-majority subtypes (both models)
d batch estimates (B1 consecutive, B2 enriched), both models
e PPV implied by each model's fixed batch operating points; B2 operating point at B1 prevalence (stars)
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nc_b import *
from ncdata import *
from sklearn.metrics import roc_curve, roc_auc_score

PMC, PMLS, PMMK = '#6a3d9a', (0, (5.5, 1.5, 1.2, 1.5)), 'h'
HFIG = 162
fig = plt.figure(figsize=(W2, HFIG * MM))
m = load_cq500()
pm = pd.read_csv(E / 'public_model_cq500' / 'public_model_per_scan.csv')
m = m.merge(pm[['scan', 'p_any']], left_on='name', right_on='scan', how='left')
FULL = m.p_any.notna().all()
mm_ = m[m.p_any.notna()].copy()
PMJ = json.load(open(OUT / 'public_model_cq500.json'))
assert PMJ['n_scans'] == len(mm_), ('public-model JSON and per-scan file come from different snapshots', PMJ['n_scans'], len(mm_))
y = m.ref_ICH.values.astype(int)
s2 = m.pred_volume_ml.values
k_all = m[['R1:ICH', 'R2:ICH', 'R3:ICH']].sum(axis=1).astype(int).values
yp = mm_.ref_ICH.values.astype(int); sp_ = mm_.p_any.values
kp = mm_[['R1:ICH', 'R2:ICH', 'R3:ICH']].sum(axis=1).astype(int).values
print('public model scans', len(mm_), 'of', len(m))

# ============================================================================ a: ROC
axA = axes_mm(fig, 13, 9, 52, 52)
axA.plot([0, 1], [0, 1], color='#cfccc5', lw=1.0, ls=(0, (3, 2)), zorder=1)
f, t, _ = roc_curve(y, s2)
axA.plot(f, t, color=M['v2'], lw=1.5, ls=LS['v2'], zorder=3, solid_joinstyle='round')
se2 = ((s2 >= 1) & (y == 1)).sum() / (y == 1).sum(); sp2 = ((s2 < 1) & (y == 0)).sum() / (y == 0).sum()
axA.plot([1 - sp2], [se2], marker=MK['v2'], ms=5.0, mfc=WHITE, mec=M['v2'], mew=1.3, ls='', zorder=6)
f, t, _ = roc_curve(yp, sp_)
axA.plot(f, t, color=PMC, lw=1.5, ls=PMLS, zorder=4, solid_joinstyle='round', dash_capstyle='butt')
sep = ((sp_ >= 0.5) & (yp == 1)).sum() / (yp == 1).sum(); spp = ((sp_ < 0.5) & (yp == 0)).sum() / (yp == 0).sum()
axA.plot([1 - spp], [sep], marker=PMMK, ms=5.4, mfc=WHITE, mec=PMC, mew=1.3, ls='', zorder=6)
loo = pd.DataFrame(N['loo'])
lr = loo[(loo.label == 'ICH') & (loo.score == 'ICH')].set_index('reader')
for rd, r in lr.iterrows():
    sl, sh = r.reader_sens_ci; pl, ph = r.reader_spec_ci
    axA.plot([1 - ph, 1 - pl], [r.reader_sens] * 2, color=INK, lw=1.0, zorder=7, solid_capstyle='butt')
    axA.plot([1 - r.reader_spec] * 2, [sl, sh], color=INK, lw=1.0, zorder=7, solid_capstyle='butt')
    axA.plot([1 - r.reader_spec], [r.reader_sens], marker=MK['reader'], ms=4.6, mfc=INK, mec=WHITE, mew=0.6, ls='', zorder=8)
a2 = [r for r in VO['cq_det_table9'] if r[0] == 'ICH'][0][1:4]
ap = PMJ['auc_all']
axA.legend(handles=legend_handles([('v2 probe, total volume (locked)', dict(color=M['v2'], marker='', ls=LS['v2'], lw=1.5)),
                                   ('Public model, probability', dict(color=PMC, marker='', ls=PMLS, lw=1.5)),
                                   ('Operating point (1.0 mL; 0.5)', dict(color=INK2, marker='o', filled=False, ms=4.8)),
                                   ('Reader (95% CI)', dict(color=INK, marker='D', ms=4.4))]),
           loc='lower right', bbox_to_anchor=(1.02, 0.30), handlelength=2.0, labelspacing=0.32, borderaxespad=0.1,
           frameon=True, facecolor=WHITE, edgecolor='none', framealpha=1.0, borderpad=0.2)
dl = PMJ['delong_public_minus_v2']
axA.text(0.985, 0.02, f"AUC (95% CI)\nv2            {fmt_ci(*a2, d=3)}\npublic model {fmt_ci(ap['auc'], ap['lo'], ap['hi'], d=3)}\n"
                      f"difference {fmt_ci(dl[0], dl[1], dl[2], d=3)}\nDeLong {fmt_p(dl[3])}",
         transform=axA.transAxes, ha='right', va='bottom', fontsize=7, color=INK, linespacing=1.22)
axA.set_xlim(-0.01, 1.01); axA.set_ylim(-0.01, 1.01); axA.set_aspect('equal')
axA.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '0.25', '0.50', '0.75', '1']); axA.set_yticks([0, 0.25, 0.5, 0.75, 1], ['0', '0.25', '0.50', '0.75', '1'])
axA.set_xlabel('1 − specificity'); axA.set_ylabel('Sensitivity')
grid(axA, 'both')
nmin, nmax = int(lr.n.min()), int(lr.n.max())
note(fig, 13, 71.0, f'Readers: each against the concordant reads of the other two\n({nmin}–{nmax} scans); not directly comparable with the curves.')
ptitle(fig, axA, 'a', 'Any ICH against the reader majority (489 scans, 205 positive)', dx_mm=11.0)

# ============================================================================ b: share called positive by vote count
axB = axes_mm(fig, 92, 9, 85, 50)
axB.axvspan(-0.5, 1.5, color=BAND, lw=0, zorder=0)
for kk in range(4):
    for off, sc, kv, thr, col, mk in [(-0.13, s2, k_all, 1.0, M['v2'], MK['v2']), (0.13, sp_, kp, 0.5, PMC, PMMK)]:
        sel = kv == kk
        pos = int((sc[sel] >= thr).sum()); n_ = int(sel.sum())
        lo, hi = wilson_ci(pos, n_)
        ci_point(axB, kk + off, pos / n_, lo, hi, col, marker=mk, filled=True, orient='v', ms=4.8 if mk == 'o' else 5.4)
        axB.text(kk + off + (0.06 if off > 0 else -0.06), pos / n_, f'{pct(pos / n_)}', fontsize=7, color=col if off > 0 else INK,
                 ha='left' if off > 0 else 'right', va='center')
axB.set_xlim(-0.5, 3.5); axB.set_ylim(0, 1.13)
axB.set_xticks(range(4), [f'{kk} of 3\n(n = {(k_all == kk).sum()})' for kk in range(4)])
axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ['0', '0.25', '0.50', '0.75', '1'])
axB.set_xlabel('Readers calling intracranial hemorrhage')
axB.set_ylabel('Share called positive (95% CI)')
grid(axB, 'y')
axB.text(0.5, 1.115, 'Reader-majority negative', fontsize=7, color=INK2, ha='center', va='top')
axB.text(2.5, 1.115, 'Reader-majority positive', fontsize=7, color=INK2, ha='center', va='top')
axB.legend(handles=legend_handles([('v2 ≥1.0 mL', dict(color=M['v2'], marker='o', ms=4.6)), ('public model ≥0.5', dict(color=PMC, marker=PMMK, ms=5.0))]),
           loc='upper left', bbox_to_anchor=(0.0, 0.92), handlelength=1.0, labelspacing=0.3, borderaxespad=0.2)
ptitle(fig, axB, 'b', 'Scans called positive by reader vote count', dx_mm=11.0)

# ============================================================================ c: sensitivity by vote split and number of subtype labels
YR2, HR2 = 93, 43
ae = pd.read_csv(OUT / 'tables/cq500_ambiguity_by_extent.csv')
ap_ = pd.DataFrame(PMJ['ambiguity']['rows'])
axC = axes_mm(fig, 6, YR2, 40, HR2)
strata = [('all', 'All majority positives'), ('single subtype', 'Single subtype'), ('two or more subtypes', 'Two or more subtypes')]
for i, (st, lab) in enumerate(strata):
    if i % 2 == 0:
        axC.axhspan(i - 0.5, i + 0.5, color=BAND, lw=0, zorder=0)
    axC.text(0.015, i - 0.45, lab, fontsize=7, color=INK, fontweight='bold', ha='left', va='top', transform=axC.get_yaxis_transform())
    for j, (vlab, filled) in enumerate([('2 of 3', False), ('3 of 3', True)]):
        for jj, (tab, col, mk) in enumerate([(ae, M['v2'], MK['v2']), (ap_, PMC, PMMK)]):
            yy = i - 0.12 + j * 0.30 + jj * 0.13
            r = tab[(tab.stratum == st) & (tab.votes == vlab)].iloc[0]
            if r.n == 0:
                if jj == 0:
                    axC.text(0.03, yy + 0.06, 'no split positive', fontsize=7, color=INK2, ha='left', va='center')
                continue
            lo, hi = wilson_ci(int(r.detected), int(r.n))
            ci_point(axC, r.detected / r.n, yy, lo, hi, col, marker=mk, filled=filled, ms=4.4 if mk == 'o' else 5.0)
        r0 = ae[(ae.stratum == st) & (ae.votes == vlab)].iloc[0]; r1 = ap_[(ap_.stratum == st) & (ap_.votes == vlab)].iloc[0]
        axC.text(1.03, i - 0.06 + j * 0.30, f'{vlab}: {int(r0.detected)}/{int(r0.n)} | {int(r1.detected)}/{int(r1.n)}' if r0.n else f'{vlab}: 0',
                 fontsize=7, color=INK, ha='left', va='center', transform=axC.get_yaxis_transform())
axC.set_yticks([]); axC.spines['left'].set_visible(False)
axC.set_ylim(len(strata) - 0.5, -0.5); axC.set_xlim(0, 1.0)
axC.set_xticks([0, 0.5, 1.0], ['0', '0.5', '1'])
axC.set_xlabel('Sensitivity (95% CI)')
grid(axC, 'x')
sd = R3['single_subtype_sens_diff']; fp_ = R3['single_subtype_fisher_p']
sdp = PMJ['ambiguity']['diff_single']; fpp = PMJ['ambiguity']['fisher_single']
note(fig, 6, 148.5, f'Single subtype, unanimous − split:\nv2 {fmt_ci(*sd)}; {fmt_p(fp_)}\npublic model {fmt_ci(*sdp)}; {fmt_p(fpp)}')
ptitle(fig, axC, 'c', 'Sensitivity by vote split and subtype count', dx_mm=4.0)

# ============================================================================ d: batch estimates
bt = {(r['model'], r['batch']): r for r in N['batches']}
b1, b2 = bt[('v2', 'B1')], bt[('v2', 'B2')]
o1, o2 = PMJ['ops_B1'], PMJ['ops_B2']
a1, a2b = PMJ['auc_B1'], PMJ['auc_B2']
axD = axes_mm(fig, 87, YR2, 26, HR2)
metrics = [('AUC', 'auc'), ('Sensitivity', 'sens'), ('Specificity', 'spec'), ('PPV', 'ppv'), ('NPV', 'npv')]


def val(model, batch, key):
    if model == 'v2':
        r = b1 if batch == 'B1' else b2
        if key == 'auc':
            return r['auc'], r['lo'], r['hi']
        return (r[key], *r[key + '_ci'])
    o = o1 if batch == 'B1' else o2
    if key == 'auc':
        a = a1 if batch == 'B1' else a2b
        return a['auc'], a['lo'], a['hi']
    return (o[key], *o[key + '_ci'])


for j, (lab, key) in enumerate(metrics):
    if j % 2 == 0:
        axD.axhspan(j - 0.5, j + 0.5, color=BAND, lw=0, zorder=0)
    txts = []
    for off, model, batch, col, mk, filled in [(-0.30, 'v2', 'B1', M['v2'], MK['v2'], False), (-0.10, 'v2', 'B2', M['v2'], MK['v2'], True),
                                                (0.10, 'pm', 'B1', PMC, PMMK, False), (0.30, 'pm', 'B2', PMC, PMMK, True)]:
        v_, lo, hi = val(model, batch, key)
        ci_point(axD, v_, j + off, lo, hi, col, marker=mk, filled=filled, ms=4.0 if mk == 'o' else 4.6)
        txts.append(f'{v_:.2f}')
    axD.text(1.04, j - 0.20, f'{txts[0]}  {txts[1]}', fontsize=7, ha='left', va='center', color=M['v2'], transform=axD.get_yaxis_transform())
    axD.text(1.04, j + 0.20, f'{txts[2]}  {txts[3]}', fontsize=7, ha='left', va='center', color=PMC, transform=axD.get_yaxis_transform())
axD.set_yticks(range(len(metrics)), [x[0] for x in metrics])
axD.tick_params(axis='y', length=0, labelcolor=INK)
axD.spines['left'].set_visible(False)
axD.set_ylim(len(metrics) - 0.5, -0.5); axD.set_xlim(0, 1.0)
axD.set_xticks([0, 0.5, 1.0], ['0', '0.5', '1'])
axD.set_xlabel('Estimate (95% CI)')
grid(axD, 'x')
axD.text(1.04, -0.78, 'B1   B2', fontsize=7, ha='left', va='center', color=INK2, transform=axD.get_yaxis_transform())
ptitle(fig, axD, 'd', 'Batch estimates', dx_mm=16.0)

# ============================================================================ e: PPV implied by fixed operating points
cf = R3['ppv_counterfactual']; cfp = PMJ['ppv_counterfactual']
axE = axes_mm(fig, 140, YR2, 37, HR2)
pi = np.linspace(0.0, 0.8, 400)


def ppv(se, sp, p):
    return se * p / (se * p + (1 - sp) * (1 - p))


LS_B1 = (0, (3.2, 1.8))
for col, mk, B1, B2, c_ in [(M['v2'], MK['v2'], b1, b2, cf), (PMC, PMMK, o1, o2, cfp)]:
    axE.plot(pi, ppv(B1['sens'], B1['spec'], pi), color=col, lw=1.5, ls=LS_B1, zorder=3)
    axE.plot(pi, ppv(B2['sens'], B2['spec'], pi), color=col, lw=1.5, zorder=3)
    p1 = B1['prev']; p2 = B2['prev']
    ci_point(axE, p1, B1['ppv'], *B1['ppv_ci'], col, marker=mk, filled=False, orient='v', ms=4.8)
    ci_point(axE, p2, B2['ppv'], *B2['ppv_ci'], col, marker=mk, filled=True, orient='v', ms=4.8)
    ci_point(axE, p1 + 0.012, c_['ppv_cf'], *c_['ppv_cf_ci'], col, marker='*', filled=True, orient='v', ms=7.0, ecolor=col)
axE.set_xlim(0, 0.8); axE.set_ylim(0, 1.0)
axE.set_xticks([0, 0.2, 0.4, 0.6, 0.8], ['0', '20', '40', '60', '80']); axE.set_yticks([0, 0.5, 1.0], ['0', '0.5', '1'])
axE.set_xlabel('Prevalence of ICH (%)'); axE.set_ylabel('PPV', labelpad=1.5)
grid(axE, 'both')
ptitle(fig, axE, 'e', 'PPV at fixed operating points', dx_mm=9.0)

hB1 = Line2D([], [], color=INK2, lw=1.5, ls=LS_B1, marker='o', ms=4.6, mfc=WHITE, mec=INK2, mew=1.0)
hB2 = Line2D([], [], color=INK2, lw=1.5, ls='-', marker='o', ms=4.6, mfc=INK2, mec=WHITE, mew=0.6)
hS = Line2D([], [], color=INK2, lw=0, marker='*', ms=7.0, mfc=INK2, mec=INK2)
fig.legend(handles=[hB1, hB2, hS], labels=[f"B1 consecutive: ICH {pct(b1['prev'])} ({b1['pos']}/{b1['n']})", f"B2 enriched: ICH {pct(b2['prev'])} ({b2['pos']}/{b2['n']})",
                                           'B2 operating point at B1 prevalence'],
           loc='upper left', bbox_to_anchor=(70 / 180, 1 - 146.5 / HFIG), ncol=1, handlelength=2.6, labelspacing=0.25, handletextpad=0.5, borderaxespad=0, borderpad=0)
txt_e = (f"PPV expected in B1 from the B2\noperating point: {cf['ppv_cf']:.2f} (v2), {cfp['ppv_cf']:.2f} (public\n"
         f"model); observed in B1: {cf['ppv_B1']:.2f} and {cfp['ppv_B1']:.2f}.")
note(fig, 131, 146.5, txt_e)
if not FULL:
    note(fig, 92, 2.0, f'PROVISIONAL: public model on {len(mm_)} of 489 scans', color=CRIT)
print('overflow', overflow_mm(fig))
save(fig, 'Figure_5_Detection', 'main_mr')
