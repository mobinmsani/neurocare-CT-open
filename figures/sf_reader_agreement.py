"""SF_reader_agreement - Structure of reader agreement on CQ500 and model agreement against the human benchmark.

a pairwise Cohen kappa by reader pair (491 released scans); b positive-call frequency of each reader; c model AUC in all and in
unanimously labelled scans (descriptive: the unanimous subset is selected); d model-reader versus reader-reader kappa at the
1 mL rule; e each reader against the concordant reads of the other two, with denominators.
All values from out/tables/cq500_reader_agreement_491.csv, out/tables/cq500_model_reader_kappa.csv, new_analyses_cq500.json
(loo, sub_unanimous, amb_auc_*), verify_original.json (overall AUCs as in the main tables) and the released reads.
"""
from nc_s1 import *
from matplotlib.colors import LinearSegmentedColormap

H = 230
fig = plt.figure(figsize=(W2, H * MM))

kap = pd.read_csv(OUT / 'tables/cq500_reader_agreement_491.csv').set_index('label').loc[ORDER14].reset_index()
reads = pd.read_csv(E / 'cq500_external/reads.csv')
assert len(reads) == 491 and len(kap) == 14
nL = len(kap)
LX = 33.0          # right edge of the row-label column (mm)

# ======================================================================= a: pairwise Cohen kappa
YA, RH = 17.0, 4.3
axA = axes_mm(fig, LX + 2, YA, 30, RH * nL)
pairs = [('cohen_R1R2', 'R1–R2'), ('cohen_R2R3', 'R2–R3'), ('cohen_R3R1', 'R3–R1')]
K = kap[[p for p, _ in pairs]].values
cmap = LinearSegmentedColormap.from_list('kgray', ['#f7f6f3', '#d9d5cc', '#8c8780', '#3c3a37', '#1d1d1f'])
axA.imshow(K, cmap=cmap, vmin=0, vmax=0.85, aspect='auto', interpolation='nearest', extent=(-0.5, 2.5, nL - 0.5, -0.5))
for i in range(nL):
    for j in range(3):
        v = K[i, j]
        axA.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=T7, color=WHITE if v > 0.45 else INK)
for x_ in (0.5, 1.5):
    axA.axvline(x_, color=WHITE, lw=1.2)
for y_ in np.arange(0.5, nL - 0.5):
    axA.axhline(y_, color=WHITE, lw=1.2)
axA.set_xticks(range(3), [l for _, l in pairs])
axA.xaxis.tick_top(); axA.tick_params(axis='x', length=0, pad=1.5, labelcolor=INK)
axA.set_yticks([])
for s_ in axA.spines.values():
    s_.set_visible(False)
for i, r in kap.iterrows():
    fig_text(fig, LX, YA + RH * (i + 0.5) + 1.2, lname(r.label), H, color=INK, ha='right')
cax = axes_mm(fig, LX + 2, YA + RH * nL + 3.0, 30, 2.0)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 0.85), cmap=cmap), cax=cax, orientation='horizontal')
cb.set_ticks([0, 0.4, 0.8], labels=['0', '0.4', '0.8']); cb.ax.tick_params(length=2.0, pad=1.2, width=1.0); cb.outline.set_linewidth(1.0)
cax.set_xlabel("Cohen’s κ", labelpad=1.5)
ptitle(fig, axA, 'a', 'Agreement by reader pair', dx_mm=LX + 2 - 3.0, dy_mm=6.0)
fig_text(fig, 3.0, YA + RH * nL + 15.0, DG + ' Label tied to a surgical criterion', H, color=INK2, ha='left')

# ======================================================================= b: positive-call frequency
axB = axes_mm(fig, 76, YA, 62, RH * nL)
cnt = {}
for i, r in kap.iterrows():
    c = [int(reads[f'R{q}:{r.label}'].sum()) for q in (1, 2, 3)]
    cnt[r.label] = c
    maj = int(r.n2 + r.n3)
    assert maj == int(r.majority_pos)
    if i % 2 == 0:
        axB.axhspan(i - 0.5, i + 0.5, color=BAND, lw=0, zorder=0)
    axB.plot([min(c), max(c)], [i, i], color='#cfcbc3', lw=2.4, zorder=1, solid_capstyle='butt')
    axB.plot([maj], [i], marker='|', ms=7.5, mew=1.4, color=INK2, zorder=2)
    for q, cc in zip(('R1', 'R2', 'R3'), c):
        axB.plot([cc], [i + {'R1': -0.2, 'R2': 0.0, 'R3': 0.2}[q]], marker='D', ms=3.6, ls='', zorder=3, **RD[q])
    lo_, hi_ = min(c), max(c)
    if lo_ == 0:
        txt = f'undefined (R3: 0; max {hi_})'
    elif hi_ / lo_ >= 10:
        txt = f'{hi_ / lo_:.0f}× ({hi_} vs {lo_})'
    else:
        txt = f'{hi_ / lo_:.1f}×'
    axB.text(1.02, i, txt, transform=axB.get_yaxis_transform(), fontsize=T7, va='center', ha='left', color=INK2)
axB.set_ylim(nL - 0.5, -0.5); axB.set_yticks([]); axB.spines['left'].set_visible(False)
axB.set_xlim(-5, 245); axB.set_xticks([0, 50, 100, 150, 200])
axB.set_xlabel('Scans called positive (of 491)')
axB.text(1.02, -0.85, 'Highest/lowest', transform=axB.get_yaxis_transform(), fontsize=T7, ha='left', va='bottom', color=INK2)
grid(axB, 'x')
axB.legend(handles=reader_handles() + [Line2D([], [], marker='|', ms=7.5, mew=1.4, ls='', color=INK2, label='Reader majority')],
           loc='upper left', bbox_to_anchor=(-0.02, -0.13), ncol=4, columnspacing=1.0, handletextpad=0.3, borderaxespad=0)
ptitle(fig, axB, 'b', 'Readers differ in positive-call frequency', dx_mm=0.0, dy_mm=6.0)

# ======================================================================= c, d: shared rows (score per row)
su = {r['sub']: r for r in N['sub_unanimous']}
t9 = [r for r in VO['cq_det_table9'] if r[0] == 'ICH'][0]
v3 = VO['cq_v3b']['all']
amb2, amb3 = N['amb_auc_ICH'], N['amb_auc_ICH_v3b']
assert [round(x, 3) for x in amb2['unanimous']] == [0.842, 0.800, 0.881] and amb2['n_un'] == 411
kp = pd.read_csv(OUT / 'tables/cq500_model_reader_kappa.csv')
ROWS = [('ICH', 'v2', 'v2 total', (t9[1], t9[2], t9[3]), amb2['unanimous'], amb2['n_un'], amb2['pos_un'], ('ICH', 'pred_volume_ml')),
        ('ICH', 'v3b', 'v3b any class', (v3['ANY']['auc'], *v3['ANY']['auc_ci']), amb3['unanimous'], amb3['n_un'], amb3['pos_un'], ('ICH', 'ANY_ml'))]
for sub in SUB:
    r = su[sub]
    assert abs(r['auc'][0] - v3[sub]['auc']) < 1e-12
    ROWS.append((sub, 'v3b', 'v3b class', (v3[sub]['auc'], *v3[sub]['auc_ci']), r['auc_un'], r['n_un'], r['pos_un'], (sub, f'{sub}_ml')))
nR = len(ROWS)
YC, RC = 106.0, 5.6
axC = axes_mm(fig, LX + 2, YC, 50, RC * nR)
axD = axes_mm(fig, 117, YC, 54, RC * nR)
band_rows([axC, axD], nR)
for i, (lab, mk, sc, a, u, nun, pun, kkey) in enumerate(ROWS):
    fig_text(fig, LX, YC + RC * (i + 0.5) + 1.2, ('Any ICH' if lab == 'ICH' else lab + (DG if lab in SURG else '')) + ' · ' + sc, H, color=INK, ha='right')
    # c: AUC in all scans (open) and in the unanimous subset (filled)
    ci_mark(axC, a[0], i - 0.17, a[1], min(a[2], 1), mk, filled=False, ms=4.2)
    ci_mark(axC, u[0], i + 0.17, u[1], min(u[2], 1), mk, filled=True, ms=4.2)
    axC.text(1.02, i, f'{pun}/{nun}', transform=axC.get_yaxis_transform(), fontsize=T7, ha='left', va='center', color=INK2)
    # d: reader-reader versus model-reader kappa at 1 mL
    rk = kp[(kp.label == kkey[0]) & (kp.score == kkey[1])].iloc[0]
    axD.plot([rk.model_reader_mean, rk.reader_reader_mean], [i, i], color='#dedad2', lw=3.2, zorder=1, solid_capstyle='butt')
    axD.plot([rk.rr_min, rk.rr_max], [i - 0.15] * 2, color=INK, lw=1.2, zorder=3, solid_capstyle='butt')
    axD.plot([rk.reader_reader_mean], [i - 0.15], marker='D', ms=4.0, mfc=INK, mec=INK, mew=1.0, ls='', zorder=4)
    ci_mark(axD, rk.model_reader_mean, i + 0.15, rk.mr_min, rk.mr_max, mk, filled=True, ms=4.2)
axC.text(1.02, -0.62, 'Unanimous\nsubset', transform=axC.get_yaxis_transform(), fontsize=T7, ha='left', va='bottom', color=INK2, linespacing=1.05)
for ax in (axC, axD):
    ax.set_ylim(nR - 0.5, -0.5); ax.set_yticks([]); ax.spines['left'].set_visible(False)
axC.set_xlim(0.55, 1.005); axC.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0], ['0.6', '0.7', '0.8', '0.9', '1'])
axC.set_xlabel('AUC (95% CI)')
grid(axC, 'x')
axD.set_xlim(0, 1.0); axD.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
axD.set_xlabel('κ at the 1 mL rule (mean and range)')
grid(axD, 'x')
axC.legend(handles=[Line2D([], [], marker='o', ms=4.2, ls='', mfc=WHITE, mec=INK2, mew=1.0, label='All 489 scans (open)'),
                    Line2D([], [], marker='o', ms=4.2, ls='', mfc=INK2, mec=INK2, mew=1.0, label='Unanimous subset (filled)')],
           loc='upper left', bbox_to_anchor=(-0.02, -0.17), ncol=2, columnspacing=1.0, handletextpad=0.3, borderaxespad=0)
axD.legend(handles=[Line2D([], [], marker='D', ms=4.0, ls='-', lw=1.2, color=INK, mfc=INK, mec=INK, label='Reader vs reader (3 pairs)'),
                    Line2D([], [], marker='o', ms=4.2, ls='-', lw=1.2, color=M['v2'], mfc=M['v2'], mec=M['v2'], label='v2 vs readers'),
                    Line2D([], [], marker='s', ms=4.2, ls='-', lw=1.2, color=M['v3b'], mfc=M['v3b'], mec=M['v3b'], label='v3b vs readers')],
           loc='upper left', bbox_to_anchor=(-0.02, -0.17), ncol=2, columnspacing=0.9, handletextpad=0.3, borderaxespad=0)
ptitle(fig, axC, 'c', 'AUC in all and in unanimously labeled scans', sub='Selected subsets (split scans removed): descriptive only',
       dx_mm=LX + 2 - 3.0, dy_mm=3.0)
ptitle(fig, axD, 'd', 'Model–reader versus reader–reader κ', sub='Reader–reader κ: human agreement benchmark',
       dx_mm=0.0, dy_mm=3.0)

# ======================================================================= e: each reader against the concordant other two
loo = pd.DataFrame(N['loo'])
labsE = ['ICH', 'EDH', 'IPH', 'IVH', 'SAH', 'SDH', 'MassEffect', 'MidlineShift']
YE, RE_ = 179.0, 5.2
axS = axes_mm(fig, LX + 2, YE, 54, RE_ * len(labsE))
axP = axes_mm(fig, 99, YE, 54, RE_ * len(labsE))
band_rows([axS, axP], len(labsE))
offs = {'R1': -0.24, 'R2': 0.0, 'R3': 0.24}
for i, lab in enumerate(labsE):
    sub = loo[(loo.label == lab) & (loo.score == lab)].set_index('reader').loc[['R1', 'R2', 'R3']]
    assert len(sub) == 3
    for q, r in sub.iterrows():
        for ax, est, ci in [(axS, r.reader_sens, r.reader_sens_ci), (axP, r.reader_spec, r.reader_spec_ci)]:
            ax.plot(ci, [i + offs[q]] * 2, color=INK2, lw=1.0, zorder=2)
            ax.plot([est], [i + offs[q]], marker='D', ms=3.4, ls='', zorder=3, **RD[q])
    fig_text(fig, LX, YE + RE_ * (i + 0.5) + 1.2, lname(lab), H, color=INK, ha='right')
    pos = ' · '.join(str(int(v)) for v in sub.pos)
    neg = ' · '.join(str(int(v)) for v in (sub.n - sub.pos))
    fig_text(fig, 156.5, YE + RE_ * (i + 0.5) - 0.2, pos, H, color=INK, ha='left')
    fig_text(fig, 156.5, YE + RE_ * (i + 0.5) + 2.6, neg, H, color=INK2, ha='left')
fig_text(fig, 156.5, YE - 4.0, 'Positives', H, color=INK, ha='left')
fig_text(fig, 156.5, YE - 1.2, 'negatives', H, color=INK2, ha='left')
ich = loo[(loo.label == 'ICH') & (loo.score == 'ICH')]
assert (round(ich.reader_sens.min(), 2), round(ich.reader_sens.max(), 2), round(ich.reader_spec.min(), 2), round(ich.reader_spec.max(), 2)) == (0.90, 0.98, 0.89, 0.98)
for ax, lab_, lo_ in [(axS, 'Sensitivity (95% CI)', 0.25), (axP, 'Specificity (95% CI); axis starts at 0.75', 0.75)]:
    ax.set_ylim(len(labsE) - 0.5, -0.5); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xlim(lo_, 1.005)
    ax.set_xlabel(lab_)
    grid(ax, 'x')
axS.set_xticks([0.25, 0.5, 0.75, 1.0], ['0.25', '0.50', '0.75', '1'])
axP.set_xticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0], ['0.75', '0.80', '0.85', '0.90', '0.95', '1'])
axP.legend(handles=reader_handles(3.4), loc='lower right', bbox_to_anchor=(1.0, 1.02), ncol=3, columnspacing=0.9, handletextpad=0.2, borderaxespad=0)
ptitle(fig, axS, 'e', 'Each reader against the concordant reads of the other two',
       sub='Subset: scans on which the other two agreed (differs by reader)', dx_mm=LX + 2 - 3.0, dy_mm=3.0)

print('overflow', overflow_mm(fig))
for lab in ORDER14:
    print(lab, cnt[lab])
for row in ROWS:
    print(row[0], row[2], [round(x, 3) for x in row[3]], [round(x, 3) for x in row[4]], row[5], row[6])
save(fig, 'SF_reader_agreement', 'supp')
