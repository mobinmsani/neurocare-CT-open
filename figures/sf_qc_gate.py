"""SF_qc_gate - Stored-intensity quality-control gate: one BHSD test scan without air-range values and its influence on
volumetric agreement."""
from nc_s2 import *

H = 130
fig = plt.figure(figsize=(W2, H * MM))

QD = E / 'neurocare_qc_gate_20261021'
qf = pd.read_csv(QD / 'QC_features_all_188.csv')
qs = pd.read_csv(QD / 'QC_features_SegCQ500.csv')
ph = pd.read_csv(E / 'neurocare_v2_20261021/phe_sich/phe_laterality_midline.csv', dtype={'case': str})
cq = pd.read_csv(E / 'cq500_external/cq500_per_scan.csv')
qres = json.load(open(QD / 'QC_gate_results.json'))
k, v1, nm, v2b, v3b, v3 = bhsd()
bha = {(r['model'], r['cohort']): r for r in S['bhsd_agreement']}
GATE = -500
assert (qf.partition == 'train').sum() == 92 and (qf.partition == 'test').sum() == 96 and len(qs) == 43 and len(ph) == 120
assert qres['flagged_ids'] == [QCID] and int(cq.qc_flag.sum()) == 0 and len(cq) == 489 and REG['qc']['train_flag'] == 0

# ======================================================================= a: counted value clusters by cohort
cohorts = [('BHSD\ntraining', qf[qf.partition == 'train'].set_index('case_id').p01),
           ('BHSD\ntest', qf[qf.partition == 'test'].set_index('case_id').p01),
           ('Seg-\nCQ500', qs.set_index('case_id').p01),
           ('PHE-SICH-\nCT-IDS', ph.set_index('case').p01)]
clusters = [('> −500: no air-range values', lambda v: v > GATE),
            ('−1,024 to −1,000: air', lambda v: (v >= -1024) & (v <= -1000)),
            ('−2,048: padding', lambda v: v == -2048),
            ('−3,024 or −3,000: padding', lambda v: (v == -3024) | (v == -3000))]
cnt = np.zeros((len(clusters), len(cohorts)), int)
for j, (_, s_) in enumerate(cohorts):
    v = s_.values
    tot = 0
    for i, (_, f) in enumerate(clusters):
        cnt[i, j] = int(f(v).sum()); tot += cnt[i, j]
    assert tot == len(v), (j, tot, len(v))          # every scan falls in exactly one cluster
assert cnt[0].tolist() == [0, 1, 0, 0]
axA = axes_mm(fig, 40, 22, 56, 36)
axA.set_xlim(-0.5, 3.6); axA.set_ylim(3.5, -0.5)
axA.set_xticks([]); axA.set_yticks([])
for s in axA.spines.values():
    s.set_visible(False)
NMAX = cnt.max()
for j in range(len(cohorts)):
    for i in range(len(clusters)):
        n = cnt[i, j]
        if n == 0:
            axA.text(j, i, '0', fontsize=F7, ha='center', va='center', color=MUTED)
            continue
        flag = i == 0
        msz = 4.0 + 15.0 * np.sqrt(n / NMAX)          # marker diameter (pt), area proportional to count
        if flag:
            axA.plot([j - 0.12], [i], marker='*', ms=9, mfc=CRIT, mec=CRIT, mew=1.0, ls='', zorder=4)
            msz = 9
        else:
            axA.plot([j - 0.12], [i], marker='o', ms=msz, mfc='#c9c4ba', mec=INK2, mew=1.0, ls='', zorder=3)
        axA.annotate(f'{n}', (j - 0.12, i), xytext=(msz / 2 + 2.0, 0), textcoords='offset points', fontsize=F7, ha='left', va='center',
                     color=CRIT if flag else INK, fontweight='bold' if flag else 'normal', zorder=5)
for j, (lab, s_) in enumerate(cohorts):
    axA.text(j, -0.62, f'{lab}\n({len(s_)})', fontsize=F7, ha='center', va='bottom', color=INK, linespacing=1.05)
for i, (lab, _) in enumerate(clusters):
    axA.text(-0.62, i, lab, fontsize=F7, ha='right', va='center', color=CRIT if i == 0 else INK)
axA.plot([-0.55, 3.55], [0.5, 0.5], color=CRIT, lw=1.0, ls=(0, (3, 2)), clip_on=False)
axA.text(3.55, 0.47, 'gate −500: abstain above', fontsize=F7, ha='right', va='bottom', color=CRIT)
ptitle(fig, axA, 'a', 'Stored-intensity gate', sub='First-percentile stored intensity, scans per value cluster', dx_mm=37.0, dy_mm=10.6)
fig_text(fig, 3, 62.5, 'Padding values (−2,048, −3,024, −3,000) are fill values outside the field of view, not tissue modes.\n'
         'CQ500: 0 of 489 scans flagged (gate prespecified).', H, linespacing=1.25)

# ======================================================================= b: v1 volumes on the BHSD test set
axB = axes_mm(fig, 124, 22, 46, 46)
isq = np.array([c == QCID for c in k])
y = v1.reference_volume_ml.values; p = v1.prediction_volume_ml.values
off = 0.05
lo_, hi_ = 0.04, 3000
axB.plot([lo_, hi_], [lo_, hi_], color=MUTED, lw=1.0, zorder=1)
axB.scatter(y[~isq] + off, p[~isq] + off, s=12, color=M['v1'], marker=MK['v1'], alpha=0.85, lw=0, zorder=3)
axB.plot(y[isq] + off, p[isq] + off, marker='*', ms=9, mfc=CRIT, mec=CRIT, ls='', zorder=5)
qy, qp = float(y[isq][0]), float(p[isq][0])
assert abs(qp - 1382.8) < 0.05 and abs(qy - 0.11) < 0.005
axB.annotate(f'flagged scan\nreference {qy:.2f} mL\npredicted {qp:,.1f} mL', (qy + off, qp + off), xytext=(7, 2), textcoords='offset points',
             fontsize=F7, color=CRIT, va='top', ha='left', linespacing=1.1)
axB.set_xscale('log'); axB.set_yscale('log'); axB.set_xlim(lo_, hi_); axB.set_ylim(lo_, hi_)
plain_log(axB, 'x', [0.1, 1, 10, 100, 1000], ['0.1', '1', '10', '100', '1,000'])
plain_log(axB, 'y', [0.1, 1, 10, 100, 1000], ['0.1', '1', '10', '100', '1,000'])
axB.set_aspect('equal')
axB.set_xlabel('Reference volume + 0.05 (mL)'); axB.set_ylabel('v1 predicted volume + 0.05 (mL)')
a_all, a_q = bha[('v1', 'all 96')], bha[('v1', 'QC-accepted 95')]
assert abs(a_all['ICC'] - qres['all_cases']['ICC_A1']) < 1e-9 and abs(a_q['ICC'] - qres['accepted_cases']['ICC_A1']) < 1e-9
grid(axB, 'both')
ptitle(fig, axB, 'b', 'Volume outlier on BHSD', sub='v1, 96 test scans; line of equality', dx_mm=12.5, dy_mm=3.2)

# ======================================================================= c: ICC(A,1) with and without the flagged scan
axC = axes_mm(fig, 40, 88, 86, 26)
items = [('v1', 'v1', 'prespecified', v1.prediction_volume_ml, 'v1'),
         ('v2-PM', 'v2 prespecified', 'prespecified', v2b.pred_ml, 'v2pm'),
         ('v2', 'v2 no post-mask', 'post hoc', nm.pred_ml, 'v2'),
         ('v3b, any class', 'v3b (any)', 'post hoc', v3b.ANY_pred_ml, 'v3b')]
for i, (lab, key, tg, pv, mk) in enumerate(items):
    a = bha[(key, 'all 96')]; b = bha[(key, 'QC-accepted 95')]
    col = M[mk]
    ya, yb = i - 0.17, i + 0.17
    ci_point(axC, a['ICC'], ya, a['ICC_ci'][0], a['ICC_ci'][1], col, marker=MK[mk], filled=False, ms=4.6, lw=1.2)
    ci_point(axC, b['ICC'], yb, b['ICC_ci'][0], b['ICC_ci'][1], col, marker=MK[mk], filled=True, ms=4.8, lw=1.6)
    axC.text(-0.02, i, f'{lab} ({tg})', transform=axC.get_yaxis_transform(), fontsize=F7, ha='right', va='center', color=INK)
    axC.text(1.03, i, nminus(f"{a['ICC']:.3f} → {b['ICC']:.3f}"), transform=axC.get_yaxis_transform(), fontsize=F7, ha='left', va='center', color=INK)
    axC.text(1.62, i, f'{float(pv.loc[QCID]):,.1f}', transform=axC.get_yaxis_transform(), fontsize=F7, ha='right', va='center',
             color=CRIT if float(pv.loc[QCID]) > 100 else INK)
axC.text(1.03, -0.95, 'ICC, all → gated', transform=axC.get_yaxis_transform(), fontsize=F7, ha='left', va='center', color=INK2, fontweight='bold')
axC.text(1.62, -0.95, 'Flagged scan (mL)', transform=axC.get_yaxis_transform(), fontsize=F7, ha='right', va='center', color=INK2, fontweight='bold')
axC.set_ylim(3.55, -0.6); axC.set_yticks([]); axC.spines['left'].set_visible(False)
axC.set_xlim(-0.1, 1.0); axC.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
axC.axvline(0, color=MUTED, lw=1.0, zorder=1)
axC.set_xlabel('Volume ICC(A,1), BHSD test (95% CI)')
grid(axC, 'x')
ptitle(fig, axC, 'c', 'Influence of the flagged scan', dx_mm=37.0, dy_mm=6.3)
key_row(fig, 40, 84.0, [('marker', INK2, 'all 96 scans', dict(filled=False, marker='o')), ('marker', INK2, 'QC-gated 95 scans', dict(marker='o'))], H)
fig_text(fig, 3, 123.0, 'Gate rule (abstain when the first-percentile stored intensity exceeds −500, i.e. no air-range values) was fixed after the v1 failure on this '
         'BHSD test scan,\nso its BHSD evaluation is post hoc; it was prespecified for v2 and for CQ500. Marker shape identifies the model.',
         H, linespacing=1.25)

print('cluster counts\n', pd.DataFrame(cnt, index=[c[0] for c in clusters], columns=[c[0].replace('\n', ' ') for c in cohorts]))
for lab, key, *_ in items:
    a = bha[(key, 'all 96')]; b = bha[(key, 'QC-accepted 95')]
    print(lab, 'all', round(a['ICC'], 3), np.round(a['ICC_ci'], 3), 'gated', round(b['ICC'], 3), np.round(b['ICC_ci'], 3))
print('overflow', overflow_mm(fig))
save(fig, 'SF_qc_gate', 'supp')
check_qa('SF_qc_gate')
