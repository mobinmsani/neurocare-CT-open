"""SF_segcq500_reference - Seg-CQ500 reference masks against CQ500 reader labels, and v2 volumes from two pipelines.

a the same model (v2) run through the Seg-CQ500 files and through the locked CQ500 DICOM pipeline (43 scans);
b v2 volume against the Seg-CQ500 reference volume, ICC with and without CT-108 (95% CIs from the stored analyses);
c Seg-CQ500 reference volume by the number of CQ500 readers calling ICH.
CT-108 and CT-187 are discordant reference sources (Seg-CQ500 mask, CQ500 majority negative); no attribution of error is made.
"""
from nc_s1 import *

H = 84
fig = plt.figure(figsize=(W2, H * MM))

segnm = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/segcq500_no_postmask.csv').set_index('case_id')
ks = sorted(segnm.index)
cq = load_cq500().set_index('name')
cp = cq.loc[ks]
ref = segnm.loc[ks, 'ref_ml'].values
nif = segnm.loc[ks, 'pred_ml'].values           # v2 (no post-mask) on the Seg-CQ500 files
dic = cp.pred_volume_ml.values                  # same model, locked CQ500 DICOM pipeline
kI = cp[['R1:ICH', 'R2:ICH', 'R3:ICH']].sum(axis=1).astype(int).values
cp_ = S['cross_pipeline']
assert len(ks) == 43 and f"{cp_['ICC']:.3f}" == '0.969' and f"{cp_['ICC_ci'][0]:.3f}–{cp_['ICC_ci'][1]:.3f}" == '0.937–0.985'
assert pd.Series(kI).value_counts().to_dict() == {3: 37, 2: 4, 1: 2}
flag = {'CQ500-CT-108': 'CT-108', 'CQ500-CT-187': 'CT-187'}
fi = [ks.index(c) for c in flag]
assert all(kI[i] == 1 for i in fi) and all(cp.ref_ICH.values[i] == 0 for i in fi)
assert f"{S['seg_ct108']['ref']:.1f}" == '450.8' and f"{ref[ks.index('CQ500-CT-187')]:.1f}" == '2.4'
ratio = dic / nif
assert [f'{ratio[i]:.2f}' for i in fi] == ['0.06', '0.10']
SA = {r['model']: r for r in S['seg_agreement']}
icc_all, icc_all_ci = SA['v2 no post-mask']['ICC'], SA['v2 no post-mask']['ICC_ci']
segx = pd.read_csv(OUT / 'tables/segcq500_excl108_sensitivity.csv')
icc_ex = float(segx[(segx.model == 'v2 no post-mask') & (segx.set == 'excl108')].ICC.iloc[0])
icc_ex_ci = X['seg_v2nm_excl108_icc_ci']
assert f'{icc_all:.3f}' == '0.104' and f'{icc_ex:.3f}' == '0.677'
assert REG['seg_v2_icc_ci'].startswith(f'{icc_all:.3f} ({icc_all_ci[0]:.3f}')

LIM = (0.07, 900)
TK = [0.1, 1, 10, 100]
HL = dict(s=30, facecolor=WHITE, edgecolor=INK, linewidths=1.4, zorder=5)    # discordant reference sources (neutral ink ring)


from matplotlib.ticker import NullLocator


def logax(ax):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(*LIM); ax.set_ylim(*LIM)
    plain_log(ax, 'x', TK); plain_log(ax, 'y', TK)
    ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
    ax.set_aspect('equal')
    ax.plot(LIM, LIM, color=MUTED, lw=1.0, zorder=1)
    grid(ax, 'both')


ok = np.ones(43, bool); ok[fi] = False
YT = 16.0      # top of the scatter axes
SZ = 52.0
# ======================================================================= a: same model, two pipelines
axA = axes_mm(fig, 16, YT, SZ, SZ)
logax(axA)
axA.scatter(nif[ok], dic[ok], s=14, color=M['v2'], alpha=0.8, lw=0, zorder=3)
axA.scatter(nif[fi], dic[fi], **HL)
for i, (dx, dy) in zip(fi, [(7, 0), (7, 7)]):
    axA.annotate(f'{flag[ks[i]]}\nDICOM {dic[i]:.1f}, file {nif[i]:.1f} mL', (nif[i], dic[i]), xytext=(dx, dy), textcoords='offset points', fontsize=T7,
                 color=INK, ha='left', va='center', linespacing=1.05, path_effects=halo(2.0))
axA.set_xlabel('v2, Seg-CQ500 file (mL)'); axA.set_ylabel('v2, CQ500 DICOM (mL)')
fig_text(fig, 16, YT - 4.6, nminus(f"ICC(A,1) {cp_['ICC']:.3f} ({cp_['ICC_ci'][0]:.3f}–{cp_['ICC_ci'][1]:.3f})"), H, color=INK, ha='left')
fig_text(fig, 16, YT - 1.6, nminus(f"DICOM − file: mean {cp_['bias']:.1f} mL (limits {cp_['loa'][0]:.1f} to {cp_['loa'][1]:.1f})"), H, color=INK2, ha='left')
ptitle(fig, axA, 'a', 'Same model, two pipelines', dx_mm=14.0, dy_mm=9.4)

# ======================================================================= b: prediction vs Seg-CQ500 reference
axB = axes_mm(fig, 86, YT, SZ, SZ)
logax(axB)
axB.scatter(ref[ok], nif[ok], s=14, color=M['v2'], alpha=0.8, lw=0, zorder=3)
axB.scatter(ref[fi], nif[fi], **HL)
for i, (dx, dy, ha, va) in zip(fi, [(-6, 5, 'right', 'bottom'), (6, -3, 'left', 'top')]):
    axB.annotate(f'{flag[ks[i]]}', (ref[i], nif[i]), xytext=(dx, dy), textcoords='offset points', fontsize=T7, color=INK, ha=ha, va=va,
                 path_effects=halo(2.0), arrowprops=dict(arrowstyle='-', lw=1.0, color=INK2, shrinkA=0, shrinkB=3))
axB.set_xlabel('Seg-CQ500 reference (mL)'); axB.set_ylabel('v2, Seg-CQ500 file (mL)')
fig_text(fig, 86, YT - 4.6, nminus(f"ICC(A,1), 43 scans: {icc_all:.3f} ({icc_all_ci[0]:.3f} to {icc_all_ci[1]:.3f})"), H, color=INK, ha='left')
fig_text(fig, 86, YT - 1.6, nminus(f"Without CT-108: {icc_ex:.3f} ({icc_ex_ci[0]:.3f}–{icc_ex_ci[1]:.3f})"), H, color=INK, ha='left')
ptitle(fig, axB, 'b', 'Against the Seg-CQ500 reference', dx_mm=14.0, dy_mm=9.4)

# ======================================================================= c: reference volume by CQ500 readers calling ICH
axC = axes_mm(fig, 147, YT, 27, SZ)
rng = np.random.default_rng(5)
xpos = {3: 0, 2: 1, 1: 2}
axC.axvspan(1.5, 2.5, color=BAND, lw=0, zorder=0)
for kk, x in xpos.items():
    v = ref[kI == kk]
    xj = x + rng.uniform(-0.22, 0.22, len(v)) if len(v) > 2 else x + np.zeros(len(v))
    isf = np.isin(np.where(kI == kk)[0], fi)
    axC.scatter(xj[~isf], v[~isf], s=12, color=INK2, alpha=0.7, lw=0, zorder=3)
    if isf.any():
        axC.scatter(xj[isf], v[isf], **HL)
    if len(v) >= 4:
        axC.plot([x - 0.3, x + 0.3], [np.median(v)] * 2, color=INK, lw=1.6, zorder=5, solid_capstyle='butt')
for i, (dx, dy, ha, va) in zip(fi, [(-7, 0, 'right', 'center'), (0, -7, 'center', 'top')]):
    axC.annotate(f"{flag[ks[i]]}", (2, ref[i]), xytext=(dx, dy), textcoords='offset points', fontsize=T7, color=INK, ha=ha, va=va,
                 path_effects=halo(2.0))
axC.set_yscale('log'); axC.set_ylim(*LIM); plain_log(axC, 'y', TK); axC.yaxis.set_minor_locator(NullLocator())
axC.set_xlim(-0.55, 2.55)
axC.set_xticks([0, 1, 2], [f'3/3\n({(kI == 3).sum()})', f'2/3\n({(kI == 2).sum()})', f'1/3\n({(kI == 1).sum()})'])
axC.set_xlabel('Readers calling ICH (scans)')
axC.set_ylabel('Seg-CQ500 reference (mL)')
grid(axC, 'y')
fig_text(fig, 147, YT - 1.6, 'Lines: medians (n ≥ 4)', H, color=INK2, ha='left')
ptitle(fig, axC, 'c', 'Reference by reader vote', dx_mm=14.0, dy_mm=9.4)

print('overflow', overflow_mm(fig))
print({flag[ks[i]]: dict(dicom=round(dic[i], 3), file=round(nif[i], 3), ref=round(ref[i], 1), ratio=round(ratio[i], 3)) for i in fi})
print('ICC all', icc_all, icc_all_ci, 'excl', icc_ex, icc_ex_ci, 'medians', {k: round(float(np.median(ref[kI == k])), 2) for k in (3, 2)})
save(fig, 'SF_segcq500_reference', 'supp')
