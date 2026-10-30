"""Supplementary figure - Subtype discrimination reflects co-occurrence and
hemorrhage burden. Panel b adds the within-ICH AUC of the public model's subtype probabilities (post hoc replication).

a co-occurrence of reader-majority subtypes among the 205 ICH-positive CQ500 scans (all 17 observed combinations + none)
b within-ICH AUC per subtype for v2 total predicted volume (locked score; v2 is binary, so descriptive), v3b class volume
  and v3b class share (post hoc), with Holm-adjusted paired DeLong tests of v3b class volume minus v2 total volume
c v3b cross-subtype call matrix at >=1 mL (diagonal: sensitivity; off-diagonal: scans with the row subtype and without the
  column subtype)
Subtype order is fixed throughout (EDH, IPH, IVH, SAH, SDH), as in ncstyle.SUB.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nc_b import *
from ncdata import *
import json as _json
PMJ = _json.load(open(OUT / 'public_model_cq500.json'))
PMS = {r['sub']: r for r in PMJ['subtypes']}
PMC, PMMK = '#6a3d9a', 'h'
from matplotlib.colors import LinearSegmentedColormap

HFIG = 150
fig = plt.figure(figsize=(W2, HFIG * MM))
m = load_cq500()
ich = m[m.ref_ICH == 1]
assert len(ich) == 205
dec = {r['sub']: r for r in N['subtype_decomposition']}
holm = R3['within_ich_delong']

# ============================================================================ a: co-occurrence (UpSet)
combo = ich[['ref_' + c for c in SUB]].astype(int)
keys = combo.apply(lambda r: tuple(c for c, v in zip(SUB, r) if v), axis=1)
cnt = keys.value_counts()
nonempty = cnt[[len(k_) > 0 for k_ in cnt.index]].sort_values(ascending=False, kind='stable')
none_n = int(cnt.get((), 0))
cols = list(nonempty.index) + [()]
vals = list(nonempty.values) + [none_n]
assert sum(vals) == 205 and len(nonempty) == 17
tot = {s_: int(ich['ref_' + s_].sum()) for s_ in SUB}
iso = {s_: int(cnt.get((s_,), 0)) for s_ in SUB}
assert iso['IVH'] == 0 and tot == {k_: int(v_) for k_, v_ in REG['cq_prev'].items() if k_ in SUB}

XM, WM = 37, 141           # matrix/bar block (mm)
axAb = axes_mm(fig, XM, 10, WM, 27)
axAd = axes_mm(fig, XM, 39.5, WM, 21)
nC = len(cols)
xs = np.arange(nC) + np.array([0] * (nC - 1) + [0.6])   # small gap before the 'no subtype label' column
COL_ONE, COL_MULTI = INK, '#a9a59d'
for xx, k_, v in zip(xs, cols, vals):
    if len(k_) == 0:
        axAb.bar(xx, v, width=0.66, color=WHITE, ec=INK, lw=1.0, zorder=3)
    else:
        axAb.bar(xx, v, width=0.66, color=COL_ONE if len(k_) == 1 else COL_MULTI, lw=0, zorder=3)
    axAb.text(xx, v + 1.5, str(v), ha='center', va='bottom', fontsize=7, color=INK)
axAb.set_xlim(xs[0] - 0.6, xs[-1] + 0.6); axAb.set_xticks([])
axAb.set_ylim(0, 92); axAb.set_yticks([25, 50, 75])
axAb.set_ylabel('Scans')
axAb.spines['bottom'].set_visible(False)
grid(axAb, 'y')
axAb.legend(handles=legend_handles([('One subtype', dict(patch=True, color=COL_ONE)), ('Two or more subtypes', dict(patch=True, color=COL_MULTI)),
                                    ('No subtype label', dict(patch=True, color=WHITE, ec=INK))]),
            loc='upper right', bbox_to_anchor=(1.0, 1.02), ncol=1, handlelength=1.2, handleheight=0.9, labelspacing=0.3)
axAb.text(xs[3] + 0.4, 88, 'Columns ordered by count', fontsize=7, color=INK2, ha='left', va='top')
for j, sname in enumerate(SUB):
    if j % 2 == 0:
        axAd.axhspan(j - 0.5, j + 0.5, color=BAND, lw=0, zorder=0)
    for xx, k_ in zip(xs, cols):
        on = sname in k_
        axAd.plot([xx], [j], marker='o', ms=4.6 if on else 3.6, color=INK if on else '#d4d0c8', mec=WHITE if on else '#d4d0c8', mew=0.6 if on else 0, ls='', zorder=3)
for xx, k_ in zip(xs, cols):
    ys_ = [SUB.index(sn) for sn in k_]
    if len(ys_) > 1:
        axAd.plot([xx, xx], [min(ys_), max(ys_)], color=INK, lw=1.4, zorder=2)
axAd.set_xlim(axAb.get_xlim()); axAd.set_ylim(len(SUB) - 0.5, -0.5)
axAd.set_xticks([]); axAd.set_yticks([])
for s_ in axAd.spines.values():
    s_.set_visible(False)
# subtype labels with isolated / total counts (text, not bars)
note(fig, XM - 1.5, 38.9, 'Isolated/total', ha='right', va='bottom', color=INK2)
for j, s_ in enumerate(SUB):
    axAd.text(-0.012, j, f'{s_}  {iso[s_]}/{tot[s_]}', transform=axAd.get_yaxis_transform(), fontsize=7, ha='right', va='center',
              color=INK, fontweight='bold' if iso[s_] == 0 else 'normal')
ptitle(fig, axAb, 'a', f'Co-occurrence of reader-majority subtypes ({len(ich)} ICH-positive scans; no scan had isolated IVH)', dx_mm=33.0)

# ============================================================================ b: within-ICH AUC
YR2, HR2 = 78, 48
axB = axes_mm(fig, 19, YR2, 47, HR2)
specs = [('auc_within_ich_v2', 'v2 total volume', M['v2'], MK['v2'], True, -0.3),
         ('auc_within_ich_v3b', 'v3b class volume', M['v3b'], MK['v3b'], True, -0.1),
         ('auc_within_ich_v3b_fraction', 'v3b class share', M['v3b'], MK['v3b'], False, 0.1),
         ('pm_within', 'public model, subtype probability', PMC, PMMK, True, 0.3)]
for i, s_ in enumerate(SUB):
    if i % 2 == 0:
        axB.axhspan(i - 0.5, i + 0.5, color=BAND, lw=0, zorder=0)
    for key, lab, col, mk, filled, off in specs:
        if key == 'pm_within':
            r_ = PMS[s_]; a, lo, hi = r_['auc_within'], r_['auc_within_lo'], r_['auc_within_hi']
        else:
            a, lo, hi = dec[s_][key]
        ci_point(axB, a, i + off, lo, hi, col, marker=mk, filled=filled, ms=4.4)
axB.axvline(0.5, color=INK2, lw=1.0, ls=(0, (2, 2)), zorder=1)
axB.set_yticks(range(len(SUB)), [f"{s_}\n{dec[s_]['pos_within_ich']} of 205" for s_ in SUB], linespacing=1.15)
axB.tick_params(axis='y', length=0, labelcolor=INK)
axB.spines['left'].set_visible(False)
axB.set_ylim(len(SUB) - 0.5, -0.5); axB.set_xlim(0.3, 1.03)
axB.set_xticks([0.3, 0.5, 0.7, 0.9, 1.0], ['0.3', '0.5', '0.7', '0.9', '1'])
axB.set_xlabel('AUC within ICH-positive scans (95% CI)')
grid(axB, 'x')
# Holm column
XH = 68.5
note(fig, XH, YR2 - 1.2, 'v3b class volume − v2 total:\nΔAUC (95% CI); P (Holm)', va='bottom', color=INK)
for i, s_ in enumerate(SUB):
    d_, lo, hi = holm[s_]['diff']
    assert abs(d_ - dec[s_]['delong_within_ich_v3b_minus_v2'][0]) < 1e-12
    ph = holm[s_]['p_holm']
    ptxt = fmt_p(ph)
    d_ = 0.0 if abs(d_) < 0.005 else d_
    axB.text(1.04, i - 0.17, fmt_ci(d_, lo, hi), transform=axB.get_yaxis_transform(), fontsize=7, color=INK, ha='left', va='center')
    axB.text(1.04, i + 0.21, ptxt, transform=axB.get_yaxis_transform(), fontsize=7, color=INK, ha='left', va='center')
fig.legend(handles=legend_handles([(lab, dict(color=col, marker=mk, filled=filled, ms=4.4)) for _, lab, col, mk, filled, _ in specs]),
           loc='upper left', bbox_to_anchor=(4 / 180, 1 - (YR2 + HR2 + 9.5) / HFIG), ncol=2, columnspacing=1.0, handletextpad=0.3, borderaxespad=0, borderpad=0)
note(fig, 4, YR2 + HR2 + 17.0, 'v2 is binary: its AUCs use total volume and carry no type information\n(descriptive). v3b scores and the public model are post hoc.')
ptitle(fig, axB, 'b', 'Within-ICH discrimination by score', dx_mm=15.0, dy_mm=8.5)

# ============================================================================ c: v3b call matrix
K = np.array(CALLM['k']); Nn = np.array(CALLM['n']); Mx = K / Nn
assert CALLM['rows'] == SUB and np.allclose(Mx, np.array(CALLM['M']), atol=1e-6)
cell = HR2 / 5
axC = axes_mm(fig, 178 - 5 * cell, YR2, 5 * cell, HR2)
cmap = LinearSegmentedColormap.from_list('v3b', ['#fdf5f0', '#f6cdb6', '#ec9a72', M['v3b'], '#8f3410'])
axC.imshow(Mx, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest', extent=(-0.5, 4.5, 4.5, -0.5))
for i in range(5):
    for j in range(5):
        bg = cmap(Mx[i, j]); tc = text_on(bg)
        axC.text(j, i - 0.17, f'{Mx[i, j]:.2f}', ha='center', va='center', fontsize=7, color=tc, fontweight='bold' if i == j else 'normal')
        axC.text(j, i + 0.2, f'{K[i, j]}/{Nn[i, j]}', ha='center', va='center', fontsize=7, color=tc)
    axC.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.6, zorder=5))
sah = SUB.index('SAH')
axC.add_patch(Rectangle((sah - 0.5, -0.5), 1, 5, fill=False, ec=INK, lw=1.0, ls=(0, (2.5, 1.5)), zorder=6))
axC.set_xticks(range(5), SUB); axC.set_yticks(range(5), [f'{s_} ({int(Nn[i, i])})' for i, s_ in enumerate(SUB)])
axC.tick_params(length=0, labelcolor=INK)
axC.xaxis.set_ticks_position('top'); axC.xaxis.set_label_position('top')
for s_ in axC.spines.values():
    s_.set_visible(False)
axC.set_xlabel('v3b class volume ≥1 mL', labelpad=2.5)
axC.set_ylabel('Reader-majority subtype (n)', labelpad=2)
off = [Mx[i, sah] for i in range(5) if i != sah]
assert abs(min(off) - R3['sah_overcall_range'][0]) < 1e-9 and abs(max(off) - R3['sah_overcall_range'][1]) < 1e-9
note(fig, 106, YR2 + HR2 + 2.5,
     f'Outlined diagonal: sensitivity. Off-diagonal: share of scans with\nthe row subtype and without the column subtype. Dashed column:\n'
     f'SAH called in {100 * min(off):.0f}–{pct(max(off))} of scans without reader-labeled SAH.')
ptitle(fig, axC, 'c', 'v3b cross-subtype calls', dx_mm=15.5, dy_mm=8.5)

print('overflow', overflow_mm(fig))
save(fig, 'SF_subtypes', 'supp_mr')
