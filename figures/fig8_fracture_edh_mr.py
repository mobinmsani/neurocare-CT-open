"""Figure 8 - CQ500 scans with reader-majority skull fracture and epidural hematoma: four scans in brain and
bone windows of the same slice, and the fracture label in all 13 scans with a majority EDH label."""
from nc_s2 import *

H = 188
fig = plt.figure(figsize=(W2, H * MM))
m = load_cq500().set_index('name')
both = m[(m.ref_Fracture == 1) & (m.ref_EDH == 1)].sort_values('pred_volume_ml')
assert len(both) == 11 and (both.ref_CalvarialFracture == 1).all()
rk = np.linspace(0, len(both) - 1, 4).round().astype(int)          # four evenly spaced ranks by v2 volume
pick = list(both.index[rk])
assert list(rk) == [0, 3, 7, 10] and pick == ['CQ500-CT-137', 'CQ500-CT-173', 'CQ500-CT-181', 'CQ500-CT-107']
SUBN = ['EDH', 'IPH', 'IVH', 'SAH', 'SDH']


def votes_(r, l):
    return int(r[f'R1:{l}'] + r[f'R2:{l}'] + r[f'R3:{l}'])


pw, gx, x0, y0 = 39.5, 3.3, 11.0, 25.0
for j in range(4):
    rb, ct_b = load_asset(f'fx_{j + 1:02d}')
    rn, ct_n = load_asset(f'fx_{j + 5:02d}')
    c = rb['case']
    assert c == pick[j] and rn['case'] == c
    r = m.loc[c]
    assert rb['titles'][1] == f"model hemorrhage {r.pred_volume_ml:.1f} mL"
    crop = head_bbox(ct_b, thr=55, margin=4)
    xx = x0 + j * (pw + gx)
    ax = axes_mm(fig, xx, y0, pw, pw)
    draw_ct(ax, ct_b, crop)
    nbox = sum(1 for o in rb['overlays'] if o['kind'] == 'box')
    for o in rb['overlays']:
        if o['kind'] == 'box':
            for sp in o['subpaths']:
                sp = np.asarray(sp)
                ax.plot(sp[:, 0], sp[:, 1], color=IMG['ref'], lw=1.0, ls=(0, (3.2, 1.6)), zorder=4)
        elif o['kind'] == 'pred':
            for sp in o['subpaths']:
                sp = np.asarray(sp)
                ax.plot(sp[:, 0], sp[:, 1], color=IMG['pred'], lw=1.0, zorder=5, solid_joinstyle='round')
    orient_marks(ax, crop, size=7.0, color='#d0d0d0')
    if c == 'CQ500-CT-137':
        assert nbox == 0
        tag(ax, 0.5, 0.97, 'Slice does not include the EDH', ha='center', fc=WHITE, tc=CRIT, weight='bold')
    ptitle(fig, ax, 'abcd'[j], c, sub=f'rank {rk[j] + 1} of 11 by v2 scan volume', dx_mm=0.0, dy_mm=3.2)
    ax2 = axes_mm(fig, xx, y0 + pw + 2.2, pw, pw)
    draw_ct(ax2, ct_n, crop)                     # identical slice and field of view
    orient_marks(ax2, crop, size=7.0, color='#d0d0d0')
    kern = 'bone kernel' if 'bone kernel' in rn['titles'][0] else 'soft-tissue kernel'
    assert (kern == 'bone kernel') == (r.bone_source == 'bone_kernel')
    subs = [f'{s} {votes_(r, s)}/3' for s in SUBN if votes_(r, s) >= 2]
    meta = [f'Scan-level v2 volume {r.pred_volume_ml:.1f} mL', f'Bone window: {kern}', f"Readers: fracture {votes_(r, 'Fracture')}/3",
            ', '.join(subs[:3]), ', '.join(subs[3:])]
    for i_, t in enumerate([t for t in meta if t]):
        fig_text(fig, xx, y0 + 2 * pw + 2.2 + 1.4 + i_ * 3.3, t, H, color=INK if i_ < 1 else INK2)
    if j == 0:
        for a_, lab in ((ax, 'Brain window'), (ax2, 'Bone window, same slice')):
            a_.text(-0.04, 0.5, lab, transform=a_.transAxes, rotation=90, fontsize=F7, fontweight='bold', color=INK, ha='right', va='center')
key_row(fig, x0, 4.5, [('line', IMG['ref'], 'BHX reference box on the displayed slice', dict(ls=(0, (3.2, 1.6)), lw=1.4)),
                       ('line', IMG['pred'], 'v2 prediction', dict(lw=1.4))], H, sw=6.0, gap_mm=6.0)
fig_text(fig, x0, 8.0, 'Selected scans with reader-majority fracture and EDH. Images do not validate fracture localization: CQ500 records fracture per scan, '
         'without location.\nPatient right on image left.', H, linespacing=1.25)

# ======================================================================= e: all 13 scans with a majority EDH label
edh = m[m.ref_EDH == 1].sort_values('pred_volume_ml')
assert len(edh) == 13 and int(edh.ref_Fracture.sum()) == 11 == REG['cq_misc']['edh_with_fx']
axE = axes_mm(fig, 40, 140, 72, 38)
for i_, (c, r) in enumerate(edh.iterrows()):
    fx = int(r.ref_Fracture) == 1
    axE.plot([r.pred_volume_ml], [i_], marker='o', ms=4.8, mfc=INK if fx else WHITE, mec=INK, mew=1.0, ls='', zorder=3)
    axE.plot([0.9, r.pred_volume_ml], [i_, i_], color=GRID, lw=1.0, zorder=1)
    axE.text(-0.02, i_, c.replace('CQ500-', ''), transform=axE.get_yaxis_transform(), fontsize=F7, ha='right', va='center',
             color=INK, fontweight='bold' if c in pick else 'normal')
    axE.text(1.03, i_, f"{votes_(r, 'Fracture')}/3", transform=axE.get_yaxis_transform(), fontsize=F7, ha='left', va='center', color=INK)
    axE.text(1.20, i_, f"{votes_(r, 'EDH')}/3", transform=axE.get_yaxis_transform(), fontsize=F7, ha='left', va='center', color=INK)
    axE.text(1.37, i_, r.Category, transform=axE.get_yaxis_transform(), fontsize=F7, ha='left', va='center', color=INK2)
for xx_, t in [(1.03, 'Fracture'), (1.20, 'EDH'), (1.37, 'Batch')]:
    axE.text(xx_, -1.25, t, transform=axE.get_yaxis_transform(), fontsize=F7, ha='left', va='center', color=INK2, fontweight='bold')
axE.text(1.03, -2.35, 'Reader votes', transform=axE.get_yaxis_transform(), fontsize=F7, ha='left', va='center', color=INK2)
axE.set_ylim(len(edh) - 0.5, -0.7); axE.set_yticks([]); axE.spines['left'].set_visible(False)
axE.set_xscale('log'); axE.set_xlim(0.9, 200)
plain_log(axE, 'x', [1, 3, 10, 30, 100], ['1', '3', '10', '30', '100'])
axE.set_xlabel('v2 scan-level predicted volume (mL)')
grid(axE, 'x')
ptitle(fig, axE, 'e', f'All {len(edh)} scans with a reader-majority EDH label: {int(edh.ref_Fracture.sum())} also carry a fracture label',
       dx_mm=37.0, dy_mm=8.6)
_bxl = pd.read_csv(E / 'bhx_scan_labels' / 'bhx_scan_labels.csv').set_index('scan')
n_edh_box = int(sum(1 for c in edh.index if c in _bxl.index and _bxl.loc[c, 'bhx_Epidural'] > 0))
assert n_edh_box == 6
fig_text(fig, 147.0, 141.0, 'Filled circle: majority\nfracture label (all\ncalvarial); open: none.\nBold: scans shown in a–d.\n'
         f'No fracture location is\nreleased; BHX marks EDH\nin {n_edh_box} of the {len(edh)} scans.', H, linespacing=1.3)
print('overflow', overflow_mm(fig))
save(fig, 'Figure_8_Fracture_EDH', 'main_mr')
check_qa('Figure_8_Fracture_EDH', 'main_mr')
