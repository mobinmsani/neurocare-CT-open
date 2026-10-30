"""SF_segcq500_errors - Seg-CQ500 voxel-error examples from the original pipeline (BLAST-CT, BLAST-CT with HU-constrained
region growing, v1) on one slice per scan."""
from nc_s2 import *

H = 178
fig = plt.figure(figsize=(W2, H * MM))
seg4 = pd.read_csv(T / 'SegCQ500_four_method_comparison.csv')
segnm = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/segcq500_no_postmask.csv').set_index('case_id')
ks = sorted(segnm.index)
R = segnm.loc[ks, 'ref_ml']
w = seg4.pivot(index='case_id', columns='method', values='dice')
pv = seg4.set_index(['case_id', 'method'])
dg = w['blast_hu45_r2'] - w['blast_model1']
zero3 = w[(w[['blast_model1', 'blast_hu45_r2', 'lightunet25d']] == 0).all(axis=1)].index.tolist()
assert dg.idxmin() == 'CQ500-CT-137' and dg.idxmax() == 'CQ500-CT-139' and zero3 == ['CQ500-CT-208']
cases = [('segm_1', 'CQ500-CT-137', nminus(f'Largest Dice decrease with HU growth ({dg.min():+.3f})')),
         ('segm_2', 'CQ500-CT-208', 'No overlap for any of the three methods'),
         ('segm_3', 'CQ500-CT-139', nminus(f'Largest Dice increase with HU growth ({dg.max():+.3f})'))]
cols = [('ct', 'CT and Seg-CQ500 reference', None), ('blast_errors', 'BLAST-CT', 'blast_model1'),
        ('growth_errors', 'BLAST-CT + HU growth', 'blast_hu45_r2'), ('v1_errors', 'v1', 'lightunet25d')]


def pred_ml(cid, mth):
    r = pv.loc[(cid, mth)]
    return float(r.prediction_volume_ml if pd.notna(r.prediction_volume_ml) else r.pred_volume_ml)


pw, gx, x0, y0, rowh = 40.0, 3.0, 9.0, 30.0, 50.0
DIM = 0.8
for i, (key, cid, why) in enumerate(cases):
    rec = json.load(open(ASSETS / f'{key}.json'))
    assert rec['case'] == cid
    ct = load_npy(ASSETS / f'{key}_ct.npy')
    crop = head_bbox(ct, thr=60, margin=4)
    yy = y0 + i * rowh
    axrow = None
    for j, (layer, ttl, mth) in enumerate(cols):
        ax = axes_mm(fig, x0 + j * (pw + gx), yy, pw, pw)
        if layer == 'ct':
            draw_ct(ax, ct, crop)
            for sp in rec['overlays']['refcontour'][0]['subpaths'] if isinstance(rec['overlays']['refcontour'], list) else []:
                pass
            for o in rec['overlays']['refcontour']:
                for sp in o['subpaths']:
                    sp = np.asarray(sp)
                    ax.plot(sp[:, 0], sp[:, 1], color=IMG['ref'], lw=1.0, zorder=4, solid_joinstyle='round')
            strip_text(ax, f'Reference {R.loc[cid]:.1f} mL', 'below', pad_mm=1.0, color=INK2)
            axrow = ax
        else:
            err = load_npy(ASSETS / f'{key}_{layer}.npy')
            ax.imshow(ct.astype(float) * DIM, cmap='gray', vmin=0, vmax=255, interpolation='bicubic', zorder=0)
            ax.imshow(err_overlay(err, alpha=0.85), interpolation='nearest', zorder=1)
            if (err == 1).any():   # keyline around true-positive voxels so white fill is not confused with bone
                ax.contour((err == 1).astype(float), [0.5], colors=['#000000'], linewidths=1.0, zorder=2)
            ax.set_xlim(crop[0], crop[1]); ax.set_ylim(crop[3], crop[2])
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.set_facecolor('black')
            strip_text(ax, f'Dice {w.loc[cid, mth]:.3f}; predicted {pred_ml(cid, mth):.1f} mL', 'below', pad_mm=1.0, color=INK2)
        orient_marks(ax, crop, size=7.0, color='#d0d0d0')
        if i == 0:
            strip_text(ax, ttl, 'above', pad_mm=1.0, weight='bold')
    ptitle(fig, axrow, 'abc'[i], f'{cid}: {why}', dx_mm=x0 - 2.0, dy_mm=1.6 if i else 5.4)
key_row(fig, x0, 14.0, [('line', IMG['ref'], 'Seg-CQ500 reference contour', dict(lw=1.5)), ('fill', WHITE, 'True positive (black keyline)', dict(ec='#000000')),
                        ('fill', IMG['fp'], 'False positive (prediction only)'), ('fill', IMG['fn'], 'False negative (reference only)')], H, gap_mm=5.0)
fig_text(fig, x0, 3.0, 'Examples from the BLAST-CT and v1 pipelines. Metrics are whole-scan; one slice is shown per scan. Error maps on a lightly dimmed CT underlay.\n'
         'Patient right on image left. Volumes assume millimeter voxel spacing.',
         H, linespacing=1.25)
print('overflow', overflow_mm(fig))
save(fig, 'SF_segcq500_errors', 'supp')
check_qa('SF_segcq500_errors')
