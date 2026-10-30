from common import *
import json
R={}
m=load_cq500()
R['mls_pos_ich_neg']=int(((m.ref_MidlineShift==1)&(m.ref_ICH==0)).sum())
R['me_pos_ich_neg']=int(((m.ref_MassEffect==1)&(m.ref_ICH==0)).sum())
R['mls_pos_total']=int(m.ref_MidlineShift.sum())
R['ich_bilateral']=int(((m.ref_ICH==1)&(m['ref_BleedLocation-Left']==1)&(m['ref_BleedLocation-Right']==1)).sum())
R['ich_noside']=int(((m.ref_ICH==1)&(m['ref_BleedLocation-Left']==0)&(m['ref_BleedLocation-Right']==0)).sum())
R['ich_unilateral']=int(((m.ref_ICH==1)&(m['ref_BleedLocation-Left']!=m['ref_BleedLocation-Right'])).sum())
R['side_without_ich']=int(((m.ref_ICH==0)&((m['ref_BleedLocation-Left']==1)|(m['ref_BleedLocation-Right']==1))).sum())
R['fx_pos_ich_neg']=int(((m.ref_Fracture==1)&(m.ref_ICH==0)).sum())
# Seg-CQ500 agreement excluding CT-108
seg4=pd.read_csv(T/'SegCQ500_four_method_comparison.csv')
segnm=pd.read_csv(E/'neurocare_v2_20261021/hemorrhage_v2/segcq500_no_postmask.csv').set_index('case_id')
v2=pd.read_csv(E/'neurocare_v2_20261021/hemorrhage_v2/v2_metrics.csv'); v2s=v2[v2.dataset=='Seg-CQ500'].set_index('case_id')
ks=sorted(segnm.index); ks2=[c for c in ks if c!='CQ500-CT-108']
rows=[]
for mth,lab in [('blast_model1','BLAST-CT'),('blast_min100mm3','BLAST-CT + size filter'),('blast_hu45_r2','BLAST-CT + HU growth'),('lightunet25d','v1')]:
    g=seg4[seg4.method==mth].set_index('case_id')
    p=g.prediction_volume_ml.fillna(g.pred_volume_ml)
    for nm_,kk in [('all43',ks),('excl108',ks2)]:
        a=agree(p.loc[kk].values,g.loc[kk,'reference_volume_ml'].values); rows.append(dict(model=lab,set=nm_,n=len(kk),dice=g.loc[kk,'dice'].mean(),**a))
for lab,(d,p) in {'v2 prespecified':(v2s['dice'],v2s['pred_ml']),'v2 no post-mask':(segnm['dice_nopostmask'],segnm['pred_ml'])}.items():
    for nm_,kk in [('all43',ks),('excl108',ks2)]:
        a=agree(p.loc[kk].values,segnm.loc[kk,'ref_ml'].values); rows.append(dict(model=lab,set=nm_,n=len(kk),dice=d.loc[kk].mean(),**a))
# CQ500 DICOM pipeline v2 volume vs Seg ref excl 108
mm=m.set_index('name')
for nm_,kk in [('all43',ks),('excl108',ks2)]:
    a=agree(mm.loc[kk,'pred_volume_ml'].values,segnm.loc[kk,'ref_ml'].values); rows.append(dict(model='v2 no post-mask, CQ500 DICOM pipeline',set=nm_,n=len(kk),dice=np.nan,**a))
df=pd.DataFrame(rows); df.to_csv(OUT/'tables'/'segcq500_excl108_sensitivity.csv',index=False)
pd.set_option('display.width',250)
print(df.round(3).to_string())
# bootstrap CI for ICC excluding 108, v2 nomask (NIfTI) and DICOM pipeline
rng=np.random.default_rng(20261028)
def bci(p,y,B=10000):
    v=[]
    n=len(y)
    for _ in range(B):
        i=rng.integers(0,n,n); v.append(icc_a1(p[i],y[i]))
    return np.percentile(v,[2.5,97.5]).tolist()
p=segnm.loc[ks2,'pred_ml'].values; y=segnm.loc[ks2,'ref_ml'].values
R['seg_v2nm_excl108_icc_ci']=bci(p,y)
p2=mm.loc[ks2,'pred_volume_ml'].values
R['seg_dicom_excl108_icc_ci']=bci(p2,y)
g=seg4[seg4.method=='blast_model1'].set_index('case_id')
R['seg_blast_excl108_icc_ci']=bci(g.loc[ks2,'prediction_volume_ml'].values,g.loc[ks2,'reference_volume_ml'].values)
# Dice CI excluding 108 for v2 nomask
d=segnm.loc[ks2,'dice_nopostmask'].values
R['seg_v2nm_excl108_dice']=(d.mean(),np.percentile([d[rng.integers(0,len(d),len(d))].mean() for _ in range(10000)],[2.5,97.5]).tolist())
save_json(R,OUT/'extra_checks.json')
print(json.dumps(R,indent=1,default=str))
