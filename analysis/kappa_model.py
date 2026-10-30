from common import *
m=load_cq500()
R={}
rows=[]
for lab,score,thr in [('ICH','pred_volume_ml',1.0),('ICH','ANY_ml',1.0)]+[(c,c+'_ml',1.0) for c in SUB]:
    X=m[[f'R{i}:{lab}' for i in (1,2,3)]].values
    pred=(m[score].values>=thr).astype(int)
    rr=[cohen_kappa(X[:,a],X[:,b]) for a,b in [(0,1),(0,2),(1,2)]]
    mr=[cohen_kappa(pred,X[:,a]) for a in range(3)]
    rows.append(dict(label=lab,score=score,reader_reader_mean=np.mean(rr),rr_min=min(rr),rr_max=max(rr),model_reader_mean=np.mean(mr),mr_min=min(mr),mr_max=max(mr),
                     model_vs_majority=cohen_kappa(pred,m['ref_'+lab].values)))
df=pd.DataFrame(rows); df.to_csv(OUT/'tables'/'cq500_model_reader_kappa.csv',index=False)
print(df.round(3).to_string())
