from common import *
import json
r=pd.read_csv(E/'cq500_external/reads.csv')
rng=np.random.default_rng(20261029)
labs=['ICH','IPH','IVH','SDH','EDH','SAH','ChronicBleed','BleedLocation-Left','BleedLocation-Right','Fracture','CalvarialFracture','OtherFracture','MassEffect','MidlineShift']
rows=[]
for l in labs:
    X=r[[f'R{i}:{l}' for i in (1,2,3)]].values; cnt=np.c_[X.sum(1),3-X.sum(1)]
    fk=fleiss_kappa(cnt); n=len(X)
    bs=[fleiss_kappa(cnt[rng.integers(0,n,n)]) for _ in range(10000)]
    lo,hi=np.nanpercentile(bs,[2.5,97.5])
    pk=[cohen_kappa(X[:,a],X[:,b]) for a,b in [(0,1),(1,2),(2,0)]]
    k=X.sum(1)
    rows.append(dict(label=l,n=n,fleiss=fk,lo=lo,hi=hi,cohen_R1R2=pk[0],cohen_R2R3=pk[1],cohen_R3R1=pk[2],n0=int((k==0).sum()),n1=int((k==1).sum()),n2=int((k==2).sum()),n3=int((k==3).sum()),
                     majority_pos=int((k>=2).sum()),split_among_majority_pos_pct=100*(k==2).sum()/max((k>=2).sum(),1),unanimous_pct=100*((k==0)|(k==3)).mean()))
df=pd.DataFrame(rows); df.to_csv(OUT/'tables'/'cq500_reader_agreement_491.csv',index=False)
print(df.round(4).to_string())
