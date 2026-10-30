from common import *
import re, json
qf=pd.read_csv(E/'neurocare_qc_gate_20261021/QC_features_all_188.csv')
mr=pd.read_csv(T/'MRBHSD_full43_compact_model_rater_metrics.csv')
mr6=pd.read_csv(T/'MRBHSD_all_per_scan_rater_metrics.csv')
pat=re.compile(r'^ID_([0-9a-f]{8})_ID_([0-9a-f]{10})$')
def parse(s):
    m_=pat.match(s); return (m_.group(1), m_.group(2)) if m_ else (None,None)
R={}
tr=qf[qf.partition=='train'].case_id.tolist(); te=qf[qf.partition=='test'].case_id.tolist(); mv=sorted(mr.case_id.unique())
for nm,ids in [('train',tr),('test',te),('mrbhsd',mv)]:
    P=[parse(i) for i in ids]
    R[nm]=dict(n=len(ids), all_match_pattern=all(p[0] for p in P), unique_patient_tokens=len(set(p[0] for p in P)), unique_study_tokens=len(set(p[1] for p in P)))
pt=lambda ids:set(parse(i)[0] for i in ids)
st=lambda ids:set(parse(i)[1] for i in ids)
R['patient_overlap_train_test']=len(pt(tr)&pt(te)); R['scan_overlap_train_test']=len(set(tr)&set(te))
R['patient_overlap_mr_train']=len(pt(mv)&pt(tr)); R['patient_overlap_mr_test']=len(pt(mv)&pt(te))
R['scan_overlap_mr_bhsd']=len(set(mv)&(set(tr)|set(te)))
R['study_overlap_any']=len(st(tr)&st(te))
# repeated patients within test
from collections import Counter
ct=Counter(parse(i)[0] for i in te); R['test_patients_multi']= {k:v for k,v in ct.items() if v>1}
ct2=Counter(parse(i)[0] for i in tr); R['train_patients_multi']={k:v for k,v in ct2.items() if v>1}
ct3=Counter(parse(i)[0] for i in mv); R['mr_patients_multi']={k:v for k,v in ct3.items() if v>1}
# does BHSD test per-scan file use the same IDs
v1=pd.read_csv(E/'neurocare_neuroimaging_advanced_20261021/per_scan_quantitative_metrics.csv')
R['v1_ids_eq_qc_test']=set(v1.case_id)==set(te)
save_json(R,OUT/'id_checks.json'); print(json.dumps(R,indent=1))
