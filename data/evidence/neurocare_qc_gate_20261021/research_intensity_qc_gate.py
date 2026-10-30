#!/usr/bin/env python3
"""Image-only intensity-encoding QC gate. Rule frozen from TRAINING scans only, then applied to test.
Abstention is reported, never hidden: coverage and all-case metrics are retained."""
import csv, gzip, json, zipfile, hashlib, sys
from pathlib import Path
import numpy as np, nibabel as nib
W=Path(sys.argv[1]); out=W/"neurocare_qc_gate_20261021"; out.mkdir(exist_ok=True)
man=list(csv.DictReader(open(W/"neurocare_jama_target_20261020/trained_baseline/provider_split_manifest.csv")))
def feats(z,r):
    b=z.read(r["image_path"].split("!/",1)[1]); assert hashlib.sha256(b).hexdigest()==r["source_image_sha256"]
    x=np.asanyarray(nib.Nifti1Image.from_bytes(gzip.decompress(b)).dataobj).astype(np.float32)
    return {"p01":float(np.percentile(x,1)),"min":float(x.min()),"frac_window":float(((x>=-40)&(x<=120)).mean())}
rows=[]
with zipfile.ZipFile(W/"research_data/bhsd_original/label_192.zip") as z:
    for r in man:
        if r["analysis_partition"] not in("train","test"):continue
        rows.append({"case_id":r["case_id"],"partition":r["analysis_partition"],**feats(z,r)})
train=[r for r in rows if r["partition"]=="train"]
# Frozen rule: air must appear as negative HU. Threshold = midpoint between HU air (-1000) and water (0),
# accepted only if it does not reject >5% of training scans (checked, not tuned on outcomes).
T=-500.0; train_flag=sum(r["p01"]>T for r in train)
lock={"rule":"flag if 1st-percentile stored intensity > -500 (air absent => non-HU encoding suspected)",
      "derived_from":"physical HU definition; feasibility checked on training only","training_scans":len(train),
      "training_flagged":train_flag,"test_outcomes_viewed_before_lock":"aggregate Dice known; per-case QC-outcome link not used",
      "action":"abstain from quantitative output, do not rescale"}
json.dump(lock,open(out/"QC_LOCK.json","w"),indent=2)
m={r["case_id"]:r for r in csv.DictReader(open(W/"neurocare_neuroimaging_advanced_20261021/per_scan_quantitative_metrics.csv"))}
test=[]
for r in rows:
    if r["partition"]!="test":continue
    q=m[r["case_id"]]; test.append({**r,"flag":r["p01"]>T,"dice":float(q["dice"]),"pv":float(q["prediction_volume_ml"]),"rv":float(q["reference_volume_ml"])})
def s(sel):
    if not sel:return None
    p=np.array([t["pv"] for t in sel]);y=np.array([t["rv"] for t in sel]);d=p-y
    k=2;R=np.c_[p,y];n=len(p);g=R.mean();msr=k*((R.mean(1)-g)**2).sum()/(n-1);msc=n*((R.mean(0)-g)**2).sum()
    e=R-R.mean(1)[:,None]-R.mean(0)+g;mse=(e**2).sum()/(n-1);icc=(msr-mse)/(msr+mse+k*(msc-mse)/n)
    return {"n":n,"mean_dice":float(np.mean([t["dice"] for t in sel])),"MAE_ml":float(abs(d).mean()),
            "RMSE_ml":float(np.sqrt((d*d).mean())),"bias_ml":float(d.mean()),"ICC_A1":float(icc)}
acc=[t for t in test if not t["flag"]];fl=[t for t in test if t["flag"]]
res={"lock":lock,"test_scans":len(test),"flagged":len(fl),"coverage":len(acc)/len(test),
     "all_cases":s(test),"accepted_cases":s(acc),"flagged_cases":s(fl),
     "flagged_ids":[t["case_id"] for t in fl],
     "interpretation":"selective-prediction result; accepted-subset metrics apply only to QC-passing scans"}
json.dump(res,open(out/"QC_gate_results.json","w"),indent=2)
with open(out/"QC_features_all_188.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(res,indent=1))
