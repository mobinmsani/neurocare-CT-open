# Per-scan inputs

Outputs of the original pipeline, keyed by the public identifiers of each resource, and its protocol records. These are the
files the analysis and figure code reads, plus the protocol locks, amendments, training records and the frozen
software environment of the original pipeline.

- `evidence/cq500_external/`  locked CQ500 evaluation of v2: per-scan outputs (`cq500_per_scan.csv`), summary results, the
  fracture pilot record, the protocol lock, the CQ500 reader file `reads.csv` (Qure.ai, CC BY-NC-SA 4.0, included unchanged
  for the label joins) and the BHX license.
- `evidence/neurocare_neuroimaging_advanced_20261021/`  v1 per-scan segmentation metrics on the BHSD test set, reference
  subtype volumes, the analysis lock and the frozen requirements of the original pipeline.
- `evidence/neurocare_qc_gate_20261021/`  quality-control gate: rule, lock, code, results and features for all 188 BHSD scans
  and for Seg-CQ500.
- `evidence/neurocare_v2_20261021/`  v2 and v2-PM per-scan metrics (BHSD, Seg-CQ500), PHE-SICH-CT-IDS laterality and midline
  outputs, the edema model's split and per-patient test metrics, protocol locks, the v2 amendment and training records.
- `evidence/v3_subtypes/`  v3 and v3b protocols, training records, BHSD per-scan metrics and CQ500 per-scan class volumes.
- `tables/`  per-scan and per-rater tables of the original pipeline (Seg-CQ500, MR-BHSD, BHSD quality-control features).

Images, reference masks and reference labels other than the CQ500 reads are not included; they are available from their
creators under their licenses (see `THIRD_PARTY_NOTICES.md` and the Data availability statement of the article).

## Post hoc additions
- `evidence/public_model_cq500/`  the public model on the 489 analyzed CQ500 scans: per-scan outputs (`public_model_per_scan.csv`,
  collected by `run/collect_public_model.py` from one JSON per scan), the DICOM identifiers read during the run
  (`cq500_series_uids.csv`), run errors (none expected) and the protocol lock (SHA-256 63260e37b5d8207e7e98c897534f67034cc1c7057a16b77e108e871e0f6741a2).
- `evidence/cq500_uids/cq500_series_uids.csv`  study and series identifiers of every series in the 489 analyzed CQ500 archives,
  read from the released headers by HTTP range requests (`run/fetch_uids_range.py`, `run/collect_uids.py`), with the series chosen
  by the locked rule marked; checked against the identifiers read during the public-model run.
- `evidence/bhx_scan_labels/bhx_scan_labels.csv`  number of BHX boxes of each type per CQ500 scan, linked by StudyInstanceUID
  (`analysis/bhx_scan_labels.py`; BHX v1.1, CC BY-SA 4.0).
- `derived/`  model-free measurements of the released Seg-CQ500 masks (`analysis/segcq500_mask_check.py`,
  `analysis/segcq500_mask_hist.py`; the list of the 43 analyzed scans is `segcq500_analyzed_ids.txt`).
- `catalogue/`  the catalogue of 27 head CT resources with hemorrhage- or fracture-related references (access tier, license, reference type and use in the study; `catalogue.json`) and the access-tier definitions (`access_tiers.json`).
