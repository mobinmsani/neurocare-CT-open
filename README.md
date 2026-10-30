# What open-access head CT references can validate: code and derived data

Code, per-scan inputs and analysis outputs for the article "Open-access head CT reference standards for intracranial hemorrhage
artificial intelligence: adequate for detection, insufficient for measurement" by Alireza Motamedi, Mobin Mobadersani and
Hassan Soleimanpour (Tabriz University of Medical Sciences). The article is under review; this repository will be linked to it on
publication.

The code regenerates every reported value (the registry in `out_delivered/registry.json`), every figure and Supplementary Data 1
from the stored per-scan outputs of the models and from model-free measurements of the released references. It contains no CT
images, image tiles, reference masks or model weights.

## Contents

- `analysis/`: statistics of the original evaluation and the post hoc analyses: the endpoint-specific validation-capacity rubric
  (`capacity_rubric_v2.py`; the first version is `capacity_rubric.py`), reader agreement (`kappa491.py`, and hemorrhage side among
  hemorrhage-positive scans in `side_agreement.py`), between-rater agreement in the MBH-Seg 2025 training release
  (`mbhseg_train_raters.py`), the quality-control and discordance ledger (`ledger.py`), the linkage of BHX boxes to CQ500 scans by
  DICOM study identifier (`bhx_scan_labels.py`, `bhx_exact.py`), the model-free description of the Seg-CQ500 masks
  (`segcq500_mask_check.py`, `segcq500_mask_hist.py`, `segcq500_mask_shape.py`), the replication with the public model
  (`public_model_cq500.py`), alternative references and lesion extent (`reference_choice.py`, `split_extent.py`), robustness of the
  agreement statistics (`volume_robust.py`), the adjudication case list (`adjudication_sample.py`), the registry of reported values
  (`build_registry.py`) and Supplementary Data 1 (`make_supp_data1_mr.py`). `compare_outputs.py` compares a fresh run with
  `out_delivered/`.
- `data/`: per-scan outputs of the original pipeline with their protocol locks, amendments and training records, the public-model
  outputs and protocol lock, CQ500 study and series identifiers, BHX box counts per CQ500 scan, Seg-CQ500 mask measurements and the
  catalogue of head CT resources with their access tiers (`data/README.md`).
- `figures/`: scripts for the main figures (`fig1_design_mr.py` to `fig8_fracture_edh_mr.py`) and the supplementary figures
  (`sf_*.py`), shared style (`ncstyle.py`), data access (`ncdata.py`) and the metadata of the image panels (`assets/`).
- `run/`: scripts that produced inputs outside the main pipeline: the public-model run on CQ500 with its protocol lock and
  batch-size check (`run/public_model/README.md`), and the collection of DICOM identifiers.
- `orig/locked_tables.md`: locked result tables of the original pipeline, used to check the reproduction.
- `out_delivered/`: the analysis outputs reported in the article.

## Running

Python 3.13 with the packages in `requirements.txt`. A full run takes about 25 minutes; the scripts run one after another, so one
CPU core is enough.

```
pip install -r requirements.txt
bash run_all.sh              # or: bash run_all.sh analysis | figures
python analysis/compare_outputs.py out out_delivered
```

`run_all.sh` writes the analysis outputs and Supplementary Data 1 to `out/` and the figures to `figures/out/`. Paths are relative to
this folder (override with `NEUROCARE_WORK`). Steps that read the released datasets themselves are optional and use the delivered
outputs when the data are absent: set `BLEEDSCENE_RAW` (BleedScene3D `raw` folder), `BHX_DIR` (folder with the BHX v1.1 CSV files),
`SEGCQ500_ZIP` (the Seg-CQ500 archive) and `MBHSEG_TRAIN_ZIP` (`MBH_Train_2025_voxel-label.zip` of the MBH-Seg 2025 training
release, read without extraction) to recompute them; these steps also need `nibabel`. The public model is run separately
(`run/public_model/README.md`).

Without the CT tiles, the figure scripts draw each image panel blank with the note "CT image not redistributed"; every plot is
unchanged. The tiles are 8-bit display crops of CQ500, Seg-CQ500 and PHE-SICH-CT-IDS slices and can be recreated from the
datasets; `figures/assets/` holds only their metadata (case, display settings and the vector overlays drawn in the published
panels). The supplementary figure of the two Seg-CQ500 masks on reader-negative scans (`sf_segref_mr.py`) needs the slices of CT-108
and CT-187 and is skipped without them; `analysis/segcq500_mask_hist.py` recreates them (set `SEGCQ500_ZIP`).

## Licenses

Code: MIT License (`LICENSE`). Derived per-scan values: CC BY 4.0, except values derived from CQ500 images (the CQ500 outputs of
every model, the CQ500 study and series identifiers and the Seg-CQ500 mask measurements), which keep the CQ500 license
(CC BY-NC-SA 4.0), and the BHX box counts, derived from BHX (CC BY-SA 4.0). All values remain subject to the terms of their source
datasets, including the RSNA 2019 terms for values derived from RSNA-derived masks. The CQ500 reader file is redistributed unchanged
under its source license. Third-party models and datasets are listed in `THIRD_PARTY_NOTICES.md`.

## Citation

Please cite the archived release (see `CITATION.cff`; the DOI is on the Zenodo record) and, once published, the article.

## Contact

Mobin Mobadersani, Faculty of Medicine, Tabriz University of Medical Sciences, Tabriz, Iran (mobinmsani@gmail.com;
ORCID 0009-0005-1869-6098).
