# Third-party software and data

## Model used, not redistributed
- **CT head hemorrhage detection** (Ian Pan; MIT License). Hugging Face `ianpan/ct-head-hemorrhage-detection`, revision
  e6d89c253fdcc2a597dd0fd0f5771821f4947b37; source code github.com/i-pan/ct-head-hemorrhage-detection, revision 914c57d. Used
  unchanged for the post hoc replication on CQ500. Weights and code are not included here; download them from the author.
- **BLAST-CT** (Monteiro et al., 2020): released pretrained model, run locally as a comparator on Seg-CQ500; not included.

## Datasets used (no images or masks redistributed)
| Dataset | License | What this repository contains |
| --- | --- | --- |
| BHSD (RSNA 2019-derived) | as released by its creators; RSNA 2019 terms of use for the images | per-scan model outputs and reference volumes derived from the released masks |
| MR-BHSD (MBH-Seg 2025 validation release) | as released by the challenge organizers; RSNA 2019 terms of use for the images | per-scan Dice and volumes |
| MBH-Seg 2025 training release (voxel-level annotations) | as released by the challenge organizers; RSNA 2019 terms of use for the images | class volumes per volume and rater, and between-rater agreement statistics (`out_delivered/tables/mbhseg_train_raters.csv`, `out_delivered/mbhseg_train_raters.json`) |
| Seg-CQ500 | as released by its creators (images from CQ500, CC BY-NC-SA 4.0) | per-scan model outputs and mask measurements (CC BY-NC-SA 4.0) |
| CQ500 (Qure.ai) | CC BY-NC-SA 4.0 | the creators' reader file (reads.csv), per-scan model outputs and DICOM study and series identifiers, all under CC BY-NC-SA 4.0 |
| BHX v1.1 (PhysioNet) | CC BY-SA 4.0 (see the PhysioNet project page) | BHX box counts per CQ500 scan, under CC BY-SA 4.0 |
| PHE-SICH-CT-IDS | CC BY 4.0 | per-patient outputs, under CC BY 4.0 |
| BleedScene3D (version 2) | as released by its creators | counts and agreement statistics only |

Cite the datasets as listed in the manuscript (Data availability) and comply with each license; files that contain CQ500 labels,
identifiers or outputs must be shared under CC BY-NC-SA 4.0 with attribution to Qure.ai.
