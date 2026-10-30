# Replication with a high-performing public model (CQ500)

These are the scripts exactly as they were run for the post hoc replication (Supplementary Methods S2). Paths are those of the
cloud workspace in which they ran (`/home/claude/work/ianpan`, `/home/claude/data/cq500_tmp`, the locked per-scan CSV of the
v2 evaluation); adjust them before re-running.

* `PROTOCOL_LOCK_public_model_CQ500.json`: the analysis plan, written and hashed (SHA-256) on 28 October 2026 at 22:48 UTC,
  before evaluation outputs were generated. It records the model revision, the SHA-256 of every weight file and of the released
  `inference.py`, the input, the reference, the scores, the thresholds and endpoints E1-E8.
* `requirements.txt`, `requirements-frozen.txt`: packages (CPU build of PyTorch 2.6.0; timm 1.0.27; pydicom 3.0.1).
* `run_cq500.py`: runs the model on the 489 CQ500 scans with the locked series rule (`cq500_select.py`), one archive at a time
  from the official CQ500 host; the archive and images are deleted after each scan, and only the outputs and DICOM identifiers are
  kept (one JSON per scan). Resumable; stops after a time budget. The first 10 scans were run with `batch_size=2` (as locked);
  after the process was stopped for lack of memory, `batch_size=1` was used for the other 479 (see `batch_check.py`).
* `bench.py`: the timing run on CQ500-CT-108 made before the lock.
* `batch_check.py`: re-runs scans of the first group with `batch_size=1` and compares every stored output.
* `fetch_uids_range.py`: reads DICOM study and series identifiers from the released archives by HTTP range requests (directory and
  the first 64 KB of one file per series); used for the 14 scans processed before identifiers were recorded and to build the
  identifier table of all 489 scans.

The model itself is not redistributed. Obtain it from its author (MIT License):
`huggingface-cli download ianpan/ct-head-hemorrhage-detection --revision e6d89c253fdcc2a597dd0fd0f5771821f4947b37 --local-dir weights`,
check the SHA-256 values in the protocol lock, and place the released `inference.py` (SHA-256
9792e50ad1d863e80a549d391ee53c5031d490abb75acba42d471b02a83fbafb, unchanged) next to the scripts. The per-scan outputs are collected with
`../collect_public_model.py` and analyzed with `../../analysis/public_model_cq500.py`.
