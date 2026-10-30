#!/bin/bash
# Regenerate every analysis output, Supplementary Data 1 and the figures from the per-scan inputs.
# Usage: bash run_all.sh [analysis|figures|all]   (default: all)
# Outputs: out/ (analysis and Supplementary Data 1), figures/out/ (figures). The outputs reported in the article are
# in out_delivered/ for comparison (python analysis/compare_outputs.py out out_delivered).
# Optional raw-data steps (otherwise the delivered results are used): BLEEDSCENE_RAW (BleedScene3D raw/ folder), BHX_DIR (BHX v1.1
# CSV files), SEGCQ500_ZIP (Seg-CQ500 archive; needs nibabel), MBHSEG_TRAIN_ZIP (MBH-Seg 2025 training archive
# MBH_Train_2025_voxel-label.zip; needs nibabel).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NEUROCARE_WORK="$ROOT"
export MPLBACKEND=Agg
STEP="${1:-all}"
if [[ "$STEP" == analysis || "$STEP" == all ]]; then
  cd "$ROOT/analysis"
  run() { echo "== analysis/$1.py"; python3 "$1.py" "${@:2}" > "$ROOT/out_log_$1.txt" 2>&1 || { echo "FAILED: $1 (see out_log_$1.txt)"; exit 1; }; }
  for s in verify_original new_analyses new_analyses_seg extra_checks extra_checks2 extra_checks3 id_checks kappa491 side_agreement kappa_model \
           thresholds_clinical build_registry table2_data posthoc_analyses; do
    run "$s"
  done
  if [[ -n "${BLEEDSCENE_RAW:-}" ]]; then
    run bleedscene_compare "$BLEEDSCENE_RAW/CQ500/psgg"; run bleedscene_counts "$BLEEDSCENE_RAW"
  else
    cp "$ROOT/out_delivered/bleedscene_cq500.json" "$ROOT/out_delivered/bleedscene_counts.json" "$ROOT/out/"
    cp "$ROOT/out_delivered/tables/bleedscene_cq500.csv" "$ROOT/out/tables/"
  fi
  if [[ -n "${MBHSEG_TRAIN_ZIP:-}" ]]; then
    run mbhseg_train_raters "$MBHSEG_TRAIN_ZIP"          # MBH-Seg 2025 training release (MBH_Train_2025_voxel-label.zip; needs nibabel)
  else
    cp "$ROOT/out_delivered/mbhseg_train_raters.json" "$ROOT/out/"; cp "$ROOT/out_delivered/tables/mbhseg_train_raters.csv" "$ROOT/out/tables/"
  fi
  run capacity_rubric                                   # first version of the rubric (compared in a supplementary table)
  if [[ -n "${BHX_DIR:-}" ]]; then run bhx_scan_labels "$BHX_DIR"; run bhx_extent "$BHX_DIR"; fi
  run bhx_exact
  run ledger
  run public_model_cq500
  run reference_choice
  run split_extent
  run volume_robust
  run capacity_rubric_v2
  run adjudication_sample
  if [[ -n "${SEGCQ500_ZIP:-}" ]]; then
    run segcq500_mask_check "$SEGCQ500_ZIP" "$ROOT/data/derived/segcq500_analyzed_ids.txt" "$ROOT/data/derived"
    run segcq500_mask_hist "$SEGCQ500_ZIP" "$ROOT/data/derived/segcq500_analyzed_ids.txt" "$ROOT/data/derived" "$ROOT/figures/assets"
    run segcq500_mask_shape "$SEGCQ500_ZIP" "$ROOT/data/derived/segcq500_analyzed_ids.txt" "$ROOT/data/derived"
  fi
  python3 make_supp_data1_mr.py "$ROOT/out/Supplementary_Data_1.xlsx"
fi
if [[ "$STEP" == figures || "$STEP" == all ]]; then
  cd "$ROOT/figures"
  for s in fig1_design_mr.py fig2_capacity_mr.py fig3_reference_mr.py fig4_examples_mr.py fig5_detection_mr.py fig6_volume_mr.py fig7_side_midline_mr.py fig8_fracture_edh_mr.py sf_*.py; do
    echo "== figures/$s"; python3 "$s" > /dev/null
  done
fi
echo "done"
