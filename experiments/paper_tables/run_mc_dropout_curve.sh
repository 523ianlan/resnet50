#!/bin/bash
# MC Dropout curve only (no random/ranking pruning)

set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

BASELINE_CONFIG="experiments/paper_tables/results/extreme_80pr_90ft/global_fisher/r50_80pr_90ft/config.json"
SAVE_ROOT="experiments/paper_tables/results"

if [[ ! -f "$BASELINE_CONFIG" ]]; then
  echo "Baseline config not found: $BASELINE_CONFIG"
  exit 1
fi

python experiments/paper_tables/run_table_mc_dropout_curve.py \
  --config "$BASELINE_CONFIG" \
  --save-root "$SAVE_ROOT" \
  --tag "mc_curve_only" \
  --layer-mode "all" \
  --max-layers 53 \
  --dropout-grid "0.05,0.1,0.15,0.2" \
  --mc-samples-grid "10,20,30" \
  --probe-batches 2 \
  --num-workers 0 \
  --seed 42

