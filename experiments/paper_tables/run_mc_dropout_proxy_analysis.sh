#!/bin/bash
# Layer-wise MC-dropout proxy analysis:
# U_l(r) vs D_rand_l(rho) vs D_rank_l(rho)

set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

BASELINE_CONFIG="experiments/paper_tables/results/extreme_80pr_90ft/global_fisher/r50_80pr_90ft/config.json"
SAVE_ROOT="experiments/paper_tables/results/mc_dropout_proxy_analysis"

if [[ ! -f "$BASELINE_CONFIG" ]]; then
  echo "Baseline config not found: $BASELINE_CONFIG"
  exit 1
fi

python experiments/paper_tables/run_table_mc_dropout_proxy_analysis.py \
  --config "$BASELINE_CONFIG" \
  --save-root "$SAVE_ROOT" \
  --tag "r50_80pr_fisher_vs_svd_taylor" \
  --layer-mode "all" \
  --max-layers 12 \
  --dropout-grid "0.05,0.1,0.15,0.2" \
  --prune-grid "0.05,0.1,0.15,0.2" \
  --mc-samples-grid "10,20,30" \
  --analysis-mc-samples 20 \
  --random-trials 8 \
  --probe-batches 2 \
  --eval-batches 2 \
  --fisher-batches 50 \
  --num-workers 0 \
  --rank-methods "fisher,svd,taylor" \
  --seed 42
