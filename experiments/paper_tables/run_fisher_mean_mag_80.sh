#!/bin/bash
# run_fisher_mean_mag_80.sh
# Fair-control ablation against:
# experiments/paper_tables/results/extreme_80pr_90ft/global_fisher/r50_80pr_90ft/config.json
# Only change: stage2-score-metric = magnitude

set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

BASELINE_CONFIG="experiments/paper_tables/results/extreme_80pr_90ft/global_fisher/r50_80pr_90ft/config.json"
OUTPUT_ROOT="experiments/paper_tables/results/extreme_80pr_magnitude_ablation"
SEED=42

if [[ ! -f "$BASELINE_CONFIG" ]]; then
    echo "Baseline config not found: $BASELINE_CONFIG"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

EXP_TIME=$(python -c "import time; print(time.strftime('%Y%m%d_%H%M%S'))")
EXP_NAME="comp80_ft90_mag_ablation_${EXP_TIME}"
EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"

echo "========================================="
echo "ABLATION: 80% Fisher-Mean Budget + Magnitude Selection"
echo "Baseline config: $BASELINE_CONFIG"
echo "Output: $EXP_DIR"
echo "Only override: --stage2-score-metric magnitude"
echo "========================================="

python main.py \
    --config "$BASELINE_CONFIG" \
    --stage2-score-metric "magnitude" \
    --save-dir "$EXP_DIR" \
    --seed "$SEED"

echo "Ablation run completed: $EXP_NAME"
