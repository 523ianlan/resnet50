#!/bin/bash
# run_paper_tables.sh
# Paper-ready ResNet-50 runs

set -e

CONFIG="./experiments/paper_runs/paper_config.json"
ROOT_RESULTS="./experiments/paper_tables/results"

mkdir -p "$ROOT_RESULTS"

# echo "================================================================"
# echo "Experiment 1: Two-Stage Framework Ablation (UFALP vs WO_FISHER)"
# echo "================================================================"
# python experiments/paper_tables/run_table_ablation_two_stage.py --target-compression 0.6 \
#   --config "$CONFIG" --save-root "$ROOT_RESULTS/ablation_two_stage_60pr"

# echo "================================================================"
# echo "Experiment 2: Scoring Metrics Ablation (Fisher vs Taylor vs Energy vs Magnitude)"
# echo "================================================================"
# python experiments/paper_tables/run_table_ablation_scoring.py --target-compression 0.6 \
#   --config "$CONFIG" --save-root "$ROOT_RESULTS/ablation_scoring_60pr"

# echo "================================================================"
# echo "Experiment 3: Budget Allocation Ablation (UFALP vs Global Fisher vs Uniform)"
# echo "================================================================"
# python experiments/paper_tables/run_table_ablation_allocation.py --target-compression 0.6 \
#   --config "$CONFIG" --save-root "$ROOT_RESULTS/ablation_allocation_60pr"

# echo "================================================================"
# echo "All 60% pruning ablation runs completed."
# echo "Results saved in subdirectories of: $ROOT_RESULTS"
# echo "================================================================"

# echo "Custom 60% pruning targeted runs completed. Results in: $ROOT_RESULTS/ablation_two_stage_60pr"

# python experiments/paper_tables/run_table_ablation_two_stage.py --target-compression 0.4 \
#   --config "$CONFIG" --save-root "$ROOT_RESULTS/ablation_two_stage"

# python experiments/paper_tables/run_table_ablation_scoring.py --target-compression 0.4 \
#   --config "$CONFIG" --save-root "$ROOT_RESULTS/ablation_scoring"

# python experiments/paper_tables/run_table_ablation_log_s.py --target-compression 0.4 \
#   --config "$CONFIG" --save-root "$ROOT_RESULTS/ablation_log_s"

# python experiments/paper_tables/run_table_ablation_allocation.py --target-compression 0.4 \
#   --config "$CONFIG" --save-root "$ROOT_RESULTS/ablation_allocation"

# python experiments/paper_tables/run_table_ranking_correlation.py --max-layers 10 --max-components 16 --oracle-batches 1 \
#   --config "$CONFIG" --save-root "$ROOT_RESULTS/ranking_correlation"

# echo "All paper table runs completed. Results in: $ROOT_RESULTS"

# echo "================================================================"
# echo "Experiment 9: Uncertainty Normalization Ablation (log / clip / var_floor)"
# echo "================================================================"
# python experiments/paper_tables/run_table_ablation_uncertainty_norm.py --target-compression 0.6 \
#   --clip-percentile 1.0 --var-floor 1e-6 \
#   --config "$CONFIG" --save-root "$ROOT_RESULTS/ablation_uncertainty_norm_60pr"

echo "================================================================"
echo "Experiment 10: Extreme Test (80% pruning, 90-epoch FT)"
echo "================================================================"
python experiments/paper_tables/run_table_extreme_80pr_90ft.py --target-compression 0.8 \
  --fine-tune-epochs 90 \
  --config "$CONFIG" --save-root "$ROOT_RESULTS/extreme_80pr_90ft"
