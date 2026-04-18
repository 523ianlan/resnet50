# Paper Table Experiments

This folder mirrors the tables you listed for the paper and provides scripts to run each table as a controlled experiment.

## Table Scripts (ResNet-50)
- `run_table_ablation_two_stage.py`
  - Variants: w/o uncertainty, w/o Fisher, full UFALP
- `run_table_ablation_scoring.py`
  - Variants: magnitude, energy, Taylor, Hessian diag, Fisher
- `run_table_ablation_log_s.py`
  - Variants: direct sigma vs log-sigma
- `run_table_ablation_allocation.py`
  - Variants: uniform, global Fisher, UFALP
- `run_table_ranking_correlation.py`
  - Produces ranking correlation (Spearman/Kendall) and flat vs decaying spectrum splits
- `run_table_mc_dropout_proxy_analysis.py`
  - Layer-wise protocol: MC-dropout probing (`U_l(r)`), random pruning distortion (`D_rand`), ranking distortion (`D_rank`)
  - Computes ranking gain, normalized gain, proxy correlation, proxy-real gap, integrated efficiency, intrinsic robustness
  - Writes curves/heatmaps/scatter plots under `visualizations/`

## MobileNetV2
- `run_table_ablation_mobilenet.py` is a placeholder. MobileNetV2 + UFALP-H (rank+channel) is not implemented in this codebase.

## Output
Each script writes:
- `complete_pruning_report.txt` in each experiment folder
- `ablation_overrides.json` in each experiment folder
- `summary.csv` in the suite root

`run_table_mc_dropout_proxy_analysis.py` writes:
- `mc_dropout_proxy_report.json`
- `layer_summary.csv`
- `analysis_summary.txt`
- plot images under `visualizations/`

Plot helper:
- `plot_summary.py --summary <path/to/summary.csv>`

## Notes
These scripts apply the paper training defaults (30 epochs, SGD, LR 5e-3, WD 1e-4). If you want to keep current training defaults, edit `common.py`.
