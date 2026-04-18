# Coverage Report

This report maps the requested paper tables to experiment scripts in `experiments/paper_tables`.

## Covered (ResNet-50)
- Ablation 1 (Two-stage necessity): `run_table_ablation_two_stage.py`
- Ablation 2 (Scoring function): `run_table_ablation_scoring.py`
- Ablation 4 (Log-singular parameterization): `run_table_ablation_log_s.py`
- Ablation 5 (Allocation strategies): `run_table_ablation_allocation.py`
- Ranking correlation: `run_table_ranking_correlation.py`
- Flat spectrum correlation: `run_table_ranking_correlation.py` (same report includes flat/decaying split)

## Not Yet Implemented
- Ablation 3 (MobileNetV2 rank-only vs rank+channel): `run_table_ablation_mobilenet.py` is a placeholder.
  This repository does not include MobileNetV2 or channel-pruning code.

## Legacy Results
- `experiments/paper_tables/legacy_fisher_results/` contains migrated `fisher_comparison_results` and `fisher_comparison_results_detailed`.
