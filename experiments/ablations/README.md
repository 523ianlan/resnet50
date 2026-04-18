# UFALP Ablation Experiments

This folder contains ablation suites for Stage 1 and Stage 2.

Scripts:
- `experiments/ablations/run_stage1_stability_ablation.py`
- `experiments/ablations/run_stage1_metric_ablation.py`
- `experiments/ablations/run_stage1_alpha_ablation.py`
- `experiments/ablations/run_stage2_scoring_ablation.py`
- `experiments/ablations/run_stage2_first_order_ablation.py`

Each script runs multiple variants and writes:
- `complete_pruning_report.txt` in each experiment directory
- `ablation_overrides.json` in each experiment directory
- `summary.csv` in the suite root

Examples:
```bash
python experiments/ablations/run_stage1_stability_ablation.py
python experiments/ablations/run_stage1_metric_ablation.py --metrics mu_over_var,mu,var
python experiments/ablations/run_stage1_alpha_ablation.py --alphas 0,0.5,1.0
python experiments/ablations/run_stage2_scoring_ablation.py
python experiments/ablations/run_stage2_first_order_ablation.py
```
