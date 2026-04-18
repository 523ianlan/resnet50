# UFALP FAST Ablations

These scripts run the same ablations as `experiments/ablations`, but with a fast configuration:
- fine_tune_epochs = 1
- train_max_batches = 200
- eval_max_batches = 50
- calib_batches = 5
- fisher_batches = 20

Scripts:
- `run_stage1_stability_ablation_fast.py`
- `run_stage1_metric_ablation_fast.py`
- `run_stage1_alpha_ablation_fast.py`
- `run_stage2_scoring_ablation_fast.py`
- `run_stage2_first_order_ablation_fast.py`

Example:
```bash
python experiments/ablations_fast/run_stage1_stability_ablation_fast.py
```
