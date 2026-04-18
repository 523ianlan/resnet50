# UFALP Batch/Convergence Sweeps

This folder contains Python sweep scripts to evaluate convergence with different batch-related hyperparameters.

Scripts:
- `experiments/batch_sweep/run_calib_batch_sweep.py`
- `experiments/batch_sweep/run_fisher_batch_sweep.py`
- `experiments/batch_sweep/run_calib_size_sweep.py`

Each run writes outputs into its own subfolder under `experiments/batch_sweep/results/*` and appends a `summary.csv`.

Examples:
```bash
python experiments/batch_sweep/run_calib_batch_sweep.py --calib-batches 5,10,20 --fisher-batches 100
python experiments/batch_sweep/run_fisher_batch_sweep.py --fisher-batches 25,50,100,200 --calib-batches 10
python experiments/batch_sweep/run_calib_size_sweep.py --calib-samples 1000,5000,10000 --use-val-transform
```

Notes:
- If you want D_cal to differ from training data, use `--calib-samples` (or `calib_split_ratio` in config) and set `--exclude-from-train` if needed.
- `summary.csv` extracts key metrics from `complete_pruning_report.txt`.
