# Paper-Ready Runs (ResNet-50)

This folder provides paper-ready command sequences that match the experimental setup you specified.

## Config file
Use the provided paper config:
- `experiments/paper_runs/paper_config.json`

Key settings:
- ImageNet-1K, pretrained
- Fine-tune 30 epochs
- SGD momentum 0.9, WD 1e-4
- Cosine LR, base LR 5e-3
- Deterministic on
- D_cal enabled (5000 samples, val transform, excluded from train)

## Core tables (ResNet-50)
```powershell
python experiments/paper_tables/run_table_ablation_two_stage.py --target-compression 0.4 ^
  --config .\experiments\paper_runs\paper_config.json --save-root .\experiments\paper_tables\results\ablation_two_stage

python experiments/paper_tables/run_table_ablation_scoring.py --target-compression 0.4 ^
  --config .\experiments\paper_runs\paper_config.json --save-root .\experiments\paper_tables\results\ablation_scoring

python experiments/paper_tables/run_table_ablation_log_s.py --target-compression 0.4 ^
  --config .\experiments\paper_runs\paper_config.json --save-root .\experiments\paper_tables\results\ablation_log_s

python experiments/paper_tables/run_table_ablation_allocation.py --target-compression 0.4 ^
  --config .\experiments\paper_runs\paper_config.json --save-root .\experiments\paper_tables\results\ablation_allocation
```

## Ranking correlation + flat spectrum
```powershell
python experiments/paper_tables/run_table_ranking_correlation.py --max-layers 10 --max-components 16 --oracle-batches 1 ^
  --config .\experiments\paper_runs\paper_config.json --save-root .\experiments\paper_tables\results\ranking_correlation
```

## Notes
- MobileNetV2 + UFALP-H is not implemented in this repo.
- Increase `--oracle-batches` or `--max-components` for stronger correlation estimates (slower).
