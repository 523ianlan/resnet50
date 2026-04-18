# run_paper_tables.ps1
# Paper-ready ResNet-50 runs (excluding MobileNetV2)

$Config = ".\experiments\paper_runs\paper_config.json"
$Root = ".\experiments\paper_tables\results"

New-Item -ItemType Directory -Force -Path $Root | Out-Null

python experiments/paper_tables/run_table_ablation_two_stage.py --target-compression 0.4 `
  --config $Config --save-root "$Root\ablation_two_stage"

python experiments/paper_tables/run_table_ablation_scoring.py --target-compression 0.4 `
  --config $Config --save-root "$Root\ablation_scoring"

python experiments/paper_tables/run_table_ablation_log_s.py --target-compression 0.4 `
  --config $Config --save-root "$Root\ablation_log_s"

python experiments/paper_tables/run_table_ablation_allocation.py --target-compression 0.4 `
  --config $Config --save-root "$Root\ablation_allocation"

python experiments/paper_tables/run_table_ranking_correlation.py --max-layers 10 --max-components 16 --oracle-batches 1 `
  --config $Config --save-root "$Root\ranking_correlation"

Write-Host "All paper table runs completed. Results in: $Root"
