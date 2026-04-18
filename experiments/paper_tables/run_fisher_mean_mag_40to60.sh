#!/bin/bash
# run_fisher_mean_mag_40to60.sh
# Budget: Fisher mean (global_fisher)
# Selection: Magnitude (stage2-score-metric magnitude)

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

BASE_DIR="D:/ImageNet_Organized"
TRAIN_PATH="${BASE_DIR}/train"
VAL_PATH="${BASE_DIR}/validation"
# 輸出到指定的混合策略資料夾
OUTPUT_DIR="experiments/paper_tables/results/FISHER mean with MAG selection result"

MC_SAMPLES=20
MC_DROPOUT_P=0.1
CALIB_BATCHES=10
CALIB_SAMPLES=5000
CALIB_SEED=42
CALIB_USE_VAL_TRANSFORM=1
CALIB_EXCLUDE_FROM_TRAIN=1
FISHER_BATCHES=100

# User requested fine-tuning parameters
FINE_TUNE_EPOCHS=90
BASE_LR=1e-4
FINE_TUNE_LR=1e-4
FREEZE_EPOCH=88    # 90 - 2 = 88
FREEZE_LR=5e-7
MIN_LR_RATIO=0.005 # 5e-7 / 1e-4 = 0.005

FINE_TUNE_WD=1e-4
WARMUP_EPOCHS=0
LAYER_DECAY=1.0
BATCH_SIZE=128
NUM_WORKERS=8
MIXED_PRECISION=1
EVAL_MAX_BATCHES=0
TRAIN_MAX_BATCHES=0
SEED=42
DETERMINISTIC=1
CUDNN_BENCHMARK=0
UNCERTAINTY_LOG=1
UNCERTAINTY_CLIP_PERCENTILE=0
UNCERTAINTY_VAR_FLOOR=0

# experiment parameters prune 40%-60%
# declare -a COMPRESSIONS=(0.4 0.5 0.6)
# declare -a PRUNING_LOWS=(0.0 0.1 0.2)
# declare -a PRUNING_HIGHS=(0.6 0.7 0.8)

declare -a COMPRESSIONS=(0.6)
declare -a PRUNING_LOWS=(0.2)
declare -a PRUNING_HIGHS=(0.8)

mkdir -p "$OUTPUT_DIR"

for i in "${!COMPRESSIONS[@]}"; do
    COMP=${COMPRESSIONS[$i]}
    COMP_PCT=$(printf "%.0f" $(echo "$COMP * 100" | bc 2>/dev/null || awk "BEGIN {printf \"%d\", $COMP * 100}"))
    CURRENT_LOW=${PRUNING_LOWS[$i]}
    CURRENT_HIGH=${PRUNING_HIGHS[$i]}

    EXP_TIME=$(date +%Y%m%d_%H%M%S 2>/dev/null || python -c "import time; print(time.strftime('%Y%m%d_%H%M%S'))")

    EXP_NAME="comp${COMP_PCT}_ft${FINE_TUNE_EPOCHS}_low${CURRENT_LOW}_high${CURRENT_HIGH}_mag_result_${EXP_TIME}"
    EXP_DIR="$OUTPUT_DIR/$EXP_NAME"

    echo "========================================="
    echo "exe: $EXP_NAME"
    echo "pr: ${COMP_PCT}%"
    echo "budget_strategy: global_fisher (fisher mean)"
    echo "selection_metric: magnitude"
    echo "fine_tune_epochs: $FINE_TUNE_EPOCHS"
    echo "========================================="

    MP_ARG=""
    if [ "$MIXED_PRECISION" = "1" ]; then
        MP_ARG="--mixed-precision"
    fi

    EVAL_ARG=""
    if [ "$EVAL_MAX_BATCHES" -gt 0 ]; then
        EVAL_ARG="--eval-max-batches $EVAL_MAX_BATCHES"
    fi
    TRAIN_ARG=""
    if [ "$TRAIN_MAX_BATCHES" -gt 0 ]; then
        TRAIN_ARG="--train-max-batches $TRAIN_MAX_BATCHES"
    fi
    CALIB_VAL_ARG=""
    if [ "$CALIB_USE_VAL_TRANSFORM" = "1" ]; then
        CALIB_VAL_ARG="--calib-use-val-transform"
    fi
    CALIB_EXCLUDE_ARG=""
    if [ "$CALIB_EXCLUDE_FROM_TRAIN" = "1" ]; then
        CALIB_EXCLUDE_ARG="--calib-exclude-from-train"
    fi
    DET_ARG=""
    if [ "$DETERMINISTIC" = "1" ]; then
        DET_ARG="--deterministic --no-cudnn-benchmark"
    fi

    python main.py \
        --train-root "$TRAIN_PATH" \
        --val-root "$VAL_PATH" \
        --target-compression $COMP \
        --pruning-clip-low $CURRENT_LOW \
        --pruning-clip-high $CURRENT_HIGH \
        --mc-samples $MC_SAMPLES \
        --mc-dropout-p $MC_DROPOUT_P \
        --calib-batches $CALIB_BATCHES \
        --calib-samples $CALIB_SAMPLES \
        --calib-seed $CALIB_SEED \
        --fisher-batches $FISHER_BATCHES \
        --fine-tune-epochs $FINE_TUNE_EPOCHS \
        --freeze-epoch $FREEZE_EPOCH \
        --freeze-lr $FREEZE_LR \
        --min-lr-ratio $MIN_LR_RATIO \
        --base-lr $BASE_LR \
        --fine-tune-lr $FINE_TUNE_LR \
        --fine-tune-weight-decay $FINE_TUNE_WD \
        --warmup-epochs $WARMUP_EPOCHS \
        --layer-decay $LAYER_DECAY \
        --allocation-strategy "global_fisher" \
        --stage2-score-metric "magnitude" \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --no-prefetch-to-gpu \
        --no-channels-last \
        --uncertainty-clip-percentile $UNCERTAINTY_CLIP_PERCENTILE \
        --uncertainty-var-floor $UNCERTAINTY_VAR_FLOOR \
        --uncertainty-log \
        $MP_ARG \
        $EVAL_ARG \
        $TRAIN_ARG \
        $CALIB_VAL_ARG \
        $CALIB_EXCLUDE_ARG \
        $DET_ARG \
        --save-dir "$EXP_DIR" \
        --seed $SEED

    echo "done: $EXP_NAME"
    echo "save to: $EXP_DIR"
    echo ""
done
echo "result save to: $OUTPUT_DIR"
