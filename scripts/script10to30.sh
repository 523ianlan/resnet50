#!/bin/bash
# run_experiments_fixed.sh

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

BASE_DIR="D:/ImageNet_Organized"
TRAIN_PATH="${BASE_DIR}/train"
VAL_PATH="${BASE_DIR}/validation"
OUTPUT_DIR="./results"

MC_SAMPLES=20
MC_DROPOUT_P=0.1
CALIB_BATCHES=10
FISHER_BATCHES=100
FINE_TUNE_EPOCHS=1
freeze_epoch=27
BASE_LR=5e-5
FINE_TUNE_LR=5e-5
FINE_TUNE_WD=1e-4
WARMUP_EPOCHS=0
LAYER_DECAY=1.0
BATCH_SIZE=256
NUM_WORKERS=8
MIXED_PRECISION=1
EVAL_MAX_BATCHES=0
TRAIN_MAX_BATCHES=0
SEED=42

# experiment parameters
declare -a COMPRESSIONS=(0.1 0.2 0.3)
declare -a PRUNING_LOWS=(0.0 0.0 0.1)
declare -a PRUNING_HIGHS=(0.3 0.4 0.5)

mkdir -p "$OUTPUT_DIR"

for i in "${!COMPRESSIONS[@]}"; do
    COMP=${COMPRESSIONS[$i]}
    COMP_PCT=$(printf "%.0f" $(echo "$COMP * 100" | bc 2>/dev/null || awk "BEGIN {printf \"%d\", $COMP * 100}"))
    CURRENT_LOW=${PRUNING_LOWS[$i]}
    CURRENT_HIGH=${PRUNING_HIGHS[$i]}

    EXP_TIME=$(date +%Y%m%d_%H%M%S 2>/dev/null)
    if [ -z "$EXP_TIME" ]; then
        EXP_TIME=$(python - <<'PY'
import time
print(time.strftime("%Y%m%d_%H%M%S"))
PY
)
    fi

    EXP_NAME="comp${COMP_PCT}_ft${FINE_TUNE_EPOCHS}_low${CURRENT_LOW}_high${CURRENT_HIGH}_result_${EXP_TIME}"
    EXP_DIR="$OUTPUT_DIR/$EXP_NAME"

    echo "========================================="
    echo "exe: $EXP_NAME"
    echo "pr: ${COMP_PCT}%"
    echo "pruning_clip_low: $CURRENT_LOW"
    echo "pruning_clip_high: $CURRENT_HIGH"
    echo "fine_tune_epochs: $FINE_TUNE_EPOCHS"
    echo "batch_size: $BATCH_SIZE"
    echo "num_workers: $NUM_WORKERS"
    echo "mixed_precision: $MIXED_PRECISION"
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

    python ../main.py \
        --train-root "$TRAIN_PATH" \
        --val-root "$VAL_PATH" \
        --target-compression $COMP \
        --pruning-clip-low $CURRENT_LOW \
        --pruning-clip-high $CURRENT_HIGH \
        --mc-samples $MC_SAMPLES \
        --mc-dropout-p $MC_DROPOUT_P \
        --calib-batches $CALIB_BATCHES \
        --fisher-batches $FISHER_BATCHES \
        --fine-tune-epochs $FINE_TUNE_EPOCHS \
        --base-lr $BASE_LR \
        --fine-tune-lr $FINE_TUNE_LR \
        --fine-tune-weight-decay $FINE_TUNE_WD \
        --warmup-epochs $WARMUP_EPOCHS \
        --layer-decay $LAYER_DECAY \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        $MP_ARG \
        $EVAL_ARG \
        $TRAIN_ARG \
        --save-dir "$EXP_DIR" \
        --seed $SEED

    echo "done: $EXP_NAME"
    echo "save to: $EXP_DIR"
    echo ""
done
echo "result save to: $OUTPUT_DIR"
# cd "D:\UFALP\resnet50\scripts"
# $env:Path
# $env:Path = "C:\Program Files\Git\bin;$env:Path"
# bash script10to30.sh