#!/bin/bash
# resume_fisher.sh
# Resume the interrupted fine-tuning for ResNet-50

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

echo "Starting resumption of fine-tuning..."
python resume_run.py
echo "Done."
