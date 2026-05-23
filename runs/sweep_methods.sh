#!/bin/bash
# Compare all four training methods on synthetic data.
# Runs bundle → refine → fallback → lehdc, prints accuracy for each.
# Use this to decide which method to invest in for your task.
#
# Rule of thumb:
#   bundle:   fastest, ~80-85%     — start here
#   refine:   +3-6% over bundle    — if accuracy plateaus
#   fallback: best for temporal    — use_snn_fallback=True
#   lehdc:    +5-10% on hard tasks — gradient + STE binarization
#
# Run as:
#   bash runs/sweep_methods.sh [n_classes]

set -e

CLASSES=${1:-4}
HIDDEN=128
DIM=2048
EPOCHS=5

echo "=== Method sweep: ${CLASSES} classes, hidden=${HIDDEN}, dim=${DIM} ==="
echo ""

echo "--- bundle (single pass, no optimizer) ---"
python train.py \
    --task synthetic --method bundle \
    --classes $CLASSES --hidden $HIDDEN --dim $DIM

echo ""
echo "--- refine (class-mean init + push/pull, ${EPOCHS} epochs) ---"
python train.py \
    --task synthetic --method refine \
    --classes $CLASSES --hidden $HIDDEN --dim $DIM \
    --epochs $EPOCHS

echo ""
echo "--- fallback (HDC + SNN delta-rule readout) ---"
python train.py \
    --task synthetic --method fallback \
    --classes $CLASSES --hidden $HIDDEN --dim $DIM

echo ""
echo "--- lehdc (gradient + STE binarization, ${EPOCHS} epochs) ---"
python train.py \
    --task synthetic --method lehdc \
    --classes $CLASSES --hidden $HIDDEN --dim $DIM \
    --epochs $EPOCHS

echo ""
echo "Sweep complete. Check accuracy above to choose your method."
