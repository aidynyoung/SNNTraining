#!/bin/bash
# Quick SNN training run on CPU (or Apple MPS on Macbooks).
# Exercises the full train.py code path: LIF → RSNN → HDC pipeline.
# Should finish in under 30 seconds with no GPU required.
#
# Run as:
#   bash runs/train_cpu.sh

set -e

echo "--- [1/4] Prototype bundling (fastest, single pass) ---"
python train.py \
    --task synthetic \
    --method bundle \
    --hidden 64 \
    --dim 1024 \
    --classes 4

echo ""
echo "--- [2/4] Prototype refinement (RefineHD, push/pull) ---"
python train.py \
    --task synthetic \
    --method refine \
    --hidden 64 \
    --dim 1024 \
    --classes 4 \
    --epochs 3

echo ""
echo "--- [3/4] SNN fallback readout (HDC + delta-rule linear readout) ---"
python train.py \
    --task synthetic \
    --method fallback \
    --hidden 64 \
    --dim 1024 \
    --classes 4

echo ""
echo "--- [4/4] LeHDC gradient training with STE binarization ---"
python train.py \
    --task synthetic \
    --method lehdc \
    --hidden 64 \
    --dim 1024 \
    --classes 4 \
    --epochs 5

echo ""
echo "All CPU runs complete."
