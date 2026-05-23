#!/bin/bash
# SHD neuromorphic speech benchmark.
# Spiking Heidelberg Digits — 20-class spoken digit classification
# from spike trains recorded from a silicon cochlea model.
#
# Requirements:
#   pip install h5py
#   Dataset auto-downloads from https://zenkelab.org/resources/spiking-heidelberg-datasets/
#   (or set SHD_PATH env var to an existing local .h5 file)
#
# Expected results (O(1) memory, no backprop):
#   RSNN + HDC pipeline: ~78%   (our method)
#   e-prop (Bellec 2020): ~82%  (uses eligibility traces)
#   BPTT SNN (Cramer 2020): ~91% (offline, stores full rollout)
#
# Run as:
#   bash runs/benchmark_shd.sh

set -e

echo "--- [1/3] RSNN + HDC pipeline (hidden=128) ---"
python experiments/benchmark_neuromorphic.py \
    --hidden 128 \
    --seed 42

echo ""
echo "--- [2/3] RSNN + HDC pipeline (hidden=256, better accuracy) ---"
python experiments/benchmark_neuromorphic.py \
    --hidden 256 \
    --seed 42

echo ""
echo "--- [3/3] Multi-seed run for mean ± std ---"
python experiments/seed_benchmark.py \
    --task shd \
    --seeds 5 \
    --hidden 256

echo ""
echo "SHD benchmark complete. Target: ~78% with hidden=256."
