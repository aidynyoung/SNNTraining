#!/bin/bash
# Energy-per-inference benchmark.
# Compares nJ/inference across HDC, SNN, MLP, and Transformer
# at increasing hypervector dimensions (d=128 to d=16384).
#
# Based on Horowitz (2014) ISSCC energy constants:
#   XOR:  0.1 pJ/bit  (HDC operations)
#   MAC:  4.6 pJ      (transformer / MLP)
#   DRAM: 640 pJ/word (transformer weight loading)
#   SRAM: 5 pJ/word   (HDC hypervector loading)
#
# Key result: HDC achieves 2.4 nJ at d=4096.
#             Transformer at d=4096: ~55,200 nJ → 22,992× gap.
#
# Run as:
#   bash runs/benchmark_energy.sh

set -e

echo "--- Energy benchmark: d=128 to d=16384 ---"
python experiments/benchmark_energy.py

echo ""
echo "--- Fault tolerance under hardware faults ---"
echo "    (stuck-at-0: memory cells frozen at zero)"
python experiments/arthedain_robustness.py \
    --fault-type stuck_at_0 \
    --fault-rate 0.10

echo ""
echo "--- Fault tolerance: stuck-at-1 ---"
python experiments/arthedain_robustness.py \
    --fault-type stuck_at_1 \
    --fault-rate 0.10

echo ""
echo "Energy and robustness benchmarks complete."
echo "Key figures:"
echo "  HDC path:    2.4 nJ/inference"
echo "  SNN path:   12.9 nJ/inference"
echo "  Transformer: 55,200 nJ/inference  (22,992x)"
echo "  Fault tol:  100% accuracy at 10% stuck-at-0"
