#!/bin/bash
# ==============================================================================
# sweep_fault_types.sh — Run artainedain_robustness.py across all SpikeFI fault types
# ==============================================================================
# This script runs the full robustness benchmark for every fault type in the
# SpikeFI taxonomy (Spyrou et al. 2024, arXiv:2412.06795):
#   - stuck_at_0:   Weights permanently set to 0
#   - stuck_at_1:   Weights permanently set to 1
#   - wbf_t:        Transient weight bit-flip
#   - wbf_p:        Permanent weight bit-flip
#   - syn_silence:  Synaptic silence (weight permanently 0)
#
# For each fault type, runs: classification and regression tasks.
# Generates results files in results/arthedain_robustness_{fault_type}_{task}.json
#
# Usage:
#   bash experiments/sweep_fault_types.sh           # Full sweep (~30-60 min)
#   bash experiments/sweep_fault_types.sh --quick    # Quick sanity check (~2 min)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python3}"
ERROR_RATES="0 0.001 0.01 0.05 0.1 0.2"
SAVE_FLAG="--save"
QUICK_FLAG=""

if [ "${1:-}" = "--quick" ]; then
    QUICK_FLAG="--quick"
    ERROR_RATES="0 0.01 0.1"
    echo "=== QUICK MODE ==="
fi

FAULT_TYPES=(
    "stuck_at_0"
    "stuck_at_1"
    "wbf_t"
    "wbf_p"
    "syn_silence"
)

echo "================================================================================"
echo "Arthedain — Multi-Fault-Type Robustness Sweep"
echo "================================================================================"
echo "Fault types: ${FAULT_TYPES[*]}"
echo "Error rates: ${ERROR_RATES}"
echo ""

for FAULT_TYPE in "${FAULT_TYPES[@]}"; do
    echo ""
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "  Fault type: $FAULT_TYPE"
    echo "────────────────────────────────────────────────────────────────────────────────"
    
    # Classification task
    echo "    [Task: classification]"
    $PYTHON experiments/arthedain_robustness.py \
        --error-rates $ERROR_RATES \
        --fault-type "$FAULT_TYPE" \
        --persistent \
        --task classification \
        --seed 42 \
        --dt 1.0 \
        $QUICK_FLAG \
        --save 2>&1 | tail -20
    
    if [ -f "results/arthedain_robustness.json" ]; then
        mv "results/arthedain_robustness.json" \
           "results/arthedain_robustness_${FAULT_TYPE}_class.json"
        echo "    -> Saved results/arthedain_robustness_${FAULT_TYPE}_class.json"
    fi
    
    # Regression task
    echo "    [Task: regression]"
    python3 experiments/arthedain_robustness.py \
        --error-rates $ERROR_RATES \
        --fault-type "$FAULT_TYPE" \
        --persistent \
        --task regression \
        --seed 42 \
        $QUICK_FLAG \
        --save 2>&1 | tail -20
    
    if [ -f "results/arthedain_robustness.json" ]; then
        mv "results/arthedain_robustness.json" \
           "results/arthedain_robustness_${FAULT_TYPE}_reg.json"
        echo "    -> Saved results/arthedain_robustness_${FAULT_TYPE}_reg.json"
    fi
done

echo ""
echo "================================================================================"
echo "Sweep complete!"
echo "================================================================================"
echo ""
echo "Results files:"
ls -la results/arthedain_robustness_*.json 2>/dev/null || echo "  No results found"
