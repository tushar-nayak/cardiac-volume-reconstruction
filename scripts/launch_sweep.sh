#!/bin/bash
set -e

# --- CONFIG ---
PROJECT="echo3d-mixed-sweep2"
ENTITY="tusharn-carnegie-mellon-university"
NUM_AGENTS=1  # ONLY 1 AGENT - LINEAR EXECUTION
RUNS_PER_AGENT=200  # 50 jobs sequentially
# --------------

echo "=========================================="
echo "Launching Mixed Mode Ablation Sweep"
echo "LINEAR EXECUTION (1 Agent, 50 Jobs)"
echo "=========================================="
echo "Project: $PROJECT"
echo "Entity: $ENTITY"

# 1. Create sweep
SWEEP_ID=$(wandb sweep --project $PROJECT --entity $ENTITY sweep_config.yaml 2>&1 | grep -o "wandb agent.*" | awk '{print $3}')

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Could not extract sweep ID"
    wandb sweep --project $PROJECT --entity $ENTITY sweep_config.yaml
    exit 1
fi

SWEEP_ID=$(echo $SWEEP_ID | tr -d "'" | tr -d '"')

echo "Sweep created with ID: $SWEEP_ID"
echo ""
echo "Starting 1 agent (LINEAR) - all output will print below..."
echo ""

# 2. Launch SINGLE agent - foreground, no &
# This prints ALL output directly and runs sequentially
wandb agent --count $RUNS_PER_AGENT $SWEEP_ID

echo ""
echo "All $RUNS_PER_AGENT jobs completed!"
echo "Monitor results at:"
echo "  https://wandb.ai/$ENTITY/$PROJECT/sweeps/$(basename $SWEEP_ID)"
