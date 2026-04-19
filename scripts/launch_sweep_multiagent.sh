#!/bin/bash
set -e

# --- CONFIG ---
PROJECT="echo3d-mixed-sweep2"
ENTITY="tusharn-carnegie-mellon-university"
NUM_AGENTS=4          # <--- SET THIS (3-4 is safe for your GPU)
RUNS_PER_AGENT=30     # Total: 4 * 13 = 52 jobs
# --------------

echo "=========================================="
echo "Launching Parallel Sweep ($NUM_AGENTS Agents)"
echo "=========================================="
echo "Project: $PROJECT"
echo "Entity: $ENTITY"
echo "Total jobs: $((NUM_AGENTS * RUNS_PER_AGENT))"
echo ""

# 1. Create sweep
SWEEP_ID=$(wandb sweep --project $PROJECT --entity $ENTITY sweep_config.yaml 2>&1 | grep -o "wandb agent.*" | awk '{print $3}')

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Could not extract sweep ID"
    wandb sweep --project $PROJECT --entity $ENTITY sweep_config.yaml
    exit 1
fi

SWEEP_ID=$(echo $SWEEP_ID | tr -d "'" | tr -d '"')

echo "Sweep ID: $SWEEP_ID"
echo ""
echo "Starting $NUM_AGENTS agents in background..."
echo ""

# 2. Launch Multiple Agents in Background
for i in $(seq 1 $NUM_AGENTS); do
    echo "[Agent $i/$NUM_AGENTS] Starting..."
    wandb agent --count $RUNS_PER_AGENT $SWEEP_ID > agent_$i.log 2>&1 &
    sleep 1
done

echo ""
echo "=================================================="
echo "All agents launched! Memory per job: ~3.1 GB"
echo "Total jobs: $((NUM_AGENTS * RUNS_PER_AGENT))"
echo "=================================================="
echo ""
echo "Watch logs: tail -f agent_1.log"
echo "Stop all:   pkill -f 'wandb agent'"
echo "Monitor:    https://wandb.ai/$ENTITY/$PROJECT/sweeps/$(basename $SWEEP_ID)"
