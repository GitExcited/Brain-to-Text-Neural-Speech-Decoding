#!/bin/bash
# =============================================================================
# Brain-to-Text Transformer - Overnight Experiment Runner
# =============================================================================
# This script runs multiple training experiments sequentially.
# If one fails (e.g., OOM error 137), it continues to the next experiment.
# Each experiment saves to its own named output folder.
#
# Usage (from WSL or inside Docker):
#   chmod +x run_experiments.sh
#   ./run_experiments.sh
#
# Or from PowerShell:
#   wsl bash ./run_experiments.sh
# =============================================================================

set -e  # Exit on error (but we handle errors per-experiment below)

# Base paths
DATA_DIR="/data/t15_copyTask_neuralData/hdf5_data_final"
OUTPUT_BASE="/outputs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Log file for the entire run
LOG_FILE="${OUTPUT_BASE}/experiment_log_${TIMESTAMP}.txt"

echo "=============================================" | tee -a "$LOG_FILE"
echo "Starting Experiment Suite: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"

# Function to run a single experiment
run_experiment() {
    local NAME=$1
    local D_MODEL=$2
    local BATCH_SIZE=$3
    local EPOCHS=$4
    local MAX_DAYS=$5
    local EXTRA_ARGS=${6:-""}
    
    local OUTPUT_DIR="${OUTPUT_BASE}/${NAME}_${TIMESTAMP}"
    
    echo "" | tee -a "$LOG_FILE"
    echo "---------------------------------------------" | tee -a "$LOG_FILE"
    echo "Experiment: $NAME" | tee -a "$LOG_FILE"
    echo "  d_model=$D_MODEL, batch_size=$BATCH_SIZE, epochs=$EPOCHS, max_days=$MAX_DAYS" | tee -a "$LOG_FILE"
    echo "  Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
    echo "  Started: $(date)" | tee -a "$LOG_FILE"
    echo "---------------------------------------------" | tee -a "$LOG_FILE"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Run training and capture exit code
    python /app/train_transformer.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --d-model "$D_MODEL" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --max-days "$MAX_DAYS" \
        --num-workers 2 \
        --cuda \
        $EXTRA_ARGS \
        2>&1 | tee "${OUTPUT_DIR}/training_output.log"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Experiment $NAME completed successfully!" | tee -a "$LOG_FILE"
    elif [ $EXIT_CODE -eq 137 ]; then
        echo "✗ Experiment $NAME failed with OOM (exit 137)" | tee -a "$LOG_FILE"
    else
        echo "✗ Experiment $NAME failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    fi
    
    echo "  Finished: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Small delay between experiments to let GPU memory clear
    sleep 10
    
    return 0  # Always return 0 so the script continues
}

# =============================================================================
# DEFINE YOUR EXPERIMENTS HERE (10 experiments)
# Format: run_experiment "NAME" D_MODEL BATCH_SIZE EPOCHS MAX_DAYS "EXTRA_ARGS"
# =============================================================================

# ----- DIMENSION SCALING (primary focus based on your results) -----

# Exp 1: d_model=512 - next step up from your successful 384
run_experiment "exp01_d512_b4" 512 4 35 45

# Exp 2: d_model=576 - between 512 and 640
run_experiment "exp02_d576_b4" 576 4 35 45

# Exp 3: d_model=640 - pushing higher
run_experiment "exp03_d640_b3" 640 3 35 45

# Exp 4: d_model=768 - BERT-base size (might OOM, but worth trying)
run_experiment "exp04_d768_b2" 768 2 30 45

# ----- TRAINING DYNAMICS -----

# Exp 5: d_model=512 with MORE epochs (see if it keeps improving)
run_experiment "exp05_d512_e60" 512 4 60 45

# Exp 6: d_model=512 with LOWER learning rate (more stable convergence)
run_experiment "exp06_d512_lr5e5" 512 4 40 45 "--lr 0.00005"

# Exp 7: d_model=512 with HIGHER learning rate (faster but riskier)
run_experiment "exp07_d512_lr3e4" 512 4 40 45 "--lr 0.0003"

# ----- BATCH SIZE EXPERIMENTS -----

# Exp 8: d_model=448 with LARGER batch (better gradient estimates)
run_experiment "exp08_d448_b8" 448 8 40 45

# Exp 9: d_model=384 with even LARGER batch (your proven config, more stable)
run_experiment "exp09_d384_b12" 384 12 45 45

# ----- AGGRESSIVE -----

# Exp 10: d_model=896 - go big or go home (likely OOM but let's see!)
run_experiment "exp10_d896_b1" 896 1 25 45

# =============================================================================
# END OF EXPERIMENTS
# =============================================================================

echo "" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "All experiments completed!" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"

# Summary
echo "" | tee -a "$LOG_FILE"
echo "Output directories created:" | tee -a "$LOG_FILE"
ls -la ${OUTPUT_BASE}/*_${TIMESTAMP}* 2>/dev/null | tee -a "$LOG_FILE" || echo "No outputs found"
