#!/bin/bash
# Run inference to generate Kaggle submission

set -e

echo "=================================================="
echo "Brain-to-Text Kaggle Submission Generator"
echo "=================================================="

# Configuration
MODEL_PATH="/outputs/exp10_d896_b1_20251128_043602/best_model.pt"
DATA_DIR="/data/t15_copyTask_neuralData/hdf5_data_final"
DICT_PATH="/app/dict.txt"
OUTPUT_CSV="/outputs/submission.csv"

# Model configuration (must match experiment 10)
D_MODEL=896
NUM_DAYS=45

echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_CSV"
echo ""

# Run inference
python /app/generate_submission.py \
    --model-path "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    --dict-path "$DICT_PATH" \
    --output-csv "$OUTPUT_CSV" \
    --d-model "$D_MODEL" \
    --num-days "$NUM_DAYS" \
    --cuda

echo ""
echo "=================================================="
echo "Done! Submission saved to: $OUTPUT_CSV"
echo "=================================================="
