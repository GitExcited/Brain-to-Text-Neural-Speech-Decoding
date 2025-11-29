# Kaggle Submission Generation

Generate a Kaggle submission using the best model from experiment 10.

## Quick Start (Docker)

Run this command in the `transformer_notebook` directory:

```bash
docker compose run --rm inference
```

This will:
1. Load the best model from experiment 10 (`outputs/exp10_d896_b1_20251128_043602/best_model.pt`)
2. Process all 1,450 test trials in chronological order
3. Generate phoneme predictions autoregressively
4. Convert phonemes to words using the CMU dictionary
5. Save the submission to `outputs/submission.csv`

## Output

The submission CSV will have two columns:
- `id`: Sequential IDs from 0 to 1449
- `text`: Predicted text (lowercase, no punctuation)

## File Locations

- **Model**: `outputs/exp10_d896_b1_20251128_043602/best_model.pt`
- **Output**: `outputs/submission.csv`
- **Data**: `data/t15_copyTask_neuralData/hdf5_data_final/*/data_test.hdf5`

## Manual Run (Python)

If you want to run outside Docker:

```bash
python generate_submission.py \
    --model-path outputs/exp10_d896_b1_20251128_043602/best_model.pt \
    --data-dir data/t15_copyTask_neuralData/hdf5_data_final \
    --dict-path ../brain-to-text-clone/nejm-brain-to-text/language_model/examples/speech/s0/dict.txt \
    --output-csv submission.csv \
    --d-model 896 \
    --num-days 45 \
    --cuda
```
