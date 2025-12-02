# Brain-to-Text V2: Conv1d + BiLSTM + TextBlob

Docker-based training and inference for the BiLSTM brain-to-text model with spell correction.

## Features

- **Conv1d + BiLSTM architecture** with CTC loss
- **SpecAugment** for data augmentation
- **TextBlob spell correction** with bonus loss term
- **GPU support** via NVIDIA Docker
- **Containerized** for reproducible training

## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit
- Data directory at `../../data` (relative to newModel folder)

### Setup

1. Ensure your data is in the correct location:
   ```
   Brain-to-Text-Neural-Speech-Decoding/
   ├── data/
   │   └── t15_copyTask_neuralData/
   │       └── hdf5_data_final/
   │           ├── t15.2023.08.11/
   │           │   ├── data_train.hdf5
   │           │   ├── data_val.hdf5
   │           │   └── data_test.hdf5
   │           └── ...
   └── transformer_notebook/
       └── newModel/
           ├── docker-compose.yml
           └── ...
   ```

2. That's it! No `.env` file needed - paths are configured to match the transformer setup.

### Run Training

Train the model:

```bash
docker-compose up bilstm-training
```

### Run Inference Only

Generate predictions from a trained model:

```bash
docker-compose up bilstm-inference
```

### Run Both Training + Inference

Train and then generate predictions:

```bash
docker-compose up bilstm-full
```

## Configuration

Default parameters are hardcoded in [docker-compose.yml](docker-compose.yml):

- `EPOCHS` - Number of training epochs (default: 20)
- `BATCH_SIZE` - Batch size (default: 32)
- `LR` - Learning rate (default: 0.003)
- `HIDDEN_DIM` - Hidden dimension size (default: 512)
- `NUM_LAYERS` - Number of LSTM layers (default: 3)
- `DROPOUT` - Dropout rate (default: 0.4)
- `MAX_DAYS` - Maximum sessions to load (default: 1000)
- `NUM_WORKERS` - Data loader workers (default: 2)
- `BONUS_WEIGHT` - TextBlob correction bonus weight (default: 0.1)
- `BONUS_EVERY_N_BATCHES` - Compute bonus every N batches (default: 5)

To change these, edit the `command:` section in docker-compose.yml or run manually in dev mode.

## Architecture

### Model Components

1. **Conv1d Encoder**: 1D convolution with stride 2 for temporal downsampling
2. **BiLSTM**: 3-layer bidirectional LSTM for sequence modeling
3. **CTC Loss**: Connectionist Temporal Classification for alignment-free training
4. **TextBlob Correction**: Spell checking with bonus reward during training

### Training Process

The model uses:
- AdamW optimizer with weight decay
- OneCycleLR scheduler with warmup
- Gradient clipping
- SpecAugment for frequency/time masking

### Correction Bonus

A novel training technique that rewards the model when TextBlob spell correction:
- Produces a perfect match with ground truth (bonus: 1.0)
- Improves edit distance (bonus: up to 0.5)
- Already produces perfect predictions (bonus: 0.3)

This bonus is subtracted from the loss to encourage learning patterns that spell-check well.

## Output Files

After training, you'll find in `./outputs`:

- `best_model.pt` - Best model checkpoint (lowest validation loss)
- `final_model.pt` - Model after final epoch
- `training_history.json` - Training metrics and losses
- `training_curves.png` - Visualization of training progress
- `submission.csv` - Predictions (if running inference)

## Development Mode

For interactive development/debugging:

```bash
docker-compose up -d bilstm-dev
docker exec -it b2t-bilstm-dev bash
```

Then run training manually:

```bash
python train_bilstm.py \
  --data-dir /data/t15_copyTask_neuralData/hdf5_data_final \
  --output-dir /outputs \
  --epochs 20 \
  --batch-size 32 \
  --cuda
```

## Project Structure

```
newModel/
├── train_bilstm.py          # Main training script
├── Dockerfile                # Docker image definition
├── docker-compose.yml        # Docker Compose services
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── outputs/                 # Output directory (created on first run)
    ├── best_model.pt
    ├── final_model.pt
    ├── training_history.json
    ├── training_curves.png
    └── submission.csv
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in docker-compose.yml or run with custom args:
```bash
docker-compose run bilstm-training --batch-size 16
```

### Slow Training

- Reduce `--num-workers` if CPU-bound
- Increase `--bonus-every-n-batches` to compute correction bonus less frequently

### Permission Errors

Ensure output directory is writable:
```bash
chmod -R 777 ./outputs
```

### Data Not Found

Verify data directory structure:
```bash
ls -la ../../data/t15_copyTask_neuralData/hdf5_data_final/
```

## Comparison with Transformer Model

This BiLSTM model differs from the transformer model in:
- Uses CTC loss instead of cross-entropy
- Character-level instead of phoneme-level
- Includes spell correction bonus
- Lighter architecture (fewer parameters)
- Different data augmentation strategy
