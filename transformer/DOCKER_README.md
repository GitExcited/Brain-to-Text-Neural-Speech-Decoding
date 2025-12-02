# Brain-to-Text Transformer - Docker Training Setup

This directory contains everything needed to train the Brain-to-Text Transformer model in a containerized environment with CUDA/GPU support.

## Prerequisites

### Windows (Docker Desktop + WSL 2)

1. **NVIDIA GPU** with CUDA support
2. **Latest NVIDIA GPU drivers** installed on Windows (regular GeForce/Studio drivers)
3. **WSL 2** enabled (Windows 10 version 21H2+ or Windows 11)
4. **Docker Desktop** installed with WSL 2 backend enabled:
   - Download from [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)
   - In Settings → General, ensure "Use WSL 2 based engine" is checked
   - In Settings → Resources → WSL Integration, enable your distro

5. **Verify GPU access in Docker** (run in PowerShell or WSL terminal):
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```
   You should see your GPU listed!

### Linux (Ubuntu/Debian)

1. **NVIDIA GPU** with CUDA support
2. **Docker** installed ([Install Docker](https://docs.docker.com/get-docker/))
3. **NVIDIA Container Toolkit** installed:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

4. **Verify GPU access**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

## Files Overview

```
transformer_notebook/
├── Dockerfile              # Container definition
├── docker-compose.yml      # Easy container orchestration
├── requirements.txt        # Python dependencies
├── train_transformer.py    # Training script
└── DOCKER_README.md        # This file
```

## Quick Start

### 1. Build the Docker Image

```bash
cd transformer_notebook

# Build the image
docker build -t brain-to-text-transformer:latest .

# Or using docker-compose
docker-compose build
```

### 2. Prepare Your Data

Your HDF5 data should be organized as:
```
/path/to/your/data/
├── session_001/
│   ├── test.hdf5
│   ├── train.hdf5
│   └── val.hdf5
├── session_002/
│   ├── test.hdf5
│   ├── train.hdf5
│   └── val.hdf5
...
```

### 3. Run Training

#### Option A: Using docker-compose (Recommended)

By default, the container looks for data in `./data` and saves outputs to `./outputs` (relative to the `transformer_notebook` folder). Just run:

```bash
# Simple run with default paths (./data and ./outputs)
docker-compose up transformer-training

# Or with custom training parameters
EPOCHS=10 \
BATCH_SIZE=32 \
LR=0.001 \
docker-compose up transformer-training

# Override data path if needed
DATA_PATH=/custom/path OUTPUT_PATH=/custom/output docker-compose up transformer-training
```

#### Option B: Using docker run directly

```bash
docker run --gpus all \
  -v ./data:/data:ro \
  -v ./outputs:/outputs \
  --shm-size=8g \
  brain-to-text-transformer:latest \
  --data-dir /data \
  --output-dir /outputs \
  --epochs 5 \
  --batch-size 16 \
  --cuda
```

### 4. Interactive Development

For debugging or interactive development:

```bash
# Start interactive container
DATA_PATH=/path/to/your/data docker-compose run transformer-dev

# Inside the container:
python train_transformer.py --data-dir /data --output-dir /outputs --epochs 1 --cuda
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | Required | Path to HDF5 data directory |
| `--output-dir` | `./outputs` | Output directory for models |
| `--epochs` | 5 | Number of training epochs |
| `--batch-size` | 16 | Batch size |
| `--lr` | 0.0005 | Learning rate |
| `--d-model` | 256 | Transformer model dimension |
| `--max-days` | 1000 | Maximum number of sessions to load |
| `--num-workers` | 4 | DataLoader workers |
| `--cuda` | True | Use CUDA (GPU) |
| `--no-cuda` | - | Disable CUDA (CPU only) |
| `--log-interval` | 40 | Batch interval for logging |
| `--save-interval` | 1 | Epoch interval for checkpoints |

## Output Files

After training, you'll find in your output directory:
```
outputs/
├── best_model.pt           # Best model (lowest PER)
├── final_model.pt          # Final model after all epochs
├── checkpoint_epoch_N.pt   # Periodic checkpoints
├── training_history.json   # Training metrics
└── training_curves.png     # Loss and PER plots
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size`
- Reduce `--d-model`
- Use fewer `--max-days`

### Container can't access GPU
```bash
# Verify NVIDIA Container Toolkit is installed
nvidia-container-cli info

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Shared memory issues with DataLoader
- The docker-compose.yml already sets `shm_size: '8gb'`
- If using docker run, add `--shm-size=8g`

### Permission denied on output directory
```bash
# Create output directory with proper permissions
mkdir -p outputs
chmod 777 outputs
```

## Using on Different Machines

1. **Copy the files** to your target machine:
   - `Dockerfile`
   - `docker-compose.yml`
   - `requirements.txt`
   - `train_transformer.py`

2. **Build and run**:
   ```bash
   docker-compose build
   DATA_PATH=/path/to/data docker-compose up transformer-training
   ```

## Cloud Deployment

### AWS (EC2 with GPU)
```bash
# On a p3.2xlarge or similar GPU instance
sudo apt-get update
sudo apt-get install -y docker.io nvidia-container-toolkit
sudo systemctl restart docker

# Clone your repo and run
cd transformer_notebook
docker-compose build
DATA_PATH=/path/to/data docker-compose up transformer-training
```

### Google Cloud (GCE with GPU)
```bash
# On a VM with NVIDIA GPU
# Install Docker and NVIDIA Container Toolkit, then:
docker-compose build
DATA_PATH=/path/to/data docker-compose up transformer-training
```

## Performance Tips

1. **Use SSD storage** for data directory
2. **Increase batch size** if GPU memory allows
3. **Use multiple GPUs** (modify docker-compose.yml `count` parameter)
4. **Enable mixed precision** training (modify train_transformer.py to use `torch.cuda.amp`)
