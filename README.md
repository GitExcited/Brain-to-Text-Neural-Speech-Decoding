# Brain-to-Text '25: Neural Speech Decoding

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Brain--to--Text--25-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/brain-to-text-25)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://www.python.org/)

Deep learning models for decoding intracortical neural activity during attempted speech into text. This repository contains our team's implementation for the **COMP 433 Fall 2025** course project, tackling the [Kaggle Brain-to-Text '25](https://www.kaggle.com/competitions/brain-to-text-25) challenge.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Team Information](#-team-information)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Models](#-models)
  - [Transformer Model](#1-transformer-model-primary)
  - [BiLSTM Model](#2-bilstm-model)
  - [Baseline RNN Model](#3-baseline-rnn-model-reference)
- [Training Instructions](#-training-instructions)
- [Inference & Submission](#-inference--submission)
- [Pre-trained Models](#-pre-trained-models)
- [Results](#-results)
- [Citation](#-citation)

---

## 🎯 Project Overview

Speech brain-computer interfaces (BCIs) aim to restore communication for people with paralysis caused by ALS or brainstem stroke by decoding speech directly from brain activity. This project develops deep learning models to map intracortical neural spiking activity to text output.

**Challenge:** Decode attempted speech from 256-electrode neural recordings into text sequences.
**Evaluation Metric:** Word Error Rate (WER)
**Baseline Performance:** ~6.7% WER
**Competition Deadline:** December 31, 2025

---

## 👥 Team Information

- **Course:** COMP 433 – Deep Learning, Fall 2025
- **Team Members:**
  - **J. David Ruiz** (40176885) - Team Lead
  - **Elion Abdyli** (40132982)
  - **Ion Turcan** (40154098)
  - **Kirill Vishnyakov** (40281175)

Individual contributions are documented in the final project submission as per course requirements.

---

## 📁 Repository Structure

```
Brain-to-Text-Neural-Speech-Decoding/
│
├── transformer/                    # Transformer model (primary model)
│   ├── train_transformer.py       # Training script
│   ├── generate_submission.py     # Inference and Kaggle submission generation
│   ├── requirements.txt           # Python dependencies
│   ├── Dockerfile                 # Docker container for GPU training
│   ├── docker-compose.yml         # Docker orchestration
│   ├── run_experiments.sh         # Batch experiment runner
│   ├── run_inference.sh           # Inference script
│   ├── outputs/                   # Trained models and experiment results
│   │   ├── exp10_d896_b1_*/       # Best model (d_model=896, batch_size=1)
│   │   │   ├── best_model.pt      # Best checkpoint (lowest PER)
│   │   │   ├── training_curves.png
│   │   │   └── training_history.json
│   │   └── submission.csv         # Kaggle submission file
│   └── DOCKER_README.md           # Docker setup instructions
│
├── bi-lstm/                        # BiLSTM model (secondary model)
│   ├── train_bilstm.py            # Training script
│   ├── requirements.txt           # Python dependencies
│   ├── Dockerfile                 # Docker container
│   ├── docker-compose.yml         # Docker orchestration
│   └── README.md                  # BiLSTM-specific documentation
│
├── rnn_trainer/                    # Baseline RNN model (reference implementation)
│   ├── train_model.py             # Main training entry point
│   ├── rnn_trainer.py             # Trainer class
│   ├── rnn_model.py               # Model architecture
│   ├── dataset.py                 # Data loading utilities
│   ├── rnn_args.yaml              # Configuration file
│   └── data_augmentations.py      # Data augmentation utilities
│
├── elion-kaggle-scripts/           # Experimental scripts and utilities
│   ├── v3.py                      # Experiment version 3
│   └── v4.py                      # Experiment version 4
│
├── docs/                           # Project documentation and reports
│   ├── pdf/                       # Final report PDFs
│   └── src/                       # LaTeX source files
│
├── brain-to-text-clone/            # Reference implementations (Stanford/UCD baselines)
├── kirill-folder/                  # Individual experimentation workspace
│
├── README.md                       # This file
├── papers.md                       # Literature review and paper references
└── proposal.md                     # Initial project proposal
```

---

## 📊 Dataset

### Obtaining the Dataset

The dataset is available from the Kaggle competition page:

**Option 1: Kaggle API (Recommended)**
```bash
# Install Kaggle API
pip install kaggle

# Set up API credentials (~/.kaggle/kaggle.json)
# Download from: https://www.kaggle.com/settings/account

# Download dataset
kaggle competitions download -c brain-to-text-25

# Extract
unzip brain-to-text-25.zip -d ./data/
```

**Option 2: Manual Download**
1. Visit [Kaggle Brain-to-Text '25 Competition](https://www.kaggle.com/competitions/brain-to-text-25/data)
2. Download `brain-to-text-25.zip` (requires Kaggle account)
3. Extract to `./data/` directory

### Dataset Structure

```
data/
└── t15_copyTask_neuralData/
    └── hdf5_data_final/
        ├── t15.2023.08.11/
        │   ├── data_train.hdf5
        │   ├── data_val.hdf5
        │   └── data_test.hdf5
        ├── t15.2023.08.13/
        │   ├── data_train.hdf5
        │   ├── data_val.hdf5
        │   └── data_test.hdf5
        └── ... (45 sessions total)
```

### Dataset Description

- **45 recording sessions** from a single participant (August 2023 - April 2025)
- **256 electrodes** in motor cortex
- **512 features per timestep** (2 features per electrode: threshold crossing rates)
- **41 phoneme classes** (39 phonemes + BLANK + silence)
- **~10,000 trials** total across train/val/test splits
- **Variable-length sequences**: 50-1500 timesteps per trial

---

## 🔧 Requirements

### Hardware Requirements

**Minimum (CPU Training):**
- CPU: Multi-core processor
- RAM: 16GB
- Storage: 50GB free space

**Recommended (GPU Training):**
- GPU: NVIDIA RTX 3080 (10GB VRAM) or better
- RAM: 32GB
- Storage: 100GB SSD

**Optimal (Cloud/Kaggle):**
- GPU: NVIDIA Tesla T4/V100/A100 (16GB+ VRAM)
- RAM: 30GB+

### Software Requirements

- **Operating System**: Linux, Windows 10/11 (with WSL2), or macOS
- **Python**: 3.10
- **CUDA**: 12.1 (for GPU training)
- **Docker**: Optional but recommended for reproducibility

---

## 📦 Installation

### Option 1: Docker (Recommended for Reproducibility)

**Prerequisites:**
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- NVIDIA Container Toolkit (for GPU support)

```bash
# Clone repository
git clone https://github.com/GitExcited/Brain-to-Text-Neural-Speech-Decoding.git
cd Brain-to-Text-Neural-Speech-Decoding

# Navigate to transformer directory
cd transformer

# Build Docker image
docker build -t brain-to-text-transformer:latest .

# Run with GPU support
docker run --gpus all \
  -v ./data:/data:ro \
  -v ./outputs:/outputs \
  brain-to-text-transformer:latest \
  --data-dir /data/t15_copyTask_neuralData/hdf5_data_final \
  --output-dir /outputs \
  --epochs 25 \
  --batch-size 1 \
  --d-model 896 \
  --cuda
```

See [transformer/DOCKER_README.md](transformer/DOCKER_README.md) for detailed Docker instructions.

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/GitExcited/Brain-to-Text-Neural-Speech-Decoding.git
cd Brain-to-Text-Neural-Speech-Decoding

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support (for GPU)
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies for transformer model
cd transformer
pip install -r requirements.txt

# OR for BiLSTM model
cd ../bi-lstm
pip install -r requirements.txt

# OR for baseline RNN
cd ../rnn_trainer
pip install -r ../transformer/requirements.txt  # Uses similar dependencies
pip install omegaconf pyyaml
```

### Option 3: Kaggle Notebooks

Our models can be trained directly on Kaggle with GPU acceleration:

1. Fork our Kaggle notebook: [Link to be added]
2. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
3. Run all cells

---

## 🤖 Models

### 1. Transformer Model (Primary)

**Architecture:** Encoder-decoder transformer with multi-head self-attention
**Best Configuration:**
- Model dimension (d_model): 896
- Attention heads: 4
- Encoder/Decoder layers: 4 each
- Batch size: 1 (eliminates padding)
- Parameters: ~12.7M

**Key Features:**
- Xavier initialization for stability
- WarmupCosineScheduler (2 epochs warmup → cosine decay)
- Label smoothing (0.1) for regularization
- Gradient clipping (max_norm=0.5)
- Early stopping (patience=10)
- Day-specific embeddings for session adaptation

**Training:**
```bash
cd transformer

# Quick training (5 epochs, small model)
python train_transformer.py \
  --data-dir /path/to/hdf5_data_final \
  --output-dir ./outputs/quick_test \
  --d-model 256 \
  --batch-size 4 \
  --epochs 5 \
  --max-days 10 \
  --cuda

# Best model configuration (25 epochs, d_model=896)
python train_transformer.py \
  --data-dir /path/to/hdf5_data_final \
  --output-dir ./outputs/best_model \
  --d-model 896 \
  --batch-size 1 \
  --epochs 25 \
  --lr 0.0001 \
  --max-days 45 \
  --cuda
```

**Training Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | Required | Path to HDF5 data directory |
| `--output-dir` | `./outputs` | Output directory for models/logs |
| `--d-model` | 256 | Transformer dimension |
| `--batch-size` | 16 | Batch size |
| `--epochs` | 5 | Number of epochs |
| `--lr` | 0.0001 | Learning rate |
| `--max-days` | 1000 | Max sessions to load |
| `--warmup-epochs` | 2 | Warmup epochs for LR scheduler |
| `--patience` | 10 | Early stopping patience |
| `--cuda` / `--no-cuda` | True | Enable/disable GPU |

### 2. BiLSTM Model

**Architecture:** Bidirectional LSTM with CTC decoding
**Location:** `bi-lstm/`

**Training:**
```bash
cd bi-lstm

# Using Docker
docker-compose up

# Or locally
python train_bilstm.py \
  --data-dir /path/to/hdf5_data_final \
  --output-dir ./outputs \
  --epochs 30 \
  --cuda
```

See [bi-lstm/README.md](bi-lstm/README.md) for detailed instructions.

### 3. Baseline RNN Model (Reference)

**Architecture:** Multi-layer GRU with day-specific input networks
**Location:** `rnn_trainer/`
**Configuration:** `rnn_trainer/rnn_args.yaml`

**Training:**
```bash
cd rnn_trainer

# Edit rnn_args.yaml to set paths and hyperparameters
# Key settings:
#   dataset.dataset_dir: /path/to/hdf5_data_final
#   output_dir: /path/to/output
#   num_training_batches: 2000
#   model.n_units: 768

python train_model.py
```

This model is provided as a reference implementation based on published baselines.

---

## 🏋️ Training Instructions

### Transformer Model - Complete Training Pipeline

**Step 1: Prepare Data**
```bash
# Ensure data is in the correct location
ls data/t15_copyTask_neuralData/hdf5_data_final/
# Should show 45 session folders (t15.2023.08.11, etc.)
```

**Step 2: Single Experiment**
```bash
cd transformer

python train_transformer.py \
  --data-dir ../data/t15_copyTask_neuralData/hdf5_data_final \
  --output-dir ./outputs/my_experiment \
  --d-model 512 \
  --batch-size 4 \
  --epochs 35 \
  --lr 0.0001 \
  --cuda
```

**Step 3: Hyperparameter Sweep (Overnight Training)**

The `run_experiments.sh` script runs multiple configurations sequentially:

```bash
cd transformer

# Edit run_experiments.sh to configure experiments
# Then run (takes 12-24 hours on T4 GPU):
chmod +x run_experiments.sh
./run_experiments.sh
```

**Step 4: Monitor Training**

Training progress is logged to:
- **Console output**: Real-time loss/PER
- **TensorBoard**: `outputs/<experiment_name>/tensorboard/`
- **Training curves**: `outputs/<experiment_name>/training_curves.png`

```bash
# View TensorBoard (in separate terminal)
tensorboard --logdir=transformer/outputs/
# Open http://localhost:6006 in browser
```

**Step 5: Resume Training (if interrupted)**

```bash
python train_transformer.py \
  --data-dir ../data/t15_copyTask_neuralData/hdf5_data_final \
  --output-dir ./outputs/my_experiment \
  --resume \
  --cuda
```

### Expected Training Time

| Configuration | GPU | Time/Epoch | Total Time (25 epochs) |
|--------------|-----|------------|------------------------|
| d_model=256, batch=4 | RTX 3080 | ~8 min | ~3.3 hours |
| d_model=512, batch=4 | RTX 3080 | ~12 min | ~5 hours |
| d_model=896, batch=1 | T4 x2 | ~18 min | ~7.5 hours |
| d_model=768, batch=2 | V100 | ~10 min | ~4.2 hours |

---

## 🔮 Inference & Submission

### Generate Kaggle Submission

**Using Pre-trained Model:**

```bash
cd transformer

python generate_submission.py \
  --model-path ./outputs/exp10_d896_b1_20251128_043602/best_model.pt \
  --data-dir ../data/t15_copyTask_neuralData/hdf5_data_final \
  --dict-path ../data/dict.txt \
  --output-csv submission.csv \
  --d-model 896 \
  --num-days 45 \
  --max-length 200 \
  --cuda
```

**Submission Arguments:**
| Argument | Required | Description |
|----------|----------|-------------|
| `--model-path` | Yes | Path to trained model checkpoint (.pt file) |
| `--data-dir` | Yes | Path to HDF5 data directory |
| `--dict-path` | Yes | Path to CMU phoneme dictionary |
| `--output-csv` | No | Output CSV filename (default: submission.csv) |
| `--d-model` | Yes | Must match training config |
| `--num-days` | Yes | Must match training config |
| `--max-length` | No | Max sequence length for generation (default: 200) |

**Output:**
- `submission.csv` with columns: `id`, `text`
- Ready to upload to Kaggle

**Submit to Kaggle:**
```bash
kaggle competitions submit -c brain-to-text-25 -f submission.csv -m "Transformer d_model=896 batch=1"
```

### Using the Shell Script (Simplified)

```bash
cd transformer
chmod +x run_inference.sh
./run_inference.sh
```

---

## 💾 Pre-trained Models

### Transformer Models (Best Configurations)

Download pre-trained checkpoints from [Google Drive / Kaggle Dataset - Link TBD]:

| Model | d_model | Batch Size | Validation PER | Download |
|-------|---------|------------|----------------|----------|
| **exp10** (Best) | 896 | 1 | 0.1056 | [Link TBD] |
| exp07 | 512 | 4 | 0.1048 | [Link TBD] |
| exp04 | 768 | 2 | 0.1541 | [Link TBD] |

**Using Pre-trained Models:**

```bash
# Download checkpoint to transformer/outputs/
cd transformer/outputs
wget [download_link] -O best_model.pt

# Run inference
cd ..
python generate_submission.py \
  --model-path ./outputs/best_model.pt \
  --data-dir ../data/t15_copyTask_neuralData/hdf5_data_final \
  --dict-path ../data/dict.txt \
  --d-model 896 \
  --num-days 45 \
  --cuda
```

### Test on Sample Data

To verify the model works without downloading the full dataset:

```bash
# Use first 5 sessions only
python generate_submission.py \
  --model-path ./outputs/best_model.pt \
  --data-dir ../data/t15_copyTask_neuralData/hdf5_data_final \
  --dict-path ../data/dict.txt \
  --d-model 896 \
  --num-days 5 \
  --cuda
```

---

## 📈 Results

### Transformer Model Performance

| Experiment | Config | Val PER | Test WER | Epochs | Training Time |
|-----------|--------|---------|----------|--------|---------------|
| exp10 | d=896, b=1 | **0.1056** | ~0.70 | 25 | 7.5h |
| exp07 | d=512, b=4, lr=3e-4 | 0.1048 | - | 38 | 8.4h |
| exp01 | d=512, b=4 | 0.1541 | - | 35 | 7h |
| exp08 | d=448, b=8 | 0.1691 | - | 40 | 5.3h |

**Note:** Large gap between validation PER (~0.10) and test WER (~0.70) is due to:
- Teacher forcing during validation vs. autoregressive generation during test
- Greedy decoding (argmax) vs. beam search
- Exposure bias (model never sees its own mistakes during training)

### Key Findings

1. **Model dimension matters**: Monotonic improvement from d_model=448 → 896
2. **Batch size = 1 is optimal**: Eliminates padding noise, improves PER by ~5%
3. **Learning rate scheduling critical**: Warmup + cosine decay enables convergence
4. **Early stopping effective**: Most models plateau after 20-30 epochs

See [docs/pdf/final_report.pdf](docs/pdf/final_report.pdf) for detailed analysis.

---

## 📚 Citation

If you use this code, please cite:

```bibtex
@misc{braintotext2025,
  author = {Ruiz, J. David and Abdyli, Elion and Turcan, Ion and Vishnyakov, Kirill},
  title = {Brain-to-Text Neural Speech Decoding with Transformers},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/GitExcited/Brain-to-Text-Neural-Speech-Decoding}}
}
```

**Competition Reference:**
```bibtex
@misc{kaggle-brain-to-text-25,
  author = {UC Davis Neuroprosthetics Lab},
  title = {Brain-to-Text '25},
  year = {2025},
  howpublished = {\url{https://www.kaggle.com/competitions/brain-to-text-25}}
}
```

---

## 🔗 Additional Resources

- **Kaggle Competition**: [https://www.kaggle.com/competitions/brain-to-text-25](https://www.kaggle.com/competitions/brain-to-text-25)
- **Project Proposal**: [proposal.md](proposal.md)
- **Literature Review**: [papers.md](papers.md)
- **Final Report**: [docs/pdf/final_report.pdf](docs/pdf/final_report.pdf)

---

## 🤝 Contributing

This repository is maintained for academic purposes (COMP 433 course project).

For questions or issues:
1. Check existing documentation in `docs/`
2. Review [papers.md](papers.md) for background literature
3. Contact team members via course channels

---

## 📜 License

This project is developed for academic purposes as part of COMP 433 at Concordia University.

Dataset is provided by UC Davis Neuroprosthetics Lab via Kaggle competition.
Baseline implementations adapted from published research (cited in `papers.md`).

---

## 🙏 Acknowledgments

- **UC Davis Neuroprosthetics Lab** for organizing the competition and providing the dataset
- **COMP 433 Course Staff** for guidance and support
- **Stanford Neural Prosthetics Translational Laboratory** for baseline implementations
- The participant (T15) who contributed neural recordings for this research

---

**Last Updated**: December 2025
**Course**: COMP 433 – Deep Learning, Fall 2025
**Institution**: Concordia University
