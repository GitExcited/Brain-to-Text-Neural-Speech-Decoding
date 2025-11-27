#!/usr/bin/env python3
"""
Brain-to-Text Transformer Training Script
Containerized training with CUDA support
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import h5py
import numpy as np
import math
import itertools
import editdistance
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

LOGIT_TO_PHONEME = [
    'BLANK',    # "BLANK" = CTC blank symbol
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    '|',    # "|" = silence token
]

# ============================================================================
# Dataset
# ============================================================================

class BrainToTextDataset(Dataset):
    def __init__(self, root_dir, split="train", max_days=1000, paths=None):
        self.root_dir = root_dir
        self.split = split
        
        if paths is None:
            paths = []
            contents = sorted(os.listdir(self.root_dir))
            days_loaded = 0
            
            for item in contents:
                if days_loaded >= max_days:
                    break
                sessionPath = os.path.join(self.root_dir, item)
                if not os.path.isdir(sessionPath):
                    continue
                    
                session = sorted(os.listdir(sessionPath))
                if len(session) != 3:
                    continue  # skip if not train, val, test in the folder
                    
                data_type = 0  # test
                if split == "train":
                    data_type = 1
                elif split == "val":
                    data_type = 2
    
                paths.append(os.path.join(sessionPath, session[data_type]))
                days_loaded += 1
                
        self.paths = paths
        self.days = len(self.paths)
        
        self.X = []
        for day_idx, train_path in enumerate(paths):
            with h5py.File(train_path, 'r') as f:
                for trial_idx, trial in enumerate(f):
                    X_trial = torch.from_numpy(np.array(f[trial]["input_features"]))
                    y_trial = torch.from_numpy(np.array(f[trial]["seq_class_ids"]))
                    self.X.append({
                        "features": X_trial,
                        "labels": y_trial,
                        "day_id": day_idx,
                        "trial_id": trial_idx
                    })
        
        print(f"Loaded {len(self.X)} trials from {self.days} days ({split})")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def custom_collate_fn(batch):
    features = [d["features"] for d in batch]
    labels = [d["labels"] for d in batch]
    day_ids = [d["day_id"] for d in batch]
    trial_ids = [d["trial_id"] for d in batch]

    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)

    feature_padding_mask = (padded_features[:, :, 0] == 0)
    label_padding_mask = (padded_labels == 0)

    return {
        "feature": padded_features,
        "label": padded_labels,
        "feature_mask": feature_padding_mask,
        "label_mask": label_padding_mask,
        "day": torch.tensor(day_ids),
        "trial": torch.tensor(trial_ids)
    }


# ============================================================================
# Transformer Model
# ============================================================================

def make_subsequent_mask(size, device):
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer so it moves with the model to GPU
        self.register_buffer('pe', pe)

    def forward(self, x):
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return x


class NeuralEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(512, d_model)
        self.pos = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, neural_feature, neural_feature_key_padding_mask=None):
        neural_feature = self.input_proj(neural_feature)
        neural_feature = self.pos(neural_feature)
        return self.encoder(
            neural_feature,
            src_key_padding_mask=neural_feature_key_padding_mask
        )


class NeuralDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, num_days=1000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.day_embed = nn.Embedding(num_days, d_model)
        self.pos = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, class_id, memory, day_idx, class_id_mask=None, 
                class_id_key_padding_mask=None, memory_key_padding_mask=None):
        class_id = self.embed(class_id)
        class_id = self.pos(class_id)
        
        day_vec = self.day_embed(day_idx)
        day_vec = day_vec.unsqueeze(1)
        class_id = class_id + day_vec
        
        out = self.decoder(
            class_id,
            memory,
            tgt_mask=class_id_mask,
            tgt_key_padding_mask=class_id_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.fc(out)


class NeuralToPhonemeTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_days=1000):
        super().__init__()
        self.encoder = NeuralEncoder(d_model=d_model)
        self.decoder = NeuralDecoder(vocab_size, d_model=d_model, num_days=num_days)

    def forward(self, neural_feature_input, class_id_tokens, 
                neural_feature_padding_mask, class_id_padding_mask, day_idx):
        class_id_in = class_id_tokens[:, :-1]
        seq_len = class_id_in.size(1)
        
        # Create mask on the same device as input
        subsequent_mask = make_subsequent_mask(seq_len, class_id_in.device)
        
        memory = self.encoder(
            neural_feature_input, 
            neural_feature_key_padding_mask=neural_feature_padding_mask
        )
        
        logits = self.decoder(
            class_id_in,
            memory,
            class_id_mask=subsequent_mask,
            class_id_key_padding_mask=class_id_padding_mask[:, :-1],
            memory_key_padding_mask=neural_feature_padding_mask,
            day_idx=day_idx
        )
        return logits


# ============================================================================
# Validation
# ============================================================================

def validation(model, loader, criterion, device):
    model.eval()
    metrics = {'losses': []}
    
    day_per = {d: {'total_edit_distance': 0, 'total_seq_length': 0}
               for d in range(len(loader.dataset.paths))}
    
    total_edit_distance = 0
    total_seq_length = 0

    with torch.no_grad():
        for batch in loader:
            inputData = batch["feature"].to(device)
            targets = batch["label"].long().to(device)
            feature_mask = batch["feature_mask"].to(device)
            label_mask = batch["label_mask"].to(device)
            day_idx = batch["day"].to(device)

            logits = model(inputData, targets, feature_mask, label_mask, day_idx)
            logits = logits.permute(0, 2, 1)

            loss = criterion(logits, targets[:, 1:])
            metrics['losses'].append(loss.item())

            batch_size = logits.shape[0]
            seq_lens = (~label_mask).sum(dim=1)

            for i in range(batch_size):
                pred_seq = torch.argmax(logits[i, :, :seq_lens[i]], dim=0).cpu().numpy()
                pred_seq = np.array([x for x, _ in itertools.groupby(pred_seq) if x != 0])
                true_seq = targets[i, 1:seq_lens[i]+1].cpu().numpy()

                edit_dist = editdistance.eval(pred_seq, true_seq)
                total_edit_distance += edit_dist
                total_seq_length += len(true_seq)

                day = day_idx[i].item()
                day_per[day]['total_edit_distance'] += edit_dist
                day_per[day]['total_seq_length'] += len(true_seq)

    avg_PER = total_edit_distance / max(total_seq_length, 1)
    metrics['day_PERs'] = day_per
    metrics['avg_PER'] = avg_PER
    metrics['avg_loss'] = sum(metrics['losses']) / max(len(metrics['losses']), 1)
    
    return metrics


# ============================================================================
# Training
# ============================================================================

def train(args):
    # Setup device
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        if args.cuda:
            print("WARNING: CUDA requested but not available, using CPU")
        else:
            print("Using CPU")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print(f"\nLoading data from: {args.data_dir}")
    train_dataset = BrainToTextDataset(
        root_dir=args.data_dir, 
        split='train', 
        max_days=args.max_days
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if args.cuda else False,
        shuffle=True
    )
    
    val_dataset = BrainToTextDataset(
        root_dir=args.data_dir, 
        split='val', 
        max_days=args.max_days,
        paths=[path.replace("train", "val") for path in train_dataset.paths]
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if args.cuda else False
    )
    
    # Create model
    print(f"\nCreating model with d_model={args.d_model}, {train_dataset.days} days")
    model = NeuralToPhonemeTransformer(
        vocab_size=len(LOGIT_TO_PHONEME),
        d_model=args.d_model,
        num_days=train_dataset.days
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'avg_per': [],
        'best_per': float('inf'),
        'best_epoch': 0
    }
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            inputData = batch["feature"].to(device)
            targets = batch["label"].long().to(device)
            feature_mask = batch["feature_mask"].to(device)
            label_mask = batch["label_mask"].to(device)
            day_idx = batch["day"].to(device)
            
            optimizer.zero_grad()
            
            logits = model(inputData, targets, feature_mask, label_mask, day_idx)
            logits = logits.permute(0, 2, 1)
            
            loss = criterion(logits, targets[:, 1:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Log progress
            if batch_idx % args.log_interval == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Validation
        metrics = validation(model, val_loader, criterion, device)
        
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(metrics['avg_loss'])
        history['avg_per'].append(metrics['avg_PER'])
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {metrics['avg_loss']:.4f}")
        print(f"  PER: {metrics['avg_PER']:.4f}")
        print("-" * 60)
        
        # Save best model
        if metrics['avg_PER'] < history['best_per']:
            history['best_per'] = metrics['avg_PER']
            history['best_epoch'] = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'per': metrics['avg_PER'],
                'loss': metrics['avg_loss']
            }, output_dir / 'best_model.pt')
            print(f"  ** New best model saved! PER: {metrics['avg_PER']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    # Save final model and history
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, output_dir / 'final_model.pt')
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, output_dir)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best PER: {history['best_per']:.4f} (Epoch {history['best_epoch']})")
    print(f"Models saved to: {output_dir}")
    
    return history


def plot_training_curves(history, output_dir):
    """Plot and save training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(history['train_losses'], label='Train Loss')
    axes[0].plot(history['val_losses'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PER curve
    axes[1].plot(history['avg_per'], label='PER', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Phoneme Error Rate')
    axes[1].set_title('Phoneme Error Rate over Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Brain-to-Text Transformer')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to the HDF5 data directory')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory for models and logs')
    parser.add_argument('--max-days', type=int, default=1000,
                        help='Maximum number of days/sessions to load')
    
    # Model arguments
    parser.add_argument('--d-model', type=int, default=256,
                        help='Transformer model dimension')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    
    # Device arguments
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use CUDA if available')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda',
                        help='Disable CUDA')
    
    # Logging arguments
    parser.add_argument('--log-interval', type=int, default=40,
                        help='Batch interval for logging')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Epoch interval for saving checkpoints')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("Brain-to-Text Transformer Training")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    print("\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    train(args)


if __name__ == '__main__':
    main()
