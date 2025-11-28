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
from torch.utils.tensorboard import SummaryWriter

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
                
                # Look for data_train.hdf5, data_val.hdf5, data_test.hdf5 pattern
                expected_file = f"data_{split}.hdf5"
                file_path = os.path.join(sessionPath, expected_file)
                
                if os.path.exists(file_path):
                    paths.append(file_path)
                    days_loaded += 1
                else:
                    # Fallback: try old pattern (test.hdf5, train.hdf5, val.hdf5)
                    session_files = sorted(os.listdir(sessionPath))
                    if len(session_files) == 3:
                        data_type = 0  # test
                        if split == "train":
                            data_type = 1
                        elif split == "val":
                            data_type = 2
                        paths.append(os.path.join(sessionPath, session_files[data_type]))
                        days_loaded += 1
                
        self.paths = paths
        self.days = len(self.paths)
        
        self.X = []
        skipped = 0
        for day_idx, train_path in enumerate(paths):
            with h5py.File(train_path, 'r') as f:
                for trial_idx, trial in enumerate(f):
                    X_trial = np.array(f[trial]["input_features"], dtype=np.float32)
                    y_trial = np.array(f[trial]["seq_class_ids"])
                    
                    # Skip trials with NaN/Inf values
                    if np.isnan(X_trial).any() or np.isinf(X_trial).any():
                        skipped += 1
                        continue
                    
                    # Skip empty or very short sequences
                    if len(X_trial) < 2 or len(y_trial) < 2:
                        skipped += 1
                        continue
                    
                    # Clip extreme values to prevent overflow
                    X_trial = np.clip(X_trial, -10.0, 10.0)
                    
                    self.X.append({
                        "features": torch.from_numpy(X_trial),
                        "labels": torch.from_numpy(y_trial),
                        "feature_len": len(X_trial),  # Store actual length
                        "label_len": len(y_trial),    # Store actual length
                        "day_id": day_idx,
                        "trial_id": trial_idx
                    })
        
        if skipped > 0:
            print(f"Skipped {skipped} trials with NaN/Inf/empty values")
        print(f"Loaded {len(self.X)} trials from {self.days} days ({split})")
        
        # Analyze label distribution
        all_labels = []
        for item in self.X:
            all_labels.extend(item["labels"].numpy().tolist())
        from collections import Counter
        label_counts = Counter(all_labels)
        total_labels = len(all_labels)
        print(f"\nLabel distribution ({split}):")
        for label_id, count in sorted(label_counts.items())[:10]:  # Top 10
            pct = 100 * count / total_labels
            name = LOGIT_TO_PHONEME[label_id] if label_id < len(LOGIT_TO_PHONEME) else f"UNK_{label_id}"
            print(f"  {label_id} ({name}): {count} ({pct:.1f}%)")
        print(f"  ... ({len(label_counts)} unique labels total)")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def custom_collate_fn(batch):
    features = [d["features"] for d in batch]
    labels = [d["labels"] for d in batch]
    day_ids = [d["day_id"] for d in batch]
    trial_ids = [d["trial_id"] for d in batch]
    feature_lens = [d["feature_len"] for d in batch]
    label_lens = [d["label_len"] for d in batch]

    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    # Use -100 as padding for labels (PyTorch's default ignore_index)
    # This way we can properly ignore padding while still learning BLANK (index 0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    batch_size = padded_features.size(0)
    max_feature_len = padded_features.size(1)
    max_label_len = padded_labels.size(1)
    
    # Create masks based on actual lengths, not values
    # True = masked (ignored), False = valid
    feature_padding_mask = torch.zeros(batch_size, max_feature_len, dtype=torch.bool)
    label_padding_mask = torch.zeros(batch_size, max_label_len, dtype=torch.bool)
    
    for i in range(batch_size):
        if feature_lens[i] < max_feature_len:
            feature_padding_mask[i, feature_lens[i]:] = True
        if label_lens[i] < max_label_len:
            label_padding_mask[i, label_lens[i]:] = True

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
    def __init__(self, d_model, max_len=10000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
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
        return self.dropout(x)


def init_weights_xavier(module):
    """Apply Xavier initialization to linear layers and embeddings with small gain for stability."""
    if isinstance(module, nn.Linear):
        # Use smaller gain for better stability
        nn.init.xavier_uniform_(module.weight, gain=0.5)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Normal init for embeddings is more stable
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class NeuralEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(512, d_model)
        self.input_norm = nn.LayerNorm(d_model)  # Normalize after projection
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, neural_feature, neural_feature_key_padding_mask=None):
        neural_feature = self.input_proj(neural_feature)
        neural_feature = self.input_norm(neural_feature)  # Normalize for stability
        neural_feature = self.pos(neural_feature)
        return self.encoder(
            neural_feature,
            src_key_padding_mask=neural_feature_key_padding_mask
        )


class NeuralDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, num_days=1000, dropout=0.1):
        super().__init__()
        # Don't use padding_idx=0 because 0=BLANK is a valid token we want to learn
        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed_scale = math.sqrt(d_model)  # Scale embeddings
        self.day_embed = nn.Embedding(num_days, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        self.embed_norm = nn.LayerNorm(d_model)  # Normalize embeddings
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, class_id, memory, day_idx, class_id_mask=None, 
                class_id_key_padding_mask=None, memory_key_padding_mask=None):
        class_id = self.embed(class_id) * self.embed_scale  # Scale embeddings
        class_id = self.pos(class_id)
        
        day_vec = self.day_embed(day_idx)
        day_vec = day_vec.unsqueeze(1)
        class_id = class_id + day_vec
        class_id = self.embed_norm(class_id)  # Normalize before decoder
        
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
        
        # Apply Xavier initialization
        self.apply(init_weights_xavier)
        print("Applied Xavier initialization to all layers")

    def forward(self, neural_feature_input, class_id_tokens, 
                neural_feature_padding_mask, class_id_padding_mask, day_idx):
        class_id_in = class_id_tokens[:, :-1]
        seq_len = class_id_in.size(1)
        
        # Safety check: ensure seq_len > 0
        if seq_len == 0:
            raise ValueError("Sequence length is 0, cannot process empty sequences")
        
        # Create mask on the same device as input
        subsequent_mask = make_subsequent_mask(seq_len, class_id_in.device)
        
        # Safety: ensure masks don't mask everything (would cause NaN in attention)
        # If all positions are masked, unmask the first position
        if neural_feature_padding_mask is not None:
            all_masked = neural_feature_padding_mask.all(dim=1)
            if all_masked.any():
                neural_feature_padding_mask = neural_feature_padding_mask.clone()
                neural_feature_padding_mask[all_masked, 0] = False
        
        if class_id_padding_mask is not None:
            # Adjust for the :-1 slice
            adjusted_mask = class_id_padding_mask[:, :-1]
            all_masked = adjusted_mask.all(dim=1)
            if all_masked.any():
                class_id_padding_mask = class_id_padding_mask.clone()
                class_id_padding_mask[all_masked, 0] = False
        
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
# Learning Rate Scheduler with Warmup
# ============================================================================

class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine decay."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = lr
        return lr
    
    def get_lr(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * (self.current_step / max(1, self.warmup_steps))
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (1 + math.cos(math.pi * progress))
    
    def state_dict(self):
        return {'current_step': self.current_step}
    
    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']


# ============================================================================
# Early Stopping
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False
    
    def state_dict(self):
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop
        }
    
    def load_state_dict(self, state_dict):
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']


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
            inputData = batch["feature"].to(device, non_blocking=True)
            targets = batch["label"].long().to(device, non_blocking=True)
            feature_mask = batch["feature_mask"].to(device, non_blocking=True)
            label_mask = batch["label_mask"].to(device, non_blocking=True)
            day_idx = batch["day"].to(device, non_blocking=True)

            try:
                logits = model(inputData, targets, feature_mask, label_mask, day_idx)
                logits = logits.permute(0, 2, 1)

                loss = criterion(logits, targets[:, 1:])
                
                # Skip NaN losses in validation too
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                    
                metrics['losses'].append(loss.item())

                batch_size = logits.shape[0]
                # Use label_mask to get actual sequence lengths
                seq_lens = (~label_mask).sum(dim=1)

                for i in range(batch_size):
                    actual_len = min(seq_lens[i].item(), logits.shape[2])
                    if actual_len <= 1:
                        continue
                    pred_seq = torch.argmax(logits[i, :, :actual_len-1], dim=0).cpu().numpy()
                    pred_seq = np.array([x for x, _ in itertools.groupby(pred_seq) if x != 0])
                    true_seq = targets[i, 1:actual_len].cpu().numpy()

                    edit_dist = editdistance.eval(pred_seq, true_seq)
                    total_edit_distance += edit_dist
                    total_seq_length += len(true_seq)

                    day = day_idx[i].item()
                    if day in day_per:
                        day_per[day]['total_edit_distance'] += edit_dist
                        day_per[day]['total_seq_length'] += len(true_seq)
            except Exception as e:
                print(f"Validation batch error: {e}")
                continue

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
        
        # Enable TF32 for Ampere GPUs (RTX 30xx, A100, etc.) - ~2x speedup
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for faster matrix operations")
    else:
        device = torch.device("cpu")
        if args.cuda:
            print("WARNING: CUDA requested but not available, using CPU")
        else:
            print("Using CPU")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=output_dir / 'tensorboard')
    print(f"TensorBoard logs: {output_dir / 'tensorboard'}")
    
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
        shuffle=True,
        persistent_workers=True if args.num_workers > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=2 if args.num_workers > 0 else None  # Prefetch batches
    )
    
    # Filter validation paths to only include files that exist
    val_paths = [path.replace("train", "val") for path in train_dataset.paths]
    val_paths = [p for p in val_paths if os.path.exists(p)]
    print(f"Found {len(val_paths)} validation files (some sessions may not have val data)")
    
    val_dataset = BrainToTextDataset(
        root_dir=args.data_dir, 
        split='val', 
        max_days=args.max_days,
        paths=val_paths
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if args.cuda else False,
        persistent_workers=True if args.num_workers > 0 else False
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
    # Use ignore_index=-100 to ignore padding tokens (not BLANK which is index 0)
    # This ensures BLANK predictions are penalized correctly
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8)
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(len(train_loader) * args.warmup_epochs, total_steps // 10)
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=args.lr * 0.01
    )
    print(f"LR Scheduler: {warmup_steps} warmup steps, {total_steps} total steps")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        mode='min'
    )
    print(f"Early stopping: patience={args.patience}, min_delta={args.min_delta}")
    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'avg_per': [],
        'learning_rates': [],
        'best_per': float('inf'),
        'best_epoch': 0
    }
    
    # Resume from checkpoint if exists
    start_epoch = 0
    resume_checkpoint = output_dir / 'latest_checkpoint.pt'
    if args.resume and resume_checkpoint.exists():
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        early_stopping.load_state_dict(checkpoint['early_stopping_state_dict'])
        history = checkpoint['history']
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}, best PER: {history['best_per']:.4f}")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = []
        epoch_lrs = []
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Use non_blocking=True with pin_memory for async CPU->GPU transfer
                inputData = batch["feature"].to(device, non_blocking=True)
                targets = batch["label"].long().to(device, non_blocking=True)
                feature_mask = batch["feature_mask"].to(device, non_blocking=True)
                label_mask = batch["label_mask"].to(device, non_blocking=True)
                day_idx = batch["day"].to(device, non_blocking=True)
                
                # Skip batches with invalid input (all zeros, NaN in features)
                if torch.isnan(inputData).any() or torch.isinf(inputData).any():
                    print(f"WARNING: NaN/Inf in input features at batch {batch_idx}, skipping...")
                    del inputData, targets, feature_mask, label_mask, day_idx
                    torch.cuda.empty_cache()
                    continue
                
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                logits = model(inputData, targets, feature_mask, label_mask, day_idx)
                logits = logits.permute(0, 2, 1)
                
                loss = criterion(logits, targets[:, 1:])
                
                # Check for NaN loss and skip batch if detected
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: NaN/Inf loss detected at batch {batch_idx}, skipping...")
                    optimizer.zero_grad(set_to_none=True)
                    del inputData, targets, feature_mask, label_mask, day_idx, logits, loss
                    torch.cuda.empty_cache()
                    continue
                
                loss.backward()
                
                # Check for NaN gradients
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"WARNING: NaN/Inf gradients detected at batch {batch_idx}, skipping...")
                    optimizer.zero_grad(set_to_none=True)
                    del inputData, targets, feature_mask, label_mask, day_idx, logits, loss
                    torch.cuda.empty_cache()
                    continue
                
                # Clip gradients with a stricter threshold
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                # Log progress (before cleanup)
                if batch_idx % args.log_interval == 0:
                    print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {scheduler.get_lr():.2e}")
                    
                    # Debug: Show what the model is predicting (first sample in batch)
                    with torch.no_grad():
                        sample_pred = torch.argmax(logits[0], dim=0)[:20].cpu().tolist()
                        sample_true = targets[0, 1:21].cpu().tolist()
                        pred_unique = len(set(sample_pred))
                        print(f"  DEBUG - Pred (first 20): {sample_pred}")
                        print(f"  DEBUG - True (first 20): {sample_true}")
                        print(f"  DEBUG - Unique predictions: {pred_unique}/20")
                
                # Clean up to free memory
                del inputData, targets, feature_mask, label_mask, day_idx, logits
                
            except torch.cuda.OutOfMemoryError:
                print(f"WARNING: CUDA OOM at batch {batch_idx}, clearing cache and skipping...")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            
            # Step the learning rate scheduler
            current_lr = scheduler.step()
            epoch_lrs.append(current_lr)
            
            epoch_losses.append(loss.item())
        
        # Validation
        metrics = validation(model, val_loader, criterion, device)
        
        # Free up GPU memory after validation
        if args.cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        avg_lr = sum(epoch_lrs) / len(epoch_lrs)
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(metrics['avg_loss'])
        history['avg_per'].append(metrics['avg_PER'])
        history['learning_rates'].append(avg_lr)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {metrics['avg_loss']:.4f}")
        print(f"  PER: {metrics['avg_PER']:.4f}")
        print(f"  Avg LR: {avg_lr:.2e}")
        print("-" * 60)
        
        # Log to TensorBoard
        writer.add_scalars('Loss', {'train': avg_train_loss, 'val': metrics['avg_loss']}, epoch + 1)
        writer.add_scalar('PER', metrics['avg_PER'], epoch + 1)
        writer.add_scalar('Learning_Rate', avg_lr, epoch + 1)
        writer.flush()
        
        # Save best model
        if metrics['avg_PER'] < history['best_per']:
            history['best_per'] = metrics['avg_PER']
            history['best_epoch'] = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'per': metrics['avg_PER'],
                'loss': metrics['avg_loss'],
                'history': history
            }, output_dir / 'best_model.pt')
            print(f"  ** New best model saved! PER: {metrics['avg_PER']:.4f}")
        
        # Save latest checkpoint (always, for resumption)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'early_stopping_state_dict': early_stopping.state_dict(),
            'history': history
        }, output_dir / 'latest_checkpoint.pt')
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pt')
        
        # Check early stopping
        if early_stopping(metrics['avg_PER']):
            print(f"\n** Early stopping triggered! No improvement for {args.patience} epochs.")
            print(f"** Best PER: {history['best_per']:.4f} at epoch {history['best_epoch']}")
            break
    
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_losses'], label='Train Loss')
    axes[0, 0].plot(history['val_losses'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PER curve
    axes[0, 1].plot(history['avg_per'], label='PER', color='green')
    axes[0, 1].axhline(y=history['best_per'], color='red', linestyle='--', label=f'Best PER: {history["best_per"]:.4f}')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Phoneme Error Rate')
    axes[0, 1].set_title('Phoneme Error Rate over Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate curve
    if 'learning_rates' in history and history['learning_rates']:
        axes[1, 0].plot(history['learning_rates'], label='Learning Rate', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # Train vs Val loss difference (overfitting indicator)
    if len(history['train_losses']) > 0 and len(history['val_losses']) > 0:
        loss_diff = [v - t for t, v in zip(history['train_losses'], history['val_losses'])]
        axes[1, 1].plot(loss_diff, label='Val - Train Loss', color='purple')
        axes[1, 1].axhline(y=0, color='gray', linestyle='--')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].set_title('Overfitting Indicator (Val - Train Loss)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
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
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (reduced default for stability)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    
    # Learning rate scheduler arguments
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='Number of warmup epochs for LR scheduler')
    
    # Early stopping arguments
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--min-delta', type=float, default=0.001,
                        help='Minimum improvement for early stopping')
    
    # Resume training
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    
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
