#!/usr/bin/env python3
"""
Brain-to-Text V2: Conv1d + BiLSTM + CTC + SpecAugment + TextBlob
Containerized training with CUDA support
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Install TextBlob if not available
try:
    from textblob import TextBlob
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', '-q', 'textblob'], check=True)
    from textblob import TextBlob

# ============================================================================
# Character Vocabulary
# ============================================================================
CHARS = [' '] + list('abcdefghijklmnopqrstuvwxyz') + ["'"]
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARS)}
BLANK_IDX = len(CHARS)

# ============================================================================
# TextBlob Correction
# ============================================================================
def correct_text(text):
    """Apply TextBlob spelling correction to text."""
    if not text or len(text.strip()) == 0:
        return text
    try:
        corrected = str(TextBlob(text).correct())
        return corrected
    except:
        return text

def edit_distance(s1, s2):
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

# ============================================================================
# Augmentation (SpecAugment)
# ============================================================================
class SpecAugment(nn.Module):
    def __init__(self, prob=0.5, freq_masks=2, time_masks=2, freq_width=20, time_width=25):
        super().__init__()
        self.prob = prob
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width

    def forward(self, x):
        if not self.training or random.random() > self.prob:
            return x

        x = x.clone()
        seq_len, num_feats = x.shape

        for _ in range(self.freq_masks):
            if num_feats > self.freq_width:
                start = random.randint(0, num_feats - self.freq_width)
                x[:, start:start + self.freq_width] = 0

        for _ in range(self.time_masks):
            if seq_len > self.time_width:
                start = random.randint(0, seq_len - self.time_width)
                x[start:start + self.time_width, :] = 0
        return x

# ============================================================================
# Dataset
# ============================================================================
class BrainDataset(Dataset):
    def __init__(self, root_dir, split="train", max_days=1000, paths=None, num_features=512):
        self.root_dir = root_dir
        self.split = split
        self.num_features = num_features
        self.is_train = (split != "test")
        self.augment = SpecAugment() if self.is_train else None

        # Build paths list if not provided (matches transformer pattern)
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
        self.samples = []

        # Index all trials (lazy loading like transformer but with indexing)
        for filepath in tqdm(self.paths, desc=f"Indexing {split}"):
            with h5py.File(filepath, 'r') as f:
                for key in f.keys():
                    self.samples.append((filepath, key))

        print(f"Loaded {len(self.samples)} trials from {self.days} sessions ({split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, key = self.samples[idx]

        with h5py.File(filepath, 'r') as f:
            trial = f[key]
            neural = np.array(trial['input_features'], dtype=np.float32)

            # Skip trials with NaN/Inf values
            if np.isnan(neural).any() or np.isinf(neural).any():
                # Return a dummy sample, will be handled in collate
                neural = np.zeros((10, self.num_features), dtype=np.float32)

            # Clip extreme values
            neural = np.clip(neural, -10.0, 10.0)
            neural = gaussian_filter1d(neural, sigma=1.0, axis=0)
            neural_len = neural.shape[0]
            neural = torch.from_numpy(neural).float()

            if self.is_train and self.augment is not None:
                neural = self.augment(neural)

            if self.is_train:
                transcription = np.array(trial['transcription'], dtype=np.int32)
                text = ''.join([chr(int(c)) for c in transcription if c > 0])
                text = text.lower()
                text = ''.join([c for c in text if c in CHAR_TO_IDX])

                # Skip if text is empty
                if len(text) == 0:
                    text = " "  # At least return a space

                char_ids = [CHAR_TO_IDX[c] for c in text]
                char_ids = torch.tensor(char_ids, dtype=torch.long)
                text_len = len(char_ids)
                return neural, neural_len, char_ids, text_len, text
            else:
                return neural, neural_len

def collate_fn(batch):
    if len(batch[0]) == 5:
        neurals, neural_lens, char_ids_list, text_lens, texts = zip(*batch)

        max_neural_len = max(neural_lens)
        num_features = batch[0][0].size(1)
        neurals_padded = []
        for neural in neurals:
            if neural.size(0) < max_neural_len:
                pad = torch.zeros(max_neural_len - neural.size(0), num_features)
                neural = torch.cat([neural, pad], dim=0)
            neurals_padded.append(neural)

        neurals_padded = torch.stack(neurals_padded)
        neural_lens = torch.tensor(neural_lens, dtype=torch.long)

        max_text_len = max(text_lens)
        char_ids_padded = []
        for char_ids in char_ids_list:
            if len(char_ids) < max_text_len:
                pad = torch.zeros(max_text_len - len(char_ids), dtype=torch.long)
                char_ids = torch.cat([char_ids, pad], dim=0)
            char_ids_padded.append(char_ids)

        char_ids_padded = torch.stack(char_ids_padded)
        text_lens = torch.tensor(text_lens, dtype=torch.long)

        return neurals_padded, neural_lens, char_ids_padded, text_lens, texts
    else:
        neurals, neural_lens = zip(*batch)
        max_neural_len = max(neural_lens)
        num_features = batch[0][0].size(1)
        neurals_padded = []
        for neural in neurals:
            if neural.size(0) < max_neural_len:
                pad = torch.zeros(max_neural_len - neural.size(0), num_features)
                neural = torch.cat([neural, pad], dim=0)
            neurals_padded.append(neural)

        neurals_padded = torch.stack(neurals_padded)
        neural_lens = torch.tensor(neural_lens, dtype=torch.long)
        return neurals_padded, neural_lens

# ============================================================================
# Model
# ============================================================================
class BrainToTextV2(nn.Module):
    def __init__(self, num_chars, num_features, hidden_dim, num_layers, dropout, bidirectional):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(num_features, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Linear(lstm_output_dim, num_chars + 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, neural, lengths):
        x = neural.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)

        new_lengths = torch.div(lengths + 1, 2, rounding_mode='floor')

        packed = nn.utils.rnn.pack_padded_sequence(
            x, new_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        output = self.dropout(output)
        logits = self.output_proj(output)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, new_lengths

# ============================================================================
# CTC Decoder
# ============================================================================
def ctc_greedy_decode(log_probs, lengths, apply_correction=False):
    batch_size = log_probs.size(0)
    predictions = []

    for i in range(batch_size):
        seq_len = lengths[i]
        pred_ids = log_probs[i, :seq_len].argmax(dim=-1).cpu().numpy()
        chars = []
        prev_idx = None
        for idx in pred_ids:
            if idx != BLANK_IDX and idx != prev_idx:
                chars.append(IDX_TO_CHAR[idx])
            prev_idx = idx
        raw_pred = ''.join(chars)

        if apply_correction:
            predictions.append(correct_text(raw_pred))
        else:
            predictions.append(raw_pred)

    return predictions

# ============================================================================
# Loss with Correction Bonus
# ============================================================================
def compute_loss_with_correction_bonus(log_probs, log_probs_ctc, new_lengths,
                                        char_ids, text_lens, true_texts,
                                        ctc_loss_fn, bonus_weight=0.1):
    """
    CTC loss + bonus if TextBlob correction improves toward ground truth.
    """
    # Standard CTC loss
    base_loss = ctc_loss_fn(log_probs_ctc, char_ids, new_lengths, text_lens)

    if torch.isnan(base_loss):
        return base_loss, float('nan'), 0.0

    # Decode predictions
    with torch.no_grad():
        raw_preds = ctc_greedy_decode(log_probs, new_lengths, apply_correction=False)
        corrected_preds = ctc_greedy_decode(log_probs, new_lengths, apply_correction=True)

    # Calculate bonus based on how well correction helps
    bonus = 0.0
    for raw, corrected, true_text in zip(raw_preds, corrected_preds, true_texts):
        true_text = true_text.strip()
        raw = raw.strip()
        corrected = corrected.strip()

        if len(true_text) == 0:
            continue

        # Calculate edit distances
        raw_dist = edit_distance(raw, true_text)
        corrected_dist = edit_distance(corrected, true_text)

        # Normalize by true text length
        max_len = max(len(true_text), 1)
        raw_error = raw_dist / max_len
        corrected_error = corrected_dist / max_len

        if corrected == true_text:
            # Perfect match after correction: big bonus
            bonus += 1.0
        elif corrected_error < raw_error:
            # Correction improved: partial bonus
            improvement = raw_error - corrected_error
            bonus += min(improvement * 2, 0.5)  # Cap at 0.5
        elif raw == true_text:
            # Raw was already perfect: small bonus
            bonus += 0.3

    bonus = bonus / max(len(raw_preds), 1)  # Normalize by batch size

    # Final loss: subtract bonus (lower loss = better)
    final_loss = base_loss - (bonus_weight * bonus)

    return final_loss, base_loss.item(), bonus

# ============================================================================
# Training
# ============================================================================
def train_model(args):
    print("\n" + "=" * 80)
    print("TRAINING: BRAIN-TO-TEXT V2 (Conv1d + BiLSTM + TextBlob)")
    print("=" * 80)
    print(f"Bonus weight: {args.bonus_weight}")
    print(f"Bonus computed every {args.bonus_every_n_batches} batches")

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

    # Load datasets using transformer pattern
    print(f"\nLoading data from: {args.data_dir}")
    train_dataset = BrainDataset(
        root_dir=args.data_dir,
        split='train',
        max_days=args.max_days,
        num_features=args.num_features
    )

    # Filter validation paths to only include files that exist
    val_paths = [path.replace("train", "val") for path in train_dataset.paths]
    val_paths = [p for p in val_paths if os.path.exists(p)]
    print(f"Found {len(val_paths)} validation files (some sessions may not have val data)")

    val_dataset = BrainDataset(
        root_dir=args.data_dir,
        split='val',
        max_days=args.max_days,
        paths=val_paths,
        num_features=args.num_features
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if args.cuda else False,
        shuffle=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if args.cuda else False,
        persistent_workers=True if args.num_workers > 0 else False
    )

    model = BrainToTextV2(
        len(CHARS), args.num_features, args.hidden_dim, args.num_layers,
        args.dropout, args.bidirectional
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    best_val_loss = float('inf')
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_bonuses': [],
        'val_bonuses': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_bonus = 0
        batch_count = 0
        bonus_count = 0

        print(f"\nEPOCH {epoch+1}/{args.epochs}")

        pbar = tqdm(train_loader, desc="Train")
        for batch_idx, (neural, neural_lens, char_ids, text_lens, texts) in enumerate(pbar):
            try:
                # Use non_blocking=True with pin_memory for async CPU->GPU transfer
                neural = neural.to(device, non_blocking=True)
                neural_lens = neural_lens.to(device, non_blocking=True)
                char_ids = char_ids.to(device, non_blocking=True)
                text_lens = text_lens.to(device, non_blocking=True)

                # Skip batches with invalid input
                if torch.isnan(neural).any() or torch.isinf(neural).any():
                    print(f"WARNING: NaN/Inf in input features at batch {batch_idx}, skipping...")
                    del neural, neural_lens, char_ids, text_lens
                    if args.cuda:
                        torch.cuda.empty_cache()
                    continue

                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                log_probs, new_lengths = model(neural, neural_lens)
                log_probs_ctc = log_probs.permute(1, 0, 2)  # (T, N, C)

                # Compute loss with or without bonus
                if batch_idx % args.bonus_every_n_batches == 0:
                    # Compute full loss with correction bonus
                    loss, base_loss, bonus = compute_loss_with_correction_bonus(
                        log_probs, log_probs_ctc, new_lengths,
                        char_ids, text_lens, texts,
                        ctc_loss_fn, args.bonus_weight
                    )
                    train_bonus += bonus
                    bonus_count += 1
                else:
                    # Just CTC loss for speed
                    loss = ctc_loss_fn(log_probs_ctc, char_ids, new_lengths, text_lens)
                    base_loss = loss.item()
                    bonus = 0.0

                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: NaN/Inf loss detected at batch {batch_idx}, skipping...")
                    optimizer.zero_grad(set_to_none=True)
                    del neural, neural_lens, char_ids, text_lens, log_probs, log_probs_ctc, loss
                    if args.cuda:
                        torch.cuda.empty_cache()
                    continue

                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += base_loss
                batch_count += 1
                pbar.set_postfix({
                    'loss': f"{base_loss:.3f}",
                    'bonus': f"{bonus:.3f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })

                # Clean up to free memory
                del neural, neural_lens, char_ids, text_lens, log_probs, log_probs_ctc

            except torch.cuda.OutOfMemoryError:
                print(f"WARNING: CUDA OOM at batch {batch_idx}, clearing cache and skipping...")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

        avg_train_loss = train_loss / max(batch_count, 1)
        avg_train_bonus = train_bonus / max(bonus_count, 1)
        history['train_losses'].append(avg_train_loss)
        history['train_bonuses'].append(avg_train_bonus)
        print(f"Train Loss: {avg_train_loss:.4f} | Avg Bonus: {avg_train_bonus:.4f}")

        # Validate
        model.eval()
        val_loss = 0
        val_bonus = 0
        val_batches = 0

        with torch.no_grad():
            for i, (neural, neural_lens, char_ids, text_lens, true_texts) in enumerate(val_loader):
                neural = neural.to(device)
                neural_lens = neural_lens.to(device)
                char_ids = char_ids.to(device)
                text_lens = text_lens.to(device)

                log_probs, new_lengths = model(neural, neural_lens)
                log_probs_ctc = log_probs.permute(1, 0, 2)

                # Compute loss with bonus for validation
                loss, base_loss, bonus = compute_loss_with_correction_bonus(
                    log_probs, log_probs_ctc, new_lengths,
                    char_ids, text_lens, true_texts,
                    ctc_loss_fn, args.bonus_weight
                )
                val_loss += base_loss
                val_bonus += bonus
                val_batches += 1

                # Show examples
                if i == 0:
                    print("-" * 60)
                    raw_preds = ctc_greedy_decode(log_probs[:3], new_lengths[:3], apply_correction=False)
                    corrected_preds = ctc_greedy_decode(log_probs[:3], new_lengths[:3], apply_correction=True)
                    for k in range(min(3, len(raw_preds))):
                        print(f"Raw:       {raw_preds[k]}")
                        print(f"Corrected: {corrected_preds[k]}")
                        print(f"True:      {true_texts[k]}")
                        print()
                    print("-" * 60)

        avg_val_loss = val_loss / max(val_batches, 1)
        avg_val_bonus = val_bonus / max(val_batches, 1)
        history['val_losses'].append(avg_val_loss)
        history['val_bonuses'].append(avg_val_bonus)
        print(f"Val Loss: {avg_val_loss:.4f} | Val Bonus: {avg_val_bonus:.4f}")

        # Free up GPU memory after validation
        if args.cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            history['best_val_loss'] = avg_val_loss
            history['best_epoch'] = epoch + 1
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print("✓ Saved Best Model")

    # Save final model and history
    torch.save(model.state_dict(), output_dir / "final_model.pt")

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    plot_training_curves(history, output_dir)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Best Val Loss: {history['best_val_loss']:.4f} (Epoch {history['best_epoch']})")
    print(f"Models saved to: {output_dir}")

    return model

# ============================================================================
# Generate Submission
# ============================================================================
def generate_submission(model, args):
    print("\n" + "=" * 80)
    print("GENERATING SUBMISSION (with TextBlob correction)")
    print("=" * 80)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load test dataset using transformer pattern
    test_dataset = BrainDataset(
        root_dir=args.data_dir,
        split='test',
        max_days=args.max_days,
        num_features=args.num_features
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    try:
        model.load_state_dict(torch.load(output_dir / "best_model.pt"))
        print("Loaded best model checkpoint.")
    except:
        print("Warning: Could not load best checkpoint, using current weights.")

    model.eval()
    predictions = []

    with torch.no_grad():
        for neural, neural_lens in tqdm(test_loader, desc="Predicting"):
            neural = neural.to(device)
            neural_lens = neural_lens.to(device)

            log_probs, new_lengths = model(neural, neural_lens)
            # Apply TextBlob correction to final predictions
            batch_preds = ctc_greedy_decode(log_probs, new_lengths, apply_correction=True)
            predictions.extend(batch_preds)

    submission = pd.DataFrame({'id': range(len(predictions)), 'text': predictions})
    submission.to_csv(output_dir / "submission.csv", index=False)
    print(f"\n[OK] Saved submission.csv ({len(predictions)} rows)")

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

    # Bonus curves
    axes[0, 1].plot(history['train_bonuses'], label='Train Bonus', color='green')
    axes[0, 1].plot(history['val_bonuses'], label='Val Bonus', color='blue')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Correction Bonus')
    axes[0, 1].set_title('Correction Bonus over Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Loss difference
    if len(history['train_losses']) > 0 and len(history['val_losses']) > 0:
        loss_diff = [v - t for t, v in zip(history['train_losses'], history['val_losses'])]
        axes[1, 0].plot(loss_diff, label='Val - Train Loss', color='purple')
        axes[1, 0].axhline(y=0, color='gray', linestyle='--')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Difference')
        axes[1, 0].set_title('Overfitting Indicator (Val - Train Loss)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Best val loss marker
    axes[1, 1].plot(history['val_losses'], label='Val Loss', color='orange')
    if 'best_epoch' in history and history['best_epoch'] > 0:
        best_epoch = history['best_epoch'] - 1
        axes[1, 1].axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch: {history["best_epoch"]}')
        axes[1, 1].scatter([best_epoch], [history['val_losses'][best_epoch]], color='red', s=100, zorder=5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Loss')
    axes[1, 1].set_title('Validation Loss with Best Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train Brain-to-Text BiLSTM Model')

    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to the HDF5 data directory')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory for models and logs')
    parser.add_argument('--max-days', type=int, default=1000,
                        help='Maximum number of days/sessions to load (0 for all)')

    # Model arguments
    parser.add_argument('--num-features', type=int, default=512,
                        help='Number of input features')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                        help='Use bidirectional LSTM')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loader workers')

    # Correction bonus arguments
    parser.add_argument('--bonus-weight', type=float, default=0.1,
                        help='Weight of correction bonus')
    parser.add_argument('--bonus-every-n-batches', type=int, default=5,
                        help='Compute bonus every N batches')

    # Device arguments
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use CUDA if available')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda',
                        help='Disable CUDA')

    # Mode arguments
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'inference', 'both'],
                        help='Mode: train, inference, or both')

    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("Brain-to-Text BiLSTM Training")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    print("\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 80)

    if args.mode in ['train', 'both']:
        model = train_model(args)

    if args.mode in ['inference', 'both']:
        if args.mode == 'inference':
            # Load model for inference-only mode
            device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
            model = BrainToTextV2(
                len(CHARS), args.num_features, args.hidden_dim, args.num_layers,
                args.dropout, args.bidirectional
            ).to(device)
        generate_submission(model, args)

if __name__ == '__main__':
    main()
