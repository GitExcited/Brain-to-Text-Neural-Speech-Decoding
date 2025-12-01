"""
Brain-to-Text V4: Conv1d + BiLSTM + Attention + CTC
--------------------------------------------------------------
Added: Multi-head self-attention after LSTM
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter1d

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Paths
    DATA_ROOT = "/kaggle/input/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final"
    OUTPUT_DIR = "/kaggle/working"
    
    # Data
    NUM_FEATURES = 512
    
    # Model
    HIDDEN_DIM = 512
    NUM_LAYERS = 3
    DROPOUT = 0.4
    BIDIRECTIONAL = True
    KERNEL_SIZE = 3
    STRIDE = 2
    NUM_ATTENTION_HEADS = 8
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-3
    NUM_EPOCHS = 20
    VAL_SPLIT = 0.1
    WEIGHT_DECAY = 1e-2
    
    # Sessions
    SESSIONS = [
        't15.2023.08.11', 't15.2023.08.13', 't15.2023.08.18', 't15.2023.08.20',
        't15.2023.08.25', 't15.2023.08.27', 't15.2023.09.01', 't15.2023.09.03',
        't15.2023.09.24', 't15.2023.09.29', 't15.2023.10.01', 't15.2023.10.06',
        't15.2023.10.08', 't15.2023.10.13', 't15.2023.10.15', 't15.2023.10.20',
        't15.2023.10.22', 't15.2023.11.03', 't15.2023.11.04', 't15.2023.11.17',
        't15.2023.11.19', 't15.2023.11.26', 't15.2023.12.03', 't15.2023.12.08',
        't15.2023.12.10', 't15.2023.12.17', 't15.2023.12.29', 't15.2024.02.25',
        't15.2024.03.03', 't15.2024.03.08', 't15.2024.03.15', 't15.2024.03.17',
        't15.2024.04.25', 't15.2024.04.28', 't15.2024.05.10', 't15.2024.06.14',
        't15.2024.07.19', 't15.2024.07.21', 't15.2024.07.28', 't15.2025.01.10',
        't15.2025.01.12', 't15.2025.03.14', 't15.2025.03.16', 't15.2025.03.30',
        't15.2025.04.13'
    ]
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print("BRAIN-TO-TEXT V4: Conv1d + BiLSTM + Attention + CTC")
print("=" * 80)
print(f"Device: {Config.DEVICE}")

# ============================================================================
# Character Vocabulary
# ============================================================================
CHARS = [' '] + list('abcdefghijklmnopqrstuvwxyz') + ["'"]
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARS)}
BLANK_IDX = len(CHARS)

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
    def __init__(self, session_files, is_train=True):
        self.session_files = session_files
        self.is_train = is_train
        self.samples = []
        self.augment = SpecAugment() if is_train else None
        
        for session, filepath in tqdm(session_files, desc="Indexing"):
            with h5py.File(filepath, 'r') as f:
                for key in f.keys():
                    self.samples.append((filepath, key))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, key = self.samples[idx]
        
        with h5py.File(filepath, 'r') as f:
            trial = f[key]
            neural = np.array(trial['input_features'], dtype=np.float32)
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
        neurals_padded = []
        for neural in neurals:
            if neural.size(0) < max_neural_len:
                pad = torch.zeros(max_neural_len - neural.size(0), Config.NUM_FEATURES)
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
        neurals_padded = []
        for neural in neurals:
            if neural.size(0) < max_neural_len:
                pad = torch.zeros(max_neural_len - neural.size(0), Config.NUM_FEATURES)
                neural = torch.cat([neural, pad], dim=0)
            neurals_padded.append(neural)
        
        neurals_padded = torch.stack(neurals_padded)
        neural_lens = torch.tensor(neural_lens, dtype=torch.long)
        return neurals_padded, neural_lens

# ============================================================================
# Model V3: Conv1d + BiLSTM + Attention
# ============================================================================
class BrainToTextV3(nn.Module):
    def __init__(self, num_chars, hidden_dim, num_layers, dropout, bidirectional, num_heads=8):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(Config.NUM_FEATURES, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(lstm_output_dim)
        
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
        
        # Self-attention with residual connection
        batch_size = output.size(0)
        max_len = output.size(1)
        mask = torch.arange(max_len, device=output.device).expand(batch_size, -1) >= new_lengths.unsqueeze(1)
        
        attn_out, _ = self.attention(output, output, output, key_padding_mask=mask)
        output = self.attn_norm(output + attn_out)  # Residual + LayerNorm
        
        output = self.dropout(output)
        logits = self.output_proj(output)
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs, new_lengths

# ============================================================================
# CTC Decoder
# ============================================================================
def ctc_greedy_decode(log_probs, lengths):
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
        predictions.append(''.join(chars))
    
    return predictions

# ============================================================================
# Training
# ============================================================================
def train_model():
    print("\n" + "=" * 80)
    print("TRAINING (V4 + Attention + CTC)")
    print("=" * 80)
    print(f"Attention heads: {Config.NUM_ATTENTION_HEADS}")
    
    train_session_files = []
    for session in Config.SESSIONS:
        train_file = Path(Config.DATA_ROOT) / session / "data_train.hdf5"
        if train_file.exists():
            train_session_files.append((session, str(train_file)))
    
    val_count = max(1, int(len(train_session_files) * Config.VAL_SPLIT))
    val_files = train_session_files[:val_count]
    train_files = train_session_files[val_count:]
    
    train_dataset = BrainDataset(train_files, is_train=True)
    val_dataset = BrainDataset(val_files, is_train=True)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                             shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    model = BrainToTextV3(
        len(CHARS), Config.HIDDEN_DIM, Config.NUM_LAYERS, 
        Config.DROPOUT, Config.BIDIRECTIONAL, Config.NUM_ATTENTION_HEADS
    ).to(Config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss = 0
        batch_count = 0
        
        print(f"\nEPOCH {epoch+1}/{Config.NUM_EPOCHS}")
        
        pbar = tqdm(train_loader, desc="Train")
        for batch_idx, (neural, neural_lens, char_ids, text_lens, texts) in enumerate(pbar):
            neural = neural.to(Config.DEVICE)
            neural_lens = neural_lens.to(Config.DEVICE)
            char_ids = char_ids.to(Config.DEVICE)
            text_lens = text_lens.to(Config.DEVICE)
            
            log_probs, new_lengths = model(neural, neural_lens)
            log_probs_ctc = log_probs.permute(1, 0, 2)
            
            loss = ctc_loss_fn(log_probs_ctc, char_ids, new_lengths, text_lens)
            
            if torch.isnan(loss):
                continue
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_train_loss = train_loss / max(batch_count, 1)
        print(f"Train Loss: {avg_train_loss:.4f}")
        
        # Validate
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for i, (neural, neural_lens, char_ids, text_lens, true_texts) in enumerate(val_loader):
                neural = neural.to(Config.DEVICE)
                neural_lens = neural_lens.to(Config.DEVICE)
                char_ids = char_ids.to(Config.DEVICE)
                text_lens = text_lens.to(Config.DEVICE)
                
                log_probs, new_lengths = model(neural, neural_lens)
                log_probs_ctc = log_probs.permute(1, 0, 2)
                
                loss = ctc_loss_fn(log_probs_ctc, char_ids, new_lengths, text_lens)
                val_loss += loss.item()
                val_batches += 1
                
                if i == 0:
                    print("-" * 60)
                    preds = ctc_greedy_decode(log_probs[:3], new_lengths[:3])
                    for k in range(min(3, len(preds))):
                        print(f"Pred: {preds[k]}")
                        print(f"True: {true_texts[k]}")
                        match = "✓" if preds[k].strip() == true_texts[k].strip() else ""
                        print(f"{match}")
                    print("-" * 60)

        avg_val_loss = val_loss / max(val_batches, 1)
        print(f"Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Path(Config.OUTPUT_DIR) / "best_model.pt")
            print("✓ Saved Best Model")
            
    return model

# ============================================================================
# Generate Submission
# ============================================================================
def generate_submission(model):
    print("\n" + "=" * 80)
    print("GENERATING SUBMISSION")
    print("=" * 80)
    
    test_files = []
    for session in Config.SESSIONS:
        test_file = Path(Config.DATA_ROOT) / session / "data_test.hdf5"
        if test_file.exists():
            test_files.append((session, str(test_file)))
    
    test_dataset = BrainDataset(test_files, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    try:
        model.load_state_dict(torch.load(Path(Config.OUTPUT_DIR) / "best_model.pt"))
        print("Loaded best model checkpoint.")
    except:
        print("Warning: Could not load best checkpoint, using current weights.")
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for neural, neural_lens in tqdm(test_loader, desc="Predicting"):
            neural = neural.to(Config.DEVICE)
            neural_lens = neural_lens.to(Config.DEVICE)
            
            log_probs, new_lengths = model(neural, neural_lens)
            batch_preds = ctc_greedy_decode(log_probs, new_lengths)
            predictions.extend(batch_preds)
    
    submission = pd.DataFrame({'id': range(len(predictions)), 'text': predictions})
    submission.to_csv(Path(Config.OUTPUT_DIR) / "submission.csv", index=False)
    print(f"\n[OK] Saved submission.csv ({len(predictions)} rows)")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = train_model()
    generate_submission(model)
    print("\nCOMPLETE!")