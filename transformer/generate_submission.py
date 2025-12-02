#!/usr/bin/env python3
"""
Generate Kaggle submission using trained transformer model.
"""

import os
import argparse
import torch
import h5py
import numpy as np
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict
from train_transformer import NeuralToPhonemeTransformer, LOGIT_TO_PHONEME, make_subsequent_mask


def load_phoneme_dict(dict_path):
    """Load CMU phoneme dictionary."""
    phoneme_to_words = defaultdict(list)

    print(f"Loading phoneme dictionary from: {dict_path}")
    with open(dict_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';;;'):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            word = parts[0]
            # Remove stress markers (0, 1, 2) from phonemes
            phonemes = tuple(re.sub(r'[0-9]', '', p) for p in parts[1:])

            # Skip punctuation and special symbols (but keep apostrophes for contractions)
            # Competition allows apostrophes in contractions like "don't"
            if any(c in word for c in ['!', '?', '.', ',', ':', ';', '"', '(', ')', '#', '%', '&', '+', '-', '/', '@']):
                continue

            word_clean = word.lower()
            phoneme_to_words[phonemes].append(word_clean)

    print(f"Loaded {len(phoneme_to_words)} phoneme sequences from dictionary")
    return phoneme_to_words


def phonemes_to_text(phoneme_indices, phoneme_dict):
    """Convert phoneme indices to text using greedy longest-match."""
    # Collapse consecutive duplicates and remove BLANK (0) and silence (40)
    phonemes = []
    prev_idx = -1
    for idx in phoneme_indices:
        if idx != prev_idx and idx > 0 and idx < len(LOGIT_TO_PHONEME) - 1:
            phonemes.append(LOGIT_TO_PHONEME[idx])
        prev_idx = idx

    if not phonemes:
        return ""

    # Greedy longest-match decoding
    words = []
    i = 0
    while i < len(phonemes):
        matched = False
        # Try to match longest sequence first (up to 10 phonemes)
        for length in range(min(10, len(phonemes) - i), 0, -1):
            phoneme_seq = tuple(phonemes[i:i+length])
            if phoneme_seq in phoneme_dict:
                words.append(phoneme_dict[phoneme_seq][0])
                i += length
                matched = True
                break

        if not matched:
            # Skip this phoneme if we can't match it
            i += 1

    return ' '.join(words)


@torch.no_grad()
def predict_sequence(model, neural_features, day_idx, device, max_length=200):
    """Generate phoneme sequence autoregressively."""
    model.eval()

    # Prepare input [1, T, 512]
    neural_features = torch.from_numpy(neural_features).float().unsqueeze(0).to(device)
    day_tensor = torch.tensor([day_idx], dtype=torch.long).to(device)

    # Encode neural features once
    memory = model.encoder(neural_features)

    # Start with silence token (index 40)
    predicted = [40]

    for step in range(max_length):
        # Current sequence as input
        tgt = torch.tensor([predicted], dtype=torch.long).to(device)

        # Create causal mask
        tgt_mask = make_subsequent_mask(len(predicted), device)

        # Decode
        logits = model.decoder(
            tgt, memory, day_tensor,
            class_id_mask=tgt_mask
        )

        # Get next token (greedy decoding)
        next_token = logits[0, -1].argmax().item()
        predicted.append(next_token)

        # Stop if predicting silence repeatedly
        if len(predicted) > 5 and all(t == 40 for t in predicted[-5:]):
            break

    return np.array(predicted)


def main():
    parser = argparse.ArgumentParser(description='Generate Kaggle submission')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to HDF5 data directory')
    parser.add_argument('--dict-path', type=str, required=True,
                        help='Path to phoneme dictionary (dict.txt)')
    parser.add_argument('--output-csv', type=str, default='submission.csv',
                        help='Output CSV file path')
    parser.add_argument('--d-model', type=int, default=896,
                        help='Model dimension (must match checkpoint)')
    parser.add_argument('--num-days', type=int, default=45,
                        help='Number of training days (must match checkpoint)')
    parser.add_argument('--max-length', type=int, default=200,
                        help='Maximum sequence length for generation')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use CUDA if available')

    args = parser.parse_args()

    # Setup device
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load phoneme dictionary
    phoneme_dict = load_phoneme_dict(args.dict_path)

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    print(f"Model checkpoint - Epoch: {checkpoint['epoch']}, PER: {checkpoint['per']:.4f}")

    model = NeuralToPhonemeTransformer(
        vocab_size=len(LOGIT_TO_PHONEME),
        d_model=args.d_model,
        num_days=args.num_days
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")

    # Get all sessions in chronological order
    print(f"\nScanning data directory: {args.data_dir}")
    sessions = sorted(os.listdir(args.data_dir))
    session_to_day = {s: i for i, s in enumerate(sessions)}
    print(f"Found {len(sessions)} sessions")

    # Collect all test trials in chronological order
    test_trials = []
    for session in sessions:
        test_file = os.path.join(args.data_dir, session, 'data_test.hdf5')
        if not os.path.exists(test_file):
            print(f"Warning: No test file for {session}")
            continue

        with h5py.File(test_file, 'r') as f:
            for trial_idx, trial_key in enumerate(f.keys()):
                # Try to parse block and trial from key if it has the expected format
                # Otherwise just use the trial index
                try:
                    parts = trial_key.split('_')
                    if len(parts) >= 4:
                        block = int(parts[1])
                        trial = int(parts[3])
                    else:
                        block = 0
                        trial = trial_idx
                except (ValueError, IndexError):
                    block = 0
                    trial = trial_idx

                test_trials.append({
                    'session': session,
                    'block': block,
                    'trial': trial,
                    'key': trial_key,
                    'file': test_file,
                    'day_idx': session_to_day[session]
                })

    # Sort by session date, then block, then trial
    test_trials.sort(key=lambda x: (x['session'], x['block'], x['trial']))
    print(f"Total test trials: {len(test_trials)}")
    print(f"First trial: {test_trials[0]['session']} - {test_trials[0]['key']}")
    print(f"Last trial: {test_trials[-1]['session']} - {test_trials[-1]['key']}")

    # Generate predictions
    print("\n" + "="*60)
    print("Generating predictions...")
    print("="*60)

    predictions = []
    import time
    start_time = time.time()

    for idx, trial in enumerate(test_trials):
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            if idx > 0:
                avg_time = elapsed / idx
                eta = avg_time * (len(test_trials) - idx)
                print(f"Processing trial {idx}/{len(test_trials)} | "
                      f"Avg: {avg_time:.2f}s/trial | ETA: {eta/60:.1f}m")
            else:
                print(f"Processing trial {idx}/{len(test_trials)}...")

        # Load trial data
        with h5py.File(trial['file'], 'r') as f:
            features = np.array(f[trial['key']]['input_features'], dtype=np.float32)

        # Clip extreme values (same as training)
        features = np.clip(features, -10.0, 10.0)

        # Generate phoneme sequence
        phoneme_ids = predict_sequence(
            model, features, trial['day_idx'],
            device, max_length=args.max_length
        )

        # Convert phonemes to text
        text = phonemes_to_text(phoneme_ids, phoneme_dict)

        # Remove remaining punctuation (but keep apostrophes for contractions)
        # Competition allows apostrophes in contractions like "don't"
        text = re.sub(r'[.,!?;:"]', '', text)

        predictions.append({'id': idx, 'text': text})

    # Save submission
    print(f"\n{'='*60}")
    print("Saving submission...")
    df = pd.DataFrame(predictions)
    df.to_csv(args.output_csv, index=False)

    print(f"✓ Saved {len(df)} predictions to: {args.output_csv}")
    print(f"✓ File size: {os.path.getsize(args.output_csv) / 1024:.2f} KB")

    # Show sample predictions
    print(f"\nSample predictions:")
    for i in range(min(10, len(df))):
        text_preview = df.iloc[i]['text'][:80]
        print(f"  {i}: {text_preview}...")

    # Validation
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")
    print(f"✓ Number of predictions: {len(df)} (expected: 1450)")
    print(f"✓ ID range: {df['id'].min()} to {df['id'].max()}")

    empty_count = (df['text'] == '').sum()
    print(f"✓ Empty predictions: {empty_count}")

    print(f"\n{'='*60}")
    print("SUBMISSION READY!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
