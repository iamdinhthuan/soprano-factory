"""
Generate audio tokens from LJSpeech-style dataset for Soprano training.

Supports both metadata.txt and metadata.csv formats.

Usage:
    python generate_dataset.py --input-dir path/to/dataset
"""
import argparse
import pathlib
import random
import json

import torchaudio
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from encoder.codec import Encoder


SAMPLE_RATE = 32000
SEED = 42
VAL_PROP = 0.05
VAL_MAX = 512


def get_args():
    parser = argparse.ArgumentParser(description="Generate audio tokens for Soprano training")
    parser.add_argument("--input-dir", required=True, type=pathlib.Path,
        help="Path to dataset directory with metadata.txt/csv and wavs/")
    parser.add_argument("--max-samples", type=int, default=None,
        help="Limit number of samples (for testing)")
    return parser.parse_args()


def load_metadata(input_dir):
    """Load metadata from txt or csv file."""
    files = []
    
    for path in [input_dir / "metadata.csv", input_dir / "metadata.txt"]:
        if path.exists():
            print(f"Loading: {path}")
            with open(path, encoding='utf-8') as f:
                lines = f.read().strip().split('\n')
            
            start = 1 if lines[0].startswith('audio_path') else 0
            for line in lines[start:]:
                if '|' in line:
                    audio, text = line.split('|', maxsplit=1)
                    files.append((audio.replace('.wav', ''), text))
            break
    
    if not files:
        raise FileNotFoundError(f"No metadata.csv or metadata.txt in {input_dir}")
    
    return files


def main():
    args = get_args()
    input_dir = args.input_dir

    print("Loading encoder...")
    encoder = Encoder()
    encoder_path = hf_hub_download(repo_id='ekwek/Soprano-Encoder', filename='encoder.pth')
    encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
    encoder.eval()
    print("Encoder ready.")

    files = load_metadata(input_dir)
    print(f"Found {len(files)} samples")
    
    if args.max_samples:
        files = files[:args.max_samples]
        print(f"Limited to {len(files)} samples")

    print("Encoding audio...")
    dataset = []
    skipped = 0
    
    for filename, transcript in tqdm(files):
        wav_path = input_dir / "wavs" / f"{filename}.wav"
        if not wav_path.exists():
            wav_path = input_dir / "wavs" / filename
            if not wav_path.exists():
                skipped += 1
                continue
        
        try:
            audio, sr = torchaudio.load(str(wav_path))
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
            
            with torch.no_grad():
                tokens = encoder(audio.cpu())
            dataset.append([transcript, tokens.squeeze(0).tolist()])
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"Error: {filename} - {e}")

    if skipped:
        print(f"Skipped {skipped} files")

    print("Creating train/val splits...")
    random.seed(SEED)
    random.shuffle(dataset)
    num_val = min(int(VAL_PROP * len(dataset)) + 1, VAL_MAX)
    train_data = dataset[num_val:]
    val_data = dataset[:num_val]

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    with open(input_dir / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open(input_dir / 'val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False)
    
    print("Done!")


if __name__ == '__main__':
    main()
