"""
Expand Soprano tokenizer with Vietnamese characters.

This script analyzes Vietnamese text, adds missing characters to tokenizer,
and resizes model embeddings accordingly.

Usage:
    python expand_tokenizer.py --input-dir ./dataset --output-dir ./vietnamese_model
"""
import argparse
import json
import pathlib
import re
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# All Vietnamese characters (lowercase and uppercase)
VIETNAMESE_CHARS = set(
    # Lowercase vowels with diacritics
    "àáảãạăằắẳẵặâầấẩẫậ"  # a variants
    "èéẻẽẹêềếểễệ"        # e variants
    "ìíỉĩị"              # i variants
    "òóỏõọôồốổỗộơờớởỡợ"  # o variants
    "ùúủũụưừứửữự"        # u variants
    "ỳýỷỹỵ"              # y variants
    "đ"                   # đ
    # Uppercase vowels with diacritics
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"  # A variants
    "ÈÉẺẼẸÊỀẾỂỄỆ"        # E variants
    "ÌÍỈĨỊ"              # I variants
    "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"  # O variants
    "ÙÚỦŨỤƯỪỨỬỮỰ"        # U variants
    "ỲÝỶỸỴ"              # Y variants
    "Đ"                   # Đ
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",
        required=True,
        type=pathlib.Path,
        help="Path to Vietnamese dataset directory containing metadata.csv or metadata.txt"
    )
    parser.add_argument("--output-dir",
        required=True,
        type=pathlib.Path,
        help="Path to save expanded tokenizer and model"
    )
    parser.add_argument("--base-model",
        default="ekwek/Soprano-80M",
        type=str,
        help="Base model to expand"
    )
    return parser.parse_args()


def load_texts(input_dir):
    """Load all texts from metadata file."""
    texts = []
    
    # Try metadata.csv first
    csv_path = input_dir / "metadata.csv"
    txt_path = input_dir / "metadata.txt"
    
    if csv_path.exists():
        with open(csv_path, encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            # Skip header if exists
            start_idx = 1 if lines[0].startswith('audio_path') else 0
            for line in lines[start_idx:]:
                if '|' in line:
                    _, text = line.split('|', maxsplit=1)
                    texts.append(text)
    elif txt_path.exists():
        with open(txt_path, encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                if '|' in line:
                    _, text = line.split('|', maxsplit=1)
                    texts.append(text)
    else:
        raise FileNotFoundError(f"No metadata.csv or metadata.txt found in {input_dir}")
    
    return texts


def analyze_characters(texts):
    """Analyze all unique characters in texts."""
    char_counter = Counter()
    for text in texts:
        char_counter.update(text)
    return char_counter


def find_missing_vietnamese_chars(texts, tokenizer):
    """Find Vietnamese characters that are not in the tokenizer vocabulary."""
    # Get all unique characters from texts
    all_chars = set()
    for text in texts:
        all_chars.update(text)
    
    # Test which characters get tokenized to UNK
    missing_chars = set()
    for char in all_chars:
        tokens = tokenizer.encode(char, add_special_tokens=False)
        # Check if any token is UNK (id=0)
        if 0 in tokens or len(tokens) == 0:
            missing_chars.add(char)
    
    # Also add all Vietnamese chars that might be missing
    for char in VIETNAMESE_CHARS:
        tokens = tokenizer.encode(char, add_special_tokens=False)
        if 0 in tokens or len(tokens) == 0:
            missing_chars.add(char)
    
    return missing_chars


def expand_tokenizer(tokenizer, new_chars, output_dir):
    """Expand tokenizer vocabulary with new characters."""
    # Sort characters for consistent ordering
    new_chars_sorted = sorted(new_chars)
    
    # Add new tokens to tokenizer
    num_added = tokenizer.add_tokens(new_chars_sorted)
    print(f"Added {num_added} new tokens to tokenizer")
    print(f"New vocabulary size: {len(tokenizer)}")
    
    # Save expanded tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"Saved expanded tokenizer to {output_dir}")
    
    return tokenizer, new_chars_sorted


def resize_model_embeddings(model, tokenizer, new_tokens, output_dir):
    """Resize model embeddings to accommodate new tokens."""
    old_vocab_size = model.config.vocab_size
    new_vocab_size = len(tokenizer)
    
    print(f"Resizing model embeddings: {old_vocab_size} -> {new_vocab_size}")
    
    # Resize token embeddings
    model.resize_token_embeddings(new_vocab_size)
    
    # Initialize new embeddings with mean of existing embeddings
    with torch.no_grad():
        # Get embedding layer
        if hasattr(model, 'model'):
            embed_tokens = model.model.embed_tokens
        elif hasattr(model, 'transformer'):
            embed_tokens = model.transformer.wte
        else:
            embed_tokens = model.get_input_embeddings()
        
        # Calculate mean of existing embeddings for initialization
        existing_embeddings = embed_tokens.weight[:old_vocab_size]
        mean_embedding = existing_embeddings.mean(dim=0)
        
        # Initialize new token embeddings with mean + small random noise
        for i in range(old_vocab_size, new_vocab_size):
            noise = torch.randn_like(mean_embedding) * 0.02
            embed_tokens.weight[i] = mean_embedding + noise
    
    # Update config
    model.config.vocab_size = new_vocab_size
    
    # Save model
    model.save_pretrained(output_dir)
    print(f"Saved resized model to {output_dir}")
    
    return model


def create_vietnamese_metadata_txt(input_dir):
    """Convert metadata.csv to metadata.txt (LJSpeech format)."""
    csv_path = input_dir / "metadata.csv"
    txt_path = input_dir / "metadata.txt"
    
    if csv_path.exists() and not txt_path.exists():
        with open(csv_path, encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        
        # Skip header and write to txt
        start_idx = 1 if lines[0].startswith('audio_path') else 0
        output_lines = []
        for line in lines[start_idx:]:
            if '|' in line:
                audio_path, text = line.split('|', maxsplit=1)
                # Remove .wav extension for LJSpeech format
                filename = audio_path.replace('.wav', '')
                output_lines.append(f"{filename}|{text}")
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        
        print(f"Created metadata.txt with {len(output_lines)} entries")
    
    return txt_path


def main():
    args = get_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base tokenizer and model
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    
    print(f"Original vocabulary size: {len(tokenizer)}")
    
    # Load texts from dataset
    print(f"Loading texts from: {args.input_dir}")
    texts = load_texts(args.input_dir)
    print(f"Loaded {len(texts)} text samples")
    
    # Analyze characters
    char_counter = analyze_characters(texts)
    print(f"Found {len(char_counter)} unique characters in dataset")
    
    # Find missing Vietnamese characters
    missing_chars = find_missing_vietnamese_chars(texts, tokenizer)
    print(f"Found {len(missing_chars)} characters not in tokenizer:")
    print(f"  {sorted(missing_chars)}")
    
    if missing_chars:
        # Expand tokenizer
        tokenizer, new_tokens = expand_tokenizer(tokenizer, missing_chars, args.output_dir)
        
        # Resize model embeddings
        model = resize_model_embeddings(model, tokenizer, new_tokens, args.output_dir)
        
        # Save token mapping for reference
        token_mapping = {char: tokenizer.convert_tokens_to_ids(char) for char in new_tokens}
        with open(args.output_dir / "vietnamese_token_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(token_mapping, f, ensure_ascii=False, indent=2)
        print(f"Saved token mapping to {args.output_dir / 'vietnamese_token_mapping.json'}")
    else:
        print("No new characters needed! Tokenizer already supports all characters.")
        tokenizer.save_pretrained(args.output_dir)
        model.save_pretrained(args.output_dir)
    
    # Create metadata.txt if needed
    create_vietnamese_metadata_txt(args.input_dir)
    
    print("\n=== Summary ===")
    print(f"Original vocab size: {model.config.vocab_size - len(missing_chars) if missing_chars else model.config.vocab_size}")
    print(f"New vocab size: {model.config.vocab_size}")
    print(f"Added tokens: {len(missing_chars)}")
    print(f"Output directory: {args.output_dir}")
    print("\nNext steps:")
    print("1. Run generate_dataset.py with --input-dir pointing to your Vietnamese dataset")
    print("2. Run train.py with --model-path pointing to the expanded model")


if __name__ == '__main__':
    main()
