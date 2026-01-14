<div align="center">
  
# Soprano-Factory

**Train your own 2000x realtime text-to-speech model**

[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/ekwek/Soprano-80M)
[![Github Repo](https://img.shields.io/badge/Github-Soprano-black?logo=github)](https://github.com/ekwek1/soprano)
[![HuggingFace Demo](https://img.shields.io/badge/HuggingFace-Demo-yellow?logo=huggingface)](https://huggingface.co/spaces/ekwek/Soprano-TTS)

</div>

## Overview

Soprano-Factory allows you to train/fine-tune Soprano TTS models with your own data. Supports **English** and **Vietnamese** (with tokenizer expansion).

**Features:**
- Train by epochs with proper evaluation
- Save top K best checkpoints automatically  
- Vietnamese language support via tokenizer expansion
- Works on CUDA, CPU, MPS

---

## Installation

```bash
git clone https://github.com/ekwek1/soprano-factory.git
cd soprano-factory
pip install -r requirements.txt
```

> **Windows CUDA**: Reinstall PyTorch with CUDA support after installation.

---

## Quick Start (English)

```bash
# 1. Prepare dataset (LJSpeech format)
python generate_dataset.py --input-dir ./my_dataset

# 2. Train
python train.py --input-dir ./my_dataset --save-dir ./checkpoints --epochs 10
```

---

## Vietnamese Training

### Step 1: Expand Tokenizer

Vietnamese requires expanding the tokenizer vocabulary:

```bash
python expand_tokenizer.py --input-dir ./vietnamese_dataset --output-dir ./vietnamese_model
```

This will:
- Analyze all Vietnamese characters in your dataset
- Add missing characters to tokenizer (ă, â, đ, ê, ô, ơ, ư, diacritics, etc.)
- Resize model embeddings automatically

### Step 2: Generate Audio Tokens

```bash
python generate_dataset.py --input-dir ./vietnamese_dataset
```

### Step 3: Train

```bash
python train.py \
    --input-dir ./vietnamese_dataset \
    --save-dir ./checkpoints \
    --model-path ./vietnamese_model \
    --epochs 10 \
    --batch-size 4
```

---

## Dataset Format

LJSpeech format with `metadata.txt` or `metadata.csv`:

```
your_dataset/
├── metadata.csv    # or metadata.txt
└── wavs/
    ├── audio001.wav
    ├── audio002.wav
    └── ...
```

**metadata.csv:**
```
audio_path|text
audio001.wav|This is the transcript for audio 001.
audio002.wav|Đây là transcript cho audio 002.
```

**metadata.txt:**
```
audio001|This is the transcript for audio 001.
audio002|Đây là transcript cho audio 002.
```

---

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | required | Dataset directory with train.json/val.json |
| `--save-dir` | required | Checkpoint output directory |
| `--model-path` | `ekwek/Soprano-80M` | Base model or expanded model path |
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 4 | Batch size |
| `--learning-rate` | 5e-4 | Maximum learning rate |
| `--eval-steps` | 500 | Evaluate every N steps |
| `--save-steps` | 500 | Save checkpoint every N steps |
| `--top-k` | 3 | Keep top K best checkpoints |
| `--device` | `cuda:0` | Training device |

---

## Output Structure

```
checkpoints/
├── checkpoint-step1000-epoch1-loss4.5/   # Top 3 best
├── checkpoint-step1500-epoch2-loss4.2/
├── checkpoint-step2000-epoch2-loss4.0/
├── checkpoints.json                       # Tracking file
└── final/                                 # Final model
```

---

## Inference

### CLI
```bash
python infer.py --model-path ./checkpoints/final --text "Xin chào, đây là tiếng Việt!"
```

### Python
```python
from soprano import SopranoTTS

model = SopranoTTS(
    model_path='./checkpoints/final',
    device='auto',
    backend='transformers'  # Required for custom models
)

audio = model.infer("Xin chào, đây là tiếng Việt!", "output.wav")
```

### Colab/Notebook
```python
# Install
!pip install soprano-tts

# Inference
from soprano import SopranoTTS
model = SopranoTTS(model_path='./checkpoints/final', backend='transformers')
audio = model.infer("Xin chào!", "output.wav")

# Play
from IPython.display import Audio
Audio("output.wav")
```

---

## Tips

- **Dataset size**: Recommend 5-10+ hours of audio for good results
- **Audio quality**: Clean audio, 32kHz sample rate preferred
- **Training time**: ~10k samples takes 4-5 hours on RTX 3090
- **Memory**: ~8GB VRAM with batch_size=4

---

## License

Apache-2.0. See `LICENSE` for details.
