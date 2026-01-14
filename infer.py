"""
Inference script for Vietnamese Soprano TTS.

Usage:
    python infer.py --model-path ./checkpoints/final --text "Xin chào"
    python infer.py --model-path ./checkpoints/final --text "Xin chào" --output output.wav
"""
import argparse
import soundfile as sf
from soprano import SopranoTTS


def main():
    parser = argparse.ArgumentParser(description="Soprano Vietnamese TTS Inference")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", "-o", default="output.wav", help="Output wav file")
    parser.add_argument("--device", "-d", default="auto", help="Device: auto, cuda, cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    model = SopranoTTS(
        model_path=args.model_path,
        device=args.device,
        backend='transformers'  # Use transformers for custom models
    )

    print(f"Synthesizing: {args.text}")
    audio = model.infer(args.text, args.output)
    
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
