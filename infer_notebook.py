"""
Inference code for Jupyter/Colab notebooks.
Copy and run this in your notebook.
"""

# === INSTALL (run once) ===
# !pip install soprano-tts

# === INFERENCE ===
from soprano import SopranoTTS

# Load model (change path to your trained model)
model = SopranoTTS(
    model_path='./checkpoints/final',  # or './vietnamese_model' for expanded tokenizer only
    device='auto',
    backend='transformers'  # Required for custom finetuned models
)

# Generate speech
text = "Xin chào, đây là tiếng Việt!"
audio = model.infer(text, "output.wav")

print("Done! Saved to output.wav")

# === PLAY AUDIO IN NOTEBOOK ===
from IPython.display import Audio, display
display(Audio("output.wav"))
