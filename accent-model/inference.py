"""
inference.py — Accent classification inference for the Accentra project.

Usage
-----
    python inference.py --audio path/to/audio.wav --model accent-model/model.pt

Output (stdout, JSON):
    {
      "accent": "Korean",
      "confidence": 0.82,
      "probabilities": {"korean": 0.82, "indian": 0.09, "spanish": 0.06, "chinese": 0.03}
    }
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Make sure parent package is importable when called from anywhere
sys.path.insert(0, str(Path(__file__).parent))

from model import AccentClassifier, ACCENT_CLASSES, load_model
from preprocess import preprocess_waveform, extract_mfcc


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def predict_accent(
    audio_path: str,
    model_path: str,
    device: str = "cpu",
) -> Dict:
    """
    Predict the accent of a speaker from an audio file.

    Parameters
    ----------
    audio_path : str
        Path to a .wav / .mp3 / .ogg audio file.
    model_path : str
        Path to the trained model checkpoint (model.pt).
    device : str
        Torch device string ('cpu' or 'cuda').

    Returns
    -------
    dict with keys:
        accent       : str  — top predicted accent class
        confidence   : float — probability of the top class (0-1)
        probabilities: dict  — per-class probabilities
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Train the model first with train.py."
        )

    # Load model and determine if it was trained with wav2vec or MFCC
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    use_wav2vec = config.get("use_wav2vec", False)
    mfcc_dim = config.get("mfcc_dim", 40)

    model = AccentClassifier(
        num_classes=len(ACCENT_CLASSES),
        freeze_encoder=True,
        use_wav2vec=use_wav2vec,
        mfcc_dim=mfcc_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Preprocess audio
    waveform = preprocess_waveform(audio_path)

    if use_wav2vec:
        # Pass raw waveform directly — model handles wav2vec encoding
        input_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
    else:
        # Extract MFCC features first
        features = extract_mfcc(waveform, n_mfcc=mfcc_dim)
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)

    probabilities = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    top_idx = int(np.argmax(probabilities))
    top_label = ACCENT_CLASSES[top_idx]
    confidence = float(probabilities[top_idx])

    prob_dict = {cls: float(probabilities[i]) for i, cls in enumerate(ACCENT_CLASSES)}

    return {
        "accent": top_label.capitalize(),
        "confidence": round(confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect accent from audio using the trained Accentra model"
    )
    parser.add_argument("--audio", required=True, help="Path to audio file (.wav/.mp3/.ogg)")
    parser.add_argument(
        "--model",
        default=os.path.join(os.path.dirname(__file__), "model.pt"),
        help="Path to trained model checkpoint (default: accent-model/model.pt)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device: 'cpu' or 'cuda' (auto-detect when not specified)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        result = predict_accent(args.audio, args.model, device=device)
        print(json.dumps(result, indent=2))
    except FileNotFoundError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(json.dumps({"error": f"Inference failed: {exc}"}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
