"""
preprocess.py — Audio preprocessing for the Accentra accent classifier.

Key steps
---------
1. Load audio file (any format supported by librosa / soundfile)
2. Resample to 16 000 Hz mono
3. Normalise amplitude
4. Either return raw waveform (for wav2vec2) or compute MFCC features

Usage
-----
    python preprocess.py --audio path/to/file.wav --mode mfcc
    python preprocess.py --audio path/to/file.wav --mode wav2vec

For dataset preparation (Speech Accent Archive layout):
    python preprocess.py --prepare-dataset \
        --data-dir accent-model/data/speech_accent_archive \
        --output-dir accent-model/data/processed \
        --metadata-csv accent-model/data/speech_accent_archive/speakers_all.csv
"""

import argparse
import os
import csv
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000          # wav2vec2 expects 16 kHz
MAX_DURATION_SEC = 15         # truncate samples longer than this
MFCC_N_MELS = 40              # number of MFCC coefficients
MIN_DURATION_SEC = 0.5        # discard clips shorter than this

# Map native_language values (Speech Accent Archive) → accent class
LANGUAGE_TO_ACCENT = {
    "korean":     "korean",
    "hindi":      "indian",
    "telugu":     "indian",
    "tamil":      "indian",
    "marathi":    "indian",
    "gujarati":   "indian",
    "punjabi":    "indian",
    "bengali":    "indian",
    "spanish":    "spanish",
    "mandarin":   "chinese",
    "cantonese":  "chinese",
    # Note: Japanese is a distinct accent and is intentionally excluded from
    # the current four-class model. Add a "japanese" class and extend
    # ACCENT_CLASSES in model.py if you need Japanese accent support.
}

ACCENT_CLASSES = ["korean", "indian", "spanish", "chinese"]


# ---------------------------------------------------------------------------
# Core audio utilities
# ---------------------------------------------------------------------------

def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load audio from *path* and return (waveform, sample_rate)."""
    try:
        import librosa
        waveform, sr = librosa.load(path, sr=None, mono=True)
    except Exception:
        import soundfile as sf
        waveform, sr = sf.read(path, always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
    return waveform.astype(np.float32), sr


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample *waveform* from *orig_sr* to *target_sr*."""
    if orig_sr == target_sr:
        return waveform
    import librosa
    return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)


def normalise(waveform: np.ndarray) -> np.ndarray:
    """Peak-normalise waveform to [-1, 1]."""
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak
    return waveform


def trim_or_pad(waveform: np.ndarray, sr: int, max_sec: float = MAX_DURATION_SEC) -> np.ndarray:
    """Truncate audio to *max_sec* seconds."""
    max_samples = int(max_sec * sr)
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
    return waveform


def preprocess_waveform(path: str) -> np.ndarray:
    """
    Full preprocessing pipeline: load → resample → normalise → trim.

    Returns
    -------
    np.ndarray shape (T,) at 16 kHz — ready to pass to wav2vec2 or MFCC.
    """
    waveform, sr = load_audio(path)
    waveform = resample(waveform, sr, SAMPLE_RATE)
    waveform = normalise(waveform)
    waveform = trim_or_pad(waveform, SAMPLE_RATE)
    return waveform


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_mfcc(waveform: np.ndarray, n_mfcc: int = MFCC_N_MELS) -> np.ndarray:
    """
    Compute mean MFCC features from a 16 kHz waveform.

    Returns
    -------
    np.ndarray shape (n_mfcc,) — averaged over time frames.
    """
    import librosa
    mfcc = librosa.feature.mfcc(
        y=waveform, sr=SAMPLE_RATE, n_mfcc=n_mfcc
    )  # (n_mfcc, T)
    return mfcc.mean(axis=1).astype(np.float32)


def extract_wav2vec_features(waveform: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    Extract wav2vec2 embeddings from a 16 kHz waveform.

    Returns
    -------
    np.ndarray shape (768,) — mean-pooled hidden states.
    """
    import torch
    from transformers import Wav2Vec2Processor, Wav2Vec2Model

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    model.eval()

    inputs = processor(
        waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    return hidden.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def map_language_to_accent(native_language: str) -> Optional[str]:
    """Return accent class string or None if language is not supported."""
    lang = native_language.strip().lower()
    return LANGUAGE_TO_ACCENT.get(lang)


def prepare_speech_accent_archive(
    data_dir: str,
    output_dir: str,
    metadata_csv: str,
    mode: str = "mfcc",
) -> None:
    """
    Preprocess the Speech Accent Archive dataset.

    Parameters
    ----------
    data_dir : str
        Root directory containing an ``recordings/`` sub-folder.
    output_dir : str
        Where to write processed .npy feature files and a manifest.json.
    metadata_csv : str
        Path to speakers_all.csv from the dataset.
    mode : str
        'mfcc' or 'wav2vec'.
    """
    recordings_dir = Path(data_dir) / "recordings"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = []
    skipped = 0

    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        native_lang = row.get("native_language", "")
        accent_label = map_language_to_accent(native_lang)
        if accent_label is None:
            skipped += 1
            continue

        # The filename pattern varies; try common extensions
        filename_base = row.get("filename", "").strip()
        audio_path = None
        for ext in (".mp3", ".wav", ".ogg"):
            candidate = recordings_dir / (filename_base + ext)
            if candidate.exists():
                audio_path = str(candidate)
                break
        if audio_path is None:
            skipped += 1
            continue

        try:
            waveform = preprocess_waveform(audio_path)
            if len(waveform) < int(MIN_DURATION_SEC * SAMPLE_RATE):
                skipped += 1
                continue

            if mode == "mfcc":
                features = extract_mfcc(waveform)
            else:
                features = extract_wav2vec_features(waveform)

            feat_file = out_path / f"{filename_base}.npy"
            np.save(str(feat_file), features)

            manifest.append({
                "filename": filename_base,
                "feature_file": str(feat_file),
                "accent": accent_label,
                "label_idx": ACCENT_CLASSES.index(accent_label),
            })
        except Exception as exc:
            print(f"[WARN] Skipping {audio_path}: {exc}")
            skipped += 1

    manifest_path = out_path / "manifest.json"
    with open(str(manifest_path), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Processed {len(manifest)} samples, skipped {skipped}.")
    print(f"Manifest saved to {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Accentra audio preprocessing")
    subparsers = parser.add_subparsers(dest="command")

    # Single-file mode
    single = subparsers.add_parser("preprocess", help="Preprocess a single audio file")
    single.add_argument("--audio", required=True, help="Path to audio file")
    single.add_argument(
        "--mode",
        choices=["mfcc", "wav2vec"],
        default="mfcc",
        help="Feature type to extract",
    )
    single.add_argument("--output", default=None, help="Output .npy path (optional)")

    # Dataset mode
    ds = subparsers.add_parser("prepare-dataset", help="Prepare full dataset")
    ds.add_argument("--data-dir", required=True)
    ds.add_argument("--output-dir", required=True)
    ds.add_argument("--metadata-csv", required=True)
    ds.add_argument("--mode", choices=["mfcc", "wav2vec"], default="mfcc")

    args = parser.parse_args()

    if args.command == "preprocess":
        waveform = preprocess_waveform(args.audio)
        if args.mode == "mfcc":
            features = extract_mfcc(waveform)
        else:
            features = extract_wav2vec_features(waveform)
        print(f"Feature shape: {features.shape}")
        if args.output:
            np.save(args.output, features)
            print(f"Saved to {args.output}")

    elif args.command == "prepare-dataset":
        prepare_speech_accent_archive(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            metadata_csv=args.metadata_csv,
            mode=args.mode,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
