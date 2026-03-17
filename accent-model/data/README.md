# Accent Model Data

This directory holds the audio dataset used to train the accent classification model.

## Supported Datasets

### 1. Speech Accent Archive (Recommended)
A curated dataset of speakers reading the same English paragraph with different accents.

**Download steps:**
```bash
# Install Kaggle CLI
pip install kaggle

# Download Speech Accent Archive
kaggle datasets download -d rtatman/speech-accent-archive
unzip speech-accent-archive.zip -d accent-model/data/speech_accent_archive
```

Dataset URL: https://www.kaggle.com/datasets/rtatman/speech-accent-archive

### 2. Mozilla Common Voice
Large crowd-sourced dataset. Filter by English speakers of different native languages.

```bash
# Download via the Common Voice downloader
pip install commonvoice-downloader

# Download English clips
# Visit https://commonvoice.mozilla.org/en/datasets and download the English dataset
# Extract to accent-model/data/common_voice
```

## Expected Directory Layout

```
data/
  speech_accent_archive/
    recordings/          # .mp3 or .wav audio files
    speakers_all.csv     # Metadata (native language, accent label, etc.)
  common_voice/
    clips/               # .mp3 audio clips
    validated.tsv        # Metadata file
```

## Accent Labels Used

The model classifies the following accent groups:
- `korean`   — Native Korean speakers
- `indian`   — Native speakers from India
- `spanish`  — Native Spanish speakers
- `chinese`  — Native Mandarin/Cantonese speakers

When preprocessing, `preprocess.py` maps the `native_language` field from the
Speech Accent Archive metadata to these four classes.

## Placeholder / Demo Mode

If you do not have a dataset, `train.py` can run in **demo mode** using
synthetically generated MFCC feature vectors so you can verify the pipeline
end-to-end before downloading real data. Set the environment variable:

```bash
DEMO_MODE=1 python train.py
```

The saved `model.pt` produced in demo mode will not predict accents accurately
but is useful for integration testing the backend.
