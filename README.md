# 🎙️ Accentra

**AI-powered accent detection, transcription, and translation.**

Accentra records or accepts uploaded audio, identifies the speaker's likely accent
(Korean, Indian, Spanish, or Chinese English), transcribes the speech with
[OpenAI Whisper](https://github.com/openai/whisper), and translates the transcript
into any of ten languages using Helsinki-NLP MarianMT models.

---

## Project Structure

```
Accentra/
├── accent-model/            # ML training pipeline
│   ├── data/
│   │   └── README.md        # Dataset download instructions
│   ├── model.py             # AccentClassifier (wav2vec2 + classification head)
│   ├── preprocess.py        # Audio preprocessing & feature extraction
│   ├── train.py             # Training loop + validation
│   ├── inference.py         # Accent prediction from audio file
│   └── requirements.txt
│
├── app/
│   ├── backend/             # FastAPI server
│   │   ├── main.py          # API endpoints
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── frontend/            # React + Tailwind UI
│       ├── src/
│       │   ├── App.jsx
│       │   └── components/
│       │       ├── AudioUploader.jsx
│       │       └── Results.jsx
│       ├── index.html
│       ├── package.json
│       ├── tailwind.config.js
│       ├── vite.config.js
│       ├── nginx.conf
│       └── Dockerfile
│
├── docker-compose.yml
└── README.md
```

---

## Architecture

```
Browser
  │  upload / record audio
  ▼
React + Tailwind (port 3000)
  │  POST /api/analyze  (FormData: file + target_language)
  ▼
FastAPI backend (port 8000)
  ├─ Whisper  → transcript
  ├─ AccentClassifier (wav2vec2 + MLP) → accent + confidence
  └─ MarianMT → translation
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | ≥ 3.10  |
| Node.js     | ≥ 18    |
| FFmpeg      | any recent |
| Docker + Compose | optional |

---

## 1 — Train the Accent Model

### 1a. Install ML dependencies

```bash
cd accent-model
pip install -r requirements.txt
```

### 1b. Obtain a dataset

See [`accent-model/data/README.md`](accent-model/data/README.md) for full
instructions. In short, download the
[Speech Accent Archive from Kaggle](https://www.kaggle.com/datasets/rtatman/speech-accent-archive)
and unzip it into `accent-model/data/speech_accent_archive/`.

### 1c. Preprocess the dataset

```bash
python preprocess.py prepare-dataset \
  --data-dir  data/speech_accent_archive \
  --output-dir data/processed \
  --metadata-csv data/speech_accent_archive/speakers_all.csv \
  --mode mfcc
```

### 1d. Train

```bash
python train.py \
  --manifest data/processed/manifest.json \
  --epochs 30 \
  --batch-size 32 \
  --output model.pt
```

#### Demo / CI mode (no dataset required)

```bash
DEMO_MODE=1 python train.py --epochs 10 --output model.pt
```

> ⚠️  A model trained in demo mode uses **synthetic data** and will not produce
> meaningful accent predictions. Use it only for integration testing.

---

## 2 — Run the Backend

```bash
cd app/backend
pip install -r requirements.txt

# Ensure the trained model is at accent-model/model.pt (default path)
uvicorn main:app --reload --port 8000
```

The API is now available at `http://localhost:8000`.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `ACCENT_MODEL_PATH` | `../../accent-model/model.pt` | Path to trained `.pt` checkpoint |

---

## 3 — Run the Frontend

```bash
cd app/frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

---

## 4 — Docker (full stack)

```bash
# First train the model (once)
cd accent-model
DEMO_MODE=1 python train.py --output model.pt
cd ..

# Build and start
docker-compose up --build
```

- Frontend: `http://localhost:3000`
- Backend API docs: `http://localhost:8000/docs`

---

## API Reference

### `POST /transcribe`

Convert audio to text.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file`    | File | Audio file (.wav, .mp3, .ogg, .flac, .m4a, .webm) |

**Response:**
```json
{ "transcript": "Hello, how are you?", "language": "en" }
```

---

### `POST /detect-accent`

Detect accent from audio.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file`    | File | Audio file |

**Response:**
```json
{
  "accent": "Korean",
  "confidence": 0.82,
  "probabilities": {
    "korean":  0.82,
    "indian":  0.09,
    "spanish": 0.06,
    "chinese": 0.03
  }
}
```

---

### `POST /translate`

Translate text.

**Body (JSON):**
```json
{ "text": "Hello world", "target_language": "spanish" }
```

**Response:**
```json
{ "original": "Hello world", "translation": "Hola mundo", "target_language": "spanish" }
```

---

### `POST /analyze`

Full pipeline: transcribe + detect accent + translate.

| Parameter         | Type   | Description |
|-------------------|--------|-------------|
| `file`            | File   | Audio file  |
| `target_language` | string | e.g. `spanish`, `french`, `korean` |

**Response:**
```json
{
  "accent":       "Korean",
  "confidence":   0.82,
  "transcript":   "Hello, my name is ...",
  "translation":  "Hola, me llamo ...",
  "probabilities": { "korean": 0.82, "indian": 0.09, ... }
}
```

---

## Accent Classes

| Class     | Description |
|-----------|-------------|
| `korean`  | Native Korean speakers |
| `indian`  | Native speakers from India (Hindi, Tamil, Telugu, …) |
| `spanish` | Native Spanish speakers |
| `chinese` | Native Mandarin / Cantonese speakers |

---

## Model Details

| Component | Description |
|-----------|-------------|
| Feature extractor | `facebook/wav2vec2-base` (frozen by default) |
| Classification head | Linear(768→256) → ReLU → Dropout(0.3) → Linear(256→4) |
| Training | Adam, LR 1e-3, StepLR scheduler |
| Input | 16 kHz mono waveform, max 15 s |

Alternatively the model can use **MFCC** features (40 coefficients) instead of
wav2vec2, which is faster and requires less memory at the cost of accuracy.

---

## Supported Translation Languages

Spanish · French · German · Chinese · Korean · Japanese · Portuguese · Russian · Arabic · Hindi

---

## License

MIT
