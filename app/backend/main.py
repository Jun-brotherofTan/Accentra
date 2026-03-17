"""
main.py — FastAPI backend for the Accentra application.

Endpoints
---------
POST /transcribe       — Convert uploaded audio to text using OpenAI Whisper
POST /detect-accent    — Detect accent using the trained accent classifier
POST /translate        — Translate text to a target language
POST /analyze          — Combined: transcribe + detect accent + translate

Requires
--------
- accent-model/model.pt to be present (produced by accent-model/train.py)
- Python packages listed in requirements.txt
"""

import io
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import torch
import whisper
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Allow importing from the accent-model directory
ACCENT_MODEL_DIR = Path(__file__).parent.parent.parent / "accent-model"
sys.path.insert(0, str(ACCENT_MODEL_DIR))

try:
    from inference import predict_accent
    ACCENT_MODEL_AVAILABLE = True
except ImportError:
    ACCENT_MODEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Accentra API",
    description="Accent detection, transcription, and translation API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy-loaded globals (loaded once on first request)
# ---------------------------------------------------------------------------

_whisper_model = None
_accent_model_path: Optional[str] = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_size = os.environ.get("WHISPER_MODEL", "base")
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def get_accent_model_path() -> str:
    global _accent_model_path
    if _accent_model_path is None:
        env_path = os.environ.get("ACCENT_MODEL_PATH")
        if env_path and Path(env_path).exists():
            _accent_model_path = env_path
        else:
            default = ACCENT_MODEL_DIR / "model.pt"
            if default.exists():
                _accent_model_path = str(default)
            else:
                raise FileNotFoundError(
                    f"Accent model not found at {default}. "
                    "Train the model first: cd accent-model && python train.py"
                )
    return _accent_model_path


# ---------------------------------------------------------------------------
# Translation helper (uses Helsinki-NLP MarianMT via transformers)
# ---------------------------------------------------------------------------

_translation_cache: dict = {}


def translate_text(text: str, target_language: str) -> str:
    """
    Translate *text* to *target_language* using Helsinki-NLP MarianMT models.

    Supported target language codes (ISO 639-1): fr, es, de, zh, ko, ja, ar, hi, pt, ru
    Falls back to returning the original text if the language/model is unavailable.
    """
    if not text.strip():
        return text

    lang_map = {
        "french": "fr", "fr": "fr",
        "spanish": "es", "es": "es",
        "german": "de", "de": "de",
        "chinese": "zh", "zh": "zh",
        "korean": "ko", "ko": "ko",
        "japanese": "ja", "ja": "ja",
        "arabic": "ar", "ar": "ar",
        "hindi": "hi", "hi": "hi",
        "portuguese": "pt", "pt": "pt",
        "russian": "ru", "ru": "ru",
    }

    code = lang_map.get(target_language.lower())
    if code is None:
        return text  # unsupported language — return original

    model_name = f"Helsinki-NLP/opus-mt-en-{code}"

    if model_name not in _translation_cache:
        try:
            from transformers import MarianMTModel, MarianTokenizer

            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            _translation_cache[model_name] = (tokenizer, model)
        except Exception:
            # Model not available — return original text
            return text

    tokenizer, model = _translation_cache[model_name]
    encoded = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        translated = model.generate(**encoded)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TranslateRequest(BaseModel):
    text: str
    target_language: str = "spanish"


class TranslateResponse(BaseModel):
    original: str
    translation: str
    target_language: str


class AnalyzeResponse(BaseModel):
    accent: str
    confidence: float
    transcript: str
    translation: str
    probabilities: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
MAX_AUDIO_SIZE_MB = 50


def validate_audio_file(file: UploadFile) -> None:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Allowed: {sorted(ALLOWED_AUDIO_EXTENSIONS)}",
        )


async def save_upload_to_tempfile(file: UploadFile) -> str:
    """Save an uploaded file to a temp path and return the path."""
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_AUDIO_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_AUDIO_SIZE_MB} MB).",
        )
    ext = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(content)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health-check endpoint."""
    return {"status": "ok", "accent_model_available": ACCENT_MODEL_AVAILABLE}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Convert uploaded audio to text using OpenAI Whisper.

    Returns
    -------
    JSON: { "transcript": "...", "language": "en" }
    """
    validate_audio_file(file)
    tmp_path = await save_upload_to_tempfile(file)

    try:
        whisper_model = get_whisper_model()
        result = whisper_model.transcribe(tmp_path)
        transcript = result.get("text", "").strip()
        if not transcript:
            raise HTTPException(status_code=422, detail="No speech detected in the audio.")
        return {"transcript": transcript, "language": result.get("language", "unknown")}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/detect-accent")
async def detect_accent(file: UploadFile = File(...)):
    """
    Detect speaker accent from audio using the trained accent classifier.

    Returns
    -------
    JSON: { "accent": "Korean", "confidence": 0.82, "probabilities": {...} }
    """
    if not ACCENT_MODEL_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Accent model module not available. Check that accent-model/ is in the Python path.",
        )

    validate_audio_file(file)
    tmp_path = await save_upload_to_tempfile(file)

    try:
        model_path = get_accent_model_path()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        result = predict_accent(tmp_path, model_path, device=device)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Accent detection failed: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest):
    """
    Translate text to a target language.

    Body: { "text": "...", "target_language": "spanish" }
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    translation = translate_text(request.text, request.target_language)
    return TranslateResponse(
        original=request.text,
        translation=translation,
        target_language=request.target_language,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    target_language: str = Form(default="spanish"),
):
    """
    Full analysis pipeline: transcribe + detect accent + translate.

    Returns
    -------
    JSON: {
      "accent": "Korean",
      "confidence": 0.82,
      "transcript": "Hello, my name is ...",
      "translation": "Hola, me llamo ...",
      "probabilities": {"korean": 0.82, ...}
    }
    """
    if not ACCENT_MODEL_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Accent model module not available.",
        )

    validate_audio_file(file)
    tmp_path = await save_upload_to_tempfile(file)

    try:
        # 1. Transcribe
        whisper_model = get_whisper_model()
        transcription = whisper_model.transcribe(tmp_path)
        transcript = transcription.get("text", "").strip()
        if not transcript:
            raise HTTPException(status_code=422, detail="No speech detected in the audio.")

        # 2. Detect accent
        model_path = get_accent_model_path()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        accent_result = predict_accent(tmp_path, model_path, device=device)

        # 3. Translate
        translation = translate_text(transcript, target_language)

        return AnalyzeResponse(
            accent=accent_result["accent"],
            confidence=accent_result["confidence"],
            transcript=transcript,
            translation=translation,
            probabilities=accent_result["probabilities"],
        )
    except (FileNotFoundError, HTTPException):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)
