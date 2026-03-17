"""
model.py — Accent classifier model definition.

Architecture:
  - Feature extractor: facebook/wav2vec2-base (frozen by default)
  - Classification head: Linear → ReLU → Dropout → Linear → Softmax

The model can also fall back to raw MFCC features when wav2vec2 is not
available (e.g., during CI / demo runs).
"""

import torch
import torch.nn as nn
from typing import Optional

# Canonical set of accent classes supported by this model.
ACCENT_CLASSES = ["korean", "indian", "spanish", "chinese"]
NUM_CLASSES = len(ACCENT_CLASSES)


class ClassificationHead(nn.Module):
    """Feed-forward classification head placed on top of the feature extractor."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AccentClassifier(nn.Module):
    """
    Accent classifier that wraps a pre-trained wav2vec2 encoder and a
    trainable classification head.

    Parameters
    ----------
    num_classes : int
        Number of accent classes to predict.
    freeze_encoder : bool
        When True (default), the wav2vec2 weights are frozen so only the
        classification head is trained (faster & less memory-intensive).
    use_wav2vec : bool
        When False the encoder is replaced by an identity-like projection
        that expects pre-computed MFCC features of shape (batch, mfcc_dim).
    mfcc_dim : int
        Expected MFCC feature dimension used when use_wav2vec=False.
    """

    WAV2VEC_HIDDEN = 768  # hidden size of wav2vec2-base

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        freeze_encoder: bool = True,
        use_wav2vec: bool = True,
        mfcc_dim: int = 40,
    ):
        super().__init__()
        self.use_wav2vec = use_wav2vec
        self.num_classes = num_classes

        if use_wav2vec:
            try:
                from transformers import Wav2Vec2Model

                self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
                if freeze_encoder:
                    for param in self.encoder.parameters():
                        param.requires_grad = False
                feature_dim = self.WAV2VEC_HIDDEN
            except Exception as exc:
                raise RuntimeError(
                    "Could not load wav2vec2-base. Install transformers and "
                    "ensure internet access, or set use_wav2vec=False to use "
                    f"MFCC features instead.\nOriginal error: {exc}"
                ) from exc
        else:
            # Lightweight projection when using MFCC features
            self.encoder = nn.Linear(mfcc_dim, 128)
            feature_dim = 128

        self.classifier = ClassificationHead(feature_dim, num_classes)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_values : Tensor
            - wav2vec mode : raw waveform (batch, time)
            - MFCC mode    : pre-computed features (batch, mfcc_dim)
        attention_mask : Tensor or None
            Only used in wav2vec mode.

        Returns
        -------
        Tensor  shape (batch, num_classes) — raw logits
        """
        if self.use_wav2vec:
            outputs = self.encoder(
                input_values, attention_mask=attention_mask
            )
            # Mean-pool over the time dimension
            hidden = outputs.last_hidden_state.mean(dim=1)  # (batch, 768)
        else:
            hidden = torch.relu(self.encoder(input_values))  # (batch, 128)

        return self.classifier(hidden)


def load_model(
    checkpoint_path: str,
    use_wav2vec: bool = True,
    mfcc_dim: int = 40,
    device: Optional[str] = None,
) -> AccentClassifier:
    """
    Load a saved AccentClassifier from disk.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pt file saved by train.py.
    use_wav2vec : bool
        Must match the value used during training.
    mfcc_dim : int
        Only relevant when use_wav2vec=False.
    device : str or None
        Target device, e.g. 'cpu' or 'cuda'. Auto-detected when None.

    Returns
    -------
    AccentClassifier in eval mode.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AccentClassifier(
        num_classes=NUM_CLASSES,
        freeze_encoder=True,
        use_wav2vec=use_wav2vec,
        mfcc_dim=mfcc_dim,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
