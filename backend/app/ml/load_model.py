from __future__ import annotations

import logging
from pathlib import Path

import torch

from app.core.config import settings
from app.ml.model import FCOSDocumentDetector

logger = logging.getLogger(__name__)

_model: FCOSDocumentDetector | None = None


def load_model() -> FCOSDocumentDetector:
    """Load (or return cached) FCOS model."""
    global _model
    if _model is not None:
        return _model

    device = torch.device(settings.DEVICE)
    model_path = Path(settings.MODEL_PATH)

    model = FCOSDocumentDetector(num_classes=FCOSDocumentDetector.NUM_CLASSES)

    if model_path.exists():
        try:
            checkpoint = torch.load(
                str(model_path), map_location=device, weights_only=False
            )
            # checkpoint is a plain state-dict (confirmed by inspection)
            missing, unexpected = model.load_state_dict(checkpoint, strict=True)
            if missing:
                logger.warning("Missing keys (%d): %s", len(missing), missing[:5])
            if unexpected:
                logger.warning("Unexpected keys (%d): %s", len(unexpected), unexpected[:5])
            logger.info("Loaded weights from %s", model_path)
        except Exception as exc:
            logger.error("Failed to load weights: %s. Using random weights.", exc)
    else:
        logger.warning("Model file not found at %s. Using random weights.", model_path)

    model.to(device).eval()
    _model = model
    return _model
