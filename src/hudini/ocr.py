"""Reads text with PP-OCRv6.

``read`` finds and reads text anywhere in a crop. ``read_line`` reads a
crop that is already one line of text, without detection. Both return
one :class:`OCRResult` per image. The models use the transformers engine
of PaddleOCR, which keeps inference on torch and never imports the
paddle runtime. Loading is quiet unless the ``hudini`` logger is at
debug level.
"""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import cv2
import numpy as np
from paddleocr import PaddleOCR
from paddlex.inference.utils.official_models import official_models
from transformers.utils.logging import disable_progress_bar

logger = logging.getLogger(__name__)

PADDLEX_LOGGER_NAME = "paddlex"


class ModelSize(StrEnum):
    """PP-OCRv6 size. Detection and recognition models come as a matched pair."""

    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"

    @property
    def detection_model(self) -> str:
        return f"PP-OCRv6_{self.value}_det"

    @property
    def recognition_model(self) -> str:
        return f"PP-OCRv6_{self.value}_rec"


@dataclass(frozen=True, slots=True)
class OCRResult:
    """One crop's reading: its text as one lowercase line, and a confidence.

    ``text`` is empty when nothing readable was found; ``score`` is then
    0.0.
    """

    text: str
    score: float

    @classmethod
    def from_lines(cls, lines: Iterable[tuple[str, float]]) -> Self:
        """Join recognised ``(text, confidence)`` lines into one read.

        Blank lines are dropped. The rest join lowercased, so observation
        keys stay stable against OCR case flicker. ``score`` is their
        mean confidence.
        """
        words = [(text.strip(), confidence) for text, confidence in lines if text.strip()]
        if not words:
            return cls(text="", score=0.0)
        return cls(
            text=" ".join(text for text, _confidence in words).lower(),
            score=sum(confidence for _text, confidence in words) / len(words),
        )


def ensure_models(model_size: ModelSize | str) -> list[Path]:
    """Download the PP-OCRv6 pair of one size when it is absent.

    Resolves through the model store the engine itself uses, so a
    later engine construction needs no network. Returns the local
    model paths, detection first.
    """
    size = ModelSize(model_size)
    return [
        Path(official_models[size.detection_model]),
        Path(official_models[size.recognition_model]),
    ]


class OCREngine:
    """PP-OCRv6 detection and recognition.

    Args:
        model_size: PP-OCRv6 size, as a :class:`ModelSize` or its string value.
        device: device string PaddleOCR understands, e.g. "gpu" or "cpu".
    """

    def __init__(
        self,
        model_size: ModelSize | str = ModelSize.TINY,
        *,
        device: str = "gpu",
    ) -> None:
        self.model_size = ModelSize(model_size)
        if not logger.isEnabledFor(logging.DEBUG):
            quiet_paddle()
        logger.info("Loading PP-OCRv6 %s (transformers engine, %s)...", self.model_size, device)

        self.paddle_ocr = PaddleOCR(
            engine="transformers",
            device=device,
            text_detection_model_name=self.model_size.detection_model,
            text_recognition_model_name=self.model_size.recognition_model,
            text_det_limit_type="max",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        self.text_recognition: Callable[[list[np.ndarray]], list[dict]] = (
            self.paddle_ocr.paddlex_pipeline.text_rec_model
        )
        logger.info("PP-OCRv6 loaded")

    def read(self, images: list[np.ndarray]) -> list[OCRResult]:
        """Find and read the text in each RGB crop.

        Each crop is detected at its own size; the detector never
        resamples.
        """
        results = []
        for image in images:
            predictions = self.paddle_ocr.predict(
                to_bgr(image), text_det_limit_side_len=max(image.shape[:2])
            )
            lines = [
                (text, float(score))
                for prediction in predictions
                for text, score in zip(
                    prediction["rec_texts"], prediction["rec_scores"], strict=True
                )
            ]
            results.append(OCRResult.from_lines(lines))
        return results

    def read_line(self, images: list[np.ndarray]) -> list[OCRResult]:
        """Read RGB crops that are each one line of text, skipping detection."""
        recognitions = self.text_recognition([to_bgr(image) for image in images])
        return [
            OCRResult.from_lines([(recognition["rec_text"], float(recognition["rec_score"]))])
            for recognition in recognitions
        ]


def quiet_paddle() -> None:
    """Drop paddlex to warnings and hide the transformers weight-loading bars.

    Must run after paddlex is imported, because its import-time setup
    forces INFO.
    The progress-bar switch is global to the transformers library.
    """
    logging.getLogger(PADDLEX_LOGGER_NAME).setLevel(logging.WARNING)
    disable_progress_bar()


def to_bgr(rgb: np.ndarray) -> np.ndarray:
    """The BGR copy PaddleOCR expects."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
