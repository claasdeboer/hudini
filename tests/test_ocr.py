"""Unit tests for hudini.ocr.

PaddleOCR is patched at the ocr namespace; only the decode logic and the
OCRResult contract run for real.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hudini.ocr import ModelSize, OCREngine, OCRResult, ensure_models


class TestModelSize:
    @pytest.mark.parametrize("size", list(ModelSize))
    def test_detection_and_recognition_names_are_a_matched_pair(self, size):
        assert size.detection_model == f"PP-OCRv6_{size.value}_det"
        assert size.recognition_model == f"PP-OCRv6_{size.value}_rec"

    def test_an_unknown_size_raises(self):
        with pytest.raises(ValueError):
            ModelSize("huge")


class TestOCRResult:
    def test_from_lines_joins_and_lowercases(self):
        read = OCRResult.from_lines([("Force", 0.9), ("BIPOLAR", 0.7)])
        assert read == OCRResult(text="force bipolar", score=pytest.approx(0.8))

    def test_blank_lines_are_dropped(self):
        read = OCRResult.from_lines([("", 0.1), ("grip", 0.8), ("   ", 0.9)])
        assert read == OCRResult(text="grip", score=0.8)

    @pytest.mark.parametrize("lines", [[], [("  ", 0.9)]])
    def test_nothing_readable_gives_empty_text_and_zero_score(self, lines):
        assert OCRResult.from_lines(lines) == OCRResult(text="", score=0.0)


@pytest.fixture
def paddle() -> MagicMock:
    engine = MagicMock()
    engine.predict.return_value = [{"rec_texts": ["Table", "Motion"], "rec_scores": [0.9, 0.7]}]
    engine.paddlex_pipeline.text_rec_model.return_value = [{"rec_text": "CUT", "rec_score": 0.7}]
    return engine


class TestOCREngine:
    def test_construction_loads_the_matched_pair(self, paddle):
        with patch("hudini.ocr.PaddleOCR", return_value=paddle) as constructor:
            OCREngine(model_size=ModelSize.SMALL, device="cpu")
        kwargs = constructor.call_args.kwargs
        assert kwargs["text_detection_model_name"] == "PP-OCRv6_small_det"
        assert kwargs["text_recognition_model_name"] == "PP-OCRv6_small_rec"
        assert kwargs["engine"] == "transformers"

    def test_read_detects_each_crop_at_its_own_size(self, paddle):
        with patch("hudini.ocr.PaddleOCR", return_value=paddle):
            model = OCREngine()
        (read,) = model.read([np.zeros((30, 200, 3), dtype=np.uint8)])
        assert paddle.predict.call_args.kwargs["text_det_limit_side_len"] == 200
        assert read == OCRResult(text="table motion", score=pytest.approx(0.8))

    def test_read_raises_on_misaligned_rec_lists(self, paddle):
        paddle.predict.return_value = [{"rec_texts": ["Table"], "rec_scores": []}]
        with patch("hudini.ocr.PaddleOCR", return_value=paddle):
            model = OCREngine()
        with pytest.raises(ValueError):
            model.read([np.zeros((30, 200, 3), dtype=np.uint8)])

    def test_read_line_skips_detection(self, paddle):
        with patch("hudini.ocr.PaddleOCR", return_value=paddle):
            model = OCREngine()
        (read,) = model.read_line([np.zeros((17, 48, 3), dtype=np.uint8)])
        assert read == OCRResult(text="cut", score=0.7)
        paddle.predict.assert_not_called()


def test_ensure_models_resolves_the_pair_of_one_size():
    store = {"PP-OCRv6_small_det": "/models/det", "PP-OCRv6_small_rec": "/models/rec"}
    with patch("hudini.ocr.official_models", store):
        paths = ensure_models("small")
    assert paths == [Path("/models/det"), Path("/models/rec")]
