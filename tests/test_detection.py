"""Unit tests for hudini.detection.

The rfdetr model is a fake. Loading guards, metadata reading, and
decoding run for real, on fabricated checkpoint files.
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from hudini.detection import (
    RFDETR_POSITION_EMBEDDINGS_KEY,
    Detection,
    Detector,
    _adopt_library_loggers,
    _checkpoint_metadata,
    load_detector,
)
from hudini.schema import Box

CLASS_NAMES = {0: "offscreen", 1: "active"}


@pytest.fixture
def make_checkpoint(tmp_path) -> object:
    """A factory for rfdetr-shaped checkpoint files with chosen faults."""

    def _make(
        model_name: str | None = "RFDETRNano",
        grid_side: int = 36,
        tokens: int | None = None,
        class_names: tuple[str, ...] = ("offscreen", "active"),
        drop: str | None = None,
    ) -> Path:
        token_count = tokens if tokens is not None else grid_side**2
        payload = {
            "model_name": model_name,
            "model": {RFDETR_POSITION_EMBEDDINGS_KEY: torch.zeros((1, token_count + 1, 4))},
            "args": {"class_names": list(class_names)},
        }
        if model_name is None:
            del payload["model_name"]
        if drop is not None:
            del payload[drop]
        path = tmp_path / "checkpoint.pt"
        torch.save(payload, path)
        return path

    return _make


def raw_detections(rows: list[tuple[float, float, float, float, int, float]]) -> SimpleNamespace:
    return SimpleNamespace(
        xyxy=[row[:4] for row in rows],
        class_id=[row[4] for row in rows],
        confidence=[row[5] for row in rows],
    )


class TestDetector:
    def test_decodes_labels_boxes_and_rounded_scores(self):
        model = MagicMock()
        model.predict.return_value = raw_detections(
            [(10.4, 20.6, 30.0, 50.0, 1, 0.987654), (0.0, 0.0, 5.0, 5.0, 7, 0.5)]
        )
        detector = Detector(model=model, class_names=CLASS_NAMES, threshold=0.5, batched=False)
        ((first, second),) = detector.detect([np.zeros((8, 8, 3), dtype=np.uint8)])
        assert first == Detection(label="active", score=0.9877, box=Box(x=10, y=21, w=20, h=29))
        assert second.label == "7"

    def test_unbatched_predicts_per_image(self):
        model = MagicMock()
        model.predict.return_value = raw_detections([])
        detector = Detector(model=model, class_names=CLASS_NAMES, threshold=0.4, batched=False)
        images = [np.zeros((8, 8, 3), dtype=np.uint8)] * 3
        assert detector.detect(images) == [[], [], []]
        assert model.predict.call_count == 3
        assert model.predict.call_args.kwargs["threshold"] == 0.4

    def test_batched_sends_one_call_for_many_images(self):
        model = MagicMock()
        model.predict.return_value = [raw_detections([]), raw_detections([])]
        detector = Detector(model=model, class_names=CLASS_NAMES, threshold=0.4, batched=True)
        assert detector.detect([np.zeros((8, 8, 3), dtype=np.uint8)] * 2) == [[], []]
        assert model.predict.call_count == 1

    def test_no_images_means_no_model_call(self):
        model = MagicMock()
        detector = Detector(model=model, class_names=CLASS_NAMES, threshold=0.4, batched=False)
        assert detector.detect([]) == []
        model.predict.assert_not_called()

    def test_labels_lists_class_names_in_id_order(self):
        detector = Detector(
            model=MagicMock(),
            class_names={1: "active", 0: "offscreen"},
            threshold=0.5,
            batched=False,
        )
        assert detector.labels == ("offscreen", "active")


class TestCheckpointMetadata:
    def test_reads_model_resolution_and_class_names(self, make_checkpoint):
        path = make_checkpoint(grid_side=36, class_names=("offscreen", "active", "inactive"))
        assert _checkpoint_metadata(path) == (
            "RFDETRNano",
            576,
            {0: "offscreen", 1: "active", 2: "inactive"},
        )

    @pytest.mark.parametrize("model_name", ["ResNet50", None])
    def test_a_missing_or_unknown_model_name_raises(self, make_checkpoint, model_name):
        with pytest.raises(ValueError, match="known rfdetr class"):
            _checkpoint_metadata(make_checkpoint(model_name=model_name))

    def test_a_non_square_token_grid_raises(self, make_checkpoint):
        with pytest.raises(ValueError, match="square"):
            _checkpoint_metadata(make_checkpoint(tokens=1290))

    @pytest.mark.parametrize("key", ["model", "args"])
    def test_a_missing_layout_key_raises_key_error(self, make_checkpoint, key):
        with pytest.raises(KeyError):
            _checkpoint_metadata(make_checkpoint(drop=key))


class TestLoadDetector:
    def test_a_missing_extra_degrades_to_none(self, tmp_path):
        checkpoint = tmp_path / "x.pt"
        checkpoint.write_bytes(b"weights")
        with patch("hudini.detection.find_spec", return_value=None):
            assert load_detector(checkpoint, threshold=0.5) is None

    def test_a_missing_checkpoint_degrades_to_none(self, tmp_path):
        assert load_detector(tmp_path / "missing.pt", threshold=0.5) is None

    def test_loads_at_the_checkpoints_own_metadata(self, make_checkpoint):
        checkpoint = make_checkpoint(grid_side=44, class_names=("offscreen", "active"))
        with patch("rfdetr.RFDETRNano") as model_class:
            detector = load_detector(checkpoint, threshold=0.5)
        model_class.assert_called_once_with(pretrain_weights=str(checkpoint), resolution=704)
        assert isinstance(detector, Detector)

    def test_a_malformed_checkpoint_raises(self, make_checkpoint):
        with pytest.raises(ValueError, match="known rfdetr class"):
            load_detector(make_checkpoint(model_name="ResNet50"), threshold=0.5)


def test_adopt_library_loggers_routes_rfdetr_through_root():
    library = logging.getLogger("rf-detr")
    backbone = logging.getLogger("rfdetr.models.backbone")
    for target in (library, backbone):
        target.handlers = [logging.StreamHandler()]
        target.setLevel(logging.INFO)
        target.propagate = False
    _adopt_library_loggers()
    for target in (library, backbone):
        assert [type(handler) for handler in target.handlers] == [logging.NullHandler]
        assert target.level == logging.NOTSET
        assert target.propagate is True
