"""Unit tests for hudini.cnn.

Forward passes use tiny fake modules; the arm-digit load runs against a
real saved checkpoint of the real net.
"""

import numpy as np
import pytest
import torch
from torch import nn

from hudini.cnn import (
    ArmDigitClassifier,
    ArmDigitNet,
    OffscreenDigitClassifier,
    StatusClassifier,
    crop_arm_region,
    end_centers,
    pad_pod_to_aspect,
)
from hudini.schema import Box, Status


class FixedLogits(nn.Module):
    """A module that answers every input with the same logits row."""

    def __init__(self, row: list[float]) -> None:
        super().__init__()
        self.row = torch.tensor(row)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.row.repeat(batch.shape[0], 1)


POD = np.zeros((40, 300, 3), dtype=np.uint8)


class TestStatusClassifier:
    def test_class_one_reads_active_with_its_probability(self):
        classifier = StatusClassifier(model=FixedLogits([0.0, 2.0]), device="cpu")
        first, second = classifier.classify([POD, POD])
        assert first == second
        status, score = first
        assert status is Status.ACTIVE
        assert score == pytest.approx(0.8808, abs=1e-4)

    def test_no_pods_means_no_forward_pass(self):
        assert StatusClassifier(model=FixedLogits([1.0, 0.0]), device="cpu").classify([]) == []


class TestArmDigitClassifier:
    def test_the_strongest_head_names_the_digit(self):
        model = FixedLogits([0.0, 0.0, 3.0, 0.0])
        model.input_size = 24
        classifier = ArmDigitClassifier(model=model, device="cpu")
        ((digit, score),) = classifier.classify([POD])
        assert digit == 3
        assert score > 0.8

    def test_load_round_trips_a_real_checkpoint(self, tmp_path):
        checkpoint = tmp_path / "arm.pt"
        torch.save(ArmDigitNet().state_dict(), checkpoint)
        classifier = ArmDigitClassifier.load(checkpoint, device="cpu")
        ((digit, score),) = classifier.classify([POD])
        assert digit in (1, 2, 3, 4)
        assert 0.0 < score <= 1.0

    def test_load_raises_on_a_missing_checkpoint(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ArmDigitClassifier.load(tmp_path / "missing.pt", device="cpu")


class TestOffscreenDigitClassifier:
    def _classifier(self, row: list[float]) -> OffscreenDigitClassifier:
        return OffscreenDigitClassifier(
            model=FixedLogits(row),
            device="cpu",
            input_size=32,
            mean=np.zeros(3, dtype=np.float32),
            std=np.ones(3, dtype=np.float32),
            threshold=0.5,
        )

    def test_the_strongest_head_names_the_arm(self):
        region = np.zeros((200, 400, 3), dtype=np.uint8)
        ((arm, score),) = self._classifier([-5.0, 4.0, -5.0, -5.0]).classify(
            region, [Box(x=50, y=20, w=200, h=14)]
        )
        assert arm == 2
        assert score > 0.9

    def test_below_the_threshold_it_abstains(self):
        region = np.zeros((200, 400, 3), dtype=np.uint8)
        ((arm, score),) = self._classifier([-5.0, -5.0, -5.0, -5.0]).classify(
            region, [Box(x=50, y=20, w=200, h=14)]
        )
        assert arm is None
        assert score < 0.5

    def test_a_box_off_the_image_abstains_without_a_forward_pass(self):
        region = np.zeros((10, 10, 3), dtype=np.uint8)
        ((arm, score),) = self._classifier([9.0, 0.0, 0.0, 0.0]).classify(
            region, [Box(x=500, y=500, w=40, h=4)]
        )
        assert (arm, score) == (None, 0.0)


class TestGeometry:
    def test_a_horizontal_bar_has_left_and_right_end_centers(self):
        side, ends = end_centers(Box(x=100, y=50, w=200, h=20))
        assert side == 30
        assert ends == [(110, 60), (290, 60)]

    def test_a_vertical_bar_has_top_and_bottom_end_centers(self):
        _side, ends = end_centers(Box(x=10, y=100, w=20, h=300))
        assert ends == [(20, 110), (20, 390)]

    def test_pad_pod_to_aspect_never_distorts(self):
        tall = np.zeros((100, 100, 3), dtype=np.uint8)
        padded = pad_pod_to_aspect(tall)
        assert padded.shape[1] / padded.shape[0] == pytest.approx(192 / 32)

    def test_the_arm_crop_is_square_at_the_left_edge(self):
        crop = crop_arm_region(np.zeros((50, 314, 3), dtype=np.uint8))
        assert crop.shape[:2] == (50, 50)
